"""原生固定步长学习数据的九基线入口；同一公开观测、反馈预算和独立payload评价。

仅显式运行指定方法，不训练、不挑选模型或超参数。MRC为额外真CSI/64数字
通道参考；其硬件、噪声与物理SNR定义不同，单独标注。
"""
import argparse
import hashlib
import importlib
from pathlib import Path
import time
import numpy as np
from baseline_common.config import rng_for
from baseline_common.controls import codes,physical_units
from baseline_common.data import Dataset,dump_json
from baseline_common.feedback import FeedbackSession
from native_sim.control_engine import NativeControlEngine

SOURCE=Path(__file__).resolve().parent
METHODS=('ttd_das','codebook','coordinate','spsa','done','de','mlp','mrc','teacher')
REQUIRED_CORE=('native_sim/config.py','native_sim/control_engine.py',
               'native_sim/native_laser.cpp','native_sim/laser_profile_019.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_core(dataset):
    recorded=dataset.manifest.get('core_source_sha256',{})
    if any(name not in recorded for name in REQUIRED_CORE):
        raise ValueError('数据缺少配置、控制引擎、激光规则或物理参数的源码哈希。')
    for name,expected in recorded.items():
        path=(SOURCE/name).resolve()
        if SOURCE not in path.parents:raise ValueError('模型源码记录路径无效。')
        if digest(path)!=expected:
            raise ValueError('模型与生成时版本不同，请使用生成源码快照：'+name)
    actual_public=digest(dataset.root/'public.npz')
    expected_public=[dataset.manifest[key] for key in ('public_file_sha256','public_sha256') if key in dataset.manifest]
    if not expected_public:raise ValueError('数据缺少公共配置哈希。')
    if any(value!=actual_public for value in expected_public):
        raise ValueError('公共配置与生成版本不同。')
    return dict(recorded),actual_public


def _scalar_metrics(values):
    """报告保留指标，完整IQ等数组由数据集/单独审计文件保存。"""
    result={}
    for key,value in values.items():
        if isinstance(value,np.generic):value=value.item()
        if value is None or isinstance(value,(str,bool,int,float)):result[key]=value
    return result


def run(root,method,output,split='test',budget=64,max_environments=1,
        carriers=None,seed=20260923,checkpoint=None,allow_incomplete=False):
    if method not in METHODS:raise ValueError('未知基线。')
    if max_environments<0:raise ValueError('环境数量必须非负；0代表全部。')
    output=Path(output)
    if output.exists():raise FileExistsError('拒绝覆盖已有评价报告。')
    dataset=Dataset(root,require_complete=not allow_incomplete)
    if dataset.manifest.get('schema')!='mwp-native-learning-v1':
        raise ValueError('此入口只适配mwp-native-learning-v1；旧数据请用run_baselines.py。')
    core,public_hash=verify_core(dataset)
    cfg=dataset.cfg
    selected=dataset.environments(split)
    if max_environments:selected=selected[:max_environments]
    carrier_set=set(dataset.carriers if not carriers else carriers)
    if not carrier_set.issubset(set(dataset.carriers)):raise ValueError('选择了数据中不存在的载频。')
    if method not in ('mrc','teacher') and budget<cfg.probes:
        raise ValueError('预算必须覆盖16个共同初始探测。')
    model=None
    if method=='mlp':
        if checkpoint is None:raise ValueError('MLP需要仅用train训练的--checkpoint。')
        model=importlib.import_module('baseline_mlp.method').load(checkpoint)
        metadata=model.metadata
        summary=metadata.get('dataset_manifest_summary',{})
        if summary and (summary.get('schema')!=dataset.manifest['schema'] or
                        summary.get('signal_config')!=dataset.manifest['signal_config']):
            raise ValueError('MLP检查点的数据schema或硬件配置不匹配。')
        train_ids=set(metadata.get('training_environment_ids',[]))
        if split=='test' and train_ids.intersection(r['environment_id'] for r in selected):
            raise ValueError('MLP训练环境ID与待评价测试环境重叠。')
    results=[]
    for environment in selected:
        for ci,carrier in enumerate(dataset.carriers):
            if carrier not in carrier_set:continue
            # 真值只在runner/模拟器层使用；在线算法收到的session只有公开观测及反馈。
            observation=dataset.observations(environment,ci)
            truth=dataset.simulator_truth(environment,ci)
            record_path=dataset.record_path(environment,ci)
            row=dict(environment_id=environment['environment_id'],split=split,carrier_ghz=carrier,
                method=method,seed=seed,budget=budget,record_sha256=digest(record_path),
                generation_fingerprint=dataset.manifest['generation_fingerprint'])
            start=time.perf_counter()
            evaluation_rng=rng_for(environment['seed'],carrier,seed,510)
            if method=='mrc':
                module=importlib.import_module('baseline_mrc.method')
                # 31 QPSK有效载荷符号，与native单记录的比特数量相同；数字参考自行抽样。
                metrics=module.evaluate(truth['channel'],float(truth['power_dbm']),cfg,
                                        evaluation_rng,payload_symbols=1)
                row.update(_scalar_metrics(metrics))
                row.update(feedback_calls=None,information='true CSI, 64 digital RF channels',
                    comparison_group='privileged_digital_reference',
                    payload_scope='independent digital-reference QPSK draw; not replay of native payload waveform',
                    snr_db_definition='physical post-combining SNR of ideal digital RF reference',
                    physical_hardware_shared_with_native=False,
                    payload_nmse_db=float(-20*np.log10(max(row['evm_percent']/100,1e-15))))
            else:
                build_start=time.perf_counter()
                engine=NativeControlEngine(cfg,truth['branch_band_w'],truth['branch_dc_w'],
                    carrier*1e9,observation['pilot_qpsk'])
                row['engine_cache_build_seconds']=time.perf_counter()-build_start
                controller_start=time.perf_counter()
                if method=='teacher':
                    label=dataset.labels(environment,ci)
                    control=label['control'];calls=None
                    row.update(information='stored offline teacher with privileged per-branch cache',
                        comparison_group='privileged_photonic_teacher',
                        offline_objective_evaluations=int(label['objective_evaluations']),
                        stored_teacher_objective=float(label['objective']))
                else:
                    def measure(u,call):
                        return engine.measure(u,observation['pilot_qpsk'],
                            rng_for(environment['seed'],carrier,seed,520,call))
                    session=FeedbackSession(cfg,observation,measure,budget)
                    if method=='mlp':
                        # 直接评价网络输出，不用测试真值在网络控制和初始码本之间选优。
                        control=model.predict(observation,cfg)
                    else:
                        module=importlib.import_module('baseline_'+method+'.method')
                        control=module.optimize(session,rng_for(seed,environment['seed'],carrier,530))
                    control=engine.project(control);calls=session.calls
                    row.update(information='same public paired probes and noisy single-APD feedback',
                        comparison_group='implementable_photonic_controller',
                        best_score_among_queried_controls=session.best_score,
                        simulation_feedback_seconds=session.simulation_feedback_seconds,
                        controller_wall_seconds=max(0.,time.perf_counter()-controller_start-session.simulation_feedback_seconds))
                    queried=[score for u,score in zip(session.controls,session.scores) if np.array_equal(u,control)]
                    row['returned_control_measured_score']=max(queried) if queried else None
                e=engine.evaluate(control)
                # 只有算法返回控制后才将payload传入独立评价，反馈回调不持有payload。
                evaluator=importlib.import_module('native_sim.evaluation').evaluate_record
                metrics=evaluator(engine,control,truth['payload_qpsk'],evaluation_rng)
                row.update(_scalar_metrics(metrics))
                delay,attenuation=physical_units(control,cfg)
                row.update(control_code=codes(control,cfg),delay_ps=delay,attenuation_db=attenuation,
                    objective=e['objective'],objective_definition=e['objective_definition'],feedback_calls=calls,
                    optical_dc_w=e['optical_dc_w'],apd_dc_a=e['apd_dc_a'],
                    noise_model=e['noise_model'],
                    snr_db_definition='-10*log10(independent payload NMSE), includes distortion',
                    physical_operating_range_verified=False,final_control_switch_seconds=cfg.switch_s,
                    measurement_seconds=None if calls is None else calls*cfg.measurement_s)
                if calls is not None:
                    row['estimated_control_latency_seconds']=calls*cfg.measurement_s+cfg.switch_s+row['controller_wall_seconds']
            for key in ('snr_db','evm_percent','bit_errors','bits_tested'):
                if key not in row or not np.isfinite(row[key]):raise ValueError('评价器缺少有限指标：'+key)
            row['wall_seconds_including_simulator_and_evaluation']=time.perf_counter()-start
            results.append(row)
    if not results:raise ValueError('没有选择任何评价样本。')
    sources=[p for p in sorted(SOURCE.rglob('*.py')) if '__pycache__' not in p.parts]
    report=dict(schema='mwp-native-baseline-report-v1',status='complete',dataset=str(dataset.root),
        method=method,model=dataset.manifest['model_selection'],dataset_status=dataset.manifest['status'],
        diagnostic_incomplete_dataset=bool(allow_incomplete),
        dataset_manifest_sha256=digest(dataset.root/'manifest.json'),public_sha256=public_hash,
        generation_fingerprint=dataset.manifest['generation_fingerprint'],verified_core_source_sha256=core,
        source_sha256={str(p.relative_to(SOURCE)):digest(p) for p in sources},
        checkpoint_sha256=None if checkpoint is None else dict(weights=digest(Path(checkpoint)/'weights.npz'),
            metadata=digest(Path(checkpoint)/'checkpoint.json')),
        repetition_scope='optimizer and additional APD feedback noise vary by seed; stored initial probes and antenna-noise record stay fixed',
        noise_scope='native fixed-step laser with fixed antenna noise; mean-power stationary Gaussian APD approximation',
        purpose='explicit evaluation only; no test-driven parameter selection or training',
        records=results,mean_snr_db=float(np.mean([r['snr_db'] for r in results])),
        snr_db_definition=results[0]['snr_db_definition'],
        mean_evm_percent=float(np.mean([r['evm_percent'] for r in results])),
        pooled_bit_errors=int(sum(r['bit_errors'] for r in results)),
        pooled_bits_tested=int(sum(r['bits_tested'] for r in results)))
    output.parent.mkdir(parents=True,exist_ok=True);dump_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--max-environments',type=int,default=1,help='0代表全部环境')
    p.add_argument('--carriers',type=int,nargs='*')
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--allow-incomplete',action='store_true',help='仅诊断现有记录；缺失文件仍报错，不用于正式评价')
    a=p.parse_args()
    run(a.dataset,a.method,a.output,a.split,a.budget,a.max_environments,a.carriers,a.seed,a.checkpoint,a.allow_incomplete)


if __name__=='__main__':main()
