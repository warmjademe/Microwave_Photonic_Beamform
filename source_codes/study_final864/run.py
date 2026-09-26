"""最终测试：先用已知验证案例核对执行器，再冻结并运行未用的864个环境。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE,require_host,sha256,source_record,verify_sources,write_json,now
from study_full_baselines.runtime_bundle import verify as verify_bundle,fingerprint
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_freeze import budget
from study_full_baselines import confirmation_execute as engine
from study_full_baselines.confirmation_batch import committed_environment,guard_load
from study_full_baselines.analyze_confirmation import check_values
from study_fair_followup.evaluate import identity_for
from our_method_joint_refinement.online_full import bind_joint,JointController
from our_method_response_control.train import precision

BASELINES=['ttd_das','codebook','coordinate','spsa','done','de',
           'mlp','dnn','cnn','rescnn','transformer','complex_cnn','jct']
PRIMARY=['cnn__joint_full_noise','cnn_warm64']
ABLATIONS=['covariance_response','complex_response_cnn','covariance__joint_full_noise','covariance_warm64']
FRESH_SHA='06c44a38ad6ac36a213671181574d415814cee6c7f26a5cb5885a94e3c3b8378'


def read(path): return json.loads(Path(path).read_text())


def comparisons(names):
    result=[]
    def add(label,family,terms):
        if set(terms)-set(names) or sum(terms.values())!=0: raise ValueError('比较清单错误')
        result.append(dict(id=label,family=family,terms=terms,
                           coefficients=[terms.get(n,0) for n in names]))
    for probes,own in [(16,PRIMARY[0]),(64,PRIMARY[1])]:
        for other in BASELINES:
            if budget(other)==probes:
                add('primary%d_minus_'%probes+other,'same_budget_'+str(probes),{own:1,other:-1})
    for prefix,b0,b1 in [('joint','covariance__joint_full_noise',PRIMARY[0]),
                         ('feedback','covariance_warm64',PRIMARY[1])]:
        a0,a1='covariance_response','complex_response_cnn'
        for label,terms in [('A_only',{a1:1,a0:-1}),('B_only',{b0:1,a0:-1}),
                            ('remove_B',{b1:1,a1:-1}),('remove_A',{b1:1,b0:-1}),
                            ('interaction',{b1:1,a1:-1,b0:-1,a0:1})]:
            add(prefix+'_'+label,'components_'+prefix,terms)
    return result


def assemble(project):
    # 沿用已核验的训练来源和核心源码，再加上本轮显式选择的清单。
    identity=identity_for(project,True)
    package=verify_bundle(Path(identity['runtime_bundle']))
    joint=bind_joint(project,package)
    for name,digest in joint['source_sha256'].items():
        if name in identity['source_sha256'] and identity['source_sha256'][name]!=digest:
            raise ValueError('源码身份冲突')
        identity['source_sha256'][name]=digest
    paths=['study_final864/run.py','study_final864/analyze.py','study_final864/PROTOCOL.md',
           'dataset_protocol/registry.py','our_method_joint_refinement/online_full.py',
           'our_method_joint_refinement/method.py','study_full_baselines/confirmation_execute.py',
           'study_full_baselines/confirmation_freeze.py','study_full_baselines/audit_results.py']
    identity['source_sha256'].update(source_record(paths))
    ordinary=BASELINES+PRIMARY+ABLATIONS
    identity.update(schema='mwp-final864-selected-with-ablation-v1',ordinary_methods=ordinary,
        methods=ordinary+['teacher','mrc'],measurement_budgets={n:budget(n) for n in ordinary},
        joint_refinement=joint,method_roles={'baselines':BASELINES,'primary':PRIMARY,
            'ablations':ABLATIONS,'references':['teacher','mrc']},
        statistics=dict(primary_metric='ber',bootstrap_repetitions=10000,bootstrap_seed=20260926,
            stratified=True,comparisons=comparisons(ordinary+['teacher','mrc']),
            metrics=['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db'],
            multiplicity='BH within each comparison family and metric',
            aggregation='error totals; sqrt mean NMSE; ratio of summed paired signal/noise powers',
            selection='validation BER: best at each measurement budget; never use final test to select'))
    verify_sources(identity['source_sha256'])
    return identity


def initialize(output,identity):
    require_host(); precision()
    if not torch.cuda.is_available(): raise RuntimeError('华硕GPU不可用')
    verify_sources(identity['source_sha256']); bundle=Path(identity['runtime_bundle'])
    if sha256(bundle/'manifest.json')!=identity['runtime_manifest_sha256']: raise ValueError('模型包改变')
    bound=identity['joint_refinement']
    for path,digest in {**bound['file_sha256'],**bound['spatial_binding']['file_sha256']}.items():
        if sha256(path)!=digest: raise ValueError('训练统计改变')
    with np.load(bundle/'public.npz') as f:
        public={k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    models={}
    with patch('numpy.load',guard_load(np.load)):
        for name in identity['ordinary_methods']:
            if name.endswith('__joint_full_noise'):
                alias=name.split('__')[0]
                base=OnlineController(bundle,'complex_response_cnn' if alias=='cnn' else 'covariance_response',public)
                if alias=='cnn' and bound['weights_sha256'] not in base.artifacts.values():
                    raise ValueError('联合校正CNN权重不符')
                spatial=np.load(Path(bound['spatial_binding']['path'])/(alias+'_residual_covariance.npy'))
                frequency=np.load(Path(bound['path'])/(alias+'_frequency_covariance.npy'))
                models[name]=JointController(base,name,spatial,frequency,True)
            else: models[name]=OnlineController(bundle,name,public)
    if list(models)!=identity['ordinary_methods']: raise ValueError('加载的方法不符')
    engine.STATE.update(output=Path(output),identity=identity,public=public,models=models)
    write_json(Path(output)/'workers'/('%d.json'%os.getpid()),dict(pid=os.getpid(),at=now(),
        gpu=torch.cuda.get_device_name(),threads=torch.get_num_threads(),methods=len(models),
        identity_sha256=fingerprint(identity)))


def preflight(project,output):
    require_host()
    if output.exists(): raise FileExistsError(output)
    identity=assemble(project); row=identity['rows'][0]
    identity.update(scope='old_runtime_rehearsal',input_sha256=identity['input_sha256'][row['environment_id']])
    output.mkdir(parents=True); (output/'workers').mkdir(); (output/'records').mkdir()
    write_json(output/'protocol.json',identity); initialize(output,identity)
    record=engine.one_environment(row)
    study=project/'dataset_simulation/baseline_results/20260925_full_baselines'
    sources={}; checked=0
    for item in record['carriers']:
        with np.load(output/item['path']) as f: current={k:f[k].copy() for k in f.files}
        check_values(current,item['carrier_ghz'],identity,engine.STATE['public'])
        for group in ['evaluation_fair3456_online','evaluation_joint_refinement_3456_online']:
            folder=study/group; prior=read(folder/'protocol.json')
            old=committed_environment(folder,prior['rows'][0],prior)
            entry=next(r for r in old['carriers'] if r['carrier_ghz']==item['carrier_ghz'])
            sources[str(folder/entry['path'])]=entry['sha256']
            with np.load(folder/entry['path']) as f:
                np.testing.assert_array_equal(current['public_X'],f['public_X'])
                for i,name in enumerate(identity['methods']):
                    chosen='evaluation_joint_refinement_3456_online' if '__joint_' in name else 'evaluation_fair3456_online'
                    if group!=chosen: continue
                    j=prior['methods'].index(name)
                    np.testing.assert_array_equal(current['control_code'][i],f['control_code'][j])
                    np.testing.assert_allclose(current['metrics'][i,:11],f['metrics'][j,:11],rtol=2e-10,atol=1e-28,equal_nan=True)
                    for key in f.files:
                        if key.startswith(name+'__trace_'):
                            np.testing.assert_array_equal(current[key],f[key])
                    checked+=1
    # 原子续跑核对：既有载频全部复用，不再推理或生成。
    resumed=engine.one_environment(row)
    if resumed['newly_computed_carriers']!=0 or resumed['reused_carriers']!=3: raise ValueError('续跑失败')
    complete=dict(status='passed',at=now(),cases=checked,methods=len(identity['methods']),
        validation_environment=row['environment_id'],carriers=[4,12,20],
        new_test_signals_generated=0,bitwise_input_and_control_equal=True,
        source_sha256=identity['source_sha256'],runtime_manifest_sha256=identity['runtime_manifest_sha256'],
        prior_record_sha256=sources,resume_reused_carriers=3,protocol_sha256=sha256(output/'protocol.json'))
    write_json(output/'complete.json',complete);print(json.dumps(complete),flush=True)


def freeze(project,registry,proof,output):
    require_host()
    if output.exists(): raise FileExistsError(output)
    identity=assemble(project); checked=read(proof/'complete.json')
    if (checked['status']!='passed' or checked['cases']!=63
            or checked['source_sha256']!=identity['source_sha256']
            or checked['runtime_manifest_sha256']!=identity['runtime_manifest_sha256']):
        raise ValueError('新清单的63案例前置核验未通过')
    meta=read(registry); marker=read(registry.parent/'complete.json')
    if marker['registry_sha256']!=sha256(registry) or marker['status']!='passed': raise ValueError('数据注册表改变')
    if [meta['splits'][r]['environments'] for r in ['train','validation','test']]!=[3456,216,864]:
        raise ValueError('数据规模改变')
    for group in meta['splits'].values():
        if sha256(group['source_manifest'])!=group['source_sha256']: raise ValueError('原数据清单改变')
    if any(meta['pairwise_seed_overlap'].values()): raise ValueError('数据重叠')
    plan=Path(meta['splits']['test']['source_manifest'])
    if sha256(plan)!=FRESH_SHA: raise ValueError('最终测试计划改变')
    identity.update(scope='fresh_confirmation',rows=read(plan)['environments'],carriers=list(range(4,21)),
        final_confirmation=True,dataset_split='test',training_environments=3456,validation_environments=216,
        fresh_plan_path=str(plan),fresh_plan_sha256=FRESH_SHA,dataset_registry=str(registry),
        dataset_registry_sha256=sha256(registry),preflight_complete=str(proof/'complete.json'),
        preflight_complete_sha256=sha256(proof/'complete.json'),frozen_at=now())
    identity.pop('input_sha256'); identity.pop('data_path'); identity.pop('new_environment_signals_generated')
    output.mkdir(parents=True)
    for name in identity['source_sha256']:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dest)
    write_json(output/'freeze.json',identity)
    print(json.dumps(dict(status='frozen',methods=len(identity['methods']),cases=864*17*len(identity['methods']),
        identity_sha256=fingerprint(identity),at=identity['frozen_at'])),flush=True)


def validate(path):
    identity=read(path);verify_sources(identity['source_sha256'])
    if identity['ordinary_methods']!=BASELINES+PRIMARY+ABLATIONS: raise ValueError('清单改变')
    if identity['statistics']['comparisons']!=comparisons(identity['methods']): raise ValueError('比较方案改变')
    for field,expected in [('fresh_plan_path','fresh_plan_sha256'),('dataset_registry','dataset_registry_sha256'),
                            ('preflight_complete','preflight_complete_sha256')]:
        if sha256(identity[field])!=identity[expected]: raise ValueError('冻结证据改变')
    if identity['rows']!=read(identity['fresh_plan_path'])['environments']: raise ValueError('测试成员改变')
    package=verify_bundle(Path(identity['runtime_bundle']))
    if package['cohort']['train_environments']!=3456: raise ValueError('训练成员改变')
    if sha256(Path(identity['runtime_bundle'])/'manifest.json')!=identity['runtime_manifest_sha256']:
        raise ValueError('模型包改变')
    for name,digest in identity['source_sha256'].items():
        if sha256(path.parent/'source_snapshot'/name)!=digest: raise ValueError('冻结源码快照改变')
    return identity


def run(frozen,output,workers):
    require_host(); identity=validate(frozen)
    if workers not in [1,2,3,4]: raise ValueError('worker只允许1至4')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        claim_path=Path(identity['project'])/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_claim.json'
        claim=dict(freeze_path=str(frozen),freeze_sha256=sha256(frozen),identity_sha256=fingerprint(identity),
                   output=str(output),fresh_plan_sha256=FRESH_SHA)
        if claim_path.exists():
            if read(claim_path)!=claim: raise ValueError('同一测试计划已经登记到其他运行')
        else:
            with claim_path.open('x') as f: json.dump(claim,f,indent=2)
        for name in ['records','workers','attempts']: (output/name).mkdir(exist_ok=True)
        if (output/'protocol.json').exists():
            if read(output/'protocol.json')!=identity: raise ValueError('续跑身份不同')
        else:
            write_json(output/'protocol.json',identity)
            shutil.copytree(frozen.parent/'source_snapshot',output/'source_snapshot')
        complete=[];pending=[]
        for row in identity['rows']:
            old=committed_environment(output,row,identity)
            if old is None: pending.append(row)
            else: complete.append(old)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))//1024
        if pending and (free<workers*3072 or available<workers*2048+4096): raise RuntimeError('当前CPU/GPU空闲内存不足')
        state=dict(status='running',pid=os.getpid(),workers=workers,total=864,completed=len(complete),
            methods=21,ordinary_methods=19,carriers=17,expected_cases=308448,
            final_confirmation=True,identity_sha256=fingerprint(identity),started_at=now(),at=now())
        attempt=output/'attempts'/('%d.json'%time.time_ns());write_json(attempt,state);write_json(output/'progress.json',state)
        if pending:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(output),identity)) as pool:
                jobs=[pool.submit(engine.one_environment,row) for row in pending]
                for job in as_completed(jobs):
                    complete.append(job.result());state.update(completed=len(complete),at=now())
                    write_json(output/'progress.json',state)
        validate(frozen)
        write_json(output/'records.json',sorted(complete,key=lambda r:r['index']))
        state.update(status='complete',completed=len(complete),at=now());write_json(attempt,state);write_json(output/'progress.json',state)
        write_json(output/'complete.json',dict(status='complete_reception_records',environments=864,methods=21,
            carriers=17,cases=308448,identity_sha256=fingerprint(identity),records_sha256=sha256(output/'records.json'),
            final_confirmation=True,scope='fresh_confirmation',at=now()))
    from study_final864.analyze import run as analyze
    analyze(frozen,output,output/'analysis')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','freeze','run'])
    for key in ['project','output','registry','proof','frozen']:p.add_argument('--'+key,type=Path)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:
        if a.action=='preflight':preflight(a.project.resolve(),a.output.resolve())
        elif a.action=='freeze':freeze(a.project.resolve(),a.registry.resolve(),a.proof.resolve(),a.output.resolve())
        else:run(a.frozen.resolve(),a.output.resolve(),a.workers)
    except BaseException:
        if a.output and a.output.exists():write_json(a.output/('failure_%d.json'%time.time_ns()),dict(at=now(),traceback=traceback.format_exc()))
        raise
