"""重跑 A × 候选反馈 F；不调用旧联合测量校正 B，不改变冻结物理核。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import (SOURCE, LEVELS, require_host, sha256,
    verify_sources, write_json, now, rng_for, atomic_npz)
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from study_full_baselines.confirmation_batch import guard_load
from our_method_response_control.train import precision
from study_uniform64.controllers import Feedback64, audit as audit_trace
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap, contrast, bh_adjust

METHODS = ['covariance_response', 'complex_response_cnn', 'covariance_warm64',
           'cnn_warm64', 'initial_select64']
LABELS = ['传统估计', 'A：复响应学习', '传统估计＋F', 'A＋F：完整流程', '64 次固定扫描']
STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def initialize(output, protocol):
    require_host(); precision(); verify_sources(protocol['source_sha256'])
    if not torch.cuda.is_available():
        raise RuntimeError('Remote GPU required')
    prior = protocol['prior']; bundle = Path(prior['runtime_bundle'])
    verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != prior['runtime_manifest_sha256']:
        raise ValueError('Frozen model package changed')
    with np.load(bundle/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    with patch('numpy.load', guard_load(np.load)):
        conventional = OnlineController(bundle, METHODS[0], public)
        learned = OnlineController(bundle, METHODS[1], public)
        initial = OnlineController(bundle, 'ttd_das', public)
        models = dict(zip(METHODS, [conventional, learned,
            Feedback64(conventional, public), Feedback64(learned, public),
            Feedback64(initial, public, fixed_catalog=True)]))
    assert type(conventional) is OnlineController and type(learned) is OnlineController
    assert not conventional.warm and not learned.warm
    STATE.update(output=Path(output), protocol=protocol, public=public, models=models)
    write_json(Path(output)/'workers'/f'{os.getpid()}.json', dict(pid=os.getpid(), at=now(),
        gpu=torch.cuda.get_device_name(), artifacts={n:m.artifacts for n,m in models.items()},
        method_configuration=protocol['configuration'], joint_correction_instantiated=False))


def original(row, carrier):
    p = STATE['protocol']; source = Path(p['source'])
    folder = source/'records'/f"environment_{row['index']:05d}"
    marker = read(folder/'complete.json')
    if marker['environment_id'] != row['environment_id'] or sha256(folder/'environment.json') != marker['environment_sha256']:
        raise ValueError('Original environment changed')
    item = next(x for x in marker['carriers'] if x['carrier_ghz'] == carrier)
    path = source/item['path']
    if sha256(path) != item['sha256']:
        raise ValueError('Original raw record changed')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    uniform = Path(p['uniform_source'])/'records'/folder.name/f'carrier_{carrier:02d}.npz'
    um = read(uniform.with_suffix('.json'))
    if sha256(uniform) != um['sha256']:
        raise ValueError('Original fixed-scan record changed')
    with np.load(uniform, allow_pickle=False) as f:
        scan = {k:f[k].copy() for k in f.files}
    return read(folder/'environment.json'), arrays, scan, dict(original=item['sha256'], scan=um['sha256'])


def compute(row, carrier):
    p, public = STATE['protocol'], STATE['public']
    env, old, scan, hashes = original(row, carrier)
    raw, observed = public_observation(env, carrier, public, p['prior']['backend'])
    np.testing.assert_array_equal(raw, old['public_X'])
    np.testing.assert_array_equal(raw, scan['public_X'])
    arrays = {'public_X':raw}; controls, costs = [], []
    with patch('numpy.load', guard_load(np.load)):
        for i, name in enumerate(METHODS):
            # 每种方法实际执行自己的测量，使用共同随机数，不借用另一种方法的评分。
            def measure(control, call):
                return observed.measure_detailed(control, rng_for(row['seed'], carrier, 620, call))['score']
            got = STATE['models'][name].decide(raw.copy(), rng_for(0, row['seed'], carrier, 630),
                                              measure if i >= 2 else None)
            controls.append(got['control_code'])
            costs.append([got['feedback_calls'], got['software_seconds'], got['feedback_simulator_seconds']])
            for field in ['trace_control_code', 'trace_scores', 'proposal_control_code']:
                if field in got:
                    arrays[name+'__'+field] = got[field]
    decisions = dict(methods=METHODS, control_code=np.asarray(controls), costs=np.asarray(costs),
                     feedback={}, teacher_objective_evaluations=None)
    scored = score_decisions(env, carrier, public, decisions, p['prior']['backend'], include_mrc=False)
    arrays.update(control_code=scored['control_code'], metrics=scored['metrics'])
    for i, name in enumerate(METHODS):
        source, names = (old, p['prior']['methods']) if i < 4 else (scan, p['uniform']['methods'])
        j = names.index(name)
        np.testing.assert_array_equal(arrays['control_code'][i], source['control_code'][j])
        np.testing.assert_array_equal(arrays['metrics'][i, [0,1,3,4,5,6,10]], source['metrics'][j, [0,1,3,4,5,6,10]])
        np.testing.assert_allclose(arrays['metrics'][i,:10], source['metrics'][j,:10], rtol=1e-12, atol=1e-28)
        if i >= 2:
            for field in ['trace_control_code', 'trace_scores']:
                np.testing.assert_array_equal(arrays[name+'__'+field], source[name+'__'+field])
    for direct, feedback in [(0,2),(1,3)]:
        np.testing.assert_array_equal(arrays[METHODS[feedback]+'__proposal_control_code'], controls[direct])
    subset = {k:v for k,v in arrays.items() if k.startswith(tuple(n+'__' for n in METHODS[2:]))}
    subset.update(public_X=raw, control_code=arrays['control_code'][2:], metrics=arrays['metrics'][2:])
    audit_trace(subset, METHODS[2:], public, carrier)
    np.testing.assert_array_equal(arrays['metrics'][:,10], [16,16,64,64,64])
    return arrays, dict(source_sha256=hashes, input_bitwise_equal=True,
        all_controls_bitwise_equal=True, integer_counts_equal=True, metrics_within_frozen_tolerance=True,
        RQ1_primary_reproduced=True, proposal_matches_direct_output=True,
        feedback_trace_equal=True, joint_correction_used=False), env


def one(row):
    output, p = STATE['output'], STATE['protocol']; started = time.perf_counter()
    folder = output/'records'/f"environment_{row['index']:05d}"; folder.mkdir(parents=True, exist_ok=True)
    records = []
    for carrier in p['carriers']:
        path = folder/f'carrier_{carrier:02d}.npz'; marker = path.with_suffix('.json')
        if marker.exists():
            entry = read(marker)
            if entry['protocol_sha256'] != fingerprint(p) or sha256(path) != entry['sha256']:
                raise ValueError('Resume identity mismatch')
        else:
            if path.exists():
                raise ValueError('Uncommitted record needs inspection')
            arrays, checks, env = compute(row, carrier)
            if (folder/'environment.json').exists() and read(folder/'environment.json') != env:
                raise ValueError('Environment changed')
            write_json(folder/'environment.json', env); atomic_npz(path, **arrays)
            entry = dict(environment_id=row['environment_id'], carrier_ghz=carrier,
                path=str(path.relative_to(output)), sha256=sha256(path),
                protocol_sha256=fingerprint(p), checks=checks, at=now())
            write_json(marker, entry)
        records.append(entry)
        write_json(output/'workers'/f'{os.getpid()}_progress.json',dict(environment=row['index'],carrier=carrier,at=now()))
    result = dict(environment_id=row['environment_id'], index=row['index'], carriers=records,
                  seconds=time.perf_counter()-started, at=now())
    write_json(folder/'complete.json', result)
    return result


def analyze(output, p):
    values=[]
    for row in p['rows']:
        current=[]
        for carrier in p['carriers']:
            path=output/'records'/f"environment_{row['index']:05d}"/f'carrier_{carrier:02d}.npz'
            entry=read(path.with_suffix('.json'))
            if sha256(path)!=entry['sha256'] or entry['protocol_sha256']!=fingerprint(p):
                raise ValueError('Analysis record changed')
            with np.load(path,allow_pickle=False) as f:
                current.append(f['metrics'].copy())
        values.append(current)
    values=np.asarray(values); s=sufficient(values); point=metrics(s.sum(0))
    draws=bootstrap(s,10000,20261029,strata=[r['joint_stratum_id'] for r in p['rows']])
    specs=[('A_without_F',[-1,1,0,0,0]),('A_with_F',[0,0,-1,1,0]),
           ('F_without_A',[-1,0,1,0,0]),('F_with_A',[0,-1,0,1,0]),
           ('interaction',[1,-1,-1,1,0]),('AF_vs_fixed_scan64',[0,0,0,1,-1])]
    comparisons=[]
    for label,c in specs:
        for field in FIELDS[:5]:
            comparisons.append(dict(id=label,coefficients=c,**contrast(s,draws,c,field)))
    for field in FIELDS[:5]:
        subset=[r for r in comparisons if r['metric']==field]
        for row,q in zip(subset,bh_adjust([r['p_value'] for r in subset])):
            row['sign_test_bh_q']=float(q)
    summary=[]
    for i,name in enumerate(METHODS):
        summary.append(dict(method=name,label=LABELS[i],**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
            ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])},
            bit_errors=int(s[:,i,0].sum()),bits_tested=int(s[:,i,1].sum()),
            probes=int(values[0,0,i,10]),mean_controller_seconds=float(values[:,:,i,11].mean()),
            mean_feedback_simulator_seconds=float(values[:,:,i,12].mean())))
    write_json(output/'summary.json',summary); write_json(output/'comparisons.json',comparisons)
    atomic_npz(output/'environment_metrics.npz',values=values,methods=np.asarray(METHODS),
        environment_ids=np.asarray([r['environment_id'] for r in p['rows']]))
    lines=['# RQ2：A 与候选反馈 F 的重跑结果','',
        '完整覆盖 864 个既有测试环境 × 17 档载频；冻结模型 seed=0，无重新训练或调参。',
        '五组均重新执行观测、控制和接收评分，并逐条件核对原结果；旧联合校正 B 未参与。','',
        '| 配置 | 测量数 | BER (%) | SER (%) | 块错误率 (%) | RMS EVM (%) | 输出 SNR (dB) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['label']} | {row['probes']} | {row['ber']*100:.6f} | {row['ser']*100:.6f} | {row['block_error_rate']*100:.6f} | {row['rms_evm_percent']:.4f} | {row['paired_output_snr_db']:.4f} |")
    lines+=['','下表为候选减对照的 BER 差，单位为百分点；负数表示候选的汇总 BER 更低。',
        '区间按环境分层配对 bootstrap；符号检验 q 值检验环境胜负概率，不是均值差检验。','',
        '| 对比 | 差值 | 95% 区间 | 胜/平/负（环境） | 符号检验 BH q |',
        '|---|---:|---|---|---:|']
    for row in comparisons:
        if row['metric']=='ber':
            lo,hi=[v*100 for v in row['ci95']]
            lines.append(f"| {row['id']} | {row['estimate']*100:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {row['wins']}/{row['ties']}/{row['losses']} | {row['sign_test_bh_q']:.6g} |")
    lines+=['','解释边界：A 的同预算增益使用 A_with_F；反馈的 F_with_A 对比包含 16→64 的测量成本增加。',
        'AF_vs_fixed_scan64 是同 64 次测量的对照。交互项不等于协同增益，必须按实际符号解释。',
        '这是对已评价测试集的修订与复现，不是新独立测试；区间不覆盖重新训练的随机性。',
        '本次并行时间记录不替代原论文的独立延迟基准。论文、大纲和审稿回复未自动修改。','']
    (output/'REPORT_zh.md').write_text('\n'.join(lines))


def run(project, output, preflight, workers):
    require_host()
    source=project/'dataset_simulation/baseline_results/20260926_final864_selected'
    uniform=project/'dataset_simulation/baseline_results/20260927_uniform64_all13'
    prior=read(source/'protocol.json'); up=read(uniform/'protocol.json')
    verify_sources(up['source_sha256'])
    if prior['rows']!=up['rows'] or prior['carriers']!=up['carriers']:
        raise ValueError('Frozen test sets differ')
    paths=[Path(__file__),Path(__file__).with_name('PROTOCOL.md')]
    sources={**up['source_sha256'],**{str(x.relative_to(SOURCE)):sha256(x) for x in paths}}
    p=dict(schema='rq2-response-learning-feedback-factorial-v1',prior=prior,uniform=up,
        source=str(source),uniform_source=str(uniform),source_sha256=sources,
        source_protocol_sha256=sha256(source/'protocol.json'),uniform_protocol_sha256=sha256(uniform/'protocol.json'),
        code_sha256=sha256(Path(__file__)),methods=METHODS,
        configuration=[dict(method=n,A=i in [1,3],F=i in [2,3],fixed_scan=i==4,
            joint_correction=False,probes=16 if i<2 else 64) for i,n in enumerate(METHODS)],
        rows=prior['rows'][:1] if preflight else prior['rows'],carriers=[4,12,20] if preflight else prior['carriers'],
        scope='preflight' if preflight else 'rerun_existing_test_posthoc_method_reframing',
        authorization='2026-10-29 user requested A plus feedback RQ2 rerun; excludes stopped E2',
        analysis=dict(bootstrap=10000,seed=20261029,unit='environment',primary='ber',
            frozen_model_seed=0,selection='none',multiplicity='six contrasts within each metric'))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'protocol.json').exists() and read(output/'protocol.json')!=p:
            raise ValueError('Protocol changed')
        if (output/'complete.json').exists():
            raise ValueError('Run already complete; do not duplicate')
        write_json(output/'protocol.json',p)
        for path,digest in sources.items():
            dest=output/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha256(dest)!=digest:
                raise ValueError('Snapshot changed')
            if not dest.exists():shutil.copyfile(SOURCE/path,dest)
        (output/'workers').mkdir(exist_ok=True)
        write_json(output/'progress.json',dict(status='running',pid=os.getpid(),environments_complete=0,total=len(p['rows']),at=now()))
        if preflight:
            initialize(output,p); records=[one(p['rows'][0])]
            assert one(p['rows'][0])['carriers']==records[0]['carriers']
        else:
            proof=read(output.parent/'preflight/complete.json')
            if proof['status']!='complete' or proof['code_sha256']!=p['code_sha256'] or proof['source_sha256']!=sources:
                raise ValueError('Current code has not passed preflight')
            records=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(output,p)) as pool:
                futures={pool.submit(one,row):row for row in p['rows']}
                for future in as_completed(futures):
                    records.append(future.result())
                    write_json(output/'progress.json',dict(status='running',pid=os.getpid(),environments_complete=len(records),total=len(p['rows']),at=now()))
                    print(f'{len(records)}/{len(p["rows"])} environments committed',flush=True)
            records.sort(key=lambda r:r['index'])
            write_json(output/'progress.json',dict(status='analyzing',pid=os.getpid(),environments_complete=len(records),total=len(p['rows']),at=now()))
            analyze(output,p)
        verify_sources(sources)
        write_json(output/'records.json',records)
        write_json(output/'complete.json',dict(status='complete',environments=len(records),
            carriers=len(p['carriers']),methods=len(METHODS),code_sha256=p['code_sha256'],source_sha256=sources,
            protocol_sha256=fingerprint(p),records_sha256=sha256(output/'records.json'),at=now(),preflight=preflight))
        write_json(output/'progress.json',dict(status='complete',environments_complete=len(records),total=len(p['rows']),at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=parser.parse_args()
    try:
        run(args.project.resolve(),args.output.resolve(),args.preflight,args.workers)
    except BaseException:
        if args.output.exists():
            write_json(args.output/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
