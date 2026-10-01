"""在华硕补测 8 个基线的 64 次反馈，保留原始冻结测试不变。"""
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
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import (SOURCE, LEVELS, require_host, sha256,
    source_record, verify_sources, atomic_npz, write_json, now, rng_for)
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_batch import (fingerprint, guard_load,
    committed_environment, check_carrier)
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from our_method_response_control.train import precision
from study_uniform64.controllers import Feedback64, DIRECT, NEW_METHODS, REUSED_METHODS, audit

STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def identity_for(source):
    prior = read(source/'protocol.json'); done = read(source/'complete.json')
    index_path=source/'records.json'
    if index_path.exists():
        if done['records_sha256'] != sha256(index_path):
            raise ValueError('原测试记录索引改变')
    elif done.get('status')=='passed':
        index_path=source/'records'/('environment_%05d'%prior['rows'][0]['index'])/'complete.json'
    else: raise ValueError('原记录索引缺失')
    verify_sources(prior['source_sha256'])
    bundle = Path(prior['runtime_bundle']); verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != prior['runtime_manifest_sha256']:
        raise ValueError('模型包改变')
    sources = dict(prior['source_sha256'])
    sources.update(source_record([str(p.relative_to(SOURCE)) for p in
        (SOURCE/'study_uniform64').iterdir() if p.suffix in ['.py', '.md']]))
    return dict(schema='uniform64-feedback-extension-v1', project=prior['project'],
        source_output=str(source), source_protocol_sha256=sha256(source/'protocol.json'),
        source_complete_sha256=sha256(source/'complete.json'),
        source_records_path=str(index_path),source_records_sha256=sha256(index_path), source_sha256=sources,
        runtime_bundle=str(bundle), runtime_manifest_sha256=prior['runtime_manifest_sha256'],
        backend=prior['backend'], rows=prior['rows'], carriers=prior['carriers'],
        methods=NEW_METHODS, reused_methods=REUSED_METHODS,
        measurement_budgets={n:64 for n in NEW_METHODS},
        prior_methods=prior['methods'], scope='followup_on_fixed_test',
        statistics=dict(repetitions=10000,seed=20260926,primary='ber',
            family='all thirteen same-budget baseline comparisons, per metric',
            metrics=['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']),
        model_selection='unchanged frozen weights; no selection using follow-up outcomes')


def initialize(output, identity):
    require_host(); precision(); verify_sources(identity['source_sha256'])
    if not torch.cuda.is_available():
        raise RuntimeError('华硕 GPU 不可用')
    bundle = Path(identity['runtime_bundle'])
    with np.load(bundle/'public.npz') as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    models = {}
    with patch('numpy.load', guard_load(np.load)):
        models['initial_select64'] = Feedback64(OnlineController(bundle,'ttd_das',public),public,True)
        for name in DIRECT:
            models[name+'_feedback64'] = Feedback64(OnlineController(bundle,name,public),public)
    STATE.update(output=Path(output),identity=identity,public=public,models=models)
    write_json(Path(output)/'workers'/('%d.json'%os.getpid()),
        dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),methods=list(models),at=now(),
             artifacts={k:v.artifacts for k,v in models.items()},identity_sha256=fingerprint(identity)))


def load_original(row, carrier, identity):
    folder = Path(identity['source_output'])/'records'/('environment_%05d'%row['index'])
    marker = read(folder/'complete.json')
    if marker['environment_id'] != row['environment_id']:
        raise ValueError('环境身份不同')
    if sha256(folder/'environment.json') != marker['environment_sha256']:
        raise ValueError('环境参数改变')
    item = next(r for r in marker['carriers'] if r['carrier_ghz']==carrier)
    path = Path(identity['source_output'])/item['path']
    if sha256(path) != item['sha256']:
        raise ValueError('原逐载频记录改变')
    with np.load(path,allow_pickle=False) as f:
        old = {k:f[k].copy() for k in f.files}
    return read(folder/'environment.json'), old, item['sha256']


def compute(row, carrier):
    identity,public,models = (STATE[k] for k in ['identity','public','models'])
    environment,old,old_sha = load_original(row,carrier,identity)
    raw,observed = public_observation(environment,carrier,public,identity['backend'])
    np.testing.assert_array_equal(raw,old['public_X'])
    cache = {}; calls = 0
    def measure(control, call):
        nonlocal calls
        calls += 1
        key = (call, tuple(observed.codes(control)))
        if key not in cache:
            cache[key] = observed.measure_detailed(control,rng_for(row['seed'],carrier,620,call))['score']
        return cache[key]
    decisions = []; arrays = {'public_X':raw}; costs = []
    with patch('numpy.load', guard_load(np.load)):
        for name,model in models.items():
            got = model.decide(raw.copy(),rng_for(0,row['seed'],carrier,630),measure)
            decisions.append(got['control_code'])
            costs.append([got['feedback_calls'],got['software_seconds'],got['feedback_simulator_seconds']])
            for field in ['trace_control_code','trace_scores','proposal_control_code']:
                if field in got: arrays[name+'__'+field]=got[field]
            if name != 'initial_select64':
                old_name = name.removesuffix('_feedback64')
                np.testing.assert_array_equal(got['proposal_control_code'],
                    old['control_code'][identity['prior_methods'].index(old_name)])
    if calls != 8*48:
        raise ValueError('每个新增方法应追加 48 次探测')
    # 旧六种方法只重放已保存控制，逐项核对同一接收器的评价一致性。
    anchors=[identity['prior_methods'].index(n) for n in REUSED_METHODS]
    all_decisions=dict(methods=NEW_METHODS+REUSED_METHODS,
        control_code=np.concatenate([decisions,old['control_code'][anchors]]),
        costs=np.concatenate([costs,old['metrics'][anchors,10:]]),
        feedback={},teacher_objective_evaluations=None)
    scored=score_decisions(environment,carrier,public,all_decisions,identity['backend'],include_mrc=False)
    np.testing.assert_allclose(scored['metrics'][8:,:10],old['metrics'][anchors,:10],rtol=1e-12,atol=1e-28)
    arrays.update(control_code=scored['control_code'][:8],metrics=scored['metrics'][:8])
    audit(arrays,NEW_METHODS,public,carrier)
    return arrays,dict(original_carrier_sha256=old_sha,public_input_bitwise_equal=True,
        direct_model_proposals_bitwise_equal=True,replayed_original_64_methods=6,
        logical_extra_measurements=calls,unique_simulator_queries=len(cache)),environment


def one(row):
    output,identity=STATE['output'],STATE['identity'];started=time.perf_counter()
    folder=output/'records'/('environment_%05d'%row['index']);folder.mkdir(exist_ok=True)
    records=[]
    for carrier in identity['carriers']:
        saved=check_carrier(folder,carrier,row,identity)
        if saved is not None: records.append(saved);continue
        arrays,checks,environment=compute(row,carrier)
        env=folder/'environment.json'
        if env.exists():
            if read(env)!=environment:raise ValueError('续跑环境改变')
        else: write_json(env,environment)
        path=folder/('carrier_%02d.npz'%carrier);atomic_npz(path,**arrays)
        record=dict(environment_id=row['environment_id'],carrier_ghz=carrier,
            sha256=sha256(path),identity_sha256=fingerprint(identity),methods=8,
            path=str(path.relative_to(output)),at=now(),checks=checks)
        write_json(path.with_suffix('.json'),record);records.append(record)
        write_json(output/'workers'/('%d_progress.json'%os.getpid()),
                   dict(environment=row['index'],carrier=carrier,at=now()))
    result=dict(index=row['index'],environment_id=row['environment_id'],
        identity_sha256=fingerprint(identity),environment_sha256=sha256(folder/'environment.json'),
        carriers=records,seconds=time.perf_counter()-started,at=now())
    write_json(folder/'complete.json',result);return result


def preflight(project, output):
    from study_uniform64.check import check_edges, check_common_wrapper
    if output.exists():raise FileExistsError(output)
    source=project/'dataset_simulation/diagnostics/20260926_final864_selected_preflight'
    identity=identity_for(source);identity.update(rows=identity['rows'][:1],carriers=[4,12,20],scope='validation_preflight')
    output.mkdir(parents=True);(output/'workers').mkdir();(output/'records').mkdir()
    write_json(output/'protocol.json',identity);initialize(output,identity)
    edge=check_edges(STATE['public']);record=one(identity['rows'][0])
    common=check_common_wrapper(identity,STATE['public'])
    resumed=one(identity['rows'][0])
    assert resumed['carriers']==record['carriers']
    verify_sources(identity['source_sha256'])
    write_json(output/'records.json',[record])
    result=dict(status='passed',at=now(),source_sha256=identity['source_sha256'],
        runtime_manifest_sha256=identity['runtime_manifest_sha256'],edges=edge,
        common_wrapper=common,added_method_cases=24,old_method_replay_cases=18,
        records_sha256=sha256(output/'records.json'),resume_equal=True)
    write_json(output/'complete.json',result);print(json.dumps(result),flush=True)


def freeze(source, proof, output):
    if output.exists():raise FileExistsError(output)
    identity=identity_for(source);checked=read(proof/'complete.json')
    if (checked['status']!='passed' or checked['source_sha256']!=identity['source_sha256']
            or checked['runtime_manifest_sha256']!=identity['runtime_manifest_sha256']):
        raise ValueError('当前源码或模型没有通过预检')
    if len(identity['rows'])!=864 or identity['carriers']!=list(range(4,21)):
        raise ValueError('必须覆盖完整最终测试')
    identity.update(preflight=str(proof/'complete.json'),preflight_sha256=sha256(proof/'complete.json'),frozen_at=now())
    output.mkdir(parents=True)
    for name in identity['source_sha256']:
        dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/name,dst)
    write_json(output/'freeze.json',identity)
    print(json.dumps(dict(status='frozen',environments=864,carriers=17,new_methods=8)),flush=True)


def run(frozen, output, workers):
    require_host();identity=read(frozen);verify_sources(identity['source_sha256'])
    if workers not in [1,2,3,4]:raise ValueError('允许 1--4 worker')
    for name,digest in identity['source_sha256'].items():
        if sha256(frozen.parent/'source_snapshot'/name)!=digest:raise ValueError('冻结快照改变')
    for path,digest in [(identity['preflight'],identity['preflight_sha256']),
            (Path(identity['source_output'])/'protocol.json',identity['source_protocol_sha256']),
            (identity['source_records_path'],identity['source_records_sha256'])]:
        if sha256(path)!=digest:raise ValueError('绑定证据改变')
    verify_bundle(Path(identity['runtime_bundle']))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ['records','workers']:(output/name).mkdir(exist_ok=True)
        if (output/'protocol.json').exists():
            if read(output/'protocol.json')!=identity:raise ValueError('续跑身份不同')
        else:
            write_json(output/'protocol.json',identity)
            shutil.copytree(frozen.parent/'source_snapshot',output/'source_snapshot')
        done=[];pending=[]
        for row in identity['rows']:
            old=committed_environment(output,row,identity)
            if old is None:pending.append(row)
            else:done.append(old)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free<workers*3072:raise RuntimeError('空闲显存不足，保留其它任务')
        state=dict(status='running',pid=os.getpid(),workers=workers,total=864,
                   completed=len(done),started_at=now(),new_methods=8,carriers=17)
        write_json(output/'progress.json',state)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(str(output),identity)) as pool:
            jobs=[pool.submit(one,row) for row in pending]
            for job in as_completed(jobs):
                done.append(job.result());write_json(output/'progress.json',dict(state,completed=len(done),at=now()))
        verify_sources(identity['source_sha256']);verify_bundle(Path(identity['runtime_bundle']))
        write_json(output/'records.json',sorted(done,key=lambda r:r['index']))
        write_json(output/'progress.json',dict(state,status='complete',completed=len(done),at=now()))
        write_json(output/'complete.json',dict(status='complete',environments=864,carriers=17,methods=8,
            cases=864*17*8,identity_sha256=fingerprint(identity),records_sha256=sha256(output/'records.json'),at=now()))
        print(json.dumps(dict(status='complete',environments=len(done))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','freeze','run'])
    for name in ['project','source','proof','freeze','output']:p.add_argument('--'+name,type=Path)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:
        if a.mode=='preflight':preflight(a.project,a.output)
        elif a.mode=='freeze':freeze(a.source,a.proof,a.output)
        else:run(a.freeze,a.output,a.workers)
    except BaseException:
        if a.output and a.output.exists():write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
