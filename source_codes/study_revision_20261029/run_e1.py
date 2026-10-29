"""完整 A+B+64 次反馈补测；只在远程运行，保持原物理核与冻结权重。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
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
from study_full_baselines.common import require_host, sha256, verify_sources, write_json, now, rng_for, atomic_npz
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from study_full_baselines.confirmation_batch import guard_load
from our_method_joint_refinement.online_full import JointController
from our_method_response_control.train import precision
from study_uniform64.controllers import Feedback64, audit as audit_trace
from study_full_baselines.paired_statistics import FIELDS, sufficient, metrics, bootstrap, contrast, bh_adjust

NEW = ['cnn_joint_feedback64', 'covariance_joint_feedback64']
STATE = {}


def read(p):
    return json.loads(Path(p).read_text())


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def initialize(project, output, protocol):
    require_host(); precision()
    if not torch.cuda.is_available():
        raise RuntimeError('Remote GPU required')
    prior = protocol['prior']
    verify_sources(prior['source_sha256'])
    bundle = Path(prior['runtime_bundle']); verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != prior['runtime_manifest_sha256']:
        raise ValueError('Runtime bundle changed')
    bound = prior['joint_refinement']
    for path, digest in {**bound['file_sha256'], **bound['spatial_binding']['file_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Frozen residual statistics changed')
    with np.load(bundle/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    models = {}
    with patch('numpy.load', guard_load(np.load)):
        for alias, base_name, label in [('cnn', 'complex_response_cnn', NEW[0]),
                                       ('covariance', 'covariance_response', NEW[1])]:
            base = OnlineController(bundle, base_name, public)
            spatial = np.load(Path(bound['spatial_binding']['path'])/(alias+'_residual_covariance.npy'))
            frequency = np.load(Path(bound['path'])/(alias+'_frequency_covariance.npy'))
            joint = JointController(base, alias+'__joint_full_noise', spatial, frequency, True)
            models[label] = Feedback64(joint, public)
    STATE.update(project=Path(project), output=Path(output), protocol=protocol, public=public, models=models)
    write_json(Path(output)/'workers'/f'{os.getpid()}.json',
               dict(pid=os.getpid(), started_at=now(), gpu=torch.cuda.get_device_name(), threads=1))


def original(row, fc):
    source = Path(STATE['protocol']['source'])
    folder = source/'records'/f"environment_{row['index']:05d}"
    marker = read(folder/'complete.json')
    if marker['environment_id'] != row['environment_id'] or sha256(folder/'environment.json') != marker['environment_sha256']:
        raise ValueError('Original environment identity changed')
    item = next(x for x in marker['carriers'] if x['carrier_ghz'] == fc)
    path = source/item['path']
    if sha256(path) != item['sha256']:
        raise ValueError('Original carrier changed')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    return read(folder/'environment.json'), arrays, item['sha256']


def compute(row, fc):
    p, public = STATE['protocol'], STATE['public']
    env, old, digest = original(row, fc)
    raw, engine = public_observation(env, fc, public, p['prior']['backend'])
    np.testing.assert_array_equal(raw, old['public_X'])
    cache = {}
    def measure(control, call):
        key = (call, tuple(engine.codes(control)))
        if key not in cache:
            cache[key] = engine.measure_detailed(control, rng_for(row['seed'], fc, 620, call))['score']
        return cache[key]
    controls, costs = [], []
    arrays = {'public_X': raw}
    prior_names = p['prior']['methods']
    with patch('numpy.load', guard_load(np.load)):
        for label, old_name in zip(NEW, ['cnn__joint_full_noise', 'covariance__joint_full_noise']):
            got = STATE['models'][label].decide(raw.copy(), rng_for(0, row['seed'], fc, 630), measure)
            np.testing.assert_array_equal(got['proposal_control_code'], old['control_code'][prior_names.index(old_name)])
            controls.append(got['control_code'])
            costs.append([got['feedback_calls'], got['software_seconds'], got['feedback_simulator_seconds']])
            for field in ['trace_control_code', 'trace_scores', 'proposal_control_code']:
                arrays[label+'__'+field] = got[field]
    # 回放原 A+反馈作锚点，防止新脚本不经意改变评分器或业务帧。
    anchor = prior_names.index('cnn_warm64')
    decisions = dict(methods=NEW+['anchor_A_feedback64'],
        control_code=np.asarray(controls+[old['control_code'][anchor]]),
        costs=np.asarray(costs+[old['metrics'][anchor, 10:].tolist()]),
        feedback={}, teacher_objective_evaluations=None)
    scored = score_decisions(env, fc, public, decisions, p['prior']['backend'], include_mrc=False)
    np.testing.assert_allclose(scored['metrics'][-1, :10], old['metrics'][anchor, :10], rtol=1e-12, atol=1e-28)
    arrays.update(control_code=scored['control_code'][:2], metrics=scored['metrics'][:2])
    audit_trace(arrays, NEW, public, fc)
    return arrays, dict(original_sha256=digest, input_bitwise_equal=True,
        proposal_equal_to_frozen_B_output=True, historical_A_score_replayed=True), env


def one(row):
    out, p = STATE['output'], STATE['protocol']; started = time.perf_counter()
    folder = out/'records'/f"environment_{row['index']:05d}"; folder.mkdir(parents=True, exist_ok=True)
    records = []
    for fc in p['carriers']:
        path = folder/f'carrier_{fc:02d}.npz'; marker = path.with_suffix('.json')
        if marker.exists():
            entry = read(marker)
            if entry['protocol_sha256'] != fingerprint(p) or sha256(path) != entry['sha256']:
                raise ValueError('Resume identity or data mismatch')
        else:
            if path.exists():
                raise ValueError('Uncommitted carrier requires inspection')
            arrays, checks, env = compute(row, fc)
            if (folder/'environment.json').exists() and read(folder/'environment.json') != env:
                raise ValueError('Environment changed during resume')
            write_json(folder/'environment.json', env)
            atomic_npz(path, **arrays)
            entry = dict(environment_id=row['environment_id'], carrier_ghz=fc,
                path=str(path.relative_to(out)), sha256=sha256(path),
                protocol_sha256=fingerprint(p), checks=checks, at=now())
            write_json(marker, entry)
        records.append(entry)
        write_json(out/'workers'/f'{os.getpid()}_progress.json',
                   dict(environment=row['index'], carrier=fc, at=now()))
    result = dict(environment_id=row['environment_id'], index=row['index'], carriers=records,
                  seconds=time.perf_counter()-started, at=now())
    write_json(folder/'complete.json', result)
    return result


def analyze(project, out, p):
    uniform = Path(project)/'dataset_simulation/baseline_results/20260927_uniform64_all13'
    up = read(uniform/'protocol.json')
    from study_uniform64.controllers import BASELINES64, NEW_METHODS
    names = [*NEW, 'cnn_warm64', 'covariance_warm64', *BASELINES64]
    values = []
    for row in p['rows']:
        current = []
        for fc in p['carriers']:
            folder = out/'records'/f"environment_{row['index']:05d}"
            entry = read(folder/f'carrier_{fc:02d}.json')
            if sha256(out/entry['path']) != entry['sha256']:
                raise ValueError('New result hash mismatch')
            with np.load(out/entry['path']) as f:
                new = f['metrics'].copy()
            _, old, _ = original(row, fc)
            uf = uniform/'records'/f"environment_{row['index']:05d}"/f'carrier_{fc:02d}.npz'
            um = read(uf.with_suffix('.json'))
            if sha256(uf) != um['sha256']:
                raise ValueError('Baseline result hash mismatch')
            with np.load(uf) as f:
                more = f['metrics'].copy()
            columns = [*new]
            for name in names[2:]:
                columns.append(more[NEW_METHODS.index(name)] if name in NEW_METHODS else old['metrics'][p['prior']['methods'].index(name)])
            current.append(columns)
        values.append(current)
    values = np.asarray(values); s = sufficient(values)
    draws = bootstrap(s, 10000, 20261029, strata=[r['joint_stratum_id'] for r in p['rows']])
    point = metrics(s.sum(0))
    rows = []
    specs = [('full_vs_'+n, 'all13', {NEW[0]:1,n:-1}) for n in BASELINES64]
    specs += [('remove_B','components',{NEW[0]:1,'cnn_warm64':-1}),
              ('remove_A','components',{NEW[0]:1,NEW[1]:-1}),
              ('B_with_conventional','components',{NEW[1]:1,'covariance_warm64':-1}),
              ('A_without_B','components',{'cnn_warm64':1,'covariance_warm64':-1}),
              ('interaction','components',{NEW[0]:1,NEW[1]:-1,'cnn_warm64':-1,'covariance_warm64':1})]
    for label, family, terms in specs:
        for field in FIELDS[:5]:
            rows.append(dict(id=label, family=family, **contrast(s, draws, [terms.get(n,0) for n in names], field)))
    for family in ['all13','components']:
        for field in FIELDS[:5]:
            subset = [r for r in rows if r['family']==family and r['metric']==field]
            for r, q in zip(subset, bh_adjust([r['p_value'] for r in subset])):
                r['sign_test_bh_q'] = float(q)
    summary = [dict(method=n, **{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
                    ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])})
               for i,n in enumerate(names)]
    write_json(out/'summary.json', summary); write_json(out/'comparisons.json', rows)
    atomic_npz(out/'environment_metrics.npz', values=values, methods=np.asarray(names),
               environment_ids=np.asarray([r['environment_id'] for r in p['rows']]))


def run(project, output, preflight, workers):
    require_host()
    source = project/'dataset_simulation/baseline_results/20260926_final864_selected'
    prior = read(source/'protocol.json')
    local = Path(__file__).parent
    p = dict(schema='sensors-revision-E1-v1', source=str(source), prior=prior,
        source_protocol_sha256=sha256(source/'protocol.json'), methods=NEW,
        code_sha256=sha256(Path(__file__)), plan_sha256=sha256(local/'PROTOCOL.md'),
        rows=prior['rows'][:1] if preflight else prior['rows'],
        carriers=[4,12,20] if preflight else prior['carriers'],
        scope='integration_check' if preflight else 'revision_followup_on_previously_evaluated_test',
        model_selection='none; no retraining, tuning or best-configuration selection')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'protocol.json').exists() and read(output/'protocol.json') != p:
            raise ValueError('Frozen revision protocol differs')
        write_json(output/'protocol.json', p)
        (output/'workers').mkdir(exist_ok=True)
        write_json(output/'progress.json',dict(status='running',environments_complete=0,total=len(p['rows']),at=now()))
        if preflight:
            initialize(project, output, p); records=[one(p['rows'][0])]
            repeated=one(p['rows'][0]); assert repeated['carriers']==records[0]['carriers']
        else:
            proof=read(output.parent/'e1_preflight/complete.json')
            if proof['status']!='complete' or proof['code_sha256']!=p['code_sha256']:
                raise ValueError('Current code has not passed preflight')
            records=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(project,output,p)) as pool:
                futures={pool.submit(one,row):row for row in p['rows']}
                for future in as_completed(futures):
                    records.append(future.result())
                    write_json(output/'progress.json',dict(status='running',environments_complete=len(records),
                        total=len(p['rows']),at=now()))
                    print(f'{len(records)}/{len(p["rows"])} environments committed',flush=True)
            records.sort(key=lambda r:r['index'])
            initialize(project, output, p); analyze(project, output, p)
        write_json(output/'records.json',records)
        write_json(output/'complete.json',dict(status='complete',environments=len(records),
            code_sha256=p['code_sha256'],protocol_sha256=fingerprint(p),at=now(),preflight=preflight))
        write_json(output/'progress.json',dict(status='complete',environments_complete=len(records),total=len(p['rows']),at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    try:
        run(args.project,args.output,args.preflight,args.workers)
    except BaseException:
        if args.output.exists():
            write_json(args.output/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
