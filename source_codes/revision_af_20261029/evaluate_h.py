"""全部验证环境上的 A/F 输入反演、候选数量及轨迹前缀诊断。"""
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
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import SOURCE, LEVELS, require_host, sha256, verify_sources, write_json, now, rng_for, atomic_npz, observation
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.paired_statistics import sufficient, metrics, FIELDS
from study_uniform64.controllers import Feedback64, audit as audit_trace
from our_method_response_control.train import precision
from our_method_response_control.physics import design, ridge_estimate, pilot_gain, measurement_noise, decode
from our_method_response_control.model import conditions
from baseline_common.feedback import FeedbackSession
from baseline_common.controls import codes
from baseline_codebook.method import optimize as scan_codebook
from dataset_protocol.registry import SOURCES

LOADINGS = [.001, .003, .01, .03, .1]
COUNTS = [1,4,16,48]
SEEDS = [0,1,2]
BUDGETS = [16,24,32,48]
DIRECT = [f'A_loading_{x:g}' for x in LOADINGS] + ['ridge_isotropic','ridge_noise_weighted','covariance']
FEEDBACK = [f'candidates_{k}_seed_{s}' for s in SEEDS for k in COUNTS]
METHODS = DIRECT + FEEDBACK + [f'prefix_{b}' for b in BUDGETS]
STATE = {}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def initialize(output, p):
    require_host(); precision(); verify_sources(p['source_sha256'])
    bundle = Path(p['runtime_bundle']); verify_bundle(bundle)
    assert sha256(bundle/'manifest.json') == p['runtime_manifest_sha256']
    with np.load(bundle/'public.npz',allow_pickle=False) as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    with patch('numpy.load',guard_load(np.load)):
        learned = OnlineController(bundle,'complex_response_cnn',public)
        conventional = OnlineController(bundle,'covariance_response',public)
    STATE.update(output=Path(output), protocol=p, public=public, learned=learned, conventional=conventional)
    write_json(Path(output)/'workers'/f'{os.getpid()}.json',dict(at=now(),pid=os.getpid(),
        gpu=torch.cuda.get_device_name(), artifacts=learned.artifacts | conventional.artifacts,
        joint_correction=False))


def weighted_initial(raw, pilots, loading, weighted=False):
    m = design(int(raw[1984])); a = m['sensing']
    g = a @ a.conj().transpose(0,2,1)
    scale = np.trace(g,axis1=1,axis2=2).real/16
    if weighted:
        r = measurement_noise(m,raw,pilots)
        diagonal = np.trace(r,axis1=1,axis2=2).real/16
        assert np.all(diagonal>0)
        penalty = loading*scale[:,None,None]*r/diagonal[:,None,None]
    else:
        penalty = loading*scale[:,None,None]*np.eye(16)
    y = pilot_gain(raw,pilots).T
    return np.einsum('knp,kp->nk',a.conj().transpose(0,2,1),np.linalg.solve(g+penalty,y[...,None])[...,0])


def predict(raw, initial):
    initial = initial[None].astype(np.complex64)
    c = conditions(raw[None],initial,STATE['public']['pilot_qpsk'])
    with torch.inference_mode():
        return STATE['learned'].model(torch.from_numpy(initial).cuda(),torch.from_numpy(c).cuda()).cpu().numpy()[0]


def feedback(raw, engine, row, carrier, proposals):
    p = STATE['public']; model = design(carrier)
    def measure(u, call):
        return engine.measure_detailed(u,rng_for(row['seed'],carrier,620,call))['score']
    session = FeedbackSession(model['cfg'],observation(raw,p),measure,64)
    seen = {tuple(codes(u,model['cfg'])) for u in session.controls}
    measured_models = 0; reused_models = 0
    start = time.perf_counter()
    for code in proposals:
        key = tuple(code)
        if key in seen:
            reused_models += 1
        elif session.remaining:
            session.evaluate(code/LEVELS); seen.add(key); measured_models += 1
    scan_codebook(session,rng_for(0,row['seed'],carrier,630))
    trace = np.asarray([codes(u,model['cfg']) for u in session.controls])
    scores = np.asarray(session.scores)
    assert trace.shape == (64,128) and len(np.unique(trace,axis=0)) == 64
    return dict(control=trace[scores.argmax()],trace=trace,scores=scores,
        cost=[64,max(0.,time.perf_counter()-start-session.simulation_feedback_seconds),session.simulation_feedback_seconds],
        measured_models=measured_models,reused_models=reused_models)


def compute(row, carrier):
    p, public = STATE['protocol'], STATE['public']
    path = Path(row['source_directory'])/'environment.json'
    assert sha256(path) == row['environment_sha256']
    env = read(path)
    raw, engine = public_observation(env,carrier,public,p['backend'])
    learned = STATE['learned']; pilot = public['pilot_qpsk']
    canonical = learned.decide(raw.copy(),rng_for(0,row['seed'],carrier,630),None)
    h = canonical['estimated_response']; base_code = canonical['control_code']
    initial_u = public['probe_controls'][int(raw[1985:2001].argmax())]
    old = ridge_estimate(raw,pilot)[0]
    np.testing.assert_allclose(weighted_initial(raw,pilot,.01),old,rtol=1e-9,atol=1e-20)
    control, cost = [], []
    for coefficient in LOADINGS:
        tick = time.perf_counter()
        # 默认系数复用冻结 lift 的运算顺序，确保基准逐位一致。
        lifted = old if coefficient == .01 else weighted_initial(raw,pilot,coefficient)
        estimated = predict(raw,lifted)
        u, _ = decode(estimated,carrier,initial_u)
        code = codes(u,design(carrier)['cfg'])
        if coefficient == .01:
            np.testing.assert_array_equal(estimated,h)
            np.testing.assert_array_equal(code,base_code)
        control.append(code); cost.append([16,time.perf_counter()-tick,0.])
    for weighted in (False,True):
        tick = time.perf_counter()
        estimated = weighted_initial(raw,pilot,.01,True) if weighted else old
        u, _ = decode(estimated,carrier,initial_u)
        control.append(codes(u,design(carrier)['cfg'])); cost.append([16,time.perf_counter()-tick,0.])
    conventional = STATE['conventional'].decide(raw.copy(),rng_for(0,row['seed'],carrier,630),None)
    control.append(conventional['control_code']); cost.append([16,conventional['software_seconds'],0.])
    arrays = {'public_X':raw}
    proposal_sets, starts, generation_times, trace_meta = [], [], [], []
    reference = None
    for seed in SEEDS:
        rng = rng_for(seed,row['seed'],carrier,910)
        proposals = [base_code]; seed_starts = [codes(initial_u,design(carrier)['cfg'])]
        times = [canonical['software_seconds']]
        for k in range(1,48):
            start_code = np.r_[rng.integers(0,77,64),np.zeros(64,dtype=int)]
            tick = time.perf_counter()
            u, _ = decode(h,carrier,start_code/LEVELS)
            times.append(time.perf_counter()-tick)
            proposals.append(codes(u,design(carrier)['cfg'])); seed_starts.append(start_code)
        proposals = np.asarray(proposals)
        proposal_sets.append(proposals); starts.append(seed_starts); generation_times.append(times)
        for k in COUNTS:
            name = f'candidates_{k}_seed_{seed}'
            result = reference if k == 1 and reference is not None else feedback(raw,engine,row,carrier,proposals[:k])
            if k == 1 and reference is None:
                reference = result
            control.append(result['control'])
            cost.append([64,result['cost'][1]+sum(times[:k]),result['cost'][2]])
            arrays[name+'__trace_control_code'] = result['trace']
            arrays[name+'__trace_scores'] = result['scores']
            arrays[name+'__proposal_control_code'] = base_code
            trace_meta.append(dict(method=name,model_candidates=k,actually_measured_models=result['measured_models'],
                reused_models=result['reused_models'],reused_single_candidate_evaluation=(k==1 and seed!=0)))
    for budget in BUDGETS:
        selected = int(reference['scores'][:budget].argmax())
        control.append(reference['trace'][selected]); cost.append([budget,0.,0.])
    arrays['candidate_codes'] = np.asarray(proposal_sets)
    arrays['candidate_start_codes'] = np.asarray(starts)
    arrays['candidate_generation_seconds'] = np.asarray(generation_times)
    if p['preflight']:
        def measure(u,call):
            return engine.measure_detailed(u,rng_for(row['seed'],carrier,620,call))['score']
        anchor = Feedback64(learned,public).decide(raw.copy(),rng_for(0,row['seed'],carrier,630),measure)
        np.testing.assert_array_equal(anchor['control_code'],reference['control'])
        np.testing.assert_array_equal(anchor['trace_control_code'],reference['trace'])
        np.testing.assert_array_equal(anchor['trace_scores'],reference['scores'])
    decisions = dict(methods=METHODS,control_code=np.asarray(control),costs=np.asarray(cost),feedback={},teacher_objective_evaluations=None)
    scored = score_decisions(env,carrier,public,decisions,p['backend'],include_mrc=False)
    arrays.update(control_code=scored['control_code'],metrics=scored['metrics'])
    expected = [16]*len(DIRECT)+[64]*len(FEEDBACK)+BUDGETS
    np.testing.assert_array_equal(arrays['metrics'][:,10],expected)
    subset = {k:v for k,v in arrays.items() if k.startswith(tuple(n+'__' for n in FEEDBACK))}
    selection = [METHODS.index(n) for n in FEEDBACK]
    subset.update(public_X=raw,control_code=arrays['control_code'][selection],metrics=arrays['metrics'][selection])
    audit_trace(subset,FEEDBACK,public,carrier)
    # 三个 seed 的 K=1 具有相同算法、噪声及输入；核对真实错误计数相同。
    for seed in (1,2):
        i,j = METHODS.index(f'candidates_1_seed_{seed}'),METHODS.index('candidates_1_seed_0')
        np.testing.assert_array_equal(arrays['metrics'][i,:11],arrays['metrics'][j,:11])
    if p['preflight']:
        repeat = score_decisions(env,carrier,public,decisions,p['backend'],include_mrc=False)
        np.testing.assert_array_equal(repeat['metrics'][:,:11],arrays['metrics'][:,:11])
    learned.verify(); STATE['conventional'].verify()
    return arrays, dict(default_A_equal=True,feedback_trace_audit=True,canonical_F_equal=p['preflight'],
        independent_payload_repeat_equal=p['preflight'],joint_correction=False,candidate_measurements=trace_meta)


def one(row):
    output,p = STATE['output'],STATE['protocol']
    folder = output/'records'/f"environment_{row['index']:05d}"
    folder.mkdir(parents=True,exist_ok=True); entries = []
    for carrier in p['carriers']:
        path = folder/f'carrier_{carrier:02d}.npz'; marker=path.with_suffix('.json')
        if marker.exists():
            e = read(marker)
            assert e['protocol_sha256'] == fingerprint(p) and sha256(path) == e['sha256']
        else:
            if path.exists():
                raise RuntimeError('Uncommitted record requires inspection: '+str(path))
            arrays,checks = compute(row,carrier)
            atomic_npz(path,**arrays)
            e = dict(environment_id=row['environment_id'],carrier_ghz=carrier,path=str(path.relative_to(output)),
                sha256=sha256(path),protocol_sha256=fingerprint(p),checks=checks)
            write_json(marker,e)
        entries.append(e)
    result = dict(index=row['index'],environment_id=row['environment_id'],carriers=entries)
    write_json(folder/'complete.json',result)
    return result


def run(project,output,workers,preflight):
    require_host()
    base = project/'dataset_simulation/revision_rq2_af_20261029/full'
    prior = read(base/'protocol.json')['prior']
    manifest,split,count,digest = SOURCES['validation']
    path = project/'dataset_simulation'/manifest
    assert sha256(path) == digest
    original = read(path); records = {r['environment_id']:r for r in read(path.parent/'records.json')}
    rows = []
    for row in original['environments']:
        if row['split'] != split: continue
        folder = path.parent/row['path']; record=records[row['environment_id']]
        expected = record['file_sha256']['environment.json']
        assert sha256(folder/'environment.json') == expected
        rows.append(dict(row,source_directory=str(folder),environment_sha256=expected,role='validation'))
    assert len(rows)==216 and len({r['seed'] for r in rows})==216
    source_hashes = dict(read(base/'protocol.json')['source_sha256'])
    source_hashes.update({str(n.relative_to(SOURCE)):sha256(n) for n in [Path(__file__),Path(__file__).with_name('H_PROTOCOL.md'),SOURCE/'dataset_protocol/registry.py']})
    verify_sources(source_hashes)
    p = dict(schema='AF-validation-sensitivity-v1',preflight=preflight,
        rows=rows[:1] if preflight else rows,carriers=[4,12,20] if preflight else list(range(4,21)),
        methods=METHODS,loadings=LOADINGS,candidate_counts=COUNTS,candidate_seeds=SEEDS,
        prefix_budgets=BUDGETS+[64],runtime_bundle=prior['runtime_bundle'],
        runtime_manifest_sha256=prior['runtime_manifest_sha256'],backend=prior['backend'],
        validation_manifest=str(path),validation_manifest_sha256=digest,
        source_sha256=source_hashes,code_sha256=sha256(Path(__file__)),joint_correction=False)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists(): raise RuntimeError('Already complete')
        if (output/'protocol.json').exists(): assert read(output/'protocol.json')==p
        write_json(output/'protocol.json',p)
        for name,h in source_hashes.items():
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): assert sha256(target)==h
            else: shutil.copyfile(SOURCE/name,target)
        (output/'workers').mkdir(exist_ok=True)
        if preflight:
            initialize(output,p); completed=[one(p['rows'][0])]
        else:
            proof=read(output.parent/'preflight'/'complete.json')
            assert proof['preflight'] and proof['source_sha256']==source_hashes
            completed=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(output,p)) as pool:
                for future in as_completed([pool.submit(one,row) for row in p['rows']]):
                    completed.append(future.result())
                    write_json(output/'progress.json',dict(status='running',completed=len(completed),total=len(p['rows']),pid=os.getpid(),at=now()))
                    print(f"H: {len(completed)}/{len(p['rows'])}",flush=True)
        completed.sort(key=lambda r:r['index'])
        values=[]
        for r in completed:
            block=[]
            for e in r['carriers']:
                f=output/e['path'];assert sha256(f)==e['sha256']
                with np.load(f,allow_pickle=False) as a: block.append(a['metrics'].copy())
            values.append(block)
        values=np.asarray(values)
        assert values.shape==(len(p['rows']),len(p['carriers']),len(METHODS),13)
        point=metrics(sufficient(values).sum(0))
        write_json(output/'summary.json',[dict(method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,n in enumerate(METHODS)])
        atomic_npz(output/'environment_metrics.npz',values=values,methods=np.asarray(METHODS),environment_ids=np.asarray([r['environment_id'] for r in completed]))
        write_json(output/'records.json',completed)
        verify_sources(source_hashes)
        write_json(output/'complete.json',dict(status='complete',preflight=preflight,environments=len(completed),
            carriers=len(p['carriers']),methods=len(METHODS),source_sha256=source_hashes,code_sha256=p['code_sha256'],
            protocol_sha256=fingerprint(p),joint_correction=False,at=now(),file_sha256={n:sha256(output/n) for n in ['protocol.json','records.json','summary.json','environment_metrics.npz']}))
        write_json(output/'progress.json',dict(status='complete',completed=len(completed),total=len(completed),at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    try: run(args.project.resolve(),args.output.resolve(),args.workers,args.preflight)
    except Exception:
        if args.output.exists(): write_json(args.output/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
