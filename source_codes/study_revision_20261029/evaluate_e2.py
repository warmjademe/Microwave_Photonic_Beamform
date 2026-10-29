"""E2独立种子接收评价：公开观测→16/64次控制→同一冻结接收评分器。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_revision_20261029 import run_e1 as e1
from study_full_baselines.common import require_host, sha256, write_json, now, atomic_npz, rng_for, verify_sources
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.confirmation_receiver import public_observation, score_decisions
from our_method_joint_refinement.online_full import JointController
from our_method_response_control.train import precision
from study_uniform64.controllers import Feedback64, audit as audit_trace
from study_full_baselines.paired_statistics import sufficient, metrics, FIELDS

NAMES = ['A_direct16','AB_direct16','A_feedback64','AB_feedback64']
STATE = {}


def initialize(project, out, protocol):
    require_host(); precision()
    prior = protocol['prior']; bundle = Path(prior['runtime_bundle'])
    verify_sources(prior['source_sha256']); verify_bundle(bundle)
    for path, digest in protocol['artifact_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen E2 artifact changed: '+path)
    with np.load(bundle/'public.npz') as f:
        public = {k:f[k].copy() for k in ['pilot_qpsk','probe_controls','catalog_controls']}
    with patch('numpy.load', guard_load(np.load)):
        base = OnlineController(bundle, 'complex_response_cnn', public)
        if protocol['seed']:
            weights = protocol['weights']
            base.model.load_state_dict(torch.load(weights, map_location='cuda', weights_only=True)); base.model.eval()
            base.artifacts[weights] = sha256(weights)
        spatial = np.load(protocol['spatial']); frequency = np.load(protocol['frequency'])
        joint = JointController(base, 'cnn__joint_full_noise', spatial, frequency, True)
        models = [base, joint, Feedback64(base, public), Feedback64(joint, public)]
    STATE.update(project=Path(project), output=Path(out), protocol=protocol, models=models, public=public)
    # Only use E1's audited historical-file reader; its compute function is unchanged.
    e1.STATE.update(protocol=protocol)


def compute(row, fc):
    p = STATE['protocol']; public = STATE['public']
    env, old, digest = e1.original(row, fc)
    raw, engine = public_observation(env, fc, public, p['prior']['backend'])
    np.testing.assert_array_equal(raw, old['public_X'])
    cache = {}
    def measure(u, call):
        key = (call, tuple(engine.codes(u)))
        if key not in cache:
            cache[key] = engine.measure_detailed(u, rng_for(row['seed'],fc,620,call))['score']
        return cache[key]
    controls, costs = [], []
    traces = {}
    with patch('numpy.load', guard_load(np.load)):
        for i, (name, model) in enumerate(zip(NAMES, STATE['models'])):
            got = model.decide(raw.copy(), rng_for(0,row['seed'],fc,630), measure if i>=2 else None)
            controls.append(got['control_code'])
            costs.append([got['feedback_calls'],got['software_seconds'],got['feedback_simulator_seconds']])
            if i>=2:
                for field in ['trace_control_code','trace_scores','proposal_control_code']:
                    traces[name+'__'+field] = got[field]
                np.testing.assert_array_equal(got['proposal_control_code'],controls[i-2])
    decision = dict(methods=NAMES, control_code=np.asarray(controls), costs=np.asarray(costs),
                    feedback={},teacher_objective_evaluations=None)
    scored = score_decisions(env,fc,public,decision,p['prior']['backend'],include_mrc=False)
    feedback_arrays = dict(public_X=raw,control_code=scored['control_code'][2:],metrics=scored['metrics'][2:],**traces)
    audit_trace(feedback_arrays,NAMES[2:],public,fc)
    np.testing.assert_array_equal(scored['metrics'][:2,10],[16,16])
    checked = dict(public_input_bitwise_equal=True,feedback_proposal_matches_direct=True)
    if p['seed']==0:
        for i, label in enumerate(['complex_response_cnn','cnn__joint_full_noise','cnn_warm64']):
            ix = p['prior']['methods'].index(label)
            np.testing.assert_array_equal(scored['control_code'][i],old['control_code'][ix])
            np.testing.assert_allclose(scored['metrics'][i,:10],old['metrics'][ix,:10],rtol=1e-12,atol=1e-28)
        f = STATE['project']/'dataset_simulation/revision_20261029/e1_full/records'/f"environment_{row['index']:05d}"/f'carrier_{fc:02d}.npz'
        meta=e1.read(f.with_suffix('.json'))
        if sha256(f)!=meta['sha256']:
            raise ValueError('E1 full-pipeline reference changed.')
        with np.load(f) as a:
            np.testing.assert_array_equal(scored['control_code'][3],a['control_code'][0])
            np.testing.assert_allclose(scored['metrics'][3,:10],a['metrics'][0,:10],rtol=1e-12,atol=1e-28)
        checked['all_four_original_configurations_reproduced']=True
    return dict(public_X=raw,control_code=scored['control_code'],metrics=scored['metrics'],**traces),dict(
        original_sha256=digest,checks=checked)


def one(row):
    out=STATE['output'];p=STATE['protocol']; fp=e1.fingerprint(p)
    folder=out/'records'/f"environment_{row['index']:05d}";folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for fc in p['carriers']:
        f=folder/f'carrier_{fc:02d}.npz'; mark=f.with_suffix('.json')
        if mark.exists():
            entry=e1.read(mark)
            if entry['protocol_sha256']!=fp or sha256(f)!=entry['sha256']:
                raise ValueError('E2 resume identity mismatch.')
        else:
            if f.exists():raise ValueError('Uncommitted E2 output requires inspection.')
            arrays,checks=compute(row,fc);atomic_npz(f,**arrays)
            entry=dict(environment_id=row['environment_id'],carrier_ghz=fc,path=str(f.relative_to(out)),
                sha256=sha256(f),protocol_sha256=fp,at=now(),**checks)
            write_json(mark,entry)
        entries.append(entry)
    result=dict(environment_id=row['environment_id'],index=row['index'],carriers=entries,at=now())
    write_json(folder/'complete.json',result);return result


def summarize(out,p):
    values=[]
    for row in p['rows']:
        current=[]
        for fc in p['carriers']:
            f=out/'records'/f"environment_{row['index']:05d}"/f'carrier_{fc:02d}.npz'
            if sha256(f)!=e1.read(f.with_suffix('.json'))['sha256']:raise ValueError('Evaluation data changed.')
            with np.load(f) as a:current.append(a['metrics'].copy())
        values.append(current)
    values=np.asarray(values);point=metrics(sufficient(values).sum(0))
    write_json(out/'summary.json',[dict(seed=p['seed'],method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)}) for i,n in enumerate(NAMES)])
    atomic_npz(out/'environment_metrics.npz',values=values,methods=np.asarray(NAMES),
               environment_ids=np.asarray([r['environment_id'] for r in p['rows']]))


def run(project,output,seed,preflight,workers):
    require_host(); source=project/'dataset_simulation/baseline_results/20260926_final864_selected'
    prior=e1.read(source/'protocol.json');root=project/'dataset_simulation/revision_20261029'
    if seed==0:
        if not preflight:raise ValueError('Seed 0 already evaluated; only calibration replay is needed.')
        bound=prior['joint_refinement']
        weights=None
        spatial=str(Path(bound['spatial_binding']['path'])/'cnn_residual_covariance.npy')
        frequency=str(Path(bound['path'])/'cnn_frequency_covariance.npy')
        artifacts={spatial:sha256(spatial),frequency:sha256(frequency)}
    else:
        folder=root/'e2_training'/('seed_%d'%seed); complete=e1.read(folder/'complete.json')
        cp=e1.read(folder/'protocol.json')
        if complete['fingerprint']!=e1.fingerprint(cp) or cp['test_used'] or cp['seed']!=seed:
            raise ValueError('Training protocol mismatch.')
        artifacts={str(folder/n):h for n,h in complete['file_sha256'].items()}
        artifacts[str(folder/'complete.json')]=sha256(folder/'complete.json')
        artifacts[str(folder/'protocol.json')]=sha256(folder/'protocol.json')
        weights=str(folder/'weights.pt');spatial=str(folder/'cnn_residual_covariance.npy');frequency=str(folder/'cnn_frequency_covariance.npy')
    p=dict(schema='sensors-revision-E2-eval-v1',seed=seed,source=str(source),prior=prior,
        source_protocol_sha256=sha256(source/'protocol.json'),weights=weights,spatial=spatial,frequency=frequency,
        artifact_sha256=artifacts,code_sha256=sha256(Path(__file__)),e1_code_sha256=sha256(Path(e1.__file__)),
        methods=NAMES,rows=prior['rows'][:1] if preflight else prior['rows'],
        carriers=[4,12,20] if preflight else prior['carriers'],preflight=preflight,
        scope='fixed final checkpoint; no selection on previously evaluated test set')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'protocol.json').exists() and e1.read(output/'protocol.json')!=p:raise ValueError('Frozen E2 protocol changed.')
        write_json(output/'protocol.json',p)
        if preflight:
            initialize(project,output,p);records=[one(p['rows'][0])]
            assert one(p['rows'][0])['carriers']==records[0]['carriers']
        else:
            proof=e1.read(root/'e2_preflight/complete.json')
            if proof['code_sha256']!=p['code_sha256'] or not proof['preflight']:raise ValueError('Current E2 code lacks preflight.')
            records=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(project,output,p)) as pool:
                for future in as_completed([pool.submit(one,row) for row in p['rows']]):
                    records.append(future.result())
                    write_json(output/'progress.json',dict(status='evaluating',seed=seed,completed=len(records),total=len(p['rows']),at=now()))
                    print(f'seed {seed}: {len(records)}/{len(p["rows"])}',flush=True)
        records.sort(key=lambda r:r['index']);summarize(output,p);write_json(output/'records.json',records)
        write_json(output/'complete.json',dict(status='complete',seed=seed,environments=len(records),
            preflight=preflight,code_sha256=p['code_sha256'],protocol_sha256=e1.fingerprint(p),at=now()))
        write_json(output/'progress.json',dict(status='complete',seed=seed,completed=len(records),total=len(p['rows']),at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[0,1,2],required=True);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--workers',type=int,default=2);a=parser.parse_args()
    try:run(a.project.resolve(),a.output.resolve(),a.seed,a.preflight,a.workers)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'failure.json',dict(at=now(),traceback=traceback.format_exc()));raise
