"""在已固定的训练案例上检验有限反馈候选，不打开测试环境。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from baseline_common.feedback import FeedbackSession
from baseline_codebook.method import optimize as codebook
from our_method_feedback_candidates.method import optimize

METHODS = ['codebook64', 'covariance_warm64', 'covariance_candidates64',
           'cnn_warm64', 'cnn_candidates64', 'covariance_base16',
           'cnn_base16', 'cnn_multi16']
REFERENCE_INDICES = [0, 4, 6]
SOURCES = ['our_method_feedback_candidates/'+n for n in
           ['method.py', 'diagnose.py', 'PROTOCOL.md']]
SOURCES += ['study_full_baselines/common.py', 'baseline_common/feedback.py',
            'baseline_common/controls.py', 'baseline_codebook/method.py',
            'our_method_response_control/physics.py', 'our_method_two_stage/control.py',
            'native_sim/control_engine.py', 'our_method_quality_rank/generate.py']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_cache(data, cache):
    manifest = check_data(data)
    protocol = json.loads((cache/'protocol.json').read_text())
    complete = json.loads((cache/'complete.json').read_text())
    if (protocol['test_used'] or protocol['frame'] != 5 or protocol['apd_draws'] != 8
            or protocol['carriers_ghz'] != [4, 12, 20]
            or protocol['metric_order'] != QUALITY_METRICS
            or protocol['data_manifest_sha256'] != sha256(data/'manifest.json')
            or complete['fingerprint'] != fingerprint(protocol)
            or complete['results_sha256'] != sha256(cache/'all_metrics.npz')):
        raise ValueError('训练诊断缓存身份或完成状态不一致。')
    verify_sources(protocol['source_sha256'])
    train = {r['environment_id']: r for r in manifest['environments'] if r['split']=='train'}
    rows = [train[eid] for eid in protocol['training_environment_ids']]
    if len(rows) != 24 or len(set(protocol['training_environment_ids'])) != 24:
        raise ValueError('训练环境数量或去重核验失败。')
    cases = []
    for i in range(72):
        path = cache/'records'/('%03d.npz'%i)
        marker = json.loads(path.with_suffix('.json').read_text())
        row = rows[i//3]; fc = [4, 12, 20][i%3]
        if (marker['fingerprint'] != complete['fingerprint']
                or marker['sha256'] != sha256(path)
                or marker['environment_id'] != row['environment_id']
                or marker['carrier_ghz'] != fc):
            raise ValueError('逐案例缓存来源不一致。')
        cases.append((row, fc, path, marker['sha256']))
    return protocol, cases


def calculate(data, public, row, fc, path):
    with np.load(path) as f:
        x = f['public_x']; h = f['response_estimate']
        reference_codes = f['control_code'][REFERENCE_INDICES]
        reference_metrics = f['metrics'][REFERENCE_INDICES]
        reference_seconds = f['seconds'][REFERENCE_INDICES]
    with np.load(data/row['path']/'data.npz') as f:
        if not np.array_equal(x, f['X'][fc-4]):
            raise ValueError('缓存公开输入与原数据不一致。')
    if h.shape != (2, 64, 31) or not np.isfinite(h).all():
        raise ValueError('响应估计形状或数值有误。')
    # 环境真值仅交给反馈/评分器，不交给控制器；控制器接收缓存公开估计。
    env = json.loads((data/row['path']/'environment.json').read_text())
    observed, _ = frame_engine(env, fc, public['pilot_qpsk'], 0)
    obs = observation(x, public)
    controls=[]; seconds=[]; simulator_seconds=[]; infos=[]; traces=[]; scores=[]
    for mi in range(5):
        def measure(u, call):
            return observed.measure_detailed(u, rng_for(row['seed'], fc, 620, call))['score']
        session = FeedbackSession(observed.cfg, obs, measure, 64)
        rng = rng_for(0, row['seed'], fc, 630)
        tick = time.perf_counter()
        if mi == 0:
            control = codebook(session, rng)
            info = dict(mode='original_codebook', total_feedback_calls=session.calls)
        else:
            mode = 'warm_codebook' if mi in [1, 3] else 'response_candidates'
            control, info = optimize(session, rng, h[0 if mi < 3 else 1], fc, mode)
        elapsed = time.perf_counter()-tick
        codes = np.rint(np.asarray(session.controls)*LEVELS).astype(np.int16)
        choice = np.rint(control*LEVELS).astype(np.int16)
        if (session.calls != 64 or codes.shape != (64, 128)
                or np.any(codes < 0) or np.any(codes > LEVELS)
                or not np.array_equal(choice, codes[int(np.argmax(session.scores))])
                or not np.array_equal(np.asarray(session.scores)[:16], obs['quality'])):
            raise ValueError('反馈预算、档位、共同初始输入或已测最优选择核验失败。')
        controls.append(control); seconds.append(max(0.,elapsed-session.simulation_feedback_seconds))
        simulator_seconds.append(session.simulation_feedback_seconds)
        traces.append(codes); scores.append(session.scores); infos.append(info)
    controls.extend(reference_codes/LEVELS)
    seconds.extend(reference_seconds); simulator_seconds.extend([0.]*3)
    engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
    clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
    metrics, codes = reception_metrics(engine, clean, controls, payload, row['seed'], fc)
    if (not np.array_equal(codes[5:], reference_codes)
            or not np.allclose(metrics[5:], reference_metrics, rtol=1e-12, atol=1e-30)):
        raise ValueError('16测量参考未复现前次相同控制的完整接收指标。')
    return dict(metrics=metrics, control_code=codes, controller_seconds=np.asarray(seconds),
                feedback_simulator_seconds=np.asarray(simulator_seconds),
                feedback_calls=np.asarray([64]*5+[16]*3), trace_control_code=np.asarray(traces),
                trace_scores=np.asarray(scores)), infos


def summarize(arrays):
    result=[]
    for mi, method in enumerate(METHODS):
        a=arrays['metrics'][:,mi]; nmse=float(a[:,2].mean())
        result.append(dict(method=method, ber=float(a[:,0].sum()/a[:,1].sum()),
            ser=float(a[:,3].sum()/a[:,4].sum()), block_error_rate=float(a[:,5].sum()/a[:,6].sum()),
            rms_evm_percent=float(100*np.sqrt(nmse)), mean_nmse=nmse,
            paired_output_snr_db=float(10*np.log10(a[:,7].sum()/a[:,8].sum())),
            mean_controller_seconds=float(arrays['controller_seconds'][:,mi].mean()),
            mean_feedback_calls=float(arrays['feedback_calls'][:,mi].mean()),
            response_estimation_time_included=False))
    return result


def run(data, cache, output, preflight_only):
    require_host()
    cached_protocol, cases = verify_cache(data, cache)
    public = public_data(data)
    protocol = dict(methods=METHODS, metric_order=QUALITY_METRICS,
        scope='training-only mechanism diagnostic; not independent test evidence',
        test_used=False, frame=5, draws=8, algorithm_seed=0,
        cache_protocol_sha256=sha256(cache/'protocol.json'),
        cached_response_weight_sha256=cached_protocol['weight_sha256'],
        cases=[dict(environment_id=r['environment_id'], carrier_ghz=fc, cache_sha256=sha)
               for r,fc,_,sha in cases],
        data_manifest_sha256=sha256(data/'manifest.json'), source_sha256=source_record(SOURCES),
        timing='stage B only; cached stage A response; simulator feedback time separate')
    fp=fingerprint(protocol)
    if preflight_only:
        if output.exists(): raise FileExistsError('前置核验文件不覆盖。')
        row,fc,path,_=cases[0]
        arrays,infos=calculate(data,public,row,fc,path)
        # 独立重放第一条新增测量，检查反馈抽样身份和保存轨迹。
        env=json.loads((data/row['path']/'environment.json').read_text())
        engine,_=frame_engine(env,fc,public['pilot_qpsk'],0)
        replay=[]
        for mi in range(5):
            u=arrays['trace_control_code'][mi,16]/LEVELS
            score=engine.measure_detailed(u,rng_for(row['seed'],fc,620,16))['score']
            if score != arrays['trace_scores'][mi,16]:
                raise ValueError('反馈轨迹逐点重放不一致。')
            replay.append(True)
        verify_sources(protocol['source_sha256'])
        write_json(output,dict(status='passed',at=now(),environment_id=row['environment_id'],
            carrier_ghz=fc,legal_codes=True,budget64=True,common_initial16=True,
            measured_best_returned=True,frame5_reference_replay=True,feedback_replay=replay,
            candidate_details=infos,source_sha256=protocol['source_sha256']))
        print(json.dumps(dict(status='passed',checks='budget/legal codes/shared input/feedback replay/frame5 reference')),flush=True)
        return
    output.mkdir(parents=True,exist_ok=True);(output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text()) != protocol:
            raise ValueError('运行协议已经改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in SOURCES:
            dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dest)
    started=time.perf_counter(); all_arrays=[]
    for i,(row,fc,cache_path,cache_sha) in enumerate(cases):
        path=output/'records'/('%03d.npz'%i);marker=path.with_suffix('.json')
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['fingerprint'] != fp or old['sha256'] != sha256(path):
                raise ValueError('已提交结果来源或内容改变。')
            with np.load(path) as f: arrays={k:f[k] for k in f.files}
        else:
            if path.exists(): raise ValueError('存在未提交文件，需核查。')
            if sha256(cache_path) != cache_sha: raise ValueError('训练缓存已改变。')
            arrays,infos=calculate(data,public,row,fc,cache_path)
            atomic_npz(path,**arrays)
            write_json(marker,dict(fingerprint=fp,sha256=sha256(path),at=now(),
                environment_id=row['environment_id'],carrier_ghz=fc,candidate_details=infos))
        all_arrays.append(arrays)
        progress=dict(status='running',completed=i+1,total=72,pid=os.getpid(),
                      seconds=time.perf_counter()-started,at=now())
        write_json(output/'progress.json',progress)
        if (i+1)%3==0: print(json.dumps(progress),flush=True)
    arrays={k:np.asarray([a[k] for a in all_arrays]) for k in all_arrays[0]}
    verify_sources(protocol['source_sha256']);verify_cache(data,cache)
    atomic_npz(output/'all_metrics.npz',**arrays)
    write_json(output/'summary.json',dict(status='complete',scope=protocol['scope'],
        records=summarize(arrays),at=now()))
    write_json(output/'complete.json',dict(status='complete',fingerprint=fp,
        results_sha256=sha256(output/'all_metrics.npz'),at=now()))
    write_json(output/'progress.json',dict(status='complete',completed=72,total=72,at=now()))
    print(json.dumps(summarize(arrays)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','cache','output']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    try: run(a.data,a.cache,a.output,a.preflight_only)
    except BaseException:
        if not a.preflight_only:
            a.output.mkdir(parents=True,exist_ok=True)
            write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
