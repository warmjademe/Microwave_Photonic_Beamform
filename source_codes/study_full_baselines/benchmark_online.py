"""单条在线控制耗时：实际推理与搜索，逐次核对的反馈重放隔离仿真成本。"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# 在导入NumPy/PyTorch之前固定CPU线程；torch.set_num_threads不控制NumPy BLAS。
for _thread_variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_thread_variable] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
import study_full_baselines.online_controller as online
from baseline_common.controls import codes as legal_codes
from our_method_response_control.train import precision


class TraceReplay:
    """仅返回先前真实反馈接口的标量；查询顺序/控制码任一变化立即失败。"""
    def __init__(self, codes, scores):
        self.codes = np.asarray(codes); self.scores = np.asarray(scores)
        if self.codes.shape != (48, 128) or self.scores.shape != (48,):
            raise ValueError('必须保存48次追加测量，不能省略查询。')
        self.position = 0

    def __call__(self, control, call):
        i = self.position
        if i >= 48 or call != i+16:
            raise ValueError('反馈调用序号不同。')
        if not np.array_equal(legal_codes(control, NativeConfig()), self.codes[i]):
            raise ValueError('反馈查询的控制码不同；禁止拿旧分数响应新控制。')
        self.position += 1
        return float(self.scores[i])

    def finish(self):
        if self.position != 48: raise ValueError('没有执行全部48次追加查询。')


def cpu_ticks():
    values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3]+values[4]


def resource_snapshot(project):
    """仅存本项目进程身份和资源，不记录可能含凭据的完整命令行。"""
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                  '--format=csv,noheader,nounits'], text=True)
    gpu = []
    for line in raw.splitlines():
        pid, memory = [x.strip() for x in line.split(',')]
        gpu.append(dict(pid=int(pid), memory_mib=int(memory)))
    processes = []
    lines = subprocess.check_output(['ps', '-eo', 'pid=,pcpu=,stat=,comm=,args='], text=True).splitlines()
    for line in lines:
        fields = line.strip().split(None, 4)
        if len(fields) != 5: continue
        pid, cpu, state, command, args = fields
        if int(pid) == os.getpid() or 'python' not in command: continue
        try: cwd = os.readlink('/proc/'+pid+'/cwd')
        except OSError: cwd = ''
        if (str(project) in args or cwd.startswith(str(project))) and float(cpu) > 1.:
            processes.append(dict(pid=int(pid), cpu_percent=float(cpu), state=state, command=command))
    before = cpu_ticks(); time.sleep(.2); after = cpu_ticks()
    busy = 1-(after[1]-before[1])/max(1, after[0]-before[0])
    return dict(at=now(), load_average=list(os.getloadavg()), cpu_busy_fraction=busy,
                gpu_processes=gpu, project_numeric_processes=processes)


def resource_gate(snapshot, final):
    reasons = []
    if any(p['pid'] != os.getpid() for p in snapshot['gpu_processes']): reasons.append('GPU有其他进程')
    if snapshot['project_numeric_processes']: reasons.append('本项目其他数值任务仍在运行')
    # 不停止虚拟机或其他用户任务；正式计时如遇资源竞争则保留失败并等待重测。
    if snapshot['cpu_busy_fraction'] > .6: reasons.append('CPU背景忙碌比例超过60%')
    if final and reasons: raise RuntimeError('正式计时资源门限未通过：'+','.join(reasons))
    return reasons


def same_result(reference, result):
    if not np.array_equal(reference['control_code'], result['control_code']):
        raise ValueError('重复执行控制码发生变化。')
    if reference['feedback_calls'] != result['feedback_calls']: raise ValueError('测量次数改变。')
    for key in ['trace_control_code', 'trace_scores']:
        if key in reference and not np.array_equal(reference[key], result[key]):
            raise ValueError('反馈轨迹发生变化：'+key)


def segmented_call(model, raw, rng, callback):
    """另一次带探针的分段诊断；不混入不带探针的正式总时间。"""
    spans = dict(estimation=0., search=0.)
    def wrap(field, function):
        def wrapped(*args, **kwargs):
            tick = time.perf_counter()
            try: return function(*args, **kwargs)
            finally: spans[field] += time.perf_counter()-tick
        return wrapped
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'estimate', wrap('estimation', model.estimate)))
        for name in ['decode', 'decode_multistart', 'warm_optimize']:
            stack.enter_context(patch.object(online, name, wrap('search', getattr(online, name))))
        if model.kind == 'classic':
            stack.enter_context(patch.object(model, 'optimize', wrap('search', model.optimize)))
        result = model.decide(raw, rng, callback)
    search = max(0., spans['search']-result['feedback_simulator_seconds'])
    return result, dict(estimation_seconds=spans['estimation'] if model.kind in
        ['ridge', 'covariance', 'response', 'linear'] else None,
        search_seconds=search, input_inference_and_other_seconds=result['software_seconds']-search,
        instrumented_software_seconds=result['software_seconds'])


def negative_checks():
    code = np.zeros((48, 128), dtype=np.int16); score = np.arange(48.)
    checks = []
    def reject(name, fn):
        try: fn()
        except ValueError: checks.append(name)
        else: raise ValueError('负例未被拒绝：'+name)
    reject('wrong_call_order', lambda: TraceReplay(code, score)(code[0], 17))
    changed = code[0].astype(float); changed[0] = 1/LEVELS[0]
    reject('wrong_control_query', lambda: TraceReplay(code, score)(changed, 16))
    reject('missing_queries', lambda: TraceReplay(code, score).finish())
    reject('short_trace', lambda: TraceReplay(code[:-1], score))
    busy = dict(gpu_processes=[dict(pid=os.getpid()+10000000)], project_numeric_processes=[],
                cpu_busy_fraction=0.)
    try: resource_gate(busy, True)
    except RuntimeError: checks.append('final_mode_rejects_gpu_contention')
    else: raise ValueError('正式计时门限无效。')
    return checks


def run(project, output, preflight, final, scales):
    require_host(); precision()
    if preflight and final: raise ValueError('前置诊断不能标成正式计时。')
    if output.exists(): raise FileExistsError('不覆盖原耗时记录。')
    if not torch.cuda.is_available(): raise RuntimeError('要求华硕GPU。')
    data = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    manifest = check_data(data)
    test = sorted([r for r in manifest['environments'] if r['split'] == 'test'], key=lambda r: r['index'])
    rows = [test[0]] if preflight else [next(r for r in test if r['factors']['power_bin'] == power)
                                      for power in range(6)]
    carriers = [4, 12, 20] if preflight else list(range(4, 21))
    methods = ['ttd_das', 'mlp', 'complex_response_cnn', 'codebook', 'spsa', 'cnn_warm64'] if preflight else online.method_names(scales)
    records = {r['environment_id']: r for r in json.loads((data/'records.json').read_text())}
    inputs = {}; identities = {}
    for row in rows:
        hashes = records[row['environment_id']]['file_sha256']
        for name, digest in hashes.items():
            if sha256(data/row['path']/name) != digest: raise ValueError('计时输入来源改变。')
        with np.load(data/row['path']/'data.npz') as f: inputs[row['index']] = f['X'].copy()
        identities[row['environment_id']] = hashes
    with np.load(data/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    names = ['study_full_baselines/'+n for n in ['online_controller.py', 'benchmark_online.py',
             'ONLINE_PROTOCOL.md', 'ONLINE_TIMING_PROTOCOL.md', 'common.py']]
    folders = ['deep_common', 'baseline_common', 'baseline_mlp', 'our_method_quality_rank',
        'our_method_response_control', 'baseline_response_realcnn', 'baseline_joint_response_linear',
        'our_method_two_stage', 'our_method_feedback_candidates']+['baseline_'+n for n in ONLINE_CLASSIC+DEEP_METHODS]
    names += [str(f.relative_to(SOURCE)) for folder in folders for f in (SOURCE/folder).glob('*.py')]
    sources = source_record(names); snapshot = resource_snapshot(project)
    output.mkdir(parents=True); (output/'records').mkdir()
    write_json(output/'resource_start.json', snapshot)
    resource_gate(snapshot, final)
    for name in sorted(set(names)):
        dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    protocol = dict(methods=methods, rows=rows, carriers=carriers, repetitions=3, batch_size=1,
        data_manifest_sha256=sha256(data/'manifest.json'), public_sha256=sha256(data/'public.npz'),
        input_file_sha256=identities, source_sha256=sources, final_fair_timing=final, preflight=preflight,
        source_selection='First old-test environment per power bin, independent of quality; preflight uses index0.',
        scope='Actual online weights and searches; callback duration excluded. No cached model predictions.',
        cpu_threads=torch.get_num_threads(), cpu_thread_environment={name: os.environ[name] for name in
            ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
        host=platform.node(), python=sys.version,
        numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), measurement_seconds=NativeConfig().measurement_s,
        switch_seconds=NativeConfig().switch_s, training_repeats_added=0, at=now())
    write_json(output/'protocol.json', protocol)
    # 评分端准备实测接口；这些成本和环境真值均不进入控制器或计时输入。
    engines = {}; fixture_times = []
    for row in rows:
        env = json.loads((data/row['path']/'environment.json').read_text())
        for fc in carriers:
            tick = time.perf_counter()
            engines[row['index'], fc] = frame_engine(env, fc, public['pilot_qpsk'], 0)[0]
            fixture_times.append(dict(environment_id=row['environment_id'], carrier_ghz=fc,
                                      simulator_preparation_seconds=time.perf_counter()-tick))
    write_json(output/'feedback_fixture.json', dict(records=fixture_times, included_in_online_time=False))
    old_load = np.load; loaded = []
    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.name in ['data.npz', 'predictions.npy', 'predicted_response.npy', 'train_response.npy']:
                raise RuntimeError('计时控制器不能读取数据文件或预测缓存。')
            loaded.append(str(path))
        return old_load(path, *args, **kwargs)
    checks = negative_checks(); all_records = []; summaries = []
    for ni, name in enumerate(methods):
        snapshot = resource_snapshot(project); reasons = resource_gate(snapshot, final)
        tick = time.perf_counter()
        with patch('numpy.load', guarded_load): model = online.OnlineController(project, name, public)
        if model.uses_cuda: torch.cuda.synchronize()
        load_seconds = time.perf_counter()-tick
        extra = model.warm or (model.kind == 'classic' and name != 'ttd_das')
        method_records = []
        for row in rows:
            for fc in carriers:
                raw = inputs[row['index']][fc-4]; engine = engines[row['index'], fc]
                query_codes = []; query_scores = []
                def measure(control, call):
                    if call != 16+len(query_codes): raise ValueError('真实反馈调用序号错误。')
                    score = engine.measure_detailed(control, rng_for(row['seed'], fc, 620, call))['score']
                    query_codes.append(legal_codes(control, NativeConfig())); query_scores.append(score)
                    return score
                with patch('numpy.load', guarded_load):
                    reference = model.decide(raw, rng_for(0, row['seed'], fc, 630), measure if extra else None)
                    def execute(segment=False):
                        replay = TraceReplay(query_codes, query_scores) if extra else None
                        rng = rng_for(0, row['seed'], fc, 630)
                        result, spans = segmented_call(model, raw, rng, replay) if segment else (model.decide(raw, rng, replay), None)
                        if replay: replay.finish()
                        same_result(reference, result)
                        return result, spans
                    # 两次真实计算热身；之后三次计时，不增加训练随机种子。
                    for _ in range(2): execute()
                    timed = [execute()[0] for _ in range(3)]
                    segmented, spans = execute(True)
                path = output/'records'/('%s_env%05d_fc%02d.npz' % (name, row['index'], fc))
                arrays = dict(control_code=reference['control_code'],
                    software_seconds=np.asarray([r['software_seconds'] for r in timed]),
                    replay_callback_seconds=np.asarray([r['feedback_simulator_seconds'] for r in timed]),
                    total_wall_seconds=np.asarray([r['total_wall_seconds'] for r in timed]))
                if extra: arrays.update(query_control_code=np.asarray(query_codes), query_scores=np.asarray(query_scores))
                atomic_npz(path, **arrays)
                record = dict(method=name, environment_id=row['environment_id'], index=row['index'], carrier_ghz=fc,
                    measurement_calls=reference['feedback_calls'],
                    real_feedback_simulator_seconds=reference['feedback_simulator_seconds'],
                    real_callback_software_seconds_diagnostic=reference['software_seconds'],
                    timing_repetitions=3, repeated_controls_and_all_queries_identical=True,
                    **spans, file=str(path.relative_to(output)), sha256=sha256(path))
                method_records.append(record); all_records.append(record)
                write_json(output/'progress.json', dict(status='running', method=name, completed_cases=len(all_records),
                    total_cases=len(methods)*len(rows)*len(carriers), pid=os.getpid(), at=now()))
        model.verify(); after = resource_snapshot(project); resource_gate(after, final)
        collected_times = []
        for record in method_records:
            with np.load(output/record['file']) as arrays: collected_times.append(arrays['software_seconds'].copy())
        values = np.concatenate(collected_times)
        calls = sorted(set(r['measurement_calls'] for r in method_records))
        if len(calls) != 1: raise ValueError('方法内部测量预算不一致。')
        nominal = calls[0]*NativeConfig().measurement_s+NativeConfig().switch_s
        result = dict(method=name, cases=len(method_records), time_samples=len(values), measurement_calls=calls[0],
            mean_software_ms=float(values.mean()*1000), median_software_ms=float(np.median(values)*1000),
            p95_software_ms=float(np.quantile(values, .95)*1000), estimated_measurement_switch_ms=nominal*1000,
            estimated_total_mean_ms=float((values.mean()+nominal)*1000), load_seconds=load_seconds,
            model_artifacts=model.artifacts, cuda=model.uses_cuda, load_before=snapshot, load_after=after,
            contention_reasons=reasons, segmented_timings_are_separate_instrumented_calls=True)
        summaries.append(result); write_json(output/(name+'.json'), result)
        print(json.dumps({k: result[k] for k in ['method', 'cases', 'mean_software_ms', 'p95_software_ms']}, ensure_ascii=False), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    verify_sources(sources)
    for summary in summaries:
        for path, digest in summary['model_artifacts'].items():
            if sha256(path) != digest: raise ValueError('计时后权重或统计参数发生变化。')
    write_json(output/'summary.json', dict(status='complete', final_fair_timing=final,
        methods=summaries, cases=all_records, negative_checks=checks, numerical_artifacts_loaded=sorted(set(loaded)),
        cases_count=len(all_records), protocol_sha256=sha256(output/'protocol.json'), at=now()))
    write_json(output/'progress.json', dict(status='complete', methods=len(methods), cases=len(all_records),
        final_fair_timing=final, at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true'); parser.add_argument('--final-idle', action='store_true')
    parser.add_argument('--scales', type=int, nargs='*', default=[216, 432, 1728, 3456])
    args = parser.parse_args()
    try: run(args.project.resolve(), args.output.resolve(), args.preflight, args.final_idle, args.scales)
    except BaseException:
        if args.output.exists(): write_json(args.output/('failure_%d.json' % time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
