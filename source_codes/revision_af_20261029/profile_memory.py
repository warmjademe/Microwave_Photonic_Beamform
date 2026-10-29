"""独立记录单个 A+F 控制器的内存；不改变冻结时延基准。"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import traceback
from unittest.mock import patch

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from study_full_baselines.common import require_host, sha256, verify_sources, write_json, now, rng_for
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_batch import guard_load
from study_full_baselines.benchmark_online import TraceReplay, resource_gate
from study_final_timing.run import timing_resource_snapshot
from study_uniform64.controllers import Feedback64
from our_method_response_control.train import precision


def read(path):
    return json.loads(Path(path).read_text())


def memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            name, number, unit = line.split()
            assert unit == 'kB'
            values[name.rstrip(':') + '_mib'] = int(number) / 1024
    values['process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    values['cuda_allocated_mib'] = torch.cuda.memory_allocated() / 2**20
    values['cuda_reserved_mib'] = torch.cuda.memory_reserved() / 2**20
    values['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    values['cuda_peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    own = []
    for line in result.splitlines():
        pid, used = (s.strip() for s in line.split(','))
        if int(pid) == os.getpid():
            own.append(int(used))
    assert len(own) == 1, own
    values['driver_process_gpu_mib'] = own[0]
    return values


def run(project, output, preflight):
    require_host()
    precision()
    assert not output.exists(), output
    here = Path(__file__).resolve().parent
    identity = {name: sha256(here / name) for name in ['profile_memory.py', 'P_MEMORY_PROTOCOL.md']}
    timing = project / 'dataset_simulation/revision_af_20261029/P/timing_full'
    if not preflight:
        done = read(timing / 'complete.json')
        assert done['status'] == 'passed' and done['formal_timing'] and not done['preflight']
        for name, digest in done['file_sha256'].items():
            assert sha256(timing / name) == digest
        timing_protocol_hash = sha256(timing / 'protocol.json')
    else:
        timing_protocol_hash = None
    anchor = project / 'dataset_simulation/revision_rq2_af_20261029/full'
    protocol = read(anchor / 'protocol.json')
    verify_sources(protocol['source_sha256'])
    rows = [protocol['rows'][0]] if preflight else [
        next(row for row in protocol['rows'] if row['factors']['power_bin'] == group) for group in range(6)]
    carriers = [4, 12, 20] if preflight else list(range(4, 21))
    if not preflight:
        timed = read(timing / 'protocol.json')
        assert rows == timed['rows'] and carriers == timed['carriers']
    output.mkdir(parents=True)
    for name in identity:
        shutil.copy2(here / name, output / name)
    snapshots = []
    def check_resources(phase):
        state = timing_resource_snapshot(project, preflight)
        state['phase'] = phase
        snapshots.append(state)
        write_json(output / 'resources.json', snapshots)
        resource_gate(state, not preflight)
    try:
        check_resources('start')
        prior = protocol['prior']
        bundle = Path(prior['runtime_bundle'])
        verify_bundle(bundle)
        assert sha256(bundle / 'manifest.json') == prior['runtime_manifest_sha256']
        with np.load(bundle / 'public.npz', allow_pickle=False) as values:
            public = {key: values[key].copy() for key in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
        with patch('numpy.load', guard_load(np.load)):
            base = OnlineController(bundle, 'complex_response_cnn', public)
            model = Feedback64(base, public)
        torch.cuda.synchronize()
        assert torch.get_num_threads() == 1
        loaded_memory = memory()
        parameters = list(base.model.parameters())
        buffers = list(base.model.buffers())
        parameter_count = sum(value.numel() for value in parameters)
        parameter_bytes = sum(value.numel() * value.element_size() for value in parameters)
        buffer_bytes = sum(value.numel() * value.element_size() for value in buffers)
        metadata = dict(at=now(), preflight=preflight, formal_memory=not preflight,
                        source_sha256=identity, anchor_protocol_sha256=sha256(anchor / 'protocol.json'),
                        timing_protocol_sha256=timing_protocol_hash, rows=rows, carriers=carriers,
                        method='cnn_warm64', initial_warmups=2, repetitions=1,
                        joint_correction=False, parameter_count=parameter_count,
                        parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes,
                        loaded_memory=loaded_memory, torch=torch.__version__, numpy=np.__version__,
                        cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(),
                        model_artifacts=model.artifacts,
                        scope='Single-controller Python replay process; RSS includes imports and record reading. '
                              'CUDA allocator peak excludes context/driver; driver values are samples, not peaks.')
        write_json(output / 'protocol.json', metadata)
        records = []
        hashes = {}
        warmed = False
        for row in rows:
            for carrier in carriers:
                folder = anchor / 'records' / f"environment_{row['index']:05d}"
                marker = read(folder / 'complete.json')
                entry = next(x for x in marker['carriers'] if x['carrier_ghz'] == carrier)
                path = anchor / entry['path']
                assert sha256(path) == entry['sha256']
                hashes[str(path)] = entry['sha256']
                with np.load(path, allow_pickle=False) as values:
                    raw = values['public_X'].copy()
                    control = values['control_code'][protocol['methods'].index('cnn_warm64')].copy()
                    trace = values['cnn_warm64__trace_control_code'].copy()
                    scores = values['cnn_warm64__trace_scores'].copy()
                    proposal = values['cnn_warm64__proposal_control_code'].copy()
                def execute():
                    callback = TraceReplay(trace[16:], scores[16:])
                    with patch('numpy.load', guard_load(np.load)):
                        result = model.decide(raw.copy(), rng_for(0, row['seed'], carrier, 630), callback)
                    torch.cuda.synchronize()
                    callback.finish()
                    assert result['feedback_calls'] == 64
                    for key, expected in [('control_code', control), ('trace_control_code', trace),
                                          ('trace_scores', scores), ('proposal_control_code', proposal)]:
                        np.testing.assert_array_equal(result[key], expected)
                if not warmed:
                    for _ in range(2):
                        execute()
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    write_json(output / 'warm_memory.json', memory())
                    warmed = True
                execute()
                records.append(dict(environment=row['index'], carrier=carrier, input_sha256=entry['sha256'],
                                    input_dtype=str(raw.dtype), input_bytes=raw.nbytes,
                                    control_and_trace_exact=True, memory=memory()))
        check_resources('end')
        base.verify()
        verify_sources(protocol['source_sha256'])
        for path, digest in hashes.items():
            assert sha256(Path(path)) == digest
        assert identity == {name: sha256(here / name) for name in identity}
        assert len(records) == len(rows) * len(carriers)
        write_json(output / 'inputs.json', hashes)
        write_json(output / 'records.json', records)
        write_json(output / 'summary.json', dict(cases=len(records), parameter_count=parameter_count,
                   parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes,
                   loaded_memory=loaded_memory,
                   final_memory=memory(),
                   max_sampled_rss_mib=max(r['memory']['VmRSS_mib'] for r in records),
                   max_sampled_driver_process_gpu_mib=max(r['memory']['driver_process_gpu_mib'] for r in records),
                   preflight=preflight, formal_memory=not preflight, scope=metadata['scope']))
        write_json(output / 'complete.json', dict(status='passed', at=now(), cases=len(records),
                   preflight=preflight, formal_memory=not preflight, source_sha256=identity,
                   file_sha256={name: sha256(output / name) for name in
                    ['summary.json', 'records.json', 'inputs.json', 'resources.json', 'protocol.json', 'warm_memory.json']}))
    except BaseException:
        write_json(output / 'failure.json', dict(at=now(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    run(args.project.resolve(), args.output.resolve(), args.preflight)
