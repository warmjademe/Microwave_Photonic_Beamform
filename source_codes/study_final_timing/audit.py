"""由逐次原始耗时独立汇总；检查覆盖、控制、预算和反馈，不重跑算法。"""
import json
from pathlib import Path
import numpy as np
from study_full_baselines.common import sha256,write_json

def read(p):return json.loads(Path(p).read_text())

def audit(folder):
    protocol=read(folder/'protocol.json');records=read(folder/'records.json');inputs=read(folder/'inputs.json')
    resources=read(folder/'resources.json')
    assert [r['phase'] for r in resources]==['start']+[n+':'+part for n in protocol['methods'] for part in ['before','after']]
    for r in resources:
        assert r['cpu_sample_seconds']>=protocol['cpu_sampling_seconds']
        fraction=1-r['cpu_idle_tick_delta']/r['cpu_tick_delta']
        assert abs(fraction-r['cpu_busy_fraction'])<1e-12
        if not protocol['preflight']:
            assert fraction<=.6 and not r['project_numeric_processes']
            assert len(r['gpu_processes'])<=1
    expected={(n,r['index'],fc) for n in protocol['methods'] for r in protocol['rows'] for fc in protocol['carriers']}
    assert {(r['method'],r['environment_index'],r['carrier_ghz']) for r in records}==expected
    assert len(records)==len(expected)
    for p,h in inputs.items():assert sha256(p)==h
    summaries=[];trace_rows=0;states=0
    for n in protocol['methods']:
        times=[];calls=[]
        for r in records:
            if r['method']!=n:continue
            path=folder/'records'/r['file'];assert sha256(path)==r['sha256']
            with np.load(path) as a:
                t=a['times'];code=a['control_code'];budget=int(a['feedback_calls'])
                assert t.shape==(3,3) and np.isfinite(t).all() and np.all(t>=0)
                assert np.all(t[:,0]+t[:,1]<=t[:,2]+1e-6)
                assert code.shape==(128,) and np.all(code>=0) and np.all(code[:64]<=76) and np.all(code[64:]<=24)
                assert np.array_equal(code,code.round())
                assert a['public_X'].shape==(2513,) and a['public_X'][1984]==r['carrier_ghz']
                assert budget in [16,64]
                if budget==64:
                    trace=a['trace_control_code'];scores=a['trace_scores']
                    assert trace.shape==(64,128) and scores.shape==(64,)
                    np.testing.assert_array_equal(code,trace[scores.argmax()])
                    trace_rows+=64
                times.extend(t[:,0].tolist());calls.append(budget);states+=128
        assert len(set(calls))==1
        nominal=calls[0]*protocol['measurement_seconds']+protocol['switch_seconds']
        summaries.append(dict(method=n,cases=len(calls),time_samples=len(times),measurement_budget=calls[0],
            mean_software_ms=float(np.mean(times)*1000),median_software_ms=float(np.median(times)*1000),
            p95_software_ms=float(np.quantile(times,.95)*1000),nominal_measurement_switch_ms=nominal*1000,
            estimated_mean_total_ms=(float(np.mean(times))+nominal)*1000))
    write_json(folder/'summary.json',dict(records=summaries,preflight=protocol['preflight'],
        final_fair_timing=protocol['final_fair_timing'],scope=f"{len(protocol['rows'])*len(protocol['carriers'])} fixed inputs; not all864 latency",
        hardware_measurement_time='nominal estimate, not measured'))
    result=dict(status='passed',cases=len(records),time_samples=len(records)*3,
        legal_control_codes_checked=states,trace_rows_checked=trace_rows,all_expected_cases=True,
        records_sha256=sha256(folder/'records.json'),inputs_sha256=sha256(folder/'inputs.json'),
        summary_sha256=sha256(folder/'summary.json'),resource_snapshots_checked=len(resources),
        resources_sha256=sha256(folder/'resources.json'))
    write_json(folder/'audit.json',result);return result
