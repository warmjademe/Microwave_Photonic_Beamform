"""发布前从正式计时原始数组重算汇总；不接受预检或并行评测耗时。"""
import json
from pathlib import Path
import numpy as np
from final_evidence import read, sha


def load_timing(project, folder, methods):
    complete=read(folder/'complete.json');protocol=read(folder/'protocol.json')
    summary=read(folder/'summary.json');audit=read(folder/'audit.json')
    if (complete['status']!='passed' or complete['preflight'] or not complete['final_fair_timing']
            or protocol['preflight'] or not protocol['final_fair_timing']):
        raise ValueError('正式计时尚未通过')
    assert complete['audit']==audit and audit['status']=='passed'
    assert sha(folder/'summary.json')==complete['summary_sha256']==audit['summary_sha256']
    assert sha(folder/'records.json')==audit['records_sha256']
    assert sha(folder/'inputs.json')==audit['inputs_sha256']
    assert sha(folder/'resources.json')==audit['resources_sha256']
    assert set(protocol['methods'])==set(methods) and len(methods)==27
    assert protocol['carriers']==list(range(4,21)) and len(protocol['rows'])==6
    assert [r['factors']['power_bin'] for r in protocol['rows']]==list(range(6))
    assert protocol['repetitions']==3 and protocol['warmup']==2 and protocol['cpu_threads']==1
    assert protocol['cpu_sampling_seconds']==5 and protocol['cpu_busy_limit']==.6
    assert not protocol['new_training'] and not protocol['test_quality_or_control_changes']
    for name,h in complete['source_sha256'].items():assert sha(folder/name)==h
    for p,h in read(folder/'inputs.json').items():assert sha(Path(p))==h
    old=project/'dataset_simulation/baseline_results/20260926_final864_selected/protocol.json'
    new=project/'dataset_simulation/baseline_results/20260927_uniform64_all13/protocol.json'
    assert sha(old)==protocol['original_protocol_sha256'] and sha(new)==protocol['extension_protocol_sha256']
    original=read(old)
    assert protocol['rows']==[next(r for r in original['rows'] if r['factors']['power_bin']==b) for b in range(6)]
    resources=read(folder/'resources.json')
    assert len(resources)==55
    assert all(r['cpu_sample_seconds']>=5 and r['cpu_busy_fraction']<=.6 for r in resources)
    records=read(folder/'records.json')
    expected={(n,r['index'],f) for n in methods for r in protocol['rows'] for f in range(4,21)}
    assert len(records)==2754 and {(r['method'],r['environment_index'],r['carrier_ghz']) for r in records}==expected
    grouped={n:[] for n in methods}
    for r in records:
        p=folder/'records'/r['file'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False) as a:
            t=a['times'];assert t.shape==(3,3) and np.isfinite(t).all() and (t>=0).all()
            grouped[r['method']].extend(t[:,0].tolist())
    assert len(summary['records'])==27
    for row in summary['records']:
        values=np.asarray(grouped[row['method']])*1000
        assert row['cases']==102 and row['time_samples']==306
        np.testing.assert_allclose([row['mean_software_ms'],row['median_software_ms'],row['p95_software_ms']],
                                  [values.mean(),np.median(values),np.quantile(values,.95)],rtol=1e-12,atol=1e-10)
    return dict(status='passed',evaluation_split='test',environments=6,inputs=102,carriers=17,
        selection='six power groups; first environment in each; all 17 carriers',
        repetitions=3,warmup=2,cpu_threads=1,gpu=protocol['gpu'],
        cpu_window_seconds=5,cpu_busy_limit=.6,model_loading_excluded=True,
        measurement_time='nominal estimate, not hardware measurement',
        source_complete_sha256=sha(folder/'complete.json'),source_summary_sha256=sha(folder/'summary.json'),
        records_sha256=audit['records_sha256'],rows=summary['records'])
