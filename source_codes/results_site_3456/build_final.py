"""仅从完整864测试的审计结果生成网站，不读取或混入旧验证成绩。"""
import argparse
import csv
import io
import json
from pathlib import Path
import socket
from datetime import datetime,timezone
from final_evidence import load_final, signal_examples, LABELS, sha, read


def build(project,output,timing=None):
    if 'huashuo' not in socket.gethostname().lower():raise RuntimeError('在华硕构建')
    if output.exists():raise FileExistsError('保留已有发布版本')
    rows,groups,comparisons,audits,protocol,extension=load_final(project)
    examples=signal_examples(project,protocol,extension)
    timing_data=None
    if timing is not None:
        from timing_evidence import load_timing
        timing_data=load_timing(project,timing,[r['method'] for r in rows if r['role']!='reference'])
    counts={role:sum(r['role']==role for r in rows) for role in ['baseline','primary','ablation','reference']}
    if counts!={'baseline':21,'primary':2,'ablation':4,'reference':2}:raise ValueError('最终方法清单不同')
    for row in rows:
        tones=[g for g in groups if g['method']==row['method'] and g['group']=='carrier_ghz']
        if len(tones)!=17 or {g['value'] for g in tones}!=set(range(4,21)) or any(g['environments']!=864 for g in tones):
            raise ValueError('频率分组覆盖不完整')
    data=dict(schema='mwp-final864-uniform64-v2',generated_at=datetime.now(timezone.utc).isoformat(),
        final_confirmation=True,evaluation_split='test',evaluation_environments=864,carriers=list(range(4,21)),
        test_records=14688,method_cases=425952,training_environments=3456,validation_environments=216,
        baseline_strategies=13,method_counts=counts,methods=[dict(r,label=LABELS[r['method']]) for r in rows],groups=groups,
        comparisons=[{k:v for k,v in r.items() if k!='coefficients'} for r in comparisons],
        audits=audits,examples=examples,timing=timing_data)
    if timing_data:
        by_name={r['method']:r for r in timing_data['rows']}
        for row in data['methods']:
            if row['method'] in by_name:
                t=by_name[row['method']]
                for key in ['mean_software_ms','median_software_ms','p95_software_ms','estimated_mean_total_ms']:
                    row['timing_'+key]=t[key]
                row['timing_inputs']=102
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    if '<script' in encoded.lower():raise ValueError('数据含脚本片段')
    view={k:v for k,v in data.items() if k not in ['groups','comparisons']}
    keys=['method','group','value','environments','ber','rms_evm_percent','ser','block_error_rate','paired_output_snr_db','effective_snr_db']
    view.update(group_keys=keys,group_rows=[[g[k] for k in keys] for g in groups],
        comparisons=[r for r in data['comparisons'] if r['metric']=='ber'])
    embedded=json.dumps(view,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
    template=(Path(__file__).parent/'final_template.html').read_text()
    if template.count('__SIGNALS_HTML__')!=1:raise ValueError('信号图插槽不符')
    template=template.replace('__SIGNALS_HTML__',(Path(__file__).parent/'final_signals.html').read_text())
    if template.count('__RESULTS_JSON__')!=1:raise ValueError('模板插槽不符')
    output.mkdir(parents=True)
    (output/'baselines.html').write_text(template.replace('__RESULTS_JSON__',embedded))
    (output/'results-20261001.json').write_text(encoded)
    for name,content in [('results-20261001.csv',data['methods']),('groups-20261001.csv',groups),('comparisons-20261001.csv',data['comparisons'])]:
        stream=io.StringIO(); fields=sorted(set().union(*(r.keys() for r in content)))
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for row in content:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        (output/name).write_text('\ufeff'+stream.getvalue())
    (output/'server.py').write_bytes((Path(__file__).parent/'final_server.py').read_bytes())
    manifest=dict(schema='mwp-final864-uniform64-release-v2',at=data['generated_at'],final_confirmation=True,
        evaluation_split='test',evaluation_environments=864,carriers=17,methods=29,method_cases=425952,
        baseline_strategies=13,main_comparison_methods=14,
        historical_public_data=False,source_audits=audits,files={p.name:sha(p) for p in output.iterdir()})
    (output/'release.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='complete_final_only_release',output=str(output),methods=29,environments=864)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timing',type=Path)
    a=p.parse_args();build(a.project.resolve(),a.output.resolve(),a.timing.resolve() if a.timing else None)
