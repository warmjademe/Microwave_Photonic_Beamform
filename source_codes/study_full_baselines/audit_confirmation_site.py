"""逐项核对网站JSON/CSV与接收分析、固定信号及计时来源。"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, write_json, now
from study_full_baselines.build_confirmation_site import read, check, check_files, check_sources


def compare_payload(page, analysis, timing, formal):
    if page['confirmation_complete'] != formal or page['preview'] == formal:
        raise ValueError('页面把旧演练和新确认混淆。')
    overall = read(analysis/'overall.json')['records']
    grouped = read(analysis/'groups.json')['records']
    count = 0
    if len(page['methods']) != len(overall):
        raise ValueError('网站方法数量不符。')
    for method, record in zip(page['methods'], overall):
        if method['id'] != record['method'] or method['overall'] != record:
            raise ValueError('网站质量数字与原始分析不同。')
        for key in ['training_environments', 'probes', 'privileged', 'photonic_hardware']:
            if method[key] != record[key]:
                raise ValueError('网站比较条件不同。')
        if method['full_online_timing_available'] != (formal and not record['privileged']):
            raise ValueError('正式时间范围不符。')
        expected = [r for r in grouped if r['method'] == record['method']]
        if page['groups'][record['method']] != expected:
            raise ValueError('网站分组结果不同。')
        count += len(record)+sum(len(r) for r in expected)
    if page['test_environments'] != overall[0]['independent_environments'] or page['carriers_per_environment'] != overall[0]['carriers_per_environment']:
        raise ValueError('页面环境/载频数量不同。')
    if page['comparisons'] != read(analysis/'comparisons.json')['records']:
        raise ValueError('网站配对统计不同。')
    original = {r['method']:r for r in read(timing/'timing/summary.json')['methods']}
    if len(page['timing_records']) != len(original):
        raise ValueError('网站实际计时方法缺失。')
    for row in page['timing_records']:
        if any(v != original[row['method']][k] for k,v in row.items()):
            raise ValueError('网站时间数字改变。')
        count += len(row)
    return count


def check_csv(path, records):
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    if len(rows) != len(records):
        raise ValueError('CSV行数不同。')
    count=0
    for row, expected in zip(rows,records):
        for key,value in expected.items():
            text=row[key]
            if value is None:
                valid=text == ''
            elif isinstance(value,(dict,list)):
                valid=json.loads(text) == value
            elif isinstance(value,bool):
                valid=text == str(value)
            elif isinstance(value,(int,float)):
                valid=float(text) == value
            else:
                valid=text == str(value)
            if not valid:
                raise ValueError('CSV和JSON字段不同：'+key)
            count+=1
    return count


def audit(site):
    require_host()
    if (site/'integrity_verification.json').exists():
        raise FileExistsError('网站核查不覆盖既有证据。')
    build=read(site/'build.json'); metadata=read(site/'audit.json'); page=read(site/'results.json')
    check_files(site,build['public_files']); check_sources(site,build['source_sha256'])
    paths={k:Path(v) for k,v in metadata['paths'].items()}
    analysis,timing,signals=paths['analysis'],paths['timing'],paths['signals']
    checked=compare_payload(page,analysis,timing,build['final_confirmation'])
    csv_count=check_csv(site/'results.csv',[dict(m['overall'],label=m['label']) for m in page['methods']])
    csv_count+=check_csv(site/'timing.csv',page['timing_records'])
    check(site/'comparisons.csv',sha256(analysis/'comparisons.csv'))
    if page['provenance'] != metadata['provenance']:
        raise ValueError('网页来源摘要不同。')
    case_methods=0
    for entry in page['examples']:
        name='carrier_%02d.json'%entry['carrier']; control='carrier_%02d_controls.csv'%entry['carrier']
        check(site/entry['url'],sha256(signals/name));check(site/entry['controls_csv'],sha256(signals/control))
        case=read(site/entry['url'])
        if case['final_confirmation'] != build['final_confirmation']:
            raise ValueError('案例正式范围不同。')
        case_methods+=len(case['methods'])
    for name,digest in read(paths['exploration']/'build.json')['public_files'].items():
        check(site/'exploration'/name,digest)
    # 证明核查确实检查数字与范围，而不是只读取“完成”标记。
    faults=[]
    for name,mutate in [('changed_quality',lambda x:x['methods'][0]['overall'].__setitem__('ber',.999)),
                        ('changed_training_count',lambda x:x['methods'][6].__setitem__('training_environments',999)),
                        ('rehearsal_as_confirmation',lambda x:x.__setitem__('confirmation_complete',not build['final_confirmation']))]:
        invalid=copy.deepcopy(page);mutate(invalid)
        try:
            compare_payload(invalid,analysis,timing,build['final_confirmation'])
        except ValueError:
            faults.append(name)
        else:
            raise AssertionError('错误未被拒绝：'+name)
    for name in build['public_files']:
        if Path(name).suffix not in ['.html','.css','.js','.json','.csv','.png','.pdf','.svg'] or name.startswith(('source_snapshot/','records/')):
            raise ValueError('公开白名单包含非展示文件。')
    report=dict(status='passed_website_data_audit',at=now(),build_sha256=sha256(site/'build.json'),
        result_sha256=sha256(site/'results.json'),public_files_checked=len(build['public_files']),
        structured_fields_checked=checked,csv_fields_checked=csv_count,fixed_signal_method_cases=case_methods,
        fault_cases_rejected=faults,final_confirmation=build['final_confirmation'],
        auditor_sha256=sha256(Path(__file__)),browser_review='separate_required')
    write_json(site/'integrity_verification.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--site',type=Path,required=True)
    args=parser.parse_args();audit(args.site.resolve())
