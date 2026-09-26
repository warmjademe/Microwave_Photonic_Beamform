"""在华硕临时目录验证发布门槛和HTTP隔离，不发布样例或改变生产站点。"""
import argparse
import json
from pathlib import Path
import socket
import tempfile
from datetime import datetime,timezone
from build_final import build
from publish_final import validate,sidecar,sha,EXPECTED


def run(project,report):
    if 'huashuo' not in socket.gethostname().lower():raise RuntimeError('检查仅在华硕运行')
    evidence={}
    with tempfile.TemporaryDirectory(prefix='mwp-publish-check-') as folder:
        stage=Path(folder)
        # 纯路由样例，不含实验成绩；正式构建仅接受真正完成的分析。
        (stage/'server.py').write_bytes((Path(__file__).parent/'final_server.py').read_bytes())
        (stage/'baselines.html').write_text('<!doctype html><h1>HTTP ROUTE TEST FIXTURE, NOT RESULTS</h1>')
        data=dict(evaluation_split='test',evaluation_environments=864,final_confirmation=True,
                  methods=[{'fixture_only':True} for _ in range(21)])
        (stage/'results-20260926.json').write_text(json.dumps(data))
        for name in ['results-20260926.csv','groups-20260926.csv','comparisons-20260926.csv']:
            (stage/name).write_text('fixture_only\ntrue\n')
        # 即使旧资产存在磁盘上，直接旧链接也必须拒绝访问。
        (stage/'deep').mkdir();(stage/'deep/index.html').write_text('OLD_DATA_MUST_NOT_BE_PUBLIC')
        (stage/'review-code.zip').write_bytes(b'OLD_DATA_MUST_NOT_BE_PUBLIC')
        m=dict(schema='mwp-final864-only-release-v1',evaluation_split='test',evaluation_environments=864,
            methods=21,carriers=17,method_cases=308448,final_confirmation=True,historical_public_data=False,
            source_audits=[dict(status='passed',environments=864)],files={n:sha(stage/n) for n in EXPECTED})
        manifest=stage/'release.json';manifest.write_text(json.dumps(m))
        validate(stage);evidence['real_http_routes']=sidecar(stage)
        for change in [dict(evaluation_split='validation'),dict(evaluation_environments=863),
                       dict(final_confirmation=False),dict(historical_public_data=True),dict(methods=20)]:
            manifest.write_text(json.dumps({**m,**change}))
            try:validate(stage)
            except ValueError:pass
            else:raise AssertionError('错误发布条件未拒绝')
        manifest.write_text(json.dumps(m));(stage/'results-20260926.csv').write_text('changed\n')
        try:validate(stage)
        except ValueError:pass
        else:raise AssertionError('篡改文件未拒绝')
        evidence['negative_gate_checks']=6
        if not (project/'dataset_simulation/baseline_results/20260926_final864_selected/analysis/complete.json').exists():
            out=stage/'must_not_publish'
            try:build(project,out)
            except RuntimeError:pass
            else:raise AssertionError('未完成的真实测试被允许构建')
            if out.exists():raise AssertionError('拒绝后仍生成了发布目录')
            evidence['live_incomplete_test_rejected']=True
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(dict(status='passed',at=datetime.now(timezone.utc).isoformat(),
        production_site_changed=False,fixture_contains_experimental_results=False,**evidence),ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='passed',gate_checks=evidence['negative_gate_checks'],
                         routes=len(evidence['real_http_routes']['checks']),production_site_changed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.report.resolve())
