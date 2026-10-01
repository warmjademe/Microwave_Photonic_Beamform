"""仅发布完整最终测试；旧网页资产移至站点外备份，逐路由核查，失败则恢复。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime,timezone

EXPECTED={'server.py','baselines.html','results-20261001.json','results-20261001.csv',
          'groups-20261001.csv','comparisons-20261001.csv'}
RETIRED=['/legacy.html','/native-calibration.html','/baselines-20260924.html','/baselines-data.json',
         '/baselines-signals.json','/baselines-results.csv','/review-code.zip','/deep/',
         '/deep/summary.json','/deep/figures/constellation.png','/results-20260926.json',
         '/results-20260926.csv','/groups-20260926.csv','/comparisons-20260926.csv']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_timing(data):
    """计时只覆盖固定102输入；避免与864环境的接收质量范围混淆。"""
    timing=data.get('timing')
    if timing is None:return
    expected=dict(status='passed',evaluation_split='test',environments=6,inputs=102,
                  carriers=17,repetitions=3,warmup=2,cpu_threads=1,
                  cpu_window_seconds=5,cpu_busy_limit=.6,model_loading_excluded=True,
                  measurement_time='nominal estimate, not hardware measurement')
    if any(timing.get(k)!=v for k,v in expected.items()):
        raise ValueError('计时范围或核查条件不符')
    names={r['method'] for r in data['methods'] if r['role']!='reference'}
    rows=timing.get('rows',[])
    if len(names)!=27 or len(rows)!=27 or {r['method'] for r in rows}!=names:
        raise ValueError('计时方法清单不完整')
    for key in ['source_complete_sha256','source_summary_sha256','records_sha256']:
        if not re.fullmatch('[0-9a-f]{64}',timing.get(key,'')):
            raise ValueError('计时来源SHA缺失')
    quality={r['method']:r for r in data['methods']}
    for r in rows:
        if r.get('cases')!=102 or r.get('time_samples')!=306 or r.get('measurement_budget') not in [16,64]:
            raise ValueError('逐方法计时样本数或预算不符')
        keys=['mean_software_ms','median_software_ms','p95_software_ms',
              'estimated_mean_total_ms','nominal_measurement_switch_ms']
        if any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) or r[k]<0 for k in keys):
            raise ValueError('计时包含无效数字')
        if not math.isclose(r['estimated_mean_total_ms'],r['mean_software_ms']+r['nominal_measurement_switch_ms'],rel_tol=1e-12):
            raise ValueError('名义总成本计算不符')
        q=quality[r['method']]
        if q.get('timing_inputs')!=102 or any(q.get('timing_'+k)!=r[k] for k in keys if k!='nominal_measurement_switch_ms'):
            raise ValueError('下载汇总与计时表不一致')


def validate(stage):
    m=json.loads((stage/'release.json').read_text())
    if (m.get('schema')!='mwp-final864-uniform64-release-v2' or m.get('final_confirmation') is not True
            or m.get('evaluation_split')!='test' or m.get('evaluation_environments')!=864
            or m.get('methods')!=29 or m.get('carriers')!=17 or m.get('method_cases')!=425952
            or m.get('historical_public_data') is not False or set(m.get('files',{}))!=EXPECTED):
        raise ValueError('拒绝发布未完成、旧划分或混入历史资产的版本')
    if len(m['source_audits'])!=2 or any(a['status']!='passed' or a['environments']!=864 for a in m['source_audits']):
        raise ValueError('最终测试审计未通过')
    for name,digest in m['files'].items():
        if sha(stage/name)!=digest:raise ValueError('发布文件SHA不符：'+name)
        if re.search(r'(?:sk-[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|olp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})',(stage/name).read_text()):
            raise ValueError('文件存在凭据特征')
    data=json.loads((stage/'results-20261001.json').read_text())
    if data['evaluation_split']!='test' or data['evaluation_environments']!=864 or len(data['methods'])!=29 or not data['final_confirmation']:
        raise ValueError('页面数据不是完整最终测试')
    validate_timing(data)
    py_compile.compile(str(stage/'server.py'),doraise=True)
    return m


def check_http(base,stage):
    checks={}
    for route in ['/','/index.html','/baselines','/baselines.html','/results-20261001.json',
                  '/results-20261001.csv','/groups-20261001.csv','/comparisons-20261001.csv']:
        with urllib.request.urlopen(base+route,timeout=10) as response:
            body=response.read()
            if response.status!=200:raise ValueError('新页面不可访问')
        name='baselines.html' if route in ['/','/index.html','/baselines'] else route[1:]
        if hashlib.sha256(body).hexdigest()!=sha(stage/name):raise ValueError('HTTP内容不符')
        checks[route]='200; byte-for-byte identical'
    for route in RETIRED+['/release.json','/server.py','/backups/','/staging/','/../review-code.zip']:
        try:urllib.request.urlopen(base+route,timeout=10)
        except urllib.error.HTTPError as error:
            if error.code not in [404,410]:raise
            checks[route]=str(error.code)+'; unavailable'
        else:raise ValueError('历史或内部地址仍公开：'+route)
    with urllib.request.urlopen(base+'/healthz',timeout=10) as response:health=json.load(response)
    if (health.get('historical_public_data') is not False or health.get('test_environments')!=864
            or health.get('methods')!=29 or health.get('main_comparison_methods')!=14):
        raise ValueError('服务器仍运行旧版')
    return dict(checks=checks,health=health)


def sidecar(stage):
    """先在临时loopback端口验证真实待发布文件，不中断当前网站。"""
    child=subprocess.Popen(['python3','-u',str(stage/'server.py'),'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        # 新服务只预压缩最终结果；启动快，外层发布进程有超时保护。
        line=child.stdout.readline()
        if not line:raise RuntimeError('待发布服务启动失败：'+child.stderr.read())
        info=json.loads(line);return check_http('http://127.0.0.1:'+str(info['port']),stage)
    finally:
        child.terminate();child.wait(timeout=10)


def run(stage,target):
    if target!=Path('/home/qyb/services/mwp-device-audit'):raise ValueError('仅允许既有微波光子网站目录')
    manifest=validate(stage);preflight=sidecar(stage)
    names=set(manifest['files'])|{'release.json'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=target.parent/('mwp-device-audit-archive-'+stamp);backup.mkdir()
    overwritten=backup/'overwritten';retired=backup/'retired';overwritten.mkdir();retired.mkdir()
    prior=[];moved=[];installed=[]
    # 精确清单先落盘；归档目录位于公开站点外，所有旧资料仍可追溯。
    old_entries=sorted(p.name for p in target.iterdir())
    (backup/'plan.json').write_text(json.dumps(dict(original_entries=old_entries,new_files=sorted(names)),indent=2))
    try:
        for name in sorted(names):
            if (target/name).exists():
                if not (target/name).is_file():raise ValueError('新文件位置被目录占用')
                shutil.copy2(target/name,overwritten/name);prior.append(name)
            temporary=target/(name+'.next');shutil.copyfile(stage/name,temporary);os.replace(temporary,target/name);installed.append(name)
        for name in old_entries:
            if name not in names:
                os.replace(target/name,retired/name);moved.append(name)
        subprocess.run(['systemctl','--user','restart','mwp-device-audit.service'],check=True)
        for attempt in range(90):
            try:
                with urllib.request.urlopen('http://127.0.0.1:4093/healthz',timeout=3) as response:
                    ready=json.load(response)
                if ready.get('historical_public_data') is False:break
            except OSError:pass
            time.sleep(.5)
        else:raise RuntimeError('新服务未恢复健康')
        checked=check_http('http://127.0.0.1:4093',stage)
        remaining={p.name for p in target.iterdir()}-names-{'__pycache__'}
        if remaining:raise ValueError('公开目录残留未声明资产')
        evidence=dict(status='published_final864_only_and_history_removed',at=stamp,
            archive=str(backup),retired_entries=moved,**checked,sidecar_preflight=preflight,
            public_browser_verification='pending',final_test_complete=True,overall_research_complete=False)
        (stage/'deployment.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
        print(json.dumps(evidence,ensure_ascii=False,indent=2))
    except Exception:
        # 只撤销本轮安装的文件，恢复归档条目和原服务；不动实验数据目录。
        for name in installed:
            if name in prior:shutil.copy2(overwritten/name,target/name)
            elif (target/name).is_file():(target/name).unlink()
        for name in moved:os.replace(retired/name,target/name)
        subprocess.run(['systemctl','--user','restart','mwp-device-audit.service'],check=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    a=p.parse_args();run(a.stage.resolve(),a.target.resolve())
