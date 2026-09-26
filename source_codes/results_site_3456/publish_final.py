"""仅发布完整最终测试；旧网页资产移至站点外备份，逐路由核查，失败则恢复。"""
import argparse
import hashlib
import json
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

EXPECTED={'server.py','baselines.html','results-20260926.json','results-20260926.csv',
          'groups-20260926.csv','comparisons-20260926.csv'}
RETIRED=['/legacy.html','/native-calibration.html','/baselines-20260924.html','/baselines-data.json',
         '/baselines-signals.json','/baselines-results.csv','/review-code.zip','/deep/',
         '/deep/summary.json','/deep/figures/constellation.png']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(stage):
    m=json.loads((stage/'release.json').read_text())
    if (m.get('schema')!='mwp-final864-only-release-v1' or m.get('final_confirmation') is not True
            or m.get('evaluation_split')!='test' or m.get('evaluation_environments')!=864
            or m.get('methods')!=21 or m.get('carriers')!=17 or m.get('method_cases')!=308448
            or m.get('historical_public_data') is not False or set(m.get('files',{}))!=EXPECTED):
        raise ValueError('拒绝发布未完成、旧划分或混入历史资产的版本')
    if len(m['source_audits'])!=1 or m['source_audits'][0]['status']!='passed' or m['source_audits'][0]['environments']!=864:
        raise ValueError('最终测试审计未通过')
    for name,digest in m['files'].items():
        if sha(stage/name)!=digest:raise ValueError('发布文件SHA不符：'+name)
        if re.search(r'(?:sk-[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{20,}|olp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})',(stage/name).read_text()):
            raise ValueError('文件存在凭据特征')
    data=json.loads((stage/'results-20260926.json').read_text())
    if data['evaluation_split']!='test' or data['evaluation_environments']!=864 or len(data['methods'])!=21 or not data['final_confirmation']:
        raise ValueError('页面数据不是完整最终测试')
    py_compile.compile(str(stage/'server.py'),doraise=True)
    return m


def check_http(base,stage):
    checks={}
    for route in ['/','/index.html','/baselines','/baselines.html','/results-20260926.json',
                  '/results-20260926.csv','/groups-20260926.csv','/comparisons-20260926.csv']:
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
    if health.get('historical_public_data') is not False or health.get('test_environments')!=864:
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
