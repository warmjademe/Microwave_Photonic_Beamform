"""在已有组件评测释放CPU后，按完全相同接收器评价实数CNN。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.evaluate_response_variants import init,one,METRICS
from study_full_baselines.evaluate_learned import summarize
from study_full_baselines.train_when_ready import alive


def run(project,output,wait_pid):
    require_host();base=project/'dataset_simulation';study=base/'baseline_results/20260925_full_baselines'
    data=base/'outputs/quality_rank_hybrid_20260925';model=study/'real_response_cnn'
    output.mkdir(parents=True,exist_ok=True)
    sources=source_record(['baseline_response_realcnn/evaluate.py',
        'study_full_baselines/evaluate_response_variants.py','study_full_baselines/evaluate_learned.py',
        'study_full_baselines/common.py','our_method_response_control/physics.py',
        'our_method_two_stage/decode_multistart.py','our_method_two_stage/control.py'])
    wait_record=dict(source_sha256=sources,wait_pid=wait_pid,scope='same old216 test and physical receiver')
    if (output/'wait_protocol.json').exists():
        if json.loads((output/'wait_protocol.json').read_text())!=wait_record:
            raise ValueError('排队来源不同。')
    else:write_json(output/'wait_protocol.json',wait_record)
    while True:
        ready=study/'evaluation_components/progress.json'
        state=json.loads(ready.read_text()) if ready.exists() else {}
        if state.get('status')=='complete':break
        if not alive(wait_pid) or list((study/'evaluation_components').glob('failure*.json')):
            raise RuntimeError('组件评测未完成但前置队列异常。')
        write_json(output/'progress.json',dict(status='waiting_for_component_evaluation',pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(sources)
    meta=json.loads((model/'complete.json').read_text());prediction=model/'predicted_response.npy'
    if meta['status']!='complete' or sha256(prediction)!=meta['prediction_sha256']:
        raise ValueError('实数响应CNN预测未完成或改变。')
    manifest=check_data(data);rows=[r for r in manifest['environments'] if r['split']=='test']
    if len(rows)!=216:raise ValueError('本轮只允许旧216测试环境。')
    methods=[dict(name='real_response_cnn__'+mode,kind='response',path=str(prediction),
        controller=mode,probes=16) for mode in ['base_2sweeps','multi_mmse']]
    bundle=dict(methods=methods,artifact_sha256={str(prediction):meta['prediction_sha256']})
    protocol=dict(bundle=bundle,data_manifest_sha256=sha256(data/'manifest.json'),
        metric_order=METRICS,frame=5,apd_draws=8,source_sha256=sources)
    fp=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('接收评分协议改变。')
    else:
        write_json(output/'protocol.json',protocol)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,dst)
    (output/'records').mkdir(exist_ok=True);records=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2,initializer=init,initargs=(str(data),bundle)) as pool:
        jobs=[pool.submit(one,row,str(output),fp) for row in rows]
        for job in as_completed(jobs):
            records.append(job.result());state=dict(status='running',completed=len(records),total=216,
                pid=os.getpid(),seconds=time.perf_counter()-started,at=now())
            write_json(output/'progress.json',state)
            if len(records)%10==0:print(json.dumps(state),flush=True)
    write_json(output/'records.json',sorted(records,key=lambda r:r['index']))
    summarize(output,rows,records,methods);verify_sources(sources)
    if sha256(prediction)!=meta['prediction_sha256']:raise ValueError('预测在评分中改变。')
    write_json(output/'progress.json',dict(status='complete',completed=216,total=216,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
    try:run(a.project,a.output,a.wait_pid)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
