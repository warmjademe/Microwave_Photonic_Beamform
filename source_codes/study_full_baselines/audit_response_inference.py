"""逐条输入响应网络后重算合法控制，检查批量浮点差异是否改变器件档位。

仅检查指定已完成模型的两轮控制器。若最后采用多起点/反馈，仍须核对完整部署链。
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from our_method_response_control.physics import ridge_estimate,decode
from our_method_response_control.model import Model as ComplexModel,conditions
from baseline_response_realcnn.model import Model as RealModel
from our_method_response_control.train import precision


def specification(project,name):
    base=project/'dataset_simulation';study=base/'baseline_results/20260925_full_baselines'
    if name=='complex_864':folder=base/'baseline_results/20260925_response_control'
    elif name=='real_864':folder=study/'real_response_cnn'
    else:
        count=int(name.split('_')[1]);schedule='_'.join(name.split('_')[2:])
        if count not in [216,432,1728,3456] or schedule not in ['fixed_epochs','equal_updates']:
            raise ValueError('未知规模模型。')
        parent=study/('scale_small' if count<864 else 'scale_%d'%count)
        folder=parent/('response_n%04d_%s'%(count,schedule))
    meta=json.loads((folder/'complete.json').read_text())
    if meta['status']!='complete':raise ValueError('模型尚未完成。')
    digest=(json.loads((folder/'inference.json').read_text())['predicted_response_sha256']
            if name=='complex_864' else meta['prediction_sha256'])
    for p,d in [(folder/'weights.pt',meta['weights_sha256']),(folder/'predicted_response.npy',digest)]:
        if sha256(p)!=d:raise ValueError('权重或响应预测改变。')
    return dict(name=name,kind='real' if name=='real_864' else 'complex',folder=str(folder),
        weights_sha256=meta['weights_sha256'],cached_response_sha256=digest)


def run(project,output,names):
    require_host();precision()
    if not torch.cuda.is_available():raise RuntimeError('响应网络推理要求华硕GPU。')
    if output.exists():raise FileExistsError('在线响应核对不覆盖已有结果。')
    data=project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    check_data(data);public=public_data(data);x,_,_,rows=load_split(data,'test',labels=False)
    specs=[specification(project,n) for n in names]
    if len(names)!=len(set(names)):raise ValueError('模型列表重复。')
    sources=source_record(['study_full_baselines/audit_response_inference.py','study_full_baselines/common.py',
        'our_method_response_control/model.py','our_method_response_control/physics.py',
        'our_method_response_control/train.py','baseline_response_realcnn/model.py'])
    protocol=dict(models=specs,data_manifest_sha256=sha256(data/'manifest.json'),source_sha256=sources,
        test_ids=[r['environment_id'] for r in rows],batch_size=1,controller='base_2sweeps',
        final_fair_timing=False,load_average_start=list(os.getloadavg()),gpu=torch.cuda.get_device_name(),
        scope='All old216 public inputs; no propagation truth or response labels used',
        timing='Single preprocessing, transfer, neural forward, CPU return and2sweep control; cached-response decode outside timer',at=now())
    output.mkdir(parents=True);write_json(output/'protocol.json',protocol);reports=[]
    for spec in specs:
        name=spec['name'];folder=Path(spec['folder']);cached=np.load(folder/'predicted_response.npy',mmap_mode='r')
        if cached.shape!=(len(x),64,31):raise ValueError('响应预测数量不符。')
        model=(RealModel() if spec['kind']=='real' else ComplexModel()).cuda()
        model.load_state_dict(torch.load(folder/'weights.pt',map_location='cuda',weights_only=True));model.eval()
        def predict(raw):
            initial=ridge_estimate(raw[None],public['pilot_qpsk']).astype(np.complex64)
            condition=conditions(raw[None],initial,public['pilot_qpsk'])
            return model(torch.from_numpy(initial).cuda(),torch.from_numpy(condition).cuda()).cpu().numpy()[0]
        codes=[];references=[];times=[];inference_times=[];relative=[];max_difference=0.
        with torch.inference_mode():
            for _ in range(5):predict(x[0])
            for i,raw in enumerate(x):
                tick=time.perf_counter()
                initial=public['probe_controls'][int(raw[1985:2001].argmax())];fc=int(raw[1984])
                h=predict(raw);forward_end=time.perf_counter()
                u,_=decode(h,fc,initial,sweeps=2)
                code=np.floor(np.clip(u,0,1)*LEVELS+.5).astype(np.int16)
                times.append(time.perf_counter()-tick);inference_times.append(forward_end-tick)
                old,_=decode(cached[i],fc,initial,sweeps=2)
                reference=np.floor(np.clip(old,0,1)*LEVELS+.5).astype(np.int16)
                codes.append(code);references.append(reference)
                delta=np.asarray(h,np.complex128)-cached[i]
                relative.append(float(np.sum(abs(delta)**2)/max(np.sum(abs(cached[i])**2),1e-30)))
                max_difference=max(max_difference,float(np.max(abs(delta))))
                if (i+1)%17==0:
                    write_json(output/'progress.json',dict(status='running',model=name,
                        completed_environments=(i+1)//17,total_environments=len(rows),pid=os.getpid(),at=now()))
        codes=np.asarray(codes);references=np.asarray(references);changed=np.any(codes!=references,axis=1)
        path=output/(name+'.npz')
        atomic_npz(path,control_code_single=codes,control_code_cached=references,
            response_relative_squared_difference=np.asarray(relative),
            online_seconds=np.asarray(times),inference_seconds=np.asarray(inference_times),
            changed_positions=np.argwhere(codes!=references))
        report=dict(model=name,samples=len(x),changed_samples=int(changed.sum()),
            changed_control_codes=int(np.sum(codes!=references)),all_controls_identical=not changed.any(),
            maximum_code_difference=int(np.max(abs(codes-references))),maximum_response_absolute_difference=max_difference,
            maximum_response_relative_squared_difference=float(np.max(relative)),
            online_mean_seconds=float(np.mean(times)),online_p95_seconds=float(np.quantile(times,.95)),
            timing_diagnostic_under_load=True,result_file=path.name,result_sha256=sha256(path),**spec)
        write_json(output/(name+'.json'),report);reports.append(report)
        print(json.dumps(report),flush=True);del model;torch.cuda.empty_cache()
    verify_sources(sources)
    for spec in specs:
        if specification(project,spec['name'])!=spec:raise ValueError('固定模型来源在核对期间改变。')
    write_json(output/'summary.json',dict(status='complete',records=reports,at=now(),
        all_controls_identical=all(r['all_controls_identical'] for r in reports),
        controller='base_2sweeps',final_fair_timing=False,
        load_average_end=list(os.getloadavg()),scope=protocol['scope']))
    write_json(output/'progress.json',dict(status='complete',models=names,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['complex_864','real_864']);a=p.parse_args()
    try:run(a.project,a.output,a.models)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
