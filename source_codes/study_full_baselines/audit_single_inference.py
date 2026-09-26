"""逐样本实际推理与批量缓存的档位核对，不覆盖已有训练或评分产物。

当前只覆盖七个直接控制基线；响应模型另行检查。
负载下的在线时间仅作诊断，不当作最终公平延迟排名。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from deep_common.preprocessing import apply
from deep_common.layers import build
from baseline_common.data import features
from baseline_mlp.method import load as load_mlp
from our_method_response_control.train import precision


def quantize(u):
    return np.floor(np.clip(u,0,1)*LEVELS+.5).astype(np.int16)


def run(data,study,output):
    require_host();precision()
    if not torch.cuda.is_available():raise RuntimeError('六深度网络要求华硕GPU。')
    if output.exists():raise FileExistsError('逐样本推理诊断不覆盖旧产物。')
    check_data(data);public=public_data(data)
    x,_,_,rows=load_split(data,'test',labels=False);learned=study/'learned'
    train_protocol=json.loads((learned/'protocol.json').read_text())
    if train_protocol['test_ids']!=[r['environment_id'] for r in rows]:
        raise ValueError('共同测试成员顺序不同。')
    if json.loads((learned/'progress.json').read_text())['status']!='complete':
        raise ValueError('七模型尚未训练完成。')
    with np.load(learned/'normalization.npz') as f:stats={k:f[k] for k in f.files}
    sources=['study_full_baselines/audit_single_inference.py','study_full_baselines/common.py',
        'baseline_common/data.py','baseline_mlp/method.py','deep_common/preprocessing.py',
        'deep_common/layers.py','our_method_response_control/train.py']
    protocol=dict(scope='single-sample versus original cached-batch quantized controls; old216 only',
        source_sha256=source_record(sources),data_manifest_sha256=sha256(data/'manifest.json'),
        learned_protocol_sha256=sha256(learned/'protocol.json'),
        normalization_sha256=sha256(learned/'normalization.npz'),
        methods=['mlp']+DEEP_METHODS,batch_size=1,load_average_start=list(os.getloadavg()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        timing='Diagnostic under current host load: input preprocessing, transfer, forward, CPU return and legal quantization; excludes model loading and physical measurement',
        final_fair_timing=False,at=now())
    output.mkdir(parents=True);write_json(output/'protocol.json',protocol);records=[]
    for method in protocol['methods']:
        folder=learned/method;meta=json.loads((folder/'complete.json').read_text())
        weight=folder/('model/weights.npz' if method=='mlp' else 'weights.pt')
        if (sha256(weight)!=meta['weights_sha256']
                or sha256(folder/'predictions.npy')!=meta['predictions_sha256']):
            raise ValueError('固定权重或参考预测改变。')
        cached=np.load(folder/'predictions.npy')
        if method=='mlp':
            model=load_mlp(folder/'model')
            def predict(raw):return model.predict_features(features(observation(raw,public)))
        else:
            model=build(method).cuda()
            model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True));model.eval()
            def predict(raw):
                prepared=apply(raw[None],stats)
                tensor=torch.from_numpy(prepared).cuda()
                return model(tensor).cpu().numpy()[0]
        got=[];timings=[];codes=[]
        with torch.inference_mode():
            for _ in range(5):predict(x[0])
            for i,raw in enumerate(x):
                tick=time.perf_counter();u=predict(raw);code=quantize(u)
                timings.append(time.perf_counter()-tick);got.append(u);codes.append(code)
                if i%512==0:
                    write_json(output/'progress.json',dict(status='running',method=method,
                        sample=i+1,total=len(x),pid=os.getpid(),at=now()))
        got=np.asarray(got);codes=np.asarray(codes);reference=quantize(cached)
        if got.shape!=cached.shape or not np.isfinite(got).all():raise ValueError('在线预测缺失。')
        changed=np.any(codes!=reference,axis=1);positions=np.argwhere(codes!=reference)
        dest=output/(method+'.npz')
        atomic_npz(dest,predictions_single=got,control_code_single=codes,control_code_cached=reference,
            changed_positions=positions,timing_seconds=np.asarray(timings))
        record=dict(method=method,samples=len(x),changed_samples=int(changed.sum()),
            changed_control_codes=int(len(positions)),unchanged_controls=not changed.any(),
            maximum_raw_prediction_difference=float(np.max(abs(got-cached))),
            maximum_code_difference=int(np.max(abs(codes-reference))),
            single_mean_seconds=float(np.mean(timings)),single_p95_seconds=float(np.quantile(timings,.95)),
            prediction_reference_sha256=meta['predictions_sha256'],weights_sha256=meta['weights_sha256'],
            result_sha256=sha256(dest),result_file=dest.name)
        write_json(output/(method+'.json'),record);records.append(record)
        print(json.dumps(record),flush=True)
        del model;torch.cuda.empty_cache()
    verify_sources(protocol['source_sha256'])
    if sha256(learned/'normalization.npz')!=protocol['normalization_sha256']:
        raise ValueError('标准化参数改变。')
    exact=all(r['unchanged_controls'] for r in records)
    write_json(output/'summary.json',dict(status='complete',records=records,
        all_controls_identical=exact,scope=protocol['scope'],final_fair_timing=False,
        conclusion=('All single-sample quantized controls match cached batch predictions.' if exact else
                    'Quantized controls differ; reception effect and deployment batch convention must be checked before final online claims.'),
        load_average_end=list(os.getloadavg()),at=now()))
    write_json(output/'progress.json',dict(status='complete',all_controls_identical=exact,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','study','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a.data,a.study,a.output)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
