"""等待响应网络逐条推理核对，再重放全部变化控制；不改写原评分。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.train_when_ready import alive


def reference_location(study,name,index):
    if name=='complex_864':phase='learned_evaluation';method='complex_response_cnn'
    elif name=='real_864':phase='evaluation_real_response_cnn';method='real_response_cnn__base_2sweeps'
    else:
        count=int(name.split('_')[1]);schedule='_'.join(name.split('_')[2:])
        phase='learned_evaluation' if count<864 else 'evaluation_scaling_%d'%count
        method='response_n%04d_%s'%(count,schedule)+('' if count<864 else '__base_2sweeps')
    return study/phase/'records'/('environment_%05d.npz'%index),method


def run(project,inference,output,wait_pid):
    require_host();output.mkdir(parents=True,exist_ok=False)
    sources=source_record(['study_full_baselines/evaluate_response_inference_changes.py',
                          'study_full_baselines/common.py'])
    write_json(output/'wait_protocol.json',dict(source_sha256=sources,wait_pid=wait_pid,at=now()))
    while True:
        f=inference/'progress.json';s=json.loads(f.read_text()) if f.exists() else {}
        if s.get('status')=='complete':break
        if not alive(wait_pid) or list(inference.glob('failure*.json')):
            raise RuntimeError('响应推理核对未完成且退出或报错。')
        write_json(output/'progress.json',dict(status='waiting_for_response_inference',pid=os.getpid(),at=now()))
        time.sleep(20)
    verify_sources(sources)
    base=project/'dataset_simulation';data=base/'outputs/quality_rank_hybrid_20260925'
    study=base/'baseline_results/20260925_full_baselines'
    protocol=json.loads((inference/'protocol.json').read_text());summary=json.loads((inference/'summary.json').read_text())
    if protocol['data_manifest_sha256']!=sha256(data/'manifest.json') or protocol['controller']!='base_2sweeps':
        raise ValueError('响应推理的数据或控制器约定不同。')
    verify_sources(protocol['source_sha256'])
    rows=[r for r in check_data(data)['environments'] if r['split']=='test']
    if protocol['test_ids']!=[r['environment_id'] for r in rows]:raise ValueError('环境顺序不同。')
    public=public_data(data);groups={};artifact={};methods=[r['model'] for r in summary['records']]
    for meta in summary['records']:
        path=inference/meta['result_file'];name=meta['model']
        if sha256(path)!=meta['result_sha256']:raise ValueError('响应档位核对产物改变。')
        artifact[str(path)]=meta['result_sha256']
        with np.load(path) as f:a=f['control_code_cached'];b=f['control_code_single']
        if a.shape!=b.shape or a.shape!=(len(rows)*17,128):raise ValueError('控制码覆盖不完整。')
        if np.any(a<0) or np.any(a>LEVELS) or np.any(b<0) or np.any(b>LEVELS):raise ValueError('控制码越界。')
        changed=np.flatnonzero(np.any(a!=b,axis=1))
        if len(changed)!=meta['changed_samples']:raise ValueError('档位差异数量不符。')
        for index in changed:groups.setdefault((int(index)//17,int(index)%17),[]).append((name,a[index],b[index]))
    write_json(output/'protocol.json',dict(data_manifest_sha256=sha256(data/'manifest.json'),source_sha256=sources,
        inference_protocol_sha256=sha256(inference/'protocol.json'),inference_artifact_sha256=artifact,
        frame=5,draws=8,scope='All changed base-controller codes; original batch receiver scores preserved',
        selection='All quantized control disagreements, no quality selection',at=now()))
    (output/'records').mkdir();differences={m:np.zeros(10) for m in methods};counts={m:0 for m in methods};details=[]
    for gi,((index,ci),items) in enumerate(sorted(groups.items())):
        row=rows[index];fc=ci+4;env=json.loads((data/row['path']/'environment.json').read_text())
        controls=np.asarray([[a,b] for _,a,b in items]).reshape(-1,128)/LEVELS
        engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
        scores,codes=reception_metrics(engine,clean,controls,payload,row['seed'],fc)
        scores=scores.reshape(len(items),2,10);codes=codes.reshape(len(items),2,128)
        path=output/'records'/('environment_%05d_fc%02d.npz'%(index,fc))
        atomic_npz(path,metrics=scores,control_code=codes,methods=np.asarray([v[0] for v in items]))
        for mi,(name,old,new) in enumerate(items):
            reference,method=reference_location(study,name,index);verified=False
            if reference.with_suffix('.json').exists():
                marker=json.loads(reference.with_suffix('.json').read_text())
                if sha256(reference)!=marker['sha256']:raise ValueError('既有接收评分记录改变。')
                rp=json.loads((reference.parents[1]/'protocol.json').read_text())
                if rp['data_manifest_sha256']!=sha256(data/'manifest.json'):raise ValueError('既有评分来自不同数据。')
                j=[m['name'] for m in rp['bundle']['methods']].index(method)
                with np.load(reference) as f:
                    if (not np.array_equal(f['control_code'][ci,j],old) or
                            not np.allclose(f['metrics'][ci,j,:10],scores[mi,0],rtol=1e-12,atol=0)):
                        raise ValueError('批量响应的控制/接收重放与正式评分不同。')
                verified=True
            differences[name]+=scores[mi,1]-scores[mi,0];counts[name]+=1
            details.append(dict(model=name,environment_id=row['environment_id'],carrier_ghz=fc,
                changed_channels=np.flatnonzero(old!=new).tolist(),maximum_code_difference=int(np.max(abs(new-old))),
                delta_bit_errors=int(scores[mi,1,0]-scores[mi,0,0]),delta_nmse=float(scores[mi,1,2]-scores[mi,0,2]),
                reference_existing_record_matched=verified,path=path.name,sha256=sha256(path)))
        write_json(output/'progress.json',dict(status='running',completed=gi+1,total=len(groups),pid=os.getpid(),at=now()))
    records=[];samples=len(rows)*17
    for name in methods:
        d=differences[name]
        records.append(dict(model=name,changed_samples=counts[name],samples=samples,
            delta_total_bit_errors=int(d[0]),delta_full_test_ber=float(d[0]/(samples*496)),
            delta_full_test_ser=float(d[3]/(samples*248)),delta_full_test_block_error_rate=float(d[5]/(samples*8)),
            delta_full_test_mean_nmse=float(d[2]/samples),delta_summed_clean_power_a2=float(d[7]),
            delta_summed_noise_power_a2=float(d[8]),sign='single minus cached batch',
            scope='Base2sweep controller only; final chosen feedback/multistart chain needs its own deployment audit'))
    verify_sources(sources)
    for path,digest in artifact.items():
        if sha256(path)!=digest:raise ValueError('核对产物在接收重放期间改变。')
    write_json(output/'summary.json',dict(status='complete',records=records,case_details=details,at=now()))
    write_json(output/'progress.json',dict(status='complete',models=methods,case_groups=len(groups),at=now()))
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','inference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
    try:run(a.project,a.inference,a.output,a.wait_pid)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
