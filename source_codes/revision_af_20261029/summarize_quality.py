"""新增质量训练控制与原 MSE 控制、A/F 的全量配对比较。"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now,atomic_npz
from study_full_baselines.paired_statistics import sufficient,metrics,FIELDS,bootstrap,contrast,bh_adjust


def read(p):return json.loads(Path(p).read_text())


def run(project):
    require_host();root=project/'dataset_simulation/revision_af_20261029/C/test';out=root.parent/'analysis';out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=read(root/'complete.json');p=read(root/'protocol.json');assert done['status']=='complete' and done['environments']==864 and done['carriers']==17 and not done['preflight']
        for name,h in done['file_sha256'].items():assert sha256(root/name)==h
        base=project/'dataset_simulation/revision_rq2_af_20261029/full';bp=read(base/'protocol.json');bv=read(base/'verification.json')
        for n in ['protocol.json','environment_metrics.npz']:assert sha256(base/n)==bv['files_sha256'][n]
        with np.load(root/'environment_metrics.npz',allow_pickle=False) as f:
            new=f['values'].copy();newnames=f['methods'].tolist();ids=f['environment_ids'].tolist()
        with np.load(base/'environment_metrics.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(ids,f['environment_ids']);references=f['values'][:,:,[1,3,4]].copy()
        assert new.shape==(864,17,8,13)
        records=read(root/'records.json');assert [r['environment_id'] for r in records]==ids
        originals=np.empty((864,17,4,13));selected={n:Counter() for n in newnames if n.endswith('feedback64')};checked=0
        oldroot=Path(bp['source']);uniform=Path(bp['uniform_source'])
        for i,r in enumerate(records):
            assert [e['carrier_ghz'] for e in r['carriers']]==list(range(4,21))
            oldfolder=oldroot/'records'/f"environment_{r['index']:05d}";om=read(oldfolder/'complete.json')
            for j,e in enumerate(r['carriers']):
                path=root/e['path'];assert sha256(path)==e['sha256'] and e['protocol_sha256']==done['protocol_sha256']
                assert e['checks']['feedback_trace_audited'] and not e['checks']['joint_correction']
                with np.load(path,allow_pickle=False) as a:
                    np.testing.assert_array_equal(new[i,j],a['metrics']);public=a['public_X'].copy()
                    for n in selected:
                        ix=int(a[n+'__trace_scores'].argmax());code=a['control_code'][newnames.index(n)]
                        np.testing.assert_array_equal(code,a[n+'__trace_control_code'][ix])
                        origin='initial' if ix<16 else ('model' if np.array_equal(code,a[n+'__proposal_control_code']) else 'geometric')
                        selected[n][origin]+=1
                oe=next(v for v in om['carriers'] if v['carrier_ghz']==e['carrier_ghz']);op=oldroot/oe['path'];assert sha256(op)==oe['sha256']
                up=uniform/'records'/oldfolder.name/f"carrier_{e['carrier_ghz']:02d}.npz";um=read(up.with_suffix('.json'));assert sha256(up)==um['sha256']
                with np.load(op,allow_pickle=False) as a:
                    np.testing.assert_array_equal(public,a['public_X'])
                    for k,n in enumerate(['cnn','transformer']):originals[i,j,2*k]=a['metrics'][bp['prior']['methods'].index(n)]
                with np.load(up,allow_pickle=False) as a:
                    np.testing.assert_array_equal(public,a['public_X'])
                    for k,n in enumerate(['cnn_feedback64','transformer_feedback64']):originals[i,j,2*k+1]=a['metrics'][bp['uniform']['methods'].index(n)]
                checked+=1
        names=newnames+['original_cnn_direct16','original_cnn_feedback64','original_transformer_direct16','original_transformer_feedback64','A_direct16','A_feedback64','fixed_scan64']
        values=np.concatenate([new,originals,references],axis=2);assert values.shape==(864,17,15,13)
        s=sufficient(values);point=metrics(s.sum(0));draws=bootstrap(s,10000,20261030,strata=[r['joint_stratum_id'] for r in p['rows']])
        summary=[dict(method=n,**{f:float(point[i,j]) for j,f in enumerate(FIELDS)},ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])}) for i,n in enumerate(names)]
        specs=[]
        for model in ['cnn','transformer']:
            for init in ['scratch','mse_init']:
                for mode in ['direct16','feedback64']:
                    left=model+'_'+init+'_'+mode
                    specs += [('quality_vs_MSE',left,'original_'+model+'_'+mode),('quality_vs_A',left,'A_'+mode)]
                specs += [('quality_F_vs_scan',model+'_'+init+'_feedback64','fixed_scan64')]
        comparisons=[]
        for family,left,right in specs:
            c=np.zeros(len(names));c[names.index(left)]=1;c[names.index(right)]=-1
            for field in FIELDS[:5]:comparisons.append(dict(family=family,left=left,right=right,**contrast(s,draws,c,field)))
        for family in ['quality_vs_MSE','quality_vs_A','quality_F_vs_scan']:
            for field in FIELDS[:5]:
                group=[r for r in comparisons if r['family']==family and r['metric']==field]
                for item,q in zip(group,bh_adjust([r['p_value'] for r in group])):item['q_value']=float(q)
        training=root.parent/'training';training_meta={name:read(training/name/'complete.json') for name in ['cnn_scratch','cnn_mse_init','transformer_scratch','transformer_mse_init']}
        write_json(out/'summary.json',summary);write_json(out/'comparisons.json',comparisons)
        write_json(out/'candidate_sources.json',{n:dict(c) for n,c in selected.items()});write_json(out/'training_cost.json',training_meta)
        atomic_npz(out/'environment_metrics.npz',values=values,methods=np.asarray(names),environment_ids=np.asarray(ids))
        write_json(out/'protocol.json',dict(code_sha256=sha256(Path(__file__)),input_files_sha256=done['file_sha256'],original_protocol_sha256=sha256(base/'protocol.json'),
            comparisons='four fixed configurations; no best test model selected',repetitions=10000,bootstrap_seed=20261030,
            inference_scope='all methods receive same public input; responses or payload are not online inputs',
            training_scope='quality training uses the same internal response supervision as A; mse_init has 80 total epochs',
            multiplicity='BH separately by comparison family and metric',statistical_unit='paired environment, 17 carriers kept together'))
        lines=['# 接收质量目标直接控制基线','',
            '四个预定配置全部报告；scratch 为40轮质量训练，mse_init 为原40轮MSE加40轮质量训练。',
            '均使用种子0，不能据此声称新增基线具有多种子稳定性。','',
            '| 配置 | BER (%) | RMS EVM (%) | 配对 SNR (dB) |','|---|---:|---:|---:|']
        for r in summary:lines.append(f"| {r['method']} | {100*r['ber']:.6f} | {r['rms_evm_percent']:.4f} | {r['paired_output_snr_db']:.4f} |")
        lines+=['','全部配对差值和区间见 comparisons.json。反馈收益包含从16次到64次测量的变化。',
            '质量代理采用整数前向与连续松弛梯度；损失下降与实际业务BER改善分别评价，不互相替代。','']
        (out/'REPORT_zh.md').write_text('\n'.join(lines))
        for n,h in done['file_sha256'].items():assert sha256(root/n)==h
        outputs=['summary.json','comparisons.json','candidate_sources.json','training_cost.json','environment_metrics.npz','protocol.json','REPORT_zh.md']
        write_json(out/'complete.json',dict(status='complete',new_records_verified=checked,original_control_records_verified=checked*2,
            at=now(),file_sha256={n:sha256(out/n) for n in outputs}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);run(p.parse_args().project.resolve())
