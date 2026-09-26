"""只重放单条/批量推理档位不同的案例，量化其对完整测试分数的影响。"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *


def run(data,study,inference,output):
    require_host()
    if output.exists():raise FileExistsError('差异评价另存，不覆盖批量评分。')
    summary=json.loads((inference/'summary.json').read_text())
    protocol=json.loads((inference/'protocol.json').read_text())
    if summary['status']!='complete' or protocol['data_manifest_sha256']!=sha256(data/'manifest.json'):
        raise ValueError('单条推理核对未完成或数据身份不同。')
    verify_sources(protocol['source_sha256'])
    manifest=check_data(data);rows=[r for r in manifest['environments'] if r['split']=='test']
    public=public_data(data);groups={};artifact_hashes={}
    methods=[r['method'] for r in summary['records']]
    for record in summary['records']:
        name=record['method'];path=inference/record['result_file']
        if sha256(path)!=record['result_sha256']:raise ValueError('推理核对产物发生变化。')
        artifact_hashes[str(path)]=record['result_sha256']
        with np.load(path) as f:
            a=f['control_code_cached'];b=f['control_code_single'];single=f['predictions_single']
        if a.shape!=b.shape or len(a)!=len(rows)*17:raise ValueError('全测试推理记录数量不同。')
        original=study/'learned'/name/'predictions.npy'
        if sha256(original)!=record['prediction_reference_sha256']:raise ValueError('原批量预测改变。')
        expected=np.floor(np.clip(np.load(original),0,1)*LEVELS+.5).astype(np.int16)
        if not np.array_equal(expected,a):raise ValueError('旧控制码不对应原批量预测。')
        if not np.array_equal(np.floor(np.clip(single,0,1)*LEVELS+.5).astype(np.int16),b):
            raise ValueError('新控制码不对应实际单条预测。')
        changed=np.flatnonzero(np.any(a!=b,axis=1))
        if len(changed)!=record['changed_samples']:raise ValueError('差异案例计数不符。')
        for index in changed:
            groups.setdefault((int(index)//17,int(index)%17),[]).append((name,a[index],b[index]))
    output.mkdir(parents=True);(output/'records').mkdir()
    source=source_record(['study_full_baselines/evaluate_single_inference_changes.py',
                         'study_full_baselines/common.py'])
    write_json(output/'protocol.json',dict(data_manifest_sha256=sha256(data/'manifest.json'),
        source_sha256=source,inference_summary_sha256=sha256(inference/'summary.json'),
        inference_artifact_sha256=artifact_hashes,frame=5,draws=8,
        selection='All and only quantized-control disagreements; no quality-based selection',
        scope='Numerical deployment sensitivity; original exploratory batch results preserved',at=now()))
    differences={m:np.zeros(10) for m in methods};details=[];sample_counts={m:0 for m in methods}
    for (index,ci),items in sorted(groups.items()):
        row=rows[index];fc=ci+4
        env=json.loads((data/row['path']/'environment.json').read_text())
        controls=np.asarray([[a,b] for _,a,b in items]).reshape(-1,128)/LEVELS
        engine,payload=frame_engine(env,fc,public['pilot_qpsk'],5)
        clean=clean_frame_engine(env,fc,public['pilot_qpsk'],payload)
        scores,codes=reception_metrics(engine,clean,controls,payload,row['seed'],fc)
        scores=scores.reshape(len(items),2,10);codes=codes.reshape(len(items),2,128)
        # 上游可能仍在评分；已有原子提交的参考必须立即匹配，未完成的留待总审计。
        reference=study/'learned_evaluation/records'/('environment_%05d.npz'%index)
        reference_checked=False
        if reference.with_suffix('.json').exists():
            marker=json.loads(reference.with_suffix('.json').read_text())
            if sha256(reference)!=marker['sha256']:raise ValueError('完整评测记录哈希不符。')
            baseline_protocol=json.loads((study/'learned_evaluation/protocol.json').read_text())
            original_methods=[m['name'] for m in baseline_protocol['bundle']['methods']]
            with np.load(reference) as f:
                for k,(name,a,b) in enumerate(items):
                    mi=original_methods.index(name)
                    if (not np.array_equal(f['control_code'][ci,mi],a)
                            or not np.allclose(f['metrics'][ci,mi,:10],scores[k,0],rtol=1e-12,atol=0)):
                        raise ValueError('批量控制的接收重放没有复现正式评分。')
            reference_checked=True
        path=output/'records'/('environment_%05d_fc%02d.npz'%(index,fc))
        atomic_npz(path,metrics=scores,control_code=codes,methods=np.asarray([x[0] for x in items]))
        for k,(name,_,_) in enumerate(items):
            differences[name]+=scores[k,1]-scores[k,0];sample_counts[name]+=1
            details.append(dict(method=name,environment_id=row['environment_id'],carrier_ghz=fc,
                changed_channels=np.flatnonzero(codes[k,0]!=codes[k,1]).tolist(),
                delta_bit_errors=int(scores[k,1,0]-scores[k,0,0]),
                delta_nmse=float(scores[k,1,2]-scores[k,0,2]),
                reference_existing_record_matched=reference_checked,path=path.name,sha256=sha256(path)))
        write_json(output/'progress.json',dict(status='running',completed_case_groups=len({(r['environment_id'],r['carrier_ghz']) for r in details}),
            total_case_groups=len(groups),at=now()))
    records=[];samples=len(rows)*17
    for name in methods:
        d=differences[name]
        records.append(dict(method=name,changed_samples=sample_counts[name],total_samples=samples,
            delta_total_bit_errors=int(d[0]),delta_full_test_ber=float(d[0]/(samples*496)),
            delta_full_test_ser=float(d[3]/(samples*248)),delta_full_test_block_error_rate=float(d[5]/(samples*8)),
            delta_full_test_mean_nmse=float(d[2]/samples),
            delta_summed_clean_power_a2=float(d[7]),delta_summed_noise_power_a2=float(d[8]),
            sign='single sample minus original cached batch',
            note='RMS EVM and physical SNR need full baseline aggregates; differences are not sums of per-case EVM/dB'))
    verify_sources(source)
    for path,digest in artifact_hashes.items():
        if sha256(path)!=digest:raise ValueError('推理记录在差异评分中改变。')
    write_json(output/'summary.json',dict(status='complete',records=records,case_details=details,
        scope='Exact full-test metric deltas for seven baselines; only changed controls require replay',
        baseline_scores_preserved=True,inference_convention_for_final_confirmation_pending=True,at=now()))
    write_json(output/'progress.json',dict(status='complete',completed_case_groups=len(groups),at=now()))
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','study','inference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a.data,a.study,a.inference,a.output)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
