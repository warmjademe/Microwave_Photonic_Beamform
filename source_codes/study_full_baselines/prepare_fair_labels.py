"""共同规模的原生教师标签：逐项核对后复用原成员，仅计算新增成员。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import *
from study_full_baselines.fair_training_common import cohort_metadata
from study_full_baselines.build_fair_view import make_link
from study_full_baselines.scale_large import freeze, fingerprint
from study_full_baselines.prepare_labels import one_environment
from study_full_baselines.train import verify_labels


def run(data, source_data, source_labels, output, workers):
    require_host()
    if workers < 1: raise ValueError('workers必须为正。')
    cohort = cohort_metadata(data); manifest = check_data(data)
    source_manifest = check_data(source_data)
    original = SOURCE.parent/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    if source_data.resolve() != original.resolve():
        raise ValueError('只复用原864成员的已核验统一教师标签。')
    original_labels = SOURCE.parent/'dataset_simulation/outputs/unified_teacher_labels_20260925'
    if source_labels.resolve() != original_labels.resolve():
        raise ValueError('来源必须为原统一教师标签。')
    old_protocol = verify_labels(source_data, source_labels)
    verify_sources(old_protocol['source_sha256'])
    fixed = dict(starts=2, sweeps=1, frame=0, uses_payload=False, uses_test=False, seed=0)
    if any(old_protocol[key] != value for key, value in fixed.items()):
        raise ValueError('原教师规则与冻结设置不同。')
    if old_protocol['data_manifest_sha256'] != sha256(source_data/'manifest.json'):
        raise ValueError('原教师标签未绑定当前来源。')
    if sha256(data/'public.npz') != sha256(source_data/'public.npz'):
        raise ValueError('公开导频或探测设置不同。')
    rows = [r for r in manifest['environments'] if r['split'] == 'train']
    old_rows = {r['environment_id']: r for r in source_manifest['environments'] if r['split'] == 'train'}
    records = {r['environment_id']: r for r in json.loads((data/'records.json').read_text())}
    old_data_records = {r['environment_id']: r for r in json.loads((source_data/'records.json').read_text())}
    label_records = {r['environment_id']: r for r in json.loads((source_labels/'records.json').read_text())}
    reusable = []
    for row in rows:
        name = row['environment_id']
        if name not in old_rows: continue
        if row != old_rows[name] or records[name]['file_sha256'] != old_data_records[name]['file_sha256']:
            raise ValueError('同名训练成员的参数或输入数据不一致。')
        for filename, digest in records[name]['file_sha256'].items():
            if sha256(data/row['path']/filename) != digest or sha256(source_data/row['path']/filename) != digest:
                raise ValueError('复用成员文件已改变。')
        record = label_records[name]; path = source_labels/'records'/record['path']
        marker = json.loads(path.with_suffix('.json').read_text())
        if record != marker or marker['fingerprint'] != fingerprint(old_protocol) or sha256(path) != marker['sha256']:
            raise ValueError('原教师记录来源不一致。')
        reusable.append((row, record, path))
    sources = dict(old_protocol['source_sha256'])
    sources.update(source_record(['study_full_baselines/'+n for n in ['prepare_fair_labels.py',
        'fair_training_common.py','build_fair_view.py','scale_large.py','train.py',
        'SCALED_FAIR_TRAINING_PROTOCOL.md']]))
    protocol = dict(**fixed, data_manifest_sha256=sha256(data/'manifest.json'),
        train_ids=cohort['train_ids'], cohort=cohort, source_sha256=sources,
        reuse_source_data_manifest_sha256=sha256(source_data/'manifest.json'),
        reuse_source_label_protocol_sha256=sha256(source_labels/'protocol.json'),
        reuse_source_label_complete_sha256=sha256(source_labels/'complete.json'),
        reused_environment_ids=[r['environment_id'] for r, _, _ in reusable])
    freeze(output, protocol); fp = fingerprint(protocol)
    if (output/'complete.json').exists():
        meta = json.loads((output/'complete.json').read_text())
        if meta['fingerprint'] != fp: raise ValueError('完成记录来源不同。')
        verify_labels(data, output); return
    (output/'records').mkdir(exist_ok=True)
    for row, old, source in reusable:
        path = output/'records'/('environment_%05d.npz'%row['index'])
        make_link(source, path, old['sha256'])
        marker = dict(index=row['index'], environment_id=row['environment_id'], path=path.name,
            sha256=old['sha256'], fingerprint=fp, seconds=0., reused=True,
            source_record_sha256=sha256(source.with_suffix('.json')),
            source_record_path=str(source), original_generation_seconds=old['seconds'])
        if path.with_suffix('.json').exists():
            if json.loads(path.with_suffix('.json').read_text()) != marker:
                raise ValueError('已复用标签的身份改变。')
        else: write_json(path.with_suffix('.json'), marker)
    started = time.perf_counter(); result_records = []; pending = []
    for row in rows:
        marker = output/'records'/('environment_%05d.json'%row['index'])
        if marker.exists():
            record = json.loads(marker.read_text())
            if (record['fingerprint'] != fp or record['environment_id'] != row['environment_id']
                    or sha256(marker.with_suffix('.npz')) != record['sha256']):
                raise ValueError('已提交教师标签改变。')
            result_records.append(record)
        else: pending.append(row)
    def progress(status):
        write_json(output/'progress.json',dict(status=status,completed=len(result_records),total=len(rows),
            reused_environments=len(reusable),workers=workers,pid=os.getpid(),
            seconds=time.perf_counter()-started,at=now()))
    progress('running')
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(one_environment,str(data),str(output),row,fp) for row in pending]
            for job in as_completed(jobs):
                result_records.append(job.result()); progress('running')
    result = []
    for record in sorted(result_records,key=lambda r:r['index']):
        path = output/'records'/record['path']
        if sha256(path) != record['sha256']: raise ValueError('汇总前记录改变。')
        with np.load(path,allow_pickle=False) as f:
            if str(f['environment_id']) != record['environment_id']:
                raise ValueError('标签数组环境身份不同。')
            if np.any(f['objectives'][:,1] < f['objectives'][:,0]-1e-12):
                raise ValueError('教师代理目标下降。')
            result.append(f['control_code'])
    codes = np.concatenate(result)
    if codes.shape != (cohort['train_samples'],128) or np.any(codes<0) or np.any(codes>LEVELS):
        raise ValueError('控制标签形状或合法档位错误。')
    np.save(output/'train_Y_code.npy',codes,allow_pickle=False)
    np.save(output/'train_Y.npy',(codes/LEVELS).astype(np.float32),allow_pickle=False)
    write_json(output/'records.json',sorted(result_records,key=lambda r:r['index']))
    verify_sources(sources); check_data(data); verify_labels(source_data,source_labels)
    write_json(output/'complete.json',dict(status='complete',fingerprint=fp,samples=len(codes),
        train_environments=len(rows),reused_environments=len(reusable),
        newly_computed_environments=len(pending),seconds_this_invocation=time.perf_counter()-started,
        file_sha256={n:sha256(output/n) for n in ['train_Y.npy','train_Y_code.npy','records.json']},at=now()))
    verify_labels(data,output); progress('complete')
    print(json.dumps(dict(status='complete',train_environments=len(rows),reused=len(reusable),
        newly_computed=len(pending))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','source-data','source-labels','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    try:run(a.data.resolve(),a.source_data.resolve(),a.source_labels.resolve(),a.output.resolve(),a.workers)
    except BaseException:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()));raise
