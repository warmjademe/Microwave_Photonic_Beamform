"""沿原采样和物理核增加独立训练环境，完整保留原训练成员，不生成新测试答案。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from generate_dataset import plan_environments
from our_method_quality_rank.generate import one_environment


def plan(project,base,target,seed):
    old=check_data(base)
    retained=[dict(r) for r in old['environments'] if r['split']=='train']
    extra=target-len(retained)
    if extra<=0 or extra%216 or target%216:raise ValueError('新增数及总数须为216的正整数倍。')
    candidates,designs=plan_environments(extra,216,seed,'joint_stratified')
    added=[dict(r) for r in candidates if r['split']=='train']
    excluded={int(r['seed']) for r in old['environments']}
    for split in ['train','test']:
        f=project/('dataset_simulation/dataset_'+split+'/environments.json')
        excluded.update(int(r['environment_id'].split('-')[-1]) for r in json.loads(f.read_text()))
    # 原864/216独立诊断数据的测试环境永不加入扩展训练。
    original=json.loads((project/'dataset_simulation/outputs/quality_rank_hybrid_20260925/manifest.json').read_text())
    excluded.update(int(r['seed']) for r in original['environments'])
    if excluded & {r['seed'] for r in added}:raise ValueError('新增训练种子与既有训练/测试交叠。')
    for r in added:
        r['index']+=len(retained)
        r['path']='train/environment_%05d'%r['index']
        r['stratum_repeat_index']+=len(retained)//216
    rows=retained+added
    groups={}
    for r in rows:groups.setdefault(r['joint_stratum_id'],[]).append(r['stratum_repeat_index'])
    if len(groups)!=216 or any(sorted(v)!=list(range(target//216)) for v in groups.values()):
        raise ValueError('扩展未保持216个联合分层及嵌套重复编号。')
    if [r['index'] for r in rows]!=list(range(target)):raise ValueError('扩展顺序错误。')
    return old,retained,added,rows,designs


def run(project,base,output,target,seed,workers,plan_only=False):
    require_host();old,retained,added,rows,designs=plan(project,base,target,seed)
    if plan_only:
        print(json.dumps(dict(status='plan_passed',retained=len(retained),added=len(added),
            total=target,samples=target*17,strata=216,seed=seed,test_samples=0)),flush=True)
        return
    self_name='study_full_baselines/expand_training.py'
    sources={**old['source_sha256'],**source_record([self_name,
        'study_full_baselines/common.py','generate_dataset.py',
        'baseline_common/channel.py','baseline_common/config.py'])}
    identity=dict(base_manifest_sha256=sha256(base/'manifest.json'),target_train_environments=target,
        additional_master_seed=seed,source_sha256=sources,public_sha256=sha256(base/'public.npz'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'expansion_identity.json').exists():
        if json.loads((output/'expansion_identity.json').read_text())!=identity:raise ValueError('扩展来源改变。')
        manifest=json.loads((output/'manifest.json').read_text())
    else:
        write_json(output/'expansion_identity.json',identity)
        shutil.copyfile(base/'public.npz',output/'public.npz')
        manifest={**old,'status':'running','schema':'quality-rank-hybrid-training-expansion-v1',
            'created_at_utc':now(),'environments':rows,'train_samples':target*17,'test_samples':0,
            'master_seed':seed,'expansion_identity':identity,'source_sha256':sources,
            'sampling_design':{'retained_from':str(base),'additional':designs['train']},
            'test_scope':'training-only expansion; existing exploratory test stays separate; fresh confirmation not generated here',
            'note':'All new records use the identical five-frame generator; no changed propagation or hardware distribution.'}
        manifest.pop('finished_at_utc',None);manifest.pop('records_sha256',None)
        write_json(output/'manifest.json',manifest)
        for name in sources:
            dst=output/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dst)
    previous={r['environment_id']:r for r in json.loads((base/'records.json').read_text())}
    records=[];todo=[];started=time.perf_counter()
    for row in rows:
        folder=output/row['path']
        if row['index']<len(retained):
            source=base/row['path'];original_record=previous[row['environment_id']]
            folder.mkdir(parents=True,exist_ok=True)
            for name,digest in original_record['file_sha256'].items():
                if sha256(source/name)!=digest:raise ValueError('保留样本的原始哈希改变。')
                if not (folder/name).exists():shutil.copyfile(source/name,folder/name)
                if sha256(folder/name)!=digest:raise ValueError('保留样本复制失败。')
            write_json(folder/'complete.json',original_record)
        marker=folder/'complete.json'
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['environment_id']!=row['environment_id'] or m['path']!=row['path']:
                raise ValueError('扩展成员身份或路径不同。')
            for name,digest in m['file_sha256'].items():
                if sha256(folder/name)!=digest:raise ValueError('完整扩展样本的哈希改变。')
            records.append(m)
        else:
            if folder.exists():raise ValueError('未完成环境残留需核查：'+str(folder))
            todo.append(row)
    def progress(status):
        value=dict(status=status,completed_environments=len(records),total_environments=len(rows),
            retained_environments=len(retained),added_environments=len(added),workers=workers,
            seconds=time.perf_counter()-started,pid=os.getpid(),at=now())
        write_json(output/'progress.json',value);return value
    progress('running')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(one_environment,str(output),row) for row in todo]
        for job in as_completed(jobs):
            records.append(job.result());value=progress('running')
            if len(records)%20==0:print(json.dumps(value),flush=True)
    verify_sources(sources);check_data(base)
    write_json(output/'records.json',sorted(records,key=lambda r:r['environment_id']))
    manifest.update(status='complete',finished_at_utc=now(),records_sha256=sha256(output/'records.json'),
        expansion_seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest);check_data(output)
    progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','base','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    try:run(a.project,a.base,a.output,a.target,a.seed,a.workers,a.plan_only)
    except BaseException:
        if not a.plan_only:
            a.output.mkdir(parents=True,exist_ok=True)
            write_json(a.output/('failure_%d.json'%time.time()),dict(traceback=traceback.format_exc(),at=now()))
        raise
