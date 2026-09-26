"""从冻结母计划逐格取固定前缀，缩减规模并复用原始记录，不按接收成绩选样本。

generation_fingerprint始终标识原始生成协议；新selection_fingerprint单独绑定成员。
NPZ硬链接后内容和SHA完全不变；目录、标记、进度与后续运行均独立。
"""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from generate_native_dataset import check_sources, digest, json_bytes, now, write_json


def select_rows(parent, train=6912, test=1728):
    """只读预先固定的分层索引，函数不接触任何接收指标或完成状态。"""
    limits = dict(train=train, test=test)
    if any(not isinstance(v,int) or v<=0 or v%216 for v in limits.values()):
        raise ValueError('每个划分须为216的正整数倍。')
    if any(limits[s]>parent['splits'][s] for s in limits):
        raise ValueError('子集不能大于母计划。')
    selected = [deepcopy(r) for r in parent['environments']
                if r['stratum_repeat_index'] < limits[r['split']]//216]
    for split,count in limits.items():
        rows = [r for r in selected if r['split']==split]
        coverage = Counter(r['joint_stratum_id'] for r in rows)
        if len(rows)!=count or len(coverage)!=216 or set(coverage.values())!={count//216}:
            raise ValueError('母计划不支持所需的精确联合分层子集。')
    if len({r['seed'] for r in selected})!=len(selected):
        raise ValueError('子集环境种子重复。')
    return selected


def verify_selection(root):
    root = Path(root); manifest=json.loads((root/'manifest.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    if digest(Path(__file__))!=selection['selector_source_sha256']:
        raise ValueError('子集选择源码与冻结版本不一致。')
    parent_path = root/'provenance/parent_manifest.json'
    if digest(parent_path)!=selection['parent_manifest_sha256']:
        raise ValueError('母计划快照SHA不符。')
    parent=json.loads(parent_path.read_text())
    expected=select_rows(parent,**manifest['splits'])
    if expected!=manifest['environments']:
        raise ValueError('数据成员与固定分层选择规则不一致。')
    if hashlib.sha256(json_bytes(expected)).hexdigest()!=selection['selected_rows_sha256']:
        raise ValueError('选后计划SHA不符。')
    if manifest['generation_fingerprint']!=parent['generation_fingerprint']:
        raise ValueError('继承的记录生成指纹不同。')
    actual=hashlib.sha256(json_bytes(selection)).hexdigest()
    if actual!=manifest['selection_fingerprint']:
        raise ValueError('子集指纹不符。')
    return dict(passed=True,selection_fingerprint=actual,total_environments=len(expected))


def resize(parent_root, output, train=6912, test=1728):
    parent_root=Path(parent_root).resolve();output=Path(output).resolve()
    # 持锁证明旧守护任务已退出，防止边复制边继续写入。
    with (parent_root/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=json.loads((parent_root/'manifest.json').read_text())
        check_sources(parent_root,parent)
        rows=select_rows(parent,train,test)
        output.mkdir(parents=True,exist_ok=False);(output/'runs').mkdir();(output/'provenance').mkdir()
        shutil.copyfile(parent_root/'manifest.json',output/'provenance/parent_manifest.json')
        shutil.copyfile(parent_root/'fingerprint_components.json',output/'fingerprint_components.json')
        shutil.copyfile(parent_root/'public.npz',output/'public.npz')
        shutil.copytree(parent_root/'source_snapshot',output/'source_snapshot',
                        ignore=shutil.ignore_patterns('__pycache__','_build'))
        selection=dict(version=1,rule='within each original split and stratum, keep stratum_repeat_index < repeat_limit',
            selector_source_sha256=digest(Path(__file__)),
            repeats_per_stratum=dict(train=train//216,test=test//216),
            parent_manifest_sha256=digest(output/'provenance/parent_manifest.json'),
            parent_generation_fingerprint=parent['generation_fingerprint'],
            selected_rows_sha256=hashlib.sha256(json_bytes(rows)).hexdigest(),
            uses_received_quality=False,uses_completion_status=False)
        write_json(output/'selection.json',selection)
        manifest=deepcopy(parent)
        for key in ('last_progress','superseded_by'):manifest.pop(key,None)
        manifest.update(status='preparing_subset',created_at_utc=now(),
            splits=dict(train=train,test=test),samples_by_split=dict(train=train*17,test=test*17),
            total_environments=len(rows),total_samples=len(rows)*17,environments=rows,
            data_selection='nested_joint_stratified_subset',parent_dataset=parent_root.name,
            generation_fingerprint_scope='original frozen generation recipe and parent plan; record bytes preserved',
            selection_fingerprint=hashlib.sha256(json_bytes(selection)).hexdigest())
        for split,count in manifest['splits'].items():
            design=manifest['sampling_design_by_split'][split]
            design.update(environment_count=count,repeats_per_joint_stratum=count//216,
                method='nested_subset_of_joint_stratified_parent',
                ordering_rule='original parent index and order preserved; no resampling or renumbering')
        write_json(output/'manifest.json',manifest)
        copied=0;reused_complete=0;reused_bytes=0
        for row in rows:
            source=parent_root/row['path'];target=output/row['path'];target.mkdir(parents=True)
            with (source/'.generation.lock').open('a') as env_lock:
                fcntl.flock(env_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                truth=source/'environment_truth.json'
                if digest(truth)!=row['truth_sha256']:raise ValueError('母环境真值哈希不同。')
                shutil.copyfile(truth,target/truth.name)
                for carrier in range(4,21):
                    name='carrier_%02d.npz'%carrier;path=source/name;marker=path.with_suffix('.json')
                    if not path.exists():continue
                    if marker.exists():
                        info=json.loads(marker.read_text())
                        if info['generation_fingerprint']!=parent['generation_fingerprint'] or digest(path)!=info['sha256']:
                            raise ValueError('已生成记录未通过校验，不得复用。')
                        shutil.copyfile(marker,target/marker.name)
                    # 已原子落盘但无marker的记录，由既有collect_environment严格恢复。
                    os.link(path,target/name);copied+=1;reused_bytes+=path.stat().st_size
                if (source/'complete.json').exists():
                    shutil.copyfile(source/'complete.json',target/'complete.json');reused_complete+=1
        report=dict(created_at_utc=now(),parent_dataset=parent_root.name,dataset=output.name,
            planned_environments=manifest['splits'],planned_samples=manifest['samples_by_split'],
            reused_records=copied,reused_complete_environments=reused_complete,
            reused_bytes=reused_bytes,copy_method='hard links for immutable NPZ; metadata copied',
            discarded_record_bytes=False,selection_independent_of_performance=True)
        manifest['status']='prepared';write_json(output/'manifest.json',manifest)
        report['selection_verification']=verify_selection(output)
        write_json(output/'reuse_report.json',report)
        parent.update(status='superseded',superseded_by=output.name,
                      superseded_at_utc=now(),superseded_reason='user reduced original fivefold target to twofold')
        write_json(parent_root/'manifest.json',parent)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train',type=int,default=6912);p.add_argument('--test',type=int,default=1728)
    a=p.parse_args();print(json.dumps(resize(a.parent,a.output,a.train,a.test),ensure_ascii=False))
