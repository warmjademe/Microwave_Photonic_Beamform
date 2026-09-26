"""将完成的原生数据按原划分提取为dataset_train/dataset_test，白名单数值数组。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from compact_dataset import SCHEMA,X_DIM,FIELDS,pack_observation,sha256,CompactDataset
from native_sim.data import NativeDataset
from generate_native_dataset import write_json,now
from run_native_baselines import verify_core

INPUT_KEYS=('combined_iq_a','quality','noise_symbol_var_a2','probe_apd_dc_a')


def export_split(source,output,split):
    source=Path(source).resolve();output=Path(output).resolve()
    dataset=NativeDataset(source);verify_core(dataset)
    rows=dataset.environments(split);n=len(rows)*len(dataset.carriers)
    version=dict(source_manifest_sha256=sha256(source/'manifest.json'),
        split=split,exporter_sha256=sha256(Path(__file__)),
        reader_sha256=sha256(Path(__file__).with_name('compact_dataset.py')))
    signature=hashlib.sha256(json.dumps(version,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=output/'manifest.json'
        if manifest_path.exists():
            meta=json.loads(manifest_path.read_text())
            if meta.get('export_fingerprint')!=signature:raise ValueError('已有目录属于另一导出版本。')
            if meta['status']=='complete':
                CompactDataset(output,verify_hashes=True);return meta
        else:
            unexpected=[p.name for p in output.iterdir() if p.name!='.export.lock']
            if unexpected:raise FileExistsError('拒绝覆盖非空目录：'+str(output))
            envs=[dict(environment_id=r['environment_id'],source_path=r['path']) for r in rows]
            write_json(output/'environments.json',envs)
            shutil.copyfile(source/'public.npz',output/'public.npz')
            meta=dict(schema=SCHEMA,status='exporting',split=split,created_at_utc=now(),
                sample_count=n,environment_count=len(rows),carriers_ghz=dataset.carriers,
                source_dataset_relative=os.path.relpath(source,output),export_fingerprint=signature,
                provenance=version,generation_fingerprint=dataset.manifest['generation_fingerprint'],
                selection_fingerprint=dataset.manifest.get('selection_fingerprint'),
                signal_config=dataset.cfg.to_dict(),X_shape=[n,X_DIM],Y_shape=[n,128],
                X_fields=FIELDS,Y_definition='first 64 delay code/76; last 64 attenuation code/24',
                Y_code_definition='uint8, delay 0..76, attenuation 0..24',
                ordering='source environment order, then 4..20GHz; index//17 selects environment',
                excluded_from_inputs=['branch optical caches','true channel','payload truth','environment path parameters'],
                normalisation='none applied; raw physical units; fit model statistics on train only',
                precision='public float64 noise/DC cast to float32; I/Q and scores preserved at source float32',
                completed_samples=0)
            write_json(manifest_path,meta)
            for name,shape,dtype in [('X.npy',(n,X_DIM),'float32'),('Y.npy',(n,128),'float32'),
                ('Y_code.npy',(n,128),'uint8'),('record_sha256.npy',(n,32),'uint8')]:
                a=np.lib.format.open_memmap(output/name,mode='w+',dtype=dtype,shape=shape);a.flush();del a
            write_json(output/'progress.json',dict(completed_samples=0,updated_at_utc=now()))
        arrays={name:np.load(output/(name+'.npy'),mmap_mode='r+',allow_pickle=False)
                for name in ('X','Y','Y_code','record_sha256')}
        progress=json.loads((output/'progress.json').read_text());begin=progress['completed_samples']
        levels=np.r_[np.full(64,76),np.full(64,24)];start=time.perf_counter()
        try:
            for ei,row in enumerate(rows):
                if (ei+1)*len(dataset.carriers)<=begin:continue
                for ci,carrier in enumerate(dataset.carriers):
                    index=ei*len(dataset.carriers)+ci
                    if index<begin:continue
                    path=dataset.record_path(row,ci);marker=json.loads(path.with_suffix('.json').read_text())
                    actual=sha256(path)
                    if actual!=marker['sha256']:raise ValueError('原始记录SHA核验失败：'+str(path))
                    with np.load(path,allow_pickle=False) as f:
                        if str(f['generation_fingerprint'].item())!=dataset.fingerprint:
                            raise ValueError('原始记录生成指纹不同。')
                        inputs={k:f[k] for k in INPUT_KEYS};code=f['control_code'];old_y=f['control']
                    if code.shape!=(128,) or not np.issubdtype(code.dtype,np.integer):
                        raise ValueError('控制码格式无效。')
                    if np.any(code<0) or np.any(code>levels):raise ValueError('控制码越界。')
                    y=(code/levels).astype(np.float32)
                    if not np.array_equal(y,old_y):raise ValueError('控制码与原始Y不符。')
                    arrays['X'][index]=pack_observation(inputs,carrier)
                    arrays['Y'][index]=y;arrays['Y_code'][index]=code
                    arrays['record_sha256'][index]=np.frombuffer(bytes.fromhex(actual),np.uint8)
                if (ei+1)%100==0 or ei==len(rows)-1:
                    for a in arrays.values():a.flush()
                    progress=dict(split=split,completed_samples=(ei+1)*len(dataset.carriers),
                                  planned_samples=n,updated_at_utc=now(),elapsed_seconds=time.perf_counter()-start)
                    write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            for a in arrays.values():a.flush()
            # 全量数值核查不依赖随机抽样；每一行已与源字段/源SHA绑定。
            for i in range(0,n,4096):
                if not np.all(np.isfinite(arrays['X'][i:i+4096])):raise ValueError('导出X非有限。')
                expected=(arrays['Y_code'][i:i+4096]/levels).astype(np.float32)
                if not np.array_equal(arrays['Y'][i:i+4096],expected):raise ValueError('导出Y码不符。')
            names=['X.npy','Y.npy','Y_code.npy','record_sha256.npy','public.npz','environments.json']
            meta.update(status='complete',completed_samples=n,finished_at_utc=now(),
                file_sha256={name:sha256(output/name) for name in names},
                file_bytes={name:(output/name).stat().st_size for name in names},
                every_source_record_sha_verified=True,every_source_Y_matched=True)
            write_json(manifest_path,meta)
            CompactDataset(output,verify_hashes=True)
            return meta
        except BaseException as exc:
            meta.update(status='interrupted_or_failed',error_type=type(exc).__name__,error=str(exc))
            write_json(manifest_path,meta);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output-parent',type=Path,required=True)
    p.add_argument('--split',choices=['train','test','both'],default='both');a=p.parse_args()
    for split in (['train','test'] if a.split=='both' else [a.split]):
        m=export_split(a.source,a.output_parent/('dataset_'+split),split)
        print(json.dumps(dict(status=m['status'],split=split,samples=m['sample_count'],bytes=sum(m['file_bytes'].values()))),flush=True)


if __name__=='__main__':main()
