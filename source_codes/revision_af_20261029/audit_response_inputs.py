"""逐环境核对两个训练视图的输入完全一致，不重新训练或生成数据。"""
import argparse
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host,sha256,write_json,now
from revision_af_20261029.evaluate_real_response import read,training_audit,fingerprint


def run(project):
    require_host();root=project/'dataset_simulation'
    bundle=root/'baseline_results/20260925_full_baselines/fair_3456/runtime_bundle'
    meta=read(bundle/'manifest.json');members=meta['cohort']['train_ids']
    roots=[Path(meta['training_provenance'][name]['training_data']) for name in ['real','response']]
    outputs=root/'revision_af_20261029/C_response'
    dest=outputs/'training_input_audit.json'
    if dest.exists():raise RuntimeError('Existing audit is not overwritten')
    records=[];bindings={};shapes=[]
    for folder in roots:
        ids=[r['environment_id'] for r in read(folder/'manifest.json')['environments'] if r['split']=='train']
        assert ids==members and len(ids)==3456
        records.append({r['environment_id']:r for r in read(folder/'records.json')})
        for name in ['manifest.json','records.json']:
            bindings[str(folder/name)]=sha256(folder/name)
    pairs=[]
    for index,member in enumerate(members):
        arrays=[];pair=[]
        for folder,rows in zip(roots,records):
            row=rows[member];assert row['samples']==17 and row['path'].startswith('train/')
            path=folder/row['path']/'data.npz'
            assert sha256(path)==row['file_sha256']['data.npz']
            with np.load(path,allow_pickle=False) as f:arrays.append(f['X'].copy())
            pair.append(row['file_sha256']['data.npz'])
        assert arrays[0].shape==arrays[1].shape==(17,2513)
        np.testing.assert_array_equal(*arrays)
        pairs.append(dict(environment_id=member,data_sha256=pair))
        if (index+1)%256==0:print('input audit',index+1,flush=True)
    audit=training_audit(project,bundle)
    for path,h in bindings.items():assert sha256(path)==h
    write_json(dest,dict(status='passed',at=now(),environments=3456,carrier_samples=58752,
        input_values_equal=True,input_shape_per_environment=[17,2513],input_records_verified=6912,
        ordered_record_pairs_sha256=fingerprint(pairs),source_metadata_sha256=bindings,
        training_audit=audit,code_sha256=sha256(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
