"""沿用已核验统计内核，声明联合校正比较，并核对全部原有方法。"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import sha256,write_json,now,require_host
from study_full_baselines.confirmation_batch import check_carrier
from study_fair_followup import analyze as kernel


def comparisons(identity):
    names=identity['methods'];result=[]
    def add(label,family,terms):
        if set(terms)-set(names) or sum(terms.values())!=0:raise ValueError('比较清单错误。')
        result.append(dict(id=label,family=family,terms=terms,coefficients=[terms.get(n,0) for n in names]))
    add('A_only','components',{'complex_response_cnn':1,'covariance_response':-1})
    for variant in ['joint_block_noise','joint_full_noise']:
        cn='cnn__'+variant;cv='covariance__'+variant
        for label,terms in [
            ('B_only',{cv:1,'covariance_response':-1}),
            ('remove_A',{cn:1,cv:-1}),
            ('remove_B',{cn:1,'complex_response_cnn':-1}),
            ('interaction',{cn:1,'complex_response_cnn':-1,cv:-1,'covariance_response':1}),
            ('vs_previous_best',{cn:1,'cnn__spatial':-1})]:
            add(variant+'__'+label,'components',terms)
    for a in ['covariance','cnn']:
        add(a+'__noise_structure','noise_structure',{a+'__joint_full_noise':1,a+'__joint_block_noise':-1})
    return result


def overlap(folder,identity):
    old=Path(identity['previous_evaluation']);old_identity=json.loads((old/'protocol.json').read_text())
    if sha256(old/'protocol.json')!=identity['previous_protocol_sha256']:raise ValueError('原全基线协议改变。')
    names=[n for n in identity['methods'] if n in old_identity['methods']]
    if len(names)!=6:raise ValueError('原有四方法及两参考覆盖不符。')
    count=0
    for row in identity['rows']:
        sub='records/environment_%05d'%row['index']
        for fc in identity['carriers']:
            for path,ids in [(folder,identity),(old,old_identity)]:
                if check_carrier(path/sub,fc,row,ids) is None:raise ValueError('缺少已提交载频。')
            with np.load(folder/sub/('carrier_%02d.npz'%fc)) as f, np.load(old/sub/('carrier_%02d.npz'%fc)) as prior:
                np.testing.assert_array_equal(f['public_X'],prior['public_X'])
                for name in names:
                    i=identity['methods'].index(name);j=old_identity['methods'].index(name)
                    np.testing.assert_array_equal(f['control_code'][i],prior['control_code'][j])
                    np.testing.assert_allclose(f['metrics'][i,:10],prior['metrics'][j,:10],rtol=2e-10,atol=1e-28,equal_nan=True)
                    count+=1
    return dict(status='passed',at=now(),overlapping_methods=names,cases=count,
        control_codes=count*128,previous_complete_sha256=sha256(old/'complete.json'))


def run(folder,output):
    require_host();identity=json.loads((folder/'protocol.json').read_text())
    verified=overlap(folder,identity)
    # 只替换预先声明的比较系数；原始审核、指标计算和统计定义完全沿用。
    with patch.object(kernel,'comparisons',comparisons):kernel.run(folder,output)
    write_json(folder/'overlap_audit.json',verified)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.evaluation.resolve(),a.output.resolve())
