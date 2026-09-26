"""重新核对训练内校正的逐案例、汇总、控制档位和测量残差。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,write_json,now,LEVELS


def run(root):
    require_host();c=json.loads((root/'complete.json').read_text());p=json.loads((root/'protocol.json').read_text())
    if c['status']!='complete' or p['test_used'] or p['extra_feedback'] or p['cases']!=72:
        raise ValueError('诊断范围不一致或尚未完成。')
    if sha256(root/'all_metrics.npz')!=c['metrics_sha256'] or sha256(root/'summary.json')!=c['summary_sha256']:
        raise ValueError('完成标记哈希不一致。')
    s=json.loads((root/'summary.json').read_text())
    with np.load(root/'all_metrics.npz') as f:m=f['metrics'];codes=f['control_code'];r=f['weighted_residuals']
    if m.shape!=(72,4,10) or codes.shape!=(72,4,128) or np.any(codes<0) or np.any(codes>LEVELS):
        raise ValueError('结果形状或控制码错误。')
    if np.any(r[:,:,1]>r[:,:,0]+1e-9*np.maximum(1,r[:,:,0])):raise ValueError('加权残差增加。')
    for i in range(72):
        file=root/'records'/('%03d.npz'%i);meta=json.loads(file.with_suffix('.json').read_text())
        if sha256(file)!=meta['sha256']:raise ValueError('案例哈希错误。')
        with np.load(file) as f:
            if not np.array_equal(f['metrics'],m[i]) or not np.array_equal(f['control_code'],codes[i]):
                raise ValueError('案例与汇总不同。')
    for i,rec in enumerate(s['records']):
        a=m[:,i]
        expected=dict(ber=a[:,0].sum()/a[:,1].sum(),ser=a[:,3].sum()/a[:,4].sum(),
            rms_evm_percent=100*np.sqrt(a[:,2].mean()),paired_output_snr_db=10*np.log10(a[:,7].sum()/a[:,8].sum()))
        for k,v in expected.items():
            if not np.isclose(v,rec[k],rtol=1e-12,atol=0):raise ValueError('汇总不同：'+k)
    report=dict(status='passed',at=now(),cases=72,training_environments=24,test_used=False,
        fields_recomputed=16,control_codes_checked=int(codes.size),public_measurements=16,extra_feedback=0,
        weighted_residual_nonincrease=True,source_sha256=sha256(Path(__file__)),summary_sha256=c['summary_sha256'])
    write_json(root/'audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--input',type=Path,required=True)
    run(a.parse_args().input.resolve())
