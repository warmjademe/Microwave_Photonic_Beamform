"""验证两轮最终测试的完整证据，合并同预算结果；不运行训练或重新选控制。"""
import hashlib
import json
from pathlib import Path
import numpy as np

DIRECT = ['mlp', 'dnn', 'cnn', 'rescnn', 'transformer', 'complex_cnn', 'jct']
LABELS = dict(ttd_das='初始实测选择', initial_select64='固定扫描选择', codebook='几何码本',
    coordinate='坐标搜索', spsa='SPSA', done='DONE', de='差分进化', mlp='MLP', dnn='DNN',
    cnn='CNN', rescnn='ResCNN', transformer='Transformer', complex_cnn='复数 CNN', jct='JCT',
    cnn_warm64='本文方法：响应学习＋反馈', cnn__joint_full_noise='本文方法：A＋B',
    complex_response_cnn='仅 A：响应学习', covariance_response='传统响应估计',
    covariance__joint_full_noise='仅 B：联合校正', covariance_warm64='传统响应估计＋反馈',
    teacher='离线教师（额外信息）', mrc='理想数字 MRC（不同硬件）')
LABELS.update({n+'_feedback64':LABELS[n]+'＋反馈' for n in DIRECT})

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''): digest.update(block)
    return digest.hexdigest()

def verified(folder):
    analysis=folder/'analysis'; complete=read(analysis/'complete.json')
    assert sha(folder/'complete.json')==complete['evaluation_complete_sha256']
    for name,digest in complete['file_sha256'].items(): assert sha(analysis/name)==digest,name
    audit=read(analysis/'audit.json'); protocol=read(folder/'protocol.json')
    assert audit['status']=='passed' and audit['environments']==864 and audit['carriers']==17
    assert sha(folder/'protocol.json')==audit['protocol_sha256']
    assert len(protocol['rows'])==864 and protocol['carriers']==list(range(4,21))
    assert len(audit['raw_file_sha256'])==864*17
    for name,digest in audit['raw_file_sha256'].items(): assert sha(folder/name)==digest,name
    terminal=read(folder/'complete.json')
    assert sha(folder/'records.json')==terminal['records_sha256']
    rows=read(analysis/'summary.json'); groups=read(analysis/'groups.json')
    for row in rows:
        tones=[g for g in groups if g['method']==row['method'] and g['group']=='carrier_ghz']
        assert len(tones)==17 and {g['value'] for g in tones}==set(range(4,21))
        assert all(g['environments']==864 for g in tones)
    # 公开的审计摘要不携带内部路径及数万条逐文件索引。
    public={k:v for k,v in audit.items() if k!='raw_file_sha256'}
    public.update(records_sha256=terminal['records_sha256'],
                  analysis_complete_sha256=sha(analysis/'complete.json'),raw_files_verified=864*17)
    return rows,groups,read(analysis/'comparisons.json'),public,protocol

def load_final(project):
    root=project/'dataset_simulation/baseline_results'
    old=root/'20260926_final864_selected'; new=root/'20260927_uniform64_all13'
    rows,groups,pairs,audit,p=verified(old)
    added,extra,comparisons,extension,q=verified(new)
    assert len(rows)==21 and len(added)==14 and sum(r['role']=='baseline' for r in added)==13
    assert p['rows']==q['rows'] and q['source_complete_sha256']==sha(old/'complete.json')
    by_name={r['method']:r for r in rows}; existing=set(by_name)
    old_groups={(g['method'],g['group'],g['value']):g for g in groups}
    for r in added:
        # 扩展统计由 role 表达参考身份；旧统计额外保存 reference 布尔字段。
        # 补齐这一等价字段后逐项比较，所有成绩、计数和预算仍必须完全相同。
        r.setdefault('reference',r['role']=='reference')
        if r['method'] in existing: assert r==by_name[r['method']],r['method']
        else: rows.append(r)
    for g in extra:
        if g['method'] in existing: assert g==old_groups[(g['method'],g['group'],g['value'])]
        else: groups.append(g)
    # 十三个基线重新构成比较族，不沿用此前五项比较的校正值。
    pairs=[r for r in pairs if r['family']!='same_budget_64']
    pairs += [dict(r,family='same_budget_64',terms={'cnn_warm64':1,r['baseline']:-1}) for r in comparisons]
    assert len(rows)==29 and len({r['method'] for r in rows})==29
    assert len([r for r in rows if r['role'] in ['primary','baseline'] and r['measurement_budget']==64])==14
    return rows,groups,pairs,[audit,extension],p,q

def signal_examples(project,old_protocol,new_protocol):
    """仅复用冻结案例的 I/Q；再次核对错误计数及与最终测试原记录的一致性。"""
    diag=project/'dataset_simulation/diagnostics'
    old=project/'dataset_simulation/baseline_results/20260926_final864_selected'
    new=project/'dataset_simulation/baseline_results/20260927_uniform64_all13'
    folder=diag/'20260927_uniform64_all13_figures'
    manifest=read(folder/'manifest.json')
    assert manifest['analysis_complete_sha256']==sha(new/'analysis/complete.json')
    specs=[(0,fc,folder/('constellation_all13_%02d.npz'%fc),'预先固定案例（索引 0）') for fc in [4,12,20]]
    selected=diag/'20260927_constellation_reconstruction'
    case=read(selected/'manifest.json')
    assert case['metrics_replayed_and_matched'] and case['input_noise_and_payload_paired']
    assert sha(selected/'reconstruction_case.npz')==case['files']['reconstruction_case.npz']
    specs.append((363,12,selected/'reconstruction_case.npz',
        '改善示例：12 GHz 下 BER、EVM 同时优于全部基线的 61/864 个案例中，取相对 EVM 改善幅度的中位附近案例'))
    cases=[]
    for index,fc,path,scope in specs:
        with np.load(path) as archive: values={k:archive[k] for k in archive.files}
        names=values['methods'].tolist(); received=values['received_qpsk']; sent=values['sent_qpsk']
        assert received.shape==(14,8,31) and sent.shape==(31,)
        metrics=values['metrics']
        with np.load(new/('records/environment_%05d/carrier_%02d.npz'%(index,fc))) as a: new_raw=a['metrics'].copy()
        with np.load(old/('records/environment_%05d/carrier_%02d.npz'%(index,fc))) as a: old_raw=a['metrics'].copy()
        groups=[]; rows=[]
        for i,name in enumerate(names):
            source,order=(new_raw,new_protocol['methods']) if name in new_protocol['methods'] else (old_raw,old_protocol['methods'])
            np.testing.assert_allclose(metrics[i],source[order.index(name),:10],rtol=1e-12,atol=1e-28)
            err=np.count_nonzero((received[i].real>=0)!=(sent.real>=0))+np.count_nonzero((received[i].imag>=0)!=(sent.imag>=0))
            nmse=float(np.mean(abs(received[i]-sent)**2)/np.mean(abs(sent)**2))
            assert err==int(metrics[i,0]) and int(metrics[i,1])==496
            np.testing.assert_allclose(nmse,metrics[i,2],rtol=1e-12)
            equal=next((j for j in range(i) if np.array_equal(received[i],received[j])),None)
            rows.append(dict(method=name,ber=err/496,rms_evm_percent=100*nmse**.5,
                             received=np.stack([received[i].real,received[i].imag],axis=-1).round(7).tolist(),
                             identical_to=names[equal] if equal is not None else None))
        cases.append(dict(environment_index=index,environment_id=old_protocol['rows'][index]['environment_id'],
            carrier_ghz=fc,scope=scope,outcome_conditioned=index==363,methods=rows,
            sent=np.stack([sent.real,sent.imag],axis=-1).round(7).tolist(),array_sha256=sha(path)))
    return cases
