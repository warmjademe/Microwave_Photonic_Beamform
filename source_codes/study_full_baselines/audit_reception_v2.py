"""六条完整评分流水线的扩展审计，覆盖参数量匹配实数CNN。

复用冻结版逐文件/汇总核验，再追加预算、参考帧、模型与精确跨流水线检查。
不修改正在运行的评分源码；部分审计不能生成完整通过状态。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.audit_results import read_phase, check_summary

PHASES = ['classic', 'learned_evaluation', 'evaluation_components',
          'evaluation_real_response_cnn', 'evaluation_scaling_1728',
          'evaluation_scaling_3456']
CROSSES = [(('classic','ttd_das'), ('learned_evaluation','public16')),
           (('learned_evaluation','covariance_response'), ('evaluation_components','covariance__base_2sweeps')),
           (('learned_evaluation','complex_response_cnn'), ('evaluation_components','cnn__base_2sweeps'))]


def check_budget(folder, value):
    protocol=json.loads((folder/'protocol.json').read_text())
    if protocol.get('frame') != 5 or protocol.get('apd_draws',protocol.get('draws')) != 8:
        raise ValueError('评价帧或APD抽样数与统一协议不同。')
    if 'bundle' in protocol:
        expected={m['name']:m.get('probes',16) for m in protocol['bundle']['methods']}
    else:
        expected={m:(16 if m=='ttd_das' else 64) for m in ONLINE_CLASSIC}
    verified=0
    for mi, method in enumerate(value['methods']):
        a=value['values'][:,:,mi]
        if method not in ['teacher','mrc']:
            index=value['metric_order'].index('feedback_calls')
            if not np.all(a[...,index]==expected[method]):
                raise ValueError('实际反馈数与方法声明不同：'+method)
            verified+=a.shape[0]*a.shape[1]
        for field in ['controller_seconds','feedback_simulator_seconds']:
            if field in value['metric_order'] and method!='mrc':
                cost=a[...,value['metric_order'].index(field)]
                if not np.isfinite(cost).all() or np.any(cost<0):
                    raise ValueError('软件/反馈模拟器耗时无效。')
    return dict(exact_feedback_budgets_checked=verified,frame5=True,paired_apd_draws8=True)


def strict_cross(arrays, left, right):
    lf,lm=left;rf,rm=right;a=arrays[lf];b=arrays[rf]
    if a is None or b is None:
        return dict(left=left,right=right,status='waiting',environments=0)
    ai=a['methods'].index(lm);bi=b['methods'].index(rm)
    common=sorted(set(a['indices']) & set(b['indices']))
    for index in common:
        ix=a['indices'].index(index);jx=b['indices'].index(index)
        if a['ids'][ix]!=b['ids'][jx]: raise ValueError('跨流水线环境身份不同。')
        if not np.array_equal(a['controls'][ix,:,ai],b['controls'][jx,:,bi]):
            raise ValueError('跨流水线相同方法的控制码不同。')
        # 功率可能远小于1e-12；禁止用1e-12的绝对容差掩盖功率差异。
        if not np.allclose(a['values'][ix,:,ai,:10],b['values'][jx,:,bi,:10],rtol=1e-12,atol=0):
            raise ValueError('跨流水线接收计数或功率不一致。')
    return dict(left=left,right=right,status='matched' if common else 'waiting',
        environments=len(common),quality_relative_tolerance=1e-12,quality_absolute_tolerance=0)


def training_artifacts(study, rows, digest):
    folder=study/'learned';protocol=json.loads((folder/'protocol.json').read_text())
    if (protocol['data_manifest_sha256']!=digest or protocol['epochs']!=40
            or protocol['seed']!=0 or protocol['validation'] or protocol['test_tuning']
            or protocol['test_ids']!=[r['environment_id'] for r in rows]):
        raise ValueError('学习基线的训练协议或共同测试成员不同。')
    verify_sources(protocol['source_sha256']);reports=[]
    expected_fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if set(protocol['train_ids']) & set(protocol['test_ids']):
        raise ValueError('训练与测试成员有交集。')
    for method in ['mlp']+DEEP_METHODS:
        model=folder/method;meta=json.loads((model/'complete.json').read_text())
        weight=model/('model/weights.npz' if method=='mlp' else 'weights.pt')
        if (meta['fingerprint']!=expected_fingerprint
                or meta['epochs']!=40 or meta['seed']!=0 or not meta['full_replay_equal']
                or sha256(weight)!=meta['weights_sha256']
                or sha256(model/'predictions.npy')!=meta['predictions_sha256']):
            raise ValueError('模型/预测身份或训练次数不符。')
        prediction=np.load(model/'predictions.npy',mmap_mode='r')
        if prediction.shape!=(len(rows)*17,128) or not np.isfinite(prediction).all():
            raise ValueError('预测数量或数值有误。')
        reports.append(dict(method=method,weights_sha256=meta['weights_sha256'],
            predictions_sha256=meta['predictions_sha256'],epochs=40,seed=0,
            training_record_reports_full_replay=True))
    return dict(status='passed_artifact_checks',models=reports,
        protocol_sha256=sha256(folder/'protocol.json'),
        normalization_sha256=sha256(folder/'normalization.npz'),
        note='This audit checks completed artifacts; weight-to-prediction numerical replay was executed by frozen training code.')


def collect(data, study, partial):
    manifest=check_data(data);rows=[r for r in manifest['environments'] if r['split']=='test']
    if len(rows)!=216 or [r['index'] for r in rows]!=list(range(len(rows))):
        raise ValueError('本入口限定旧216环境探索，独立确认使用另行冻结协议。')
    digest=sha256(data/'manifest.json');reports={};arrays={}
    for phase in PHASES:
        folder=study/phase
        info,value=read_phase(folder,rows,digest,partial)
        if value is not None: info.update(check_budget(folder,value))
        if info['status']=='complete': info.update(check_summary(folder,value,rows))
        reports[phase]=info;arrays[phase]=value
    crosses=[strict_cross(arrays,l,r) for l,r in CROSSES]
    complete=all(r['status']=='complete' for r in reports.values()) and all(
        c['status']=='matched' and c['environments']==len(rows) for c in crosses)
    report=dict(status='passed' if complete else 'partial_not_complete',at=now(),
        data_sha256=digest,phases=reports,cross_pipeline_checks=crosses,
        all_required_phases_complete=complete,
        training_artifacts=training_artifacts(study,rows,digest),
        scientific_scope='Old216 exploratory receiver results; not sealed confirmation',
        source_sha256=source_record(['study_full_baselines/audit_reception_v2.py',
                                    'study_full_baselines/audit_results.py']))
    return report,arrays,rows


def run(data,study,output,partial):
    require_host();report,_,_=collect(data,study,partial)
    if output.exists(): raise FileExistsError('审计报告不能覆盖。')
    write_json(output,report)
    print(json.dumps(dict(status=report['status'],counts={k:v['count'] for k,v in report['phases'].items()},
        cross_checks=report['cross_pipeline_checks'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','study','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    run(a.data,a.study,a.output,a.allow_partial)
