"""完成核查：绑定已逐项重算的研究证据、当前模型文件与实际公开页面。"""
import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root):
    if 'huashuo' not in socket.gethostname().lower():
        raise RuntimeError('仅在远程实验工作站核查研究结果')
    evidence=root/'dataset_simulation/ops/research_completion_20261001'
    release=root/'dataset_simulation/site_releases/final864_uniform64_timing_20261001'
    full=read(evidence/'full_audit/audit.json')
    data=read(release/'results-20261001.json')
    sys.path.insert(0,str(root/'source_codes/results_site_3456'))
    from publish_final import validate
    from timing_evidence import load_timing
    validate(release)
    checks=[]

    def passed(requirement, detail, paths):
        checks.append(dict(requirement=requirement,status='passed',detail=detail,
                           evidence={str(p.relative_to(root)):sha(p) for p in paths}))

    assert full['status']=='passed_quality_and_scale_audit'
    assert sha(root/'source_codes/diagnostics/audit_research_goal_20261001.py')==full['source_sha256']
    mapping={
        'ttd_das':['ttd_das'],'codebook':['codebook'],'coordinate':['coordinate'],
        'spsa':['spsa'],'done':['done'],'de':['de'],'mlp':['mlp'],'dnn':['dnn'],
        'cnn':['cnn'],'rescnn':['rescnn'],'transformer':['transformer'],
        'complex_cnn':['complex_cnn'],'jct':['jct'],'teacher':['teacher'],'mrc':['mrc'],
        'frequency_prior':['global_prior','frequency_prior'],
        'joint_response_linear':['joint_relative__base_2sweeps','joint_absolute__base_2sweeps'],
        'response_realcnn':['real_response_cnn__base_2sweeps']}
    dirs={p.name.removeprefix('baseline_') for p in (root/'source_codes').glob('baseline_*')
          if p.is_dir() and p.name!='baseline_common'}
    assert dirs==set(mapping)
    evaluated=set(full['other_evaluated_methods'])
    assert all(set(v)<=evaluated for v in mapping.values())
    assert len(full['all_baseline_phases'])==6
    assert all(p['count']==p['expected']==216 and p['maximum_absolute_difference']<1e-10
               for p in full['all_baseline_phases'].values())
    passed('全部既有baseline代码均完成重跑',dict(directory_mapping=mapping,phases=6),[evidence/'full_audit/audit.json'])

    assert full['train_validation_test_seed_overlap']==0
    assert data['training_environments']==3456 and data['validation_environments']==216
    assert data['evaluation_environments']==864 and data['test_records']==14688
    assert len(data['methods'])==29 and data['method_cases']==425952
    assert all(a['status']=='passed' and a['environments']==864 and a['carriers']==17
               for a in full['final_audits'])
    passed('独立数据划分与完整17载频最终评价',dict(train=3456,validation=216,test=864,seed_overlap=0,
           final_method_cases=425952),[evidence/'full_audit/audit.json',release/'release.json'])

    metrics=['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']
    ranks={}
    for budget,own,expected in [(16,'cnn__joint_full_noise',9),(64,'cnn_warm64',14)]:
        peers=[r for r in data['methods'] if r['measurement_budget']==budget and r['role'] in ['primary','baseline']]
        assert len(peers)==expected
        ranks[budget]={}
        for metric in metrics:
            ordered=sorted(peers,key=lambda r:r[metric],reverse=metric=='paired_output_snr_db')
            assert ordered[0]['method']==own
            ranks[budget][metric]=dict(best=own,value=ordered[0][metric],compared=expected)
    assert ranks=={int(k):v for k,v in full['final_quality_ranks'].items()}
    passed('同信息和同预算的五项接收指标总体领先',ranks,[release/'results-20261001.json'])

    assert len(full['component_effects'])==5
    assert all(e['metric']=='ber' and e['ci95'][1]<0 for e in full['component_effects'])
    passed('组件A、B分别与组合的必要消融',dict(comparisons=5,all_ber_ci_upper_below_zero=True,
           scope='16-probe A/B factorial; 64-probe frozen primary uses A plus feedback'),[evidence/'full_audit/audit.json'])

    scale=full['training_scale']
    assert scale['counts']==[216,432,864,1728,3456] and scale['strictly_nested']
    assert scale['unique_models']==9 and scale['actual_optimizer_states_verified']
    assert scale['evaluation_split']=='validation' and scale['evaluation_environments']==216
    for p,h in scale['source_sha256'].items():
        assert sha(root/p)==h,p
    # 五档×两预算共十个表格位置；864档两预算使用同一模型，共九套权重。
    assert len(scale['budgets'])==10
    assert len({b['weights_sha256'] for b in scale['budgets']})==9
    shared=[b for b in scale['budgets'] if b['training_environments']==864]
    assert len(shared)==2 and len({b['weights_sha256'] for b in shared})==1
    for b in scale['budgets']:
        assert b['seed']==0 and b['used_environments']==b['training_environments']
        assert b['optimizer_updates']==(9200 if b['schedule']=='equal_updates' else
            {216:2320,432:4600,864:9200,1728:18360,3456:36720}[b['training_environments']])
    passed('五档训练规模及匹配优化次数对照',dict(counts=scale['counts'],models=9,
           conclusion='data coverage and optimization updates both matter; no monotone gain at equal updates'),
           [evidence/'full_audit/audit.json'])

    timing=root/'dataset_simulation/diagnostics/20261001_final_timing_run03'
    fresh=load_timing(root,timing,[r['method'] for r in data['methods'] if r['role']!='reference'])
    assert fresh==data['timing']
    passed('控制结果保持一致的正式计算成本',dict(configurations=27,inputs=102,repetitions=3,
           cases=2754,time_samples=8262,hardware_measured=False),[timing/'complete.json',timing/'audit.json'])

    deployment=read(release/'deployment.json');browser=read(evidence/'public_browser_verification.json')
    assert deployment['status']=='published_final864_only_and_history_removed'
    assert len(deployment['checks'])==27
    assert deployment['health']['baselines_html_sha256']==sha(release/'baselines.html')
    assert browser['status']=='passed' and browser['consoleErrors']==[] and browser['noWorkstationNickname']
    assert len(browser['signals'])==4 and '38.210' in browser['timing64'] and '53.026' in browser['timing16']
    assert '10.0136' in browser['snrRanking'] and '12.3131' in browser['rank16']
    assert (evidence/'timing_public_64.jpg').stat().st_size>10000
    gates=read(evidence/'publish_timing_checks.json')
    assert gates['negative_gate_checks']==11 and gates['negative_timing_checks']==10
    passed('完整结果网站上线、旧公开资产下线及交互核验',dict(http_routes=27,negative_checks=21,
           signal_cases=4,budgets=[16,64],historical_public_data=False),
           [release/'deployment.json',evidence/'public_browser_verification.json',evidence/'timing_public_64.jpg'])

    report=root/'dataset_simulation/docs/研究实验完成报告_20261001.txt'
    assert all(s in report.read_text() for s in ['1.2726','0.5894','9200','38.210','14 GHz','数据量不是'])
    passed('最终中文报告与复现入口',dict(quality_scope='all864',timing_scope='fixed102',
           limitations=['simulation only','single training seed','not fastest','not every condition best',
                        'no complete A+B-plus-feedback independent run']),
           [report,root/'README.md',root/'RESEARCH_GOAL.md',root/'source_codes/results_site_3456/README.md'])

    result=dict(status='complete_requirements_verified',at=datetime.now(timezone.utc).isoformat(),
                goal_scope='all existing baselines; multi-metric method improvement; training-size study; final confirmation and website',
                checks=checks,unresolved_required_items=[],source_sha256=sha(Path(__file__)))
    (evidence/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=result['status'],requirements=len(checks),unresolved=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    run(p.parse_args().project.resolve())
