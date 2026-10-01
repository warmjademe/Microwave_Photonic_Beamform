"""对全基线、最终多指标、消融及五档训练规模做当前文件核验，不重新训练。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'results_site_3456'))
from study_full_baselines.report_training_scale import inspect_model,quality_rows,COUNTS
from study_full_baselines.audit_results import read_phase,check_summary
from study_full_baselines.common import require_host,sha256,write_json,verify_sources
from final_evidence import load_final

def read(p):return json.loads(Path(p).read_text())

def run(project,output):
    require_host()
    if output.exists():raise FileExistsError(output)
    base=project/'dataset_simulation';root=base/'baseline_results/20260925_full_baselines'
    data=base/'outputs/quality_rank_hybrid_20260925'
    validation=[r for r in read(data/'manifest.json')['environments'] if r['split']=='test']
    old_audit=read(root/'analysis/audit.json');phases={};method_names=set()
    for name in old_audit['phases']:
        audit,arrays=read_phase(root/name,validation,sha256(data/'manifest.json'))
        audit.update(check_summary(root/name,arrays,validation))
        method_names.update(arrays['methods'])
        phases[name]={k:v for k,v in audit.items() if k!='record_sha256'}
    required=['ttd_das','codebook','coordinate','spsa','done','de','mlp','dnn','cnn','rescnn',
        'transformer','complex_cnn','jct','teacher','mrc','global_prior','frequency_prior',
        'ridge_response','covariance_response','complex_response_cnn']
    assert set(required)<=method_names
    assert any('real_response_cnn' in n for n in method_names)
    assert any('joint_relative' in n for n in method_names)
    assert any('quality_' in n for n in method_names)
    budgets=[];members={};files={}
    for n in COUNTS:
        for schedule in ['fixed_epochs','equal_updates']:
            if n==864 and schedule=='equal_updates':budgets.append(dict(budgets[-1],schedule=schedule));continue
            row,ids,hashes,resume,weights=inspect_model(project,n,schedule)
            if n in members:assert members[n]==ids
            members[n]=ids;files.update(hashes);budgets.append(row)
    for a,b in zip(COUNTS[:-1],COUNTS[1:]):assert set(members[a])<set(members[b])
    assert {r['optimizer_updates'] for r in budgets if r['schedule']=='equal_updates'}=={9200}
    quality,paired,quality_sources=quality_rows(project,False)
    assert len(quality)==15
    rows,groups,pairs,audits,p,q=load_final(project)
    verify_sources(p['source_sha256']);verify_sources(q['source_sha256'])
    registry=read(p['dataset_registry']);assert sha256(p['dataset_registry'])==p['dataset_registry_sha256']
    assert [registry['splits'][n]['environments'] for n in ['train','validation','test']]==[3456,216,864]
    assert not any(registry['pairwise_seed_overlap'].values())
    ranks={}
    for budget,own in [(16,'cnn__joint_full_noise'),(64,'cnn_warm64')]:
        selected=[r for r in rows if r['role'] in ['primary','baseline'] and r['measurement_budget']==budget]
        assert len(selected)==(9 if budget==16 else 14)
        ranks[budget]={}
        for metric in ['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db']:
            ordered=sorted(selected,key=lambda r:r[metric],reverse=metric=='paired_output_snr_db')
            ranks[budget][metric]=dict(best=ordered[0]['method'],value=ordered[0][metric],compared=len(ordered))
            assert ordered[0]['method']==own
    component=[r for r in pairs if r['family']=='components_joint' and r['metric']=='ber']
    assert len(component)==5 and all(r['ci95'][1]<0 for r in component)
    output.mkdir(parents=True)
    report=dict(status='passed_quality_and_scale_audit',source_sha256=sha256(Path(__file__)),
        all_baseline_phases=phases,required_methods_covered=required,other_evaluated_methods=sorted(method_names),
        final_audits=audits,final_quality_ranks=ranks,component_effects=component,
        training_scale=dict(counts=COUNTS,budgets=budgets,strictly_nested=True,unique_models=9,
            actual_optimizer_states_verified=True,quality=quality,paired=paired,source_sha256=files,
            quality_source_sha256=quality_sources,evaluation_split='validation',evaluation_environments=216),
        dataset_registry_sha256=sha256(p['dataset_registry']),train_validation_test_seed_overlap=0,
        new_training=False,remaining_requirements=['final controlled timing audit','website verification evidence archive'])
    write_json(output/'audit.json',report)
    print(json.dumps(dict(status=report['status'],baseline_phases=len(phases),final_configs=len(rows),scales=COUNTS)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.output.resolve())
