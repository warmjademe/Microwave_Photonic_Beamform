"""核对共同模型包接收演练：117案例、实际反馈轨迹和完整复用。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host,sha256,write_json,now,LEVELS,source_record,verify_sources
from study_full_baselines.confirmation_batch import committed_environment
from study_full_baselines.runtime_bundle import verify as verify_bundle


def run(project,folder):
    require_host();output=folder/'integrity_verification.json'
    if output.exists(): raise FileExistsError('不覆盖接收演练核查。')
    identity=json.loads((folder/'protocol.json').read_text())
    if identity['scope']!='old_runtime_rehearsal' or identity['final_confirmation']:
        raise ValueError('此核查限定原测试0的三个频率。')
    if len(identity['rows'])!=1 or identity['rows'][0]['index']!=0 or identity['carriers']!=[4,12,20]:
        raise ValueError('演练范围改变。')
    verify_sources(identity['source_sha256']);verify_bundle(Path(identity['runtime_bundle']))
    for name,digest in identity['source_sha256'].items():
        if sha256(folder/'source_snapshot'/name)!=digest: raise ValueError('执行源码快照改变。')
    checkpoint=json.loads((folder/'before_noop.json').read_text())
    for relative,item in checkpoint['files'].items():
        f=folder/relative
        if sha256(f)!=item['sha256'] or f.stat().st_mtime_ns!=item['mtime_ns']:
            raise ValueError('完整复用时改写已提交文件。')
    attempts=[json.loads(f.read_text()) for f in sorted((folder/'attempts').glob('*.json'))]
    if [(a['reused_environments'],a['newly_computed_environments']) for a in attempts]!=[(0,1),(1,0)]:
        raise ValueError('未完成一次计算加一次完整复用。')
    workers=sorted(f.name for f in (folder/'workers').glob('*.json') if not f.name.endswith('_progress.json'))
    if workers!=checkpoint['worker_files']: raise ValueError('完整复用时仍新建worker。')
    reference=project/'dataset_simulation/diagnostics/20260926_confirmation_receiver_preflight'
    proof=json.loads((reference/'summary.json').read_text())
    old=json.loads((reference/'protocol.json').read_text())['methods']
    indices=[old.index(name) for name in identity['methods']]
    record=committed_environment(folder,identity['rows'][0],identity)
    if record is None: raise ValueError('演练环境未完整提交。')
    cases=0; traces=0; files={};counts=[0,1,3,4,5,6]
    for carrier in record['carriers']:
        fc=carrier['carrier_ghz'];path=folder/carrier['path'];ref=reference/'records'/('carrier_%02d.npz'%fc)
        if sha256(ref)!=proof['result_and_reference_sha256'][str(ref)]: raise ValueError('参考接收记录改变。')
        with np.load(path) as f: actual={k:f[k].copy() for k in f.files}
        with np.load(ref) as f: previous={k:f[k].copy() for k in f.files}
        np.testing.assert_array_equal(actual['control_code'],previous['control_code'][indices])
        np.testing.assert_array_equal(actual['metrics'][:,counts],previous['metrics_gpu'][indices][:,counts])
        np.testing.assert_allclose(actual['metrics'][:,:10],previous['metrics_gpu'][indices,:10],rtol=1e-12,atol=0.,equal_nan=True)
        np.testing.assert_allclose(actual['mrc_physical_reference_snr_db'],previous['mrc_physical_reference_snr_db'],rtol=1e-12,atol=0.)
        source=Path(identity['data_path'])/identity['rows'][0]['path']/'data.npz'
        with np.load(source) as f: np.testing.assert_array_equal(actual['public_X'],f['X'][fc-4])
        if actual['control_code'].shape!=(39,128): raise ValueError('不是全部39项方法。')
        np.testing.assert_array_equal(actual['control_code'][-1],np.full(128,-1))
        for i,name in enumerate(identity['ordinary_methods']):
            if actual['metrics'][i,10]!=identity['measurement_budgets'][name]: raise ValueError('测量预算不同。')
            trace_key=name+'__trace_control_code';score_key=name+'__trace_scores'
            if trace_key in actual:
                np.testing.assert_array_equal(actual[trace_key],previous[trace_key])
                np.testing.assert_allclose(actual[score_key],previous[score_key],rtol=1e-12,atol=0.)
                traces+=len(actual[trace_key])
        if np.any(actual['control_code'][:-1]<0) or np.any(actual['control_code'][:-1]>LEVELS):
            raise ValueError('控制档位非法。')
        cases+=39;files[str(path)]=sha256(path);files[str(ref)]=sha256(ref)
    claim=project/'dataset_simulation/ops/full_baselines_20260925/fresh_confirmation_claim.json'
    if claim.exists(): raise ValueError('旧演练意外登记了新留出计划。')
    result=dict(status='passed_runtime_confirmation_rehearsal',at=now(),cases=cases,methods=39,
        all_controls_exact=True,all_public_inputs_bitwise_equal=True,all_integer_quality_fields_exact=True,
        all_ten_quality_fields_matched=True,quality_relative_tolerance=1e-12,quality_absolute_tolerance=0.,
        feedback_trace_rows_checked=traces,preserved_files=len(checkpoint['files']),no_worker_on_complete_resume=True,
        source_sha256=source_record(['study_full_baselines/audit_confirmation_execute.py']),
        result_and_reference_sha256=files,protocol_sha256=sha256(folder/'protocol.json'),
        final_confirmation=False,new_environment_signals_generated=0,new_plan_claim_created=False,
        formal_speed_comparison=False)
    write_json(output,result);print(json.dumps({k:v for k,v in result.items() if k!='result_and_reference_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    a=p.parse_args();run(a.project.resolve(),a.input.resolve())
