"""用真实旧接收记录检查分析器会拒绝关键数据错误；不生成新的传播数据。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.analyze_confirmation import check_values
import numpy as np
from study_full_baselines.common import require_host, sha256, source_record, write_json, now


def run(project, output):
    require_host()
    if output.exists():
        raise FileExistsError('不覆盖已有错误注入检查。')
    root = project/'dataset_simulation/diagnostics/20260926_confirmation_runtime_rehearsal'
    identity = json.loads((root/'protocol.json').read_text())
    source = root/'records/environment_00000/carrier_04.npz'
    with np.load(source) as f:
        original = {k:f[k].copy() for k in f.files}
    with np.load(Path(identity['runtime_bundle'])/'public.npz') as f:
        public = {k:f[k].copy() for k in f.files}
    rows = check_values(original, 4, identity, public)
    checks = []
    def reject(name, mutation):
        arrays = {k:v.copy() for k,v in original.items()}
        mutation(arrays)
        try:
            check_values(arrays, 4, identity, public)
        except (ValueError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError('错误输入没有被拒绝：'+name)
    reject('incorrect_measurement_budget', lambda a:a['metrics'].__setitem__((0, 10), 64))
    reject('fractional_error_count', lambda a:a['metrics'].__setitem__((0, 0), .5))
    reject('mrc_falsely_has_photonic_snr', lambda a:a['metrics'].__setitem__((-1, 7), 1.))
    reject('changed_shared_initial_score', lambda a:a['codebook__trace_scores'].__setitem__(0, 1.))
    reject('missing_additional_query', lambda a:a.__setitem__('codebook__trace_scores', a['codebook__trace_scores'][:-1]))
    def alter_selected(a):
        a['control_code'][0, 0] = (int(a['control_code'][0, 0])+1)%77
    reject('selected_control_different_from_measured_best', alter_selected)
    write_json(output, dict(status='passed_actual_record_fault_checks', at=now(),
        valid_trace_rows=rows, rejected=checks, source_record_sha256=sha256(source),
        protocol_sha256=sha256(root/'protocol.json'), no_new_signals=True, final_confirmation=False,
        source_sha256=source_record(['study_full_baselines/check_confirmation_analysis.py',
                                    'study_full_baselines/analyze_confirmation.py'])))
    print(json.dumps(dict(status='passed_actual_record_fault_checks', rejected=len(checks))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.output.resolve())
