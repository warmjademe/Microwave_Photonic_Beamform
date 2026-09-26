"""把共同训练模型包接入冻结的真实在线计时器；只更换权重根目录。"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 先导入原计时入口，确保NumPy/PyTorch导入前固定CPU线程数。
import study_full_baselines.benchmark_online as benchmark
import numpy as np
from study_full_baselines.common import SOURCE, require_host, source_record, verify_sources, sha256, write_json, now
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.confirmation_freeze import validate as validate_freeze

SOURCES = ['study_full_baselines/'+name for name in [
    'benchmark_bundle_online.py', 'audit_bundle_online.py', 'BUNDLE_TIMING_PROTOCOL.md',
    'benchmark_online.py', 'audit_online_timing.py', 'runtime_bundle.py', 'confirmation_freeze.py']]
PREFLIGHT_METHODS = ['ttd_das', 'mlp', 'complex_response_cnn', 'codebook', 'spsa', 'cnn_warm64']


def run(project, output, preflight, freeze):
    require_host()
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('共同模型包计时不得覆盖已有结果。')
    if bool(preflight) == bool(freeze):
        raise ValueError('必须选择旧案例演练或者已冻结的正式计时。')
    if preflight:
        bundle = project/'dataset_simulation/diagnostics/20260926_runtime_bundle_preflight_run02/bundle'
        frozen = None
    else:
        freeze = freeze.resolve()
        frozen = validate_freeze(freeze)
        if Path(frozen['project']).resolve() != project:
            raise ValueError('冻结来源不属于当前项目。')
        bundle = Path(frozen['runtime_bundle']).resolve()
    data = project/'dataset_simulation/outputs/quality_rank_hybrid_20260925'
    with np.load(data/'public.npz') as f:
        public = {k: f[k].copy() for k in ['pilot_qpsk', 'probe_controls', 'catalog_controls']}
    package = verify_bundle(bundle, public)
    if preflight and package['cohort']['train_environments'] != 864:
        raise ValueError('旧案例演练固定原864环境权重。')
    methods = PREFLIGHT_METHODS if preflight else package['methods']
    sources = source_record(SOURCES)
    binding = dict(schema='common-cohort-online-timing-v1', at=now(), project=str(project),
        runtime_bundle=str(bundle), runtime_manifest_sha256=sha256(bundle/'manifest.json'),
        common_training_cohort=package['cohort'], public_sha256=package['public_sha256'],
        preflight=preflight, final_fair_timing=not preflight, methods=methods,
        freeze_path=str(freeze) if freeze else None, freeze_sha256=sha256(freeze) if freeze else None,
        source_sha256=sources, numerical_algorithm='unchanged benchmark_online and OnlineController',
        input_scope='fixed old-test timing cases; no new holdout signals generated',
        clock_scope='actual preprocessing/inference/search; simulator callback excluded',
        training_repeats_added=0)
    output.mkdir(parents=True)
    write_json(output/'binding.json', binding)
    for name in sources:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    original = benchmark.online.OnlineController
    constructed = []
    def from_bundle(requested_project, name, requested_public):
        if Path(requested_project).resolve() != project or name not in methods:
            raise ValueError('计时器请求了非声明的项目或方法。')
        for key in public:
            np.testing.assert_array_equal(requested_public[key], public[key])
        # 仅切换模型文件根目录；decide、反馈顺序和计时实现沿用原代码。
        model = original(bundle, name, requested_public)
        constructed.append(name)
        return model
    try:
        with patch.object(benchmark.online, 'OnlineController', from_bundle):
            benchmark.run(project, output/'timing', preflight, not preflight, [])
        if constructed != methods:
            raise ValueError('实际加载的方法缺失、重复或顺序不同。')
        verify_sources(sources)
        if verify_bundle(bundle, public) != package:
            raise ValueError('计时过程中模型包改变。')
        if frozen is not None and validate_freeze(freeze) != frozen:
            raise ValueError('计时过程中最终方案改变。')
        write_json(output/'execution.json', dict(status='complete_actual_timing', at=now(),
            constructed_methods=constructed, binding_sha256=sha256(output/'binding.json'),
            timing_summary_sha256=sha256(output/'timing/summary.json'),
            final_fair_timing=not preflight))
        from study_full_baselines.audit_bundle_online import audit
        audit(project, output)
    except BaseException:
        write_json(output/'failure.json', dict(at=now(), traceback=traceback.format_exc(),
            constructed_methods=constructed, binding_sha256=sha256(output/'binding.json')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    run(args.project, args.output, args.preflight, args.freeze)
