"""从冻结结果提取绘图输入。原实验文件只读，输入副本逐文件记录 SHA-256。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SOURCES = {
    'rq1_metrics.npz': 'baseline_results/20260927_uniform64_all13/analysis/environment_metrics.npz',
    'rq1_groups.json': 'baseline_results/20260927_uniform64_all13/analysis/groups.json',
    'rq1_summary.json': 'baseline_results/20260927_uniform64_all13/analysis/summary.json',
    'rq1_complete.json': 'baseline_results/20260927_uniform64_all13/analysis/complete.json',
    'selection_origins.json': 'diagnostics/20260927_uniform64_all13_figures/selection_origins.json',
    'selection_manifest.json': 'diagnostics/20260927_uniform64_all13_figures/manifest.json',
    'constellation.npz': 'diagnostics/20260927_constellation_reconstruction/reconstruction_case.npz',
    'constellation_manifest.json': 'diagnostics/20260927_constellation_reconstruction/manifest.json',
    'ablation_summary.json': 'revision_rq2_af_20261029/full/summary.json',
    'ablation_comparisons.json': 'revision_rq2_af_20261029/full/comparisons.json',
    'ablation_verification.json': 'revision_rq2_af_20261029/full/verification.json',
    'sensitivity_summary.json': 'revision_af_20261029/R/analysis/summary.json',
    'sensitivity_comparisons.json': 'revision_af_20261029/R/analysis/comparisons.json',
    'sensitivity_complete.json': 'revision_af_20261029/R/analysis/complete.json',
    'training_curves.json': 'diagnostics/20260927_paper_rq3_figures_02/manifest.json',
    'array_response.npz': 'revision_scope_20261030/array/array_response.npz',
    'array_summary.json': 'revision_scope_20261030/array/summary.json',
    'array_complete.json': 'revision_scope_20261030/array/complete.json',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(project, output):
    # 若已有输入包，必须显式指定新的目录，避免重写已核验的数据版本。
    if output.exists():
        raise FileExistsError(f'Input directory already exists: {output}')
    source = project / 'dataset_simulation'
    checks = []
    for folder, marker, key, names in [
        ('baseline_results/20260927_uniform64_all13/analysis', 'complete.json', 'file_sha256',
         ['environment_metrics.npz', 'summary.json', 'groups.json']),
        ('revision_rq2_af_20261029/full', 'verification.json', 'files_sha256',
         ['summary.json', 'comparisons.json']),
        ('revision_af_20261029/R/analysis', 'complete.json', 'file_sha256',
         ['summary.json', 'comparisons.json']),
        ('revision_scope_20261030/array', 'complete.json', 'outputs',
         ['array_response.npz', 'summary.json']),
    ]:
        proof = read(source / folder / marker)
        assert proof['status'] in ['passed', 'complete'], folder
        for name in names:
            assert sha(source / folder / name) == proof[key][name], name
            checks.append(str(Path(folder) / name))
    case = source / 'diagnostics/20260927_constellation_reconstruction'
    assert sha(case / 'reconstruction_case.npz') == read(case / 'manifest.json')['files']['reconstruction_case.npz']
    scale = read(source / SOURCES['training_curves.json'])
    for name, digest in scale['sources'].items():
        assert sha(project / name) == digest, name
    # 用 AST 读取已经冻结的阵列常数，不导入仿真器、更不重新执行接收实验。
    config = project / 'source_codes/native_sim/config.py'
    array_summary = read(source / SOURCES['array_summary.json'])
    assert sha(config) == array_summary['source_sha256']['source_codes/native_sim/config.py']
    tree = ast.parse(config.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeConfig')
    fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
    def constant(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return constant(node.left) / constant(node.right)
        raise ValueError('Unsupported configuration expression')
    geometry = {k: constant(fields[k]) for k in ['nx', 'ny', 'pitch_m', 'sample_rate_hz']}
    output.mkdir(parents=True)
    files = {}
    for name, relative in SOURCES.items():
        src = source / relative
        shutil.copyfile(src, output / name)
        files[name] = {'source': 'dataset_simulation/' + relative, 'sha256': sha(src),
                       'bytes': src.stat().st_size, 'transformation': 'byte-identical copy'}
    (output / 'array_geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    files['array_geometry.json'] = {'source': 'source_codes/native_sim/config.py',
        'source_sha256': sha(config), 'sha256': sha(output / 'array_geometry.json'),
        'transformation': 'AST extraction of four frozen array constants'}
    manifest = {'schema_version': 1, 'files': files, 'checked_source_bindings': checks,
        'scope': 'Plotting inputs only; no training, inference, receiver replay or new case selection.',
        'manuscripts': {n: sha(project / 'latex' / n) for n in ['main.tex', 'main_zh.tex']}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(files)} checked inputs in {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=HERE.parents[1])
    p.add_argument('--output', type=Path, default=HERE / 'inputs')
    a = p.parse_args()
    prepare(a.project.resolve(), a.output.resolve())
