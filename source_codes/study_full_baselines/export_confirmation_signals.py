"""固定确认环境索引0，回放全部方法的实际控制并导出波形与星座。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import SOURCE, require_host, sha256, source_record, verify_sources, write_json, now
from study_full_baselines.confirmation_freeze import validate as validate_freeze
from study_full_baselines.confirmation_execute import rehearsal_identity
from study_full_baselines.confirmation_batch import committed_environment
from study_full_baselines.confirmation_receiver import scoring_engines
from study_full_baselines.runtime_bundle import verify as verify_bundle
from study_full_baselines.analyze_confirmation import check_values
import study_full_baselines.export_signal_examples as original

SOURCES = original.SOURCES+['study_full_baselines/'+name for name in [
    'export_confirmation_signals.py', 'CONFIRMATION_SIGNAL_PROTOCOL.md',
    'confirmation_freeze.py', 'confirmation_execute.py', 'confirmation_batch.py',
    'confirmation_receiver.py', 'runtime_bundle.py', 'analyze_confirmation.py']]


def read(path):
    return json.loads(path.read_text())


def run(project, folder, analysis, output, freeze=None, rehearse=False):
    require_host()
    project, folder, analysis, output = [p.resolve() for p in [project, folder, analysis, output]]
    if output.exists():
        raise FileExistsError('固定信号导出不覆盖已有记录。')
    if bool(freeze) == bool(rehearse):
        raise ValueError('只选择旧案例演练或者正式冻结案例。')
    if freeze is not None:
        freeze = freeze.resolve()
    identity = rehearsal_identity(project) if rehearse else validate_freeze(freeze)
    if read(folder/'protocol.json') != identity or Path(identity['project']).resolve() != project:
        raise ValueError('接收记录与冻结项目不同。')
    verified = read(analysis/'complete.json')
    expected = 'complete_old_record_rehearsal' if rehearse else 'complete_frozen_confirmation_statistics'
    if (verified['status'] != expected or verified['input_protocol_sha256'] != sha256(folder/'protocol.json')
            or verified['final_confirmation'] != (not rehearse)
            or verified['freeze_sha256'] != (sha256(freeze) if freeze else None)):
        raise ValueError('完整接收核查与统计尚未通过，或来源不同。')
    for name, digest in verified['file_sha256'].items():
        if sha256(analysis/name) != digest:
            raise ValueError('完整分析产物改变。')
    verify_sources(verified['source_sha256'])
    audit = read(analysis/'audit.json')
    if (audit['complete_sha256'] != sha256(folder/'complete.json')
            or audit['records_sha256'] != sha256(folder/'records.json')
            or read(folder/'progress.json')['status'] != 'complete'):
        raise ValueError('接收源已经改变或未完成。')
    row = next(r for r in identity['rows'] if r['index'] == 0)
    carriers = identity['carriers']
    if not rehearse and carriers != list(range(4, 21)):
        raise ValueError('正式固定案例必须导出全部17载频。')
    record = committed_environment(folder, row, identity)
    if record is None:
        raise ValueError('固定案例尚未完整提交。')
    env = read(folder/'records/environment_00000/environment.json')
    if env['seed'] != row['seed']:
        raise ValueError('固定案例种子不符。')
    bundle = Path(identity['runtime_bundle'])
    package = verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != identity['runtime_manifest_sha256']:
        raise ValueError('模型包身份不同。')
    with np.load(bundle/'public.npz') as f:
        public = {k:f[k].copy() for k in f.files}
    names = identity['methods']
    # 仅为原导出函数的载频索引接口留槽；缺失频点保持NaN，绝不导出为成绩。
    values = np.full((17, len(names), len(identity['metric_order'])), np.nan)
    codes = np.full((17, len(names), 128), -1, dtype=np.int16)
    digital = np.full(17, np.nan)
    inputs = {}
    for item in record['carriers']:
        if audit['raw_file_sha256'].get(item['path']) != item['sha256']:
            raise ValueError('案例不属于已经完整审计的结果。')
        fc = item['carrier_ghz']
        with np.load(folder/item['path']) as f:
            arrays = {k:f[k].copy() for k in f.files}
        check_values(arrays, fc, identity, public)
        values[fc-4], codes[fc-4] = arrays['metrics'], arrays['control_code']
        digital[fc-4], inputs[fc] = arrays['mrc_physical_reference_snr_db'], arrays['public_X']
    records = {'confirmation':dict(methods=names, values=values, codes=codes,
                                  digital_snr=digital, metric_order=identity['metric_order'])}
    sources = {**identity['source_sha256'], **source_record(SOURCES)}
    scope = 'old_runtime_rehearsal_signals' if rehearse else 'fresh_confirmation_fixed_example'
    protocol = dict(schema='confirmation-fixed-signal-v1', scope=scope, at=now(),
        environment_index=0, environment_id=row['environment_id'], seed=row['seed'],
        carriers_ghz=carriers, methods=names, frame=5, metric_draws=8, display_draw=0,
        record_protocol_sha256=sha256(folder/'protocol.json'), analysis_complete_sha256=sha256(analysis/'complete.json'),
        fixed_environment_record=record, runtime_manifest_sha256=identity['runtime_manifest_sha256'],
        training_environments=package['cohort']['train_environments'], source_sha256=sources,
        freeze_sha256=sha256(freeze) if freeze else None, final_confirmation=not rehearse,
        backend=identity['backend'], new_environment_generation=False,
        control_policy='replay committed codes; no optimization or case selection',
        quality_relative_tolerance=1e-12, quality_absolute_tolerance=0)
    output.mkdir(parents=True)
    write_json(output/'protocol.json', protocol)
    write_json(output/'environment.json', env)
    for name in sources:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, dest)
    entries = []
    for fc in carriers:
        engine, clean_engine, payload = scoring_engines(env, fc, public, identity['backend'])
        def frame_for_export(e, carrier, pilots, number):
            if e != env or carrier != fc or number != 5:
                raise ValueError('导出器试图更换固定输入或评分帧。')
            np.testing.assert_array_equal(pilots, public['pilot_qpsk'])
            return engine, payload
        def clean_for_export(e, carrier, pilots, sent):
            if e != env or carrier != fc:
                raise ValueError('无噪参考不是同一环境与载频。')
            np.testing.assert_array_equal(pilots, public['pilot_qpsk'])
            np.testing.assert_array_equal(sent, payload)
            return clean_engine
        # 原I/Q回放器保持不变，只接入正式使用的GPU FFT接收引擎。
        with patch.object(original, 'frame_engine', frame_for_export), patch.object(original, 'clean_frame_engine', clean_for_export):
            entry = original.export_carrier(output, fc, env, inputs[fc], public, records)
        path = output/('carrier_%02d.json'%fc)
        page = read(path)
        page.update(source_scope=scope, final_confirmation=not rehearse,
                    environment_id=row['environment_id'], training_environments=package['cohort']['train_environments'])
        write_json(path, page)
        entry['files'] = {name:sha256(output/name) for name in entry['files']}
        entries.append(entry)
        write_json(output/'progress.json', dict(status='running', completed_carriers=len(entries),
            total_carriers=len(carriers), scope=scope, pid=os.getpid(), at=now()))
        print(json.dumps(dict(carrier_ghz=fc, verified_quality_values=entry['verified_quality_values'])), flush=True)
    if committed_environment(folder, row, identity) != record:
        raise ValueError('导出期间原始控制或评分改变。')
    verify_sources(sources)
    if sha256(bundle/'manifest.json') != identity['runtime_manifest_sha256']:
        raise ValueError('导出期间模型包改变。')
    write_json(output/'summary.json', dict(status='passed_preflight' if rehearse else 'complete',
        scope=scope, final_confirmation=not rehearse, entries=entries, all_quality_replays_passed=True,
        protocol_sha256=sha256(output/'protocol.json'), at=now()))
    write_json(output/'progress.json', dict(status='complete', scope=scope, carriers=len(entries), at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'input', 'analysis', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', type=Path)
    mode.add_argument('--rehearse-old', action='store_true')
    args = parser.parse_args()
    run(args.project, args.input, args.analysis, args.output, args.freeze, args.rehearse_old)
