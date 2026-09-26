"""核对网站JSON、控制CSV、原始波形数组与已提交接收记录的一致性。"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import require_host, sha256, source_record, verify_sources, write_json, now, NativeConfig
from study_full_baselines.export_signal_examples import quality, complex_list, checked_equal


def read(path):
    return json.loads(path.read_text())


def run(records, data):
    require_host()
    dest = data/'integrity_verification.json'
    if dest.exists():
        raise FileExistsError('不覆盖信号展示核查。')
    protocol, summary = read(data/'protocol.json'), read(data/'summary.json')
    if summary['protocol_sha256'] != sha256(data/'protocol.json') or not summary['all_quality_replays_passed']:
        raise ValueError('信号导出来源未核验。')
    if sha256(records/'protocol.json') != protocol['record_protocol_sha256']:
        raise ValueError('接收源协议改变。')
    verify_sources(protocol['source_sha256'])
    cfg = NativeConfig()
    cases, csv_rows, numeric_fields, missing_fields = 0, 0, 0, 0
    if [e['carrier_ghz'] for e in summary['entries']] != protocol['carriers_ghz']:
        raise ValueError('载频覆盖或顺序不同。')
    for entry, committed in zip(summary['entries'], protocol['fixed_environment_record']['carriers']):
        fc = entry['carrier_ghz']
        if fc != committed['carrier_ghz'] or sha256(records/committed['path']) != committed['sha256']:
            raise ValueError('原始接收记录改变。')
        for name, digest in entry['files'].items():
            if sha256(data/name) != digest:
                raise ValueError('展示文件改变。')
        stem = 'carrier_%02d'%fc
        with np.load(records/committed['path']) as f:
            reference = {k:f[k].copy() for k in f.files}
        with np.load(data/(stem+'.npz')) as f:
            arrays = {k:f[k].copy() for k in f.files}
        page = read(data/(stem+'.json'))
        np.testing.assert_array_equal(arrays['public_x_frame0'], reference['public_X'])
        np.testing.assert_array_equal(arrays['control_code'], reference['control_code'][:-1])
        checked_equal(arrays['metrics'], reference['metrics'][:-1, :10], 'photonic arrays')
        checked_equal(arrays['mrc_metrics'], reference['metrics'][-1, :10], 'MRC array')
        if [r['method'].split('/')[-1] for r in page['methods']] != protocol['methods']:
            raise ValueError('网站方法覆盖不同。')
        if page['sent_qpsk'] != complex_list(arrays['payload_qpsk']) or page['tx_iq_normalized'] != complex_list(arrays['tx_iq_normalized']):
            raise ValueError('发送数据的JSON与原始数组不同。')
        if page['time_us'] != arrays['time_us'].tolist():
            raise ValueError('时间轴改变。')
        for mi, case in enumerate(page['methods']):
            target = reference['metrics'][mi, :10]
            if case['quality'] != quality(target):
                raise ValueError('网站指标不是已提交接收结果。')
            numeric_fields += int(np.isfinite(target).sum())
            missing_fields += int(np.isnan(target).sum())
            if case['photonic']:
                if case['method'] != str(arrays['methods'][mi]):
                    raise ValueError('波形数组与页面方法顺序不同。')
                if case['received'] != complex_list(arrays['received_qpsk'][mi, 0]) or case['raw_iq_a'] != complex_list(arrays['received_iq_a'][mi, 0]):
                    raise ValueError('页面没有使用固定抽样0。')
                code = reference['control_code'][mi]
                np.testing.assert_array_equal(case['delay_code'], code[:64])
                np.testing.assert_array_equal(case['attenuation_code'], code[64:])
                np.testing.assert_allclose(case['delay_ps'], code[:64]/cfg.sample_rate_hz*1e12, rtol=1e-12, atol=0)
                np.testing.assert_array_equal(case['attenuation_db'], code[64:]*.5)
            else:
                if case['method'].split('/')[-1] != 'mrc' or not case['own_digital_payload']:
                    raise ValueError('数字MRC被当作光子方法。')
                if case['received'] != complex_list(arrays['mrc_equalized'][0, :, 0]) or case['sent'] != complex_list(arrays['mrc_sent_qpsk'][0, :, 0]):
                    raise ValueError('数字参考符号不同。')
            cases += 1
        with (data/(stem+'_controls.csv')).open() as stream:
            table = list(csv.DictReader(stream))
        if len(table) != 38*64:
            raise ValueError('控制CSV缺项或多项。')
        for index, row in enumerate(table):
            mi, antenna = divmod(index, 64)
            code = reference['control_code'][mi]
            if row['method'] != str(arrays['methods'][mi]) or [int(row[k]) for k in ['antenna_index','array_row','array_column','delay_code','attenuation_code']] != [antenna, antenna//8, antenna%8, int(code[antenna]), int(code[64+antenna])]:
                raise ValueError('CSV方法、阵列位置或档位错误。')
            checked_equal([float(row[k]) for k in ['delay_ps','attenuation_db','total_optical_loss_db']],
                [code[antenna]/cfg.sample_rate_hz*1e12, .5*code[64+antenna], 5+.5*code[64+antenna]], 'control CSV')
            csv_rows += 1
    write_json(dest, dict(status='passed_signal_arrays_json_csv_audit', at=now(), cases=cases,
        numeric_quality_fields_checked=numeric_fields, not_applicable_quality_fields_checked=missing_fields,
        control_csv_rows_checked=csv_rows, final_confirmation=protocol['final_confirmation'],
        protocol_sha256=sha256(data/'protocol.json'), summary_sha256=sha256(data/'summary.json'),
        source_sha256=source_record(['study_full_baselines/audit_confirmation_signals.py']),
        scope=protocol['scope']))
    print(json.dumps(dict(status='passed_signal_arrays_json_csv_audit', cases=cases,
        numeric_quality_fields=numeric_fields, control_csv_rows=csv_rows)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    run(args.records.resolve(), args.data.resolve())
