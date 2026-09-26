"""固定24个旧测试环境的快速探索，不替代全216环境比较和新留出确认。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_full_baselines.common import *
from study_full_baselines.evaluate_response_variants import bundle_for, controls_for, METRICS
from study_full_baselines.audit_results import statistics


def run(project, output):
    require_host(); base = project/'dataset_simulation'
    data = base/'outputs/quality_rank_hybrid_20260925'; study = base/'baseline_results/20260925_full_baselines'
    manifest = check_data(data); public = public_data(data); chosen = {}
    for row in sorted(manifest['environments'], key=lambda r:r['index']):
        if row['split']!='test':
            continue
        key = (row['factors']['power_bin'], row['factors']['rays'])
        chosen.setdefault(key, row)
    rows = list(chosen.values())
    if len(rows)!=24:
        raise ValueError('六档功率×四档路径数未完整覆盖。')
    bundle = bundle_for(project, study, 'components')
    protocol = dict(scope='old-test exploration; not final independent evidence',
        selection='smallest test index within each of six power bins and four ray counts',
        environment_ids=[r['environment_id'] for r in rows], carriers_ghz=[4,12,20],
        bundle=bundle, frame=5, apd_draws=8, metric_order=METRICS,
        data_manifest_sha256=sha256(data/'manifest.json'), source_sha256=source_record([
            'study_full_baselines/pilot_components.py', 'study_full_baselines/common.py',
            'study_full_baselines/evaluate_response_variants.py', 'study_full_baselines/audit_results.py',
            'our_method_response_control/physics.py', 'our_method_two_stage/control.py',
            'our_method_two_stage/decode_multistart.py']))
    fp = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True); (output/'records').mkdir(exist_ok=True)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:
            raise ValueError('探索性小组协议不同。')
    else:
        write_json(output/'protocol.json', protocol)
        for name in protocol['source_sha256']:
            dest = output/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE/name, dest)
    arrays = {path:np.load(path, mmap_mode='r') for path in bundle['artifact_sha256']}
    values = []; codes = []; started = time.perf_counter()
    for ri, row in enumerate(rows):
        with np.load(data/row['path']/'data.npz') as f:
            raw = f['X']
        for ci, fc in enumerate([4,12,20]):
            path = output/'records'/('%03d.npz'%(ri*3+ci)); marker = path.with_suffix('.json')
            if marker.exists():
                meta = json.loads(marker.read_text())
                if meta['fingerprint']!=fp or sha256(path)!=meta['sha256']:
                    raise ValueError('已有探索记录改变。')
                with np.load(path) as f:
                    result = f['metrics']; code = f['control_code']
            else:
                if path.exists():
                    raise ValueError('未提交探索记录须核查。')
                controls, seconds = controls_for(raw[fc-4], row['index']*17+fc-4,
                    public, bundle, arrays)
                env = json.loads((data/row['path']/'environment.json').read_text())
                engine, payload = frame_engine(env, fc, public['pilot_qpsk'], 5)
                clean = clean_frame_engine(env, fc, public['pilot_qpsk'], payload)
                metrics, code = reception_metrics(engine, clean, controls, payload, row['seed'], fc)
                result = np.column_stack([metrics, np.full(len(controls),16), seconds])
                atomic_npz(path, metrics=result, control_code=code)
                write_json(marker, dict(fingerprint=fp, sha256=sha256(path),
                    environment_id=row['environment_id'], environment_index=row['index'], carrier_ghz=fc))
            values.append(result); codes.append(code)
            state = dict(status='running', completed=len(values), total=72,
                seconds=time.perf_counter()-started, pid=os.getpid(), at=now())
            write_json(output/'progress.json', state)
            if len(values)%6==0:
                print(json.dumps(state), flush=True)
    scores = np.asarray(values); summary = []
    for mi, method in enumerate(bundle['methods']):
        summary.append(dict(method=method['name'], **statistics(scores[:,mi], method['name']),
            mean_decoder_seconds=float(scores[:,mi,-1].mean())))
    verify_sources(protocol['source_sha256'])
    for path, digest in bundle['artifact_sha256'].items():
        if sha256(path)!=digest:
            raise ValueError('模型产物在探索中改变。')
    atomic_npz(output/'all_metrics.npz', metrics=scores, control_code=np.asarray(codes))
    write_json(output/'summary.json', dict(status='complete', records=summary,
        scope=protocol['scope'], environments=24, cases=72, at=now()))
    write_json(output/'progress.json', dict(status='complete', completed=72, total=72, at=now()))
    print(json.dumps(summary), flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try:
        run(a.project, a.output)
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        write_json(a.output/('failure_%d.json'%time.time()), dict(traceback=traceback.format_exc(), at=now()))
        raise
