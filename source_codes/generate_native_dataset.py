"""生成有监督数据：独立环境划分→17载频→原生离散激光→合光/APD→控制标签。

prepare固定计划和源码；run支持独立分片、原子写入与逐文件校验续跑；status核数。
本批接受已披露的仿真偏差，不改写历史OSD验收结论。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import time
import traceback
import numpy as np
from baseline_common.channel import make_environment, serialize_environment
from baseline_common.config import qpsk, rng_for
from baseline_common.controls import probe_codebook, physical_units
from generate_dataset import plan_environments
from native_sim.config import NativeConfig
from native_sim.laser import build, load_profile, simulate
from native_sim.waveforms import make_drive
from native_sim.optical_cache import make_cache
from native_sim.control_engine import NativeControlEngine, optimize_teacher
from native_sim.evaluation import evaluate_record

SOURCE = Path(__file__).resolve().parent
CORE = ['generate_native_dataset.py', 'generate_dataset.py',
        'baseline_common/channel.py', 'baseline_common/config.py', 'baseline_common/controls.py',
        'native_sim/config.py', 'native_sim/laser.py', 'native_sim/native_laser.cpp',
        'native_sim/laser_profile_019.json', 'native_sim/waveforms.py',
        'native_sim/optical_cache.py', 'native_sim/control_engine.py', 'native_sim/evaluation.py']


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+'\n').encode('utf-8')


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    temp.write_bytes(json_bytes(value)); temp.replace(path)


def hashes(): return {name: digest(SOURCE/name) for name in CORE}


def prepare(root, train=17280, test=4320, seed=20260925):
    rows, designs = plan_environments(train, test, seed, 'joint_stratified')
    cfg = NativeConfig(); controls, angles = probe_codebook(cfg)
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    public = root/'public.npz'
    np.savez(public, pilot_qpsk=qpsk(rng_for(seed, 2), (31, 2)),
             probe_controls=controls, probe_angles_deg=angles,
             positions_m=cfg.positions, offsets_hz=cfg.offsets_hz)
    manifest = dict(schema='mwp-native-learning-v1', status='preparing',
        created_at_utc=now(), master_seed=seed, signal_config=cfg.to_dict(),
        carriers_ghz=list(range(4, 21)), splits=dict(train=train, test=test),
        samples_by_split=dict(train=train*17, test=test*17),
        total_environments=len(rows), total_samples=len(rows)*17,
        validation_split=False, split_unit='propagation environment before carrier expansion',
        sampling_method='joint_stratified', sampling_design_by_split=designs,
        model_selection='native_fixed_step_51g_v1', osd_end_to_end_calibrated=False,
        known_deviations_accepted=True,
        acceptance=dict(date='2026-09-23', purpose='learning control relationships within frozen simulation',
            scope='proceed with data despite disclosed long-frame device approximation'),
        independent_interferer=False, temperature_drift=False, device_execution_error=False,
        time_variation=False, optical_combining='64 optical routes, one final APD',
        public_file_sha256=digest(public), core_source_sha256=hashes(),
        teacher=dict(starts=2, sweeps=1, uses_payload_targets=False,
            privileged_branch_cache=True, globally_optimal=False),
        noise_model='thermal antenna noise before laser; stationary Gaussian APD using mean optical power',
        evaluation='two pilots set equalizer; independent payload gives EVM/BER/effective SNR',
        snr_definition='-10 log10(payload NMSE), includes distortion; not thermal SNR alone',
        waveform='31 QPSK OFDM tones; two pilots and one payload; periodic ideal rectangular band limit',
        random_streams=dict(public=[seed, 2], payload=['environment_seed', 3],
            antenna=['environment_seed', 'carrier_ghz', 4],
            probes=['environment_seed', 'carrier_ghz', 5, 'probe_index'],
            teacher=['environment_seed', 'carrier_ghz', 6],
            evaluation=['environment_seed', 'carrier_ghz', 8]),
        environments=rows)
    write_json(root/'manifest.json', manifest)
    for row in rows:
        folder = root/row['path']; folder.mkdir(parents=True)
        write_json(folder/'environment_truth.json',
                   serialize_environment(make_environment(row['seed'], row['factors'])))
        row['truth_sha256'] = digest(folder/'environment_truth.json')
    fingerprint = dict(master_seed=seed, signal_config=manifest['signal_config'],
        public_sha256=manifest['public_file_sha256'], core_source_sha256=manifest['core_source_sha256'],
        plan_sha256=hashlib.sha256(json_bytes(rows)).hexdigest(), carriers=manifest['carriers_ghz'])
    manifest.update(status='prepared', generation_fingerprint=hashlib.sha256(json_bytes(fingerprint)).hexdigest())
    for name in CORE:
        destination = root/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE/name, destination)
    write_json(root/'manifest.json', manifest)
    write_json(root/'fingerprint_components.json', fingerprint)
    (root/'runs').mkdir()
    print(json.dumps(dict(status='prepared', environments=len(rows), samples=len(rows)*17)), flush=True)
    return manifest


def check_sources(root, manifest):
    if hashes() != manifest['core_source_sha256']:
        raise ValueError('生成源码与冻结快照不同；不可将不同版本混入同一数据集。')
    if digest(root/'public.npz') != manifest['public_file_sha256']:
        raise ValueError('公共探测/导频文件发生变化。')


def validate_arrays(arrays, fingerprint):
    shapes = {'combined_iq_a':(16,31,2), 'pilot_iq_a':(16,248), 'quality':(16,),
        'noise_symbol_var_a2':(16,31), 'probe_apd_dc_a':(16,), 'control':(128,),
        'control_code':(128,), 'delay_ps':(64,), 'attenuation_db':(64,),
        'branch_band_w':(64,255), 'branch_dc_w':(64,), 'channel':(64,31),
        'payload_qpsk':(31,), 'teacher_iq_a':(512,), 'best_probe_iq_a':(512,)}
    for name, shape in shapes.items():
        if np.shape(arrays[name]) != shape: raise ValueError('形状错误：'+name)
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.dtype.kind in 'biufc' and not np.all(np.isfinite(a)):
            raise ValueError('非有限数值：'+name)
    if str(np.asarray(arrays['generation_fingerprint']).item()) != fingerprint:
        raise ValueError('记录指纹不同。')
    levels = np.r_[np.full(64,76), np.full(64,24)]
    code = arrays['control_code']
    if np.any(code < 0) or np.any(code > levels): raise ValueError('控制码超出范围。')
    if not np.allclose(arrays['control'], code/levels, rtol=0, atol=1e-7):
        raise ValueError('标签与实际控制码不同。')
    if float(arrays['objective'])+1e-12 < float(arrays['initial_best_objective']):
        raise ValueError('教师自身目标比其起点更差。')
    if not 0 <= int(arrays['bit_errors']) <= 62: raise ValueError('误码计数无效。')


def one_record(environment, carrier, public, cfg, profile, fingerprint):
    seed = environment['seed']
    payload = qpsk(rng_for(seed, 3), (31,))
    drive, details = make_drive(environment, carrier*1e9, public['pilot_qpsk'], payload,
                                rng_for(seed, carrier, 4), cfg)
    fields, current_bounds = simulate(drive, cfg.sample_rate_hz, profile)
    del drive
    centers = np.array([r['optical_frequency_hz'] for r in profile])
    band, dc = make_cache(fields, centers, carrier*1e9, cfg)
    del fields
    engine = NativeControlEngine(cfg, band, dc, carrier*1e9, public['pilot_qpsk'])
    controls = public['probe_controls']
    measured = [engine.measure_detailed(u, rng_for(seed, carrier, 5, i)) for i,u in enumerate(controls)]
    control, teacher = optimize_teacher(engine, controls, rng_for(seed, carrier, 6), starts=2, sweeps=1)
    # 初始对照由公开带噪反馈选出，不读payload。用同一APD基础随机数作配对评价。
    best_probe_index = int(np.argmax([m['score'] for m in measured]))
    before = evaluate_record(engine, controls[best_probe_index], payload, rng_for(seed, carrier, 8))
    after = evaluate_record(engine, control, payload, rng_for(seed, carrier, 8))
    delay, attenuation = physical_units(control, cfg)
    result = dict(combined_iq_a=np.stack([m['symbols'] for m in measured]).astype(np.complex64),
        pilot_iq_a=np.stack([m['pilot_iq'] for m in measured]).astype(np.complex64),
        quality=np.array([m['score'] for m in measured], np.float32),
        noise_symbol_var_a2=np.stack([m['noise_symbol_var'] for m in measured]),
        probe_apd_dc_a=np.array([m['apd_dc_a'] for m in measured]),
        control=control.astype(np.float32), control_code=engine.codes(control),
        delay_ps=delay, attenuation_db=attenuation,
        branch_band_w=band, branch_dc_w=dc,
        channel=details['channel_coefficients'][:,np.arange(-15,16)*8+127].astype(np.complex64),
        power_dbm=np.float64(environment['power_dbm']), payload_qpsk=payload,
        teacher_iq_a=after['iq_a'].astype(np.complex64),
        best_probe_iq_a=before['iq_a'].astype(np.complex64),
        teacher_received_qpsk=after['received_qpsk'].astype(np.complex64),
        best_probe_received_qpsk=before['received_qpsk'].astype(np.complex64),
        best_probe_index=np.int16(best_probe_index), current_bounds_a=current_bounds,
        generation_fingerprint=np.array(fingerprint), carrier_ghz=np.int16(carrier))
    for key in ('objective', 'initial_best_objective', 'objective_evaluations'):
        result[key] = np.array(teacher[key])
    for key in ('evm_percent', 'bit_errors', 'bits_tested', 'snr_db', 'payload_nmse', 'ber'):
        result[key] = np.array(after[key]); result['initial_'+key] = np.array(before[key])
    validate_arrays(result, fingerprint)
    return result


def collect_environment(root_text, row, fingerprint):
    root = Path(root_text); folder = root/row['path']; start = time.perf_counter()
    with open(folder/'.generation.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if digest(folder/'environment_truth.json') != row['truth_sha256']:
            raise ValueError('传播环境真值被改变。')
        environment = json.loads((folder/'environment_truth.json').read_text())
        with np.load(root/'public.npz', allow_pickle=False) as f:
            public = {k:f[k] for k in f.files}
        cfg = NativeConfig(); profile = load_profile(); records = []
        for carrier in range(4,21):
            path = folder/('carrier_%02d.npz' % carrier)
            marker = path.with_suffix('.json')
            if path.exists() and marker.exists():
                old = json.loads(marker.read_text())
                if old['generation_fingerprint'] != fingerprint or old['sha256'] != digest(path):
                    raise ValueError('续跑记录校验失败：'+str(path))
                records.append(old); continue
            if path.exists():
                # 原子NPZ已写完但进程来不及写marker：只在完整校验后补marker。
                with np.load(path, allow_pickle=False) as f:
                    arrays = {k:f[k] for k in f.files}
                validate_arrays(arrays, fingerprint)
            else:
                if shutil.disk_usage(root).free < 10*1024**3:
                    raise OSError('剩余磁盘不足10GiB，保留现场停止。')
                arrays = one_record(environment, carrier, public, cfg, profile, fingerprint)
                temp = path.with_name(path.name+'.%d.tmp' % os.getpid())
                with temp.open('wb') as stream: np.savez_compressed(stream, **arrays)
                temp.replace(path)
            info = dict(carrier_ghz=carrier, sha256=digest(path), bytes=path.stat().st_size,
                generation_fingerprint=fingerprint, completed_at_utc=now(),
                evm_percent=float(arrays['evm_percent']), initial_evm_percent=float(arrays['initial_evm_percent']),
                bit_errors=int(arrays['bit_errors']), initial_bit_errors=int(arrays['initial_bit_errors']))
            write_json(marker, info); records.append(info)
        summary = dict(environment_id=row['environment_id'], split=row['split'],
            generation_fingerprint=fingerprint, carrier_count=len(records), records=records,
            seconds=time.perf_counter()-start, completed_at_utc=now())
        write_json(folder/'complete.json', summary)
        return dict(environment_id=row['environment_id'], split=row['split'], seconds=summary['seconds'],
                    bytes=sum(r['bytes'] for r in records))


def run(root, workers=4, shard=0, shards=1, smoke=False):
    root = Path(root).resolve(); manifest = json.loads((root/'manifest.json').read_text())
    if not (1<=workers<=32 and 0<=shard<shards): raise ValueError('worker/分片参数无效。')
    check_sources(root, manifest); build()
    selected = [r for i,r in enumerate(manifest['environments']) if i%shards==shard]
    if smoke:
        selected = [next(r for r in selected if r['split']==s) for s in ('train','test')]
    fingerprint = manifest['generation_fingerprint']; started = time.perf_counter()
    run_id = '%s_%s_p%d_s%dof%d' % (time.strftime('%Y%m%d_%H%M%S'), socket.gethostname(), os.getpid(), shard, shards)
    progress_path = root/'runs'/(run_id+'.json')
    progress = dict(run_id=run_id, status='running', started_at_utc=now(),
        generation_fingerprint=fingerprint, selected_environments=len(selected),
        workers=workers, shard=shard, shards=shards, smoke=smoke, completed=0,
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        pid=os.getpid(), completed_bytes=0)
    write_json(progress_path, progress)
    # 所有分片只写自己的run文件。manifest最终状态由status统一核数后更新。
    errors = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = iter(selected); pending = {}
            def submit_next():
                row = next(iterator, None)
                if row is not None:
                    pending[executor.submit(collect_environment, str(root), row, fingerprint)] = row
            for _ in range(workers): submit_next()
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    try: result = future.result()
                    except Exception as exc:
                        error = dict(environment_id=row['environment_id'], time=now(),
                                     type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                        errors.append(error)
                        write_json(root/'runs'/(run_id+'_failure_'+row['environment_id']+'.json'), error)
                    else:
                        progress['completed'] += 1; progress['completed_bytes'] += result['bytes']
                    if not errors: submit_next()
                progress.update(updated_at_utc=now(), elapsed_seconds=time.perf_counter()-started,
                                failed=len(errors), status='stopping_after_failure' if errors else 'running')
                write_json(progress_path, progress)
                print(json.dumps({k:progress[k] for k in ('run_id','status','completed','selected_environments',
                    'elapsed_seconds','failed')}), flush=True)
        progress.update(status='failed' if errors else 'complete', finished_at_utc=now())
    except BaseException:
        progress.update(status='interrupted', finished_at_utc=now()); raise
    finally:
        write_json(progress_path, progress)
    if errors: raise RuntimeError('分片生成失败，错误已保存；不删除或替换失败环境。')
    return progress


def status(root, finalize=False):
    root = Path(root).resolve(); manifest = json.loads((root/'manifest.json').read_text())
    counts = dict(train=0,test=0); partial = dict(train=0,test=0); byte_count = 0
    for row in manifest['environments']:
        folder = root/row['path']; complete = folder/'complete.json'
        if complete.exists():
            value = json.loads(complete.read_text())
            if (value['generation_fingerprint']!=manifest['generation_fingerprint'] or
                    value['carrier_count']!=17 or len(value['records'])!=17 or
                    sorted(r['carrier_ghz'] for r in value['records'])!=list(range(4,21))):
                raise ValueError('环境完成标记无效：'+row['path'])
            for record in value['records']:
                path = folder/('carrier_%02d.npz' % record['carrier_ghz'])
                if not path.exists() or path.stat().st_size!=record['bytes']:
                    raise ValueError('完成标记存在但记录缺失/大小不同：'+str(path))
                if finalize and digest(path)!=record['sha256']:
                    raise ValueError('最终SHA校验失败：'+str(path))
                byte_count += record['bytes']
            counts[row['split']] += 1
        else:
            partial[row['split']] += len(list(folder.glob('carrier_*.json')))
    finished = counts==manifest['splits']
    if finalize and not finished: raise ValueError('环境尚未齐全，不能完成数据集。')
    # 普通查询不能撤销已经完成的最终核验；只有全量SHA核验能首次置complete。
    previously_verified = manifest.get('status')=='complete' and manifest.get('last_progress',{}).get('final_sha_verification',False)
    verified = bool(finished and (finalize or previously_verified))
    state = 'complete' if verified else 'generating'
    result = dict(status=state, updated_at_utc=now(), completed_environments=counts,
        planned_environments=manifest['splits'], completed_samples={s:counts[s]*17+partial[s] for s in counts},
        planned_samples=manifest['samples_by_split'], completed_bytes=byte_count,
        final_sha_verification=verified, generation_fingerprint=manifest['generation_fingerprint'])
    write_json(root/'progress.json', result)
    manifest.update(status=state, last_progress=result)
    write_json(root/'manifest.json', manifest)
    print(json.dumps(result,ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','status'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-environments',type=int,default=17280)
    p.add_argument('--test-environments',type=int,default=4320)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--smoke',action='store_true');p.add_argument('--finalize',action='store_true')
    a = p.parse_args()
    if a.command=='prepare': prepare(a.output,a.train_environments,a.test_environments,a.seed)
    elif a.command=='run': run(a.output,a.workers,a.shard,a.shards,a.smoke)
    else: status(a.output,a.finalize)


if __name__=='__main__': main()
