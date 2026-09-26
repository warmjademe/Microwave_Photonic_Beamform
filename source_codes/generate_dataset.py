"""分阶段采集：先产生训练/测试传播环境，再由已选接收模型生成观测与标签。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from baseline_common.channel import sampling_plan, make_environment, response, serialize_environment
from baseline_common.config import Config, qpsk, rng_for
from baseline_common.controls import probe_codebook
from baseline_common.data import SCHEMA, dump_json

SOURCE = Path(__file__).resolve().parent


def source_hashes():
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob('*.py')) if '__pycache__' not in p.parts}


def plan_environments(train_environments=800, test_environments=200, seed=20260923,
                      sampling='marginal_balanced'):
    """只生成划分清单，供预审覆盖和种子；此函数没有文件或信道计算。"""
    if min(train_environments, test_environments) < 1:
        raise ValueError('首批须同时包含训练与测试环境。')
    rows, designs, seeds = [], {}, set()
    for si, (split, count) in enumerate((('train', train_environments), ('test', test_environments))):
        factors, designs[split] = sampling_plan(count, seed+100*si, sampling)
        repeats = {}
        for index, factor in enumerate(factors):
            env_seed = int(np.random.SeedSequence([seed, 1000+si, index]).generate_state(1)[0])
            if env_seed in seeds:
                raise ValueError('环境随机种子碰撞。')
            seeds.add(env_seed)
            row = dict(split=split, index=index, seed=env_seed,
                       path=str(Path(split)/f'environment_{index:05d}'),
                       environment_id=f'{split}-{env_seed:010d}', factors=factor)
            if sampling == 'joint_stratified':
                stratum = 'L%d_T%d_A%d_P%d' % tuple(int(factor[k]) for k in
                                                   ('rays', 'max_delay_ns', 'angular_std_deg', 'power_bin'))
                row.update(joint_stratum_id=stratum,
                           stratum_repeat_index=repeats.get(stratum, 0))
                repeats[stratum] = row['stratum_repeat_index']+1
            rows.append(row)
        designs[split]['environment_seed_rule'] = f'SeedSequence([master_seed, {1000+si}, index])'
        if sampling == 'joint_stratified':
            designs[split]['repeat_index_rule'] = 'zero-based occurrence within the shuffled split plan'
    return rows, designs


def make_channels(root, train_environments=800, test_environments=200, seed=20260923,
                  require_osd_equivalence=True, sampling='marginal_balanced'):
    # 先检查完整方案；非法联合分层数量不会留下半成品输出目录。
    planned_rows, designs = plan_environments(train_environments, test_environments, seed, sampling)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    cfg = Config()
    controls, angles = probe_codebook(cfg)
    pilots = qpsk(rng_for(seed, 2), (cfg.tones, cfg.pilot_symbols)).astype(np.complex64)
    np.savez(root/'public.npz', pilot_qpsk=pilots, probe_controls=controls.astype(np.float32),
             probe_angles_deg=angles, positions_m=cfg.positions, offsets_hz=cfg.offsets_hz)
    carriers = list(range(4, 21))
    manifest = dict(schema=SCHEMA, status='generating_channels', master_seed=seed,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    signal_config=cfg.to_dict(), carriers_ghz=carriers,
                    splits={'train': train_environments, 'test': test_environments},
                    sampling_method=sampling, sampling_design_by_split=designs,
                    validation_split=False, split_unit='independent propagation environment, before frequency expansion',
                    signal_format='64-bin OFDM-QPSK frequency-domain blocks; CP 300 ns; block-static channel',
                    pilot_meaning='128 OFDM blocks per setting, 128 QPSK symbols per bin',
                    model_selection=('osd_native_equivalent_pending_calibration' if require_osd_equivalence
                                     else 'exploratory_test_fixture'),
                    osd_equivalence_required=bool(require_osd_equivalence), osd_end_to_end_calibrated=False,
                    independent_interferer=False, temperature_drift=False, device_execution_error=False,
                    time_variation=False, environments=[], python=platform.python_version(), numpy=np.__version__,
                    source_sha256_at_channel_generation=source_hashes())
    dump_json(root/'manifest.json', manifest)
    start = time.perf_counter()
    try:
        for row in planned_rows:
            split, index = row['split'], row['index']
            count = manifest['splits'][split]
            e = make_environment(row['seed'], row['factors'])
            folder = root/row['path']
            folder.mkdir(parents=True)
            dump_json(folder/'environment_truth.json', serialize_environment(e))
            h = np.stack([response(e, cfg, fc*1e9) for fc in carriers]).astype(np.complex64)
            np.savez(folder/'truth_not_ai_input.npz', channel=h,
                     power_dbm=np.float64(e['power_dbm']),
                     input_noise_psd_w_hz=np.float64(1.380649e-23*cfg.temperature_k))
            manifest['environments'].append(row)
            if (index+1) % 50 == 0 or index+1 == count:
                print(json.dumps({'stage': 'channels', 'split': split, 'completed': index+1, 'total': count}), flush=True)
                dump_json(root/'manifest.json', manifest)
        manifest.update(status='channels_complete', total_environments=len(planned_rows),
                        total_samples=len(planned_rows)*len(carriers),
                        samples_by_split={k: v*len(carriers) for k, v in manifest['splits'].items()},
                        channel_generation_seconds=time.perf_counter()-start,
                        source_sha256_after_channel_generation=source_hashes())
    except Exception as exc:
        manifest.update(status='failed_channels', failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        dump_json(root/'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-environments', type=int, default=800)
    p.add_argument('--test-environments', type=int, default=200)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--sampling', choices=['marginal_balanced', 'joint_stratified'],
                   default='marginal_balanced',
                   help='默认保留原边际均衡；joint_stratified要求训练和测试各为216倍数。')
    p.add_argument('--channels-only', action='store_true', required=True,
                   help='此入口只生成独立于光子模型的传播真值，不声称已有监督标签。')
    args = p.parse_args()
    make_channels(args.output, args.train_environments, args.test_environments, args.seed,
                  sampling=args.sampling)


if __name__ == '__main__':
    main()
