"""三个冻结训练种子的 A/F 结果：完整原始覆盖、配对区间及种子间变异。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from study_full_baselines.common import require_host, sha256, now, write_json, atomic_npz
from study_full_baselines.paired_statistics import sufficient, metrics, FIELDS, bootstrap, contrast, bh_adjust


def read(path):
    return json.loads(Path(path).read_text())


def run(project):
    require_host()
    root = project / "dataset_simulation/revision_af_20261029/S"
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = project / "dataset_simulation/revision_rq2_af_20261029/full"
        verification = read(base / "verification.json")
        assert verification["status"] == "passed"
        path = base / "environment_metrics.npz"
        assert sha256(path) == verification["files_sha256"][path.name]
        protocol = read(base / "protocol.json")
        assert sha256(base / "protocol.json") == verification["files_sha256"]["protocol.json"]
        sources = {str(path): sha256(path), str(base / "verification.json"): sha256(base / "verification.json")}
        with np.load(path, allow_pickle=False) as a:
            assert a["methods"].tolist() == ["covariance_response", "complex_response_cnn",
                "covariance_warm64", "cnn_warm64", "initial_select64"]
            reference = a["values"][:, :, [0,2,4]].copy()
            seed0 = a["values"][:, :, [1,3]].copy()
            ids = a["environment_ids"].copy()
        values, coverage = [seed0], []
        for seed in (1,2):
            folder = root / f"seed_{seed}"
            done, p = read(folder / "complete.json"), read(folder / "protocol.json")
            assert done["status"] == "complete" and done["seed"] == seed and not done["preflight"]
            assert done["environments"] == 864 and done["carriers"] == 17
            assert not done["joint_correction"] and p["rows"] == protocol["rows"]
            for name, expected in done["file_sha256"].items():
                assert sha256(folder / name) == expected
                sources[str(folder / name)] = expected
            records = read(folder / "records.json")
            assert [r["environment_id"] for r in records] == ids.tolist()
            checked = 0
            for r in records:
                assert [e["carrier_ghz"] for e in r["carriers"]] == list(range(4,21))
                for entry in r["carriers"]:
                    assert entry["protocol_sha256"] == done["protocol_sha256"]
                    assert sha256(folder / entry["path"]) == entry["sha256"]
                    assert entry["checks"]["feedback_audit_passed"] and not entry["checks"]["joint_correction_used"]
                    checked += 1
            assert checked == 14688
            with np.load(folder / "environment_metrics.npz", allow_pickle=False) as a:
                np.testing.assert_array_equal(a["environment_ids"], ids)
                assert a["methods"].tolist() == ["A_direct16", "A_feedback64"]
                values.append(a["values"].copy())
            coverage.append(dict(seed=seed,raw_records_verified=checked,
                reused_conditions=done["reused_conditions"],newly_evaluated_conditions=done["newly_evaluated_conditions"]))
        values = np.asarray(values)
        assert values.shape == (3,864,17,2,13)
        names = ["A_direct16", "A_feedback64", "conventional_direct16", "conventional_feedback64", "fixed_scan64"]
        points, per_seed, comparisons = [], [], []
        specs = [("A_vs_conventional", [1,0,-1,0,0]), ("AF_vs_conventional_F", [0,1,0,-1,0]),
                 ("AF_vs_fixed_scan", [0,1,0,0,-1]), ("F_with_A", [-1,1,0,0,0])]
        for seed in (0,1,2):
            combined = np.concatenate([values[seed], reference], axis=2)
            s = sufficient(combined)
            point = metrics(s.sum(0))
            points.append(point[:2])
            draws = bootstrap(s, 10000, 20261029, strata=[r["joint_stratum_id"] for r in protocol["rows"]])
            for i, name in enumerate(names):
                per_seed.append(dict(seed=seed, method=name,
                    **{f:float(point[i,j]) for j,f in enumerate(FIELDS)},
                    ci95={f:np.quantile(draws[:,i,j],[.025,.975]).tolist() for j,f in enumerate(FIELDS[:5])}))
            for label, coefficients in specs:
                for field in FIELDS[:5]:
                    comparisons.append(dict(seed=seed, id=label, coefficients=coefficients,
                        **contrast(s,draws,coefficients,field)))
        for field in FIELDS[:5]:
            group = [r for r in comparisons if r["metric"] == field]
            assert len(group) == 12
            # contrast 将精确符号检验的字段直接合并在返回字典中。
            adjusted = bh_adjust([r["p_value"] for r in group])
            for r, q in zip(group, adjusted):
                r["q_value"] = float(q)
        points = np.asarray(points)
        summary = [dict(method=name, seeds=[0,1,2], metrics={
            field: dict(mean=float(points[:,i,j].mean()), sample_sd=float(points[:,i,j].std(ddof=1)),
                        values=points[:,i,j].tolist())
            for j,field in enumerate(FIELDS)}) for i,name in enumerate(names[:2])]
        write_json(output / "per_seed.json", per_seed)
        write_json(output / "comparisons.json", comparisons)
        write_json(output / "mean_and_seed_sd.json", summary)
        write_json(output / "coverage.json", coverage)
        atomic_npz(output / "environment_metrics.npz", values=values, methods=np.asarray(names[:2]), environment_ids=ids)
        write_json(output / "protocol.json", dict(code_sha256=sha256(Path(__file__)), sources=sources,
            seeds=[0,1,2], selection="none; all fixed 40-epoch checkpoints", bootstrap_repetitions=10000,
            bootstrap_seed=20261029, multiplicity="12 contrasts per metric",
            uncertainty="environment intervals and across-training-seed sample SD reported separately",
            existing_test_set=True))
        lines = ["# 三个训练种子的 A/F 接收结果", "",
                 "所有种子使用相同 3456 个训练环境与固定 40 轮末尾权重，评价既有 864×17 测试条件。",
                 "不选择最好种子，不以本轮测试重新调参；seed 0 是原论文预先固定的模型。", "",
                 "| 种子 | 配置 | BER (%) | RMS EVM (%) | 配对 SNR (dB) |",
                 "|---:|---|---:|---:|---:|"]
        for r in per_seed:
            if r["method"] in names[:2]:
                lines.append(f"| {r['seed']} | {r['method']} | {100*r['ber']:.6f} | {r['rms_evm_percent']:.4f} | {r['paired_output_snr_db']:.4f} |")
        lines += ["", "## 种子间变异", ""]
        for r in summary:
            m = r["metrics"]["ber"]
            lines.append(f"- {r['method']}：BER 均值 {100*m['mean']:.6f}%，种子间样本标准差 {100*m['sample_sd']:.6f} 个百分点。")
        lines += ["", "三个种子的样本标准差描述本轮重训练变异，环境 bootstrap 区间不包含这种变异。",
                  "原始数据复用数量与新计算数量分别见 coverage.json；全部两种子原始记录重新核对 SHA-256。", ""]
        (output / "REPORT_zh.md").write_text("\n".join(lines))
        for name, digest in sources.items():
            assert sha256(name) == digest
        outputs = ["per_seed.json","comparisons.json","mean_and_seed_sd.json","coverage.json",
                   "environment_metrics.npz","protocol.json","REPORT_zh.md"]
        write_json(output / "complete.json", dict(status="complete",seeds=[0,1,2],
            environments_per_seed=864,carriers=17,joint_correction=False,
            file_sha256={n:sha256(output/n) for n in outputs},at=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    run(parser.parse_args().project.resolve())
