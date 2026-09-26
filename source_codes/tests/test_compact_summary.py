"""临时合成九方法结果：核对轴、误码池化、环境聚类统计和拒错。

不读取正式实验结果或训练/测试数组。来源审计 verify_training 单独 mock，
避免用统计夹具伪装完整训练模型；来源审计由真实流程另行执行。
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from compact_dataset import sha256
from generate_native_dataset import json_bytes, write_json
from run_compact_baselines import GROUPS, METRICS
import summarize_compact_baselines as summary


SEEDS = [2, 0, 1]  # 故意不是数值排序，检测错误地把数组下标当 seed 的情况。
CARRIERS = list(range(4, 21))
BOOTSTRAP_SEED = 5289
BOOTSTRAP_REPLICATES = 257


def synthetic_metrics(method, environment_count):
    """返回 [seed, environment, carrier, metric]，各轴值均不同。

    NMSE 含方法×环境交互，避免配对差值恰好是常数而掩盖聚类错误。
    所有记录保持真实评测的62比特；参考方法的未定义指标显式置NaN。
    """
    mi = summary.ORDER.index(method)
    result = np.full((len(SEEDS), environment_count, 17, len(METRICS)), np.nan)
    for si, seed in enumerate(SEEDS):
        for ei in range(environment_count):
            for ci in range(17):
                nmse = 0.1 + 0.03 * mi + 0.007 * seed + 0.05 * ei + 0.001 * ci + 0.002 * mi * ei
                errors = (mi * 13 + seed * 7 + ei * 3 + ci) % 63
                values = dict(evm_percent=100 * np.sqrt(nmse), payload_nmse=nmse,
                              effective_snr_db=-10 * np.log10(nmse),
                              bit_errors=errors, bits_tested=62)
                if method == "mrc":
                    values["physical_reference_snr_db"] = 20 + 0.1 * seed + ci
                else:
                    values.update(pilot_objective=-nmse * 0.9, optical_dc_w=0.01,
                                  apd_dc_a=0.3, simulation_feedback_seconds=0.002)
                    if method == "teacher":
                        values.update(normalized_control_mse=0., delay_mae_ps=0., attenuation_mae_db=0.)
                    else:
                        calls = 16 if method in ("mlp", "ttd_das") else 64
                        wall = 0.0002 + mi * 0.00001 + seed * 0.000001 + ei * 0.0000001
                        values.update(feedback_calls=calls, controller_wall_seconds=wall,
                                      estimated_control_latency_seconds=wall + calls * 2.24e-6,
                                      normalized_control_mse=0.02 * mi + 0.001 * ei,
                                      delay_mae_ps=10. + mi + ei,
                                      attenuation_mae_db=0.1 + 0.01 * mi)
                for key, value in values.items():
                    result[si, ei, ci, METRICS.index(key)] = value
    return result


def write_phase(root, methods, expected, budget=64):
    root.mkdir()
    (root / "records").mkdir()
    count = next(iter(expected.values())).shape[1]
    protocol = dict(metric_order=METRICS, environment_count=count, sample_count=count * 17,
                    methods=methods, seeds=SEEDS, budget=budget,
                    test_manifest_sha256="synthetic-test-manifest",
                    carriers_ghz=CARRIERS, generation_fingerprint="synthetic-generation",
                    selection_fingerprint="synthetic-selection", verified_native_core={"fixture": "sha"},
                    groups={method: GROUPS[method] for method in methods},
                    test_dataset="not-opened-by-statistics-test", checkpoints=[])
    write_json(root / "protocol.json", protocol)
    write_json(root / "progress.json", dict(status="complete", final_sha_verification=True))
    fingerprint = hashlib.sha256(json_bytes(protocol)).hexdigest()
    for ei in range(count):
        values = np.stack([expected[method][:, ei] for method in methods])
        controls = np.zeros((len(methods), len(SEEDS), 17, 128), np.int16)
        for mi, method in enumerate(methods):
            if method == "mrc":
                controls[mi] = -1
            else:
                controls[mi, :, :, :64] = (mi + ei) % 77
                controls[mi, :, :, 64:] = (mi + ei) % 25
        path = root / "records" / f"environment_{ei:05d}.npz"
        identity = "synthetic-test-environment-" + str(ei)
        np.savez_compressed(path, metrics=values, control_code=controls,
                            protocol_fingerprint=np.array(fingerprint), environment_id=np.array(identity))
        write_json(path.with_suffix(".json"), dict(protocol_fingerprint=fingerprint,
                                                   sha256=sha256(path), environment_id=identity))
    return protocol


def fixture(root, count=5, learned_budget=64):
    expected = {method: synthetic_metrics(method, count) for method in summary.ORDER}
    # 文件内顺序故意与汇总展示顺序不同，防止错误地拼接方法轴。
    methods = ["mrc", "coordinate", "teacher", "de", "ttd_das", "done", "codebook", "spsa"]
    first = write_phase(root / "nonlearning", methods, expected)
    second = write_phase(root / "learned", ["mlp"], expected, budget=learned_budget)
    study = dict(test_manifest_sha256=first["test_manifest_sha256"], seeds=SEEDS,
                 budget=64, test_environments=count, test_samples=count * 17,
                 statistics=dict(bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_replicates=BOOTSTRAP_REPLICATES))
    write_json(root / "study_protocol.json", study)
    return expected, first, second


def replace_record(path, **changes):
    """用于构造内容合法但语义错误的记录；重新签SHA以触发更深层校验。"""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)
    marker = json.loads(path.with_suffix(".json").read_text())
    marker["sha256"] = sha256(path)
    write_json(path.with_suffix(".json"), marker)


class CompactSummaryTests(unittest.TestCase):
    def test_nine_methods_axes_pooled_ber_and_environment_clusters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected, _, _ = fixture(root)
            training_evidence = {"synthetic_mock_only": True, "environment_overlap": 0}
            with patch.object(summary, "verify_training", return_value=training_evidence) as verify:
                with contextlib.redirect_stdout(io.StringIO()):
                    report = summary.summarize(root)
            verify.assert_called_once()
            self.assertEqual(report["training_audit"], training_evidence)
            self.assertEqual(list(report["summary"]), summary.ORDER)
            self.assertEqual(report["independent_test_environments"], 5)
            self.assertEqual(report["test_samples"], 5 * 17)
            self.assertEqual(report["total_method_case_evaluations"], 5 * 17 * 3 * 9)
            self.assertEqual(report["seeds"], SEEDS)
            draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
                0, 5, (BOOTSTRAP_REPLICATES, 5))
            env_values = {}
            for method, data in expected.items():
                with self.subTest(method=method):
                    actual = report["summary"][method]
                    for metric in METRICS:
                        raw = data[..., METRICS.index(metric)]
                        if np.all(np.isnan(raw)):
                            self.assertIsNone(actual["mean_" + metric])
                            self.assertIsNone(actual["std_" + metric + "_across_seeds"])
                        else:
                            per_seed = np.asarray([raw[si].mean() for si in range(len(SEEDS))])
                            self.assertAlmostEqual(actual["mean_" + metric], float(per_seed.mean()), places=12)
                            self.assertAlmostEqual(actual["std_" + metric + "_across_seeds"],
                                                   float(np.std(per_seed, ddof=1)), places=12)
                    errors = data[..., METRICS.index("bit_errors")]
                    bits = data[..., METRICS.index("bits_tested")]
                    self.assertEqual(actual["pooled_bit_errors"], int(errors.sum()))
                    self.assertEqual(actual["pooled_bits_tested"], 3 * 5 * 17 * 62)
                    self.assertAlmostEqual(actual["pooled_ber"], float(errors.sum() / bits.sum()), places=14)
                    np.testing.assert_allclose(actual["ber_by_seed"],
                        [errors[si].sum() / bits[si].sum() for si in range(len(SEEDS))], rtol=0, atol=1e-15)
                    nmse = data[..., METRICS.index("payload_nmse")]
                    # 先保留独立环境，环境内部同时平均三个种子和17载频。
                    env_values[method] = np.asarray([nmse[:, ei, :].mean() for ei in range(5)])
                    means = np.asarray([np.mean(env_values[method][indices]) for indices in draws])
                    np.testing.assert_allclose(actual["payload_nmse_environment_bootstrap_ci95"],
                                               np.percentile(means, [2.5, 97.5]), rtol=0, atol=1e-14)
                    self.assertEqual(len(report["by_frequency"][method]), 17)
                    for ci, by_frequency in enumerate(report["by_frequency"][method]):
                        self.assertEqual(by_frequency["carrier_ghz"], ci + 4)
                        self.assertAlmostEqual(by_frequency["mean_payload_nmse"], nmse[:, :, ci].mean(), places=14)
                        self.assertAlmostEqual(by_frequency["mean_evm_percent"],
                                               data[:, :, ci, METRICS.index("evm_percent")].mean(), places=12)
                        self.assertAlmostEqual(by_frequency["pooled_ber"],
                                               errors[:, :, ci].sum() / bits[:, :, ci].sum(), places=14)
            self.assertEqual(set(report["paired_environment_analysis"]), set(summary.ORDER[:6]))
            for method in summary.ORDER[:6]:
                paired = report["paired_environment_analysis"][method]
                delta = env_values["mlp"] - env_values[method]
                means = np.asarray([delta[indices].mean() for indices in draws])
                self.assertEqual(paired["independent_units"], 5)
                self.assertTrue(paired["lower_is_better_for_mlp"])
                self.assertAlmostEqual(paired["mean_nmse_difference_mlp_minus_baseline"], delta.mean(), places=14)
                np.testing.assert_allclose(paired["environment_bootstrap_ci95"],
                                           np.percentile(means, [2.5, 97.5]), rtol=0, atol=1e-14)
            disk_report = json.loads((root / "summary.json").read_text())
            self.assertEqual(disk_report, report)
            self.assertTrue((root / "summary.md").is_file())

    def test_single_seed_has_no_seed_std_and_strict_finite_json(self):
        """单seed只报告环境不确定性，不能制造跨seed标准差或NaN。"""
        with tempfile.TemporaryDirectory() as temporary, patch(__name__ + ".SEEDS", [0]):
            root = Path(temporary)
            expected, _, _ = fixture(root, count=3)
            with patch.object(summary, "verify_training", return_value={"synthetic_mock_only": True}):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = summary.summarize(root)
            self.assertEqual(report["seeds"], [0])
            self.assertEqual(report["test_samples"], 3 * 17)
            self.assertEqual(report["independent_test_environments"], 3)
            self.assertEqual(report["total_method_case_evaluations"], 3 * 17 * 1 * 9)
            for method in summary.ORDER:
                with self.subTest(method=method):
                    result = report["summary"][method]
                    self.assertEqual(result["seeds"], [0])
                    self.assertEqual(result["test_samples_per_seed"], 3 * 17)
                    for metric in METRICS:
                        self.assertIsNone(result["std_" + metric + "_across_seeds"])
                    errors = expected[method][..., METRICS.index("bit_errors")]
                    self.assertEqual(result["pooled_bits_tested"], 3 * 17 * 62)
                    self.assertEqual(result["pooled_bit_errors"], int(errors.sum()))
                    self.assertEqual(len(result["ber_by_seed"]), 1)
                    self.assertAlmostEqual(result["pooled_ber"], errors.sum() / (3 * 17 * 62), places=14)
            def reject_nonstandard_constant(value):
                raise AssertionError("汇总JSON不得包含NaN或Infinity：" + value)
            text = (root / "summary.json").read_text()
            disk_report = json.loads(text, parse_constant=reject_nonstandard_constant)
            self.assertEqual(disk_report, report)
            # 内存报告也必须能以严格JSON序列化，None应写作null。
            json.dumps(report, allow_nan=False)
            markdown = (root / "summary.md").read_text()
            self.assertIn("459", markdown)

    def test_cross_phase_protocol_and_predeclared_study_mismatch_rejected(self):
        for mode in ("phase", "study"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root, learned_budget=65 if mode == "phase" else 64)
                if mode == "study":
                    study = json.loads((root / "study_protocol.json").read_text())
                    study["budget"] = 63
                    write_json(root / "study_protocol.json", study)
                with patch.object(summary, "verify_training", return_value={}):
                    with self.assertRaisesRegex(ValueError, "比较条件不同|预定研究协议不同"):
                        summary.summarize(root)
                self.assertFalse((root / "summary.json").exists())

    def test_record_sha_and_internal_protocol_corruption_rejected(self):
        for mode in ("sha", "inside", "marker"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root)
                path = root / "nonlearning/records/environment_00000.npz"
                if mode == "sha":
                    path.write_bytes(path.read_bytes() + b"uncommitted mutation")
                elif mode == "inside":
                    replace_record(path, protocol_fingerprint=np.array("different-protocol"))
                else:
                    marker = json.loads(path.with_suffix(".json").read_text())
                    marker["protocol_fingerprint"] = "different-protocol"
                    write_json(path.with_suffix(".json"), marker)
                with self.assertRaisesRegex(ValueError, "SHA/协议|内部协议"):
                    summary.read_phase(root / "nonlearning")

    def test_incomplete_missing_duplicate_and_nonfinite_records_rejected(self):
        for mode in ("incomplete", "missing", "duplicate", "nan", "invalid_control"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root)
                phase = root / "nonlearning"
                path = phase / "records/environment_00001.npz"
                if mode == "incomplete":
                    write_json(phase / "progress.json", dict(status="running", final_sha_verification=False))
                elif mode == "missing":
                    path.unlink()
                elif mode == "duplicate":
                    replace_record(path, environment_id=np.array("synthetic-test-environment-0"))
                elif mode == "nan":
                    with np.load(path, allow_pickle=False) as archive:
                        values = archive["metrics"].copy()
                    values[0, 1, 3, METRICS.index("evm_percent")] = np.nan
                    replace_record(path, metrics=values)
                else:
                    with np.load(path, allow_pickle=False) as archive:
                        control = archive["control_code"].copy()
                    control[1, 1, 2, 0] = 77
                    replace_record(path, control_code=control)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    summary.read_phase(phase)

    def test_environment_order_mismatch_and_missing_method_rejected(self):
        for mode in ("environment_order", "missing_method"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                expected, _, _ = fixture(root)
                if mode == "environment_order":
                    paths = [root / "learned/records" / f"environment_{i:05d}.npz" for i in (0, 1)]
                    replace_record(paths[0], environment_id=np.array("synthetic-test-environment-1"))
                    replace_record(paths[1], environment_id=np.array("synthetic-test-environment-0"))
                else:
                    # 构造本身提交完整、但缺少一个方法的阶段，不能只依赖文件完整判断。
                    other = root / "incomplete_method_phase"
                    write_phase(other, [m for m in summary.ORDER if m not in ("mlp", "de")], expected)
                    (root / "nonlearning").rename(root / "original_nonlearning")
                    other.rename(root / "nonlearning")
                with patch.object(summary, "verify_training", return_value={}):
                    with self.assertRaisesRegex(ValueError, "环境顺序不同|九基线未齐全"):
                        summary.summarize(root)
                self.assertFalse((root / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
