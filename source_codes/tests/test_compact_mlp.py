"""仅用临时合成训练数组，核对旧 MLP 等价性、恢复及公开特征读取。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baseline_common.data import features
from baseline_mlp.method import fit, load
from baseline_mlp.train_compact import fit_stream, feature_cache, main, _sha256
from compact_dataset import CompactDataset, pack_observation
from native_sim.config import NativeConfig


class InterruptedForTest(RuntimeError):
    pass


def compact_fixture(root, count=9):
    """不创建任何原始仿真或测试集文件，只创建 train 的合法轻量格式。"""
    root.mkdir()
    rng = np.random.default_rng(932)
    cfg = NativeConfig()
    pilots = (rng.choice([-1, 1], (31, 2)) + 1j * rng.choice([-1, 1], (31, 2))) / np.sqrt(2)
    public = dict(pilot_qpsk=pilots, probe_controls=rng.uniform(size=(16, 128)),
                  probe_angles_deg=np.zeros((16, 2)), positions_m=cfg.positions,
                  offsets_hz=cfg.offsets_hz)
    np.savez(root / "public.npz", **public)
    x = []
    for _ in range(count):
        iq = (rng.normal(size=(16, 31, 2)) + 1j * rng.normal(size=(16, 31, 2))).astype(np.complex64)
        x.append(pack_observation(dict(combined_iq_a=iq, quality=np.zeros(16),
                    noise_symbol_var_a2=np.full((16, 31), 1e-10), probe_apd_dc_a=np.ones(16)), 4))
    np.save(root / "X.npy", np.asarray(x, np.float32))
    codes = np.concatenate([rng.integers(0, 77, (count, 64)),
                            rng.integers(0, 25, (count, 64))], axis=1).astype(np.uint8)
    np.save(root / "Y_code.npy", codes)
    np.save(root / "Y.npy", (codes / np.r_[np.full(64, 76), np.full(64, 24)]).astype(np.float32))
    np.save(root / "record_sha256.npy", np.zeros((count, 32), np.uint8))
    environments = [dict(env_id="synthetic-train-" + str(i), source_path="train/e" + str(i))
                    for i in range(count)]
    (root / "environments.json").write_text(json.dumps(environments))
    manifest = dict(schema="mwp-compact-xy-v1", status="complete", split="train",
                    signal_config=cfg.to_dict(), sample_count=count, environment_count=count,
                    carriers_ghz=[4], source_dataset_relative="../intentionally_absent_original",
                    generation_fingerprint="synthetic-only", selection_fingerprint="synthetic-subset",
                    file_sha256={p.name: _sha256(p) for p in root.iterdir()})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class CompactMLPTests(unittest.TestCase):
    def test_single_epoch_matches_existing_fit_with_constant_columns(self):
        rng = np.random.default_rng(310)
        x = rng.normal(size=(53, 37)).astype(np.float32)
        x[:, 3:13] = np.arange(10, dtype=np.float32) / 7
        y = rng.uniform(size=(53, 128)).astype(np.float32)
        old = fit(x, y, epochs=1, batch_size=11, hidden=9, seed=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "x.npy", x)
            streamed = fit_stream(np.load(root / "x.npy", mmap_mode="r"), y,
                                  epochs=1, batch_size=11, hidden=9, seed=8,
                                  statistics_batch_size=13)
        for name in old.arrays:
            np.testing.assert_allclose(streamed.arrays[name], old.arrays[name], rtol=2e-12, atol=2e-14)
        self.assertEqual(streamed.settings, old.settings)
        self.assertEqual(streamed.metadata["constant_features"], 10)
        np.testing.assert_allclose([r["mse"] for r in streamed.history],
                                   [r["mse"] for r in old.history], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(streamed.predict_features(x), old.predict_features(x), atol=2e-14)

    def test_epoch_resume_is_exact_and_mismatch_rejected(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(41, 23)).astype(np.float32)
        x[:, 5:10] = 0.125
        y = rng.uniform(0.65, 0.9, size=(41, 128)).astype(np.float32)
        expected = fit_stream(x, y, epochs=3, batch_size=8, hidden=12, seed=4)
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            def interrupt(record):
                if record["epoch"] == 1:
                    raise InterruptedForTest()
            with self.assertRaises(InterruptedForTest):
                fit_stream(x, y, epochs=3, batch_size=8, hidden=12, seed=4,
                           checkpoint_dir=state, callback=interrupt, provenance={"fixture": "train"})
            self.assertTrue((state / "epoch_000001.json").exists())
            with self.assertRaisesRegex(ValueError, "配置/训练来源"):
                fit_stream(x, y, epochs=3, batch_size=8, hidden=12, seed=5,
                           checkpoint_dir=state, resume=True, provenance={"fixture": "train"})
            actual = fit_stream(x, y, epochs=3, batch_size=8, hidden=12, seed=4,
                                checkpoint_dir=state, resume=True, provenance={"fixture": "train"})
            for name in expected.arrays:
                np.testing.assert_array_equal(actual.arrays[name], expected.arrays[name])
            self.assertEqual(actual.history, expected.history)
            self.assertLess(actual.history[-1]["mse"], actual.history[0]["mse"])
            self.assertEqual(len(list(state.glob("epoch_*.json"))), 4)
            last_state = state / "epoch_000003.npz"
            last_state.write_bytes(last_state.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "哈希"):
                fit_stream(x, y, epochs=3, batch_size=8, hidden=12, seed=4,
                           checkpoint_dir=state, resume=True, provenance={"fixture": "train"})

    def test_public_feature_cache_and_cli_old_load_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = compact_fixture(root / "dataset_train")
            dataset = CompactDataset(train, verify_hashes=True)
            cache = root / "features"
            manifest_sha = _sha256(train / "manifest.json")
            with patch.object(dataset, "source_record", side_effect=AssertionError("不要打开原始真值")):
                x, metadata = feature_cache(dataset, cache, manifest_sha)
            self.assertIsInstance(x, np.memmap)
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(x.shape, (9, 3537))
            for index in range(len(dataset)):
                np.testing.assert_array_equal(x[index], features(dataset.observation(index)))
            x2, metadata2 = feature_cache(dataset, cache, manifest_sha)
            self.assertEqual(metadata, metadata2)
            np.testing.assert_array_equal(x, x2)
            with self.assertRaisesRegex(ValueError, "来源或哈希"):
                feature_cache(dataset, cache, "different-manifest")
            output = root / "checkpoint"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--dataset", str(train), "--output", str(output), "--epochs", "1",
                      "--batch-size", "4", "--hidden", "5", "--feature-cache", str(cache)])
            model = load(output)
            expected = fit(x, dataset.Y, epochs=1, batch_size=4, hidden=5, seed=0)
            np.testing.assert_allclose(model.predict_features(x), expected.predict_features(x),
                                       rtol=1e-12, atol=1e-13)
            u = model.predict(dataset.observation(0), dataset.cfg)
            self.assertEqual(u.shape, (128,))
            self.assertTrue(np.all((u >= 0) & (u <= 1)))
            np.testing.assert_allclose(u[:64] * 76, np.rint(u[:64] * 76), atol=1e-12)
            np.testing.assert_allclose(u[64:] * 24, np.rint(u[64:] * 24), atol=1e-12)
            self.assertFalse(model.metadata["source_dataset_opened"])
            self.assertEqual(model.metadata["compact_manifest_sha256"], manifest_sha)
            self.assertEqual(model.metadata["training_compact_manifest_sha256"], manifest_sha)
            self.assertEqual(model.metadata["generation_fingerprint"], "synthetic-only")
            self.assertEqual(model.metadata["selection_fingerprint"], "synthetic-subset")
            self.assertEqual(model.metadata["signal_config"], dataset.cfg.to_dict())
            self.assertEqual(model.metadata["training_environment_count"], 9)
            self.assertGreaterEqual(model.metadata["constant_features"], 2048)
            self.assertFalse(dataset.source_root.exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--dataset", str(train), "--output", str(output)])

    def test_reject_test_header_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"split": "test"}))
            with patch("compact_dataset.CompactDataset", side_effect=AssertionError("未授权读测试数组")):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--dataset", str(root), "--output", str(root / "output")])


if __name__ == "__main__":
    unittest.main()
