"""轻量训练集上的流式 MLP：固定轮数，只读 train，不打开原始仿真真值。

运行示例（在线程环境变量已设置的进程中）：
  python -m baseline_mlp.train_compact --dataset ../dataset_simulation/dataset_train \
      --output ../dataset_simulation/baseline_results/mlp_seed0 --seed 0
中断后使用相同命令追加 --resume；每轮保存 Adam、随机数状态和权重。
最终目录保持旧 baseline_mlp.method.load() 格式；训练过程存于输出名.training。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_common.data import features
from baseline_mlp.method import MLP, _sigmoid


STREAM_SCHEMA = "mwp-numpy-mlp-stream-v1"
CACHE_SCHEMA = "mwp-compact-mlp-features-v1"
WEIGHTS = ("w1", "b1", "w2", "b2")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    """清单最后原子替换；未完成的临时文件不会当成检查点。"""
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def feature_cache(dataset, directory, manifest_sha256, callback=None):
    """将既有 features() 逐行写入 float32 mmap；三种训练种子可复用同一缓存。

    原始 X 的 quality/DC/标定噪声继续保存在数据集中；本基线沿用既有
    3537 维特征定义，不擅自改变输入。此函数不访问 source_root 或 Y。
    """
    if dataset.split != "train":
        raise ValueError("特征缓存只允许训练划分。")
    if len(dataset.X) == 0:
        raise ValueError("训练数据为空。")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "baseline_common/data.py"
    identity = dict(schema=CACHE_SCHEMA, compact_manifest_sha256=manifest_sha256,
                    feature_source_sha256=_sha256(source), rows=len(dataset.X),
                    columns=len(features(dataset.observation(0))), dtype="float32")
    data_path, metadata_path = directory / "features.npy", directory / "cache.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity or _sha256(data_path) != metadata.get("sha256"):
            raise ValueError("特征缓存来源或哈希不匹配；请选用新的缓存目录。")
        cached = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if cached.shape != (identity["rows"], identity["columns"]) or cached.dtype != np.float32:
            raise ValueError("特征缓存形状/类型不匹配。")
        return cached, metadata
    # 并发构建不抢写。异常锁保留，以免一个进程误用另一个未完成的缓存。
    lock = directory / "building.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()) + "\n")
    temporary = directory / f"features.npy.tmp.{os.getpid()}"
    started = time.perf_counter()
    try:
        cached = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(identity["rows"], identity["columns"]))
        for index in range(identity["rows"]):
            row = features(dataset.observation(index))
            if row.shape != (identity["columns"],) or not np.all(np.isfinite(row)):
                raise ValueError(f"训练特征第 {index} 行无效。")
            cached[index] = row
            if callback is not None and (index + 1) % 4096 == 0:
                callback(dict(stage="feature_cache", rows=index + 1, total=identity["rows"]))
        cached.flush()
        del cached
        temporary.replace(data_path)
        metadata = dict(identity=identity, sha256=_sha256(data_path),
                        elapsed_s=time.perf_counter() - started)
        _json(metadata_path, metadata)
    except BaseException:
        # 保留失败文件与锁；换用新目录即可重建，绝不把部分结果当完整缓存。
        raise
    else:
        lock.unlink()
    return np.load(data_path, mmap_mode="r", allow_pickle=False), metadata


def _training_statistics(x, y, block_size):
    """两遍 float64 归约，不分配全训练集的 float64 或标准化副本。"""
    total = np.zeros(x.shape[1], np.float64)
    minimum = np.full(x.shape[1], np.inf)
    maximum = np.full(x.shape[1], -np.inf)
    for begin in range(0, len(x), block_size):
        xb = np.asarray(x[begin:begin + block_size], np.float64)
        yb = np.asarray(y[begin:begin + block_size], np.float64)
        if (not np.all(np.isfinite(xb)) or not np.all(np.isfinite(yb))
                or np.any(yb < 0) or np.any(yb > 1)):
            raise ValueError("训练数据必须有限，Y 必须在 [0,1]。")
        total += np.sum(xb, axis=0)
        minimum = np.minimum(minimum, np.min(xb, axis=0))
        maximum = np.maximum(maximum, np.max(xb, axis=0))
    mean = total / len(x)
    constant = minimum == maximum
    mean[constant] = minimum[constant]
    variance_sum = np.zeros_like(mean)
    for begin in range(0, len(x), block_size):
        centered = np.asarray(x[begin:begin + block_size], np.float64) - mean
        variance_sum += np.sum(centered ** 2, axis=0)
    scale = np.sqrt(variance_sum / len(x))
    scale = np.where(scale < 1e-12, 1.0, scale)
    # 仅删去严格相同的列，不把近似常量、低方差特征当零。
    return mean, scale, np.flatnonzero(~constant)


def fit_stream(x, y, epochs=40, batch_size=64, hidden=128, seed=0,
               learning_rate=0.001, callback=None, checkpoint_dir=None,
               resume=False, provenance=None, statistics_batch_size=2048):
    """与旧 fit 相同的初始化/损失/Adam，按批读取 mmap，并在每轮落盘。

    callback 在该轮状态已保存之后执行。resume 使用相同固定轮数和来源，
    不靠测试分数或训练损失早停。常量输入权重保持最初随机值、梯度为零。
    """
    if (getattr(x, "ndim", None) != 2 or len(x) == 0 or x.shape[1] == 0
            or getattr(y, "shape", None) != (len(x), 128)):
        raise ValueError("需要 X[N,D] 和 Y[N,128] 的非空训练数组。")
    if any(int(v) != v or v <= 0 for v in (epochs, batch_size, hidden, statistics_batch_size)):
        raise ValueError("轮数、批量、隐层和统计分块大小必须为正整数。")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("学习率必须为正。")
    epochs, batch_size, hidden = int(epochs), int(batch_size), int(hidden)
    settings = dict(epochs=epochs, batch_size=batch_size, hidden=hidden, seed=int(seed),
                    learning_rate=float(learning_rate), adam_beta1=0.9, adam_beta2=0.999,
                    adam_epsilon=1e-8, loss="mean squared error over normalized controls",
                    training_rows=len(x), shuffle_each_epoch=True)
    protocol = dict(schema=STREAM_SCHEMA, settings=settings, input_dimension=x.shape[1],
                    statistics_batch_size=int(statistics_batch_size), provenance=provenance or {})
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if resume and directory is None:
        raise ValueError("恢复训练必须指定 checkpoint_dir。")
    started, elapsed_before = time.perf_counter(), 0.0
    rng = np.random.default_rng(seed)
    if directory is not None:
        if resume:
            stored = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
            if stored != protocol:
                raise ValueError("恢复配置/训练来源不一致；拒绝改变已有训练协议。")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            _json(directory / "protocol.json", protocol)
    if resume:
        snapshots = sorted(directory.glob("epoch_*.json"))
        if not snapshots:
            raise ValueError("未发现完整 epoch 检查点；请使用新的输出目录重新开始。")
        document = json.loads(snapshots[-1].read_text(encoding="utf-8"))
        state_path = directory / document["arrays_file"]
        if _sha256(state_path) != document["arrays_sha256"]:
            raise ValueError("恢复检查点数组哈希不一致。")
        with np.load(state_path, allow_pickle=False) as state:
            mean, scale, active = (state[name].copy() for name in ("mean", "scale", "active"))
            full_w1 = state["full_w1"].copy()
            weights = {name: state[name].copy() for name in WEIGHTS}
            first = {name: state["first_" + name].copy() for name in WEIGHTS}
            second = {name: state["second_" + name].copy() for name in WEIGHTS}
        history = document["history"]
        update_count, completed_epoch = document["update_count"], document["epoch"]
        rng.bit_generator.state = document["rng_state"]
        elapsed_before = document["elapsed_s"]
    else:
        mean, scale, active = _training_statistics(x, y, int(statistics_batch_size))
        # 必须先按原 D 初始化所有行，再缩小计算矩阵；RNG 序列与原 fit 相同。
        full_w1 = rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], hidden))
        weights = dict(w1=full_w1[active].copy(), b1=np.zeros(hidden),
                       w2=rng.normal(0, np.sqrt(1 / hidden), (hidden, 128)), b2=np.zeros(128))
        first = {name: np.zeros_like(value) for name, value in weights.items()}
        second = {name: np.zeros_like(value) for name, value in weights.items()}
        update_count, completed_epoch = 0, 0
        history = []

    def normalized(rows):
        # 先按行取小批量，再取活跃列；没有全矩阵 float64/normalized 拷贝。
        return (np.asarray(x[rows][:, active], np.float64) - mean[active]) / scale[active]

    def training_mse():
        total = 0.0
        for begin in range(0, len(x), max(batch_size, 256)):
            rows = slice(begin, begin + max(batch_size, 256))
            activation = np.maximum(normalized(rows) @ weights["w1"] + weights["b1"], 0)
            prediction = _sigmoid(activation @ weights["w2"] + weights["b2"])
            total += float(np.sum((prediction - np.asarray(y[rows], np.float64)) ** 2))
        return total / (len(x) * 128)

    def save_epoch(epoch):
        full_w1[active] = weights["w1"]
        elapsed = elapsed_before + time.perf_counter() - started
        if directory is not None:
            filename = f"epoch_{epoch:06d}.npz"
            temporary = directory / (filename + f".tmp.{os.getpid()}")
            arrays = dict(weights, full_w1=full_w1, mean=mean, scale=scale, active=active)
            arrays.update({"first_" + name: value for name, value in first.items()})
            arrays.update({"second_" + name: value for name, value in second.items()})
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(directory / filename)
            document = dict(schema=STREAM_SCHEMA, epoch=epoch, update_count=update_count,
                            rng_state=rng.bit_generator.state, elapsed_s=elapsed, history=history,
                            arrays_file=filename, arrays_sha256=_sha256(directory / filename))
            _json(directory / f"epoch_{epoch:06d}.json", document)
            _json(directory / "training_loss.json", history)
        return elapsed

    if not history:
        initial_loss = training_mse()
        if not np.isfinite(initial_loss):
            raise FloatingPointError("初始训练损失非有限。")
        history.append(dict(epoch=0, mse=initial_loss))
        save_epoch(0)
    for epoch in range(completed_epoch + 1, epochs + 1):
        epoch_started = time.perf_counter()
        order = rng.permutation(len(x))
        for begin in range(0, len(x), batch_size):
            rows = order[begin:begin + batch_size]
            xb, yb = normalized(rows), np.asarray(y[rows], np.float64)
            pre_activation = xb @ weights["w1"] + weights["b1"]
            activation = np.maximum(pre_activation, 0)
            prediction = _sigmoid(activation @ weights["w2"] + weights["b2"])
            delta2 = 2 * (prediction - yb) * prediction * (1 - prediction) / yb.size
            delta1 = (delta2 @ weights["w2"].T) * (pre_activation > 0)
            gradients = dict(w1=xb.T @ delta1, b1=np.sum(delta1, axis=0),
                             w2=activation.T @ delta2, b2=np.sum(delta2, axis=0))
            update_count += 1
            for name in weights:
                first[name] = 0.9 * first[name] + (1 - 0.9) * gradients[name]
                second[name] = 0.999 * second[name] + (1 - 0.999) * gradients[name] ** 2
                unbiased_first = first[name] / (1 - 0.9 ** update_count)
                unbiased_second = second[name] / (1 - 0.999 ** update_count)
                weights[name] -= learning_rate * unbiased_first / (np.sqrt(unbiased_second) + 1e-8)
        loss = training_mse()
        if not np.isfinite(loss):
            raise FloatingPointError("训练损失出现非有限值。")
        history.append(dict(epoch=epoch, mse=loss))
        elapsed = save_epoch(epoch)
        if callback is not None:
            callback(dict(history[-1], elapsed_s=elapsed,
                          epoch_elapsed_s=time.perf_counter() - epoch_started,
                          active_features=len(active), total_features=x.shape[1]))
    full_w1[active] = weights["w1"]
    model = MLP(dict(weights, w1=full_w1, mean=mean, scale=scale), settings, history)
    model.metadata = dict(streaming_schema=STREAM_SCHEMA,
                          active_features=len(active), constant_features=x.shape[1] - len(active),
                          elapsed_s=elapsed_before + time.perf_counter() - started,
                          resume_used=bool(resume), checkpoint_dir=str(directory) if directory else None)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="仅 dataset_train")
    parser.add_argument("--output", type=Path, required=True, help="全新、可由旧 load() 加载的最终目录")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--feature-cache", type=Path, help="三种种子共用的磁盘 float32 特征缓存目录")
    parser.add_argument("--resume", action="store_true", help="从输出名.training 的最后完整 epoch 恢复")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("最终输出已存在，拒绝覆盖。")
    manifest_path = args.dataset / "manifest.json"
    header = json.loads(manifest_path.read_text(encoding="utf-8"))
    if header.get("split") != "train":
        parser.error("只接受 split=train 的轻量目录；未打开该目录的数据数组。")
    from compact_dataset import CompactDataset
    dataset = CompactDataset(args.dataset, verify_hashes=True)
    if dataset.split != "train":
        raise ValueError("只允许训练划分。")
    manifest_sha = _sha256(manifest_path)
    source = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), source / "baseline_mlp/method.py",
                    source / "baseline_common/data.py", source / "baseline_common/controls.py",
                    source / "compact_dataset.py"]
    environments = [str(row.get("env_id", row.get("environment_id", row.get("source_path"))))
                    for row in dataset.environments]
    metadata = dict(training_environment_ids=environments, training_environment_count=len(environments),
                    training_sample_count=len(dataset.X), train_carriers_ghz=header["carriers_ghz"],
                    compact_manifest_sha256=manifest_sha, compact_schema=header["schema"],
                    training_compact_manifest_sha256=manifest_sha,
                    generation_fingerprint=header["generation_fingerprint"],
                    selection_fingerprint=header.get("selection_fingerprint"),
                    signal_config=dataset.cfg.to_dict(),
                    dataset_manifest_sha256=manifest_sha,
                    dataset_manifest_summary={key: header[key] for key in
                        ("schema", "status", "signal_config", "carriers_ghz", "generation_fingerprint",
                         "selection_fingerprint", "source_manifest_sha256") if key in header},
                    native_config=dataset.cfg.to_dict(), source_dataset_path=str(dataset.source_root),
                    source_dataset_opened=False,
                    model_selection="Fixed final epoch; no validation/test data loaded or scored.",
                    source_sha256={str(p.relative_to(source)): _sha256(p) for p in source_files},
                    thread_environment={key: os.environ.get(key) for key in
                                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                         "MKL_NUM_THREADS")})
    def report(record):
        print(json.dumps(dict(split="train", **record), ensure_ascii=False), flush=True)
    directory = args.output.with_name(args.output.name + ".training")
    if directory.exists() and not args.resume:
        parser.error("中间训练目录已存在；恢复请追加 --resume，或选择新的输出目录。")
    cache_dir = args.feature_cache or args.dataset.with_name(args.dataset.name + "_mlp_features_" + manifest_sha[:12])
    started = time.perf_counter()
    x, cache_metadata = feature_cache(dataset, cache_dir, manifest_sha, callback=report)
    metadata["feature_cache"] = dict(directory=str(cache_dir), **cache_metadata)
    model = fit_stream(x, dataset.Y, epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
                       seed=args.seed, learning_rate=args.learning_rate, callback=report,
                       checkpoint_dir=directory, resume=args.resume, provenance=metadata)
    metadata.update(model.metadata)
    metadata.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    current_invocation_elapsed_s=time.perf_counter() - started)
    model.save(args.output, metadata=metadata)
    report(dict(status="complete", checkpoint=str(args.output), training_rows=len(x),
                final_training_mse=model.history[-1]["mse"], elapsed_s=model.metadata["elapsed_s"]))


if __name__ == "__main__":
    main()
