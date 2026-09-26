"""单隐层 ReLU + sigmoid 控制回归；不读取数据集或仿真真值。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "mwp-numpy-mlp-v1"


def _sigmoid(value):
    # 两侧分别计算，避免 exp(大正数) 溢出。
    exponential = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MLP:
    def __init__(self, arrays, settings, history):
        self.arrays = {name: np.asarray(value, np.float64).copy() for name, value in arrays.items()}
        self.settings = dict(settings)
        self.history = list(history)
        self.metadata = {}
        self._validate()

    def _validate(self):
        a = self.arrays
        if set(a) != {"w1", "b1", "w2", "b2", "mean", "scale"}:
            raise ValueError("检查点数组字段不完整。")
        if a["w1"].ndim != 2 or a["w2"].ndim != 2:
            raise ValueError("MLP 权重必须是矩阵。")
        inputs, hidden = a["w1"].shape
        if (a["w2"].shape != (hidden, 128) or a["b1"].shape != (hidden,)
                or a["b2"].shape != (128,) or a["mean"].shape != (inputs,)
                or a["scale"].shape != (inputs,) or np.any(a["scale"] <= 0)
                or not all(np.all(np.isfinite(v)) for v in a.values())):
            raise ValueError("MLP 检查点尺寸或数值无效。")

    def predict_features(self, x):
        """对公开测量特征推理；返回未量化的 [128] 或 [批量,128]。"""
        x = np.asarray(x, np.float64)
        single = x.ndim == 1
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.arrays["mean"]):
            raise ValueError("输入特征维度与检查点不一致。")
        if not np.all(np.isfinite(x)):
            raise ValueError("输入特征含非有限值。")
        a = self.arrays
        normalized = (np.atleast_2d(x) - a["mean"]) / a["scale"]
        hidden = np.maximum(normalized @ a["w1"] + a["b1"], 0)
        result = _sigmoid(hidden @ a["w2"] + a["b2"])
        return result[0] if single else result

    def predict(self, observation, cfg):
        """使用公共特征入口和公共硬件量化器；不读取任何信道真值。"""
        from baseline_common.data import features
        from baseline_common.controls import project
        if cfg.n != 64:
            raise ValueError("当前检查点输出限定为 64 路双控制。")
        return project(self.predict_features(features(observation)), cfg).astype(np.float64)

    def save(self, output, metadata=None):
        """只创建全新检查点目录；已有目录/文件一律拒绝覆盖。"""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "weights.npz", **self.arrays)
        _write_json(output / "training_loss.json", self.history)
        document = dict(schema=SCHEMA, created_at_utc=datetime.now(timezone.utc).isoformat(),
                        numpy_version=np.__version__, settings=self.settings,
                        input_dimension=len(self.arrays["mean"]), output_dimension=128,
                        normalization="mean/std fitted only on supplied training X; constant scale=1",
                        selection="fixed final epoch; no validation or test model selection",
                        metadata=self.metadata if metadata is None else metadata,
                        weights_sha256=_sha256(output / "weights.npz"),
                        training_loss_sha256=_sha256(output / "training_loss.json"))
        # manifest 最后写；中途失败的目录不会被 load 当成完整检查点。
        _write_json(output / "checkpoint.json", document)


def load(output):
    """加载本实现检查点，校验哈希和数组结构，不加载 pickle。"""
    output = Path(output)
    document = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError("未知 MLP 检查点格式。")
    for name, field in (("weights.npz", "weights_sha256"),
                        ("training_loss.json", "training_loss_sha256")):
        if _sha256(output / name) != document[field]:
            raise ValueError("MLP 检查点哈希不一致：" + name)
    with np.load(output / "weights.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    model = MLP(arrays, document["settings"], json.loads(
        (output / "training_loss.json").read_text(encoding="utf-8")))
    model.metadata = document.get("metadata", {})
    return model


def fit(x, y, epochs=40, batch_size=64, hidden=128, seed=0,
        learning_rate=0.001, callback=None):
    """固定轮数训练；调用方只能传入训练数据，不做早停或模型挑选。"""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if (x.ndim != 2 or y.shape != (len(x), 128) or len(x) == 0 or x.shape[1] == 0
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(y < 0) or np.any(y > 1)):
        raise ValueError("需要有限训练 X[N,D] 和范围 [0,1] 的 Y[N,128]。")
    if any(int(v) != v or v <= 0 for v in (epochs, batch_size, hidden)):
        raise ValueError("轮数、批量和隐层维数必须为正整数。")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("学习率必须为正。")
    epochs, batch_size, hidden = int(epochs), int(batch_size), int(hidden)
    rng = np.random.default_rng(seed)
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    normalized = (x - mean) / scale
    weights = dict(w1=rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], hidden)),
                   b1=np.zeros(hidden), w2=rng.normal(0, np.sqrt(1 / hidden), (hidden, 128)),
                   b2=np.zeros(128))
    first_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    second_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    update_count = 0

    def training_mse():
        # 分块计算完整训练损失，避免额外分配整个训练集的隐层矩阵。
        total = 0.0
        for begin in range(0, len(x), max(batch_size, 256)):
            end = begin + max(batch_size, 256)
            activation = np.maximum(normalized[begin:end] @ weights["w1"] + weights["b1"], 0)
            prediction = _sigmoid(activation @ weights["w2"] + weights["b2"])
            total += float(np.sum((prediction - y[begin:end]) ** 2))
        return total / y.size

    history = [{"epoch": 0, "mse": training_mse()}]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for begin in range(0, len(x), batch_size):
            rows = order[begin:begin + batch_size]
            xb, yb = normalized[rows], y[rows]
            pre_activation = xb @ weights["w1"] + weights["b1"]
            activation = np.maximum(pre_activation, 0)
            prediction = _sigmoid(activation @ weights["w2"] + weights["b2"])
            delta2 = 2 * (prediction - yb) * prediction * (1 - prediction) / yb.size
            delta1 = (delta2 @ weights["w2"].T) * (pre_activation > 0)
            gradients = dict(w1=xb.T @ delta1, b1=np.sum(delta1, axis=0),
                             w2=activation.T @ delta2, b2=np.sum(delta2, axis=0))
            update_count += 1
            for name in weights:
                first_moment[name] = beta1 * first_moment[name] + (1 - beta1) * gradients[name]
                second_moment[name] = beta2 * second_moment[name] + (1 - beta2) * gradients[name] ** 2
                unbiased_first = first_moment[name] / (1 - beta1 ** update_count)
                unbiased_second = second_moment[name] / (1 - beta2 ** update_count)
                weights[name] -= learning_rate * unbiased_first / (np.sqrt(unbiased_second) + epsilon)
        loss = training_mse()
        if not np.isfinite(loss):
            raise FloatingPointError("训练损失出现非有限值。")
        record = {"epoch": epoch, "mse": loss}
        history.append(record)
        if callback is not None:
            callback(dict(record))
    settings = dict(epochs=epochs, batch_size=batch_size, hidden=hidden, seed=int(seed),
                    learning_rate=learning_rate, adam_beta1=beta1, adam_beta2=beta2,
                    adam_epsilon=epsilon, loss="mean squared error over normalized controls",
                    training_rows=len(x), shuffle_each_epoch=True)
    return MLP(dict(weights, mean=mean, scale=scale), settings, history)
