# NumPy 监督 MLP 基线

本目录提供单隐层 `ReLU → sigmoid(128)` 的监督学习基线，不依赖 PyTorch。输入只由 `baseline_common.data.features` 从公共测量生成，输出为 64 路延时和 64 路光衰减的归一化建议。最终用公共 `project` 转换为合法量化控制。

## 训练与推理

在 `source_codes` 目录运行，数据集必须已经完整生成：

```sh
../dataset_simulation/.venv/bin/python -m baseline_mlp.train \
  --dataset ../dataset_simulation/outputs/your_dataset \
  --output checkpoints/mlp_seed0 \
  --epochs 40 --batch-size 64 --hidden 128 --seed 0
```

也支持直接运行 `baseline_mlp/train.py`。上面的数据集名称是占位路径，不代表已经生成该数据。输出目录必须不存在；程序拒绝覆盖旧检查点。

```python
from baseline_mlp import load

model = load("checkpoints/mlp_seed0")
control = model.predict(observation, cfg)  # float64[128]，已限幅和量化
```

可独立测试的训练接口为 `fit(X, Y, epochs=40, batch_size=64, hidden=128, seed=0)`，返回 `MLP`。`X` 是训练特征矩阵，`Y` 是 `[样本数,128]` 的归一化控制标签。`predict_features` 返回未量化输出，用于核对保存/加载的一致性；真实控制评价使用 `predict`。

## 数据隔离与固定设置

CLI 只遍历 `Dataset.environments('train')`，只读取这些环境的公开观测和监督标签。不调用 `simulator_truth`，不读取测试观测，不建立验证集。特征均值和标准差只从传入训练矩阵拟合；近常量特征的缩放因子设为 1。测试推理沿用检查点中的训练统计量。

默认固定训练 40 轮，批量 64，隐层 128；Adam 学习率 `0.001`，一阶/二阶矩系数 `0.9/0.999`，epsilon `1e-8`。每轮仅打乱训练样本，优化所有归一化控制项的平均平方误差。隐层使用 He 初始化，输出层按隐层维数初始化。没有测试调参、早停或从多个 epoch 中挑选最好测试模型；保存最后一轮的模型。

检查点包含 `weights.npz`、`training_loss.json`、`checkpoint.json`。记录从第零轮到最终轮的训练 MSE、所有训练设置、训练环境 ID、载频、数据 manifest 的 SHA-256 摘要以及特征/控制映射与本目录代码哈希。加载时拒绝 pickle，并核对权重与损失文件哈希。此格式用于推理与审计，没有保存 Adam 动量以续训。

## 与 Transformer 公平比较

两种网络应使用相同训练环境划分、公开测量、控制标签和硬件量化器。不能给其中一种额外真实角度、真实信道、测试集统计量或更多在线探测。若 Transformer 使用更丰富的观测表示，应把信息差异单独说明并加入共享特征的对照。报告模型参数量、训练时间、推理时间和全部测量开销。

控制 MSE 是训练损失，不等于通信性能。不同控制可能产生相近输出，必须将预测送入共同器件模型，用独立检验符号评价 SNR/EVM 等指标。若实施多种随机种子，须预先固定种子列表，不能根据测试效果选种子。

当前仅完成小型合成数据的功能验证，未执行完整研究数据集训练；合成集损失下降不能作为微波光子控制效果的证据。
