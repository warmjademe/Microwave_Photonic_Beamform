# CNN

三层一维卷积，沿31个子载波学习局部变化。

输入是统一的2513维公开探测数据；输出128个0–1连续控制，经共同量化器转为64个延时码和64个衰减码。不读取真实信道、逐阵元缓存或测试标签。

这是文献架构的监督控制适配，不是原论文实验复现。[正式论文](https://doi.org/10.1109/TMC.2024.3462960)；[详细适配与公平比较协议](../../dataset_simulation/docs/六种深度学习基线实验设计_20260924.md)。

实现见 [method.py](method.py)。在华硕项目根目录运行 `bash source_codes/run_deep_campaign.sh`，六种网络共享40轮、seed=0、batch=256、Adam学习率0.001及训练集拟合的标准化。单独加载模型：`from baseline_cnn.method import Model`。
