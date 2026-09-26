# 用接收质量作监督的候选控制排序

这是一项尚待结果判定的受控方法实验，使用监督学习，不使用强化学习。先读[固定实验协议](PROTOCOL.md)。原来的训练/测试数据和九基线、六深度网络结果均保留。

## 大白话的工作过程

1. 仿真一个有多条传播路径、衰减和正常噪声的环境。
2. 接收器试16套已知设置，得到公开的合路测量，共2513个输入数字。AI看到的不是64路真实信道。
3. 离线生成标签时，再发4帧独立数据，在同一传播环境内计算64套候选设置各自的接收误差。这些答案供训练使用，运行中的AI看不到。
4. AI学习给设置打分。如果两套设置几乎一样好，就允许两者都获得较高监督权重，避免把两套延时数字直接取平均。
5. 运行时只根据第2步的测量选一套设置，转换成64个延时码和64个衰减码。当前候选目录的衰减固定为0 dB，所以本轮只验证方向时延控制，不宣称同时优化了衰减能力。
6. 最后另发一帧没有用于标签计算的数据，用同一接收流程比较各方法误码率和失真。

候选排序是否优于传统16次探测选优，需要看完整测试；这里没有写入“必须超过传统方法”的停止条件。

## 文件职责

| 文件 | 作用 |
|---|---|
| `common.py` | 固定64候选、完整批量接收评分、五项模型/损失/控制映射 |
| `generate.py` | 检查数值门限，按环境划分后生成17载频输入和标签 |
| `train.py` | 只读取训练成员，40轮seed0监督训练 |
| `evaluate.py` | GPU推理、独立帧评分、传播分组、配对区间和权重回放 |
| `test_common.py` | 批量评分与原I/Q→FFT评价的独立一致性检查；等质标签检查 |
| `preflight.py` | 一个旧训练环境、17频率的生成流程及五模型GPU梯度检查 |
| `../diagnostics/hybrid_centered.py` | 弱响应小信号计算、强响应回退非线性RK4 |

## 数据布局

运行目录：`dataset_simulation/outputs/quality_rank_hybrid_20260925`。该目录的`train/`与`test/`各按环境保存，不改变旧`dataset_train/`与`dataset_test/`。

每个环境的`data.npz`包含：

- `X`：17×2513，网络唯一动态输入。
- `single_nmse`：17×64，单个独立标签帧的候选接收误差。
- `robust_nmse`：17×64，4个独立标签帧的平均候选误差。
- `measured64_choice`：仅供明确多48次反馈的参照，不能进入普通网络。
- `nonlinear_route_counts`：17×5，各频率、各帧采用完整非线性计算的通道数，属于生成审计。

`public.npz`保存固定已知导频、候选设置和角度。`environment.json`是真实传播参数，只允许生成器和评价器读取。网络训练器的测试读取入口禁止加载监督标签。

## 在华硕复现

从远程项目的`source_codes`目录执行，设置`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。依赖是该项目`.venv_dl`中的NumPy、PyTorch及系统C++17编译器。先按协议完成非线性参考、混合核检查，再运行：

```bash
../.venv_dl/bin/python our_method_quality_rank/generate.py --project PROJECT --output DATA --workers 6
../.venv_dl/bin/python our_method_quality_rank/train.py --data DATA --output RESULTS
../.venv_dl/bin/python our_method_quality_rank/evaluate.py --data DATA --run-root RESULTS --workers 6
```

`PROJECT`为华硕项目根，`DATA`和`RESULTS`替换为新的空目标目录。程序拒绝覆盖已有目录；复现时不删除原实验。全部输入、模型、结果由SHA绑定；运行开始后不修改冻结源码，后续改变须建立新版本。

## 公平性边界

五个网络和`public16`都使用16次公开测量；`feedback64`另用48次测量，单列其成本；`offline_catalog_reference`能读取仿真产生的质量答案，是额外信息参照。所有方法在同样的独立帧和配对噪声下评分。EVM来自接收失真，`-10log10(NMSE)`是等效质量指标，不是单独的热噪声SNR。

数值版本在6个训练环境、全部17载频的102案例检查中通过预设门限；它是明确近似的研究接收器，不是完整OSD等价或实体硬件认证。新测试只评价固定范围内的新传播环境，不能代表真实超视距外场或未建模时变、多普勒。
