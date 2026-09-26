# Microwave Photonic Beamforming

面向超视距无线接收的微波光子波束控制：复响应学习与联合测量校正。

本仓库提供 64 阵元接收仿真、监督学习与传统基线代码，以及冻结的训练数据、验证数据、最终测试记录、监督标签和模型。传播场景仅包含期望信号的多径、衰减、衰落及接收噪声，不加入独立干扰源或温漂。

## 数据与下载

| 划分 | 独立环境 | 每环境载频 | 环境—载频记录 |
|---|---:|---:|---:|
| 训练 | 3,456 | 4–20 GHz，共 17 档 | 58,752 |
| 验证 | 216 | 17 档 | 3,672 |
| 最终测试 | 864 | 17 档 | 14,688 |

同一环境的全部载频属于同一划分。验证集的历史目录名为 `test`，当前角色由三组数据注册表定义。

代码和说明保存在主分支；全量二进制数据保存在本仓库的 [Release](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-09-26)。在仓库根目录执行：

```bash
python3 source_codes/release_tools/fetch_artifacts.py --list
python3 source_codes/release_tools/fetch_artifacts.py --extract
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

下载工具需要 Python 3.9 或更新版本，使用标准库。它先检查附件 SHA-256，再解压和逐文件核对；不会运行训练或仿真。各附件的范围、大小与校验值见 [RELEASE_DATA.json](dataset_simulation/RELEASE_DATA.json)。

## 输入、监督与输出

- 一个环境的输入 `X` 为 `17 × 2513`，包括 16 组探测下的复导频观测、载频和允许的接收统计量。
- 直接控制监督 `control_code` 为 `17 × 128`：前 64 个数是延时码，后 64 个数是光衰减码。
- 响应学习监督包含逐路、逐频率的等效复响应。它与直接控制标签采用相同训练环境，但学习目标不同。
- 最终测试逐载频保存 `public_X`、21 个方法/配置的控制、指标和反馈轨迹。测试参考信息用于计分，不用于模型训练或控制选择。

延时码范围为 0–76，步长 19.53125 ps；衰减码范围为 0–24，步长 0.5 dB。64 路光信号总合并后由同一个光电探测器接收。

## 代码与实验入口

| 路径 | 作用 |
|---|---|
| `source_codes/native_sim/` | 冻结器件模型、波形与控制计算 |
| `source_codes/diagnostics/` | 当前生成过程所需的数值修正与支撑模块 |
| `source_codes/dataset_protocol/` | 环境划分、来源和种子核对 |
| `source_codes/baseline_*/` | 传统搜索与监督学习基线 |
| `source_codes/our_method_response_control/` | 组件 A：复响应学习及共同控制求解 |
| `source_codes/our_method_joint_refinement/` | 组件 B：联合测量校正 |
| `source_codes/study_final864/` | 冻结后的最终测试与统计分析 |
| `source_codes/study_full_baselines/` | 训练、验证及超参数研究的共享实现 |

正式比较包含 13 个基线、两个探测预算下的主方法、四个必要消融配置；离线教师与理想数字 MRC 另列为参考，共 21 项。完整定义见[最终测试协议](source_codes/study_final864/PROTOCOL.md)。

原始实验环境为 Linux、Python 3.11.16、PyTorch 2.8.0+cu128、NumPy 1.26.4 与 RTX 4090。依赖见 `source_codes/requirements-deep.txt`。冻结脚本保留原实验的路径、主机检查及文件身份校验；更换运行环境需要显式适配这些运行约束，并重新检查数值一致性。下载和检查文件完整性不依赖原实验主机。

## 结果与复现资料

最终 864 环境评测已经完成，逐载频结果与统计汇总随测试附件发布；正文中的主比较和消融以这些冻结记录为依据。历史预实验、失败诊断及旧权重不混入当前训练/测试入口。模型包中的部分共享辅助模型由原冻结包校验要求保留，不增加正式比较的方法数量。

数据路径、字段、标签对应关系与模型来源见 [数据说明](dataset_simulation/README.md)。仿真采用作者工程参数和经过记录范围内核对的研究模型，校准范围见随附说明。
