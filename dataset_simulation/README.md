# 当前仿真数据、监督标签与最终结果

本目录对应冻结的微波光子接收研究版本。仅考虑期望信号的多径、衰减、衰落、天线噪声与光电检测噪声，不设置独立干扰源、温漂或额外随机器件失配。

公开仓库：https://github.com/warmjademe/Microwave_Photonic_Beamform 。完整数据、监督标签、模型与逐载频结果见同一仓库的 [Release](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-09-26)。

## 三组独立环境

| 当前角色 | 环境数 | 环境—载频记录数 | 数据位置 |
|---|---:|---:|---|
| 训练 | 3,456 | 58,752 | `outputs/scaling_train_3456_20260925/train` |
| 验证 | 216 | 3,672 | `outputs/quality_rank_hybrid_20260925/test` |
| 最终测试 | 864 | 14,688 | `baseline_results/20260926_final864_selected/records` |

每个环境有 4–20 GHz 共 17 档载频。完整注册表为 `ops/dataset_split_20260926/registry.json`。验证集历史目录名为 `test`，这里的 216 环境已用于方法选择，不能作为最终独立测试报告。

`outputs/fair_view_3456_20260926` 将同一批训练环境和验证环境组织为共享读取视图；兼容视图和历史来源目录不增加独立样本数。保留这些路径是为了维持原始清单、文件哈希和模型来源关系。

## 输入、标签与字段

训练目录中，每个 `environment_XXXXX/data.npz` 的 `X` 为 `float32[17,2513]`。它由 16 套初始探测的 I/Q 导频、载频、导频质量及允许的接收统计量组成。`single_nmse`、`robust_nmse` 等字段是离线候选评价资料，不是新增在线输入。

直接控制监督位于 `baseline_results/20260925_full_baselines/fair_3456/teacher_labels/records`，其中 `control_code` 为 `int16[17,128]`。前 64 维是延时码 0–76，后 64 维是衰减码 0–24；分别乘以 19.53125 ps 和 0.5 dB 才是物理设置。

响应监督位于 `baseline_results/20260925_full_baselines/scale_3456/targets/train_response.npy`，形状为 `complex64[58752,64,31]`；同内容的 `fair_3456/targets` 用于共享训练视图。使用对应协议及环境清单确定记录顺序，不按文件系统遍历顺序拼接。

最终测试每个环境有 17 个 `carrier_XX.npz`。其中 `public_X` 为 `[2513]`，`control_code` 为 `[21,128]`，`metrics` 为 `[21,13]`，并保存使用反馈的方法的实际控制轨迹。方法顺序和指标顺序分别由同级 `protocol.json` 的 `methods` 与 `metric_order` 定义；测试参考信息仅用于评价。

## 模型与结果

当前冻结模型包为 `baseline_results/20260925_full_baselines/fair_3456/runtime_bundle`。组件 A 的最终权重来源为 `scale_3456/response_n3456_fixed_epochs`，组件 B 的频率和空间校正统计位于 `diagnostics/20260926_joint_refinement_train3456` 与 `diagnostics/20260926_measurement_refinement_fit_3456`。

最终测试已完成 864/864 环境、14,688 个环境—载频条件及 308,448 个方法—条件评价。`baseline_results/20260926_final864_selected/analysis` 保存汇总、分组、比较和审计。21 项包含 13 个基线、两个主方法、四个必要消融及两个额外信息参考。

## 下载与完整性核对

在公开仓库根目录执行：

```bash
python3 source_codes/release_tools/fetch_artifacts.py --list
python3 source_codes/release_tools/fetch_artifacts.py --extract
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

`RELEASE_DATA.json` 列出附件大小、下载地址和 SHA-256；`release_indices/` 提供逐文件校验清单。附件恢复原目录，并保留冻结模型包依赖的硬链接关系。

本机工作目录另保留 OSD 参数来源、逐节点校准与文献核查记录；它们与当前训练数据分开。公开附件采用经核对的研究仿真结果，局部校准不扩大解释为所有工况下对完整 OSD 系统的逐点等价。OptiSystem 安装程序和许可材料不属于数据附件。
