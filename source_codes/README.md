# 微波光子接收仿真与监督控制代码

当前版本对应 3,456 个训练环境、216 个验证环境和 864 个最终测试环境；每个环境覆盖 4–20 GHz 的 17 档载频。最终 864 环境的全部 21 项方法/配置已完成评价。数据与冻结结果见 [dataset_simulation](../dataset_simulation/README.md)。

公开仓库：https://github.com/warmjademe/Microwave_Photonic_Beamform 。二进制数据、监督标签和模型通过同一仓库的 Release 下载，`release_tools/fetch_artifacts.py` 负责下载和文件完整性核对。

## 当前入口

| 目录 | 作用 |
|---|---|
| `native_sim/` | 固定器件参数、波形、离散控制和共同光电接收计算 |
| `diagnostics/` | 当前版本仍依赖的数值修正与生成支撑模块 |
| `dataset_protocol/` | 三组环境注册、来源校验及种子隔离 |
| `baseline_*/` | 正式传统与深度学习基线，以及冻结包依赖的辅助模型 |
| `our_method_response_control/` | 组件 A：复响应学习与共同控制求解器 |
| `our_method_joint_refinement/` | 组件 B：联合测量校正 |
| `our_method_measurement_refinement/` | 联合校正使用的训练残差统计 |
| `study_final864/` | 最终清单、冻结、逐载频测试与统计分析 |
| `study_full_baselines/` | 训练、验证、数据规模与超参数研究的共享实现 |
| `results_site_3456/build_final.py` | 仅展示完整 864 环境结果的网站构建入口 |

正式清单为 13 个基线、两个预算下的主方法、四个必要消融及教师/MRC 两个参考。详情见 [PROTOCOL.md](study_final864/PROTOCOL.md)。代码中的一些辅助结构属于冻结执行器依赖，不代表新增正式比较方法。

当前输入为 2,513 维公开测量，最终执行输出为 64 个延时码和 64 个光衰减码。控制回归使用离线控制标签，响应学习使用等效复响应标签；两类方法共享训练环境与可观测信息。

## 执行与版本约束

原始实验使用华硕 Linux、Python 3.11.16、PyTorch 2.8.0+cu128 和 NumPy 1.26.4。依赖见 `requirements-deep.txt`。本项目数值实验仍仅在华硕执行；Mac 用于编辑、同步、文件校验与论文处理。

已冻结的计算源码、参数和协议继续保留原字节及哈希。旧启动器、退出使用的原型与编译缓存从当前目录清理；必要历史材料另行压缩留档。更换运行环境时应显式适配原脚本的路径、主机检查及硬链接校验，不能直接把修改后的输出称为原冻结实验。
