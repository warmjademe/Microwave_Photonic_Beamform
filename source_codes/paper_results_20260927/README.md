# 论文结果图

绘图和固定接收重放在华硕主机运行，使用既有的完整 864 环境测试记录。
`build.py` 核验冻结统计与源代码哈希，绘制六种 64 次探测方法的分组曲线和指标关系，
并在固定测试索引 0、4/12/20 GHz 重放已经保存的控制。不会训练模型或重新优化控制。
星座使用八次噪声、每次 31 个业务符号；回放指标须与保存的原始评分在相对容差 1e-12 内一致。

`build_rq3.py` 从既有规模实验统计绘制 216 个验证环境上的 BER、EVM 曲线。
所需依赖为原实验环境，以及 Matplotlib 和 CairoSVG。为避免修改实验环境，本次绘图依赖安装在
华硕 `/tmp/mwp-figure-tools-20260927`，通过 `PYTHONPATH` 使用。

有效图形输出位于：

- `dataset_simulation/diagnostics/20260927_paper_final_figures_02`
- `dataset_simulation/diagnostics/20260927_paper_rq3_figures_02`

`_01` 保留为字体显示异常的诊断记录，不能用于论文图形交付。
`_02` 将 SVG 字形转换为矢量路径后生成 PDF，避免 CJK 字体子集显示异常。
论文采用的四幅 PDF 位于 `latex/figures`；数据来源和哈希保存在各输出目录的 `manifest.json`。

## 统一 64 次探测的全部基线比较

`build_uniform64.py` 使用 `20260927_uniform64_all13/analysis` 的完整审计结果，
绘制全部 13 个基线与本文方法的分组曲线、指标关系、配对差值和固定案例星座。
它还核对逐次测量轨迹，统计最终控制来自共同初始集合、模型候选或新增码本的比例。
模型与评分程序均保持冻结，星座仍使用测试索引 0 和 4/12/20 GHz。

`tables_uniform64.py` 从同一完整统计结果生成 RQ1 数值表和 RQ3 的 16/64 次预算对照表。
两项程序只接受已写入 `complete.json` 且哈希通过的分析输出，不使用部分结果。
统一比较的图形和表格目录分别为
`dataset_simulation/diagnostics/20260927_uniform64_all13_figures` 与
`dataset_simulation/diagnostics/20260927_uniform64_paper_tables`；以各目录的完整清单为完成依据。
