# 先估计64路复响应，再计算器件控制

本方法正在检验中，尚无训练后接收质量结论。实验只在华硕运行。

大白话流程是：接收器先试16套已知设置，得到同一份2513个数字。传统反演、训练协方差估计器和复数CNN分别根据这些数字估计“各路信号有多强、相位怎样、各频率有什么差异”。随后同一个计算器比较每路允许的延时和衰减档位，输出64个延时码和64个衰减码。CNN不读取测试环境的真实多径参数，也不额外试探硬件。

输出码范围沿用当前软件模型：延时码0–76，衰减码0–24。它们须按当前采样间隔和0.5 dB步进换算，不直接当作六个物理开关的二进制命令。最终通信质量使用同一完整混合接收模型中的独立数据帧评价。

- `PROTOCOL.md`：查看新测试结果前固定的假设、对照、预算与评价方式。
- `physics.py`：导频提取、传统估计、噪声协方差和128维控制搜索。
- `model.py`：复数CNN；`prepare.py`：只生成训练响应标签与训练协方差。
- `train.py`：seed0、40轮、最终模型；`evaluate.py`：独立帧评分与权重重放。
- `audit.py`：重新读取公开输入，重算全部下发控制并逐码核对。
- `test_model.py`、`test_noise.py`、`preflight.py`：机制、噪声和旧训练环境的完整流程检查。
- `run_when_ready.py`：等待质量排序实验完成并释放GPU，然后依次训练、评分和审计。

从华硕项目的`source_codes`目录运行（各输出目录必须尚不存在）：

```bash
../.venv_dl/bin/python our_method_response_control/train.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --output ../dataset_simulation/baseline_results/20260925_response_control
../.venv_dl/bin/python our_method_response_control/evaluate.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --run-root ../dataset_simulation/baseline_results/20260925_response_control --workers 6
../.venv_dl/bin/python our_method_response_control/audit.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --run-root ../dataset_simulation/baseline_results/20260925_response_control --workers 6
```

队列已经运行时不要重复执行上述命令。噪声校验中的多次抽样用于估计噪声统计量，与多次随机种子训练不同；每个网络仍只训练一次。

目前的线性响应监督是控制代理，最终混合接收器仍包含强响应路的非线性。报告时分别列出CPU预处理、GPU前向和CPU控制搜索耗时；这些计时不包括实际硬件探测与主机到GPU传输，不能直接写成整个真实设备的控制延迟。
