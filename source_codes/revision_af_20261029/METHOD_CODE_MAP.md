# 当前论文方法与冻结代码的对应关系

当前论文的方法名称为 Response Learning with Feedback（RLF），由 Complex Response Learning（CRL，复响应学习）与 Candidate Feedback Selection（CFS，候选反馈选择）构成。历史实验标识 A 和 F 分别对应 CRL 和 CFS。

## 输入到输出

1. 输入公开的 2,513 维观测：16 套初始控制的导频 I/Q、实测评分、载频与已知噪声/器件信息。内部传播参数和业务正确比特不提供给控制器。
2. CRL 将观测提升为 64×31 的初始复响应，用复数卷积预测响应残差，恢复物理尺度。推理使用冻结权重。
3. 共同离散求解器从初始实测最佳控制开始，在 64 个延时码（每个 0–76）和 64 个衰减码（每个 0–24）上进行两轮逐坐标枚举。目标是预测响应与控制相关噪声形成的带内平均 MMSE 代理；不查询业务 BER。
4. CFS 先实际评价模型候选，再按固定随机顺序扫描几何码本。初始 16 次测量计入总预算 64；重复控制复用评分，不额外计费。按导频评分选择最优已测控制，并列时保留先测设置。
5. 输出 128 个合法整数码，用于 64 路延时和光衰减。反馈不重新训练网络，不修改预测响应，不读取业务参考答案。

## 代码定位

| 论文内容 | 实现位置 |
|---|---|
| 探测反演、初值正则化与尺度 | `our_method_response_control/physics.py` 的 `design`、`pilot_gain`；`model.py` 的 `conditions` |
| 复数卷积和响应残差网络 CRL | `our_method_response_control/model.py`；训练目标见同目录训练代码 |
| 传统响应替代 | `our_method_response_control/physics.py::covariance_estimate` |
| 共同离散求解器、控制相关噪声 | `our_method_response_control/physics.py::decode` |
| 64 次候选反馈 CFS | `study_uniform64/controllers.py::Feedback64.decide` |
| 固定随机顺序几何候选 | `baseline_codebook/method.py::optimize`：41×21 网格，方位 −35° 至 35°，俯仰 −15° 至 15° |
| 导频评分、实际测量计数与最优控制 | `baseline_common/feedback.py::FeedbackSession` |
| 主方法控制器载入 | `study_full_baselines/online_controller.py` 的 `complex_response_cnn`；外包 `Feedback64` |
| 当前 CRL/CFS 消融入口 | `study_rq2_af_20261029/run.py` |

固定扫描参照采用公开的 64 套控制；CFS 的后续几何候选来自更密的网格，两者不混称为同一个码本。

## 消融和证据

四格为传统响应估计、CRL、传统估计+CFS、RLF；另列固定扫描 64 次。前两格直接执行求解器候选，使用 16 次初始探测；后两格使用完整 64 次反馈。移除 CFS 与将 CFS 的预算设为 16 不同：后者只能选择初始已测控制。

`dataset_simulation/revision_rq2_af_20261029/full/verification.json` 记录了 864×17 条件、73,440 个输出的核查。RLF 与原 RQ1 的 `cnn_warm64` 控制逐条件一致；全部五组整数接收错误计数与原记录一致。当前消融不实例化联合校正控制器。

完整流程 BER 为 10.013619%，传统响应+CFS 为 10.245073%，CRL 直接输出为 12.902498%。CRL 的同预算收益与 CFS 连同额外测量的收益分别解释；两者的 SNR 对比区间包含零，不能将 BER 结论推广为全部指标都改善。

## 旧组件处理

`our_method_joint_refinement/` 与 `our_method_measurement_refinement/` 不属于当前方法。它们仍是旧冻结执行器和历史实验的依赖，保留以复查原稿问题及负结果；本次没有删除或改写这些记录，也没有改动已冻结的物理核、权重和控制算法。论文主方法、图和消融不再使用 B。

预算参数 B 是测量次数的数学符号，与已经移除的旧组件名称无关。当前两个组件统一称 CRL 和 CFS。
