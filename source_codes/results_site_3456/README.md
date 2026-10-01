# 完整独立测试结果网站

当前入口：https://guangzi.qyb.ink/baselines.html 。本目录只构建和发布已核验结果，不训练模型。

## 当前数据范围

仅使用 `baseline_results/20260926_final864_selected` 和 `baseline_results/20260927_uniform64_all13` 的最终记录：864环境、17载频、29个预算配置，共425,952方法案例。29项包括21个基线预算实现、2个主方法、4个共用消融和2个参考；64次主比较为13基线加本文方法。教师/MRC与必要消融单列。

正式计时来自 `diagnostics/20261001_final_timing_run03`：六个功率组各自首个环境×17载频=102输入，每配置热身2次、计时3次。不是864环境全量计时。公开页面保留硬件规格，不展示工作站名称。`timing_evidence.py`逐原始数组重算并核对来源SHA、资源检查和控制结果；计时不改变冻结质量结果。

`final_evidence.py`核对两批最终数据的记录与控制、指标汇总、分组和配对统计。`final_signals.html`展示3个预先固定案例和1个明确标注的改善案例；共同控制的星座相同，不伪造差别。验证集上的规模实验只保存在研究材料和论文中，不混入当前网站。

## 构建与核验

在远程Linux实验工作站的项目根目录执行；输出使用不存在的新目录：

```sh
.venv_dl/bin/python source_codes/results_site_3456/build_final.py \
  --project . \
  --timing dataset_simulation/diagnostics/20261001_final_timing_run03 \
  --output dataset_simulation/site_releases/新的发布目录
.venv_dl/bin/python source_codes/results_site_3456/check_final_publish.py \
  --project . --release dataset_simulation/site_releases/新的发布目录 \
  --report dataset_simulation/ops/新的发布核查.json
```

2026-10-01版本为 `final864_uniform64_timing_20261001`。完成27个HTTP路径检查、11个基础拒绝条件和10个计时拒绝条件；浏览器实际检查两预算、指标、消融与4个信号案例，无控制台错误。

## 发布与恢复

将发布文件同步到NAS站点外的暂存目录，运行 `publish_final.py --stage 暂存目录 --target /home/qyb/services/mwp-device-audit`。程序先用临时loopback端口验证真实待发布文件，再备份并原子替换，重启既有用户服务；失败恢复旧文件和服务。首页与 `/baselines.html` 指向相同最终结果。

公开内容仅有HTML、结果JSON和3份CSV。旧历史页、校准页、`/deep/`、旧JSON/CSV和代码包返回410；内部清单、源码、备份路径不开放。旧公开资产备份位于站点外，不删除研究原始记录。部署记录、截图和浏览器观察位于 `dataset_simulation/ops/research_completion_20261001/` 与发布目录。

旧 `build.py`、`template.html` 和 `wait_publish_nas.py` 保留作历史来源，不用于当前发布。重新发布前应使用本文件的最终构建流程；不能恢复混合验证与测试结果的旧站点。
