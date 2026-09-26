# 3,456 环境完整结果网站

## 当前发布规则：只保留完整864测试结果（2026-09-26）

用户要求实验完成后清理旧网站数据。新的生产链为`build_final.py → final_template.html + final_server.py → publish_final.py`；NAS的`wait_publish_nas.py`等待华硕全864测试及审计完成后执行。旧`build.py`和`template.html`仅保留作历史发布来源，不再用于本次最终上线。

`build_final.py`只读取`baseline_results/20260926_final864_selected`，并核对864环境、17载频、21项方法、308448方法案例及原始记录SHA。首页、`/baselines`和`/baselines.html`统一展示最终测试；下载仅含这批汇总、分组和配对比较。13个正式基线和2个主方法进入同预算主比较，4个必要消融及2个参考单列。

`publish_final.py`先用临时loopback端口测试待发布服务，再备份和替换。旧站所有非本轮资产移至`/home/qyb/services/mwp-device-audit-archive-时间戳/`，公开目录只保留最终页面、四份数据、服务及发布清单。旧历史页、校准页、`/deep/`、旧JSON/CSV、旧代码包地址返回410；内部清单、源码及备份路径不开放。研究目录内原始实验数据不删除。切换失败会恢复原服务与文件。

`check_final_publish.py`在华硕临时目录完成23个HTTP路径检查、6个错误发布条件检查，以及未完成真实测试的拒绝发布检查；不修改生产站点。以下内容为此前验证阶段的发布记录。

本目录只生成和发布已核验结果，不训练模型，也不运行新的评测。固定入口为 `https://guangzi.qyb.ink/baselines.html`。

## 数据来源

读取 `dataset_simulation/baseline_results/20260925_full_baselines/` 下两轮：

- `evaluation_fair3456_online`：41 个普通方法及变体、2 个参考。
- `evaluation_joint_refinement_3456_online`：8 个普通配置、2 个参考，其中 6 个与上一轮重复，新增 4 个配置。

合并后为 45 个普通方法及变体、2 个额外信息参考。共同训练 3,456 环境；旧 216 环境×17 载频做探索性比较。不能称为独立 864 环境确认。

构建时校验两轮完成标记、分析文件及原始逐载频文件的 SHA-256、6 个重叠方法的审核证据，并逐项核对重叠汇总与分组数值。重叠方法只显示一次。配对统计保留各轮原比较族，不把原 BH q 值解释成跨两轮重新校正。

## 构建与部署

在华硕项目根目录运行，输出使用新目录：

```sh
.venv_dl/bin/python source_codes/results_site_3456/build.py \
  --project /home/qyb/RESEARCH/wangluqiang_2026_paper_1 \
  --output /home/qyb/RESEARCH/wangluqiang_2026_paper_1/dataset_simulation/site_releases/发布版本名
```

将生成的 HTML、JSON、3 份 CSV 与 `release.json` 同步到 NAS 暂存目录，同时加入器件网站的 `server.py`、`native-calibration.html`、`baselines-20260924.html`。再在 NAS 执行 `publish.py --stage 暂存目录 --target /home/qyb/services/mwp-device-audit`。脚本检查文件和凭据模式，备份当前页面，原子替换文件，重启现有用户服务，逐字节核验本机 HTTP 输出。启动预压缩受原有 CPU 配额限制，健康检查允许 45 秒。

发布后用外网浏览器核验图表、预算/指标筛选、方法检索、条件分组和消融切换。自动 HTTP 请求可能受到 Cloudflare 的客户端规则拦截，不能据此直接认定浏览器访问故障。

## 保留范围

首页仍保留 OSD 历史核查，顶部入口指向最新结果。旧九基线改为有历史提示的单独页面，旧 JSON 和星座图不变；历史六网络及传播演示路由保留。公开文件不包含原始逐环境数组、模型权重、私有路径或凭据。

独立确认、正式在线时延、本轮固定案例星座图及 I/Q 尚未完成，页面明确标注；本页发布不代表整体研究目标完成。
