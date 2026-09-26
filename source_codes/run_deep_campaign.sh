#!/usr/bin/env bash
# 只能在华硕运行：训练、完整评分、图表汇总顺序执行，失败立即保留现场。
set -euo pipefail
cd /home/qyb/RESEARCH/wangluqiang_2026_paper_1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=.venv_dl/bin/python
OUT=dataset_simulation/baseline_results/20260924_huashuo_deep_six
mkdir -p "$OUT"
exec 9>"$OUT/campaign.lock"
flock -n 9 || { echo '已有实验持有运行锁'; exit 1; }
"$PY" source_codes/run_deep_baselines.py --train dataset_simulation/dataset_train --test dataset_simulation/dataset_test --output "$OUT"
"$PY" source_codes/evaluate_deep_baselines.py --test dataset_simulation/dataset_test --output "$OUT" --workers 4
"$PY" source_codes/summarize_deep_baselines.py --output "$OUT"
"$PY" source_codes/analyze_deep_conditions.py --output "$OUT"
"$PY" -m pip freeze > "$OUT/requirements.lock.txt"
"$PY" source_codes/verify_deep_campaign.py --output "$OUT"
