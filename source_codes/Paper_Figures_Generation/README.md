# Paper analysis figure generation

This self-contained package regenerates the **eight quantitative figures (Figures 3–10)** in the current manuscript. Figure 1 is a scenario illustration and Figure 2 is the method diagram; they are not numerical result plots and are not generated here.

The package reads frozen results. It does not train networks, run the optical receiver, select a new constellation case, or alter experiment outputs. It uses the current names **RLF**, **CRL**, and **CFS**. Original internal method IDs remain in the input files for traceability.

## Run

Python 3.9 or later is required. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_figures.py
```

The default generates English and Chinese PDF/PNG files in `generated/en/` and `generated/zh/`. Open **[the gallery](generated/index.html)**, or the combined eight-page PDFs: **[English](generated/en/all_analysis_figures.pdf)** / **[Chinese](generated/zh/all_analysis_figures.pdf)**.

Chinese labels require a CJK font. The script detects Noto Sans CJK, PingFang, FandolHei, Microsoft YaHei or SimHei. If detection fails, specify a font explicitly:

```bash
.venv/bin/python generate_figures.py --zh-font /path/to/NotoSansCJK-Regular.ttc
# English only, with no Chinese font dependency:
.venv/bin/python generate_figures.py --language en --output generated_english
# Verify the input package and numerical aggregates without drawing:
.venv/bin/python generate_figures.py --verify-only
```

Plotting is portable and needs neither CUDA nor a training machine. Training and receiver evaluation remain separate tasks under the project's experiment protocol. Fonts and Matplotlib versions can change layout; input values and aggregation rules are checked independently of pixel-level appearance.

## Figure and source mapping

| Paper figure | Output stem | Frozen input |
|---|---|---|
| 3: metric relationships | `rq1_metric_relations_all13_64` | `rq1_metrics.npz`: 864 environments × 17 carriers × 14 methods |
| 4: carrier/power profiles | `rq1_quality_profiles_all13_64` | `rq1_groups.json`; carrier values cross-checked against raw metric arrays |
| 5: array pattern | `array_ttd_phase` | `array_response.npz`, `array_geometry.json`: fixed geometry and saved quantised delays |
| 6: control-selection sources | `rq1_selection_origins_64` | `selection_origins.json`: initial/model/codebook counts |
| 7: same-input constellation | `rq1_constellation_all13_12_64` | `constellation.npz`: original case 363, 12 GHz, 248 received symbols per method |
| 8: sensitivity | `rq1_sensitivity` | `sensitivity_summary.json`, `sensitivity_comparisons.json`: all 11 conditions |
| 9: CRL/CFS ablation | `rq2_af_ablation` | `ablation_summary.json`, `ablation_comparisons.json`: four cells and fixed scan |
| 10: training scale | `rq3_training_quality` | `training_curves.json`: fixed epochs and fixed updates on the same validation set |

These eight PDF stems match the current LaTeX source. The program writes its own output directory and does not replace files in `latex/`.

## Numerical checks

* Verify the SHA-256 of all 19 bundled input files before plotting. `inputs/manifest.json` records original project-relative paths and transformations.
* Read method IDs from the raw array and reorder by name. The archive order differs from the paper's presentation order.
* Aggregate errors and denominators before computing BER/SER/BLER. Compute EVM as `100*sqrt(mean(NMSE))`, and paired SNR as `10*log10(sum(signal power)/sum(noise power))`.
* Recompute all five overall RQ1 metrics and the 17 carrier profiles, then compare with frozen aggregates at `rtol=1e-12`.
* Check that initial/model/codebook counts sum to 14,688 for every controller, and that RLF's RQ2 BER matches RQ1.
* Recompute case BER and EVM from received complex symbols. The 12 baselines with exactly identical controls and outputs share one panel; coordinate search and RLF have their own panels. No artificial jitter or modified baseline values are used.
* Draw array-factor azimuth cuts from frozen geometry/delay codes. Their values at the steering directions must agree with the saved deterministic array diagnostic. These cuts exclude propagation multipath and receiver electronics; they are not new BER evaluations.
* Read confidence intervals from completed analyses. The plotting program does not rerun bootstrap or introduce a new statistical seed.

`generated/plot_data/` exports environment-level metrics, array cuts and constellation metrics as CSV. `generated/manifest.json` records versions, input and code hashes, figure mapping and output hashes.

## Refresh inputs from a complete project checkout

The included input package is sufficient for plotting. To rebuild it from the original project data, run:

```bash
.venv/bin/python prepare_inputs.py --project /path/to/wangluqiang_2026_paper_1 --output inputs_new
.venv/bin/python generate_figures.py --inputs inputs_new --output generated_new
```

The exporter checks the original completion/checksum records, copies selected arrays/summaries byte for byte, and extracts four array constants from the frozen configuration. It refuses to overwrite an existing input directory. Original full datasets and records remain available through `dataset_simulation/RELEASE_DATA.json`; this package is the smaller plotting-input subset.

See [中文说明](README_zh.md) for a step-by-step explanation.
