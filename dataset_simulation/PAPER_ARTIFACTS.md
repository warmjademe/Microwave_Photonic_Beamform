# Paper-to-artifact guide

This guide maps the manuscript's dataset, research questions, and figures to the released numerical artifacts. All data paths below are relative to `dataset_simulation/`. The [complete release manifest](RELEASE_DATA.json) combines the eight unchanged base groups with the two `data-2026-10-30` revision groups. Run the downloader from the repository root; it restores the original relative paths and checks each file's SHA-256.

The proposed method is A+F. Historical B sources, correction statistics and full-pipeline results are retained for traceability and are not relabelled as F. Revision studies reuse the existing test environments; transformed conditions and repeated training are not new independent propagation samples.

## Dataset and supervision

| Item | Location | Content |
|---|---|---|
| Split registry | `ops/dataset_split_20260926/registry.json` | 3,456 training / 216 validation / 864 test environments, membership, and seed separation |
| Training inputs | `outputs/scaling_train_3456_20260925/train/` | Per-environment measurements and propagation metadata; 17 carriers per environment |
| Validation inputs | `outputs/quality_rank_hybrid_20260925/test/` | The 216 validation environments; the folder name does not define its experimental role |
| Shared training view | `outputs/fair_view_3456_20260926/` | Common inputs for the final trained methods; these are not additional environments |
| Control targets | `baseline_results/20260925_full_baselines/fair_3456/teacher_labels/records/` | Offline delay/attenuation labels and associated records |
| Response targets | `baseline_results/20260925_full_baselines/scale_3456/targets/` | Per-branch, per-frequency complex-response supervision |
| Final model bundle | `baseline_results/20260925_full_baselines/fair_3456/runtime_bundle/` | Frozen model weights, shared inputs, and model provenance |
| Historical correction statistics | `diagnostics/20260926_joint_refinement_train3456/` and `diagnostics/20260926_measurement_refinement_fit_3456/` | Original B statistics, retained for historical records rather than the revised A/F method |
| Original final-test archive | `baseline_results/20260926_final864_selected/` | Public measurements, controls, raw metric arrays, feedback traces, propagation plans, and scoring seeds |
| 64-probe extension | `baseline_results/20260927_uniform64_all13/` | Eight additional baseline configurations over the same final-test conditions |

The two final-test archives contain 21 and eight configurations, respectively, across 864 environments and 17 carriers. Together they contain 425,952 method–condition evaluations. The main comparison selects 13 baselines and the proposed method at 64 probes; other configurations support budget comparisons, ablations, or separate references.

The stored numerical dataset consists of measurements, supervision, control settings, propagation configurations/seeds, and evaluation records. Raw constellation arrays are also provided for the paper's examples. Long passband waveforms for every condition are regenerated from the fixed simulation and seeds; they are not a separate full-waveform archive.

## RQ1: reception quality against 13 baselines

| Evidence | Data | Code under `source_codes/` |
|---|---|---|
| Base controls and raw errors/powers | `baseline_results/20260926_final864_selected/records/` | `study_final864/run.py` |
| Added 64-probe baseline records and measurement traces | `baseline_results/20260927_uniform64_all13/records/` | `study_uniform64/run.py`, `controllers.py`, `check.py` |
| All-method aggregates, carrier/power groups, and paired comparisons | `baseline_results/20260927_uniform64_all13/analysis/` | `study_uniform64/analyze.py` |
| Manuscript numerical tables | `diagnostics/20260927_uniform64_paper_tables/` | `paper_results_20260927/tables_uniform64.py` |
| Figure inputs and selected-control origins | `diagnostics/20260927_uniform64_all13_figures/` | `paper_results_20260927/build_uniform64.py` |
| English publication figures | `diagnostics/20261001_english_figures/` | `paper_results_20260927/build_english.py` |

The proposed 64-probe configuration is `cnn_warm64`: component A generates a candidate through the shared discrete solver, followed by measured candidate feedback. The additional fixed-scan identifier is `initial_select64`; network extensions use identifiers ending in `_feedback64`. Existing 64-probe methods are reused from the base archive after consistency checks. Read `protocol.json` and the analysis method lists to resolve array order.

The all-method aggregate must reproduce the paper's proposed BER of **10.0136%**, fixed-scan BER of **10.3705%**, and paired difference of **−0.3569 percentage points**. BER/SER/BLER aggregate error counts; EVM aggregates normalized squared errors before taking a square root; SNR aggregates signal and noise powers before taking a logarithm.

### Constellations

`diagnostics/20260927_constellation_reconstruction/reconstruction_case.npz` contains transmitted QPSK symbols, received symbols, controls, and metrics for the same input across all 14 methods. Its manifest records the selected test index 363 at 12 GHz and the case-selection rule. This illustrative case was selected from the 61 environments whose proposed BER and EVM were both better than every baseline at that carrier; it is not a random representative case.

`diagnostics/20260927_constellation_grouped/` contains the mapping from methods to panels. Methods with identical controls and received symbols share a panel; their data have not been perturbed to create visual differences. The fixed index-0 examples at 4, 12, and 20 GHz are retained in `diagnostics/20260927_uniform64_all13_figures/`.

## RQ2: component ablations

The revised comparison is conventional covariance response, A only, conventional response+F, and A+F, with BERs **13.7654%, 12.9025%, 10.2451%, and 10.0136%**, respectively. Direct output uses 16 probes; F uses 64, including the initial 16. Fixed scanning at 64 probes is an additional reference with BER **10.3705%**. Comparisons that remove A retain the same probing budget; comparisons that remove F also remove 48 measurements.

The revision records are in `revision_rq2_af_20261029/full/`: `verification.json` binds the raw records and analyses, `comparisons.json` contains paired effects and intervals, and `selection_diagnostics.json` records candidate origins and payload outcomes. They are included in `revision_core`. The A+F controls reproduce the original RQ1 `cnn_warm64` controls. `study_rq2_af_20261029/run.py` and `report.py` implement this experiment and analysis; A and F remain implemented by the original shared controller modules.

The original A/B direct-output comparison in `baseline_results/20260926_final864_selected/` remains historical evidence. Its correction statistics and B implementations are not components of the revised method. The original archives also provide the nominal controls, payloads, and noise records reused to pair the revised analyses.

### Completed revision supplements

The following are scientific-artifact locations, not additional dataset partitions. Keep the original model bundles and source snapshots when reproducing them. The first six rows and physical-model audit are in `revision_core`; the sensitivity, formal timing and memory records are in `revision_sensitivity_cost`.

| Study | Record locations | Coverage |
|---|---|---|
| A/F factorial ablation | `revision_rq2_af_20261029/full/` | Complete and checked |
| Three training seeds | `revision_af_20261029/S/` and `revision_20261029/e2_training/` | Complete and checked |
| Fixed-trajectory and inversion studies | `revision_af_20261029/H/` | Complete and checked; 216 validation environments |
| Quality-objective CNN/Transformer controls | `revision_af_20261029/C/` | All four configurations complete and retained |
| Matched real-valued response estimator | `revision_af_20261029/C_response/` | Complete and checked |
| Power-group and constellation analysis | `revision_af_20261029/D/` | Complete and checked |
| Frozen-controller sensitivity | `revision_af_20261029/R/full/`, `R/analysis/`, `R/figures/` | All 11 conditions, 864 environments and 17 carriers; 161,568 records |
| Physical-model audit | `revision_af_20261029/P/static/` | Formula and parameter checks; not hardware validation |
| Formal software timing | `revision_af_20261029/P/timing_full/` and `P/timing_analysis/` | Four controllers, 102 inputs, raw repetitions and full-call analysis |
| Formal process/GPU memory | `revision_af_20261029/P/memory_full/` | Separate A+F process, same 102 inputs |

The new supplementary comparisons reuse the existing 864 test environments. Their interpretation is partly retrospective, and transformed conditions are not counted as newly sampled independent environments.

## RQ3: hyperparameters and training scale

| Study | Released evidence |
|---|---|
| 16/64-probe quality comparison | Both final-test archives and `baseline_results/20260927_uniform64_all13/analysis/` |
| 216/432-environment trained models | `baseline_results/20260925_full_baselines/scale_small/` |
| 864-environment trained model | `baseline_results/20260925_response_control/` |
| 1,728-environment trained models | `baseline_results/20260925_full_baselines/scale_1728/` |
| 3,456-environment trained models | `baseline_results/20260925_full_baselines/scale_3456/` |
| Nested subset inputs and labels | Corresponding `outputs/` manifests, shared views, and training-study target directories |
| Raw validation evaluations | `baseline_results/20260925_full_baselines/evaluation_components/`, `evaluation_scaling_1728/`, and `evaluation_scaling_3456/` |
| Training budget/coverage audit and curves | `diagnostics/20260926_training_scale_budget/` |
| Training-quality figure source | `diagnostics/20260927_paper_rq3_figures_02/` |
| Search and feedback candidate studies | `baseline_results/20260925_full_baselines/evaluation_feedback_warm/`, `diagnostics/20260926_feedback_candidates_train/`, and `diagnostics/20260926_feedback_candidates_training_statistics/` |

The five training sizes are 216, 432, 864, 1,728, and 3,456 independent environments. Fixed-40-epoch and fixed-9,200-update schedules share the 864-environment point, producing nine distinct trained networks. The release includes final weights, optimizer checkpoints, training histories, coverage records where applicable, predicted responses, and validation results. `study_full_baselines/report_training_scale.py` verifies actual optimizer state and maps it to the quality curves.

These studies use the same **216 validation environments**. They are not additional independent final-test results. Candidate-count and search comparisons retain both improved and unimproved configurations; only the prescribed settings are used in the final comparison.

## Supplementary timing and verification

The original release retains `diagnostics/20261001_final_timing_run03/`: 27 configurations × 102 fixed inputs, with three timed repetitions after warm-up. It records software time, callback time, actual controls, resource checks, and independent audit results. `study_final_timing/run.py` and `audit.py` implement the replay and verification. Earlier interrupted attempts retain their failure status and are excluded from timing aggregates.

The revision's four-controller timing uses `revision_af_20261029/benchmark_af.py`. Each of 102 predetermined inputs has two warm-ups, three uninstrumented repetitions and three separate component-timing repetitions. `summarize_timing.py` reads the original arrays and computes the entire outer-call wall time minus measurement-replay callbacks. This includes return-value packaging omitted by the controller's narrower internal timer; it requires no new timing samples. Use `P/timing_analysis/summary.json` for the reported 28.551 ms A+F mean, while retaining `P/timing_full/` as raw evidence. Network forward timing is nested within response estimation. Assumed acquisition and switching are separate from measured software cost and do not establish hardware-loop latency.

`profile_memory.py` runs A+F alone on the same 102 inputs in a fresh process. `P/memory_full/` reports parameter bytes, process RSS, CUDA allocator peaks and driver samples with their distinct accounting boundaries. Preflight measurements are retained separately and do not enter the formal time or memory summaries.

`ops/research_completion_20261001/full_audit/audit.json` records checks on the final results, component effects, and nine scale-study models. `site_releases/final864_uniform64_timing_20261001/` preserves the final dashboard and downloadable summaries. The authoritative numerical results remain the per-condition arrays and the checksum-bound analyses, not screenshots.

## Integrity and execution

Every release member is listed with its SHA-256 in the compressed indices under `release_indices/`. The manifest identifies both the original base assets and this release's supplements. `fetch_artifacts.py --verify-only` checks restored bytes without running a numerical experiment.

Original protocols and source snapshots retain their bytes and paths for provenance. Some runners enforce the original Linux environment, absolute paths, and model-bundle relationships. Consult the [source guide](../source_codes/README.md) before running them in another environment; downloading raw data and independently inspecting arrays does not require those host checks.
