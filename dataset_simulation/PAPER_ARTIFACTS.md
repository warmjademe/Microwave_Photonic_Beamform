# Paper-to-artifact guide

This guide maps the manuscript's dataset, research questions, and figures to the released numerical artifacts. All data paths below are relative to `dataset_simulation/`. The [complete release manifest](RELEASE_DATA.json) combines unchanged base archives with the `data-2026-10-01` supplements. Run the downloader from the repository root; it restores the original relative paths and checks each file's SHA-256.

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
| Correction statistics | `diagnostics/20260926_joint_refinement_train3456/` and `diagnostics/20260926_measurement_refinement_fit_3456/` | Training-fitted frequency/spatial residual statistics |
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

Use `baseline_results/20260926_final864_selected/records/` and its `analysis/` directory. The four 16-probe direct-output configurations compare conventional covariance response, A only, conventional response plus B, and A+B. Their BERs are **13.7654%, 12.9025%, 13.5856%, and 12.3131%**, respectively.

The same archive contains the 64-probe candidate-generation comparison: conventional response plus feedback versus A plus feedback. Both use the same feedback procedure; their BERs are **10.2451%** and **10.0136%**. The 64-probe result does not represent a full A+B-plus-feedback experiment.

`study_final864/analyze.py` and `study_full_baselines/paired_statistics.py` implement aggregation and environment-paired statistics. `our_method_response_control/` and `our_method_joint_refinement/` implement A and B; the immutable run protocols define their settings.

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

`diagnostics/20261001_final_timing_run03/` contains the successful timing study: 27 configurations × 102 fixed inputs, with three timed repetitions after warm-up. It records software time, callback time, actual controls, resource checks, and independent audit results. `study_final_timing/run.py` and `audit.py` implement the replay and verification. Earlier interrupted attempts retain their failure status and are excluded from timing aggregates.

`ops/research_completion_20261001/full_audit/audit.json` records checks on the final results, component effects, and nine scale-study models. `site_releases/final864_uniform64_timing_20261001/` preserves the final dashboard and downloadable summaries. The authoritative numerical results remain the per-condition arrays and the checksum-bound analyses, not screenshots.

## Integrity and execution

Every release member is listed with its SHA-256 in the compressed indices under `release_indices/`. The manifest identifies both the original base assets and this release's supplements. `fetch_artifacts.py --verify-only` checks restored bytes without running a numerical experiment.

Original protocols and source snapshots retain their bytes and paths for provenance. Some runners enforce the original Linux environment, absolute paths, and model-bundle relationships. Consult the [source guide](../source_codes/README.md) before running them in another environment; downloading raw data and independently inspecting arrays does not require those host checks.
