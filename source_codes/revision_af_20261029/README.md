# Complex response learning and candidate feedback: revision experiments

These source files accompany the revised A/F formulation. The main method remains the frozen `cnn_warm64` controller; these experiments do not change its weights, solver, or measured-score selection rule.

| Study | Evaluation / analysis | Protocol |
|---|---|---|
| Three training seeds | `evaluate_seeds.py`, `summarize_seeds.py` | `SEEDS_PROTOCOL.md` |
| Loading, noise weighting, candidate count and measurement prefixes | `evaluate_h.py`, `summarize_h.py` | `H_PROTOCOL.md` |
| Propagation, RIN and static device sensitivity | `evaluate_r.py`, `perturbed_receiver.py`, `summarize_r.py` | `R_PROTOCOL.md` |
| Quality-objective control CNN and Transformer | `train_quality.py`, `evaluate_quality.py`, `summarize_quality.py` | `C_PROTOCOL.md` |
| Matched real-valued response estimator | `audit_response_inputs.py`, `evaluate_real_response.py`, `analyze_real_response.py` | `C_RESPONSE_PROTOCOL.md` |
| Power groups and constellation display | `analyze_power.py`, `render_constellation.py` | Original frozen test records |
| Physical-model audit | `physical_model_audit.py` | Formula checks, not hardware measurements |
| Software timing and full-call analysis | `benchmark_af.py`, `summarize_timing.py` | `P_TIMING_PROTOCOL.md` |
| Single-controller process and GPU memory | `profile_memory.py` | `P_MEMORY_PROTOCOL.md` |

`render_ablation.py` and `render_sensitivity.py` draw only checksum-bound completed analyses. Queue scripts coordinate the original execution and preserve failure states; they do not make incomplete jobs into completed evidence. The original protocols are retained byte-for-byte for provenance, including their Chinese descriptions.

The A/F factorial study is in `../study_rq2_af_20261029/`. A is implemented by `../our_method_response_control/model.py`, the shared discrete solver by that module's `physics.py::decode`, and F by `../study_uniform64/controllers.py::Feedback64`. Historical joint correction B is not a component of A/F.

The data roots and download groups are documented in [the paper-to-artifact guide](../../dataset_simulation/PAPER_ARTIFACTS.md). Training uses 3,456 environments; the H study uses 216 validation environments; test comparisons use the existing 864 test environments and all 17 carriers. The sensitivity study transforms the same environments rather than introducing a new independent holdout. All configurations and unfavourable results are retained.

The frozen runners retain the original host, path and source-identity checks. Reproduction on a different machine requires an explicit environment/path adaptation and a new run directory, followed by the specified zero-change and control-replay checks. Do not edit the archived identity manifests to make a changed implementation appear identical. Downloading and verifying the artifacts does not require executing these runners.

All listed studies are complete. `R/analysis/complete.json` verifies 161,568 environment–carrier–condition records. `P/timing_full/` retains the original timing arrays; `P/timing_analysis/` uses the outer call wall time minus replay-callback time, including return-value packaging. The latter corrects the reporting boundary without new timing samples or changes to execution. `P/memory_full/` records the separate 102-input memory measurement. These R/P files are in the `revision_sensitivity_cost` group; the remaining revised analyses are in `revision_core`.
