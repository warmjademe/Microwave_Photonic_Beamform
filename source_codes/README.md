# Simulation, learning, and evaluation code

The code accompanies the dataset with 3,456 training, 216 validation, and 864 independent test environments, each evaluated at 17 carriers from 4 to 20 GHz. See the [paper-to-artifact guide](../dataset_simulation/PAPER_ARTIFACTS.md) for the raw records and model files associated with each research question.

**Current method:** RLF combines Complex Response Learning (CRL) followed by a shared discrete solver and Candidate Feedback Selection (CFS). The release manifest preserves all original data groups and adds the completed revision experiments; historical joint correction B is not part of the current method.

## Entry points

| Module | Purpose |
|---|---|
| `native_sim/` | Device parameters, waveform generation, discrete control, and common photodetection |
| `diagnostics/` | Numerical support, calibration, and consistency audits |
| `dataset_protocol/` | Split registry, provenance, and seed isolation |
| `baseline_*/` | Nonlearning and supervised baselines, plus auxiliary models required by frozen bundles |
| `our_method_response_control/` | Component CRL: complex response learning and shared control solving |
| `our_method_joint_refinement/` | Historical B implementation, excluded from the revised method |
| `our_method_measurement_refinement/` | Historical correction statistics and frozen-run dependencies |
| `study_full_baselines/` | Training, validation, training-scale, and hyperparameter studies |
| `study_final864/` | Base final-test execution and paired statistical analysis |
| `study_uniform64/` | Candidate feedback CFS, eight additional baseline configurations, and the combined 64-probe comparison |
| `study_final_timing/` | Timing replay with control and feedback-trajectory verification |
| `paper_results_20260927/` | Numerical tables, constellation reconstruction, and English paper figures |
| `results_site_3456/build_final.py` | Dashboard built from complete final-test records |
| `release_tools/fetch_artifacts.py` | Artifact download, extraction, and integrity verification |

The main comparison contains 13 baselines and the proposed method at 64 actual probes. The base and extended archives contain 29 configurations across budgets, ablations, and privileged references. [The base protocol](study_final864/PROTOCOL.md) and [the 64-probe extension protocol](study_uniform64/PROTOCOL.md) specify their relationship. Auxiliary code does not imply additional methods in the main ranking.

Each online input has 2,513 real values. Executed controls contain 64 integer delay codes and 64 integer attenuation codes. Control networks learn offline control labels; the response network learns per-branch complex responses. Both use the same observable inputs and training environments.

CRL is implemented in `our_method_response_control/model.py`; its output enters `physics.py::decode`. CFS is implemented by `study_uniform64/controllers.py::Feedback64` and uses `baseline_common/feedback.py::FeedbackSession` for measured-score selection. The frozen main configuration is `cnn_warm64`. The revised factorial ablation compares conventional response estimation, CRL, conventional estimation+CFS, and RLF; fixed 64-probe scanning is an additional reference. The two direct-output configurations use 16 probes, while CFS uses 64. Historical B code is retained to read and reproduce earlier records, not renamed as CFS.

## Execution environment

The experiments use Linux, Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 1.26.4, and an NVIDIA RTX 4090. Python dependencies are listed in [requirements-deep.txt](requirements-deep.txt). Figure-generation modules also use Matplotlib and CairoSVG; Chinese figure variants require a CJK font. The native simulator requires a C++ compiler.

Frozen runners preserve absolute paths, machine checks, source hashes, and hardlink relationships used in the original experiment. Downloading and inspecting the released arrays is portable. Executing the original training or evaluation runners on another machine requires explicit path/environment adaptation and consistency checks; changing a frozen source file also requires a new run identity. Original protocols and source snapshots are supplied for comparison.

Primary models use seed 0 and the final checkpoint after 40 epochs. The revised repeated-training study evaluates CRL at seeds 0, 1, and 2. Quality-objective CNN/Transformer controls receive 40 additional quality-training epochs; MSE-initialized controls therefore have 80 total epochs. Training-scale experiments additionally hold the optimizer update count fixed. The original weights and configurations preceded their test evaluation. Revised component interpretation and supplementary comparisons reuse the same test environments and are partly retrospective; no test-selected checkpoint or best-seed replacement is introduced.

## Inspect artifacts without rerunning experiments

From the repository root:

```bash
python3 source_codes/release_tools/fetch_artifacts.py --list
python3 source_codes/release_tools/fetch_artifacts.py --extract
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

These commands use the Python standard library and do not train or simulate. The complete manifest includes both base datasets and paper supplements. Per-condition errors and powers, selected controls, model/checkpoint files, and constellation arrays remain available independently of the dashboard.

## Current paper entry points

See [CURRENT_PAPER.md](CURRENT_PAPER.md) for the current names RLF/CRL/CFS, their internal IDs, the 13 baselines, ablations and parameter studies. [Paper_Figures_Generation](Paper_Figures_Generation/README.md) is the single portable entry for the eight quantitative manuscript figures. Retired process queues are excluded; simulation, controller, training and scoring files bound to frozen records remain unchanged.
