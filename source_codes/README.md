# Simulation, learning, and evaluation code

The code accompanies the dataset with 3,456 training, 216 validation, and 864 independent test environments, each evaluated at 17 carriers from 4 to 20 GHz. See the [paper-to-artifact guide](../dataset_simulation/PAPER_ARTIFACTS.md) for the raw records and model files associated with each research question.

## Entry points

| Module | Purpose |
|---|---|
| `native_sim/` | Device parameters, waveform generation, discrete control, and common photodetection |
| `diagnostics/` | Numerical support, calibration, and consistency audits |
| `dataset_protocol/` | Split registry, provenance, and seed isolation |
| `baseline_*/` | Nonlearning and supervised baselines, plus auxiliary models required by frozen bundles |
| `our_method_response_control/` | Component A: complex response learning and shared control solving |
| `our_method_joint_refinement/` | Component B: joint measurement correction |
| `our_method_measurement_refinement/` | Training residual statistics used by correction |
| `study_full_baselines/` | Training, validation, training-scale, and hyperparameter studies |
| `study_final864/` | Base final-test execution and paired statistical analysis |
| `study_uniform64/` | Eight additional baseline configurations and the combined 64-probe comparison |
| `study_final_timing/` | Timing replay with control and feedback-trajectory verification |
| `paper_results_20260927/` | Numerical tables, constellation reconstruction, and English paper figures |
| `results_site_3456/build_final.py` | Dashboard built from complete final-test records |
| `release_tools/fetch_artifacts.py` | Artifact download, extraction, and integrity verification |

The main comparison contains 13 baselines and the proposed method at 64 actual probes. The base and extended archives contain 29 configurations across budgets, ablations, and privileged references. [The base protocol](study_final864/PROTOCOL.md) and [the 64-probe extension protocol](study_uniform64/PROTOCOL.md) specify their relationship. Auxiliary code does not imply additional methods in the main ranking.

Each online input has 2,513 real values. Executed controls contain 64 integer delay codes and 64 integer attenuation codes. Control networks learn offline control labels; the response network learns per-branch complex responses. Both use the same observable inputs and training environments.

## Execution environment

The experiments use Linux, Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 1.26.4, and an NVIDIA RTX 4090. Python dependencies are listed in [requirements-deep.txt](requirements-deep.txt). Figure-generation modules also use Matplotlib and CairoSVG; Chinese figure variants require a CJK font. The native simulator requires a C++ compiler.

Frozen runners preserve absolute paths, machine checks, source hashes, and hardlink relationships used in the original experiment. Downloading and inspecting the released arrays is portable. Executing the original training or evaluation runners on another machine requires explicit path/environment adaptation and consistency checks; changing a frozen source file also requires a new run identity. Original protocols and source snapshots are supplied for comparison.

Model training uses one seed (0) and the final checkpoint after 40 epochs. Training-scale experiments additionally hold the optimizer update count fixed. The final-test set does not select checkpoints or component configurations.

## Inspect artifacts without rerunning experiments

From the repository root:

```bash
python3 source_codes/release_tools/fetch_artifacts.py --list
python3 source_codes/release_tools/fetch_artifacts.py --extract
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

These commands use the Python standard library and do not train or simulate. The complete manifest includes both base datasets and paper supplements. Per-condition errors and powers, selected controls, model/checkpoint files, and constellation arrays remain available independently of the dashboard.
