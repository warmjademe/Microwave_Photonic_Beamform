# Current manuscript: code and data map

The method is **Response Learning with Feedback (RLF)**, comprising **Complex Response Learning (CRL)** and **Candidate Feedback Selection (CFS)**. The shared discrete solver translates the response estimate into a candidate. The stored identifiers `A`, `F`, and `cnn_warm64` remain unchanged so that existing records, model bundles and source snapshots can be checked.

## Receiver and control flow

| Step | Current implementation | Input and output |
|---|---|---|
| Simulated receiver | `native_sim/config.py`, `laser.py`, `control_engine.py`, `evaluation.py` | Desired-signal propagation, 64 optical branches and common photodetection |
| Observable measurements | `baseline_common/feedback.py::FeedbackSession` | 16 common initial probes; 2,513 real input values |
| CRL response prediction | `our_method_response_control/model.py`; `physics.py` | Measurements to 64 × 31 complex responses |
| Shared discrete solver | `our_method_response_control/physics.py::decode` | Response to 64 delay codes (0–76) and 64 attenuation codes (0–24) |
| CFS measured selection | `study_uniform64/controllers.py::Feedback64.decide` | Model candidate, then seeded geometric candidates; select the highest measured pilot score |
| Complete RLF controller | `study_rq2_af_20261029/run.py::initialize` | `OnlineController(..., 'complex_response_cnn', ...)` wrapped by `Feedback64` |

The nominal delay step is 19.53125 ps and the attenuation step is 0.5 dB. The main comparison has 64 actual probes including the initial 16. Duplicate controls reuse their recorded score. Independent payload symbols are used only after control selection for BER, SER, BLER, EVM and SNR evaluation. There is no independent interfering transmitter in the nominal dataset.

## RQ1: the 13 baselines

| Baseline | Code directory or controller | Stored main-comparison ID |
|---|---|---|
| Fixed scan | `Feedback64(..., fixed_catalog=True)` | `initial_select64` |
| Geometric codebook | `baseline_codebook` | `codebook` |
| Coordinate search | `baseline_coordinate` | `coordinate` |
| SPSA | `baseline_spsa` | `spsa` |
| DONE | `baseline_done` | `done` |
| Differential evolution | `baseline_de` | `de` |
| MLP | `baseline_mlp` with `Feedback64` | `mlp_feedback64` |
| DNN | `baseline_dnn` with `Feedback64` | `dnn_feedback64` |
| CNN | `baseline_cnn` with `Feedback64` | `cnn_feedback64` |
| ResCNN | `baseline_rescnn` with `Feedback64` | `rescnn_feedback64` |
| Transformer | `baseline_transformer` with `Feedback64` | `transformer_feedback64` |
| Complex CNN | `baseline_complex_cnn` with `Feedback64` | `complex_cnn_feedback64` |
| JCT | `baseline_jct` with `Feedback64` | `jct_feedback64` |

RLF is stored as `cnn_warm64`. All 14 methods use the same 864 test environments and 17 carriers at 64 probes. Read method names from each archive rather than assuming array order. `study_uniform64/analyze.py` combines the base records and the eight extended configurations. Teacher and digital MRC use additional information or different receiver hardware and remain separate references.

## RQ2 and RQ3

`study_rq2_af_20261029/run.py` and `report.py` implement the four-way ablation: conventional response, CRL, conventional response + CFS, and RLF. Fixed scanning is an additional reference. Replacing CRL retains the measurement budget; removing CFS removes the 48 additional probes as well. The RLF controls and error counts match the RQ1 records.

Training-scale and initial search studies are implemented in `study_full_baselines/`. `revision_af_20261029/` contains direct experiment and analysis entry points for repeated training, regularization/noise weighting, candidate count, fixed-trajectory budgets, sensitivity, matched response supervision, software time and memory. These studies use the documented validation/test splits; they do not add new independent environments.

## Data and figures

- [Paper-to-artifact guide](../dataset_simulation/PAPER_ARTIFACTS.md): exact dataset, model and result paths.
- [Dataset guide](../dataset_simulation/README.md): 3,456 training, 216 validation and 864 test environments; release download and verification.
- [Paper_Figures_Generation](Paper_Figures_Generation/README.md): a standalone, checked plotting package for all eight quantitative figures, with source arrays, English/Chinese PDF/PNG output and CSV values.

## Frozen dependencies and cleanup

The scientific Python/C++ implementations, trained models and released experiment records retain their original bytes. Historical correction modules remain dependencies of the frozen runners and source snapshots. They are excluded from RLF, but removing them would break historical replay or evidence inspection. The original protocols retain their original configuration identifiers.

Fifteen unreferenced process queues and temporary dispatchers were removed. They are not numerical training, simulation, control or scoring implementations. Use the direct experiment entry points and their documented arguments. Original runners contain environment/path checks; replay on a different machine requires explicit path adaptation and a new run identity rather than silently changing frozen provenance.
