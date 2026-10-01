# Microwave Photonic Receive Beam Control

Simulation datasets and research code for **Microwave Photonic Receive Beam Control for Beyond-Line-of-Sight Wireless Communications: Complex Response Learning and Joint Measurement Correction**.

**Authors:** Luqiang Wang, Tao Pu, Jin Li, Jilin Zheng, and Hua Zhou.

[Results dashboard](https://guangzi.qyb.ink/baselines.html) · [Data and model release](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-10-01) · [Paper-to-artifact guide](dataset_simulation/PAPER_ARTIFACTS.md) · [Download manifest](dataset_simulation/RELEASE_DATA.json)

The study addresses receive control when multipath propagation, attenuation, fading, and receiver noise distort a desired wireless signal. A supervised model estimates the complex response of a microwave photonic receiver, and a constrained solver converts that estimate into per-branch delay and optical attenuation settings. The repository provides the simulation and learning code, baseline implementations, and downloadable data, supervision, models, and evaluation records.

The main contributions are a simulation dataset for hardware-constrained receive control, a method combining complex response learning with joint measurement correction, and shared artifacts for reproducible research.

## Communication and receiver model

The simulated receive architecture follows the directly modulated laser, true-time-delay, optical attenuation, optical combining, and photodetection chain described by Wang et al. in *Advanced Over-the-horizon Communications with Microwave Photonic Technologies (Invited)*, Acta Photonica Sinica, 2026, 55(3), 0355110 ([DOI](https://doi.org/10.3788/gzxb20265503.0355110)).

```text
Transmitted QPSK–OFDM signal
    → multipath propagation, attenuation, fading, and receiver noise
    → 8 × 8 receive array: 64 electrical inputs
    → 64 directly modulated laser branches
    → per-branch true time delay and optical attenuation
    → total optical combining of all 64 branches
    → one common avalanche photodiode (APD)
    → demodulation, equalization, and payload evaluation
```

All 64 optical branches are combined before common photodetection. The scenario contains only the desired signal; independent interfering transmitters, temperature drift, and additional random device mismatch are outside the current model. Fixed insertion loss and receiver noise are included.

| Parameter | Setting |
|---|---|
| Receive array | 8 × 8 elements; fixed spacing of approximately 37.4741 mm |
| Carrier frequencies | 4, 5, …, 20 GHz: 17 separate operating conditions |
| Nominal signal bandwidth | 100 MHz per carrier |
| Waveform | QPSK–OFDM; 31 active subcarriers; two pilot blocks and one payload block |
| Number of propagation paths | 4, 8, 16, or 24 |
| Maximum relative path delay | 50, 100, or 200 ns |
| Azimuth angular spread | Standard deviation of 1°, 3°, or 6°; elevation spread is half this value |
| Nominal reference received power | −105 to −75 dBm, stratified into six 5 dB bins |
| Delay control | 64 integer codes in 0–76; 19.53125 ps per code step |
| Optical attenuation control | 64 integer codes in 0–24; 0.5 dB per code step |
| Fixed optical insertion loss | 5 dB from branch and total-combiner losses |

Array spacing remains fixed across carriers. The 17 carrier conditions are distinct from the 31 frequencies within each signal band. The dataset uses a statistical propagation model and a calibrated research receiver model; calibration covers the compared device nodes and records, rather than every operating condition of a complete optical frequency-conversion system.

## Method and input/output interface

The controller uses combined pilot measurements under known probe settings. It does not require direct digital access to all 64 antenna signals or true propagation parameters.

| Quantity | Per-carrier representation | Role |
|---|---|---|
| Observable input `X` | 2,513 real values | Combined pilot I/Q, carrier frequency, pilot scores, APD noise statistics, and DC optical power |
| Response supervision | 64 × 31 complex values | Equivalent response of each branch at each active frequency; available during training |
| Control supervision | 128 integer codes | Offline teacher targets for supervised control-regression baselines |
| Executed control | 64 delay codes + 64 attenuation codes | Legal settings applied to the simulated receiver |

The input contains 1,984 pilot I/Q values from 16 probes, 31 frequencies, and two pilot blocks, plus one carrier value, 16 pilot scores, 496 noise-variance values, and 16 optical-power values. Noise and power statistics use the simulator's device calibration interface; a physical implementation would require corresponding receiver calibration or monitoring.

The method has two components:

1. **A — Complex response learning:** learn the spatial and frequency structure of the equivalent response from the shared observations.
2. **B — Joint measurement correction:** refine the response estimate using current measurements and residual statistics fitted on training data.

A shared discrete solver maps the estimated response to legal delay and attenuation codes. When further probes are available, a common candidate-feedback procedure measures proposed controls and chooses the one with the highest measured pilot score. Offline response targets and teacher controls are supervision, not additional online inputs.

The paper evaluates two configurations selected on validation data: **A+B with direct output at 16 probes**, and **A with candidate feedback at 64 probes**. The reported 64-probe result uses the latter configuration; B is evaluated through the direct-output component ablations below. Probe budget is an experimental hyperparameter.

## Dataset

| Split | Independent propagation environments | Carriers per environment | Environment–carrier records |
|---|---:|---:|---:|
| Training | 3,456 | 17 | 58,752 |
| Validation | 216 | 17 | 3,672 |
| Independent test | 864 | 17 | 14,688 |

Sampling uses 216 strata formed by path count, maximum relative delay, angular spread, and reference power. Each stratum contributes 16 training, one validation, and four test environments. Environment seeds are disjoint, and all carriers and noise repetitions of an environment stay in the same split. Frequency records are not counted as independent environments.

All learned methods use the same training environments and observable inputs. Control-regression networks learn teacher controls, whereas component A learns equivalent complex responses. Model selection uses the validation split. After control selection, an independent payload frame is used for scoring; test payload symbols do not enter control selection or equalizer fitting.

## Main results: 13 baselines at 64 probes

The main comparison includes six nonlearning strategies—fixed scan, geometric codebook, coordinate search, simultaneous perturbation stochastic approximation (SPSA), DONE, and differential evolution (DE)—and seven supervised networks: MLP, DNN, CNN, ResCNN, Transformer, complex CNN, and a joint CNN–Transformer (JCT) adaptation.

All methods are evaluated on the same 864 test environments and 17 carriers, using 64 actual probes: 16 common initial measurements and 48 additional measurements. The seven control networks and the proposed method share candidate ordering, deduplication, and measured-score selection for the added feedback. Network weights remain fixed.

| Method | BER (%) ↓ | SER (%) ↓ | BLER (%) ↓ | EVM (%) ↓ | Output SNR (dB) ↑ |
|---|---:|---:|---:|---:|---:|
| Fixed scan | 10.3705 | 17.0532 | 54.8169 | 60.1493 | 25.4530 |
| Geometric codebook | 10.4813 | 17.2334 | 55.0500 | 60.4077 | 25.3999 |
| Coordinate search | 16.6829 | 26.8450 | 69.1457 | 76.0784 | 23.3626 |
| SPSA | 16.2792 | 26.2529 | 68.1134 | 74.0961 | 23.6413 |
| DONE | 16.1035 | 25.9752 | 67.8198 | 74.0780 | 24.0099 |
| DE | 15.1392 | 24.5039 | 66.1952 | 72.9201 | 23.7382 |
| MLP + feedback | 10.5384 | 17.3017 | 55.2568 | 60.6193 | 25.3514 |
| DNN + feedback | 10.5462 | 17.3126 | 55.2585 | 60.6381 | 25.3581 |
| CNN + feedback | 10.5445 | 17.3077 | 55.2551 | 60.6411 | 25.3584 |
| ResCNN + feedback | 10.5438 | 17.3096 | 55.2688 | 60.6393 | 25.3583 |
| Transformer + feedback | 10.5438 | 17.3082 | 55.2671 | 60.6236 | 25.3570 |
| Complex CNN + feedback | 10.5466 | 17.3125 | 55.2577 | 60.6471 | 25.3544 |
| JCT + feedback | 10.5445 | 17.3110 | 55.2543 | 60.6324 | 25.3581 |
| **Proposed: A + feedback** | **10.0136** | **16.4256** | **52.2723** | **58.8287** | **25.7133** |

BER, SER, and BLER denote bit, symbol, and block error rates. EVM is root-mean-square error vector magnitude after equalization; paired output SNR is measured before equalization. These measure different aspects of reception and should not be treated as interchangeable metrics.

Compared with fixed scan, the lowest-BER baseline, the proposed method reduces BER by **0.3569 percentage points (3.44% relative)**. The environment-paired 95% bootstrap interval for the BER difference is **[−0.5027, −0.2112] percentage points**. It wins in 511 environments, ties in 32, and loses in 321 after aggregation across carriers. The improvement is an overall result, not a claim of winning in every environment or carrier band.

The offline teacher accesses internal responses, and ideal digital maximum-ratio combining (MRC) assumes different receive hardware. Both are reference results rather than baselines in the same-information ranking.

### Component ablations

The direct-output comparison holds the budget at 16 probes and uses the full independent test set.

| Response/control configuration | BER (%) ↓ |
|---|---:|
| Conventional covariance response | 13.7654 |
| A only | 12.9025 |
| Conventional response + B | 13.5856 |
| A+B | 12.3131 |

Removing A or B from A+B increases BER by 1.2726 or 0.5894 percentage points, respectively. At 64 probes, learned-response candidate generation achieves 10.0136% BER versus 10.2451% for conventional-response candidate generation under the same feedback procedure.

The [dashboard](https://guangzi.qyb.ink/baselines.html) provides full-test metrics, paired comparisons, carrier and power groups, constellation examples, and timing measurements. Machine-readable summaries are available as [JSON](https://guangzi.qyb.ink/results-20261001.json) and [CSV](https://guangzi.qyb.ink/results-20261001.csv).

## Download the data and models

Source code is stored in the main branch. The complete download manifest is published with [release `data-2026-10-01`](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-10-01).

**Artifact coverage:** the manifest includes training, validation, and final-test data; control and response supervision; frozen models and correction statistics; raw records for all 13 baselines at 64 probes; component ablations; training-scale and hyperparameter studies; constellation arrays; and timing records. Unchanged base archives remain in `data-2026-09-26`, while the new release supplies the supplements. **The downloader retrieves both automatically.** The [paper-to-artifact guide](dataset_simulation/PAPER_ARTIFACTS.md) identifies the exact records and code for each research question.

```bash
git clone https://github.com/warmjademe/Microwave_Photonic_Beamform.git
cd Microwave_Photonic_Beamform

# Inspect available artifact groups without downloading.
python3 source_codes/release_tools/fetch_artifacts.py --list

# Download, verify, extract, and check all released files.
python3 source_codes/release_tools/fetch_artifacts.py --extract

# Recheck the extracted files without network access.
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

The downloader needs Python 3.9 or newer and only uses the standard library. It verifies SHA-256 checksums for download parts, reconstructed archives, and extracted files. Completed parts can be reused after an interrupted download. It does not run simulation or training.

| Artifact group | Contents | Compressed size |
|---|---|---:|
| `train_validation` | Training and validation inputs, environment manifests, and split metadata | 600.3 MiB |
| `supervision` | Control labels and complex response targets | 1,674.1 MiB |
| `models` | Frozen model weights and fitted correction statistics | 566.3 MiB |
| `final_test` | Released full-test records, controls, feedback traces, and summaries | 543.1 MiB |
| `evidence` | Supporting validation and experiment records | 220.8 MiB |
| `uniform64` | Additional full-test records for all baselines at 64 probes and combined analysis | 430.2 MiB |
| `training_studies` | Complete scale-study inputs, targets, checkpoints, and validation records | 3,750.0 MiB |
| `paper_evidence` | Constellation arrays, plot data, timing records, and supporting checks | 269.0 MiB |

The complete download totals approximately **7.87 GiB**. Archives are split into parts of at most 64 MiB. Allow additional space for extraction and cached archives. Exact sizes and checksums are listed in [RELEASE_DATA.json](dataset_simulation/RELEASE_DATA.json). Supporting exploratory and failed-run records retain their original status and are separate from the final results.

To download only training inputs and supervision:

```bash
python3 source_codes/release_tools/fetch_artifacts.py --only train_validation supervision --extract
python3 source_codes/release_tools/fetch_artifacts.py --only train_validation supervision --verify-only
```

### Released data layout

Paths below are relative to `dataset_simulation/` after extraction.

| Data | Path |
|---|---|
| Training inputs | `outputs/scaling_train_3456_20260925/train/` |
| Validation inputs | `outputs/quality_rank_hybrid_20260925/test/` |
| Split registry | `ops/dataset_split_20260926/registry.json` |
| Teacher control labels | `baseline_results/20260925_full_baselines/fair_3456/teacher_labels/records/` |
| Training response targets | `baseline_results/20260925_full_baselines/scale_3456/targets/train_response.npy` |
| Frozen runtime bundle | `baseline_results/20260925_full_baselines/fair_3456/runtime_bundle/` |
| Released test records | `baseline_results/20260926_final864_selected/records/` |
| Additional 64-probe test records | `baseline_results/20260927_uniform64_all13/records/` |
| Combined all-baseline test analysis | `baseline_results/20260927_uniform64_all13/analysis/` |

Each training environment stores `X` as `float32[17, 2513]`; its control labels have shape `int16[17, 128]`. The response target array is `complex64[58752, 64, 31]`. Use the manifests to align inputs and targets, rather than filesystem traversal order. The directory named `quality_rank_hybrid_20260925/test` contains the **216 validation environments**, not the final test split; roles are defined by the split registry.

The base test archive contains `public_X[2513]`, `control_code[21, 128]`, and `metrics[21, 13]` per carrier. The `uniform64` supplement adds eight baseline configurations over the same 14,688 conditions, yielding 29 configurations across budgets, ablations, and references—not 29 different baseline strategies. Method and metric order come from the associated protocols. See the [data guide](dataset_simulation/README.md) for field and provenance details.

## Code organization and execution environment

| Path | Purpose |
|---|---|
| [source_codes/native_sim/](source_codes/native_sim/) | Receiver model, signal generation, and control computation |
| [source_codes/diagnostics/](source_codes/diagnostics/) | Numerical support and calibration routines |
| [source_codes/dataset_protocol/](source_codes/dataset_protocol/) | Split registration, provenance, and seed checks |
| `source_codes/baseline_*/` | Nonlearning and supervised baseline implementations |
| [source_codes/our_method_response_control/](source_codes/our_method_response_control/) | Component A and shared control solving |
| [source_codes/our_method_joint_refinement/](source_codes/our_method_joint_refinement/) | Component B |
| [source_codes/study_full_baselines/](source_codes/study_full_baselines/) | Shared training, validation, and hyperparameter studies |
| [source_codes/study_final864/](source_codes/study_final864/) | Released full-test runner and statistical analysis |
| [source_codes/study_uniform64/](source_codes/study_uniform64/) | All-baseline comparison at 64 actual probes |
| [source_codes/study_final_timing/](source_codes/study_final_timing/) | Controlled timing replay and audits |
| [source_codes/paper_results_20260927/](source_codes/paper_results_20260927/) | Paper tables, figures, and constellation reconstruction |
| [source_codes/release_tools/](source_codes/release_tools/) | Artifact download and integrity verification |

Experiments use Linux, Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 1.26.4, and an NVIDIA RTX 4090. Dependencies are recorded in [requirements-deep.txt](source_codes/requirements-deep.txt). Neural networks use seed 0 and the final checkpoint after 40 epochs; no checkpoint is selected on final-test performance.

The released experiment runners preserve absolute paths, machine checks, and artifact identity checks from the frozen environment. Running them on another system requires adapting those execution constraints and checking numerical consistency. Downloading and checking file integrity is independent of the original experiment machine. Receiver calibration, simulation evidence, and one trained seed do not establish physical-deployment performance or variability across retraining runs.

## Contact

Corresponding author: **Tao Pu**, [nj_putao@163.com](mailto:nj_putao@163.com). Use [GitHub issues](https://github.com/warmjademe/Microwave_Photonic_Beamform/issues) for questions about the code or released artifacts.
