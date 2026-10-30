# Microwave Photonic Receive Beam Control

Simulation datasets and research code for **Microwave Photonic Receive Beam Control for Beyond-Line-of-Sight Wireless Communications: Complex Response Learning and Candidate Feedback**.

**Revision, 30 October 2026:** the method, ablation and supplementary evidence use complex response learning A and candidate feedback F. Joint correction B is historical evidence, not a component of the revised method. The download manifest combines unchanged base datasets with the completed revision studies.

**Authors:** Luqiang Wang, Tao Pu, Jin Li, Jilin Zheng, and Hua Zhou.

[Original-results dashboard](https://guangzi.qyb.ink/baselines.html) · [Revision data release](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-10-30) · [Paper-to-artifact guide](dataset_simulation/PAPER_ARTIFACTS.md) · [Download manifest](dataset_simulation/RELEASE_DATA.json)

The study addresses receive control when multipath propagation, attenuation, fading, and receiver noise distort a desired wireless signal. A supervised model estimates the complex response of a microwave photonic receiver, and a constrained solver converts that estimate into per-branch delay and optical attenuation settings. The repository provides the simulation and learning code, baseline implementations, and downloadable data, supervision, models, and evaluation records.

The main contributions are a simulation dataset for hardware-constrained receive control, a method combining complex response learning with candidate feedback selection, and shared artifacts for reproducible research.

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

All 64 optical branches are combined before common photodetection. The nominal dataset contains only the desired wireless signal, fixed insertion loss, and receiver noise. It excludes independent interfering transmitters and temperature drift. A separate completed sensitivity study examines static device offsets, laser relative intensity noise, and changes to propagation conditions without retraining the controller.

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

The nominal device parameters come from the supplied OptiSystem project. A parameter audit matched 896 laser entries, five APD parameters and the effective fixed optical loss. The research receiver extends the original inputs, optical detection window, control grid and numerical record length. The primary task concerns propagation impairments with nominal device settings; temperature drift and device-error compensation are separate future studies. All reported reception results are simulations, with physical receiver and coded-protocol evaluation planned separately.

[Appendix A code and protocol](source_codes/revision_scope_20261030/README.md) and [complete array results](dataset_simulation/revision_scope_20261030/array/) compare true delay with centre-frequency-matched phase control over 483,021 deterministic conditions. Their maximum absolute directional-gain difference is 0.1574 dB in the specified 100 MHz band, and high-frequency spatial replicas remain present. This array calculation isolates geometry and control phase; it does not compare complete photonic and electronic hardware or change the receiver benchmark scores.

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
2. **F — Candidate feedback selection:** measure the model candidate and geometric-codebook settings, then select the setting with the highest observed pilot score.

A shared discrete solver maps A's estimated response to legal delay and attenuation codes. F tests that candidate first and spends the remaining probe budget on a fixed seeded geometric-codebook order, reusing measurements of duplicate settings. It returns the highest-scoring measured control, including the initial settings. F does not update network weights or the estimated response. Offline response targets and teacher controls provide training supervision, not additional online inputs.

The main configuration is **A+F with 64 probes**, corresponding to `cnn_warm64` in the frozen code. The direct-output ablation executes A's solver candidate after 16 initial probes and removes F. This differs from retaining F with a budget of 16, which can only choose among initially measured controls. The original B implementation and records remain historical evidence; B is not a component of the revised method.

## Dataset

| Split | Independent propagation environments | Carriers per environment | Environment–carrier records |
|---|---:|---:|---:|
| Training | 3,456 | 17 | 58,752 |
| Validation | 216 | 17 | 3,672 |
| Test, reused in the revision | 864 | 17 | 14,688 |

Sampling uses 216 strata formed by path count, maximum relative delay, angular spread, and reference power. Each stratum contributes 16 training, one validation, and four test environments. Environment seeds are disjoint, and all carriers and noise repetitions of an environment stay in the same split. Frequency records are not counted as independent environments.

All learned methods use the same training environments and observable inputs. Control-regression networks learn teacher controls, whereas component A learns equivalent complex responses. The original weights and operating configurations were selected before their test evaluation. The revised component interpretation and supplementary comparisons reuse the same 864 test environments; these are not a new independent confirmation set. After control selection, an independent payload frame is used for scoring; test payload symbols do not enter control selection or equalizer fitting.

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

Compared with fixed scan, the lowest-BER of the 13 initial baselines, the proposed method reduces BER by **0.3569 percentage points (3.44% relative)**. The environment-paired 95% bootstrap interval for the BER difference is **[−0.5027, −0.2112] percentage points**. It wins in 511 environments, ties in 32, and loses in 321 after aggregation across carriers. The improvement is an overall result, not a claim of winning in every environment or carrier band.

A supplementary real-valued response CNN with matched training supervision, solver, and feedback reaches **10.0671% BER**, compared with **10.0136%** for A+F. The real-minus-complex difference is **0.0535 percentage points**, with paired 95% interval **[0.0291, 0.0774]**. This compares whole estimators: their convolution, activation, and conditioning differ, so it does not isolate one complex-valued operation. Four CNN/Transformer controls trained with a reception-quality objective reach **10.5295%–10.5457% BER** with feedback; their differences from their original control-regression versions have paired intervals that include zero.

The offline teacher accesses internal responses, and ideal digital maximum-ratio combining (MRC) assumes different receive hardware. Both are reference results rather than baselines in the same-information ranking.

### Component ablations

The revised factorial comparison uses all 864 existing test environments and 17 carriers. Replacing A retains the same solver and output procedure. Removing F also removes 48 additional measurements.

| Response/control configuration | Probes | BER (%) ↓ |
|---|---:|---:|
| Conventional covariance response | 16 | 13.7654 |
| A only | 16 | 12.9025 |
| Conventional response + F | 64 | 10.2451 |
| A+F | 64 | 10.0136 |
| Fixed scan | 64 | 10.3705 |

At 64 probes, replacing A with conventional response estimation increases BER by **0.2315 percentage points**, with paired 95% interval **[0.2018, 0.2606]**. Removing F increases BER by **2.8889 percentage points**, with interval **[2.7246, 3.0549]**, while reducing measurements from 64 to 16. These comparisons support A's same-budget contribution and F's benefit together with its added measurement cost. The corresponding SNR intervals include zero, and the BER interaction indicates a smaller absolute benefit of A when F is present rather than a superadditive gain.

Three training seeds retain the BER comparison directions. A+F has mean **9.9941% BER** and between-seed sample standard deviation **0.0181 percentage points**; A alone has mean **12.8619%** and standard deviation **0.0844 percentage points**. Seed 0 remains the primary model, and the three runs use the same test environments.

## Sensitivity and online cost

The 11-condition sensitivity study retains all 864 existing environments and 17 carriers, comparing A+F, conventional estimation+F and fixed scanning at 64 probes. Conditions change relative path delay, angular offsets, desired-signal power, independent laser RIN, static branch delay or static optical loss. All conditions and negative results are retained. A+F-minus-fixed-scan BER intervals are below zero in ten conditions; at RIN −130 dB/Hz the difference is −0.0210 percentage points, with paired 95% interval [−0.1686, 0.1205]. A+F BER rises to 21.6008% there. These are transformations of the existing simulated environments, not independent field data or time-varying-channel tests.

Software timing uses 102 inputs chosen without quality scores, two warm-ups and three uninstrumented repetitions per input. Controls and complete feedback traces match the saved experiment. The full controller-call boundary includes result packaging and excludes measurement-replay callbacks, model loading and external correctness checks.

| Controller | Probes | Mean software time (ms) | Median (ms) | 95th percentile (ms) |
|---|---:|---:|---:|---:|
| A only | 16 | 16.358 | 16.331 | 16.761 |
| A+F | 64 | 28.551 | 28.503 | 28.943 |
| Conventional estimation+F | 64 | 30.550 | 30.523 | 30.968 |
| Fixed scan | 64 | 8.187 | 8.150 | 8.381 |

Separate instrumented runs give A+F response estimation 1.746 ms, including 1.397 ms network forward, discrete solving 14.480 ms, and other control operations 12.329 ms. The network time is nested, not added twice. Assuming 1.24 microseconds acquisition and 1 microsecond switching per probe, plus one final switch, adds 0.14436 ms at 64 probes. Software plus that assumption is 28.695 ms, excluding unmeasured device communication and settling; this is not hardware-loop latency.

The 17,540 parameters occupy 70,160 bytes. A separate single-controller process reaches 1005.4 MiB peak RSS, 33.1 MiB CUDA peak allocated memory and 34.0 MiB reserved memory; the largest sampled driver-reported usage is 518 MiB. Process/library overhead and driver/context storage differ from parameter storage. Reception benefit must be weighed against this implementation's computation and the required channel-stability interval.

The [original-release dashboard](https://guangzi.qyb.ink/baselines.html) provides the original full-test results. Its [JSON](https://guangzi.qyb.ink/results-20261001.json) and [CSV](https://guangzi.qyb.ink/results-20261001.csv) identify that frozen version; they do not yet include the revision supplements described above. Use the revision release and artifact guide for the current A/F studies.

## Download the data and models

Source code is stored in the main branch. The complete download manifest accompanies [release `data-2026-10-30`](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-10-30).

**Artifact coverage:** the manifest includes training, validation, and test data; control and response supervision; frozen models; raw records for all 13 baselines at 64 probes; A/F ablations; repeated training; training-scale and hyperparameter studies; constellation arrays; sensitivity, timing and memory records. Original B code and statistics remain historical evidence. All eight previous download groups are retained byte-for-byte; the revision adds `revision_core` and `revision_sensitivity_cost`. **The downloader retrieves their dependencies from the corresponding releases automatically.** The [paper-to-artifact guide](dataset_simulation/PAPER_ARTIFACTS.md) identifies the records and code for each research question.

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
| `revision_core` | A/F ablation, seeds, hyperparameters, training objectives, matched response estimator, and historical B evidence | See manifest |
| `revision_sensitivity_cost` | All 11 sensitivity conditions, raw formal timing, full-call timing analysis, and memory measurements | See manifest |

Archives are split into parts of at most 64 MiB. Allow additional space for extraction and cached archives. Exact sizes and checksums are listed in [RELEASE_DATA.json](dataset_simulation/RELEASE_DATA.json). Supporting exploratory and failed-run records retain their original status and are separate from final results. The revision archives have been extracted through the public reader and checked against every indexed member hash.

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
| [source_codes/our_method_joint_refinement/](source_codes/our_method_joint_refinement/) | Historical B implementation; excluded from the revised method |
| [source_codes/study_full_baselines/](source_codes/study_full_baselines/) | Shared training, validation, and hyperparameter studies |
| [source_codes/study_final864/](source_codes/study_final864/) | Released full-test runner and statistical analysis |
| [source_codes/study_uniform64/](source_codes/study_uniform64/) | Candidate feedback F and all-baseline comparison at 64 actual probes |
| [source_codes/study_final_timing/](source_codes/study_final_timing/) | Controlled timing replay and audits |
| [source_codes/paper_results_20260927/](source_codes/paper_results_20260927/) | Paper tables, figures, and constellation reconstruction |
| [source_codes/release_tools/](source_codes/release_tools/) | Artifact download and integrity verification |

Experiments use Linux, Python 3.11.16, PyTorch 2.8.0+cu128, NumPy 1.26.4, and an NVIDIA RTX 4090. Dependencies are recorded in [requirements-deep.txt](source_codes/requirements-deep.txt). Primary networks use seed 0 and the final checkpoint after 40 epochs. The repeated-training study evaluates A with seeds 0, 1, and 2. Quality-objective controls receive 40 quality-training epochs; models initialized from their original MSE checkpoint have 80 total training epochs. No checkpoint is selected on final-test performance.

The released experiment runners preserve absolute paths, machine checks, and artifact identity checks from the frozen environment. Running them on another system requires adapting those execution constraints and checking numerical consistency. Downloading and checking file integrity is independent of the original experiment machine. The evidence concerns the simulated receiver and evaluated statistical channels; the three training seeds describe limited retraining variation, while measured hardware and propagation remain separate validation tasks.

## Contact

Corresponding author: **Tao Pu**, [nj_putao@163.com](mailto:nj_putao@163.com). Use [GitHub issues](https://github.com/warmjademe/Microwave_Photonic_Beamform/issues) for questions about the code or released artifacts.
