# Simulation data, supervision, and experimental records

The dataset models a desired signal subject to multipath, attenuation, fading, antenna noise, and photodetection noise. It includes fixed insertion loss but no independent interferer, temperature drift, or additional random device mismatch. See the [paper-to-artifact guide](PAPER_ARTIFACTS.md) for the exact files supporting each research question.

## Independent splits

| Role | Environments | Environment–carrier records | Input location |
|---|---:|---:|---|
| Training | 3,456 | 58,752 | `outputs/scaling_train_3456_20260925/train/` |
| Validation | 216 | 3,672 | `outputs/quality_rank_hybrid_20260925/test/` |
| Final test | 864 | 14,688 | `baseline_results/20260926_final864_selected/records/` |

Each environment covers 17 carriers from 4 to 20 GHz. Membership and seed separation are recorded in `ops/dataset_split_20260926/registry.json`. The 216-environment folder named `test` is the validation split; it is not the independent final test set. Shared views, nested training subsets, and repeated noise draws do not add independent environments.

## Stored arrays and their roles

- **Observable input:** each training `environment_XXXXX/data.npz` stores `X` as `float32[17, 2513]`. It contains combined pilot I/Q from 16 probes, carrier frequency, pilot quality, and permitted receiver calibration statistics. Fields such as `single_nmse` and `robust_nmse` are offline candidate-evaluation information, not extra online inputs.
- **Control supervision:** `baseline_results/20260925_full_baselines/fair_3456/teacher_labels/records/` contains `control_code` as `int16[17, 128]`. The first 64 columns are delay codes 0–76; the remaining 64 are optical attenuation codes 0–24. Multiply by 19.53125 ps and 0.5 dB, respectively, to obtain physical settings.
- **Response supervision:** `baseline_results/20260925_full_baselines/scale_3456/targets/train_response.npy` is `complex64[58752, 64, 31]`. The corresponding `fair_3456/targets/` view serves the same training cohort. Match records using the protocols and environment manifests.
- **Base test records:** each `carrier_XX.npz` in `20260926_final864_selected/records/` includes `public_X[2513]`, `control_code[21, 128]`, `metrics[21, 13]`, and applicable feedback traces.
- **64-probe extension:** `20260927_uniform64_all13/records/` adds eight configurations over the same environments and carriers. Its analysis combines the 13 baselines and proposed method at the common budget.

Always read each archive's `protocol.json` for method names, metric order, and provenance. True propagation parameters are used for generation and offline supervision. Test payload symbols are used for scoring after control selection. Neither is additional input to the deployed controller.

## Models and result records

The frozen runtime bundle is `baseline_results/20260925_full_baselines/fair_3456/runtime_bundle/`. Component A uses the model from `scale_3456/response_n3456_fixed_epochs/`. Component B uses fitted statistics from `diagnostics/20260926_joint_refinement_train3456/` and `diagnostics/20260926_measurement_refinement_fit_3456/`.

The base and extended test records together cover 29 configurations and **425,952 method–condition evaluations**. These include two probe budgets, necessary ablations, and the privileged teacher/digital MRC references. They are not 29 distinct baselines. The paper's main comparison uses 13 baselines plus the proposed method, each with 64 probes.

The supplements also include the nine distinct training-scale models and their checkpoints, raw validation results, transmitted/received constellation arrays, and the 102-input timing study. Failed or exploratory runs retain their original status and do not enter the final ranking.

## Download and verify

From the repository root:

```bash
python3 source_codes/release_tools/fetch_artifacts.py --list
python3 source_codes/release_tools/fetch_artifacts.py --extract
python3 source_codes/release_tools/fetch_artifacts.py --verify-only
```

[Release `data-2026-10-01`](https://github.com/warmjademe/Microwave_Photonic_Beamform/releases/tag/data-2026-10-01) provides the complete manifest. Unchanged base assets remain in `data-2026-09-26`; the downloader resolves both releases automatically. The full download is approximately 7.87 GiB, with additional disk space needed for extraction and cached archives.

[RELEASE_DATA.json](RELEASE_DATA.json) records asset locations, sizes, and SHA-256 digests. Compressed indices in `release_indices/` identify every restored file. Downloads preserve the original relative paths and required model-bundle hardlinks. Simulator installers and license materials are not part of the research dataset.
