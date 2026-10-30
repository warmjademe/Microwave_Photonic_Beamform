# Array geometry and matched phase-control diagnostic

This deterministic calculation supports the receiver-model formulation, RQ1 analysis and reply to reviewer 4.4. It compares true time delay with a phase setting that matches the same quantized delays at each centre frequency. It does not train a controller or rerun the receiver benchmark.

The two controls share the 8 × 8 geometry, 37.47405725 mm spacing, equal amplitudes, and delay codes from the nominal receiver. The fixed grid has 17 carriers from 4 to 20 GHz, 861 steering directions, and 33 frequency offsets: 483,021 paired conditions. `protocol.json` records the grid, checks, reporting rules, and exclusions established before execution.

## Files and results

Results are committed under `dataset_simulation/revision_scope_20261030/array/`:

- `array_response.npz`: `carriers_hz` (17), `steering_deg` (861 × 2), `offsets_hz` (33), `delay_codes` (861 × 64), and `normalised_power` (17 × 861 × 33 × 2). The last axis is true delay followed by matched phase.
- `summary.json`: all carrier summaries, spatial replicas, checks, and source hashes.
- `array_ttd_phase.pdf` and `.png`: the array cuts used in the RQ1 frequency analysis.
- `complete.json`: output and executed-source SHA-256 hashes.

The maximum absolute directional-gain difference is 0.157403265 dB. Either control can have the larger gain at individual offsets because frequency-dependent phase interacts with the shared quantization error. High-frequency spatial replicas remain present with true time delay. These results quantify the array response in the specified 100 MHz band; they do not compare complete photonic and electronic receivers or predict multipath BER.

## Reproduction

The executed script is preserved without changing its source hash. It imports the repository's `native_sim.config.NativeConfig` and `baseline_common.controls.direction`, and requires Python, NumPy and Matplotlib (3.8 or later). As an execution safeguard for the original study, it checks for Linux and the project root `/home/qyb/RESEARCH/wangluqiang_2026_paper_1`. Use that checkout location on a Linux machine to reproduce the archived script unchanged. An alternative checkout requires explicitly adapting the path guard; that adaptation changes the source hash but not the mathematical formulas.

From the repository root, use a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source_codes/revision_scope_20261030/array_diagnostic.py --output dataset_simulation/revision_scope_20261030/array_reproduced
```

Completed output directories cannot be overwritten by the script. Checks cover centre-frequency equality, unit broadside gain, analytical spatial replicas, legal delay codes and finite outputs. Numerical arrays can be compared with the archive; timestamps and PDF metadata need not have identical file hashes across runs.

The calculation excludes multipath, laser dynamics, detector noise, hardware loss differences and coding. Existing receiver scores, model weights, dataset partitions and simulator code were not changed.
