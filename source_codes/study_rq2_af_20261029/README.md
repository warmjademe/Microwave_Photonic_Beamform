# A/F factorial ablation

`run.py` evaluates conventional response estimation, learned response A, conventional estimation with feedback F, and A+F. A fifth configuration performs fixed 64-probe scanning. All use the original 864 test environments and 17 carriers; no new training or checkpoint selection is performed.

The direct-output configurations use 16 initial probes. The feedback configurations use 64 probes, including those initial measurements. Consequently, replacing A is a same-budget comparison, whereas removing F also removes 48 measurements. Feedback with budget 16 selects an initially measured control and is not the same procedure as directly executing A's unmeasured candidate.

`report.py` produces the factorial effects, paired environment intervals and selection diagnostics from the completed records. The new A+F controls and integer error counts are checked against the original RQ1 `cnn_warm64` records. The comparison reuses a previously evaluated test set and supports revision and reproducibility analysis rather than a newly sampled confirmation claim.

`PROTOCOL.md` is the byte-preserved protocol bound to the execution. Its original administrative instructions describe the stage when that experiment was authorised; current manuscript and release status are given by the repository README. Historical B is not instantiated by this study.

Raw records and analyses are under `dataset_simulation/revision_rq2_af_20261029/full/`, supplied in the `revision_core` download group. Consult the release manifest for member hashes and attachment locations.
