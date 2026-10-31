# Historical joint-correction check and repeated training

This directory preserves the earlier full-pipeline check involving joint correction B. B is not part of the revised CRL/CFS method. The full historical result and its unfavourable comparison are retained for traceability, not relabelled as candidate feedback.

`run_e1.py` and its frozen `PROTOCOL.md` produced the historical full A+B-plus-feedback and conventional-estimation-plus-B-plus-feedback records. Their archived outputs are under `dataset_simulation/revision_20261029/e1_full/`.

`train_e2.py` produced the additional seed-1 and seed-2 response models using the original training protocol. The revised CRL/CFS seed evaluation is implemented in `../revision_af_20261029/`; it does not call B during control. Old B-statistic files and interrupted B-evaluation records retain their historical roles. `continue_e2.py`, `evaluate_e2.py` and `summarize_e2.py` are historical entry points or shared implementation dependencies and must not be mistaken for the current CRL/CFS study.

Source files and original protocol bytes are preserved because experiment records bind their hashes. The repository's current method description and artifact guide determine which results belong to the revised manuscript. Do not launch old queues merely because an archived command exists.
