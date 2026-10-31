# Complex Response Learning (CRL) and shared discrete control solving

This module implements component CRL of the revised method. Candidate feedback CFS is implemented separately by `study_uniform64/controllers.py::Feedback64`. The complete path is shared pilot observations → CRL → shared discrete solver → CFS → receive-branch controls. Joint correction B is not used in this path.

Sixteen initial settings provide the same 2,513-value observation to each estimator. Regularised inversion, covariance estimation and the complex CNN estimate branch amplitudes and frequency-dependent phases. The same discrete solver searches legal delay and attenuation values. The CNN receives neither true test-channel parameters nor reference payload symbols; candidate feedback subsequently measures actual pilot quality.

The output consists of 64 delay codes in 0–76 and 64 attenuation codes in 0–24. Their steps are 19.53125 ps and 0.5 dB. These integer codes are not a direct six-switch hardware command. An independent payload frame is evaluated with the shared hybrid receiver after control selection.

- `PROTOCOL.md`: byte-preserved original response-learning experiment protocol.
- `physics.py`: pilot extraction, conventional estimators, noise calculation and discrete control search.
- `model.py`: the complex CNN; `prepare.py`: training response labels and covariance statistics.
- `train.py`: original seed-0, 40-epoch training; `evaluate.py`: independent-payload evaluation.
- `audit.py`: public-input replay and exact control-code checks.
- `test_model.py`, `test_noise.py`, `preflight.py`: mechanism, noise and pipeline checks.
- `run_when_ready.py`: historical experiment orchestration, not the revision's current queue.

The following commands document the original experiment from `source_codes/`; their data paths and scale are historical. Current CRL/CFS experiments and source mappings are described in [the revision directory](../revision_af_20261029/README.md). A fresh run requires unused output directories and the original environment checks or an explicit, verified adaptation.

```bash
../.venv_dl/bin/python our_method_response_control/train.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --output ../dataset_simulation/baseline_results/20260925_response_control
../.venv_dl/bin/python our_method_response_control/evaluate.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --run-root ../dataset_simulation/baseline_results/20260925_response_control --workers 6
../.venv_dl/bin/python our_method_response_control/audit.py --data ../dataset_simulation/outputs/quality_rank_hybrid_20260925 --targets ../dataset_simulation/diagnostics/20260925_response_control_targets --run-root ../dataset_simulation/baseline_results/20260925_response_control --workers 6
```

Do not restart an existing queue. Repeated noise draws measure receiver variability and are distinct from retraining with different seeds. The revision reports CRL at seeds 0, 1 and 2, retaining seed 0 for the primary comparison.

The linearised response is a control surrogate, while the final hybrid receiver retains nonlinear computation for branches exceeding its frozen state thresholds. The revision measures total software time separately from instrumented components and from assumed pilot-acquisition and switching costs. Measured-device latency and hardware deployment remain separate validation tasks.
