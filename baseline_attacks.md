# Baseline Attack Pipeline Log

## Objective
Generate legitimate baseline adversarial samples for CICIoT2023 using torchattacks with:
- PGD attack
- C&W (Carlini-Wagner) attack

Required outputs:
- Terminal preview of adversarial samples
- Text artifact: generated_baseline_samples.txt
- Saved arrays and metrics for reproducibility

## Scope and Design
- Script: baseline_attacks.py
- Dataset source: data/processed (derived from data/ciciot2023/ciciot2023_base.csv preprocessing)
- Victim model: models/nids/nids_cnnlstm.pt (fallback: nids_cnnlstm_best.pt)
- Label source: models/nids/label_encoder.pkl (fallback: data/processed/label_encoder.pkl)

## Engineering Choices
1. Reproducibility
- Fixed seed for Python, NumPy, and PyTorch.
- Deterministic attack config captured in JSON report.

2. Input Domain Correctness
- Existing model expects standardized tabular features, while torchattacks is designed around [0,1] bounded inputs.
- Implemented a wrapper model that attacks in unit space and maps back to standardized space before inference.
- This preserves model compatibility and keeps perturbation bounds meaningful.

3. Legitimacy Checks
- Input NaN/Inf validation before attack.
- Output NaN/Inf checks for each adversarial batch.
- Metrics include:
  - clean accuracy on evaluated subset
  - ASR on all samples
  - ASR on clean-correct samples
  - average L2 and Linf perturbation magnitudes

4. Artifacts
- results/baseline/x_clean_unit.npy
- results/baseline/adv_pgd_unit.npy
- results/baseline/adv_cw_unit.npy
- results/baseline/y_test_encoded.npy
- results/baseline/baseline_attack_report.json
- generated_baseline_samples.txt

## Progress Tracker
- [x] Reviewed existing model and preprocessing code
- [x] Implemented baseline_attacks.py with PGD and C&W (torchattacks)
- [x] Installed/verified torchattacks runtime dependency
- [x] Executed pipeline and generated artifacts
- [x] Logged execution metrics and sample previews

## Latest Run Summary (2026-04-14)
- Command:
  python baseline_attacks.py --n-samples 64 --batch-size 32 --preview-count 5
- Device: cuda
- Samples attacked: 64
- PGD metrics:
  clean_acc=0.9375, asr_all=0.8906, asr_clean_correct=0.8833, l2_mean=0.31816, linf_mean=0.08000
- CW metrics:
  clean_acc=0.9375, asr_all=0.7656, asr_clean_correct=0.7500, l2_mean=0.06712, linf_mean=0.04863

## Implementation Notes from Debugging
- torchattacks uses gradient backpropagation through the LSTM during attack generation.
- cuDNN RNN backward in eval mode raised a runtime error.
- Resolution: run attack steps with cuDNN disabled inside the attack loop, keeping inference behavior unchanged.

## Latest Stratified Run Summary (2026-04-14)
- Command:
  python baseline_attacks.py --n-samples 1280 --batch-size 128 --preview-count 3
- Sampling policy:
  Seeded stratified selection in load_data (balanced per class when feasible)
- Observed class counts in saved y_test_encoded.npy:
  Benign=256, DDoS=256, DoS=256, Mirai=256, Recon=256
- Metrics:
  PGD: clean_acc=0.8453, asr_all=0.8961, asr_clean_correct=0.8808, l2_mean=0.31490, linf_mean=0.08000
  CW:  clean_acc=0.8453, asr_all=0.7797, asr_clean_correct=0.7394, l2_mean=0.08054, linf_mean=0.05359

## Larger Stratified Run Summary (2026-04-14)
- Command:
  python baseline_attacks.py --n-samples 2500 --batch-size 128 --preview-count 3
- Observed class counts in saved y_test_encoded.npy:
  Benign=500, DDoS=500, DoS=500, Mirai=500, Recon=500
- Metrics:
  PGD: clean_acc=0.8420, asr_all=0.8916, asr_clean_correct=0.8736, l2_mean=0.31570, linf_mean=0.08000
  CW:  clean_acc=0.8420, asr_all=0.7868, asr_clean_correct=0.7468, l2_mean=0.08344, linf_mean=0.05577

## Stability Check (256/class vs 500/class)
- PGD asr_all: 0.8961 -> 0.8916 (delta -0.0045)
- PGD asr_clean_correct: 0.8808 -> 0.8736 (delta -0.0072)
- CW asr_all: 0.7797 -> 0.7868 (delta +0.0071)
- CW asr_clean_correct: 0.7394 -> 0.7468 (delta +0.0074)
- Interpretation:
  Metrics are broadly stable under increased sample size and balanced class coverage, supporting thesis-level reliability better than the small 32-sample run.

## How Execution Works
1. Load processed train/test arrays and labels.
2. Build and load CNN-LSTM victim model checkpoint.
3. Normalize test features into [0,1] using train min/max.
4. Wrap model to map unit-space attacks back to standardized model space.
5. Run PGD and C&W in batches.
6. Evaluate attack success and perturbation sizes.
7. Print sample previews in terminal.
8. Save preview text and attack artifacts.

## Run Command
python baseline_attacks.py --n-samples 256 --batch-size 64 --preview-count 5

## Notes
- If torchattacks is missing: pip install torchattacks
- You can scale attack strength with:
  - --pgd-eps --pgd-alpha --pgd-steps
  - --cw-c --cw-kappa --cw-steps --cw-lr
- torchattacks pins requests~=2.25.1, which can conflict with JupyterLab dependencies.
- Local resolution used here: reinstall requests>=2.31 after torchattacks install; torchattacks import remains functional.
