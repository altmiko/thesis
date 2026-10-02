# Classifier evaluation (test split, training seed 42)

These numbers come from the stored test-set metrics of the existing checkpoints. Nothing was retrained or re-inferred.
Each metric was also recomputed from the stored test confusion matrix and agrees with the stored value
(largest absolute difference 2.2e-16). The category-head checkpoints are the FINAL-suite victims: their
SHA-256 hashes appear in `FINAL_OUTPUTS/runs/<dataset>/baselines_untargeted/` (CICIDS2017: `mlp`, `cnn`,
`ft_transformer`; CICIDS2018: `mlp-s42`, `cnn-s42`, `ft_transformer-s42`).

All values are percentages on the held-out test split (chronological within-label 70/15/15). Precision and recall are
macro averages over classes, so macro recall equals balanced accuracy.

## Category head (5 classes: Benign, DoS, DDoS, Recon, BruteForce), the attacked victims

| Dataset | Model | Test n | Accuracy | Macro precision | Macro recall | Macro F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| CICIDS2017-DistriNet | SimpleMLP | 312,056 | 98.45 | 96.69 | 99.03 | 97.73 | 99.03 |
| CICIDS2017-DistriNet | CNNOnly | 312,056 | 98.43 | 96.41 | 98.91 | 97.53 | 98.91 |
| CICIDS2017-DistriNet | FT-Transformer | 312,056 | 98.46 | 96.77 | 99.10 | 97.80 | 99.10 |
| CSE-CIC-IDS2018-DistriNet | SimpleMLP | 125,032 | 99.77 | 99.77 | 99.80 | 99.79 | 99.80 |
| CSE-CIC-IDS2018-DistriNet | CNNOnly | 125,032 | 99.88 | 99.86 | 99.88 | 99.87 | 99.88 |
| CSE-CIC-IDS2018-DistriNet | FT-Transformer | 125,032 | 99.95 | 99.95 | 99.96 | 99.96 | 99.96 |

## Binary head (Benign vs Attack)

| Dataset | Model | Test n | Accuracy | Macro precision | Macro recall | Macro F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| CICIDS2017-DistriNet | SimpleMLP | 312,056 | 98.42 | 96.64 | 98.77 | 97.65 | 98.77 |
| CICIDS2017-DistriNet | CNNOnly | 312,056 | 98.47 | 96.68 | 98.90 | 97.74 | 98.90 |
| CICIDS2017-DistriNet | FT-Transformer | 312,056 | 98.46 | 96.68 | 98.84 | 97.71 | 98.84 |
| CSE-CIC-IDS2018-DistriNet | SimpleMLP | 125,032 | 99.92 | 99.89 | 99.92 | 99.90 | 99.92 |
| CSE-CIC-IDS2018-DistriNet | CNNOnly | 125,032 | 99.89 | 99.85 | 99.88 | 99.87 | 99.88 |
| CSE-CIC-IDS2018-DistriNet | FT-Transformer | 125,032 | 99.95 | 99.93 | 99.96 | 99.94 | 99.96 |

## Notes

- CICIDS2017 macro precision (96.4–96.8%) sits below accuracy because of Recon precision of about 84% on every
  category model: roughly 4,510–4,517 of the 247,164 test Benign flows are predicted as Recon.
- The CICIDS2018 test split is controlled, not natural-prevalence: Benign, DoS and DDoS were time-stratified
  down during preprocessing, while Recon and BruteForce keep every row. Its scores are therefore not directly comparable
  with CICIDS2017.
- BruteForce is small on CICIDS2017 (1,042 test flows), so its contribution to the macro averages is unstable.
- Single training seed (42). These numbers say nothing about seed robustness. For CICIDS2018, seeds 123 and 2024 are in
  `outputs/cicids2018distrinet/classifiers_multiseed/per_run_metrics.csv`.

## Provenance

| Dataset | Model | Head | Best epoch | Checkpoint (SHA-256, first 12) | Metrics source |
|---|---|---|---:|---|---|
| CICIDS2017-DistriNet | SimpleMLP | binary | 7 | `65a2e89bdbbe` | `outputs/cicids2017distrinet/metrics/mlp_binary_metrics.json` |
| CICIDS2017-DistriNet | SimpleMLP | category | 9 | `a896ffccf88b` | `outputs/cicids2017distrinet/metrics/mlp_category_metrics.json` |
| CICIDS2017-DistriNet | CNNOnly | binary | 8 | `61c87cee9a6b` | `outputs/cicids2017distrinet/metrics/cnn_binary_metrics.json` |
| CICIDS2017-DistriNet | CNNOnly | category | 8 | `51849974ae7b` | `outputs/cicids2017distrinet/metrics/cnn_category_metrics.json` |
| CICIDS2017-DistriNet | FT-Transformer | binary | 3 | `78c751a165e8` | `outputs/cicids2017distrinet_ft/metrics/ft_transformer_binary_metrics.json` |
| CICIDS2017-DistriNet | FT-Transformer | category | 7 | `1afad63e3ec6` | `outputs/cicids2017distrinet_ft/metrics/ft_transformer_category_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | SimpleMLP | binary | 10 | `1a86828a61f6` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/mlp_binary_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | SimpleMLP | category | 10 | `1ac1c5708dcb` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/mlp_category_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | CNNOnly | binary | 5 | `8ca8437081cd` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/cnn_binary_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | CNNOnly | category | 10 | `0550b76328d3` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/cnn_category_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | FT-Transformer | binary | 10 | `f69ce2df556b` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/ft_transformer_binary_metrics.json` |
| CSE-CIC-IDS2018-DistriNet | FT-Transformer | category | 6 | `a5cbc18a93da` | `outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/metrics/ft_transformer_category_metrics.json` |
