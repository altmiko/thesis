# CICIDS2017-DistriNet Attack Results — validator_v2

Every saved `X_adv_raw` was independently reloaded and revalidated; runner-provided validity masks were not trusted. Denominator: clean-correct malicious samples. Plausibility is reported separately from structural validity.

## Metric definitions

- **Untargeted ASR:** adversarial prediction changes from the true malicious class to any other class.
- **Attack→Benign ASR:** adversarial prediction is specifically Benign (class 0). This is the primary NIDS-evasion metric.
- **v2-valid ASR:** attack success AND `validator_v2.hybrid_valid`, over the same clean-correct denominator.
- **Valid+ID ASR:** attack success AND hybrid validity AND in-distribution plausibility.

## Evaluation protocol

- Four attack classes: DoS, DDoS, Recon, BruteForce.
- Four victim models: MLP, CNN, LSTM, Serial CNN-LSTM.
- 256 sampled test rows per attack class; seed 42; CPU.
- 16 class × model cells and 4,055 clean-correct malicious rows per method.
- Strict validity is validator_v2 `hybrid_valid` only; no PAVE gate.

## Overall results

| Method | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted ASR | v2-valid Attack→Benign ASR | In-distribution | Valid+ID untargeted ASR | Valid+ID Attack→Benign ASR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Input PGD | 4055 | 99.73% | 99.58% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Primitive Direct | 4055 | 60.12% | 54.16% | 100.00% | 60.12% | 54.16% | 46.71% | 21.53% | 18.74% |
| VAE Latent Primitive | 4055 | 8.85% | 8.85% | 100.00% | 8.85% | 8.85% | 97.51% | 8.78% | 8.78% |
| VAE Latent Raw | 4055 | 68.80% | 68.66% | 0.00% | 0.00% | 0.00% | 0.47% | 0.00% | 0.00% |
| VAE Latent Masked | 4055 | 4.36% | 2.61% | 99.98% | 4.36% | 2.61% | 96.15% | 3.18% | 1.45% |

## Structural-validity layers

| Method | SCHEMA | EXTRACTOR | PROTOCOL | MINED | Hard structural | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| Input PGD | 0.00% | 0.00% | 0.00% | 0.30% | 0.00% | 0.00% |
| Primitive Direct | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| VAE Latent Primitive | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| VAE Latent Raw | 0.00% | 17.61% | 100.00% | 0.47% | 0.00% | 0.00% |
| VAE Latent Masked | 100.00% | 100.00% | 100.00% | 99.98% | 100.00% | 99.98% |

## Per-class summary

### Input PGD

| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1020 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | 1020 | 98.92% | 98.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | 1019 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | 996 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Primitive Direct

| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1020 | 64.90% | 64.90% | 100.00% | 64.90% | 64.90% | 15.39% |
| DDoS | 1020 | 98.14% | 74.51% | 100.00% | 98.14% | 74.51% | 24.02% |
| Recon | 1019 | 0.69% | 0.59% | 100.00% | 0.69% | 0.59% | 0.39% |
| BruteForce | 996 | 77.11% | 77.11% | 100.00% | 77.11% | 77.11% | 35.54% |

### VAE Latent Primitive

| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1020 | 29.02% | 29.02% | 100.00% | 29.02% | 29.02% | 28.73% |
| DDoS | 1020 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| Recon | 1019 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 0.39% |
| BruteForce | 996 | 5.92% | 5.92% | 100.00% | 5.92% | 5.92% | 5.92% |

### VAE Latent Raw

| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1020 | 90.88% | 90.78% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | 1020 | 78.24% | 78.04% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | 1019 | 32.29% | 32.09% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | 996 | 73.90% | 73.80% | 0.00% | 0.00% | 0.00% | 0.00% |

### VAE Latent Masked

| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1020 | 12.84% | 5.88% | 99.90% | 12.84% | 5.88% | 5.78% |
| DDoS | 1020 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| Recon | 1019 | 4.51% | 4.51% | 100.00% | 4.51% | 4.51% | 0.00% |
| BruteForce | 996 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |

## Detailed class × model results

### Input PGD

| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 256 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DoS | cnn | 256 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DoS | lstm | 253 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DoS | serial | 255 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | mlp | 255 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | cnn | 255 | 99.22% | 99.22% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | lstm | 255 | 98.43% | 97.25% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | serial | 255 | 98.04% | 96.86% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | mlp | 255 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | cnn | 255 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | lstm | 255 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | serial | 254 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | mlp | 249 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | cnn | 249 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | lstm | 250 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | serial | 248 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Primitive Direct

| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 256 | 14.84% | 14.84% | 100.00% | 14.84% | 14.84% | 0.39% | 0.00% |
| DoS | cnn | 256 | 47.66% | 47.66% | 100.00% | 47.66% | 47.66% | 0.39% | 0.39% |
| DoS | lstm | 253 | 98.81% | 98.81% | 100.00% | 98.81% | 98.81% | 21.34% | 21.34% |
| DoS | serial | 255 | 98.82% | 98.82% | 100.00% | 98.82% | 98.82% | 40.00% | 40.00% |
| DDoS | mlp | 255 | 92.55% | 20.78% | 100.00% | 92.55% | 20.78% | 50.20% | 0.78% |
| DDoS | cnn | 255 | 100.00% | 78.43% | 100.00% | 100.00% | 78.43% | 4.31% | 3.14% |
| DDoS | lstm | 255 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 89.80% | 89.80% |
| DDoS | serial | 255 | 100.00% | 98.82% | 100.00% | 100.00% | 98.82% | 3.53% | 2.35% |
| Recon | mlp | 255 | 0.78% | 0.39% | 100.00% | 0.78% | 0.39% | 95.29% | 0.39% |
| Recon | cnn | 255 | 0.78% | 0.78% | 100.00% | 0.78% | 0.78% | 95.29% | 0.39% |
| Recon | lstm | 255 | 0.78% | 0.78% | 100.00% | 0.78% | 0.78% | 95.29% | 0.39% |
| Recon | serial | 254 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 95.67% | 0.39% |
| BruteForce | mlp | 249 | 40.16% | 40.16% | 100.00% | 40.16% | 40.16% | 0.40% | 0.40% |
| BruteForce | cnn | 249 | 81.93% | 81.93% | 100.00% | 81.93% | 81.93% | 0.80% | 0.80% |
| BruteForce | lstm | 250 | 99.60% | 99.60% | 100.00% | 99.60% | 99.60% | 64.40% | 64.40% |
| BruteForce | serial | 248 | 86.69% | 86.69% | 100.00% | 86.69% | 86.69% | 89.92% | 76.61% |

### VAE Latent Primitive

| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 256 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 96.88% | 0.39% |
| DoS | cnn | 256 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 98.05% | 0.39% |
| DoS | lstm | 253 | 61.26% | 61.26% | 100.00% | 61.26% | 61.26% | 97.63% | 60.87% |
| DoS | serial | 255 | 54.51% | 54.51% | 100.00% | 54.51% | 54.51% | 96.86% | 53.73% |
| DDoS | mlp | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.43% | 0.00% |
| DDoS | cnn | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.04% | 0.00% |
| DDoS | lstm | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 95.69% | 0.00% |
| DDoS | serial | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 96.08% | 0.00% |
| Recon | mlp | 255 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 95.69% | 0.39% |
| Recon | cnn | 255 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 95.69% | 0.39% |
| Recon | lstm | 255 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 95.69% | 0.39% |
| Recon | serial | 254 | 0.39% | 0.39% | 100.00% | 0.39% | 0.39% | 95.67% | 0.39% |
| BruteForce | mlp | 249 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| BruteForce | cnn | 249 | 0.40% | 0.40% | 100.00% | 0.40% | 0.40% | 100.00% | 0.40% |
| BruteForce | lstm | 250 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| BruteForce | serial | 248 | 23.39% | 23.39% | 100.00% | 23.39% | 23.39% | 100.00% | 23.39% |

### VAE Latent Raw

| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 256 | 99.22% | 99.22% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DoS | cnn | 256 | 78.91% | 78.52% | 0.00% | 0.00% | 0.00% | 1.56% | 0.00% |
| DoS | lstm | 253 | 96.05% | 96.05% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DoS | serial | 255 | 89.41% | 89.41% | 0.00% | 0.00% | 0.00% | 0.78% | 0.00% |
| DDoS | mlp | 255 | 83.53% | 83.53% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | cnn | 255 | 45.88% | 45.88% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | lstm | 255 | 94.12% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| DDoS | serial | 255 | 89.41% | 89.41% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | mlp | 255 | 56.86% | 56.86% | 0.00% | 0.00% | 0.00% | 0.39% | 0.00% |
| Recon | cnn | 255 | 71.37% | 71.37% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Recon | lstm | 255 | 0.78% | 0.00% | 0.00% | 0.00% | 0.00% | 4.71% | 0.00% |
| Recon | serial | 254 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | mlp | 249 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | cnn | 249 | 71.49% | 71.49% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | lstm | 250 | 33.60% | 33.20% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| BruteForce | serial | 248 | 90.73% | 90.73% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### VAE Latent Masked

| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 256 | 14.84% | 0.39% | 100.00% | 14.84% | 0.39% | 98.05% | 0.39% |
| DoS | cnn | 256 | 10.55% | 0.78% | 99.61% | 10.55% | 0.78% | 98.05% | 0.39% |
| DoS | lstm | 253 | 3.95% | 0.40% | 100.00% | 3.95% | 0.40% | 98.81% | 0.40% |
| DoS | serial | 255 | 21.96% | 21.96% | 100.00% | 21.96% | 21.96% | 98.04% | 21.96% |
| DDoS | mlp | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.43% | 0.00% |
| DDoS | cnn | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.43% | 0.00% |
| DDoS | lstm | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.43% | 0.00% |
| DDoS | serial | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 98.43% | 0.00% |
| Recon | mlp | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 96.08% | 0.00% |
| Recon | cnn | 255 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 96.08% | 0.00% |
| Recon | lstm | 255 | 14.51% | 14.51% | 100.00% | 14.51% | 14.51% | 71.76% | 0.00% |
| Recon | serial | 254 | 3.54% | 3.54% | 100.00% | 3.54% | 3.54% | 88.19% | 0.00% |
| BruteForce | mlp | 249 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| BruteForce | cnn | 249 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| BruteForce | lstm | 250 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| BruteForce | serial | 248 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |

## Top failed v2 rules by method

### Input PGD
- `SCH_0024 Bwd Packet Length Max integer`: 4055
- `SCH_0117 Bwd Init Win Bytes integer`: 4055
- `SCH_0083 ACK Flag Count integer`: 4053
- `EXT_0007 Total Length of Bwd Packet ~= Total Bwd packets * Bwd Packet Length Mean`: 4053
- `SCH_0006 Protocol in [0.0, 6.0, 17.0]`: 4052
- `EXT_0001 Packet Length Variance ~= Packet Length Std^2`: 4051
- `SCH_0095 Fwd Bytes/Bulk Avg integer`: 4050
- `SCH_0097 Fwd Packet/Bulk Avg integer`: 4048
- `SCH_0056 Bwd PSH Flags integer`: 4047
- `SCH_0079 RST Flag Count integer`: 4046
- `EXT_0006 Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean`: 4045
- `SCH_0103 Bwd Packet/Bulk Avg integer`: 4042

### Primitive Direct
No v2 rule failures among clean-correct samples.

### VAE Latent Primitive
No v2 rule failures among clean-correct samples.

### VAE Latent Raw
- `SCH_0062 Fwd Header Length integer`: 4055
- `SCH_0077 SYN Flag Count integer`: 4048
- `SCH_0010 Total Fwd Packet integer`: 4039
- `SCH_0064 Bwd Header Length integer`: 4039
- `SCH_0083 ACK Flag Count integer`: 4034
- `SCH_0121 Fwd Seg Size Min integer`: 4030
- `SCH_0012 Total Bwd packets integer`: 3994
- `SCH_0115 FWD Init Win Bytes integer`: 3935
- `SCH_0006 Protocol in [0.0, 6.0, 17.0]`: 3862
- `SCH_0036 Flow IAT Min integer`: 3761
- `SCH_0079 RST Flag Count integer`: 3682
- `SCH_0016 Total Length of Bwd Packet integer`: 3659

### VAE Latent Masked
- `MINED_0001 Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max`: 1

## Conclusions

1. Input PGD and VAE Latent Raw obtain high raw ASR but 0% hybrid validity, so none of their successes survive v2.
2. Primitive Direct and VAE Latent Primitive are 100% structurally valid; their raw and v2-valid ASRs are identical.
3. VAE Latent Masked is 99.98% hybrid-valid; one DoS/CNN row violates the forward packet-length ordering rule.
4. Attack→Benign ASR is lower than untargeted ASR when attacks redirect samples to another malicious class rather than Benign.
5. In-distribution plausibility remains a separate gate and can substantially reduce validity-aware success.

## Reproduction artifacts

- `outputs/v2_validity_report/results.json` — machine-readable metrics.
- `outputs/v2_validity_report/report.md` — generated report.
- `python -m validation.evaluation.attack_artifact_validity` — regenerate both reports.
