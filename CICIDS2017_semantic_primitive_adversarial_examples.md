# CICIDS2017 — Semantically-Admissible Primitive-Realizable Adversarial Examples

Best (cheapest, realizability-validated) targeted `Attack -> Benign` adversarial flow per class from the **semantic-capability-gated** direct primitive attack (`src/attack/run_cicids2017_primitive_attack.py`, `infer_capabilities` gate). Two per-flow primitives are optimized — forward packet-length augmentation `p` (bytes/fwd packet) and forward timing dilation `alpha` — only where the source flow supports them; a discrete realizability projection then recomputes every dependent CICFlowMeter feature.

- **Artifacts:** `outputs/cicids2017_primitive_attack/attack_artifacts/<Class>_<victim>_seed<seed>.npz`
- **Class ids:** Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4
- **Raw units** are the pristine CICFlowMeter feature space, i.e. the inverse RobustScaler transform of the model input (`raw = scaled*scale + center`); this identity was re-verified against the stored raw vector (float32-exact, max rel err ~6e-5).


## DoS

- **source:** `DoS_serial_seed43.npz` row 435 | victim `serial`, seed 43 | selection tier `evasion+strict+in_dist`
- primitives: `p = 115.215` -> `p_real = 115` bytes;  `alpha = 1.5791` -> `alpha_real = 1.5791`  |  normalized cost = 0.2392
- capability: pad `PAD_ALLOWED`, timing `TIMING_ALLOWED`  |  strict-valid = True, in-distribution = True, inverse-transform verified = True
- **true class = DoS  ->  clean pred = DoS  ->  adversarial pred = Benign**  (Benign logit -2.40 -> 7.46)

| # | feature | clean (raw) | adversarial (raw) | delta |
|---|---------|-------------|-------------------|-------|
| 0 | Src Port | 5.888e+04 | 5.888e+04 | +0 |
| 1 | Dst Port | 80 | 80 | +0 |
| 2 | Protocol | 6 | 6 | +0 |
| 3 | **Flow Duration** | 1.6e+05 | 2.526e+05 | +9.265e+04 |
| 4 | Total Fwd Packet | 9 | 9 | +0 |
| 5 | Total Bwd packets | 7 | 7 | +0 |
| 6 | **Total Length of Fwd Packet** | 400 | 1435 | +1035 |
| 7 | Total Length of Bwd Packet | 1.16e+04 | 1.16e+04 | +0 |
| 8 | **Fwd Packet Length Max** | 400 | 515 | +115 |
| 9 | **Fwd Packet Length Min** | 0 | 115 | +115 |
| 10 | **Fwd Packet Length Mean** | 44.44 | 159.4 | +115 |
| 11 | Fwd Packet Length Std | 133.3 | 133.3 | +0 |
| 12 | Bwd Packet Length Max | 5792 | 5792 | +0 |
| 13 | Bwd Packet Length Min | 0 | 0 | +0 |
| 14 | Bwd Packet Length Mean | 1656 | 1656 | +0 |
| 15 | Bwd Packet Length Std | 1948 | 1948 | +0 |
| 16 | **Flow Bytes/s** | 7.498e+04 | 5.158e+04 | -2.34e+04 |
| 17 | **Flow Packets/s** | 100 | 63.33 | -36.68 |
| 18 | **Flow IAT Mean** | 1.067e+04 | 1.684e+04 | +6176 |
| 19 | Flow IAT Std | 3.391e+04 | 3.391e+04 | +0 |
| 20 | **Flow IAT Max** | 1.321e+05 | 2.248e+05 | +9.265e+04 |
| 21 | Flow IAT Min | 0 | 0 | +0 |
| 22 | **Fwd IAT Total** | 1.6e+05 | 2.526e+05 | +9.265e+04 |
| 23 | **Fwd IAT Mean** | 2e+04 | 3.158e+04 | +1.158e+04 |
| 24 | **Fwd IAT Std** | 5.074e+04 | 8.012e+04 | +2.938e+04 |
| 25 | **Fwd IAT Max** | 1.45e+05 | 2.29e+05 | +8.397e+04 |
| 26 | Fwd IAT Min | 0 | 0 | +0 |
| 27 | Bwd IAT Total | 1.466e+05 | 1.466e+05 | +0 |
| 28 | Bwd IAT Mean | 2.444e+04 | 2.444e+04 | +0 |
| 29 | Bwd IAT Std | 5.305e+04 | 5.305e+04 | +0 |
| 30 | Bwd IAT Max | 1.321e+05 | 1.321e+05 | +0 |
| 31 | Bwd IAT Min | 45 | 45 | +0 |
| 32 | Fwd PSH Flags | 1 | 1 | +0 |
| 33 | Bwd PSH Flags | 1 | 1 | +0 |
| 34 | Fwd URG Flags | 0 | 0 | +0 |
| 35 | Bwd URG Flags | 0 | 0 | +0 |
| 36 | Fwd Header Length | 284 | 284 | +0 |
| 37 | Bwd Header Length | 232 | 232 | +0 |
| 38 | **Fwd Packets/s** | 56.26 | 35.63 | -20.63 |
| 39 | **Bwd Packets/s** | 43.76 | 27.71 | -16.05 |
| 40 | Packet Length Min | 0 | 0 | +0 |
| 41 | Packet Length Max | 5792 | 5792 | +0 |
| 42 | **Packet Length Mean** | 749.7 | 814.4 | +64.69 |
| 43 | **Packet Length Std** | 1486 | 1454 | -31.92 |
| 44 | **Packet Length Variance** | 2.209e+06 | 2.115e+06 | -9.385e+04 |
| 45 | FIN Flag Count | 1 | 1 | +0 |
| 46 | SYN Flag Count | 2 | 2 | +0 |
| 47 | RST Flag Count | 2 | 2 | +0 |
| 48 | PSH Flag Count | 2 | 2 | +0 |
| 49 | ACK Flag Count | 14 | 14 | +0 |
| 50 | URG Flag Count | 0 | 0 | +0 |
| 51 | CWR Flag Count | 0 | 0 | +0 |
| 52 | ECE Flag Count | 0 | 0 | +0 |
| 53 | Down/Up Ratio | 0.7778 | 0.7778 | +0 |
| 54 | **Average Packet Size** | 749.7 | 814.4 | +64.69 |
| 55 | **Fwd Segment Size Avg** | 44.44 | 159.4 | +115 |
| 56 | Bwd Segment Size Avg | 1656 | 1656 | +0 |
| 57 | Fwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 58 | Fwd Packet/Bulk Avg | 0 | 0 | +0 |
| 59 | Fwd Bulk Rate Avg | 0 | 0 | +0 |
| 60 | Bwd Bytes/Bulk Avg | 1.16e+04 | 1.16e+04 | +0 |
| 61 | Bwd Packet/Bulk Avg | 5 | 5 | +0 |
| 62 | Bwd Bulk Rate Avg | 2.411e+07 | 2.411e+07 | +0 |
| 63 | Subflow Fwd Packets | 0 | 0 | +0 |
| 64 | Subflow Fwd Bytes | 25 | 25 | +0 |
| 65 | Subflow Bwd Packets | 0 | 0 | +0 |
| 66 | Subflow Bwd Bytes | 724 | 724 | +0 |
| 67 | FWD Init Win Bytes | 2.92e+04 | 2.92e+04 | +0 |
| 68 | Bwd Init Win Bytes | 235 | 235 | +0 |
| 69 | Fwd Act Data Pkts | 1 | 1 | +0 |
| 70 | Fwd Seg Size Min | 20 | 20 | +0 |
| 71 | Active Mean | 0 | 0 | +0 |
| 72 | Active Std | 0 | 0 | +0 |
| 73 | Active Max | 0 | 0 | +0 |
| 74 | Active Min | 0 | 0 | +0 |
| 75 | Idle Mean | 0 | 0 | +0 |
| 76 | Idle Std | 0 | 0 | +0 |
| 77 | Idle Max | 0 | 0 | +0 |
| 78 | Idle Min | 0 | 0 | +0 |

## DDoS

- **source:** `DDoS_lstm_seed43.npz` row 945 | victim `lstm`, seed 43 | selection tier `evasion+strict+in_dist`
- primitives: `p = 27.915` -> `p_real = 28` bytes;  `alpha = 1.1849` -> `alpha_real = 1.1849`  |  normalized cost = 0.0811
- capability: pad `PAD_ALLOWED`, timing `TIMING_ALLOWED`  |  strict-valid = True, in-distribution = True, inverse-transform verified = True
- **true class = DDoS  ->  clean pred = DDoS  ->  adversarial pred = Benign**  (Benign logit 0.72 -> 10.25)

| # | feature | clean (raw) | adversarial (raw) | delta |
|---|---------|-------------|-------------------|-------|
| 0 | Src Port | 6.464e+04 | 6.464e+04 | +0 |
| 1 | Dst Port | 80 | 80 | +0 |
| 2 | Protocol | 6 | 6 | +0 |
| 3 | **Flow Duration** | 6.622e+06 | 7.846e+06 | +1.224e+06 |
| 4 | Total Fwd Packet | 9 | 9 | +0 |
| 5 | Total Bwd packets | 7 | 7 | +0 |
| 6 | **Total Length of Fwd Packet** | 20 | 272 | +252 |
| 7 | Total Length of Bwd Packet | 1.16e+04 | 1.16e+04 | +0 |
| 8 | **Fwd Packet Length Max** | 20 | 48 | +28 |
| 9 | **Fwd Packet Length Min** | 0 | 28 | +28 |
| 10 | **Fwd Packet Length Mean** | 2.222 | 30.22 | +28 |
| 11 | Fwd Packet Length Std | 6.667 | 6.667 | +0 |
| 12 | Bwd Packet Length Max | 5840 | 5840 | +0 |
| 13 | Bwd Packet Length Min | 0 | 0 | +0 |
| 14 | Bwd Packet Length Mean | 1656 | 1656 | +0 |
| 15 | Bwd Packet Length Std | 2290 | 2290 | +0 |
| 16 | **Flow Bytes/s** | 1754 | 1512 | -241.6 |
| 17 | **Flow Packets/s** | 2.416 | 2.039 | -0.3771 |
| 18 | **Flow IAT Mean** | 4.415e+05 | 5.231e+05 | +8.163e+04 |
| 19 | Flow IAT Std | 1.558e+06 | 1.558e+06 | +0 |
| 20 | **Flow IAT Max** | 6.053e+06 | 7.277e+06 | +1.224e+06 |
| 21 | Flow IAT Min | 1 | 1 | +0 |
| 22 | **Fwd IAT Total** | 6.622e+06 | 7.846e+06 | +1.224e+06 |
| 23 | **Fwd IAT Mean** | 8.277e+05 | 9.808e+05 | +1.531e+05 |
| 24 | **Fwd IAT Std** | 2.12e+06 | 2.512e+06 | +3.92e+05 |
| 25 | **Fwd IAT Max** | 6.053e+06 | 7.172e+06 | +1.119e+06 |
| 26 | Fwd IAT Min | 1 | 1 | +0 |
| 27 | Bwd IAT Total | 5.426e+05 | 5.426e+05 | +0 |
| 28 | Bwd IAT Mean | 9.043e+04 | 9.043e+04 | +0 |
| 29 | Bwd IAT Std | 2.132e+05 | 2.132e+05 | +0 |
| 30 | Bwd IAT Max | 5.254e+05 | 5.254e+05 | +0 |
| 31 | Bwd IAT Min | 47 | 47 | +0 |
| 32 | Fwd PSH Flags | 1 | 1 | +0 |
| 33 | Bwd PSH Flags | 1 | 1 | +0 |
| 34 | Fwd URG Flags | 0 | 0 | +0 |
| 35 | Bwd URG Flags | 0 | 0 | +0 |
| 36 | Fwd Header Length | 192 | 192 | +0 |
| 37 | Bwd Header Length | 152 | 152 | +0 |
| 38 | **Fwd Packets/s** | 1.359 | 1.147 | -0.2121 |
| 39 | **Bwd Packets/s** | 1.057 | 0.8921 | -0.165 |
| 40 | Packet Length Min | 0 | 0 | +0 |
| 41 | Packet Length Max | 5840 | 5840 | +0 |
| 42 | **Packet Length Mean** | 725.9 | 741.7 | +15.75 |
| 43 | **Packet Length Std** | 1678 | 1671 | -7.2 |
| 44 | **Packet Length Variance** | 2.816e+06 | 2.792e+06 | -2.411e+04 |
| 45 | FIN Flag Count | 2 | 2 | +0 |
| 46 | SYN Flag Count | 2 | 2 | +0 |
| 47 | RST Flag Count | 1 | 1 | +0 |
| 48 | PSH Flag Count | 2 | 2 | +0 |
| 49 | ACK Flag Count | 15 | 15 | +0 |
| 50 | URG Flag Count | 0 | 0 | +0 |
| 51 | CWR Flag Count | 0 | 0 | +0 |
| 52 | ECE Flag Count | 0 | 0 | +0 |
| 53 | Down/Up Ratio | 0.7778 | 0.7778 | +0 |
| 54 | **Average Packet Size** | 725.9 | 741.7 | +15.75 |
| 55 | **Fwd Segment Size Avg** | 2.222 | 30.22 | +28 |
| 56 | Bwd Segment Size Avg | 1656 | 1656 | +0 |
| 57 | Fwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 58 | Fwd Packet/Bulk Avg | 0 | 0 | +0 |
| 59 | Fwd Bulk Rate Avg | 0 | 0 | +0 |
| 60 | Bwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 61 | Bwd Packet/Bulk Avg | 0 | 0 | +0 |
| 62 | Bwd Bulk Rate Avg | 0 | 0 | +0 |
| 63 | Subflow Fwd Packets | 0 | 0 | +0 |
| 64 | Subflow Fwd Bytes | 1 | 1 | +0 |
| 65 | Subflow Bwd Packets | 0 | 0 | +0 |
| 66 | Subflow Bwd Bytes | 724 | 724 | +0 |
| 67 | FWD Init Win Bytes | 8192 | 8192 | +0 |
| 68 | Bwd Init Win Bytes | 229 | 229 | +0 |
| 69 | Fwd Act Data Pkts | 1 | 1 | +0 |
| 70 | Fwd Seg Size Min | 20 | 20 | +0 |
| 71 | Active Mean | 5.69e+05 | 5.69e+05 | +0 |
| 72 | Active Std | 0 | 0 | +0 |
| 73 | Active Max | 5.69e+05 | 5.69e+05 | +0 |
| 74 | Active Min | 5.69e+05 | 5.69e+05 | +0 |
| 75 | Idle Mean | 6.053e+06 | 6.053e+06 | +0 |
| 76 | Idle Std | 0 | 0 | +0 |
| 77 | Idle Max | 6.053e+06 | 6.053e+06 | +0 |
| 78 | Idle Min | 6.053e+06 | 6.053e+06 | +0 |

## Recon

- **source:** `Recon_lstm_seed43.npz` row 773 | victim `lstm`, seed 43 | selection tier `evasion+strict`
- primitives: `p = 0.000` -> `p_real = 0` bytes;  `alpha = 3.9352` -> `alpha_real = 3.9352`  |  normalized cost = 0.5154
- capability: pad `NO_FORWARD_PAYLOAD`, timing `TIMING_ALLOWED`  |  strict-valid = True, in-distribution = False, inverse-transform verified = True
- **true class = Recon  ->  clean pred = Recon  ->  adversarial pred = Benign**  (Benign logit -0.45 -> 8.67)

| # | feature | clean (raw) | adversarial (raw) | delta |
|---|---------|-------------|-------------------|-------|
| 0 | Src Port | 5.806e+04 | 5.806e+04 | +0 |
| 1 | Dst Port | 80 | 80 | +0 |
| 2 | Protocol | 6 | 6 | +0 |
| 3 | **Flow Duration** | 597 | 2349 | +1752 |
| 4 | Total Fwd Packet | 2 | 2 | +0 |
| 5 | Total Bwd packets | 1 | 1 | +0 |
| 6 | Total Length of Fwd Packet | 0 | 0 | +0 |
| 7 | Total Length of Bwd Packet | 0 | 0 | +0 |
| 8 | Fwd Packet Length Max | 0 | 0 | +0 |
| 9 | Fwd Packet Length Min | 0 | 0 | +0 |
| 10 | Fwd Packet Length Mean | 0 | 0 | +0 |
| 11 | Fwd Packet Length Std | 0 | 0 | +0 |
| 12 | Bwd Packet Length Max | 0 | 0 | +0 |
| 13 | Bwd Packet Length Min | 0 | 0 | +0 |
| 14 | Bwd Packet Length Mean | 0 | 0 | +0 |
| 15 | Bwd Packet Length Std | 0 | 0 | +0 |
| 16 | Flow Bytes/s | 0 | 0 | +0 |
| 17 | **Flow Packets/s** | 5025 | 1277 | -3748 |
| 18 | **Flow IAT Mean** | 298.5 | 1174 | +876 |
| 19 | Flow IAT Std | 399.5 | 399.5 | +0 |
| 20 | **Flow IAT Max** | 581 | 2333 | +1752 |
| 21 | Flow IAT Min | 16 | 16 | +0 |
| 22 | **Fwd IAT Total** | 597 | 2349 | +1752 |
| 23 | **Fwd IAT Mean** | 597 | 2349 | +1752 |
| 24 | Fwd IAT Std | 0 | 0 | +0 |
| 25 | **Fwd IAT Max** | 597 | 2349 | +1752 |
| 26 | **Fwd IAT Min** | 597 | 2349 | +1752 |
| 27 | Bwd IAT Total | 0 | 0 | +0 |
| 28 | Bwd IAT Mean | 0 | 0 | +0 |
| 29 | Bwd IAT Std | 0 | 0 | +0 |
| 30 | Bwd IAT Max | 0 | 0 | +0 |
| 31 | Bwd IAT Min | 0 | 0 | +0 |
| 32 | Fwd PSH Flags | 0 | 0 | +0 |
| 33 | Bwd PSH Flags | 0 | 0 | +0 |
| 34 | Fwd URG Flags | 0 | 0 | +0 |
| 35 | Bwd URG Flags | 0 | 0 | +0 |
| 36 | Fwd Header Length | 44 | 44 | +0 |
| 37 | Bwd Header Length | 24 | 24 | +0 |
| 38 | **Fwd Packets/s** | 3350 | 851.4 | -2499 |
| 39 | **Bwd Packets/s** | 1675 | 425.7 | -1249 |
| 40 | Packet Length Min | 0 | 0 | +0 |
| 41 | Packet Length Max | 0 | 0 | +0 |
| 42 | Packet Length Mean | 0 | 0 | +0 |
| 43 | Packet Length Std | 0 | 0 | +0 |
| 44 | Packet Length Variance | 0 | 0 | +0 |
| 45 | FIN Flag Count | 0 | 0 | +0 |
| 46 | SYN Flag Count | 2 | 2 | +0 |
| 47 | RST Flag Count | 1 | 1 | +0 |
| 48 | PSH Flag Count | 0 | 0 | +0 |
| 49 | ACK Flag Count | 1 | 1 | +0 |
| 50 | URG Flag Count | 0 | 0 | +0 |
| 51 | CWR Flag Count | 0 | 0 | +0 |
| 52 | ECE Flag Count | 0 | 0 | +0 |
| 53 | Down/Up Ratio | 0.5 | 0.5 | +0 |
| 54 | Average Packet Size | 0 | 0 | +0 |
| 55 | Fwd Segment Size Avg | 0 | 0 | +0 |
| 56 | Bwd Segment Size Avg | 0 | 0 | +0 |
| 57 | Fwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 58 | Fwd Packet/Bulk Avg | 0 | 0 | +0 |
| 59 | Fwd Bulk Rate Avg | 0 | 0 | +0 |
| 60 | Bwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 61 | Bwd Packet/Bulk Avg | 0 | 0 | +0 |
| 62 | Bwd Bulk Rate Avg | 0 | 0 | +0 |
| 63 | Subflow Fwd Packets | 0 | 0 | +0 |
| 64 | Subflow Fwd Bytes | 0 | 0 | +0 |
| 65 | Subflow Bwd Packets | 0 | 0 | +0 |
| 66 | Subflow Bwd Bytes | 0 | 0 | +0 |
| 67 | FWD Init Win Bytes | 1024 | 1024 | +0 |
| 68 | Bwd Init Win Bytes | 2.92e+04 | 2.92e+04 | +0 |
| 69 | Fwd Act Data Pkts | 0 | 0 | +0 |
| 70 | Fwd Seg Size Min | 20 | 20 | +0 |
| 71 | Active Mean | 0 | 0 | +0 |
| 72 | Active Std | 0 | 0 | +0 |
| 73 | Active Max | 0 | 0 | +0 |
| 74 | Active Min | 0 | 0 | +0 |
| 75 | Idle Mean | 0 | 0 | +0 |
| 76 | Idle Std | 0 | 0 | +0 |
| 77 | Idle Max | 0 | 0 | +0 |
| 78 | Idle Min | 0 | 0 | +0 |

## BruteForce

- **source:** `BruteForce_serial_seed44.npz` row 51 | victim `serial`, seed 44 | selection tier `evasion+strict`
- primitives: `p = 30.009` -> `p_real = 30` bytes;  `alpha = 1.3205` -> `alpha_real = 1.3205`  |  normalized cost = 0.1003
- capability: pad `PAD_ALLOWED`, timing `TIMING_ALLOWED`  |  strict-valid = True, in-distribution = False, inverse-transform verified = True
- **true class = BruteForce  ->  clean pred = BruteForce  ->  adversarial pred = Benign**  (Benign logit 1.02 -> 5.82)

| # | feature | clean (raw) | adversarial (raw) | delta |
|---|---------|-------------|-------------------|-------|
| 0 | Src Port | 5.905e+04 | 5.905e+04 | +0 |
| 1 | Dst Port | 21 | 21 | +0 |
| 2 | Protocol | 6 | 6 | +0 |
| 3 | **Flow Duration** | 8.423e+06 | 1.112e+07 | +2.7e+06 |
| 4 | Total Fwd Packet | 11 | 11 | +0 |
| 5 | Total Bwd packets | 17 | 17 | +0 |
| 6 | **Total Length of Fwd Packet** | 110 | 440 | +330 |
| 7 | Total Length of Bwd Packet | 188 | 188 | +0 |
| 8 | **Fwd Packet Length Max** | 19 | 49 | +30 |
| 9 | **Fwd Packet Length Min** | 0 | 30 | +30 |
| 10 | **Fwd Packet Length Mean** | 10 | 40 | +30 |
| 11 | Fwd Packet Length Std | 8.136 | 8.136 | +0 |
| 12 | Bwd Packet Length Max | 34 | 34 | +0 |
| 13 | Bwd Packet Length Min | 0 | 0 | +0 |
| 14 | Bwd Packet Length Mean | 11.06 | 11.06 | +0 |
| 15 | Bwd Packet Length Std | 14.23 | 14.23 | +0 |
| 16 | **Flow Bytes/s** | 35.38 | 56.46 | +21.08 |
| 17 | **Flow Packets/s** | 3.324 | 2.518 | -0.8069 |
| 18 | **Flow IAT Mean** | 3.119e+05 | 4.119e+05 | +9.998e+04 |
| 19 | Flow IAT Std | 8.81e+05 | 8.81e+05 | +0 |
| 20 | **Flow IAT Max** | 2.867e+06 | 5.566e+06 | +2.7e+06 |
| 21 | Flow IAT Min | 3 | 3 | +0 |
| 22 | **Fwd IAT Total** | 8.422e+06 | 1.112e+07 | +2.7e+06 |
| 23 | **Fwd IAT Mean** | 8.422e+05 | 1.112e+06 | +2.7e+05 |
| 24 | **Fwd IAT Std** | 1.352e+06 | 1.785e+06 | +4.333e+05 |
| 25 | **Fwd IAT Max** | 2.911e+06 | 3.844e+06 | +9.331e+05 |
| 26 | **Fwd IAT Min** | 3 | 4 | +1 |
| 27 | Bwd IAT Total | 8.422e+06 | 8.422e+06 | +0 |
| 28 | Bwd IAT Mean | 5.264e+05 | 5.264e+05 | +0 |
| 29 | Bwd IAT Std | 1.107e+06 | 1.107e+06 | +0 |
| 30 | Bwd IAT Max | 2.867e+06 | 2.867e+06 | +0 |
| 31 | Bwd IAT Min | 3 | 3 | +0 |
| 32 | Fwd PSH Flags | 7 | 7 | +0 |
| 33 | Bwd PSH Flags | 7 | 7 | +0 |
| 34 | Fwd URG Flags | 0 | 0 | +0 |
| 35 | Bwd URG Flags | 0 | 0 | +0 |
| 36 | Fwd Header Length | 360 | 360 | +0 |
| 37 | Bwd Header Length | 528 | 528 | +0 |
| 38 | **Fwd Packets/s** | 1.306 | 0.989 | -0.317 |
| 39 | **Bwd Packets/s** | 2.018 | 1.528 | -0.4899 |
| 40 | Packet Length Min | 0 | 0 | +0 |
| 41 | **Packet Length Max** | 34 | 49 | +15 |
| 42 | **Packet Length Mean** | 10.64 | 22.43 | +11.79 |
| 43 | **Packet Length Std** | 12.03 | 18.75 | +6.72 |
| 44 | **Packet Length Variance** | 144.8 | 351.7 | +206.9 |
| 45 | FIN Flag Count | 2 | 2 | +0 |
| 46 | SYN Flag Count | 2 | 2 | +0 |
| 47 | RST Flag Count | 2 | 2 | +0 |
| 48 | PSH Flag Count | 14 | 14 | +0 |
| 49 | ACK Flag Count | 25 | 25 | +0 |
| 50 | URG Flag Count | 0 | 0 | +0 |
| 51 | CWR Flag Count | 0 | 0 | +0 |
| 52 | ECE Flag Count | 0 | 0 | +0 |
| 53 | Down/Up Ratio | 1.545 | 1.545 | +0 |
| 54 | **Average Packet Size** | 10.64 | 22.43 | +11.79 |
| 55 | **Fwd Segment Size Avg** | 10 | 40 | +30 |
| 56 | Bwd Segment Size Avg | 11.06 | 11.06 | +0 |
| 57 | Fwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 58 | Fwd Packet/Bulk Avg | 0 | 0 | +0 |
| 59 | Fwd Bulk Rate Avg | 0 | 0 | +0 |
| 60 | Bwd Bytes/Bulk Avg | 0 | 0 | +0 |
| 61 | Bwd Packet/Bulk Avg | 0 | 0 | +0 |
| 62 | Bwd Bulk Rate Avg | 0 | 0 | +0 |
| 63 | Subflow Fwd Packets | 0 | 0 | +0 |
| 64 | Subflow Fwd Bytes | 3 | 3 | +0 |
| 65 | Subflow Bwd Packets | 0 | 0 | +0 |
| 66 | Subflow Bwd Bytes | 6 | 6 | +0 |
| 67 | FWD Init Win Bytes | 2.92e+04 | 2.92e+04 | +0 |
| 68 | Bwd Init Win Bytes | 0 | 0 | +0 |
| 69 | Fwd Act Data Pkts | 7 | 7 | +0 |
| 70 | Fwd Seg Size Min | 32 | 32 | +0 |
| 71 | Active Mean | 0 | 0 | +0 |
| 72 | Active Std | 0 | 0 | +0 |
| 73 | Active Max | 0 | 0 | +0 |
| 74 | Active Min | 0 | 0 | +0 |
| 75 | Idle Mean | 0 | 0 | +0 |
| 76 | Idle Std | 0 | 0 | +0 |
| 77 | Idle Max | 0 | 0 | +0 |
| 78 | Idle Min | 0 | 0 | +0 |