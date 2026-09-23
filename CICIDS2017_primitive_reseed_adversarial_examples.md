# CICIDS2017-DistriNet — reseeded primitive-direct adversarial attack

Direct primitive-domain attack (`method_id = primitive_direct`): the attacker optimizes two per-flow primitives — forward packet-length augmentation `p` (bytes/fwd packet) and forward timing dilation `α` — through a differentiable realizability map (NO VAE in the gradient path); the VAE is used only as a val-anchored Mahalanobis realism gate (IDR).

- **Re-run with new seeds:** `[45, 46, 47, 48, 49]` (evaluation rows fixed across seeds; only optimizer initialization varies). All effectiveness numbers below are the mean±std **over these seeds**.
- **Denominator:** clean-correct malicious test rows per (class, victim).
- **Threat model:** targeted Attack->Benign (untargeted also reported).
- **Strict valid:** PAVE (Level-A) ∧ mined-density ∧ internal primitive-realizability.
- **Class ids:** `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`.
- All ASR / validity / IDR figures are **percentages**.

---

## 1. Effectiveness averaged over new seeds (mean±std %, per class × victim)

| Class | Victim | N clean-correct | Untargeted ASR | Targeted-Benign ASR | Strict Validity | Targeted Strict-Valid ASR | IDR | True-IDSR |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| DoS | mlp | 1019 | 16.3±0.2 | 16.3±0.2 | 99.8±0.0 | 16.3±0.2 | 5.5±0.3 | 0.7±0.2 |
| DoS | cnn | 1017 | 47.6±0.3 | 47.6±0.3 | 99.8±0.0 | 47.6±0.3 | 23.4±1.0 | 10.8±0.7 |
| DoS | lstm | 1002 | 98.2±0.1 | 98.2±0.1 | 99.8±0.0 | 98.2±0.1 | 30.6±0.5 | 29.7±0.5 |
| DoS | serial | 1018 | 98.1±0.0 | 98.1±0.0 | 99.8±0.0 | 98.1±0.0 | 60.6±0.6 | 60.6±0.5 |
| DDoS | mlp | 1022 | 92.7±0.5 | 16.6±0.5 | 100.0±0.0 | 16.6±0.5 | 0.0±0.0 | 0.0±0.0 |
| DDoS | cnn | 1022 | 100.0±0.0 | 75.9±0.5 | 100.0±0.0 | 75.9±0.5 | 0.0±0.0 | 0.0±0.0 |
| DDoS | lstm | 1022 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| DDoS | serial | 1022 | 100.0±0.0 | 99.2±0.1 | 100.0±0.0 | 99.2±0.1 | 0.0±0.0 | 0.0±0.0 |
| Recon | mlp | 1022 | 100.0±0.0 | 97.2±0.0 | 100.0±0.0 | 97.2±0.0 | 0.0±0.0 | 0.0±0.0 |
| Recon | cnn | 1019 | 100.0±0.0 | 99.4±0.2 | 100.0±0.0 | 99.4±0.2 | 0.0±0.0 | 0.0±0.0 |
| Recon | lstm | 1020 | 100.0±0.0 | 99.9±0.1 | 100.0±0.0 | 99.9±0.1 | 0.0±0.0 | 0.0±0.0 |
| Recon | serial | 1019 | 100.0±0.0 | 99.7±0.1 | 100.0±0.0 | 99.7±0.1 | 0.0±0.0 | 0.0±0.0 |
| BruteForce | mlp | 1004 | 42.7±0.1 | 42.7±0.1 | 100.0±0.0 | 42.7±0.1 | 0.0±0.0 | 0.0±0.0 |
| BruteForce | cnn | 1000 | 80.6±0.5 | 80.6±0.5 | 100.0±0.0 | 80.6±0.5 | 0.0±0.0 | 0.0±0.0 |
| BruteForce | lstm | 1000 | 99.6±0.2 | 99.6±0.2 | 100.0±0.0 | 99.6±0.2 | 0.0±0.0 | 0.0±0.0 |
| BruteForce | serial | 1000 | 85.5±0.7 | 85.5±0.7 | 100.0±0.0 | 85.5±0.7 | 0.0±0.0 | 0.0±0.0 |

## 2. Per-class macro (mean over victims) and overall, %

| Class | Untargeted ASR | Targeted-Benign ASR | Targeted Strict-Valid ASR | Strict Validity | IDR | True-IDSR |
|---|--:|--:|--:|--:|--:|--:|
| DoS | 65.1 | 65.1 | 65.1 | 99.8 | 30.1 | 25.5 |
| DDoS | 98.2 | 72.9 | 72.9 | 100.0 | 0.0 | 0.0 |
| Recon | 100.0 | 99.1 | 99.1 | 100.0 | 0.0 | 0.0 |
| BruteForce | 77.1 | 77.1 | 77.1 | 100.0 | 0.0 | 0.0 |
| **OVERALL** | 85.1 | 78.5 | 78.5 | 100.0 | 7.5 | 6.4 |

*Macro = unweighted mean over victims (and over classes for OVERALL).*

---

## 3. Best targeted adversarial examples (inverse-transformed, base vs adversarial)

For each attack class: the smallest normalized-cost row that was clean-correct and evaded to **Benign** while passing strict validity, chosen across all new seeds and victims. Values are **inverse-transformed raw CICFlowMeter units**.

### DoS → Benign  (victim = cnn, seed = 45, row = 14)

- primitives: forward packet-length augmentation `p = 36.971` → `p_real = 37` bytes;  timing dilation `α = 1.0002` → `α_real = 1.0002`  |  normalized cost = 0.0654
- **true class = DoS  →  adversarial prediction = Benign**

#### Base vs. adversarial — all features (raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 54392 | 54392 |
| 1 | Dst Port | F | 80 | 80 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 1.1588e+08 | 1.1590e+08 |
| 4 | Total Fwd Packet | F | 3 | 3 |
| 5 | Total Bwd packets | F | 3 | 3 |
| 6 | Total Length of Fwd Packet | Dp | 8 | 119 |
| 7 | Total Length of Bwd Packet | F | 483 | 483 |
| 8 | Fwd Packet Length Max | Dp | 8 | 45 |
| 9 | Fwd Packet Length Min | Dp | 0 | 37 |
| 10 | Fwd Packet Length Mean | Dp | 2.6667 | 39.6667 |
| 11 | Fwd Packet Length Std | I | 4.6188 | 4.6188 |
| 12 | Bwd Packet Length Max | F | 483 | 483 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 161 | 161 |
| 15 | Bwd Packet Length Std | F | 278.8602 | 278.8602 |
| 16 | Flow Bytes/s | R | 4.2372 | 5.1940 |
| 17 | Flow Packets/s | R | 0.0518 | 0.0518 |
| 18 | Flow IAT Mean | Dt | 2.3176e+07 | 2.3181e+07 |
| 19 | Flow IAT Std | Fᶜ | 5.1818e+07 | 5.1818e+07 |
| 20 | Flow IAT Max | C | 1.1587e+08 | 1.1590e+08 |
| 21 | Flow IAT Min | Fᶜ | 2 | 2 |
| 22 | Fwd IAT Total | Dt | 1.1588e+08 | 1.1590e+08 |
| 23 | Fwd IAT Mean | Dt | 5.7940e+07 | 5.7952e+07 |
| 24 | Fwd IAT Std | Dt | 8.1928e+07 | 8.1945e+07 |
| 25 | Fwd IAT Max | Dt | 1.1587e+08 | 1.1590e+08 |
| 26 | Fwd IAT Min | Dt | 7704 | 7706 |
| 27 | Bwd IAT Total | F | 1.1588e+08 | 1.1588e+08 |
| 28 | Bwd IAT Mean | F | 5.7938e+07 | 5.7938e+07 |
| 29 | Bwd IAT Std | F | 8.1937e+07 | 8.1937e+07 |
| 30 | Bwd IAT Max | F | 1.1588e+08 | 1.1588e+08 |
| 31 | Bwd IAT Min | F | 2 | 2 |
| 32 | Fwd PSH Flags | F | 1 | 1 |
| 33 | Bwd PSH Flags | F | 1 | 1 |
| 34 | Fwd URG Flags | F | 0 | 0 |
| 35 | Bwd URG Flags | F | 0 | 0 |
| 36 | Fwd Header Length | F | 84 | 84 |
| 37 | Bwd Header Length | F | 96 | 96 |
| 38 | Fwd Packets/s | R | 0.0259 | 0.0259 |
| 39 | Bwd Packets/s | R | 0.0259 | 0.0259 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 483 | 483 |
| 42 | Packet Length Mean | Dp | 81.8333 | 100.3333 |
| 43 | Packet Length Std | Dp | 196.5568 | 188.4947 |
| 44 | Packet Length Variance | Dp | 38634.5664 | 35530.2656 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 0 | 0 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 2 | 2 |
| 49 | ACK Flag Count | F | 5 | 5 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1 | 1 |
| 54 | Average Packet Size | Dp | 81.8333 | 100.3333 |
| 55 | Fwd Segment Size Avg | Dp | 2.6667 | 39.6667 |
| 56 | Bwd Segment Size Avg | F | 161 | 161 |
| 57 | Fwd Bytes/Bulk Avg | Fᶜ | 0 | 0 |
| 58 | Fwd Packet/Bulk Avg | Fᶜ | 0 | 0 |
| 59 | Fwd Bulk Rate Avg | Fᶜ | 0 | 0 |
| 60 | Bwd Bytes/Bulk Avg | F | 0 | 0 |
| 61 | Bwd Packet/Bulk Avg | F | 0 | 0 |
| 62 | Bwd Bulk Rate Avg | F | 0 | 0 |
| 63 | Subflow Fwd Packets | F | 0 | 0 |
| 64 | Subflow Fwd Bytes | Fᶜ | 1 | 1 |
| 65 | Subflow Bwd Packets | F | 0 | 0 |
| 66 | Subflow Bwd Bytes | F | 80 | 80 |
| 67 | FWD Init Win Bytes | F | 229 | 229 |
| 68 | Bwd Init Win Bytes | F | 235 | 235 |
| 69 | Fwd Act Data Pkts | Fᶜ | 0 | 0 |
| 70 | Fwd Seg Size Min | F | 20 | 20 |
| 71 | Active Mean | Fᶜ | 49 | 49 |
| 72 | Active Std | Fᶜ | 0 | 0 |
| 73 | Active Max | Fᶜ | 49 | 49 |
| 74 | Active Min | Fᶜ | 49 | 49 |
| 75 | Idle Mean | Fᶜ | 1.1587e+08 | 1.1587e+08 |
| 76 | Idle Std | Fᶜ | 0 | 0 |
| 77 | Idle Max | Fᶜ | 1.1587e+08 | 1.1587e+08 |
| 78 | Idle Min | Fᶜ | 1.1587e+08 | 1.1587e+08 |

---

### DDoS → Benign  (victim = lstm, seed = 48, row = 127)

- primitives: forward packet-length augmentation `p = 31.482` → `p_real = 31` bytes;  timing dilation `α = 1.0489` → `α_real = 1.0489`  |  normalized cost = 0.0706
- **true class = DDoS  →  adversarial prediction = Benign**

#### Base vs. adversarial — all features (raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 63572 | 63572 |
| 1 | Dst Port | F | 80 | 80 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 7.1218e+06 | 7.4699e+06 |
| 4 | Total Fwd Packet | F | 10 | 10 |
| 5 | Total Bwd packets | F | 7 | 7 |
| 6 | Total Length of Fwd Packet | Dp | 20 | 330 |
| 7 | Total Length of Bwd Packet | F | 11595 | 11595 |
| 8 | Fwd Packet Length Max | Dp | 20 | 51 |
| 9 | Fwd Packet Length Min | Dp | 0 | 31 |
| 10 | Fwd Packet Length Mean | Dp | 2 | 33 |
| 11 | Fwd Packet Length Std | I | 6.3246 | 6.3246 |
| 12 | Bwd Packet Length Max | F | 4380 | 4380 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 1656.4286 | 1656.4286 |
| 15 | Bwd Packet Length Std | F | 2119.4155 | 2119.4155 |
| 16 | Flow Bytes/s | R | 1630.8965 | 1596.4000 |
| 17 | Flow Packets/s | R | 2.3870 | 2.2758 |
| 18 | Flow IAT Mean | Dt | 445115.6250 | 466870.7500 |
| 19 | Flow IAT Std | Fᶜ | 1.4935e+06 | 1.4935e+06 |
| 20 | Flow IAT Max | C | 5.9430e+06 | 6.2910e+06 |
| 21 | Flow IAT Min | Fᶜ | 0 | 0 |
| 22 | Fwd IAT Total | Dt | 7.1218e+06 | 7.4699e+06 |
| 23 | Fwd IAT Mean | Dt | 791316.6875 | 829992.4375 |
| 24 | Fwd IAT Std | Dt | 1.9698e+06 | 2.0661e+06 |
| 25 | Fwd IAT Max | Dt | 5.9430e+06 | 6.2334e+06 |
| 26 | Fwd IAT Min | Dt | 0 | 0 |
| 27 | Bwd IAT Total | F | 1.1586e+06 | 1.1586e+06 |
| 28 | Bwd IAT Mean | F | 193095.5000 | 193095.5000 |
| 29 | Bwd IAT Std | F | 466032.8438 | 466032.8438 |
| 30 | Bwd IAT Max | F | 1.1443e+06 | 1.1443e+06 |
| 31 | Bwd IAT Min | F | 16 | 16 |
| 32 | Fwd PSH Flags | F | 1 | 1 |
| 33 | Bwd PSH Flags | F | 1 | 1 |
| 34 | Fwd URG Flags | F | 0 | 0 |
| 35 | Bwd URG Flags | F | 0 | 0 |
| 36 | Fwd Header Length | F | 212 | 212 |
| 37 | Bwd Header Length | F | 152 | 152 |
| 38 | Fwd Packets/s | R | 1.4041 | 1.3387 |
| 39 | Bwd Packets/s | R | 0.9829 | 0.9371 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 4380 | 4380 |
| 42 | Packet Length Mean | Dp | 683.2353 | 701.4706 |
| 43 | Packet Length Std | Dp | 1545.6077 | 1537.1246 |
| 44 | Packet Length Variance | Dp | 2.3889e+06 | 2.3628e+06 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 2 | 2 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 2 | 2 |
| 49 | ACK Flag Count | F | 16 | 16 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 0.7000 | 0.7000 |
| 54 | Average Packet Size | Dp | 683.2353 | 701.4706 |
| 55 | Fwd Segment Size Avg | Dp | 2 | 33 |
| 56 | Bwd Segment Size Avg | F | 1656.4286 | 1656.4286 |
| 57 | Fwd Bytes/Bulk Avg | Fᶜ | 0 | 0 |
| 58 | Fwd Packet/Bulk Avg | Fᶜ | 0 | 0 |
| 59 | Fwd Bulk Rate Avg | Fᶜ | 0 | 0 |
| 60 | Bwd Bytes/Bulk Avg | F | 0 | 0 |
| 61 | Bwd Packet/Bulk Avg | F | 0 | 0 |
| 62 | Bwd Bulk Rate Avg | F | 0 | 0 |
| 63 | Subflow Fwd Packets | F | 0 | 0 |
| 64 | Subflow Fwd Bytes | Fᶜ | 1 | 1 |
| 65 | Subflow Bwd Packets | F | 0 | 0 |
| 66 | Subflow Bwd Bytes | F | 682 | 682 |
| 67 | FWD Init Win Bytes | F | 8192 | 8192 |
| 68 | Bwd Init Win Bytes | F | 229 | 229 |
| 69 | Fwd Act Data Pkts | Fᶜ | 1 | 1 |
| 70 | Fwd Seg Size Min | F | 20 | 20 |
| 71 | Active Mean | Fᶜ | 1.1789e+06 | 1.1789e+06 |
| 72 | Active Std | Fᶜ | 0 | 0 |
| 73 | Active Max | Fᶜ | 1.1789e+06 | 1.1789e+06 |
| 74 | Active Min | Fᶜ | 1.1789e+06 | 1.1789e+06 |
| 75 | Idle Mean | Fᶜ | 5.9430e+06 | 5.9430e+06 |
| 76 | Idle Std | Fᶜ | 0 | 0 |
| 77 | Idle Max | Fᶜ | 5.9430e+06 | 5.9430e+06 |
| 78 | Idle Min | Fᶜ | 5.9430e+06 | 5.9430e+06 |

---

### Recon → Benign  (victim = cnn, seed = 45, row = 179)

- primitives: forward packet-length augmentation `p = 27.709` → `p_real = 28` bytes;  timing dilation `α = 1.0000` → `α_real = 1.0000`  |  normalized cost = 0.0904
- **true class = Recon  →  adversarial prediction = Benign**

#### Base vs. adversarial — all features (raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 43959 | 43959 |
| 1 | Dst Port | F | 1271 | 1271 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 122 | 122 |
| 4 | Total Fwd Packet | F | 1 | 1 |
| 5 | Total Bwd packets | F | 1 | 1 |
| 6 | Total Length of Fwd Packet | Dp | 0 | 28 |
| 7 | Total Length of Bwd Packet | F | 0 | 0 |
| 8 | Fwd Packet Length Max | Dp | 0 | 28 |
| 9 | Fwd Packet Length Min | Dp | 0 | 28 |
| 10 | Fwd Packet Length Mean | Dp | 0 | 28 |
| 11 | Fwd Packet Length Std | I | 0 | 0 |
| 12 | Bwd Packet Length Max | F | 0 | 0 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 0 | 0 |
| 15 | Bwd Packet Length Std | F | 0 | 0 |
| 16 | Flow Bytes/s | R | 0 | 229508.2031 |
| 17 | Flow Packets/s | R | 16393.4434 | 16393.4434 |
| 18 | Flow IAT Mean | Dt | 122 | 122 |
| 19 | Flow IAT Std | Fᶜ | 0 | 0 |
| 20 | Flow IAT Max | C | 122 | 122 |
| 21 | Flow IAT Min | Fᶜ | 122 | 122 |
| 22 | Fwd IAT Total | Dt | 0 | 0 |
| 23 | Fwd IAT Mean | Dt | 0 | 0 |
| 24 | Fwd IAT Std | Dt | 0 | 0 |
| 25 | Fwd IAT Max | Dt | 0 | 0 |
| 26 | Fwd IAT Min | Dt | 0 | 0 |
| 27 | Bwd IAT Total | F | 0 | 0 |
| 28 | Bwd IAT Mean | F | 0 | 0 |
| 29 | Bwd IAT Std | F | 0 | 0 |
| 30 | Bwd IAT Max | F | 0 | 0 |
| 31 | Bwd IAT Min | F | 0 | 0 |
| 32 | Fwd PSH Flags | F | 0 | 0 |
| 33 | Bwd PSH Flags | F | 0 | 0 |
| 34 | Fwd URG Flags | F | 0 | 0 |
| 35 | Bwd URG Flags | F | 0 | 0 |
| 36 | Fwd Header Length | F | 24 | 24 |
| 37 | Bwd Header Length | F | 20 | 20 |
| 38 | Fwd Packets/s | R | 8196.7217 | 8196.7217 |
| 39 | Bwd Packets/s | R | 8196.7217 | 8196.7217 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 0 | 28 |
| 42 | Packet Length Mean | Dp | 0 | 14 |
| 43 | Packet Length Std | Dp | 0 | 19.7990 |
| 44 | Packet Length Variance | Dp | 0 | 392 |
| 45 | FIN Flag Count | F | 0 | 0 |
| 46 | SYN Flag Count | F | 1 | 1 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 0 | 0 |
| 49 | ACK Flag Count | F | 1 | 1 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1 | 1 |
| 54 | Average Packet Size | Dp | 0 | 14 |
| 55 | Fwd Segment Size Avg | Dp | 0 | 28 |
| 56 | Bwd Segment Size Avg | F | 0 | 0 |
| 57 | Fwd Bytes/Bulk Avg | Fᶜ | 0 | 0 |
| 58 | Fwd Packet/Bulk Avg | Fᶜ | 0 | 0 |
| 59 | Fwd Bulk Rate Avg | Fᶜ | 0 | 0 |
| 60 | Bwd Bytes/Bulk Avg | F | 0 | 0 |
| 61 | Bwd Packet/Bulk Avg | F | 0 | 0 |
| 62 | Bwd Bulk Rate Avg | F | 0 | 0 |
| 63 | Subflow Fwd Packets | F | 0 | 0 |
| 64 | Subflow Fwd Bytes | Fᶜ | 0 | 0 |
| 65 | Subflow Bwd Packets | F | 0 | 0 |
| 66 | Subflow Bwd Bytes | F | 0 | 0 |
| 67 | FWD Init Win Bytes | F | 1024 | 1024 |
| 68 | Bwd Init Win Bytes | F | 0 | 0 |
| 69 | Fwd Act Data Pkts | Fᶜ | 0 | 0 |
| 70 | Fwd Seg Size Min | F | 24 | 24 |
| 71 | Active Mean | Fᶜ | 0 | 0 |
| 72 | Active Std | Fᶜ | 0 | 0 |
| 73 | Active Max | Fᶜ | 0 | 0 |
| 74 | Active Min | Fᶜ | 0 | 0 |
| 75 | Idle Mean | Fᶜ | 0 | 0 |
| 76 | Idle Std | Fᶜ | 0 | 0 |
| 77 | Idle Max | Fᶜ | 0 | 0 |
| 78 | Idle Min | Fᶜ | 0 | 0 |

---

### BruteForce → Benign  (victim = serial, seed = 45, row = 31)

- primitives: forward packet-length augmentation `p = 28.565` → `p_real = 29` bytes;  timing dilation `α = 1.2582` → `α_real = 1.2582`  |  normalized cost = 0.0946
- **true class = BruteForce  →  adversarial prediction = Benign**

#### Base vs. adversarial — all features (raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 59010 | 59010 |
| 1 | Dst Port | F | 21 | 21 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 9.3832e+06 | 1.1806e+07 |
| 4 | Total Fwd Packet | F | 11 | 11 |
| 5 | Total Bwd packets | F | 17 | 17 |
| 6 | Total Length of Fwd Packet | Dp | 122 | 441 |
| 7 | Total Length of Bwd Packet | F | 188 | 188 |
| 8 | Fwd Packet Length Max | Dp | 27 | 56 |
| 9 | Fwd Packet Length Min | Dp | 0 | 29 |
| 10 | Fwd Packet Length Mean | Dp | 11.0909 | 40.0909 |
| 11 | Fwd Packet Length Std | I | 9.5964 | 9.5964 |
| 12 | Bwd Packet Length Max | F | 34 | 34 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 11.0588 | 11.0588 |
| 15 | Bwd Packet Length Std | F | 14.2323 | 14.2323 |
| 16 | Flow Bytes/s | R | 33.0377 | 53.2779 |
| 17 | Flow Packets/s | R | 2.9841 | 2.3717 |
| 18 | Flow IAT Mean | Dt | 347526.4375 | 437259.7812 |
| 19 | Flow IAT Std | Fᶜ | 990693.0625 | 990693.0625 |
| 20 | Flow IAT Max | C | 3.5113e+06 | 5.9341e+06 |
| 21 | Flow IAT Min | Fᶜ | 2 | 2 |
| 22 | Fwd IAT Total | Dt | 9.3831e+06 | 1.1806e+07 |
| 23 | Fwd IAT Mean | Dt | 938314.1250 | 1.1806e+06 |
| 24 | Fwd IAT Std | Dt | 1.5196e+06 | 1.9120e+06 |
| 25 | Fwd IAT Max | Dt | 3.5554e+06 | 4.4734e+06 |
| 26 | Fwd IAT Min | Dt | 3 | 4 |
| 27 | Bwd IAT Total | F | 9.3832e+06 | 9.3832e+06 |
| 28 | Bwd IAT Mean | F | 586447.1250 | 586447.1250 |
| 29 | Bwd IAT Std | F | 1.2457e+06 | 1.2457e+06 |
| 30 | Bwd IAT Max | F | 3.5113e+06 | 3.5113e+06 |
| 31 | Bwd IAT Min | F | 2 | 2 |
| 32 | Fwd PSH Flags | F | 7 | 7 |
| 33 | Bwd PSH Flags | F | 7 | 7 |
| 34 | Fwd URG Flags | F | 0 | 0 |
| 35 | Bwd URG Flags | F | 0 | 0 |
| 36 | Fwd Header Length | F | 360 | 360 |
| 37 | Bwd Header Length | F | 528 | 528 |
| 38 | Fwd Packets/s | R | 1.1723 | 0.9317 |
| 39 | Bwd Packets/s | R | 1.8117 | 1.4399 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 34 | 56 |
| 42 | Packet Length Mean | Dp | 11.0714 | 22.4643 |
| 43 | Packet Length Std | Dp | 12.4154 | 19.0428 |
| 44 | Packet Length Variance | Dp | 154.1429 | 362.6283 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 2 | 2 |
| 47 | RST Flag Count | F | 2 | 2 |
| 48 | PSH Flag Count | F | 14 | 14 |
| 49 | ACK Flag Count | F | 25 | 25 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1.5455 | 1.5455 |
| 54 | Average Packet Size | Dp | 11.0714 | 22.4643 |
| 55 | Fwd Segment Size Avg | Dp | 11.0909 | 40.0909 |
| 56 | Bwd Segment Size Avg | F | 11.0588 | 11.0588 |
| 57 | Fwd Bytes/Bulk Avg | Fᶜ | 0 | 0 |
| 58 | Fwd Packet/Bulk Avg | Fᶜ | 0 | 0 |
| 59 | Fwd Bulk Rate Avg | Fᶜ | 0 | 0 |
| 60 | Bwd Bytes/Bulk Avg | F | 0 | 0 |
| 61 | Bwd Packet/Bulk Avg | F | 0 | 0 |
| 62 | Bwd Bulk Rate Avg | F | 0 | 0 |
| 63 | Subflow Fwd Packets | F | 0 | 0 |
| 64 | Subflow Fwd Bytes | Fᶜ | 4 | 4 |
| 65 | Subflow Bwd Packets | F | 0 | 0 |
| 66 | Subflow Bwd Bytes | F | 6 | 6 |
| 67 | FWD Init Win Bytes | F | 29200 | 29200 |
| 68 | Bwd Init Win Bytes | F | 0 | 0 |
| 69 | Fwd Act Data Pkts | Fᶜ | 7 | 7 |
| 70 | Fwd Seg Size Min | F | 32 | 32 |
| 71 | Active Mean | Fᶜ | 0 | 0 |
| 72 | Active Std | Fᶜ | 0 | 0 |
| 73 | Active Max | Fᶜ | 0 | 0 |
| 74 | Active Min | Fᶜ | 0 | 0 |
| 75 | Idle Mean | Fᶜ | 0 | 0 |
| 76 | Idle Std | Fᶜ | 0 | 0 |
| 77 | Idle Max | Fᶜ | 0 | 0 |
| 78 | Idle Min | Fᶜ | 0 | 0 |

---

## 4. Feature deltas only (changed features per best example)

### DoS → Benign  (victim = cnn, seed = 45, row = 14)

| idx | feature | role | base | → | adversarial | Δ (adv − base) |
|---:|---|:--:|---:|:--:|---:|---:|
| 3 | Flow Duration | C | 1.1588e+08 | → | 1.1590e+08 | 24344 |
| 6 | Total Length of Fwd Packet | Dp | 8 | → | 119 | 111 |
| 8 | Fwd Packet Length Max | Dp | 8 | → | 45 | 37 |
| 9 | Fwd Packet Length Min | Dp | 0 | → | 37 | 37 |
| 10 | Fwd Packet Length Mean | Dp | 2.6667 | → | 39.6667 | 37.0000 |
| 16 | Flow Bytes/s | R | 4.2372 | → | 5.1940 | 0.9568 |
| 18 | Flow IAT Mean | Dt | 2.3176e+07 | → | 2.3181e+07 | 4868 |
| 20 | Flow IAT Max | C | 1.1587e+08 | → | 1.1590e+08 | 24344 |
| 22 | Fwd IAT Total | Dt | 1.1588e+08 | → | 1.1590e+08 | 24344 |
| 23 | Fwd IAT Mean | Dt | 5.7940e+07 | → | 5.7952e+07 | 12172 |
| 24 | Fwd IAT Std | Dt | 8.1928e+07 | → | 8.1945e+07 | 17208 |
| 25 | Fwd IAT Max | Dt | 1.1587e+08 | → | 1.1590e+08 | 24336 |
| 26 | Fwd IAT Min | Dt | 7704 | → | 7706 | 2 |
| 42 | Packet Length Mean | Dp | 81.8333 | → | 100.3333 | 18.5000 |
| 43 | Packet Length Std | Dp | 196.5568 | → | 188.4947 | -8.0620 |
| 44 | Packet Length Variance | Dp | 38634.5664 | → | 35530.2656 | -3104.3008 |
| 54 | Average Packet Size | Dp | 81.8333 | → | 100.3333 | 18.5000 |
| 55 | Fwd Segment Size Avg | Dp | 2.6667 | → | 39.6667 | 37.0000 |

### DDoS → Benign  (victim = lstm, seed = 48, row = 127)

| idx | feature | role | base | → | adversarial | Δ (adv − base) |
|---:|---|:--:|---:|:--:|---:|---:|
| 3 | Flow Duration | C | 7.1218e+06 | → | 7.4699e+06 | 348082 |
| 6 | Total Length of Fwd Packet | Dp | 20 | → | 330 | 310 |
| 8 | Fwd Packet Length Max | Dp | 20 | → | 51 | 31 |
| 9 | Fwd Packet Length Min | Dp | 0 | → | 31 | 31 |
| 10 | Fwd Packet Length Mean | Dp | 2 | → | 33 | 31 |
| 16 | Flow Bytes/s | R | 1630.8965 | → | 1596.4000 | -34.4965 |
| 17 | Flow Packets/s | R | 2.3870 | → | 2.2758 | -0.1112 |
| 18 | Flow IAT Mean | Dt | 445115.6250 | → | 466870.7500 | 21755.1250 |
| 20 | Flow IAT Max | C | 5.9430e+06 | → | 6.2910e+06 | 348082 |
| 22 | Fwd IAT Total | Dt | 7.1218e+06 | → | 7.4699e+06 | 348082 |
| 23 | Fwd IAT Mean | Dt | 791316.6875 | → | 829992.4375 | 38675.7500 |
| 24 | Fwd IAT Std | Dt | 1.9698e+06 | → | 2.0661e+06 | 96274 |
| 25 | Fwd IAT Max | Dt | 5.9430e+06 | → | 6.2334e+06 | 290463 |
| 38 | Fwd Packets/s | R | 1.4041 | → | 1.3387 | -0.0654 |
| 39 | Bwd Packets/s | R | 0.9829 | → | 0.9371 | -0.0458 |
| 42 | Packet Length Mean | Dp | 683.2353 | → | 701.4706 | 18.2353 |
| 43 | Packet Length Std | Dp | 1545.6077 | → | 1537.1246 | -8.4830 |
| 44 | Packet Length Variance | Dp | 2.3889e+06 | → | 2.3628e+06 | -26151 |
| 54 | Average Packet Size | Dp | 683.2353 | → | 701.4706 | 18.2353 |
| 55 | Fwd Segment Size Avg | Dp | 2 | → | 33 | 31 |

### Recon → Benign  (victim = cnn, seed = 45, row = 179)

| idx | feature | role | base | → | adversarial | Δ (adv − base) |
|---:|---|:--:|---:|:--:|---:|---:|
| 6 | Total Length of Fwd Packet | Dp | 0 | → | 28 | 28 |
| 8 | Fwd Packet Length Max | Dp | 0 | → | 28 | 28 |
| 9 | Fwd Packet Length Min | Dp | 0 | → | 28 | 28 |
| 10 | Fwd Packet Length Mean | Dp | 0 | → | 28 | 28 |
| 16 | Flow Bytes/s | R | 0 | → | 229508.2031 | 229508.2031 |
| 41 | Packet Length Max | C | 0 | → | 28 | 28 |
| 42 | Packet Length Mean | Dp | 0 | → | 14 | 14 |
| 43 | Packet Length Std | Dp | 0 | → | 19.7990 | 19.7990 |
| 44 | Packet Length Variance | Dp | 0 | → | 392 | 392 |
| 54 | Average Packet Size | Dp | 0 | → | 14 | 14 |
| 55 | Fwd Segment Size Avg | Dp | 0 | → | 28 | 28 |

### BruteForce → Benign  (victim = serial, seed = 45, row = 31)

| idx | feature | role | base | → | adversarial | Δ (adv − base) |
|---:|---|:--:|---:|:--:|---:|---:|
| 3 | Flow Duration | C | 9.3832e+06 | → | 1.1806e+07 | 2.4228e+06 |
| 6 | Total Length of Fwd Packet | Dp | 122 | → | 441 | 319 |
| 8 | Fwd Packet Length Max | Dp | 27 | → | 56 | 29 |
| 9 | Fwd Packet Length Min | Dp | 0 | → | 29 | 29 |
| 10 | Fwd Packet Length Mean | Dp | 11.0909 | → | 40.0909 | 29.0000 |
| 16 | Flow Bytes/s | R | 33.0377 | → | 53.2779 | 20.2402 |
| 17 | Flow Packets/s | R | 2.9841 | → | 2.3717 | -0.6124 |
| 18 | Flow IAT Mean | Dt | 347526.4375 | → | 437259.7812 | 89733.3438 |
| 20 | Flow IAT Max | C | 3.5113e+06 | → | 5.9341e+06 | 2.4228e+06 |
| 22 | Fwd IAT Total | Dt | 9.3831e+06 | → | 1.1806e+07 | 2.4228e+06 |
| 23 | Fwd IAT Mean | Dt | 938314.1250 | → | 1.1806e+06 | 242280 |
| 24 | Fwd IAT Std | Dt | 1.5196e+06 | → | 1.9120e+06 | 392383.8750 |
| 25 | Fwd IAT Max | Dt | 3.5554e+06 | → | 4.4734e+06 | 918024 |
| 26 | Fwd IAT Min | Dt | 3 | → | 4 | 1 |
| 38 | Fwd Packets/s | R | 1.1723 | → | 0.9317 | -0.2406 |
| 39 | Bwd Packets/s | R | 1.8117 | → | 1.4399 | -0.3718 |
| 41 | Packet Length Max | C | 34 | → | 56 | 22 |
| 42 | Packet Length Mean | Dp | 11.0714 | → | 22.4643 | 11.3929 |
| 43 | Packet Length Std | Dp | 12.4154 | → | 19.0428 | 6.6274 |
| 44 | Packet Length Variance | Dp | 154.1429 | → | 362.6283 | 208.4854 |
| 54 | Average Packet Size | Dp | 11.0714 | → | 22.4643 | 11.3929 |
| 55 | Fwd Segment Size Avg | Dp | 11.0909 | → | 40.0909 | 29.0000 |
