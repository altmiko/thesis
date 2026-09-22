# CICIDS2017-DistriNet — realizability-aware primitive-control adversarial examples

Successful adversarial network flows from the **primitive-control** attack: the attacker optimizes only two primitives per flow — forward packet-length augmentation `p` (bytes/fwd packet) and forward timing dilation `α` (delay factor) — then applies a discrete realizability projection (round `p` to integer bytes; µs-quantize timing) and recomputes every dependent CICFlowMeter feature.

- **Artifacts:** `outputs/cicids2017_primitive_attack/attack_artifacts/<Class>_<victim>_seed42.npz`
- **Class ids:** `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`
- **Feature role:** `P` = primitive-controlled, `D` = direct-derived (exact identity), `C` = conditional-derived (structure-dependent / conservative projection), `R` = rate (count·byte / projected duration), `F` = frozen (copied verbatim), `Fᶜ` = Level-C frozen (would change under real packet edits but not reconstructable from the aggregate flow).
- Values are the **pristine raw** original CICFlowMeter units; the example per class is the smallest normalized-cost row that was clean-correct and evaded to **Benign** while passing PAVE ∧ mined ∧ internal-realizability (strict valid).

---

## DoS → Benign  (victim = cnn, row = 14)

- primitives: forward packet-length augmentation `p = 40.994` → `p_real = 41` bytes;  timing dilation `α = 1.0002` → `α_real = 1.0002`  |  normalized cost = 0.0720
- **true class = DoS  →  adversarial prediction = Benign**

### Base vs. adversarial (all 79 features, raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 54392 | 54392 |
| 1 | Dst Port | F | 80 | 80 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 1.1588e+08 | 1.1590e+08 |
| 4 | Total Fwd Packet | F | 3 | 3 |
| 5 | Total Bwd packets | F | 3 | 3 |
| 6 | Total Length of Fwd Packet | P | 8 | 131 |
| 7 | Total Length of Bwd Packet | F | 483 | 483 |
| 8 | Fwd Packet Length Max | P | 8 | 49 |
| 9 | Fwd Packet Length Min | P | 0 | 41 |
| 10 | Fwd Packet Length Mean | D | 2.6667 | 43.6667 |
| 11 | Fwd Packet Length Std | F | 4.6188 | 4.6188 |
| 12 | Bwd Packet Length Max | F | 483 | 483 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 161 | 161 |
| 15 | Bwd Packet Length Std | F | 278.8602 | 278.8602 |
| 16 | Flow Bytes/s | R | 4.2372 | 5.2976 |
| 17 | Flow Packets/s | R | 0.0518 | 0.0518 |
| 18 | Flow IAT Mean | D | 2.3176e+07 | 2.3180e+07 |
| 19 | Flow IAT Std | Fᶜ | 5.1818e+07 | 5.1818e+07 |
| 20 | Flow IAT Max | C | 1.1587e+08 | 1.1589e+08 |
| 21 | Flow IAT Min | Fᶜ | 2 | 2 |
| 22 | Fwd IAT Total | P | 1.1588e+08 | 1.1590e+08 |
| 23 | Fwd IAT Mean | D | 5.7940e+07 | 5.7951e+07 |
| 24 | Fwd IAT Std | P | 8.1928e+07 | 8.1944e+07 |
| 25 | Fwd IAT Max | P | 1.1587e+08 | 1.1589e+08 |
| 26 | Fwd IAT Min | P | 7704 | 7705 |
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
| 42 | Packet Length Mean | D | 81.8333 | 102.3333 |
| 43 | Packet Length Std | D | 196.5568 | 187.7335 |
| 44 | Packet Length Variance | D | 38634.5664 | 35243.8672 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 0 | 0 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 2 | 2 |
| 49 | ACK Flag Count | F | 5 | 5 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1 | 1 |
| 54 | Average Packet Size | D | 81.8333 | 102.3333 |
| 55 | Fwd Segment Size Avg | D | 2.6667 | 43.6667 |
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
| 71 | Active Mean | F | 49 | 49 |
| 72 | Active Std | F | 0 | 0 |
| 73 | Active Max | F | 49 | 49 |
| 74 | Active Min | F | 49 | 49 |
| 75 | Idle Mean | F | 1.1587e+08 | 1.1587e+08 |
| 76 | Idle Std | F | 0 | 0 |
| 77 | Idle Max | F | 1.1587e+08 | 1.1587e+08 |
| 78 | Idle Min | F | 1.1587e+08 | 1.1587e+08 |

### Delta — changed features only

| idx | feature | role | from | → | to |
|---:|---|:--:|---:|:--:|---:|
| 3 | Flow Duration | C | 1.1588e+08 | → | 1.1590e+08 |
| 6 | Total Length of Fwd Packet | P | 8 | → | 131 |
| 8 | Fwd Packet Length Max | P | 8 | → | 49 |
| 9 | Fwd Packet Length Min | P | 0 | → | 41 |
| 10 | Fwd Packet Length Mean | D | 2.6667 | → | 43.6667 |
| 16 | Flow Bytes/s | R | 4.2372 | → | 5.2976 |
| 18 | Flow IAT Mean | D | 2.3176e+07 | → | 2.3180e+07 |
| 20 | Flow IAT Max | C | 1.1587e+08 | → | 1.1589e+08 |
| 22 | Fwd IAT Total | P | 1.1588e+08 | → | 1.1590e+08 |
| 23 | Fwd IAT Mean | D | 5.7940e+07 | → | 5.7951e+07 |
| 24 | Fwd IAT Std | P | 8.1928e+07 | → | 8.1944e+07 |
| 25 | Fwd IAT Max | P | 1.1587e+08 | → | 1.1589e+08 |
| 26 | Fwd IAT Min | P | 7704 | → | 7705 |
| 42 | Packet Length Mean | D | 81.8333 | → | 102.3333 |
| 43 | Packet Length Std | D | 196.5568 | → | 187.7335 |
| 44 | Packet Length Variance | D | 38634.5664 | → | 35243.8672 |
| 54 | Average Packet Size | D | 81.8333 | → | 102.3333 |
| 55 | Fwd Segment Size Avg | D | 2.6667 | → | 43.6667 |

**Impossible-case checks (all must be False):** fwd mean<min=False, fwd max>total=False, pkt mean<min=False, fwd IAT max>total=False, fwd IAT total>dur=False, flow IAT mean>max=False, any rate<0=False, duration<=0=False

---

## DDoS → Benign  (victim = lstm, row = 127)

- primitives: forward packet-length augmentation `p = 29.656` → `p_real = 30` bytes;  timing dilation `α = 1.1251` → `α_real = 1.1251`  |  normalized cost = 0.0787
- **true class = DDoS  →  adversarial prediction = Benign**

### Base vs. adversarial (all 79 features, raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 63572 | 63572 |
| 1 | Dst Port | F | 80 | 80 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 7.1218e+06 | 8.0128e+06 |
| 4 | Total Fwd Packet | F | 10 | 10 |
| 5 | Total Bwd packets | F | 7 | 7 |
| 6 | Total Length of Fwd Packet | P | 20 | 320 |
| 7 | Total Length of Bwd Packet | F | 11595 | 11595 |
| 8 | Fwd Packet Length Max | P | 20 | 50 |
| 9 | Fwd Packet Length Min | P | 0 | 30 |
| 10 | Fwd Packet Length Mean | D | 2 | 32 |
| 11 | Fwd Packet Length Std | F | 6.3246 | 6.3246 |
| 12 | Bwd Packet Length Max | F | 4380 | 4380 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 1656.4286 | 1656.4286 |
| 15 | Bwd Packet Length Std | F | 2119.4155 | 2119.4155 |
| 16 | Flow Bytes/s | R | 1630.8965 | 1486.9917 |
| 17 | Flow Packets/s | R | 2.3870 | 2.1216 |
| 18 | Flow IAT Mean | D | 445115.6250 | 500801.3750 |
| 19 | Flow IAT Std | Fᶜ | 1.4935e+06 | 1.4935e+06 |
| 20 | Flow IAT Max | C | 5.9430e+06 | 6.8339e+06 |
| 21 | Flow IAT Min | Fᶜ | 0 | 0 |
| 22 | Fwd IAT Total | P | 7.1218e+06 | 8.0128e+06 |
| 23 | Fwd IAT Mean | D | 791316.6875 | 890313.5625 |
| 24 | Fwd IAT Std | P | 1.9698e+06 | 2.2162e+06 |
| 25 | Fwd IAT Max | P | 5.9430e+06 | 6.6865e+06 |
| 26 | Fwd IAT Min | P | 0 | 0 |
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
| 38 | Fwd Packets/s | R | 1.4041 | 1.2480 |
| 39 | Bwd Packets/s | R | 0.9829 | 0.8736 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 4380 | 4380 |
| 42 | Packet Length Mean | D | 683.2353 | 700.8823 |
| 43 | Packet Length Std | D | 1545.6077 | 1537.3965 |
| 44 | Packet Length Variance | D | 2.3889e+06 | 2.3636e+06 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 2 | 2 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 2 | 2 |
| 49 | ACK Flag Count | F | 16 | 16 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 0.7000 | 0.7000 |
| 54 | Average Packet Size | D | 683.2353 | 700.8823 |
| 55 | Fwd Segment Size Avg | D | 2 | 32 |
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
| 71 | Active Mean | F | 1.1789e+06 | 1.1789e+06 |
| 72 | Active Std | F | 0 | 0 |
| 73 | Active Max | F | 1.1789e+06 | 1.1789e+06 |
| 74 | Active Min | F | 1.1789e+06 | 1.1789e+06 |
| 75 | Idle Mean | F | 5.9430e+06 | 5.9430e+06 |
| 76 | Idle Std | F | 0 | 0 |
| 77 | Idle Max | F | 5.9430e+06 | 5.9430e+06 |
| 78 | Idle Min | F | 5.9430e+06 | 5.9430e+06 |

### Delta — changed features only

| idx | feature | role | from | → | to |
|---:|---|:--:|---:|:--:|---:|
| 3 | Flow Duration | C | 7.1218e+06 | → | 8.0128e+06 |
| 6 | Total Length of Fwd Packet | P | 20 | → | 320 |
| 8 | Fwd Packet Length Max | P | 20 | → | 50 |
| 9 | Fwd Packet Length Min | P | 0 | → | 30 |
| 10 | Fwd Packet Length Mean | D | 2 | → | 32 |
| 16 | Flow Bytes/s | R | 1630.8965 | → | 1486.9917 |
| 17 | Flow Packets/s | R | 2.3870 | → | 2.1216 |
| 18 | Flow IAT Mean | D | 445115.6250 | → | 500801.3750 |
| 20 | Flow IAT Max | C | 5.9430e+06 | → | 6.8339e+06 |
| 22 | Fwd IAT Total | P | 7.1218e+06 | → | 8.0128e+06 |
| 23 | Fwd IAT Mean | D | 791316.6875 | → | 890313.5625 |
| 24 | Fwd IAT Std | P | 1.9698e+06 | → | 2.2162e+06 |
| 25 | Fwd IAT Max | P | 5.9430e+06 | → | 6.6865e+06 |
| 38 | Fwd Packets/s | R | 1.4041 | → | 1.2480 |
| 39 | Bwd Packets/s | R | 0.9829 | → | 0.8736 |
| 42 | Packet Length Mean | D | 683.2353 | → | 700.8823 |
| 43 | Packet Length Std | D | 1545.6077 | → | 1537.3965 |
| 44 | Packet Length Variance | D | 2.3889e+06 | → | 2.3636e+06 |
| 54 | Average Packet Size | D | 683.2353 | → | 700.8823 |
| 55 | Fwd Segment Size Avg | D | 2 | → | 32 |

**Impossible-case checks (all must be False):** fwd mean<min=False, fwd max>total=False, pkt mean<min=False, fwd IAT max>total=False, fwd IAT total>dur=False, flow IAT mean>max=False, any rate<0=False, duration<=0=False

---

## Recon → Benign  (victim = cnn, row = 179)

- primitives: forward packet-length augmentation `p = 30.915` → `p_real = 31` bytes;  timing dilation `α = 3.0269` → `α_real = 1.0000`  |  normalized cost = 0.1000
- **true class = Recon  →  adversarial prediction = Benign**

### Base vs. adversarial (all 79 features, raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 43959 | 43959 |
| 1 | Dst Port | F | 1271 | 1271 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 122 | 122 |
| 4 | Total Fwd Packet | F | 1 | 1 |
| 5 | Total Bwd packets | F | 1 | 1 |
| 6 | Total Length of Fwd Packet | P | 0 | 31 |
| 7 | Total Length of Bwd Packet | F | 0 | 0 |
| 8 | Fwd Packet Length Max | P | 0 | 31 |
| 9 | Fwd Packet Length Min | P | 0 | 31 |
| 10 | Fwd Packet Length Mean | D | 0 | 31 |
| 11 | Fwd Packet Length Std | F | 0 | 0 |
| 12 | Bwd Packet Length Max | F | 0 | 0 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 0 | 0 |
| 15 | Bwd Packet Length Std | F | 0 | 0 |
| 16 | Flow Bytes/s | R | 0 | 254098.3594 |
| 17 | Flow Packets/s | R | 16393.4434 | 16393.4434 |
| 18 | Flow IAT Mean | D | 122 | 122 |
| 19 | Flow IAT Std | Fᶜ | 0 | 0 |
| 20 | Flow IAT Max | C | 122 | 122 |
| 21 | Flow IAT Min | Fᶜ | 122 | 122 |
| 22 | Fwd IAT Total | P | 0 | 0 |
| 23 | Fwd IAT Mean | D | 0 | 0 |
| 24 | Fwd IAT Std | P | 0 | 0 |
| 25 | Fwd IAT Max | P | 0 | 0 |
| 26 | Fwd IAT Min | P | 0 | 0 |
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
| 41 | Packet Length Max | C | 0 | 31 |
| 42 | Packet Length Mean | D | 0 | 15.5000 |
| 43 | Packet Length Std | D | 0 | 21.9203 |
| 44 | Packet Length Variance | D | 0 | 480.5000 |
| 45 | FIN Flag Count | F | 0 | 0 |
| 46 | SYN Flag Count | F | 1 | 1 |
| 47 | RST Flag Count | F | 1 | 1 |
| 48 | PSH Flag Count | F | 0 | 0 |
| 49 | ACK Flag Count | F | 1 | 1 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1 | 1 |
| 54 | Average Packet Size | D | 0 | 15.5000 |
| 55 | Fwd Segment Size Avg | D | 0 | 31 |
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
| 71 | Active Mean | F | 0 | 0 |
| 72 | Active Std | F | 0 | 0 |
| 73 | Active Max | F | 0 | 0 |
| 74 | Active Min | F | 0 | 0 |
| 75 | Idle Mean | F | 0 | 0 |
| 76 | Idle Std | F | 0 | 0 |
| 77 | Idle Max | F | 0 | 0 |
| 78 | Idle Min | F | 0 | 0 |

### Delta — changed features only

| idx | feature | role | from | → | to |
|---:|---|:--:|---:|:--:|---:|
| 6 | Total Length of Fwd Packet | P | 0 | → | 31 |
| 8 | Fwd Packet Length Max | P | 0 | → | 31 |
| 9 | Fwd Packet Length Min | P | 0 | → | 31 |
| 10 | Fwd Packet Length Mean | D | 0 | → | 31 |
| 16 | Flow Bytes/s | R | 0 | → | 254098.3594 |
| 41 | Packet Length Max | C | 0 | → | 31 |
| 42 | Packet Length Mean | D | 0 | → | 15.5000 |
| 43 | Packet Length Std | D | 0 | → | 21.9203 |
| 44 | Packet Length Variance | D | 0 | → | 480.5000 |
| 54 | Average Packet Size | D | 0 | → | 15.5000 |
| 55 | Fwd Segment Size Avg | D | 0 | → | 31 |

**Impossible-case checks (all must be False):** fwd mean<min=False, fwd max>total=False, pkt mean<min=False, fwd IAT max>total=False, fwd IAT total>dur=False, flow IAT mean>max=False, any rate<0=False, duration<=0=False

---

## BruteForce → Benign  (victim = serial, row = 270)

- primitives: forward packet-length augmentation `p = 30.235` → `p_real = 30` bytes;  timing dilation `α = 1.2825` → `α_real = 1.2825`  |  normalized cost = 0.1021
- **true class = BruteForce  →  adversarial prediction = Benign**

### Base vs. adversarial (all 79 features, raw units)

| idx | feature | role | base (raw) | adversarial (raw) |
|---:|---|:--:|---:|---:|
| 0 | Src Port | F | 59502 | 59502 |
| 1 | Dst Port | F | 21 | 21 |
| 2 | Protocol | F | 6 | 6 |
| 3 | Flow Duration | C | 9.9320e+06 | 1.2738e+07 |
| 4 | Total Fwd Packet | F | 11 | 11 |
| 5 | Total Bwd packets | F | 17 | 17 |
| 6 | Total Length of Fwd Packet | P | 122 | 452 |
| 7 | Total Length of Bwd Packet | F | 188 | 188 |
| 8 | Fwd Packet Length Max | P | 27 | 57 |
| 9 | Fwd Packet Length Min | P | 0 | 30 |
| 10 | Fwd Packet Length Mean | D | 11.0909 | 41.0909 |
| 11 | Fwd Packet Length Std | F | 9.5964 | 9.5964 |
| 12 | Bwd Packet Length Max | F | 34 | 34 |
| 13 | Bwd Packet Length Min | F | 0 | 0 |
| 14 | Bwd Packet Length Mean | F | 11.0588 | 11.0588 |
| 15 | Bwd Packet Length Std | F | 14.2323 | 14.2323 |
| 16 | Flow Bytes/s | R | 31.2124 | 50.2435 |
| 17 | Flow Packets/s | R | 2.8192 | 2.1982 |
| 18 | Flow IAT Mean | D | 367850.2500 | 471776.6250 |
| 19 | Flow IAT Std | Fᶜ | 1.0453e+06 | 1.0453e+06 |
| 20 | Flow IAT Max | C | 3.5157e+06 | 6.3217e+06 |
| 21 | Flow IAT Min | Fᶜ | 3 | 3 |
| 22 | Fwd IAT Total | P | 9.9319e+06 | 1.2738e+07 |
| 23 | Fwd IAT Mean | D | 993191.1875 | 1.2738e+06 |
| 24 | Fwd IAT Std | P | 1.6007e+06 | 2.0529e+06 |
| 25 | Fwd IAT Max | P | 3.5599e+06 | 4.5656e+06 |
| 26 | Fwd IAT Min | P | 4 | 5 |
| 27 | Bwd IAT Total | F | 9.9319e+06 | 9.9319e+06 |
| 28 | Bwd IAT Mean | F | 620741.1875 | 620741.1875 |
| 29 | Bwd IAT Std | F | 1.3140e+06 | 1.3140e+06 |
| 30 | Bwd IAT Max | F | 3.5157e+06 | 3.5157e+06 |
| 31 | Bwd IAT Min | F | 3 | 3 |
| 32 | Fwd PSH Flags | F | 7 | 7 |
| 33 | Bwd PSH Flags | F | 7 | 7 |
| 34 | Fwd URG Flags | F | 0 | 0 |
| 35 | Bwd URG Flags | F | 0 | 0 |
| 36 | Fwd Header Length | F | 360 | 360 |
| 37 | Bwd Header Length | F | 528 | 528 |
| 38 | Fwd Packets/s | R | 1.1075 | 0.8636 |
| 39 | Bwd Packets/s | R | 1.7116 | 1.3346 |
| 40 | Packet Length Min | C | 0 | 0 |
| 41 | Packet Length Max | C | 34 | 57 |
| 42 | Packet Length Mean | D | 11.0714 | 22.8571 |
| 43 | Packet Length Std | D | 12.4154 | 19.4226 |
| 44 | Packet Length Variance | D | 154.1429 | 377.2381 |
| 45 | FIN Flag Count | F | 2 | 2 |
| 46 | SYN Flag Count | F | 2 | 2 |
| 47 | RST Flag Count | F | 2 | 2 |
| 48 | PSH Flag Count | F | 14 | 14 |
| 49 | ACK Flag Count | F | 25 | 25 |
| 50 | URG Flag Count | F | 0 | 0 |
| 51 | CWR Flag Count | F | 0 | 0 |
| 52 | ECE Flag Count | F | 0 | 0 |
| 53 | Down/Up Ratio | F | 1.5455 | 1.5455 |
| 54 | Average Packet Size | D | 11.0714 | 22.8571 |
| 55 | Fwd Segment Size Avg | D | 11.0909 | 41.0909 |
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
| 71 | Active Mean | F | 0 | 0 |
| 72 | Active Std | F | 0 | 0 |
| 73 | Active Max | F | 0 | 0 |
| 74 | Active Min | F | 0 | 0 |
| 75 | Idle Mean | F | 0 | 0 |
| 76 | Idle Std | F | 0 | 0 |
| 77 | Idle Max | F | 0 | 0 |
| 78 | Idle Min | F | 0 | 0 |

### Delta — changed features only

| idx | feature | role | from | → | to |
|---:|---|:--:|---:|:--:|---:|
| 3 | Flow Duration | C | 9.9320e+06 | → | 1.2738e+07 |
| 6 | Total Length of Fwd Packet | P | 122 | → | 452 |
| 8 | Fwd Packet Length Max | P | 27 | → | 57 |
| 9 | Fwd Packet Length Min | P | 0 | → | 30 |
| 10 | Fwd Packet Length Mean | D | 11.0909 | → | 41.0909 |
| 16 | Flow Bytes/s | R | 31.2124 | → | 50.2435 |
| 17 | Flow Packets/s | R | 2.8192 | → | 2.1982 |
| 18 | Flow IAT Mean | D | 367850.2500 | → | 471776.6250 |
| 20 | Flow IAT Max | C | 3.5157e+06 | → | 6.3217e+06 |
| 22 | Fwd IAT Total | P | 9.9319e+06 | → | 1.2738e+07 |
| 23 | Fwd IAT Mean | D | 993191.1875 | → | 1.2738e+06 |
| 24 | Fwd IAT Std | P | 1.6007e+06 | → | 2.0529e+06 |
| 25 | Fwd IAT Max | P | 3.5599e+06 | → | 4.5656e+06 |
| 26 | Fwd IAT Min | P | 4 | → | 5 |
| 38 | Fwd Packets/s | R | 1.1075 | → | 0.8636 |
| 39 | Bwd Packets/s | R | 1.7116 | → | 1.3346 |
| 41 | Packet Length Max | C | 34 | → | 57 |
| 42 | Packet Length Mean | D | 11.0714 | → | 22.8571 |
| 43 | Packet Length Std | D | 12.4154 | → | 19.4226 |
| 44 | Packet Length Variance | D | 154.1429 | → | 377.2381 |
| 54 | Average Packet Size | D | 11.0714 | → | 22.8571 |
| 55 | Fwd Segment Size Avg | D | 11.0909 | → | 41.0909 |

**Impossible-case checks (all must be False):** fwd mean<min=False, fwd max>total=False, pkt mean<min=False, fwd IAT max>total=False, fwd IAT total>dur=False, flow IAT mean>max=False, any rate<0=False, duration<=0=False

---
