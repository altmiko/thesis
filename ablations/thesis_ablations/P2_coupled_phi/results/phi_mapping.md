# P2 reduced recomputation path: definition

Derived from code by `ablations/common/phi_mapping.py` before any P2 run. The P2 search (`recompute_mode="direct_only"`) writes the direct features exactly as φ does and leaves every derived feature at the source flow's value. The reference (`recompute_mode="full_phi"`) is φ unchanged.

# cicids2017_distrinet

## φ dependency map (traced from code)

Source: `src/attack/realizability/cicids2017.py:CICIDS2017PrimitiveModel.generate` (sha256 of the function source `ebed83e1a77fd33d…`). Rule: direct = the written value's backward slice (local assignments of generate) reaches the primitive controls and source columns only; derived = the slice reads a value that generate writes to another feature.

### primitive → direct (base) quantities → recomputed dependent features

| primitive | direct / base features (kept in P2) | dependent features recomputed by φ (frozen at source in P2) |
|---|---|---|
| `p` | Total Length of Fwd Packet, Fwd Packet Length Min, Fwd Packet Length Max | Fwd Packet Length Mean, Fwd Segment Size Avg, Packet Length Max, Packet Length Min, Packet Length Mean, Average Packet Size, Packet Length Variance, Packet Length Std, Flow Bytes/s |
| `delay` | Fwd IAT Total, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min | Fwd IAT Mean, Flow Duration, Flow IAT Mean, Flow IAT Max, Fwd Packets/s, Bwd Packets/s, Flow Packets/s, Flow Bytes/s |
| `shape` | Fwd IAT Std, Fwd IAT Max, Fwd IAT Min | - |

### Write sites

| feature | kind | φ code block | primitives | gated by | parents (written values read) | direct bases | written value |
|---|---|---|---|---|---|---|---|
| Total Length of Fwd Packet | direct | forward packet-length augmentation | p | p | - | Total Length of Fwd Packet | `tl_fwd` |
| Fwd Packet Length Min | direct | forward packet-length augmentation | p | p | - | Fwd Packet Length Min | `fmin` |
| Fwd Packet Length Max | direct | forward packet-length augmentation | p | p | - | Fwd Packet Length Max | `fmax` |
| Fwd Packet Length Mean | derived | forward packet-length augmentation | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `fmean` |
| Fwd Segment Size Avg | derived | forward packet-length augmentation | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `fmean` |
| Fwd IAT Total | direct | affine allocation of total forward delay | delay | delay | - | Fwd IAT Total | `fit` |
| Fwd IAT Mean | derived | affine allocation of total forward delay | delay | delay | Fwd IAT Total | Fwd IAT Total | `fimean` |
| Fwd IAT Std | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Std | `fistd` |
| Fwd IAT Max | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Max | `fimax` |
| Fwd IAT Min | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Min | `fimin` |
| Flow Duration | derived | affine allocation of total forward delay | delay | delay | Fwd IAT Total | Fwd IAT Total | `dur` |
| Flow IAT Mean | derived | affine allocation of total forward delay | delay | delay | Flow Duration | Fwd IAT Total | `flow_iat_mean` |
| Flow IAT Max | derived | affine allocation of total forward delay | delay | delay | Flow Duration | Fwd IAT Total | `flow_iat_max` |
| Packet Length Max | derived | combined packet-length statistics | p | p | Fwd Packet Length Max | Fwd Packet Length Max | `cmax` |
| Packet Length Min | derived | combined packet-length statistics | p | p | Fwd Packet Length Min | Fwd Packet Length Min | `cmin` |
| Packet Length Mean | derived | combined packet-length statistics | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `pmean` |
| Average Packet Size | derived | combined packet-length statistics | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `pmean` |
| Packet Length Variance | derived | combined packet-length statistics | p | p | Fwd Packet Length Mean, Fwd Segment Size Avg | Total Length of Fwd Packet | `pvar` |
| Packet Length Std | derived | combined packet-length statistics | p | p | Packet Length Variance | Total Length of Fwd Packet | `pstd` |
| Fwd Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `Nf_raw / dur_s` |
| Bwd Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `Nb / dur_s` |
| Flow Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `N / dur_s` |
| Flow Bytes/s | derived | rates | p, delay | p, delay | Flow Duration, Total Length of Fwd Packet | Fwd IAT Total, Total Length of Fwd Packet | `(tl_fwd + tl_bwd) / dur_s` |

56 of 79 features are never written by φ and stay at the source value in both arms.

### Empirical check on real flows

82908 source flows admitting padding and timing; each primitive switched on alone.

* `p` changes 12 features; all inside the traced slice: True; traced but unchanged here: none.
* `delay` changes 12 features; all inside the traced slice: True; traced but unchanged here: none.
* `shape` changes 3 features; all inside the traced slice: True; traced but unchanged here: none.
* reduced vector changes only direct features: True; equals φ on direct features: True; φ − reduced differ only on derived features: True.

# cicids2018_distrinet

## φ dependency map (traced from code)

Source: `src/attack/realizability/cicids2017.py:CICIDS2017PrimitiveModel.generate` (sha256 of the function source `ebed83e1a77fd33d…`). Rule: direct = the written value's backward slice (local assignments of generate) reaches the primitive controls and source columns only; derived = the slice reads a value that generate writes to another feature.

### primitive → direct (base) quantities → recomputed dependent features

| primitive | direct / base features (kept in P2) | dependent features recomputed by φ (frozen at source in P2) |
|---|---|---|
| `p` | Total Length of Fwd Packet, Fwd Packet Length Min, Fwd Packet Length Max | Fwd Packet Length Mean, Fwd Segment Size Avg, Packet Length Max, Packet Length Min, Packet Length Mean, Average Packet Size, Packet Length Variance, Packet Length Std, Flow Bytes/s |
| `delay` | Fwd IAT Total, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min | Fwd IAT Mean, Flow Duration, Flow IAT Mean, Flow IAT Max, Fwd Packets/s, Bwd Packets/s, Flow Packets/s, Flow Bytes/s |
| `shape` | Fwd IAT Std, Fwd IAT Max, Fwd IAT Min | - |

### Write sites

| feature | kind | φ code block | primitives | gated by | parents (written values read) | direct bases | written value |
|---|---|---|---|---|---|---|---|
| Total Length of Fwd Packet | direct | forward packet-length augmentation | p | p | - | Total Length of Fwd Packet | `tl_fwd` |
| Fwd Packet Length Min | direct | forward packet-length augmentation | p | p | - | Fwd Packet Length Min | `fmin` |
| Fwd Packet Length Max | direct | forward packet-length augmentation | p | p | - | Fwd Packet Length Max | `fmax` |
| Fwd Packet Length Mean | derived | forward packet-length augmentation | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `fmean` |
| Fwd Segment Size Avg | derived | forward packet-length augmentation | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `fmean` |
| Fwd IAT Total | direct | affine allocation of total forward delay | delay | delay | - | Fwd IAT Total | `fit` |
| Fwd IAT Mean | derived | affine allocation of total forward delay | delay | delay | Fwd IAT Total | Fwd IAT Total | `fimean` |
| Fwd IAT Std | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Std | `fistd` |
| Fwd IAT Max | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Max | `fimax` |
| Fwd IAT Min | direct | affine allocation of total forward delay | delay, shape | delay | - | Fwd IAT Min | `fimin` |
| Flow Duration | derived | affine allocation of total forward delay | delay | delay | Fwd IAT Total | Fwd IAT Total | `dur` |
| Flow IAT Mean | derived | affine allocation of total forward delay | delay | delay | Flow Duration | Fwd IAT Total | `flow_iat_mean` |
| Flow IAT Max | derived | affine allocation of total forward delay | delay | delay | Flow Duration | Fwd IAT Total | `flow_iat_max` |
| Packet Length Max | derived | combined packet-length statistics | p | p | Fwd Packet Length Max | Fwd Packet Length Max | `cmax` |
| Packet Length Min | derived | combined packet-length statistics | p | p | Fwd Packet Length Min | Fwd Packet Length Min | `cmin` |
| Packet Length Mean | derived | combined packet-length statistics | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `pmean` |
| Average Packet Size | derived | combined packet-length statistics | p | p | Total Length of Fwd Packet | Total Length of Fwd Packet | `pmean` |
| Packet Length Variance | derived | combined packet-length statistics | p | p | Fwd Packet Length Mean, Fwd Segment Size Avg | Total Length of Fwd Packet | `pvar` |
| Packet Length Std | derived | combined packet-length statistics | p | p | Packet Length Variance | Total Length of Fwd Packet | `pstd` |
| Fwd Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `Nf_raw / dur_s` |
| Bwd Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `Nb / dur_s` |
| Flow Packets/s | derived | rates | delay | delay | Flow Duration | Fwd IAT Total | `N / dur_s` |
| Flow Bytes/s | derived | rates | p, delay | p, delay | Flow Duration, Total Length of Fwd Packet | Fwd IAT Total, Total Length of Fwd Packet | `(tl_fwd + tl_bwd) / dur_s` |

56 of 79 features are never written by φ and stay at the source value in both arms.

### Empirical check on real flows

5161 source flows admitting padding and timing; each primitive switched on alone.

* `p` changes 12 features; all inside the traced slice: True; traced but unchanged here: none.
* `delay` changes 12 features; all inside the traced slice: True; traced but unchanged here: none.
* `shape` changes 3 features; all inside the traced slice: True; traced but unchanged here: none.
* reduced vector changes only direct features: True; equals φ on direct features: True; φ − reduced differ only on derived features: True.

