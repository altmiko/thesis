# Synthetic violation report — cicids2017_distrinet

Clean accepted base flows: **10,000** (validator_v2-accepted `test` flows). Each corruption injects ONE controlled violation.

**Overall known-invalid detection rate (mean over corruptions): 0.9998**

| corruption | target layer | applied | detection rate | top firing rule |
|---|---|---|---|---|
| min_gt_max (Fwd Packet Length Min set above Max) | MINED (order chain) | 8,773 | 1.0000 | `MINED:Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max` |
| mean_above_max (Packet Length Mean pushed above Max) | MINED (order chain) | 10,000 | 1.0000 | `EXTRACTOR:Average Packet Size ~= Packet Length Mean` |
| negative_count (Total Fwd Packet set negative) | PROTOCOL (nonnegativity) | 10,000 | 1.0000 | `PROTOCOL:Total Fwd Packet >= 0` |
| variance_neq_std_sq (Packet Length Variance != Std^2 (3x)) | EXTRACTOR (square) | 8,594 | 1.0000 | `EXTRACTOR:Packet Length Variance ~= Packet Length Std^2` |
| variance_subtle_2pct (Variance off by a subtle 2%) | EXTRACTOR (square) | 8,594 | 0.9980 | `EXTRACTOR:Packet Length Variance ~= Packet Length Std^2` |
| flow_pkts_inconsistent (Flow Packets/s inconsistent with Fwd+Bwd) | EXTRACTOR (sum) | 10,000 | 1.0000 | `EXTRACTOR:Flow Packets/s ~= Fwd Packets/s + Bwd Packets/s` |
| avg_size_neq_mean (Average Packet Size != Packet Length Mean) | EXTRACTOR (equality) | 10,000 | 1.0000 | `EXTRACTOR:Average Packet Size ~= Packet Length Mean` |
| totlen_neq_count_mean (Total Length Fwd != count*mean (1.5x)) | EXTRACTOR (product) | 10,000 | 1.0000 | `EXTRACTOR:Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean` |
| count_fractional (Integer packet count made fractional (+0.5)) | SCHEMA (integer) | 10,000 | 1.0000 | `SCHEMA:Total Fwd Packet integer` |
| mean_below_min_subtle (Mean set 0.01 below Min (subtle)) | MINED (order chain) | 5,100 | 1.0000 | `MINED:Packet Length Min <= Packet Length Mean <= Packet Length Max` |

Detection = fraction of corrupted rows for which `structurally_valid` becomes FALSE.