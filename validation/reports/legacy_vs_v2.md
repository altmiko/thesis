# Legacy vs validator_v2

Legacy = mined ConstraintEngine semantics (`old_constraints/cicids2017_distrinet/mined.json`: 8 monotone + 6 product identities). v2 shown at `hard_structural_valid` (SCHEMA+EXTRACTOR+PROTOCOL) and `hybrid_valid` (+MINED).

## Clean data (100,000 untouched `test` flows)

### Legacy vs v2 hard_structural_valid

| | v2 = True | v2 = False |
|---|---|---|
| **legacy = True** | 100000 | 0 |
| **legacy = False** | 0 | 0 |

Agreement: 1.000000

### Legacy vs v2 hybrid_valid

| | v2 = True | v2 = False |
|---|---|---|
| **legacy = True** | 100000 | 0 |
| **legacy = False** | 0 | 0 |

Agreement: 1.000000

## Corrupted data (36,000 rows: clean flows with one injected violation each)

This is where the two validators diverge — legacy only checks 14 mined monotone/product rules, whereas v2 also enforces SCHEMA type facts, EXTRACTOR identities and PROTOCOL non-negativity.

### Legacy vs v2 hybrid_valid (corrupted)

| | v2 = True | v2 = False |
|---|---|---|
| **legacy = True** | 22 | 9246 |
| **legacy = False** | 0 | 26732 |

Agreement: 0.743167

## Disagreements (corrupted set)

- legacy accepts but v2-hybrid rejects: **9246**
- v2-hybrid accepts but legacy rejects: **0**

v2 rules responsible (legacy=True, v2=False):

- `EXTRACTOR: Flow Packets/s ~= Fwd Packets/s + Bwd Packets/s` — 4000
- `EXTRACTOR: Packet Length Variance ~= Packet Length Std^2` — 3978
- `SCHEMA: Total Fwd Packet integer` — 730
- `EXTRACTOR: Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean` — 730
- `EXTRACTOR: Average Packet Size ~= Packet Length Mean` — 538


> This comparison is for auditability. v2 is intentionally NOT tuned to reproduce legacy; where they differ, the responsible rules are listed above so the difference is explainable.