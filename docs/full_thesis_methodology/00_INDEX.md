# Full Thesis Methodology — Documentation Index

Audit-grounded technical documentation for the final LaTeX write-up. Every claim
here was checked against the actual implementation, configs, mined artifacts, and
experiment outputs on `main` (git `3380e4a`, working tree dirty). Where code and
prior documentation disagree, the discrepancy is stated explicitly and the **code
is treated as ground truth**.

## Scope note (important)

The repository contains **two datasets**. The root `CLAUDE.md` describes the
*older* **CICIoT2023** study (39 features, 8 categories, `MixedInputBetaVAE`
Path-C attacks). The *current, thesis-critical* work is on **CICIDS2017-DistriNet**
(79 CICFlowMeter features, 5 categories, PrimAttack + validator_v2 + feature-space
baselines). Most CICIoT2023 code is retired/CICIoT-only; this documentation set is
CICIDS2017-centric and flags CICIoT-only or stale code wherever it appears.

## Documents

| # | File | Detail | Subject |
|---|------|--------|---------|
| — | [`00_CODE_MAP.md`](00_CODE_MAP.md) | — | File→responsibility map; live vs stale vs archived |
| — | [`00_OPEN_ISSUES.md`](00_OPEN_ISSUES.md) | — | Discrepancies, defects, claim boundaries |
| 1 | [`01_preprocessing_cicids2017_distrinet.md`](01_preprocessing_cicids2017_distrinet.md) | **MAX** | Raw CSV → transformed arrays/artifacts |
| 2 | [`02_primattack.md`](02_primattack.md) | **MAX** | Primitive-domain white-box attack (p, α) |
| 3 | [`03_validator_v2.md`](03_validator_v2.md) | **MAX** | SCHEMA/PROTOCOL/EXTRACTOR/MINED validator |
| 4 | [`04_statistical_evaluation.md`](04_statistical_evaluation.md) | **MAX** | Cochran/McNemar/Friedman/Wilcoxon/Holm |
| 5 | [`05_victim_classifiers.md`](05_victim_classifiers.md) | MOD | MLP / CNN / FT-Transformer |
| 6 | [`06_semantic_preservation.md`](06_semantic_preservation.md) | MOD/HIGH | PASS / FAIL / NOT_FULLY_TESTABLE, SP-ASR |
| 7 | [`07_attack_metrics.md`](07_attack_metrics.md) | MOD | ASR family, denominators, pooling |
| 8 | [`08_feature_space_pgd_cw.md`](08_feature_space_pgd_cw.md) | MOD | PGD/C&W baselines; what really ran |
| 9 | [`09_vae_adversarial_method.md`](09_vae_adversarial_method.md) | MOD | β-VAE latent attack; experiment status |
| 10 | [`10_reproducibility_provenance.md`](10_reproducibility_provenance.md) | MOD | Seeds, hashes, manifests, hardware |

## How to read this set for the thesis

- **Methods chapter**: docs 1, 5, 9 (data + models + generator) and 3 (validity model).
- **Attack chapter**: doc 2 (PrimAttack, the core contribution) + doc 8 (baselines).
- **Evaluation chapter**: docs 7 (metrics), 6 (semantic preservation), 4 (statistics).
- **Reproducibility appendix**: doc 10.
- **Threats-to-validity / limitations**: `00_OPEN_ISSUES.md` is the single collected list.

## Global claim boundary (applies to every result)

All CICIDS2017 attack work operates **offline on aggregate CICFlowMeter flow
features**. No PCAP is edited, no packet is constructed, no attack is replayed, and
no application/target state is observed. "Validity", "primitive feasibility", and
"flow-level semantic preservation" are all **feature-space proxies**. Packet-level
realizability and complete malicious functionality are **not established** anywhere
in the code (`attack/realizability/base.py:NullPacketBackend` explicitly declares
Level-C packet verification unavailable).
