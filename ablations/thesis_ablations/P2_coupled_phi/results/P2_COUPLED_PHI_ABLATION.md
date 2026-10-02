# P2 - Coupled feature-recomputation ablation (`NoCoupledPhi`)

## Question

PrimAttack changes a flow through three primitives `u = (p, D, s)`: padding bytes per forward
packet, total added forward delay, and the delay allocation `shape`. The canonical map
`φ(x, u)` (`CICIDS2017PrimitiveModel.generate`) writes the statistics each primitive sets
directly and then recomputes every statistic that depends on them. P2 asks whether that coupled
recomputation is needed to reach valid adversarial feature states, or whether moving only the
directly affected statistics gives the same result.

## φ mapping, traced from code

`ablations/common/phi_mapping.py` parses `generate` and follows the data flow of each of its 23
write sites. A write is *direct* when its value depends only on the primitives and on source
columns. It is *derived* when φ computes it from a value it has already written to another
feature. No feature name enters the rule. Full table, per-write parents and the empirical
check: `phi_mapping.md` / `phi_mapping.json`.

| primitive | direct features (P2 keeps φ's values) | derived features (P2 holds them at the source value) |
|---|---|---|
| `p` | Total Length of Fwd Packet, Fwd Packet Length Min, Fwd Packet Length Max | Fwd Packet Length Mean, Fwd Segment Size Avg, Packet Length Max/Min/Mean, Average Packet Size, Packet Length Variance/Std, Flow Bytes/s |
| `D` | Fwd IAT Total, Fwd IAT Std/Max/Min | Fwd IAT Mean, Flow Duration, Flow IAT Mean, Flow IAT Max, Fwd/Bwd/Flow Packets/s, Flow Bytes/s |
| `s` | Fwd IAT Std/Max/Min | none |

Flow Duration is derived because φ writes `max(duration + D, Fwd IAT Total', Bwd IAT Total)`,
which reads the recomputed Fwd IAT Total. The 56 features φ never writes stay at the source
value in both arms. On 82,908 (2017) and 5,161 (2018) train flows that admit both primitives,
each primitive alone changes only features inside its traced slice, and the reduced vector
equals φ on the direct features and differs from it only on derived features.

The reduced map (`DirectOnlyPrimitiveModel`, `recompute_mode="direct_only"`) runs φ and resets
the 16 derived coordinates to the source flow. Projection, integer rounding, quantization,
capability mask M(x) and per-flow box are the canonical ones.

## Protocol

| | `full_phi` (reference) | `direct_only` (P2) |
|---|---|---|
| map seen by the search (candidates and gradients) | φ | direct-only map |
| search success predicate | Benign on the realized flow ∧ validator_v2 | Benign on the reduced flow |
| scored outcome | the realized φ flow | the same primitives realized again through φ |

Everything else is shared and taken from the FINAL suite: CICIDS2017 and CICIDS2018 DistriNet;
victims `mlp`, `cnn`, `ft_transformer` (2017) and `*-s42` (2018); the frozen 800 clean-correct
test flows per class (DoS, DDoS, Recon, BruteForce) from `baselines_untargeted/selection.json`,
sha256-verified; targeted → Benign; Hybrid Search with `PRIM_ARGS` (40 steps, lr 0.1, restarts
until 256 victim evaluations per flow); joint mode; capability-aware M(x); train-fit p75 and
envelope-only unbounded budgets; attack seeds 42/2024/2026. That gives 12 dataset × victim ×
budget cells with 3,200 flows per seed, the cell grid of A2 and D6.

Decisions:

* The shared `ablations/reference` arm is untargeted, so the targeted reference was rerun inside
  P2. It reproduces the FINAL targeted Hybrid cells flow for flow (sanity checks below).
* validator_v2 is left out of the P2 search predicate. It would judge the reduced flow, which
  breaks φ's identities by construction, and the search would then almost never record a
  success. Amendment A6 reran every FINAL Hybrid configuration (targeted, p75 and unbounded,
  both datasets, all seeds) without the gate and found every final flow bit-identical, so this
  setting does not separate the two arms.
* The full-φ re-evaluation of P2's returned primitives is an outcome measurement and is not
  charged to the 256-evaluation budget. Mean evaluations per flow are 188.5–190.8 in both arms,
  maximum 255.
* Tests follow the ablation-suite convention: paired McNemar per cell at attack seed 42 (exact
  binomial below 25 discordant flows), Newcombe 95% CI, Holm over the 12 cells. Seeds 2024 and
  2026 get their own Holm families and are reported in `report.md`.
* Results live in `ablations/thesis_ablations/P2_coupled_phi/` (requested rename from A4, which already names the
  mimicry ablation). Like every ablation here, P2 is an added analysis outside the locked FINAL
  protocol.

## Results

Mean ± SD over the three attack seeds, in %. *Reduced-space Raw* is what the P2 search believed
(Benign on the reduced flow). *Full-φ* is the same primitives through canonical φ. Only full-φ
Valid is compared with the reference. `*` marks a significant seed-42 McNemar after Holm.

| dataset | victim | budget | Ref Raw | Ref Valid | P2 reduced-space Raw | P2 full-φ Raw | P2 full-φ Valid | Δ Valid vs ref (pp) | seed-42 P2-only / ref-only |
|---|---|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.16 ± 0.00 | 3.08 ± 0.09 | 3.08 ± 0.09 | −1.01* | 0 / 34 |
| 2017 | mlp | unb | 22.94 ± 0.00 | 22.94 ± 0.00 | 28.91 ± 0.00 | 14.90 ± 0.02 | 14.90 ± 0.02 | −8.04* | 0 / 257 |
| 2017 | cnn | p75 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.41 ± 0.00 | 12.07 ± 0.15 | 12.07 ± 0.15 | −1.18* | 0 / 34 |
| 2017 | cnn | unb | 59.69 ± 0.00 | 59.69 ± 0.00 | 60.81 ± 0.00 | 53.75 ± 0.29 | 53.75 ± 0.29 | −5.94* | 0 / 188 |
| 2017 | ft_transformer | p75 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.00 | 0 / 0 |
| 2017 | ft_transformer | unb | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.41 ± 0.00 | 0.45 ± 0.02 | 0.45 ± 0.02 | −0.10 | 0 / 4 |
| 2018 | mlp-s42 | p75 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.69 ± 0.00 | 0.69 ± 0.00 | 0.69 ± 0.00 | −0.09 | 0 / 3 |
| 2018 | mlp-s42 | unb | 24.80 ± 0.02 | 24.80 ± 0.02 | 1.61 ± 0.02 | 24.02 ± 0.04 | 24.02 ± 0.04 | −0.78* | 0 / 23 |
| 2018 | cnn-s42 | p75 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 | 0 / 0 |
| 2018 | cnn-s42 | unb | 26.09 ± 0.00 | 26.09 ± 0.00 | 20.03 ± 0.00 | 25.64 ± 0.02 | 25.64 ± 0.02 | −0.45* | 0 / 14 |
| 2018 | ft_transformer-s42 | p75 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 | 0 / 0 |
| 2018 | ft_transformer-s42 | unb | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.00 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.00 | 0 / 0 |

Apparent (reduced-space) successes after the same primitives go through φ, all seeds pooled:

| dataset | victim | budget | apparent | lost after full φ | full-φ hits the reduced search did not see | median norm. cost of valid successes, P2 / ref |
|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 399 | 121 (30.3%) | 18 | 0.59 / 0.63 |
| 2017 | mlp | unb | 2,775 | 1,348 (48.6%) | 3 | 0.27 / 0.30 |
| 2017 | cnn | p75 | 1,287 | 164 (12.7%) | 36 | 0.57 / 0.60 |
| 2017 | cnn | unb | 5,838 | 774 (13.3%) | 96 | 0.54 / 0.39 |
| 2017 | ft_transformer | p75 | 12 | 0 | 0 | 0.30 / 0.29 |
| 2017 | ft_transformer | unb | 39 | 14 (35.9%) | 18 | 0.35 / 0.05 |
| 2018 | mlp-s42 | p75 | 66 | 12 (18.2%) | 12 | 0.52 / 0.48 |
| 2018 | mlp-s42 | unb | 155 | 76 (49.0%) | 2,227 | 1.00 / 0.60 |
| 2018 | cnn-s42 | p75 | 0 | 0 | 0 | n/a |
| 2018 | cnn-s42 | unb | 1,923 | 44 (2.3%) | 582 | 0.80 / 0.16 |
| 2018 | ft_transformer-s42 | p75 | 0 | 0 | 0 | n/a |
| 2018 | ft_transformer-s42 | unb | 0 | 0 | 12 | 1.00 / 0.10 |

Every lost apparent success was lost through the victim's prediction: under full φ the flow is
no longer classified Benign. No loss came from validator_v2, which accepts every full-φ P2 flow
that the victim calls Benign.

Which recomputed features undo the lost successes (2,416 rows on 2017 and 132 on 2018 that were
lost by prediction and whose two verdicts re-score exactly). Each φ code block's derived
features are swapped between the reduced and the full-φ flow of the same primitives:

| dataset | derived group (φ code block) | full-φ values alone end the success | reverting them to source restores it |
|---|---|---|---|
| 2017 | affine allocation of total forward delay (Fwd IAT Mean, Flow Duration, Flow IAT Mean, Flow IAT Max) | 96.9% | 92.5% |
| 2017 | rates | 7.5% | 3.1% |
| 2017 | forward / combined packet-length statistics | 0% (never changed) | 0% |
| 2018 | affine allocation of total forward delay | 100.0% | 97.7% |
| 2018 | rates | 2.3% | 0.0% |
| 2018 | forward / combined packet-length statistics | 0% (never changed) | 0% |

Single features, "alone ends the success": 2017 Fwd IAT Mean 78.7%, Flow IAT Mean 60.7%, Flow
Duration 31.0%, Flow Bytes/s 7.7%; 2018 Flow Duration 93.2%, Fwd IAT Mean 68.9%, Flow IAT Mean
28.0%. Per-row primitives, both vectors, logits, changed features and validator layers are in
`diagnostics/` (CSV; vectors in the git-ignored `lost_successes_vectors.npz`).

Consistency of the reduced flows themselves (rows with a non-zero primitive; `report.md`):
99.7–100% violate at least one of φ's exact identities, almost always
`Fwd IAT Mean = Fwd IAT Total / (Nf − 1)`. validator_v2 still accepts 8.3–78.5% of them per
cell. Its EXTRACTOR layer holds 7 identities (`validation/rules/*/extractor_rules.yaml`), none
of them on forward IAT mean, IAT totals or flow duration. At seed 42 every one of the 14,067
rejected reduced flows fails the MINED layer (ordering rules such as Fwd IAT Min ≤ Fwd IAT
Mean ≤ Fwd IAT Max), and 6 of them also fail EXTRACTOR; the per-layer verdicts are stored as
`reduced_<layer>_valid` in the P2 npz. Of the reduced flows validator_v2 accepted, 99.6–100% per
cell violate an exact φ identity and 96.9–100% fail the Level-B timing-order check of
`attack/realizability/validator.py`.

## Answers

**Does ignoring dependent-feature recomputation inflate apparent Raw ASR?** On the 2017 MLP and
CNN it does. Reduced-space Raw exceeds the full-φ Raw of the same primitives by 1.1–14.0 pp, and
in the unbounded cells it also exceeds the reference's Raw (MLP 28.91% vs 22.94%, CNN 60.81% vs
59.69%). On the 2018 MLP and CNN unbounded cells the error goes the other way. The reduced
search sees 1.61% and 20.03% while its own primitives evade at 24.02% and 25.64% under φ, so
2,227 and 582 full-φ successes were invisible to it. FT-Transformer and the 2018 p75 cells move
by at most 0.12 pp. Holding the derived features fixed therefore misestimates achievable
evasion, and whether it over- or underestimates depends on the victim.

**How many apparent successes survive full φ?** 7,929 of 10,350 on CICIDS2017 (76.6%) and 2,012
of 2,144 on CICIDS2018 (93.8%), all seeds and cells pooled. Per cell the loss ranges from 0 to
49%. It is largest where the reduced search was most successful (2017 MLP unbounded: 1,348 of
2,775 lost).

**Does full coupled recomputation improve final Valid ASR?** In 6 of 12 cells yes, significantly
after Holm: 2017 MLP −1.01 pp (p75) and −8.04 pp (unbounded), 2017 CNN −1.18 pp and −5.94 pp,
2018 MLP unbounded −0.78 pp, 2018 CNN unbounded −0.45 pp for P2. In the other six cells
(FT-Transformer on both datasets, 2018 MLP p75, the two zero cells) the difference is at most 4
discordant flows and not significant. P2 never has a valid success the reference lacks: 0
P2-only flows in all 36 seed × cell comparisons. The effect is concentrated on the 2017 MLP and
CNN. On CICIDS2018 it is under 1 pp.

**Which dependent feature groups account for the discrepancy?** The derived timing features of
φ's delay block (Fwd IAT Mean, Flow Duration, Flow IAT Mean, Flow IAT Max). Substituting their
full-φ values alone removes 96.9% (2017) and 100% (2018) of the lost successes, and reverting
them restores 92.5% and 97.7%. The rates matter in a few percent of rows. The padding-derived
groups play no part, because P2 used padding on only 114 of 115,200 flows and none of its valid
successes pad (the same holds for the reference). These substitutions describe these victims'
decisions on these flows, one group at a time. They do not separate interactions between
groups and are not a causal decomposition.

## Interpretation

The tested coupling matters on the 2017 MLP and CNN, where it costs P2 1–8 pp of Valid ASR and
up to half of its apparent successes, and on the 2018 MLP and CNN unbounded cells, where the
reduced search misses most of the evasion its own primitives achieve. On FT-Transformer and on
the 2018 p75 cells it has no measurable effect, which matches their near-zero Valid ASR in both
arms. A search that changes primitive-associated features independently misjudges which
primitive actions evade, in either direction. Keeping the timing statistics consistent with the
added delay (mean forward IAT, flow duration, flow IAT) is what aligns the search with the flow
the victim will actually see.

validator_v2 does not enforce the IAT/duration identities that the reduced flows break. Its
verdict on a candidate generated without coupled recomputation would therefore overstate that
candidate's consistency. This is the second validator_v2 gap after D6 (delay on
single-forward-packet flows) and, like that one, it does not affect capability-aware PrimAttack,
whose realized flows pass the Level-B identity and timing checks on all 115,200 rows of both
arms.

All statements are feature-space proxies on CICFlowMeter aggregates (`CLAUDE.md` claim
boundary); no PCAP was edited or replayed.

## Sanity checks (`sanity_checks.json`, `report.md`)

* `full_phi` reproduces the FINAL targeted Hybrid cells (p75 vs
  `primattack_hybrid_objective_targeted`, unbounded vs `primattack_targeted_budgets`): 115,200 of
  115,200 realized flows bit-identical, with identical predictions, validator verdicts and
  valid-success outcomes.
* Pairing: all 36 cells have identical sample ids, positional indices and per-flow budget boxes
  (`p_hi`, `delay_hi`) in both arms.
* Primitive identity: on all 115,200 P2 rows the stored `(p, D, s)` is a fixed point of the
  canonical projection; φ of the stored primitives equals the stored full-φ flow, the direct-only
  map equals the stored reduced flow, and the two flows differ only on derived features. These
  are also asserted during the run (`phi_mapping.realize_full_phi`).
* Capability inference active: 0 padding or timing capability violations in either arm; the
  stored masks equal `infer_capabilities` on all rows.
* Final validator unchanged: validator_v2 recomputed on every stored flow of both arms
  reproduces the stored verdict on all 115,200 rows each.
* Final counting: 0 P2 valid successes without a full-φ Benign prediction, 0 without full-φ
  validator_v2 acceptance, and stored P2 raw success equals Benign on the full-φ logits on every
  row.
* Lost-success verdicts re-scored from the stored vectors: 2,553 of 2,553 full-φ verdicts and
  2,548 of 2,553 reduced hits reproduce. The 5 reduced hits that flip under a different batch
  composition are excluded from the group/feature substitutions.

## Files

| file | content |
|---|---|
| `phi_mapping.md`, `phi_mapping.json` | traced φ map, reduced-path definition, empirical check (written before the run) |
| `per_flow.parquet` | one row per (dataset, victim, budget, seed, flow): both arms' outcomes, predictions, primitives, costs |
| `per_seed.csv` | every metric per seed and cell |
| `aggregate.csv` | reference vs P2 mean ± SD |
| `reduced_vs_full.csv` | reduced-space vs full-φ outcome table |
| `tests.csv` | McNemar + Newcombe + Holm (P2 vs reference; reduced hit vs full-φ Valid) |
| `diagnostics/` | lost successes (`lost_successes.csv`, vectors npz), group/feature substitution rankings |
| `reduced_flow_audit_per_seed.csv`, `reduced_flow_identity_violations.csv` | validator_v2 and Level-B checks on the reduced flows |
| `sanity_checks.json` | all checks above |
| `report.md` | generated tables, including per-seed McNemar |
| `<dataset>/cells.json`, `<dataset>/artifacts/*.npz` | per-cell summaries and per-row artifacts (npz git-ignored) |
