# 6. Semantic-Preservation Framework (MODERATE/HIGH DETAIL)

Implemented in `src/attack/flow_semantics.py` (`FlowSemanticValidator`), consumed by
`run_cicids2017_primitive_attack.py` and the sweep. This is a **flow-level
semantic-preservation proxy** — it tests measurable attack-related flow properties that
CICIDS2017 aggregate rows actually expose. It is **not** packet-level functionality.

---

## 6.1 Three distinct gates (do not conflate)

1. **Domain validity** — validator_v2 `hybrid_valid` (doc 3). "Does this row satisfy
   the structural schema/extractor/protocol/mined rules?" *Not in this file; the
   separate VAE IDR gate measures distributional realism.*
2. **Primitive feasibility** — projected controls + realized costs inside the hard
   per-flow box, plus internal transform consistency (doc 2 §2.10,
   `flow_semantics.py` `primitive_budget_compliance` + `RealizabilityValidator`).
3. **Flow-level semantic preservation** — `FlowSemanticValidator.evaluate` (this file):
   did the edit preserve the attack-related and structural flow properties?

A vector can be domain-valid yet fail a class attack-intensity rule, and vice-versa —
hence the deliberate separation.

---

## 6.2 Outcome labels (`evaluate :393-418`)

Per-check status `CheckStatus ∈ {PASS, FAIL, NOT_TESTABLE}` (`:23-26`). Aggregated to a
per-sample `SemanticStatus ∈ {PASS, FAIL, NOT_FULLY_TESTABLE}` (`:29-32`) by counting,
over **required** checks only, how many pass / fail / are not-testable, then:
```
status = PASS
if not_testable > 0:      status = NOT_FULLY_TESTABLE     (:417)
if required_failed > 0:   status = FAIL                    (:418)   # FAIL overrides
```
Precedence: **FAIL > NOT_FULLY_TESTABLE > PASS**.

- **PASS**: every required check passed and none was untestable.
- **FAIL**: at least one required check FAILED (a real semantic violation).
- **NOT_FULLY_TESTABLE**: no failures, but ≥1 required property cannot be tested from a
  single aggregate flow row. **This is NOT a semantic failure** — it is an
  *unavailability of evidence*. Confirmed in code: it is never promoted to PASS and never
  counted as FAIL; it has its own rate/count (`run :522-526`).

### Generic required checks (all classes, `evaluate :286-385`)
protocol unchanged; service ports unchanged; packet counts unchanged (incl. Fwd Act Data
Pkts, subflow packets); TCP control/flag aggregates unchanged; source/dst IP endpoints &
direction unchanged; all generated features finite; represented traffic volume not
decreased (`added_byte_quantity ≥ −atol`); **only declared primitive dependencies
changed** (`:348-368` — φ may write only features declared by active primitives);
**primitive budget compliance** (`:370-386` — p and delay integral, `0≤p≤⌊p_hi⌋`,
`0≤delay≤⌊delay_hi⌋`, `0≤shape≤shape_hi`, realized relative-duration ≤ budget,
added bytes ≥ 0). Endpoint/label checks become NOT_TESTABLE if metadata is
absent.

---

## 6.3 Per-class rules

- **DoS / DDoS — `RateRetentionRule`** (`:104-118`): required check
  `class_attack_like_flow_rate`: **adversarial `Flow Packets/s` ≥ class-train p05**
  (absolute floor; DoS 0.6248, DDoS 1.0654). Reason `RATE_BELOW_TRAIN_P05`. **A DoS/DDoS
  flow whose rate dilation drops it below the class's normal packet rate FAILS** — the
  attack would no longer look like a flood. (Note: the doc's ratio `R_adv/R₀` is only a
  reported cost metric, not the gate — see `00_OPEN_ISSUES.md#A4`.)
- **Recon — `ReconRule`** (`:121-137`): emits 3 required **NOT_TESTABLE** checks for
  `complete_scanned_port_set`, `scan_sequence`, `distinct_connection_attempt_count`
  (reason `NOT_TESTABLE_FROM_FLOW_DATA`). A single aggregate row lacks the port set, scan
  order, and distinct connection-attempt sequence → Recon successes are `NOT_FULLY_TESTABLE`.
- **BruteForce — `BruteForceRule`** (`:140-156`): 3 required NOT_TESTABLE checks for
  `authentication_attempt_count`, `credential_or_payload_semantics`,
  `server_authentication_outcome`. Aggregate flow features expose none of these → always
  `NOT_FULLY_TESTABLE` (unless a generic check fails first).

Cost quantities (`primitive_costs :159-204`): duration/byte/rate deltas, `rate_retention
= R_adv/max(R₀,ε)`, padding % of forward mean, normalized primitive magnitudes — reported
per sample, used for medians in results, and (rate_retention) for interpretation.

---

## 6.4 SP-ASR

$$\mathrm{SP\text{-}ASR}=\frac{\sum(E \wedge V \wedge P \wedge [S=\text{PASS}])}{N_{\text{eligible}}}$$
where E = targeted-Benign success, V = validator_v2 `hybrid_valid`, P = primitive
feasibility, and S = semantic status. The standalone runner and full paired driver
compute the same conjunction.

**Observed in the current v2 campaign** (reference seed 42, joint p75, 3,200
clean-correct rows per victim): SP-ASR is 6.19% for MLP, 10.97% for CNN, and 0.22%
for FT-Transformer. These rates are lower than valid targeted ASR because semantic
`PASS` is an additional gate; Recon/BruteForce successes are
`NOT_FULLY_TESTABLE`, not silently counted as passes. The historical replaced
`(p, alpha)` sweep had SP-ASR 0 and only 10/4064 classifier successes. Always report
PASS/FAIL/NOT_FULLY_TESTABLE coverage and never drop untestable rows from the
denominator.

---

## 6.5 Assumptions · Limitations · Claims
- **This is flow-level, not packet-level** (docstring `:1-6`). Packet realization,
  CICFlowMeter re-extraction, and isolated replay against the target/service are out of
  scope (`NullPacketBackend`).
- **NOT_FULLY_TESTABLE ≠ failure** — it is honest reporting of missing evidence; do not
  count it as either success or violation.
- **Can claim**: for the properties CICIDS2017 exposes (protocol/ports/counts/flags/
  endpoints/volume/declared-dependency discipline; DoS/DDoS rate floor), PrimAttack
  outputs were checked against frozen train-derived proxy tests.
- **Must NOT claim**: preserved *complete* malicious functionality, deployment behavior,
  or packet-trace validity for any class — and especially not for Recon/BruteForce, whose
  defining semantics are untestable from a flow row.
