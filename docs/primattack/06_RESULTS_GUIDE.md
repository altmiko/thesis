# PrimAttack results guide

## Primary metrics

All ASRs use eligible clean-correct malicious source samples as the denominator.

$$\text{Raw Targeted ASR}=\frac{\sum E}{N_{eligible}}$$

$$\text{Valid Targeted ASR}=\frac{\sum(E\land V)}{N_{eligible}}$$

$$\text{Primitive-Feasible Targeted ASR}=\frac{\sum(E\land V\land P)}{N_{eligible}}$$

$$\text{SP-ASR}=\frac{\sum(E\land V\land P\land[S=PASS])}{N_{eligible}}$$

where $E$ is targeted Benign success, $V$ is validator_v2 domain validity, $P$ is primitive
feasibility, and $S$ is the flow-level semantic proxy status.

**SP-ASR means flow-level semantic-preservation proxy ASR.** It does not measure complete
malicious functionality.

## Coverage and status

Always report:

- semantic PASS count/rate;
- semantic FAIL count/rate;
- NOT_FULLY_TESTABLE count/rate;
- semantic testability rate.

Recon/PortScan and BruteForce contain critical application/sequence properties unavailable from
a single flow row, so their conservative NOT_FULLY_TESTABLE outcomes must not be dropped from
results.

## Cost/behavior summaries

Report medians because the distributions are skewed:

- relative duration change;
- relative byte change;
- rate retention;
- number of changed features;
- normalized primitive cost.

For DoS/DDoS, rate retention and the absolute train-P05 rate check are both relevant. A high
retention ratio on an already atypically low-rate flow does not automatically pass the
class-conditional threshold.

## Per-sample audit artifact

Each NPZ contains:

- `sample_id`, attack class, victim, optimizer, budget, primitive mode;
- original/adversarial predictions, target, targeted success;
- domain validity and violation reason;
- primitive feasibility and violation reasons;
- semantic status, failure reasons, unavailable-test reasons, and test counts;
- original/adversarial duration, delta, and relative change;
- original/adversarial represented bytes, added bytes, and relative change;
- original/adversarial rate and retention;
- requested/projected $p$ and $\alpha$, per-flow caps, and normalized magnitudes;
- changed feature names and count;
- source, code, preprocessing, scaler, victim, VAE/IDR, and calibration provenance.

Unavailable quantities are not fabricated.

## Sensitivity figures

`scripts/budget_sweep_primitive.py` writes:

1. `01_raw_asr_vs_budget.png`;
2. `02_valid_asr_vs_budget.png`;
3. `03_sp_asr_vs_budget.png`;
4. `04_semantic_pass_vs_budget.png`;
5. `05_rate_retention_vs_timing_budget.png`;
6. `06_primitive_cost_vs_asr.png`;
7. `07_timing_padding_combined.png`.

Read the plots together. Raw ASR without validity/feasibility/semantic gates is not the final
claim. Timing-only versus padding-only identifies which primitive drives evasion; semantic PASS
and rate retention identify which primitive degrades the flow-level proxy.

The completed full tables, class/victim breakdowns, paired tests, and figure links are in
[`docs/primattack_budget_results.md`](../primattack_budget_results.md).


## Interpretation boundary

Permitted claim: measurable attack-related flow properties available in CICIDS2017 were
preserved according to frozen train-derived proxy tests.

No result establishes complete malicious behavior, deployment behavior, or packet-trace
validity. Full verification would require packet realization, CICFlowMeter re-extraction, and
isolated replay against the relevant target/service; that work is outside this thesis.
