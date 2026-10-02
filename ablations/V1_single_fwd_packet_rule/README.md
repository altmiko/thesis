# V1 - Toggle-able validator rule: no forward IAT on single-forward-packet flows

## Why

Ablation D6 (`../D6_capability_inference`) found a gap in validator_v2:

- PrimAttack can add forward delay to a flow with only one forward packet. Example:
  `Fwd IAT Total` goes from 0 to 1.2e7 µs while `Total Fwd Packet = 1`.
- That flow passes all four validator_v2 layers.
- Only the capability mask M(x) prevents it.

## Rule

`ablations/common/extra_rules.py`, `single_fwd_packet_no_fwd_iat` (definitional, CICFlowMeter):

    Total Fwd Packet ≤ 1  ⇒  Fwd IAT Total = Fwd IAT Mean = Fwd IAT Std = Fwd IAT Max = Fwd IAT Min = 0

validator_v2 is the locked FINAL validator and is **not modified**. The rule is evaluated next
to it:

- `--rule on`: the rule is ANDed into the search success predicate and into validity.
- `--rule off`: it is not.

Every arm stores the rule's per-row verdict, so every arm reports both outcomes:

- `extended_valid_success` = objective met ∧ validator_v2 ∧ rule (primary);
- `valid_success` = objective met ∧ validator_v2, as in all other ablations.

## Arms

| Arm | PrimAttack | Rule |
|---|---|---|
| `capaware_rule_off` | capability-aware (canonical) | off |
| `capaware_rule_on` | capability-aware | on |
| `nocap_rule_off` | capability-ablated, u = (p, D, s) (= D6) | off |
| `nocap_rule_on` | capability-ablated | on |

Protocol: Hybrid Search, untargeted, p75 and unbounded budgets. Both datasets, three victims,
four classes × 800 frozen clean-correct flows, attack seeds 42/2024/2026.

```
python ablations/V1_single_fwd_packet_rule/run.py --device cuda                      # all four arms
python ablations/V1_single_fwd_packet_rule/run.py --rule on --capability aware      # one arm
python ablations/V1_single_fwd_packet_rule/run.py --skip-run                         # re-analyze
```

Full tables: `results/report.md`. CSVs: `summary.csv`, `tests.csv`, `gap_audit.csv`,
`genuine_flow_check.csv`.

## Checks

- **The rule is sound on real traffic.**
  - It accepts every genuine flow of every split (train/val/test) of both datasets, in all
    classes including Benign: 2,913,931 flows, 0 violations.
  - Of these, 616,539 are single-forward-packet flows, and all of them carry zero forward IAT.
- **The rule-off arms reproduce the earlier runs exactly.**
  - `capaware_rule_off` = `reference`: 115,200 / 115,200 identical adversarial flows.
  - `nocap_rule_off` = D6 `no_capability`: 115,200 / 115,200 identical.

## Results

**Capability-aware PrimAttack (canonical): the rule never binds.**

- Valid ASR is identical with the rule on or off in all 12 (dataset, victim, budget) cells, with
  0 discordant flows at seed 42.
- 0 validator_v2 successes violate the rule.
- PrimAttack's timing primitive already requires ≥ 2 forward packets, so the canonical results
  in FINAL and in every other ablation stand unchanged under the stricter validator.

**Capability-ablated PrimAttack: the rule closes the D6 gap.**

Successes are pooled over seeds and classes. `rejected` counts validator_v2 successes that
violate the rule.

| dataset | victim | budget | validator_v2 successes (rule off) | rejected by the rule | validator_v2 + rule |
|---|---|---|---|---|---|
| 2018 | mlp-s42 | unb | 3,698 | 1,461 | 2,237 |
| 2018 | cnn-s42 | unb | 468 | 25 | 443 |
| 2018 | ft_transformer-s42 | unb | 33 | 24 | 9 |
| 2018 | mlp-s42 | p75 | 21 | 2 | 19 |

Mean Valid ASR over seeds, under validator_v2 alone vs with the rule:

| dataset | victim | budget | validator_v2 | validator_v2 + rule |
|---|---|---|---|---|
| 2018 | mlp-s42 | unb | 38.52% | 23.30% |
| 2018 | cnn-s42 | unb | 4.87% | 4.61% |
| 2018 | ft_transformer-s42 | unb | 0.34% | 0.09% |

On CICIDS2017 no ablated success violates the rule, because PROTOCOL already rejects the
ablated arm's violations there (D6).

**Putting the rule inside the search finds no replacement successes.**

- `nocap_rule_on` and `nocap_rule_off` have identical `validator_v2 + rule` Valid ASR in every
  cell (0 discordant flows at seed 42).
- For flows whose only validator_v2 success was a single-packet delay, the search with the rule
  in its predicate finds no other valid evasion.
- Its incumbent then falls back to the lowest-margin failure; 1,084 of those still pass
  validator_v2 but violate the rule (2018 MLP unbounded).

## Findings

- **The rule is a sound fix.** It has 0 false rejections on 2.9 M genuine flows and removes
  every single-forward-packet timing success from the capability-ablated attack. With it,
  ablated 2018 MLP unbounded Valid ASR falls from 38.52% to 23.30%. D6's conclusion is
  strengthened: without M(x), Valid ASR is below the capability-aware reference (44.36%) in
  every cell once validity is complete.
- **For the thesis numbers it is a no-op.** Capability-aware PrimAttack never produces such
  flows, so adopting the rule into validator_v2 would change no FINAL or ablation result for
  PrimAttack.
- Adding the rule would make validator_v2 safe to use as the only feasibility gate for attacks
  without a capability model, e.g. other feature-space attacks. That requires amending the
  locked validator and re-running Exp F (genuine-flow acceptance). Exp F's outcome is predicted
  by the 0-violation genuine-flow check above but was not re-run here.
