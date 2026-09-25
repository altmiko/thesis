# Open Issues, Discrepancies, and Claim Boundaries

Collected findings where the code disagrees with documentation, where artifacts are
missing, or where a naive reading would over-claim. Cite these in the
threats-to-validity / limitations section. Each item names the evidence.

## A. Discrepancies between code and prior docs

1. **`CLAUDE.md` describes the wrong (older) study.** It documents CICIoT2023
   (39 features, 8 classes, `MixedInputBetaVAE` Path-C, latent PGD/C&W). The
   thesis-critical current work is CICIDS2017-DistriNet (79 features, 5 classes,
   PrimAttack). `CLAUDE.md` also states "no FT-Transformer in code" — false: a
   local `FTTransformer` exists (`src/classifiers/ft_transformer.py`) with trained
   checkpoints. Treat `CLAUDE.md` as legacy context, not current spec.

2. **"Original / non-budget" is not a PrimAttack method name.** The standalone
   runner still requires `budget_name ∈ {restricted, intermediate,
   maximum-evaluated}`. The full paired driver now additionally evaluates
   `unbounded_calibration`: it sets the class `p_max` and relative-duration cap to
   infinity while retaining the same train-p99 envelope, capability gates, integer
   projection, and DoS/DDoS rate floor. This committed condition appears as `unb`
   in `outputs/full_adv_eval_primattack_v2`. Call it **envelope-only**, not
   unconstrained or original. `maximum-evaluated` (p75) remains the largest named
   empirical budget; input-space PGD/C&W are the genuinely non-primitive-constrained
   baselines.

3. **`validation/evaluation/attack_artifact_validity.py` is STALE.** It expects npz
   fields `evasion`/`benign` and output dirs `outputs/v2_validity_*` that the
   current PrimAttack runner never produces (it saves `targeted_success`,
   `domain_valid`, …). Its "MLP and CNN, 8 cells" prose and Conclusions are
   hard-coded narrative, not measured from live artifacts. Do not cite it.

4. **Semantic rate-retention doc vs code.** `docs/primattack/04_FLOW_SEMANTIC_PRESERVATION.md`
   foregrounds a *ratio* `R_adv/max(R₀,ε)`; the actual DoS/DDoS gate
   (`flow_semantics.py:107-118`, `RateRetentionRule`) is an **absolute** floor:
   `adversarial Flow Packets/s ≥ class-train p05`. The ratio is only a reported
   `PrimitiveCosts` metric and gates nothing.

5. **Historical versus current victim coverage.** The old 72-artifact
   `outputs/primattack_budget_sensitivity_full/` sweep contains MLP/CNN, seed 42
   only. The current paired v2 campaign in `outputs/full_adv_eval_primattack_v2/`
   contains MLP, CNN, and FT-Transformer for attack seeds `42,123,2024`. Do not use
   the historical sweep's coverage limitations to describe the v2 campaign.

6. **Argparse defaults ≠ shipped training.** Classifier trainer argparse defaults are
   `epochs=5, patience=2`, but the checkpoints were trained with `epochs=10,
   patience=3` (CLI override; best epochs mlp=9, cnn=8, ft=7). Report the values
   actually used, not the argparse defaults.

## B. Statistical-validity concerns (see doc 4)

Items 7–10 describe the **historical Adam/sigmoid budget sweep and its report**.
The v2 optimizer comparison instead keeps analyses within victim, uses a frozen
800-row/class roster, reports three attack seeds descriptively, and includes paired
rate differences with Newcombe CIs. Attack-seed variation still does not replace
victim-training seed replication.

7. **Victim pseudoreplication / non-independence (primary concern).** In
   `analyze_primattack_experiments.py`, the pairing key is
   `_KEY = [sample_id, attack_class, victim_model, seed]`, and `analyze()` pools
   **all victims** into one Cochran's Q / McNemar / Friedman / Wilcoxon per budget.
   Each unique source flow therefore appears **twice** (once for MLP, once for CNN)
   as if independent → N is inflated (pooled N = 4064 = 2 victims × ~2032 eligible
   flows). Recommended safer presentation: run the paired tests **per victim** and
   report them side-by-side, or add victim as a blocking factor; do not pool.

8. **Single seed.** All committed sweep artifacts are `seed=42` only (runner default
   is `42,43,44` but only 42 was run). No across-seed variance; no seed as a random
   effect. State results as single-seed; do not claim seed robustness.

9. **Missing effect sizes / CIs in the sweep report.** `analyze_primattack_experiments.py`
   emits only test statistics + p-values (+Holm). It does **not** emit effect sizes
   (e.g. Cochran/Cohen's g, rank-biserial for Wilcoxon) or confidence intervals.
   Note: `src/evaluation/paired_validity_gap.py` *does* compute a Newcombe paired CI
   and Haldane–Anscombe OR, but that engine is used for the raw-vs-valid gap, not for
   the mode/budget sweep. For the sweep, effect sizes/CIs are absent.

10. **Degenerate outcomes make most tests trivial.** Because raw ASR ≈ 0 across
    budgets (max 10/4064 = 0.25%), most binary McNemar tables have b=c=0 (p=1.0),
    and raw=valid=primitive-feasible counts are equal (validator/feasibility rejected
    none of the classifier successes). The only non-trivial binary result is at
    `maximum-evaluated`: padding/joint > timing-only (Cochran Q p=4.54e-5). Continuous
    tests (duration/byte/rate) are highly significant simply because timing changes
    duration and padding changes bytes by construction — significance here is
    mechanical, not evidence of an interesting effect.

## C. Substantive result boundaries

11. **Budgeted PrimAttack: structurally valid evasion is victim-dependent, and in-distribution
    evasion is ≈0.** The earlier conclusion ("essentially fails to evade", from the replaced
    Adam/sigmoid `(p, α)` optimizer — `docs/primattack_budget_results.md`,
    `outputs/full_adv_eval` `prim_opt_*`) was an optimizer artifact. With the v2 search
    (`outputs/full_adv_eval_primattack_v2`, `PRIMATTACK_V2_OPTIMIZER_COMPARISON.md`; 800 frozen
    clean-correct flows/class, reference seed 42, classes pooled within victim) joint p75
    targeted ∧ hybrid_valid ∧ feasible is mlp 6.28%, cnn 34.91%, ft_transformer 5.19%
    (old: 0.16%, 1.19%, 4.66%; McNemar Holm p ≤ 1.7e-4 for every victim). Joint p75
    targeted ∧ valid ∧ feasible ∧ SP is mlp 6.19%, cnn 10.97%, ft 0.22%.
    **True-IDSR (∧ in_distribution) remains 0–0.09% for every victim/budget/mode**: the
    successful flows are structurally valid and within the calibrated box but lie outside
    the per-class VAE IDR gate. Validity and primitive feasibility were observed at 100%
    in every v2 PrimAttack cell; this is an empirical campaign result, not a construction guarantee.
    Caveat: the frozen roster was inspected diagnostically before the v2 design, so the
    old-vs-new contrast is post-hoc on the same rows.

12. **Feature-space PGD/C&W: high raw ASR, zero valid ASR.**
    `outputs/cicids2017_baseline_pgd_cw/`: raw ASR ≈ 1.0 but domain-valid /
    realizable / valid-real-evasion = 0.0 and **no per-sample npz** (aggregated
    json+md only). These are **untargeted** (source-class CE ascent), not
    targeted→Benign. The targeted→Benign feature-PGD variant
    (`run_cicids2017_input_baseline.py`) was **never run** (no artifacts). This is the
    intended baseline story: unconstrained feature attacks trivially evade but are
    domain-invalid.

13. **VAE latent attack has no final aggregated CICIDS2017 result.** Only the four
    per-class generators (`outputs/cicids2017_vae_stage_a/`) and **two smoke npz**
    exist. All aggregated latent-attack results are archived and were built against
    the **retired** LSTM/serial victims. The current-roster VAE latent attack is
    IMPLEMENTED but its final run is MISSING.

14. **Global claim boundary (everywhere).** All validity, primitive-feasibility, and
    semantic-preservation results are **feature-space proxies** on aggregate
    CICFlowMeter statistics. `NullPacketBackend` declares packet-level (Level-C)
    verification unavailable. Never claim packet realizability, PCAP validity, or
    complete malicious functionality.

## D. Provenance mismatches (see doc 10)

15. Victims trained on Python 3.11.15 + CUDA (RTX 4070 Ti SUPER); attacks/calibration
    run on Python 3.12.3 + CPU. All run manifests record `dirty: true`. Reproducibility
    is anchored on SHA-256 of preprocessing manifest/scaler/checkpoints, not on git
    (outputs, checkpoints, `*.pkl`, `data/` are git-ignored). `requirements.txt`
    contains non-portable `file://` pins and xgboost not present in `environment.yml`.
