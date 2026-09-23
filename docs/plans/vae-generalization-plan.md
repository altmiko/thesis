# VAE Generalization Plan — dataset/constraint plug-and-play

Goal: refactor the existing `MixedInputBetaVAE` in place so it is no longer
coupled to the CICIoT2023 39-feature schema. A new tabular dataset + constraints
must plug in through configuration while preserving the current β-VAE design,
training behavior, and generic decoder bodies. Decoder constraints become
manifest-driven (mined IR, not literal indices); this does not introduce a
second VAE architecture.

## What is hardwired to CICIoT2023 (six couplings)

| Coupling | Location | Hardcoded content |
|---|---|---|
| Feature partition | `vae/schema.py:get_partition` | fixed `continuous_idx`/`independent_binary_idx`/`derived_binary_idx`/`protocol_idx` for D=39 |
| Categorical = "Protocol" only | `vae/schema.py` L19-33, `model.py` L29,107,124 | one categorical col, `PROTOCOL_ALLOWLIST=[0,1,2,6,17,47]`, single embedding + single softmax head |
| Derived one-hots | `schema.derive_binaries_from_protocol_index`, `model.decode_to_39` L432 | protocol argmax -> TCP/UDP/ICMP/IGMP columns (a functional dependency) |
| Decoder by-construction constraints | `model._structure_continuous_raw` L248-323 | literal indices 2 (TTL), 31/32/33 (Min/AVG/Max ordering), 34 (Std), 37 (Number int), 38 (Var=Std^2), 30/35 (Tot sum/size); TTL `sigmoid*255`, softplus chains |
| Constraint penalties | `losses._compute_constraint_loss` / `_compute_physics_constraint_loss` | same indices restated |
| Output width / scatter | `decode_to_39`, `raw_postprocess`, `validator.G1-G8` | hard `39`, index lists, G-rules |

Plus `config.CLASSES` = the 8 CICIoT categories; `decode_to_39` name/signature.

Key insight: `CLAUDE.md` THE REFACTOR PRINCIPLE + Target architecture already IS
the generalization mechanism — `FeatureManifest` as single source of truth, fed
by `PerturbabilityScorer` + `ConstraintMiner`. None of those modules exist yet
(confirmed: no `*manifest*`/`*scorer*`/`*miner*` files). Generalizing the VAE =
building that manifest layer and making the VAE consume ONLY it. This plan adds
the piece the target arch under-specifies: a generic decoder-constraint engine.

## Research grounding (generic tabular VAEs)

- Type-specific likelihoods / heterogeneous heads — HI-VAE, VAEM (NeurIPS 2020),
  CardiCat (arxiv 2501.17324): one likelihood per feature TYPE, not per-dataset.
  Generalizes the 4 fixed heads.
- Categorical handling — learnable per-categorical embeddings on encode;
  Gumbel-softmax detokenizer for differentiable one-hot on decode (TABCF, arxiv
  2410.10463). Generalizes single protocol embedding + argmax.
- Mode-specific normalization per continuous column (CTGAN/TVAE) — replaces the
  single global RobustScaler.
- On-manifold adversarial attacks for tabular data (arxiv 2507.10998) — validates
  the latent-attack + constraint-projection design as dataset-agnostic.
- Plug-and-play decoder as isolated swappable stage (Flash-VAED 2602.19161;
  NVIDIA PiD) — precedent for decoder driven by external spec.

No prior work ships a by-construction constraint decoder driven by mined rules —
that stays the novel contribution; we make it read a constraint IR, not indices.

## Principles

- Everything mined on TRAIN ONLY; flows through one artifact (`FeatureManifest`).
- VAE/attacks import ONLY the manifest.
- One pipeline stage per commit.
- Keep CICIoT hardcoded constants in-tree until the mined CICIoT manifest is
  proven equivalent to them (equivalence = acceptance gate).

## Phases

### Phase 0 — `DatasetSpec` ingestion (typing block)
New `preprocessing/dataset_spec.py`. Input: raw dataframe + minimal user config
(`label_column`, optional per-column type overrides). Output on TRAIN split only:
- per-feature type: `continuous | binary | categorical | integer` (inferred via
  cardinality/dtype/value-set, override hook);
- per-categorical: cardinality + value allowlist (generalizes `PROTOCOL_ALLOWLIST`);
- per-column fitted transform (continuous->Robust/mode-specific; categorical->
  integer codes; binary->passthrough) — replaces the one global RobustScaler;
- optional label->category map (generalizes 34->8; identity if none).
Generalizes `preprocessing/schema.py`. CICIoT config must reproduce current
`FEATURE_NAMES`/`BINARY`/`INTEGER`/`CATEGORY_MAP` exactly (test).

### Phase 1 — `PerturbabilityScorer` + `ConstraintMiner` -> `FeatureManifest`
Per target architecture. Load-bearing new piece: the constraint IR the miner
emits — a closed set of typed primitives, each with a differentiable projection
(by-construction) and penalty (loss):

```
nonneg(f)                     # relu / softplus         (G1)
bound(f, lo, hi)              # sigmoid-scaled          (G7 TTL 0..255)
order([f1..fk])               # cumulative softplus     (G5 Min<=AVG<=Max)
equality(target, expr)        # set target := expr      (G6 Var=Std^2; Tot size=AVG; Tot sum=Number*AVG)
integer(f)                    # round + straight-through (G8 Number)
onehot(group)                 # softmax / Gumbel        (protocol)
depends(derived_group = table[categorical])  # functional dependency (proto->TCP/UDP/ICMP/IGMP)
```

Miner discovers each from train data, keeps a rule iff violation-rate <= epsilon,
reports soundness/completeness/F1. Acceptance gate: on CICIoT, mined IR must
recover G1-G8 + protocol allowlist + TTL/ordering/variance/integer rules. `expr`
for `equality`/`depends` is a restricted arithmetic form (`a`, `a*b`, `a^2`,
table-lookup) — enough for observed rules, closed enough to stay differentiable
and validatable.

### Phase 2 — parameterize the existing `MixedInputBetaVAE`
Keep `vae/model.py:MixedInputBetaVAE`, its encoder/decoder MLP bodies,
reparameterization, β-VAE objective, per-class training flow, and scheduler.
Refactor only the schema-dependent edges:
- Constructor takes `manifest` instead of `partition`; nothing indexes features
  by CICIoT-specific literal integers.
- Existing encoder path remains, but instantiates one embedding for each
  manifest categorical instead of special-casing Protocol.
- Existing decoder body remains. Its current continuous, binary, and categorical
  output heads are sized from the manifest; multiple categorical heads replace
  the single Protocol-only head.
- A generic constraint projector replaces `_structure_continuous_raw`: it
  interprets the IR primitives in topological dependency order. This is the
  existing decoder's constraint stage made configurable, not a new decoder.
- Rename `decode_to_39` to dataset-neutral `decode`; scatter outputs into
  `manifest.n_features`, including derived fields from dependency tables. Migrate
  all callers, then remove the old fixed-width method.
- Preserve current module/state-dict names where their semantics are unchanged,
  so existing CICIoT checkpoints remain loadable under the equivalent CICIoT
  manifest. CICIDS2017 still needs its own trained weights because its
  input/output dimensions differ, but it uses the same VAE class and code path.

### Phase 3 — generic losses (`vae/losses.py`)
- Reconstruction = sum of per-type likelihoods from manifest (Gaussian/Laplace
  continuous, BCE binary, CE per categorical).
- `_compute_constraint_loss` -> generic penalty walker over the same IR. Deletes
  `_compute_constraint_loss`/`_compute_physics_constraint_loss` index literals.

### Phase 4 — `MinedValidator` + generic postprocess
`raw_postprocess` and `validator.validate_batch` become IR-driven: iterate
manifest constraints, apply projection (postprocess) / check (validator). Deletes
G1-G8 literals and `VALID_PROTOCOLS`. Equivalence test vs current validator.

### Phase 5 — rewire attacks + config
- `attack/latent_infra.PerturbationMask`, `attack/validator`,
  `attack/constrained_input_baselines`, `latent_restarts` import manifest for
  tiers/rules/epsilon. Protocol/mask reimposition in PGD/CW inner loop reads
  manifest categoricals/onehot generically.
- `config.CLASSES` derived from the label encoder; per-class loop iterates
  `manifest.classes`.

## Verification (run, not inspect)

- P0: CICIoT config reproduces `schema.py` constants exactly.
- P1: mined CICIoT IR superset of G1-G8 + protocol rules; report F1.
- P2: `torch.allclose` — new engine on CICIoT IR vs current
  `_structure_continuous_raw` on real decoder outputs.
- P3/P4: ELBO + validator numerically match current on a CICIoT batch.
- P5: run canonical `attack/run_all_models_attack_rerun.py`; ASR/validity
  unchanged on CICIoT.
- New-dataset smoke test: ingest a second tabular IDS set (UNSW-NB15/NSL-KDD)
  end-to-end (spec -> manifest -> train 1 class -> decode -> validate) with zero
  code edits, config only — the plug-and-play acceptance criterion.

## Biggest risk / decision

The `depends` functional-dependency primitive (protocol->derived one-hots) is the
most CICIoT-shaped rule. Options: (a) generic `table[categorical]->onehot group`
primitive the miner can discover on any dataset (RECOMMENDED — fully general,
mineable); (b) leave it a dataset-specific plugin. Recommend (a): marginally more
work, avoids a per-dataset code hook, satisfies "nothing hand-authored."
