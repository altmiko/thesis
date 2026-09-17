# Audit — `feature_groups.py` & `netdiffuser_categorization.py`

*Scope: `src/preprocessing/feature_groups.py` and
`src/preprocessing/netdiffuser_categorization.py`. Explains exactly what each
module does, how the feature groups were derived (report-ready), how they are
consumed downstream, and every bug / correctness concern found. All invariant
claims below were verified programmatically (results in §3.5); data-distribution
facts are cited from `data/processed/ciciot2023_labeled_full_manifest.json` and
`docs/data/ciciot2023_building_from_download.md`.*

---

## 0. TL;DR

- **`feature_groups.py` is a pure declarative config module** — the single
  source of truth for the 39-feature schema, its type groups, its
  adversarial-robustness (perturbation) taxonomy, and the 34→8 category map. It
  is imported by **34 modules** across `attack/`, `evaluation/`, `vae/`,
  `thesis_eval/`, and `preprocessing/`. It contains almost no logic — just
  named lists/dicts plus one small derived array.
- **`netdiffuser_categorization.py` implements one algorithm** — NetDiffuser
  "Algorithm 1": it auto-partitions the features into *discrete* (independent)
  vs *relative* (correlated-group) using hierarchical clustering on a
  Spearman-correlation distance, choosing the cut height by Calinski–Harabasz
  score. Its output is a second input to the 3-tier perturbation mask.
- **Neither file computes the near-zero governance or the final mask itself** —
  those are computed at pipeline runtime in
  `src/preprocessing/ciciot2023/pipeline.py`
  (`build_near_zero_report`, `build_netdiffuser`, `build_perturbation_mask`).
  These two files only supply the *constants and the algorithm* those functions
  consume.
- **Correctness:** the schema and all group partitions are internally
  consistent and verified (§3.5). Findings are: one medium methodological issue
  in NetDiffuser (§4.4 F1), one latent NaN bug on the EDA path (F2), and dead /
  misleading config in `feature_groups.py` (F3–F4). None corrupts the current
  training arrays, but F1 and F3 are worth fixing and disclosing.

---

## 1. `feature_groups.py` — what it defines

It is a flat module of constants (no classes, no I/O). Every symbol:

| Symbol | Type | Size | Purpose |
|---|---|--:|---|
| `LABEL_COLUMN` | str | — | `"Label"` |
| `FEATURE_NAMES` | list[str] | 39 | canonical feature order = CSV header minus `Label` |
| `EXPECTED_COLUMNS` | list[str] | 40 | `FEATURE_NAMES + ["Label"]` |
| `IMMUTABLE_FEATURES` | list[str] | 4 | fixed by the network stack |
| `QUASI_IMMUTABLE_FEATURES` | list[str] | 13 | app/protocol-layer indicators + TTL |
| `BINARY_FEATURES` | list[str] | 15 | protocol one-hot indicators, values in {0,1} |
| `INTEGER_FEATURES` | list[str] | 12 | TCP-flag numbers, flag counts, packet `Number` |
| `BASE_MUTABLE` | list[str] | 17 | attacker-craftable continuous/flag features |
| `MUTABLE_FEATURES` | list[str] | 22 | `BASE_MUTABLE` + the 4 flag counts + `Number` |
| `FULL_PERTURBABLE_OVERRIDE_FEATURES` | list[str] | 11 | high-IQR continuous → always full-perturb |
| `NEAR_ZERO_IQR_THRESHOLD` | float | — | `1e-6` IQR cutoff |
| `RARE_SIGNAL_NONZERO_THRESHOLD` | float | — | `1e-3` non-zero-fraction cutoff |
| `NEAR_ZERO_FREEZE_POLICY` | dict | 3 | `constant→auto_freeze`, `rare_signal→allow`, `concentrated→manual` |
| `MANUAL_CONCENTRATED_DECISIONS` | dict | 9 | human overrides for concentrated features |
| `CATEGORY_MAP` | dict | 34 | 34-class label → 8-way category |
| `get_feature_indices` | func | — | names → positional indices |
| `PERTURBATION_MASK` | np.ndarray | 39 | binary {0,1} mask over `MUTABLE_FEATURES` |

`FEATURE_NAMES` order was confirmed **byte-for-byte identical** to the raw shard
header (spot-checked on `XSS.pcap.csv`) and to the labelled
`ciciot2023_base.csv` header (which appends `Label`). Order is a hard contract:
every downstream `.npy` array, the `RobustScaler`, and the perturbation mask
index features by position, so reordering this list silently corrupts every
trained artifact.

---

## 2. How the feature groups were derived (report narrative)

This is the "how I created the feature groups" story, reconstructed from the
module's own annotations and the way each group is consumed. There are **three
independent groupings**, created for three different purposes:

### 2.1 The schema list (`FEATURE_NAMES`)
Transcribed directly from the vendor CSV header. CIC's shipped CICIoT2023 CSV
distribution ("Modified Schema A" — a 39-column subset of CICFlowMeter's
46-feature output; see `ciciot2023_building_from_download.md` §4). No feature
selection was performed here — the 39 columns are exactly what the vendor
shipped, in the order they appear. The module docstring records what this schema
*has* vs *lacks* relative to full CICFlowMeter output (missing `flow_duration`,
`Srate`/`Drate`, `urg_count`, `Magnitude`, `Radius`, `Covariance`, `Weight`).

### 2.2 The data-type grouping (`BINARY` / `INTEGER` / implicit float)
Created by **feature semantics**, to drive canonical rounding after cleaning:
- **`BINARY_FEATURES` (15)** — the protocol one-hot indicators (HTTP, HTTPS,
  DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC). Each
  is a presence flag ∈ {0,1}.
- **`INTEGER_FEATURES` (12)** — the seven TCP `*_flag_number` fields, the four
  flag `*_count` fields, and `Number` (packet count). Counts/flags are integer
  by nature.
- **Everything else (12)** — continuous floats (`Rate`, `IAT`, size/statistic
  aggregates, `Header_Length`, `Time_To_Live`) plus the categorical
  `Protocol Type`.

These types are enforced in `pipeline.clip_round` (`pipeline.py:140-143`):
integer columns are `round`ed and clipped to ≥0; binary columns are `round`ed
and clipped to {0,1}. They are also used by the domain validator and EDA typing.

### 2.3 The adversarial-robustness grouping (`IMMUTABLE` / `QUASI_IMMUTABLE` / `MUTABLE`)
This is the thesis-specific taxonomy — the **perturbation policy**: which
features an attacker can realistically change by crafting packets, versus which
are fixed by the environment. It answers "what is a *valid* adversarial
perturbation in feature space?"
- **`IMMUTABLE_FEATURES` (4)** — `Protocol Type`, `TCP`, `UDP`, `ICMP`.
  Determined by the L3/L4 protocol the attack uses; changing them changes the
  attack itself, not a perturbation of it.
- **`QUASI_IMMUTABLE_FEATURES` (13)** — the application-layer service indicators
  (HTTP…IGMP minus the L4 ones) plus `Time_To_Live`. Determined by the service
  and OS/network stack; not freely craftable without changing semantics.
- **`MUTABLE_FEATURES` (22)** — rates, packet-size statistics, TCP flag
  fields/counts, `Header_Length`, `IAT`, `Number`. These an attacker can shape
  through packet crafting.
- **`FULL_PERTURBABLE_OVERRIDE_FEATURES` (11)** — the subset of mutable features
  that are genuinely high-IQR continuous aggregates; flagged for *full*
  perturbation regardless of the near-zero governance below.

**Verified partition property (§3.5):** the frozen set (everything not in
`MUTABLE`) equals exactly `BINARY_FEATURES ∪ {Protocol Type, Time_To_Live}` —
i.e. **no binary protocol bit is ever mutable**. This is the intended invariant
(an attacker cannot flip "this flow is HTTP" while keeping it valid).

### 2.4 The near-zero-IQR governance
A refinement of the mutable set, motivated by the data: after clip+scale, the
DDoS-dominated train set (72.65% DDoS — manifest `per_category_rows`) collapses
some mutable features to a near-constant column (Q25==Q75). Perturbing such a
column is meaningless (truly constant) or dangerous (rare on/off signal). The
policy is data-driven and recomputed each run, but the **thresholds and the
manual human decisions** live in this file:
- `constant` (1 unique value) → `auto_freeze`.
- `rare_signal` (non-zero fraction < 1e-3) → `allow`.
- `concentrated` (varies but IQR≈0) → `manual`, resolved via
  `MANUAL_CONCENTRATED_DECISIONS` (all set to `allow_mutable` here, with an
  inline justification for `Min` and `Number`).

### 2.5 The category map (`CATEGORY_MAP`)
The CIC-documented 34-class → 8-category taxonomy (DDoS 12, DoS 4, Mirai 3,
Recon 5, Spoofing 2, Web 6, BruteForce 1, Benign 1). Keys are UPPERCASE to match
the `Label` values written by the parquet builder.

---

## 3. Downstream consumption (what actually uses each symbol)

Verified by grepping every importer under `src/`.

| Symbol | Consumers | Role |
|---|---|---|
| `FEATURE_NAMES` | ~30 modules (build, pipeline, vae, attack, evaluation) | positional feature contract |
| `CATEGORY_MAP` | build_labeled, pipeline, eda, validate_full_dataset | 34→8 labelling |
| `EXPECTED_COLUMNS` | build_labeled | header assertion |
| `BINARY_FEATURES` | pipeline (rounding), validator, EDA | {0,1} enforcement |
| `INTEGER_FEATURES` | pipeline (rounding), sample_exhibit, validate_full_dataset, EDA | integer enforcement |
| `MUTABLE_FEATURES` | pipeline.build_perturbation_mask, attack/latent_infra | mutable set |
| `FULL_PERTURBABLE_OVERRIDE_FEATURES` | pipeline, latent_infra | force full-perturb tier |
| `MANUAL_CONCENTRATED_DECISIONS` | pipeline, latent_infra | resolve concentrated features |
| `NEAR_ZERO_FREEZE_POLICY` | pipeline.build_near_zero_report | classify near-zero features |
| `IMMUTABLE_FEATURES` | sample_exhibit | constraint reporting |
| `QUASI_IMMUTABLE_FEATURES` | **none** | dead (see F3) |
| `PERTURBATION_MASK` (module const) | **none** | dead / misleading (see F3) |
| `get_feature_indices` | **none** (only builds the dead const) | dead (see F4) |

### 3.1 How the real (3-tier) perturbation mask is built
The authoritative mask is **not** the module-level `PERTURBATION_MASK`. It is the
39-value `perturbation_mask.npy` produced at runtime by
`pipeline.build_perturbation_mask` (`pipeline.py:289-331`) combining three
inputs: the near-zero report, the NetDiffuser discrete/relative split, and the
static mutable/override sets. Decision order per feature:

```
if feature auto-frozen (constant, or manual force_freeze)      -> 0.0 (frozen)
elif mutable AND in FULL_PERTURBABLE_OVERRIDE                   -> 1.0 (full)
elif mutable AND (rare_signal or concentrated near-zero)        -> 0.3 (partial)
elif mutable AND discrete (NetDiffuser)                         -> 1.0 (full)
elif mutable AND relative (NetDiffuser)                         -> 0.3 (partial)
else                                                            -> 0.0 (frozen)
```

So `feature_groups.py` supplies the *policy inputs*; the *tiers* (0.0/0.3/1.0)
are assigned in the pipeline.

### 3.2 Correctness strength worth citing
`build_perturbation_mask` has a **fail-loud guard** (`pipeline.py:301-303`): if
the data ever yields a `concentrated` mutable feature that is *not* covered by
`MANUAL_CONCENTRATED_DECISIONS`, it raises `RuntimeError` rather than silently
guessing. This means the checked-in manual decisions can't drift out of sync
with the data without a hard failure — good defensive design; safe to state in
the methodology.

### 3.3 Rounding path (type groups)
`clip_round` clips all features to `[0, train-99.99th-pct]`, then rounds
`INTEGER_FEATURES` to ≥0 integers and `BINARY_FEATURES` to {0,1}. Uses direct
`FEATURE_NAMES.index(f)` (raises on a typo — safe), **not** the silent
`get_feature_indices` helper.

### 3.4 NetDiffuser wiring
`pipeline.build_netdiffuser` (`pipeline.py:267-286`) seeds a ≤200k-row sample of
the subsampled train set, **drops constant-in-sample columns** (`usable`
filter), calls `categorize_features`, then folds the dropped constants into
`discrete`. Result is written to `netdiffuser_categorization.json` and consumed
both by the mask builder and by `attack/latent_infra.py:138`.

### 3.5 Verified invariants (ran against the constants)
```
n_features = 39, all unique
|MUTABLE| = 22, |frozen| = 17, disjoint, union = 39           ✓
no BINARY feature is in MUTABLE                                ✓
no IMMUTABLE feature is in MUTABLE                             ✓
FULL_PERTURBABLE_OVERRIDE ⊆ MUTABLE                            ✓
BINARY ∩ INTEGER = ∅                                          ✓
frozen == BINARY ∪ {Protocol Type, Time_To_Live}             ✓
every name in every group ∈ FEATURE_NAMES (no typos)          ✓
```

---

## 4. `netdiffuser_categorization.py` — what it does

Single public function `categorize_features(df, feature_cols, method='spearman',
h_grid_points=100)`.

### 4.1 Algorithm (NetDiffuser Algorithm 1)
1. **Correlation** — `|Spearman ρ|` matrix over `feature_cols`; diagonal set to
   1.0. (`.copy()` is needed because numpy 2.x returns a read-only view that
   `fill_diagonal` cannot mutate — correctly handled.)
2. **Distance** — `d = sqrt(2·(1−|ρ|))`, the standard correlation (chord)
   distance; `max(…,0)` guards float noise. Perfectly correlated features →
   distance 0; uncorrelated → √2 ≈ 1.414.
3. **Clustering** — condense to a vector (`squareform`, `checks=False`) →
   **average-linkage** hierarchical clustering (`scipy.linkage`).
4. **Cut sweep** — for each height `h` in `linspace(0.1, 1.0, 100)`, `fcluster`
   by distance; record (a) the Calinski–Harabasz score of the resulting
   partition and (b) a "non-trivial" flag = at least 3 clusters of size ≥ 2.
5. **Cut selection** — prefer CH **local maxima** that are non-trivial; fall
   back to the global non-trivial max; final fallback to any valid max.
6. **Partition** — features in singleton clusters → `discrete`; features in
   clusters of size > 1 → `relative`.
7. Returns both lists plus the linkage matrix, chosen cut, CH curve,
   non-trivial flags, grid, and the correlation matrix (for the F7 dendrogram
   figure).

### 4.2 Interpretation
- **`relative`** features move together (a correlated block) → the pipeline caps
  them at the **0.3 partial** perturbation tier (perturbing one in isolation is
  unrealistic).
- **`discrete`** features are statistically independent → eligible for the
  **1.0 full** tier.

### 4.3 What is correct / robust
- Spearman is rank-based, so the correlation/clustering step is **scale- and
  monotone-invariant** — appropriate for mixed-magnitude NIDS features.
- The distance is a valid metric; `average` linkage + `fcluster` are
  deterministic; the fallbacks guarantee a partition is always returned.
- Diagonal handling and the read-only-view `.copy()` are correct.

### 4.4 Findings / bugs

**F1 — MEDIUM (methodological): the cut is selected in a different geometry than
the clustering.** The tree is built on rank-correlation distance
(scale-invariant), but the Calinski–Harabasz score that *chooses the cut height*
is computed on `df[feature_cols].values.T` — i.e. each feature is a point whose
coordinates are its **raw, unscaled** values across the sampled rows
(`categorize_features` line 25, 37). Because the pipeline passes **cleaned but
unscaled** `X` (and the EDA path passes raw parquet values), Euclidean distances
between feature-points are dominated by the high-magnitude features (`Rate`,
`IAT`, `Tot sum`, whose values are in the thousands) while binaries contribute
~nothing. So the "optimal" cut is chosen by a magnitude-dominated criterion that
is inconsistent with the correlation-based tree it is cutting. It does not crash
and still returns a plausible split, but the selection rationale is shaky.
- *Fix:* either z-score/standardize the feature-vectors before the CH score, or
  (better) evaluate cut quality in the same space as the tree — e.g.
  `silhouette_score(dist, labels, metric='precomputed')` on the correlation
  distance matrix. CH cannot take a precomputed distance matrix, which is
  precisely why the code fell back to raw Euclidean space.
- *Impact:* affects only which mutable features land in the 0.3 vs 1.0 tier via
  the `discrete`/`relative` split; does not affect labels, splits, scaling, or
  the training arrays. Disclose in the methodology; consider re-running with a
  space-consistent selection and checking whether the partition changes.

**F2 — LOW→MEDIUM (latent NaN bug on the EDA path): no guard for zero-variance
columns.** If any `feature_col` is constant over the sample, `df.corr()`
produces `NaN` for that row/column → `NaN` in `dist` → `squareform`/`linkage`
silently propagate `NaN` (with `checks=False`) and the partition is garbage
(no exception guaranteed). The pipeline is safe because
`build_netdiffuser` pre-drops constant columns (`usable` filter). But
`evaluation/eda_figures_part2.py:200` calls
`categorize_features(nd_sample, FEATURE_NAMES)` on **all 39** columns, and the
rare protocol indicators (e.g. `IRC`, `Telnet`, `IGMP`) can be all-zero in a
50k-row sample → NaN corruption of the F7 NetDiffuser figure/JSON.
- *Fix:* move the constant-column drop **inside** `categorize_features` (or add
  `assert not np.isnan(corr).any()`), so both callers are protected and the two
  code paths can't disagree.

**F3 — LOW (hygiene, but misleading): dead + stale config in
`feature_groups.py`.**
- `PERTURBATION_MASK` (module constant, lines 166-168) is imported **nowhere**.
  It is a binary {0,1} mask over `MUTABLE_FEATURES` only — it does **not**
  implement the near-zero freeze or the 0.3 partial tier. The real mask is the
  3-tier `perturbation_mask.npy` built in the pipeline. A future reader could
  `from feature_groups import PERTURBATION_MASK` believing it authoritative and
  get a wrong 2-tier mask.
  *Fix:* delete it, or replace it with a clear comment pointing at
  `pipeline.build_perturbation_mask`.
- `QUASI_IMMUTABLE_FEATURES` is imported nowhere. Freezing is actually enforced
  by "not in `MUTABLE`" + the near-zero policy, not by this list. It documents
  intent but is never read.
  *Fix:* keep only if used as documentation; otherwise remove to avoid implying
  it is enforced.

**F4 — LOW (silent failure, confined to dead code): `get_feature_indices`
swallows typos.** `[FEATURE_NAMES.index(f) for f in feats if f in FEATURE_NAMES]`
silently skips any name not in `FEATURE_NAMES`, so a typo yields a
wrong-length mask with no error. It is only used to build the dead
`PERTURBATION_MASK`, so it is currently harmless — but if that mask is ever
revived, this hides bugs.
- *Fix:* drop the `if f in FEATURE_NAMES` guard so unknown names raise, matching
  the pipeline's own direct-`index` convention.

**F5 — INFORMATIONAL (reproducibility, not a bug): two callers seed the sample
differently.** `pipeline.build_netdiffuser` samples with `paths.SEED`;
`eda_figures_part2` samples with `random_state=42` and skips the constant-column
drop. So the pipeline's `netdiffuser_categorization.json` and the EDA figure's
version can differ. If the report shows the F7 figure as "the" partition, note
it may not be byte-identical to the one the mask was built from. Prefer
regenerating the figure from the pipeline's JSON.

### 4.5 Non-issues (checked, fine)
- `h_grid` starting at 0.1 excludes very tight cuts — documented as an
  intentional "publication stability" constraint, not a bug.
- `h_grid` topping out at 1.0 (< √2 max distance) is fine: it just means only
  feature pairs with |ρ| ≳ 0.5 can merge within the searched range.
- Plateau handling (`>=` on both sides marks a local max) is acceptable; ties
  are broken by CH value.

---

## 5. Recommended actions (in priority order)

1. **(F2)** Move the zero-variance-column guard into `categorize_features` (or
   assert no NaN in the correlation matrix). One-line safety fix; protects the
   EDA path. Behaviour-preserving for the pipeline.
2. **(F1)** Make the NetDiffuser cut-selection space-consistent (standardize
   before CH, or switch to precomputed-distance silhouette). Re-run and diff the
   `discrete`/`relative` partition; disclose the choice in the methodology.
3. **(F3)** Delete the dead `PERTURBATION_MASK` module constant (and
   `get_feature_indices`) or replace with a pointer to
   `pipeline.build_perturbation_mask`; decide whether to keep
   `QUASI_IMMUTABLE_FEATURES` as documentation.
4. Fix the stale "archived feature_groups module" comment in
   `build_labeled_ciciot2023_dataset.py:51` — it imports the **live**, canonical
   module, not an archived one.

None of these change the labels, splits, scaler, or training arrays already on
disk. F1/F3 are the only ones with report-methodology relevance.

---

## 6. One-paragraph methodology wording (drop-in)

> Feature definitions are centralised in `feature_groups.py`, which fixes the
> 39-feature order transcribed verbatim from the CIC-shipped CSV header and
> defines three orthogonal groupings: a data-type grouping (15 binary protocol
> indicators, 12 integer flag/count features, the remaining continuous
> features) used to enforce canonical value ranges after cleaning; and an
> adversarial-robustness grouping (immutable network-stack features,
> quasi-immutable service/OS features, and 22 attacker-mutable features) that
> defines the perturbation policy. On top of the mutable set, a data-driven
> near-zero-IQR governance (recomputed per run from the training split, with
> human overrides recorded in the module) and a NetDiffuser correlation-based
> discrete/relative partition together assign each feature to one of three
> perturbation tiers — frozen (0.0), partial (0.3), or full (1.0) — in
> `pipeline.build_perturbation_mask`. The NetDiffuser partition clusters
> features by absolute Spearman correlation under average linkage and selects
> the cut height by a Calinski–Harabasz criterion. The 34→8 category map follows
> CIC's documented taxonomy.
