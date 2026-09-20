"""Finalize provenance and write the thesis-quality corrected preprocessing report."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing.ciciot2023 import pipeline as pl
from src.preprocessing.ciciot2023.reporting import markdown_table
from src.preprocessing.schema import FEATURE_NAMES


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                result[str(path.relative_to(root)).replace("\\", "/")] = _sha256(path)
    return result


def run(root: Path) -> dict:
    manifest_path = root / "run_manifest.json"
    manifest = _json(manifest_path)
    audit = _json(root / "audits" / "audit_summary.json")
    duplicate = _json(root / "audits" / "duplicate_audit.json")
    analysis = _json(root / "eda" / "feature_analysis_summary.json")
    selected = _json(root / "feature_selection" / "selected_features.json")
    build = _json(root / "ciciot2023_labeled_full_manifest.json")
    clipping = pd.read_csv(root / "audits" / "clipping_impact.csv")
    raw_stats = pd.read_csv(root / "audits" / "raw_feature_stats.csv").set_index("feature")
    train_stats = pd.read_csv(root / "audits" / "train_preclean_stats.csv").set_index("feature")
    pairs = pd.read_csv(root / "feature_selection" / "correlation_pairs.csv")
    variability = pd.read_csv(root / "eda" / "feature_variability_global.csv")

    manifest["finalized_timestamp"] = datetime.now(timezone.utc).isoformat()
    manifest["duplicate_overlap_summary"] = duplicate
    manifest["optional_feature_selection"] = {
        "mode": selected["mode"],
        "threshold": selected["threshold"],
        "fit_population": selected["fit_population"],
        "selected_features": selected["model_schema"],
        "full_domain_schema": FEATURE_NAMES,
    }
    manifest["analysis_provenance"] = {
        "pearson": analysis["pearson"],
        "spearman": analysis["spearman"],
        "feature_selection_fit_population": "natural training only",
        "validation_test_used_for_policy_fitting": False,
    }
    manifest["report_artifact_hashes"] = _relative_hashes(
        root,
        ("audits/*.csv", "audits/*.json", "eda/*.csv", "eda/*.json", "feature_selection/*.csv", "feature_selection/*.json"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    near = variability[variability["near_constant"]]
    high = pairs[pairs["abs_spearman_rho"] >= 0.95]
    clip_totals = clipping.groupby("split", as_index=False).agg(
        affected_feature_cells=("affected_rows", "sum"),
        maximum_absolute_change=("maximum_absolute_change", "max"),
    )
    split_counts = pd.DataFrame([
        {"partition": "natural train before sampling", "rows": manifest["train_before_sampling"]},
        {"partition": "saved sampled train", "rows": manifest["train_after_sampling"]},
        {"partition": "full validation", "rows": manifest["split_row_counts"]["val"]},
        {"partition": "full test", "rows": manifest["split_row_counts"]["test"]},
    ])
    overlap = pd.DataFrame.from_dict(duplicate["intersections"], orient="index").reset_index(names="comparison")

    report = f"""# CICIoT2023 semantics-corrected preprocessing report

## 1. Dataset source

The run uses `{manifest['source_paths']['labelled_parquet']}`, built from **{build['n_files']}** class-separated CSV shards. The builder read **{build['total_rows_read']:,}** rows, retained **{build['kept_rows']:,}**, and removed **{build['dropped_nan_inf']:,}** rows containing NaN or infinity. The canonical feature order is the frozen 39-column Modified Schema A in `src/preprocessing/schema.py`.

Official feature-generation reference: [CICIoT2023 paper, Section 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/). The authors group extracted packet features into windows of 10 or 100 packets and calculate means. The local CSV rows must therefore be interpreted at window-aggregate level.

## 2. Original CICIoT2023 feature-generation semantics

A packet-level flag, service indicator, packet count, or protocol identifier is discrete before aggregation. Its mean over a packet window is continuous. Values such as `HTTP=0.2`, `syn_flag_number=0.1`, or a fractional count in a compatible source release are meaningful aggregate statistics rather than malformed bits/integers.

## 3. Why previous binary/integer assumptions were incorrect

The legacy cleaner mapped every positive value in 22 flag/service/protocol columns to one and rounded named count fields. This destroyed occurrence-frequency information and changed the empirical manifold. The corrected cleaner performs neither operation. Regression tests exercise `0.2` indicator and `0.06` count examples.

## 4. Corrected feature schema

`FeatureSpec` records define name, semantic family, CSV representation type, structural bounds where justified, source semantics, derived status, and a non-authoritative mutability placeholder. Structural constraints, empirical support, mutability, and relational constraints are separate concepts. The full domain schema always contains all 39 features.

## 5. Source-build verification

Raw/Parquet reconciliation: **{audit['raw_rows']:,} = {audit['parquet_rows']:,} + {audit['dropped_nonfinite_rows']:,}**. Exact per-feature raw, Parquet, train-preclean, train-postclean, validation, and test statistics are under `audits/`. Feature order and width are checked in `schema_audit.json`.

## 6. Split strategy

{markdown_table(split_counts)}

The split is performed before clipping, scaling, sampling, correlation, feature selection, or importance estimation. Validation and test are complete natural holdouts; only training is sampled.

## 7. Chronology validation

Chronology is **not demonstrated**. No timestamp is present; repository evidence proves source concatenation and deterministic natural shard ordering, not wall-clock ordering. The corrected term is **forward source-order split**. See `audits/chronology_report.md`.

## 8. Train-only transformations

`RobustScaler` is fitted on all {manifest['train_before_sampling']:,} cleaned natural-training rows before sampling. Clipping bounds, correlation, mutual information, dependency groups, and optional feature selection use training only. Validation/test do not influence fitted policies.

## 9. Clipping audit

Canonical mode: **`{manifest['clipping_mode']}`**. A counterfactual {pl.CLIP_PERCENTILE}th-percentile model clip was fitted on training only:

{markdown_table(clip_totals, 8)}

Clipping is a model-preprocessing option, not a validity bound. Rare values are not invalid by rarity.

## 10. Scaling

Scaling is an affine RobustScaler transform. The scaler is fitted before downsampling, so its median/IQR describe natural training rather than the sampled class mixture. `preprocessing_verification.txt` records the independent fit check.

## 11. Training sampling

Majority-category caps remain 200,000 with seeded cluster-proportional allocation, floor 500, and seed 42; BruteForce and Web remain whole. Because all released features are continuous aggregates, clustering now uses all 39 scaled features rather than the old 23-feature binary-exclusion subset. Sampling is still confined to training.

## 12. Feature distribution comparison

`Header_Length` is 0–60 in the raw CSV, labelled Parquet, and natural training; the former 99.99th bound of 60 was source-derived, not a clipping or column-mapping bug. This local source formulation differs materially from paper Table 5 (median about 54; maximum about 9.9 million).

Local `IAT` natural-training median is **{train_stats.loc['IAT', 'p50']:.10g}** and 99.99th percentile is **{train_stats.loc['IAT', 'p99_99']:.10g}**. Median `Rate × IAT` is approximately one, supporting seconds relative to a per-second rate. No repository preprocessing conversion exists. Paper-scale ~83 million IAT values describe a different extractor representation/unit; no speculative conversion was added.

## 13. Duplicate leakage analysis

{markdown_table(overlap, 8)}

Training contains {duplicate['splits']['train']['duplicate_percentage']:.4f}% duplicate rows. **{duplicate['cross_label']['unique_vectors_with_multiple_labels']:,}** unique feature vectors occur with multiple fine labels, covering **{duplicate['cross_label']['rows_on_cross_label_vectors']:,}** rows. Natural evaluation retains all duplicates. `novel_val_indices.npy` and `novel_test_indices.npy` provide non-destructive split-local novel-pattern views.

## 14. Correlation analysis

Pearson uses complete natural training via streaming sufficient statistics. Spearman uses a deterministic, category-stratified **{analysis['spearman']['rows']:,}-row training sample** and is explicitly approximate. High Spearman pairs at `|rho| >= 0.95`:

{markdown_table(high[['feature_a', 'feature_b', 'pearson_r', 'spearman_rho']], 7)}

## 15. Dependency groups

The strongest groups are Rate–IAT; flag-number/count pairs; ARP–IPv–LLC; AVG–Tot size; and Std–Variance. These are useful relational evidence. They are not perturbability permissions and do not imply that either feature is unnecessary to the domain validator.

## 16. Optional feature reduction

Threshold `|sampled Spearman rho| >= {analysis['selection_threshold']}` yields an optional {len(selected['model_schema'])}-feature classifier schema. Proposed ablation-only drops: **{', '.join(analysis['dropped_features'])}**. All remain in the full domain schema. The reduced schema is not canonical and must be retrained/evaluated before adoption.

## 17. VAE/adversarial implications

The corrected VAE uses continuous reconstruction for all 39 features and sigmoid-bounded continuous outputs for aggregate indicators. Bernoulli/BCE targets, protocol class embedding, derived protocol bits, integer Number projection, and hard binary attack projection were removed from canonical paths. Existing VAE checkpoints are incompatible and require retraining. Historical diagnostics listed in `audits/vae_adversarial_semantics_audit.md` remain excluded until migrated.

## 18. Generated artifacts

- arrays, labels, scaler, encoders, kept indices, feature schema, and `run_manifest.json` at the output root;
- exact stage statistics, clipping, chronology, duplicate, schema, before/after, and VAE audits under `audits/`;
- Pearson/Spearman matrices, heatmaps, variability, and mutual information under `eda/`;
- ranked pairs, dependency groups, decision matrix, optional schemas, and ablation config under `feature_selection/`.

## 19. Verification/tests

The regression suite covers fractional preservation, train-only fitting, untouched holdouts, frozen order, finite outputs, alignment, duplicate fixtures, deterministic selection/order, seed reproducibility, and continuous VAE bounded outputs. End-to-end verification checks every saved row against the source-row transform and validates manifest hashes.

## 20. Remaining caveats

- Spearman and per-category variability are deterministic samples, not exact 33-million-row ranks; Pearson and global variability are complete-training calculations.
- A 64-bit stable row hash is used for duplicate matching; collision risk is negligible but not mathematically zero.
- Source order is not proven chronology.
- Local Header_Length/IAT representations differ from paper Table 5; raw evidence is preserved rather than coerced.
- No corrected full-vs-reduced classifier retraining or adversarial rerun was performed automatically. Performance equivalence and attack impact remain experimental questions.

## Feature-reduction decision

1. **Constant features:** none.
2. **Near-constant features:** {', '.join(analysis['near_constant_features'])} (global dominant fraction >= 99.5%); none is auto-dropped because per-category variation can matter.
3. **Highly redundant features:** the high-correlation table above, especially exact AVG–Tot size, exact monotonic Std–Variance, and flag/count pairs.
4. **Meaningful derived relations:** Rate–IAT, AVG–Tot size, Std–Variance, size-statistic groups, and paired flag/count features.
5. **Classifier removals:** only the generated optional ablation candidates; no canonical removal without measured clean/adversarial equivalence.
6. **Full domain schema:** all 39, including every optional classifier drop.
7. **Threshold:** 0.95 is a reasonable conservative ablation threshold here; 0.70 would collapse substantively distinct features. The sampled nature of Spearman must remain disclosed.
8. **Reduced classifier performance:** not measured; configuration generated.
9. **Adversarial impact:** not measured; corrected VAE/classifier retraining is required.
10. **Recommendation:** use **all 39 features as the primary corrected thesis configuration now**. Treat reduced-classifier + full-validator schema as a prespecified secondary ablation; promote it only if clean and adversarial metrics remain materially equivalent.

## Reproduction commands

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.feature_audit --output-dir outputs/ciciot2023_semantics_corrected
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.feature_analysis --output-dir outputs/ciciot2023_semantics_corrected
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.pipeline --output-dir outputs/ciciot2023_semantics_corrected --clipping-mode none
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.verify_corrected --output-dir outputs/ciciot2023_semantics_corrected
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.compare_preprocessing --old-root outputs/ciciot2023_fixed --new-root outputs/ciciot2023_semantics_corrected
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.generate_report --output-dir outputs/ciciot2023_semantics_corrected
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m pytest src/preprocessing/ciciot2023/tests -q
```
"""
    (root / "preprocessing_report.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=pl.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"manifest": str(args.output_dir / 'run_manifest.json'), "report": str(args.output_dir / 'preprocessing_report.md'), "artifact_hashes": len(result['artifact_hashes'])}, indent=2))


if __name__ == "__main__":
    main()
