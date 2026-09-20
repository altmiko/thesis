"""Training-only CICIoT2023 dependency, variability, and feature-selection analysis.

Pearson correlation is accumulated over the complete natural training split.
Spearman, per-category variability, and mutual information use deterministic
stratified training samples and record that approximation explicitly. Feature
reduction is an optional classifier ablation; the full 39-feature domain schema
is never reduced.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif

from config import paths
from src.preprocessing.ciciot2023 import pipeline as pl
from src.preprocessing.ciciot2023.reporting import markdown_table
from src.preprocessing.schema import FEATURE_METADATA, FEATURE_NAMES, FEATURE_SPECS


@dataclass(frozen=True)
class AnalysisConfig:
    output_dir: Path = pl.DEFAULT_OUTPUT_DIR
    spearman_per_category: int = 100_000
    importance_per_category: int = 25_000
    variability_per_category: int = 100_000
    selection_threshold: float = 0.95
    dependency_threshold: float = 0.90
    seed: int = paths.SEED
    pearson_chunk_size: int = 500_000

_PRIMITIVE_PRIORITY = {
    "IAT": 0, "Rate": 1,
    "AVG": 0, "Tot size": 1,
    "Std": 0, "Variance": 1,
    "IPv": 0, "LLC": 1, "ARP": 2,
    "fin_flag_number": 0, "fin_count": 1,
    "syn_flag_number": 0, "syn_count": 1,
    "rst_flag_number": 0, "rst_count": 1,
    "ack_flag_number": 0, "ack_count": 1,
}


def _primitive_rank(name: str) -> tuple[int, int]:
    """Semantic primitive preference, then frozen schema order."""

    return (_PRIMITIVE_PRIORITY.get(name, 0), FEATURE_NAMES.index(name))


def _stratified_indices(labels: np.ndarray, per_category: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for code in np.unique(labels):
        candidates = np.flatnonzero(labels == code)
        take = min(per_category, candidates.size)
        selected.append(rng.choice(candidates, take, replace=False))
    result = np.concatenate(selected).astype(np.int64)
    result.sort()
    return result


def _streaming_pearson(X: np.ndarray, indices: np.ndarray, chunk_size: int) -> np.ndarray:
    n_features = X.shape[1]
    count = 0
    total = np.zeros(n_features, dtype=np.float64)
    cross = np.zeros((n_features, n_features), dtype=np.float64)
    for start in range(0, indices.size, chunk_size):
        chosen = indices[start:start + chunk_size]
        chunk = np.asarray(X[chosen], dtype=np.float64)
        count += chunk.shape[0]
        total += chunk.sum(axis=0)
        cross += chunk.T @ chunk
    covariance = (cross - np.outer(total, total) / count) / max(count - 1, 1)
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(scale, scale)
    correlation = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _sampled_spearman(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(X[indices], dtype=np.float64)
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        ranks[:, column] = rankdata(values[:, column], method="average")
    return np.corrcoef(ranks, rowvar=False)


def _plot_matrix(matrix: np.ndarray, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    positions = np.arange(len(FEATURE_NAMES))
    ax.set_xticks(positions, FEATURE_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(positions, FEATURE_NAMES, fontsize=6)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _correlation_pairs(pearson: np.ndarray, spearman: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left in range(len(FEATURE_NAMES)):
        for right in range(left + 1, len(FEATURE_NAMES)):
            p = float(pearson[left, right])
            s = float(spearman[left, right])
            rows.append({
                "feature_a": FEATURE_NAMES[left],
                "feature_b": FEATURE_NAMES[right],
                "pearson_r": p,
                "abs_pearson_r": abs(p) if math.isfinite(p) else np.nan,
                "spearman_rho": s,
                "abs_spearman_rho": abs(s) if math.isfinite(s) else np.nan,
            })
    return pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False, na_position="last", ignore_index=True)


def _components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    adjacency = {index: set() for index in range(matrix.shape[0])}
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            value = matrix[left, right]
            if np.isfinite(value) and abs(value) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for root in range(matrix.shape[0]):
        if root in seen or not adjacency[root]:
            continue
        stack = [root]
        group: list[int] = []
        seen.add(root)
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(group))
    return groups


def _dependency_groups(matrix: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for number, indices in enumerate(_components(matrix, threshold), start=1):
        features = [FEATURE_NAMES[index] for index in indices]
        pairs: list[dict[str, Any]] = []
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1:]:
                pairs.append({
                    "feature_a": FEATURE_NAMES[left],
                    "feature_b": FEATURE_NAMES[right],
                    "spearman_rho": float(matrix[left, right]),
                })
        primitive = min(features, key=_primitive_rank)
        derived = [name for name in features if FEATURE_METADATA[name].derived]
        families = sorted({FEATURE_METADATA[name].semantic_family for name in features})
        groups.append({
            "group": number,
            "threshold": threshold,
            "features": features,
            "pairwise_correlations": pairs,
            "semantic_families": families,
            "possible_semantic_relation": "shared family or empirically coupled window aggregates",
            "candidate_primitive_feature": primitive,
            "candidate_derived_features": derived,
        })
    return groups


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _variability_rows(
    X: np.ndarray,
    indices: np.ndarray,
    category: str,
    basis: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, feature in enumerate(FEATURE_NAMES):
        values = np.asarray(X[indices, column], dtype=np.float64)
        unique, counts = np.unique(values, return_counts=True)
        median = np.median(values)
        dominant_position = int(np.argmax(counts))
        rows.append({
            "category": category,
            "basis": basis,
            "rows": int(values.size),
            "feature": feature,
            "variance": float(values.var(ddof=1)) if values.size > 1 else 0.0,
            "mad": float(np.median(np.abs(values - median))),
            "unique_count": int(unique.size),
            "dominant_value": float(unique[dominant_position]),
            "dominant_value_fraction": float(counts[dominant_position] / values.size),
            "zero_fraction": float(np.count_nonzero(values == 0.0) / values.size),
            "nonzero_fraction": float(np.count_nonzero(values != 0.0) / values.size),
            "entropy_bits": _entropy(counts),
            "constant": bool(unique.size == 1),
            "near_constant": bool(counts[dominant_position] / values.size >= 0.995),
            "highly_sparse": bool(np.count_nonzero(values == 0.0) / values.size >= 0.995),
        })
    return rows


def _mutual_information(
    X: np.ndarray,
    category_labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    values = np.asarray(X[indices], dtype=np.float64)
    coarse = category_labels[indices]
    binary = (coarse != 0).astype(np.int32)
    coarse_mi = mutual_info_classif(values, coarse, discrete_features=False, random_state=seed)
    binary_mi = mutual_info_classif(values, binary, discrete_features=False, random_state=seed)
    return pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mutual_information_binary": binary_mi,
        "mutual_information_coarse_8class": coarse_mi,
        "fit_population": "deterministic category-stratified natural-training sample",
        "sample_rows": indices.size,
    })


def _selection(
    spearman: np.ndarray,
    threshold: float,
    variability: pd.DataFrame,
    importance: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]]]:
    global_var = variability[variability["category"] == "ALL"].set_index("feature")
    mi = importance.set_index("feature")["mutual_information_coarse_8class"]
    dropped: list[dict[str, Any]] = []
    selected = set(FEATURE_NAMES)
    for group_number, indices in enumerate(_components(spearman, threshold), start=1):
        def score(index: int) -> tuple[float, float, float, float, int]:
            name = FEATURE_NAMES[index]
            spec = FEATURE_METADATA[name]
            primitive_penalty = float(_PRIMITIVE_PRIORITY.get(name, 0))
            derived_penalty = 1.0 if spec.derived else 0.0
            near_constant_penalty = 1.0 if bool(global_var.loc[name, "near_constant"]) else 0.0
            return (primitive_penalty, derived_penalty, near_constant_penalty, -float(mi.loc[name]), index)
        keep_index = min(indices, key=score)
        keep = FEATURE_NAMES[keep_index]
        for index in indices:
            if index == keep_index:
                continue
            name = FEATURE_NAMES[index]
            selected.discard(name)
            dropped.append({
                "feature": name,
                "group": group_number,
                "kept_feature": keep,
                "max_abs_spearman_in_group": float(max(abs(spearman[index, other]) for other in indices if other != index)),
                "reason": "optional correlation ablation: deterministic primitive/non-derived, variability, training-MI, then schema-order ranking",
                "full_domain_schema_status": "retained",
            })
    selected_ordered = [name for name in FEATURE_NAMES if name in selected]
    return selected_ordered, dropped


def _write_dependency_markdown(groups: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Correlation dependency groups",
        "",
        "Groups are empirical dependencies, not perturbability permissions and not domain-validity rules.",
        "",
    ]
    for group in groups:
        lines.extend([
            f"## Group {group['group']}", "",
            f"Features: {', '.join(group['features'])}", "",
            f"Candidate primitive: `{group['candidate_primitive_feature']}`", "",
            f"Candidate derived features: {', '.join(group['candidate_derived_features']) or 'none'}", "",
            markdown_table(pd.DataFrame(group["pairwise_correlations"]), 6), "",
        ])
    output.write_text("\n".join(lines), encoding="utf-8")


def run(config: AnalysisConfig) -> dict[str, Any]:
    eda_dir = config.output_dir / "eda"
    selection_dir = config.output_dir / "feature_selection"
    eda_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(paths.LABELED_PARQUET, columns=FEATURE_NAMES + ["Label", "category", "source_csv_filename"])
    X = np.ascontiguousarray(frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False))
    split, _, _ = pl.compute_split(frame[["source_csv_filename", "Label", "category"]])
    train_indices = np.flatnonzero(split == 0)
    category_names = sorted(frame["category"].astype(str).unique())
    category_map = {name: index for index, name in enumerate(category_names)}
    category_codes = frame["category"].astype(str).map(category_map).to_numpy(np.int32)
    train_categories = category_codes[train_indices]

    pearson = _streaming_pearson(X, train_indices, config.pearson_chunk_size)
    spearman_local = _stratified_indices(train_categories, config.spearman_per_category, config.seed)
    spearman_indices = train_indices[spearman_local]
    spearman = _sampled_spearman(X, spearman_indices)
    pearson_frame = pd.DataFrame(pearson, index=FEATURE_NAMES, columns=FEATURE_NAMES)
    spearman_frame = pd.DataFrame(spearman, index=FEATURE_NAMES, columns=FEATURE_NAMES)
    pearson_frame.to_csv(eda_dir / "train_pearson_correlation.csv")
    spearman_frame.to_csv(eda_dir / "train_spearman_correlation.csv")
    _plot_matrix(pearson, "Pearson correlation: complete natural training partition", eda_dir / "train_pearson_correlation.png")
    _plot_matrix(spearman, "Spearman correlation: deterministic stratified natural-training sample", eda_dir / "train_spearman_correlation.png")

    pairs = _correlation_pairs(pearson, spearman)
    pairs.to_csv(selection_dir / "correlation_pairs.csv", index=False)
    for threshold in (0.70, 0.80, 0.90, 0.95, 0.99):
        pairs[pairs["abs_spearman_rho"] >= threshold].to_csv(
            selection_dir / f"correlation_pairs_ge_{str(threshold).replace('.', '_')}.csv", index=False
        )

    dependency_groups = _dependency_groups(spearman, config.dependency_threshold)
    (selection_dir / "dependency_groups.json").write_text(json.dumps(dependency_groups, indent=2), encoding="utf-8")
    _write_dependency_markdown(dependency_groups, selection_dir / "dependency_groups.md")

    variability_rows = _variability_rows(X, train_indices, "ALL", "complete natural training")
    variability_sample_local = _stratified_indices(train_categories, config.variability_per_category, config.seed)
    variability_sample = train_indices[variability_sample_local]
    by_class_rows: list[dict[str, Any]] = []
    for code, name in enumerate(category_names):
        indices = variability_sample[category_codes[variability_sample] == code]
        by_class_rows.extend(_variability_rows(X, indices, name, "deterministic natural-training sample"))
    variability_global = pd.DataFrame(variability_rows)
    variability_by_class = pd.DataFrame(by_class_rows)
    variability_global.to_csv(eda_dir / "feature_variability_global.csv", index=False)
    variability_by_class.to_csv(eda_dir / "feature_variability_by_class.csv", index=False)
    near = variability_global[variability_global["constant"] | variability_global["near_constant"] | variability_global["highly_sparse"]]
    (eda_dir / "near_constant_report.md").write_text(
        "# Constant, near-constant, and sparse features\n\nNo feature is removed automatically. Per-category results are sampled and may reveal category-specific variation.\n\n"
        + markdown_table(near, 8) + "\n",
        encoding="utf-8",
    )

    importance_local = _stratified_indices(train_categories, config.importance_per_category, config.seed)
    importance_indices = train_indices[importance_local]
    importance = _mutual_information(X, category_codes, importance_indices, config.seed)
    importance.to_csv(eda_dir / "training_mutual_information.csv", index=False)

    selected, dropped = _selection(spearman, config.selection_threshold, variability_global, importance)
    selected_payload = {
        "mode": "optional_classifier_ablation",
        "fit_population": "natural training only",
        "threshold": config.selection_threshold,
        "full_domain_schema": FEATURE_NAMES,
        "model_schema": selected,
    }
    (selection_dir / "selected_features.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    (selection_dir / "dropped_features.json").write_text(json.dumps(dropped, indent=2), encoding="utf-8")

    top_map: dict[str, tuple[str, float]] = {}
    for feature in FEATURE_NAMES:
        candidates = pairs[(pairs["feature_a"] == feature) | (pairs["feature_b"] == feature)]
        if candidates.empty:
            top_map[feature] = ("", np.nan)
        else:
            row = candidates.iloc[candidates["abs_spearman_rho"].argmax()]
            other = row["feature_b"] if row["feature_a"] == feature else row["feature_a"]
            top_map[feature] = (str(other), float(row["abs_spearman_rho"]))

    raw_stats_path = config.output_dir / "audits" / "raw_feature_stats.csv"
    train_stats_path = config.output_dir / "audits" / "train_preclean_stats.csv"
    raw_stats = pd.read_csv(raw_stats_path).set_index("feature") if raw_stats_path.exists() else None
    train_stats = pd.read_csv(train_stats_path).set_index("feature") if train_stats_path.exists() else None
    global_var = variability_global.set_index("feature")
    dropped_names = {entry["feature"] for entry in dropped}
    decision_rows: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        top_feature, top_correlation = top_map[spec.name]
        decision_rows.append({
            "Feature": spec.name,
            "Semantic family": spec.semantic_family,
            "Raw min": float(raw_stats.loc[spec.name, "min"]) if raw_stats is not None else np.nan,
            "Raw max": float(raw_stats.loc[spec.name, "max"]) if raw_stats is not None else np.nan,
            "Train min": float(train_stats.loc[spec.name, "min"]) if train_stats is not None else np.nan,
            "Train max": float(train_stats.loc[spec.name, "max"]) if train_stats is not None else np.nan,
            "Train 25%": float(train_stats.loc[spec.name, "p25"]) if train_stats is not None else np.nan,
            "Train 50%": float(train_stats.loc[spec.name, "p50"]) if train_stats is not None else np.nan,
            "Train 75%": float(train_stats.loc[spec.name, "p75"]) if train_stats is not None else np.nan,
            "Unique count": int(global_var.loc[spec.name, "unique_count"]),
            "Zero %": float(100 * global_var.loc[spec.name, "zero_fraction"]),
            "Dominant value %": float(100 * global_var.loc[spec.name, "dominant_value_fraction"]),
            "Variance": float(global_var.loc[spec.name, "variance"]),
            "MAD": float(global_var.loc[spec.name, "mad"]),
            "Top correlated feature": top_feature,
            "Max abs Spearman": top_correlation,
            "Possible derived feature": spec.derived,
            "Bounded": spec.expected_max is not None,
            "Candidate classifier decision": "drop in optional ablation" if spec.name in dropped_names else "keep",
            "Reason": "correlation-cluster ranking" if spec.name in dropped_names else "canonical 39-feature baseline",
            "Potential perturbability notes": "must be inferred separately from training data and threat model",
            "Potential validator/dependency notes": "retain in full domain schema; structural, empirical, mutability, and relational constraints remain separate",
        })
    decision = pd.DataFrame(decision_rows)
    decision.to_csv(selection_dir / "feature_decision_matrix.csv", index=False)
    (selection_dir / "feature_decision_matrix.md").write_text(
        "# Feature-reduction decision matrix\n\nClassifier reduction is separate from perturbability and domain validity. All 39 features remain in the full domain schema.\n\n"
        + markdown_table(decision, 7) + "\n",
        encoding="utf-8",
    )

    (selection_dir / "feature_selection_report.md").write_text(
        "\n".join([
            "# Optional correlation-based classifier reduction",
            "",
            f"Threshold: `abs(sampled Spearman rho) >= {config.selection_threshold}`.",
            f"Selected classifier features: {len(selected)} of {len(FEATURE_NAMES)}.",
            "",
            "This is an ablation recommendation, not a canonical preprocessing change. Correlated features remain in the full domain schema for dependency mining and validity checks. Low correlation does not imply perturbability; high correlation does not imply immutability.",
            "",
            "## Selected",
            "",
            ", ".join(selected),
            "",
            "## Optional drops",
            "",
            markdown_table(pd.DataFrame(dropped), 6) if dropped else "None at this threshold.",
            "",
            "Clean-model and adversarial equivalence are not inferred from correlation. The generated ablation configuration requires retraining/evaluation before adoption.",
            "",
        ]),
        encoding="utf-8",
    )

    ablation = {
        "seed": config.seed,
        "full_domain_schema": FEATURE_NAMES,
        "models": {
            "all_39": {"model_features": FEATURE_NAMES},
            "correlation_reduced": {"model_features": selected, "selection_fit": "natural training only", "threshold": config.selection_threshold},
        },
        "required_clean_metrics": ["accuracy", "macro_f1", "weighted_f1", "per_class_recall", "benign_fpr"],
        "required_adversarial_metrics": ["target_to_benign_asr", "valid_asr", "raw_asr", "perturbation_magnitude", "modified_feature_count", "constraint_violation_rate"],
        "status": "configuration_only; expensive retraining and attacks not run automatically",
    }
    (selection_dir / "full_vs_reduced_ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")

    summary = {
        "fit_population": "natural training only",
        "natural_training_rows": int(train_indices.size),
        "pearson": "complete natural training, streaming sufficient statistics",
        "spearman": {"method": "deterministic category-stratified sample", "rows": int(spearman_indices.size)},
        "dependency_threshold": config.dependency_threshold,
        "selection_threshold": config.selection_threshold,
        "selected_features": selected,
        "dropped_features": [entry["feature"] for entry in dropped],
        "constant_features": variability_global.loc[variability_global["constant"], "feature"].tolist(),
        "near_constant_features": variability_global.loc[variability_global["near_constant"], "feature"].tolist(),
    }
    (eda_dir / "feature_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=pl.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spearman-per-category", type=int, default=100_000)
    parser.add_argument("--importance-per-category", type=int, default=25_000)
    parser.add_argument("--variability-per-category", type=int, default=100_000)
    parser.add_argument("--selection-threshold", type=float, default=0.95)
    parser.add_argument("--dependency-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    args = parser.parse_args()
    result = run(AnalysisConfig(
        output_dir=args.output_dir,
        spearman_per_category=args.spearman_per_category,
        importance_per_category=args.importance_per_category,
        variability_per_category=args.variability_per_category,
        selection_threshold=args.selection_threshold,
        dependency_threshold=args.dependency_threshold,
        seed=args.seed,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
