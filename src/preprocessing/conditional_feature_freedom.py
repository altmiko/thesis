"""Conditional Feature Freedom (CFF) for repository network-flow model features.

CFF is a class-conditional, data-driven proxy for ranking candidate perturbable
features. For each model feature X_i and traffic class c, LightGBM predicts X_i
from every other canonical model feature using explicit out-of-fold prediction.
The robust residual width is normalized by the feature's robust natural width::

    r_i = X_i - f_i(X_-i)
    W_r(i) = Q_0.95(r_i) - Q_0.05(r_i)
    W_x(i) = Q_0.95(X_i) - Q_0.05(X_i)
    CFF_i = clip(W_r(i) / (W_x(i) + epsilon), 0, 1)

Higher CFF means greater observed conditional freedom; lower CFF means greater
conditional predictability or structural constraint. CFF does not prove attacker
accessibility, packet-level mutability, functionality preservation, or causal
controllability. Those properties require separate problem-space validation.

Near-constant detection catches a single repeated value. Class-relative span
degeneracy separately catches a feature whose within-class robust span is tiny
in absolute terms or relative to its pooled training-class robust span.

Repository-native inputs are saved TRAINING artifact directories containing
X_train.npy and y_train_cat.npy. CICIoT2023 uses the fixed order from
``src.preprocessing.schema.FEATURE_NAMES``; CICIDS2017-DistriNet uses the exact
79-feature order and category-ID mapping persisted in its preprocessing manifest
and label encoders. Both matrices are model-ready RobustScaler space. Validation
and test artifacts are never opened. A train-only CSV or Parquet file is also
supported when its parent artifact directory provides the feature contract.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import SEED  # noqa: E402
from src.preprocessing.schema import FEATURE_NAMES  # noqa: E402

LOGGER = logging.getLogger("cff")

DEFAULT_INPUT = _REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "cff"
DEFAULT_SAMPLE_PER_CLASS = 20_000
DEFAULT_N_ESTIMATORS = 150
DEFAULT_CV_FOLDS = 3
DEFAULT_HARD_EXCLUDE = frozenset(
    {
        "Flow ID",
        "Src IP",
        "Source IP",
        "Dst IP",
        "Destination IP",
        "Timestamp",
        "Label",
        "Protocol",
        "Protocol Type",
        "Destination Port",
        "Dst Port",
    }
)
SELECTION_FRACTIONS = {"top10": 0.10, "top25": 0.25, "top50": 0.50}


@dataclass(frozen=True)
class TrainingData:
    """One training-only matrix plus its exact feature and class contracts."""

    X: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    source: Path
    representation: str
    label_source: str
    feature_order_source: str


@dataclass(frozen=True)
class RunConfig:
    """Resolved CFF settings shared by every feature and class."""

    cv_folds: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    near_constant_threshold: float
    span_degeneracy_ratio: float
    min_absolute_span: float
    epsilon: float
    seed: int


def require_lightgbm() -> tuple[Any, str]:
    """Import LightGBM lazily and fail without changing estimators silently."""

    try:
        import lightgbm
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for CFF. Install it in the repository environment with:\n"
            "    python -m pip install lightgbm"
        ) from exc
    return LGBMRegressor, str(lightgbm.__version__)


def _validate_training_path(path: Path) -> None:
    """Reject an explicitly supplied tabular holdout path."""

    if path.is_file() and not re.search(r"(^|[_\-.])train([_\-.]|$)", path.stem, re.IGNORECASE):
        raise ValueError(
            f"CFF accepts training data only; file name must identify a train split: {path}"
        )


def _scan_finite(X: np.ndarray, chunk_rows: int = 250_000) -> None:
    """Fail on non-finite model values without materializing the full memmap."""

    for start in range(0, X.shape[0], chunk_rows):
        block = np.asarray(X[start : start + chunk_rows])
        invalid = int(block.size - np.count_nonzero(np.isfinite(block)))
        if invalid:
            raise ValueError(
                f"training matrix contains {invalid} non-finite values in rows "
                f"[{start}, {min(start + chunk_rows, X.shape[0])}); CFF does not invent imputation"
            )


def _decode_array_labels(encoded: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    values = np.asarray(encoded)
    if values.ndim != 1:
        raise ValueError(f"label array must be one-dimensional, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"encoded label array must be integer, got {values.dtype}")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class names must be non-empty and unique")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= len(class_names)):
        raise ValueError("encoded labels are outside the configured class-name mapping")
    names_array = np.asarray([str(name) for name in class_names], dtype=object)
    return names_array[values.astype(np.int64, copy=False)]


def _directory_contract(directory: Path) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Resolve repository-native feature and category orders from artifacts."""

    manifest_path = directory / "preprocessing_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_names = tuple(str(name) for name in manifest.get("modelling_feature_names", []))
        if not feature_names:
            raise ValueError(f"{manifest_path} has no modelling_feature_names")
        encoders_path = directory / "label_encoders.json"
        if not encoders_path.exists():
            raise FileNotFoundError(f"training artifact directory is missing: {encoders_path.name}")
        encoders = json.loads(encoders_path.read_text(encoding="utf-8"))
        category_mapping = encoders.get("category", {})
        if not isinstance(category_mapping, dict) or not category_mapping:
            raise ValueError(f"{encoders_path} has no category mapping")
        ordered = sorted(category_mapping.items(), key=lambda item: int(item[1]))
        if [int(index) for _, index in ordered] != list(range(len(ordered))):
            raise ValueError(f"{encoders_path} category IDs must be contiguous from zero")
        class_names = tuple(str(name) for name, _ in ordered)
        return feature_names, class_names, str(manifest_path)

    names_path = directory / "category_names.json"
    if not names_path.exists():
        raise FileNotFoundError(f"training artifact directory is missing: {names_path.name}")
    raw_names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(raw_names, list) or not raw_names:
        raise ValueError(f"invalid class-name list: {names_path}")
    return tuple(FEATURE_NAMES), tuple(str(name) for name in raw_names), (
        "src.preprocessing.schema.FEATURE_NAMES"
    )


def resolve_feature_columns(
    frame: pd.DataFrame, label_column: str, feature_names: Sequence[str]
) -> list[str]:
    """Validate and return the configured model feature order for tabular input."""

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"training table is missing configured model features: {missing}")
    if label_column not in frame.columns:
        raise ValueError(f"training table is missing label column {label_column!r}")
    ignored = [
        name for name in frame.columns if name not in feature_names and name != label_column
    ]
    if ignored:
        LOGGER.info("Ignoring non-model metadata columns: %s", ignored)
    return list(feature_names)


def load_training_data(input_path: Path, label_column: str) -> TrainingData:
    """Load repository-native NumPy artifacts or a train-only Parquet/CSV table."""

    path = input_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"training input does not exist: {path}")
    _validate_training_path(path)
    contract_dir = path if path.is_dir() else path.parent
    feature_names, class_names, feature_order_source = _directory_contract(contract_dir)

    if path.is_dir():
        x_path = path / "X_train.npy"
        y_path = path / "y_train_cat.npy"
        missing = [p.name for p in (x_path, y_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"training artifact directory is missing: {missing}")
        X = np.load(x_path, mmap_mode="r")
        encoded = np.load(y_path, mmap_mode="r")
        labels = _decode_array_labels(encoded, class_names)
        source = x_path
        representation = "model-ready RobustScaler space"
        label_source = str(y_path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        columns = resolve_feature_columns(frame, label_column, feature_names)
        X = np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float32, copy=False))
        labels = frame[label_column].astype(str).to_numpy()
        source = path
        representation = "cleaned raw table values"
        label_source = f"{path}:{label_column}"
    elif path.suffix.lower() in {".csv", ".csv.gz"} or path.name.lower().endswith(".csv.gz"):
        frame = pd.read_csv(path)
        columns = resolve_feature_columns(frame, label_column, feature_names)
        X = np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float32, copy=False))
        labels = frame[label_column].astype(str).to_numpy()
        source = path
        representation = "cleaned raw table values"
        label_source = f"{path}:{label_column}"
    else:
        raise ValueError("--input must be a training-artifact directory, train Parquet, or train CSV")

    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(
            f"training matrix must have shape (n, {len(feature_names)}), got {X.shape}"
        )
    if labels.shape != (X.shape[0],):
        raise ValueError(f"feature/label row mismatch: {X.shape[0]} versus {labels.shape}")
    if X.shape[0] < 2:
        raise ValueError("training input must contain at least two rows")
    _scan_finite(X)
    return TrainingData(
        X=X,
        labels=np.asarray(labels, dtype=object),
        feature_names=feature_names,
        class_names=class_names,
        source=source,
        representation=representation,
        label_source=label_source,
        feature_order_source=feature_order_source,
    )


def resolve_classes(
    labels: np.ndarray,
    requested: Sequence[str] | None,
    configured: Sequence[str] | None = None,
) -> list[str]:
    """Resolve exact classes; by default preserve configured non-benign order."""

    observed = {str(value) for value in np.unique(labels)}
    if requested:
        classes = list(dict.fromkeys(str(value) for value in requested))
        missing = [name for name in classes if name not in observed]
        if missing:
            raise ValueError(f"requested classes are absent from training labels: {missing}")
    else:
        ordered = list(configured or sorted(observed))
        classes = [
            name
            for name in ordered
            if name in observed and name.casefold() not in {"benign", "normal"}
        ]
        if not classes:
            raise ValueError("no non-benign classes were found; pass --classes explicitly")
    LOGGER.info("Selected classes: %s", ", ".join(classes))
    return classes


def deterministic_class_sample(
    X: np.ndarray,
    labels: np.ndarray,
    class_name: str,
    sample_per_class: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Return a deterministic row sample from one training class only."""

    indices = np.flatnonzero(labels == class_name)
    total = int(indices.size)
    if total == 0:
        raise ValueError(f"class {class_name!r} has no training rows")
    if sample_per_class <= 0:
        raise ValueError("sample-per-class must be positive")
    if total > sample_per_class:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=sample_per_class, replace=False))
    sampled = np.ascontiguousarray(X[indices], dtype=np.float32)
    return sampled, total


def detect_degenerate_feature(
    values: np.ndarray, near_constant_threshold: float
) -> tuple[int, float, bool, bool]:
    """Return unique-count, dominant share, constant, and near-constant flags."""

    unique, counts = np.unique(values, return_counts=True)
    n_unique = int(unique.size)
    dominant_fraction = float(counts.max() / values.size)
    is_constant = n_unique <= 1
    is_near_constant = bool(not is_constant and dominant_fraction >= near_constant_threshold)
    return n_unique, dominant_fraction, is_constant, is_near_constant


def make_lgbm_regressor(config: RunConfig) -> Any:
    """Create the fixed deterministic LightGBM regressor used by every target."""

    LGBMRegressor, _ = require_lightgbm()
    return LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=config.seed,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _oof_predictions(
    predictors: np.ndarray,
    predictor_names: Sequence[str],
    target: np.ndarray,
    config: RunConfig,
    model_factory: Callable[[], Any],
) -> np.ndarray:
    """Generate explicit shuffled-KFold predictions using training rows only."""

    if target.size < config.cv_folds:
        raise ValueError(f"{target.size} rows cannot support {config.cv_folds}-fold CV")
    splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    if (
        len(predictor_names) != predictors.shape[1]
        or len(set(predictor_names)) != len(predictor_names)
    ):
        raise ValueError("predictor names must map one-to-one to predictor columns")
    predictor_frame = pd.DataFrame(predictors, columns=list(predictor_names), copy=False)
    predictions = np.empty(target.shape[0], dtype=np.float64)
    for train_idx, fold_idx in splitter.split(predictor_frame):
        model = model_factory()
        model.fit(predictor_frame.iloc[train_idx], target[train_idx])
        predictions[fold_idx] = model.predict(predictor_frame.iloc[fold_idx])
    if not np.isfinite(predictions).all():
        raise RuntimeError("regressor produced non-finite out-of-fold predictions")
    return predictions


def _base_feature_record(
    class_name: str,
    feature: str,
    feature_index: int,
    values: np.ndarray,
    hard_excluded: bool,
    global_feature_span: float,
    near_constant_threshold: float,
    span_degeneracy_ratio: float,
    min_absolute_span: float,
) -> dict[str, Any]:
    n_unique, dominant_fraction, is_constant, is_near_constant = detect_degenerate_feature(
        values, near_constant_threshold
    )
    feature_q05, feature_q95 = np.quantile(values, [0.05, 0.95])
    feature_span = float(feature_q95 - feature_q05)
    is_span_degenerate = bool(
        feature_span < min_absolute_span
        or feature_span < span_degeneracy_ratio * global_feature_span
    )
    return {
        "class": class_name,
        "feature": feature,
        "feature_index": feature_index,
        "cff_score": 0.0,
        "cff_raw": 0.0,
        "residual_q05": np.nan,
        "residual_q95": np.nan,
        "residual_span": np.nan,
        "feature_q05": float(feature_q05),
        "feature_q95": float(feature_q95),
        "feature_span": feature_span,
        "global_feature_span": float(global_feature_span),
        "cv_r2": np.nan,
        "normalized_mae": np.nan,
        "n_unique": n_unique,
        "dominant_fraction": dominant_fraction,
        "hard_excluded": bool(hard_excluded),
        "is_constant": is_constant,
        "is_near_constant": is_near_constant,
        "is_span_degenerate": is_span_degenerate,
        "eligible": bool(
            not hard_excluded
            and not is_constant
            and not is_near_constant
            and not is_span_degenerate
        ),
        "suspicious_model_fit": False,
        "rank": pd.NA,
        "selected_top10": False,
        "selected_top25": False,
        "selected_top50": False,
    }


def compute_feature_cff(
    X_class: np.ndarray,
    class_name: str,
    feature_index: int,
    feature_names: Sequence[str],
    global_feature_spans: dict[str, float],
    hard_exclude: set[str],
    config: RunConfig,
    model_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Compute one feature's class-conditional CFF record."""

    feature = feature_names[feature_index]
    target = np.asarray(X_class[:, feature_index], dtype=np.float64)
    record = _base_feature_record(
        class_name,
        feature,
        feature_index,
        target,
        feature in hard_exclude,
        global_feature_spans[feature],
        config.near_constant_threshold,
        config.span_degeneracy_ratio,
        config.min_absolute_span,
    )
    if not record["eligible"]:
        return record

    predictors = np.delete(X_class, feature_index, axis=1)
    predictor_names = [
        name for index, name in enumerate(feature_names) if index != feature_index
    ]
    factory = model_factory or (lambda: make_lgbm_regressor(config))
    predicted = _oof_predictions(predictors, predictor_names, target, config, factory)
    residual = target - predicted
    residual_q05, residual_q95 = np.quantile(residual, [0.05, 0.95])
    residual_span = float(residual_q95 - residual_q05)
    feature_span = float(record["feature_span"])
    cff_raw = residual_span / (feature_span + config.epsilon)
    score = float(np.clip(cff_raw, 0.0, 1.0))
    cv_r2 = float(r2_score(target, predicted))
    normalized_mae = float(mean_absolute_error(target, predicted) / (feature_span + config.epsilon))
    record.update(
        {
            "cff_score": score,
            "cff_raw": float(cff_raw),
            "residual_q05": float(residual_q05),
            "residual_q95": float(residual_q95),
            "residual_span": residual_span,
            "cv_r2": cv_r2,
            "normalized_mae": normalized_mae,
            "suspicious_model_fit": bool(cv_r2 < -1.0),
        }
    )
    return record


def select_top_fraction(frame: pd.DataFrame, fraction: float, column: str) -> None:
    """Mark the ceil(fraction * eligible_count) highest-ranked eligible rows."""

    eligible_count = int(frame["eligible"].sum())
    selected_count = math.ceil(fraction * eligible_count) if eligible_count else 0
    frame[column] = frame["eligible"] & (frame["rank"].fillna(eligible_count + 1) <= selected_count)


def _rank_class_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic ranks and all requested selection fractions."""

    ranked = frame.copy()
    eligible_indices = (
        ranked.loc[ranked["eligible"]]
        .sort_values(["cff_score", "feature_index"], ascending=[False, True], kind="mergesort")
        .index
    )
    ranked["rank"] = pd.array([pd.NA] * len(ranked), dtype="Int64")
    ranked.loc[eligible_indices, "rank"] = np.arange(1, len(eligible_indices) + 1)
    for name, fraction in SELECTION_FRACTIONS.items():
        select_top_fraction(ranked, fraction, f"selected_{name}")
    return ranked.sort_values("feature_index").reset_index(drop=True)


def compute_class_cff(
    X_class: np.ndarray,
    class_name: str,
    feature_names: Sequence[str],
    global_feature_spans: dict[str, float],
    hard_exclude: set[str],
    config: RunConfig,
) -> pd.DataFrame:
    """Compute and rank all canonical features for one class sequentially."""

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(feature_names):
        feature_started = time.perf_counter()
        record = compute_feature_cff(
            X_class,
            class_name,
            feature_index,
            feature_names,
            global_feature_spans,
            hard_exclude,
            config,
        )
        records.append(record)
        status = "eligible" if record["eligible"] else "excluded/degenerate"
        LOGGER.info(
            "  %02d/%02d %-24s CFF=%.4f (%s, %.2fs)",
            feature_index + 1,
            len(feature_names),
            feature,
            record["cff_score"],
            status,
            time.perf_counter() - feature_started,
        )
    frame = _rank_class_frame(pd.DataFrame.from_records(records))
    LOGGER.info("Completed %s in %.2fs", class_name, time.perf_counter() - started)
    return frame


def compute_all_classes(
    data: TrainingData,
    classes: Sequence[str],
    sample_per_class: int,
    hard_exclude: set[str],
    config: RunConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, float]]:
    """Compute pooled training spans, then CFF independently for every class."""

    sampled_by_class: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for class_name in classes:
        X_class, total = deterministic_class_sample(
            data.X,
            data.labels,
            class_name,
            sample_per_class,
            config.seed,
        )
        sampled_by_class[class_name] = X_class
        counts[class_name] = {
            "total_training_rows": total,
            "sampled_rows": int(X_class.shape[0]),
        }

    pooled = np.concatenate([sampled_by_class[name] for name in classes], axis=0)
    pooled_q05, pooled_q95 = np.quantile(pooled, [0.05, 0.95], axis=0)
    global_feature_spans = {
        feature: float(pooled_q95[index] - pooled_q05[index])
        for index, feature in enumerate(data.feature_names)
    }
    del pooled

    frames: list[pd.DataFrame] = []
    for class_name in classes:
        X_class = sampled_by_class[class_name]
        count = counts[class_name]
        LOGGER.info(
            "Class %s: total training rows=%d, sampled rows=%d, feature count=%d",
            class_name,
            count["total_training_rows"],
            count["sampled_rows"],
            X_class.shape[1],
        )
        frames.append(
            compute_class_cff(
                X_class,
                class_name,
                data.feature_names,
                global_feature_spans,
                hard_exclude,
                config,
            )
        )
    return pd.concat(frames, ignore_index=True), counts, global_feature_spans


def build_feature_mask(
    class_frame: pd.DataFrame,
    selection_column: str,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Build and validate one bool mask in configured model feature order."""

    selected = class_frame.loc[class_frame[selection_column], "feature"].tolist()
    index = {name: idx for idx, name in enumerate(feature_names)}
    if len(index) != len(feature_names):
        raise AssertionError("configured feature names are not unique")
    if len(selected) != len(set(selected)) or any(name not in index for name in selected):
        raise AssertionError("selected feature does not map exactly once to configured order")
    mask = np.zeros(len(feature_names), dtype=bool)
    for name in selected:
        mask[index[name]] = True
    if mask.shape != (len(feature_names),) or mask.dtype != np.bool_:
        raise AssertionError("generated mask has the wrong dimensionality or dtype")
    return mask


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "class"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_summary(
    scores: pd.DataFrame,
    sample_counts: dict[str, dict[str, int]],
    classes: Sequence[str],
    feature_names: Sequence[str],
) -> str:
    """Generate the conservative report-friendly Markdown summary."""

    lines = [
        "# Conditional Feature Freedom summary",
        "",
        "CFF is a class-conditional statistical proxy for ranking candidate perturbable features. "
        "It measures observed conditional freedom; it does not establish physical mutability or "
        "packet-level controllability.",
        "",
    ]
    for class_name in classes:
        frame = scores[scores["class"] == class_name].copy()
        eligible = frame[frame["eligible"]].sort_values("rank")
        bottom = eligible.sort_values(["cff_score", "feature_index"], ascending=[True, True])
        selected = (
            frame.loc[frame["selected_top25"]]
            .sort_values("rank")["feature"]
            .tolist()
        )
        counts = sample_counts[class_name]
        constant_count = int(frame["is_constant"].sum())
        near_count = int(frame["is_near_constant"].sum())
        span_degenerate_count = int(frame["is_span_degenerate"].sum())
        lines.extend(
            [
                f"## {class_name}",
                "",
                f"- Total training rows: {counts['total_training_rows']:,}",
                f"- Rows sampled: {counts['sampled_rows']:,}",
                f"- Total model features: {len(feature_names)}",
                f"- Hard-excluded feature count: {int(frame['hard_excluded'].sum())}",
                f"- Constant feature count: {constant_count}",
                f"- Near-constant feature count: {near_count}",
                f"- Class-span-degenerate feature count: {span_degenerate_count}",
                f"- Eligible feature count: {int(frame['eligible'].sum())}",
                f"- Mean eligible CFF: {eligible['cff_score'].mean():.6f}",
                f"- Median eligible CFF: {eligible['cff_score'].median():.6f}",
                f"- Median eligible OOF R²: {eligible['cv_r2'].median():.6f}",
                f"- Median eligible normalized MAE: {eligible['normalized_mae'].median():.6f}",
                "",
                "### Top 10 CFF-ranked candidate features",
                "",
                "| Rank | Feature | CFF | OOF R² | Normalized MAE |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in eligible.head(10).itertuples(index=False):
            lines.append(
                f"| {int(row.rank)} | {row.feature} | {row.cff_score:.6f} | "
                f"{row.cv_r2:.6f} | {row.normalized_mae:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Bottom 10 eligible features",
                "",
                "| Rank | Feature | CFF | OOF R² | Normalized MAE |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in bottom.head(10).itertuples(index=False):
            lines.append(
                f"| {int(row.rank)} | {row.feature} | {row.cff_score:.6f} | "
                f"{row.cv_r2:.6f} | {row.normalized_mae:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Selected top-25% candidate features",
                "",
                ", ".join(selected) if selected else "None",
                "",
            ]
        )
    return "\n".join(lines)


def plot_rankings(scores: pd.DataFrame, classes: Sequence[str], output_dir: Path) -> None:
    """Optionally save a compact top-20 eligible ranking plot per class."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plots requires matplotlib") from exc
    for class_name in classes:
        frame = (
            scores[(scores["class"] == class_name) & scores["eligible"]]
            .sort_values("rank")
            .head(20)
            .sort_values("cff_score")
        )
        fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(frame))))
        ax.barh(frame["feature"], frame["cff_score"], color="#3569a8")
        ax.set_xlabel("Conditional Feature Freedom")
        ax.set_title(f"{class_name}: top CFF-ranked candidate features")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(output_dir / f"cff_{_safe_filename(class_name)}_ranking.png", dpi=180)
        plt.close(fig)


def save_results(
    scores: pd.DataFrame,
    output_dir: Path,
    classes: Sequence[str],
    feature_names: Sequence[str],
    sample_counts: dict[str, dict[str, int]],
    metadata: dict[str, Any],
    make_plots: bool,
) -> None:
    """Save score tables, class masks, feature order, metadata, and summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    scores.to_csv(output_dir / "cff_scores_all.csv", index=False)
    masks_json: dict[str, dict[str, list[str]]] = {}

    for class_name in classes:
        class_frame = scores[scores["class"] == class_name].sort_values("feature_index")
        safe_name = _safe_filename(class_name)
        class_frame.to_csv(output_dir / f"cff_scores_{safe_name}.csv", index=False)
        masks_json[class_name] = {}
        eligible_mask = build_feature_mask(
            class_frame.assign(selected_eligible=class_frame["eligible"]),
            "selected_eligible",
            feature_names,
        )
        np.save(masks_dir / f"{safe_name}_eligible.npy", eligible_mask)
        masks_json[class_name]["eligible"] = [
            feature_names[idx] for idx in np.flatnonzero(eligible_mask)
        ]
        for selection_name in SELECTION_FRACTIONS:
            column = f"selected_{selection_name}"
            mask = build_feature_mask(class_frame, column, feature_names)
            np.save(masks_dir / f"{safe_name}_{selection_name}.npy", mask)
            masks_json[class_name][selection_name] = [
                feature_names[idx] for idx in np.flatnonzero(mask)
            ]

    _json_dump(output_dir / "cff_masks.json", masks_json)
    _json_dump(output_dir / "feature_order.json", list(feature_names))
    _json_dump(output_dir / "cff_run_metadata.json", metadata)
    (output_dir / "cff_summary.md").write_text(
        generate_summary(scores, sample_counts, classes, feature_names), encoding="utf-8"
    )
    if make_plots:
        plot_rankings(scores, classes, output_dir)


def _extra_trees_factory(config: RunConfig) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=min(config.n_estimators, 150),
        min_samples_leaf=1,
        random_state=config.seed,
        n_jobs=-1,
    )


def run_estimator_check(
    X_class: np.ndarray,
    class_name: str,
    primary: pd.DataFrame,
    feature_names: Sequence[str],
    global_feature_spans: dict[str, float],
    hard_exclude: set[str],
    config: RunConfig,
) -> float:
    """Compare eligible LightGBM and ExtraTrees CFF ranks for one class."""

    from scipy.stats import spearmanr

    records = []
    for feature_index in range(len(feature_names)):
        records.append(
            compute_feature_cff(
                X_class,
                class_name,
                feature_index,
                feature_names,
                global_feature_spans,
                hard_exclude,
                config,
                model_factory=lambda: _extra_trees_factory(config),
            )
        )
    secondary = _rank_class_frame(pd.DataFrame.from_records(records))
    joined = primary.loc[primary["eligible"], ["feature", "rank"]].merge(
        secondary.loc[secondary["eligible"], ["feature", "rank"]],
        on="feature",
        suffixes=("_lgbm", "_extra_trees"),
        validate="one_to_one",
    )
    if len(joined) < 2:
        raise ValueError("estimator check needs at least two eligible features")
    rho = float(spearmanr(joined["rank_lgbm"], joined["rank_extra_trees"]).statistic)
    LOGGER.info("Spearman rho between LightGBM-CFF and ExtraTrees-CFF: %.6f", rho)
    return rho


def run_synthetic_self_test(seed: int = SEED) -> None:
    """Check predictable, independent, degenerate, and excluded feature behavior."""

    require_lightgbm()
    rng = np.random.default_rng(seed)
    n = 1_200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2 + rng.normal(scale=0.01, size=n)
    x4 = rng.normal(size=n)
    x5 = np.ones(n)
    narrow_local = rng.normal(scale=1e-4, size=n)
    excluded = rng.normal(size=n)
    X = np.column_stack([x1, x2, x3, x4, x5, narrow_local, excluded]).astype(np.float32)
    names = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "class_local_narrow",
        "hard_excluded_feature",
    ]
    second_class = X.copy()
    second_class[:, 5] = rng.normal(scale=1.0, size=n)
    pooled_q05, pooled_q95 = np.quantile(
        np.concatenate([X, second_class], axis=0), [0.05, 0.95], axis=0
    )
    global_spans = pooled_q95 - pooled_q05
    config = RunConfig(
        cv_folds=3,
        n_estimators=75,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        near_constant_threshold=0.995,
        span_degeneracy_ratio=0.02,
        min_absolute_span=1e-3,
        epsilon=1e-12,
        seed=seed,
    )

    def compute(index: int, excluded_flag: bool = False) -> dict[str, Any]:
        target = np.asarray(X[:, index], dtype=np.float64)
        record = _base_feature_record(
            "synthetic",
            names[index],
            index,
            target,
            excluded_flag,
            float(global_spans[index]),
            config.near_constant_threshold,
            config.span_degeneracy_ratio,
            config.min_absolute_span,
        )
        if not record["eligible"]:
            return record
        predictors = np.delete(X, index, axis=1)
        predictor_names = [
            name for other_index, name in enumerate(names) if other_index != index
        ]
        predicted = _oof_predictions(
            predictors, predictor_names, target, config, lambda: make_lgbm_regressor(config)
        )
        residual = target - predicted
        rq05, rq95 = np.quantile(residual, [0.05, 0.95])
        cff_raw = float((rq95 - rq05) / (record["feature_span"] + config.epsilon))
        record["cff_score"] = float(np.clip(cff_raw, 0.0, 1.0))
        record["cv_r2"] = float(r2_score(target, predicted))
        return record

    x3_result = compute(2)
    x4_result = compute(3)
    x5_result = compute(4)
    narrow_result = compute(5)
    excluded_result = compute(6, excluded_flag=True)
    assert x3_result["cff_score"] < x4_result["cff_score"], (x3_result, x4_result)
    assert x3_result["cv_r2"] > x4_result["cv_r2"] + 0.5, (x3_result, x4_result)
    assert x5_result["cff_score"] == 0.0 and x5_result["is_constant"]
    assert narrow_result["is_span_degenerate"] and not narrow_result["eligible"]
    assert narrow_result["cff_score"] == 0.0
    assert excluded_result["cff_score"] == 0.0 and excluded_result["hard_excluded"]
    LOGGER.info(
        "Self-test passed: CFF(x3)=%.4f < CFF(x4)=%.4f; R²(x3)=%.4f > R²(x4)=%.4f; "
        "constant, class-span-degenerate, and hard-excluded scores are zero",
        x3_result["cff_score"],
        x4_result["cff_score"],
        x3_result["cv_r2"],
        x4_result["cv_r2"],
    )


def _resolved_numeric(value: int | None, normal: int, fast: int, fast_mode: bool) -> int:
    return int(value if value is not None else (fast if fast_mode else normal))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Training-artifact directory or train-only Parquet/CSV (default: data/processed)",
    )
    parser.add_argument(
        "--label-column",
        default="category",
        help="Class column for tabular input; ignored for repository NumPy artifacts",
    )
    parser.add_argument("--classes", nargs="+", help="Exact training classes; default: all non-benign")
    parser.add_argument("--sample-per-class", type=int, default=None, help="Default 20000; --fast default 10000")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-feature", action="append", default=[], help="Additional exact feature name; repeatable")
    parser.add_argument("--near-constant-threshold", type=float, default=0.995)
    parser.add_argument("--span-degeneracy-ratio", type=float, default=0.02)
    parser.add_argument("--min-absolute-span", type=float, default=1e-3)
    parser.add_argument("--cv", type=int, default=None, help="OOF folds; default 3, --fast default 2")
    parser.add_argument("--n-estimators", type=int, default=None, help="Default 150; --fast default 75")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Save optional top-20 ranking plots")
    parser.add_argument(
        "--estimator-check",
        action="store_true",
        help="Compare LightGBM and ExtraTrees ranks for the first selected class",
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic logical checks and exit")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("sample_per_class", "cv", "n_estimators", "num_leaves", "min_child_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.near_constant_threshold <= 1.0:
        raise ValueError("--near-constant-threshold must be in (0, 1]")
    if not 0.0 <= args.span_degeneracy_ratio <= 1.0:
        raise ValueError("--span-degeneracy-ratio must be in [0, 1]")
    if args.min_absolute_span < 0.0:
        raise ValueError("--min-absolute-span must be non-negative")
    if args.learning_rate <= 0 or args.epsilon <= 0:
        raise ValueError("--learning-rate and --epsilon must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        run_synthetic_self_test(args.seed)
        return

    args.sample_per_class = _resolved_numeric(
        args.sample_per_class, DEFAULT_SAMPLE_PER_CLASS, 10_000, args.fast
    )
    args.n_estimators = _resolved_numeric(
        args.n_estimators, DEFAULT_N_ESTIMATORS, 75, args.fast
    )
    args.cv = _resolved_numeric(args.cv, DEFAULT_CV_FOLDS, 2, args.fast)
    _validate_args(args)
    _, lightgbm_version = require_lightgbm()
    config = RunConfig(
        cv_folds=args.cv,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        near_constant_threshold=args.near_constant_threshold,
        span_degeneracy_ratio=args.span_degeneracy_ratio,
        min_absolute_span=args.min_absolute_span,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    started = time.perf_counter()
    data = load_training_data(args.input, args.label_column)
    feature_names = data.feature_names
    hard_exclude = set(DEFAULT_HARD_EXCLUDE) | set(args.exclude_feature)
    unknown_requested_exclusions = sorted(set(args.exclude_feature) - set(feature_names))
    if unknown_requested_exclusions:
        LOGGER.warning(
            "Requested exclusions absent from configured model schema: %s",
            unknown_requested_exclusions,
        )
    LOGGER.info(
        "Default hard exclusions present in model schema: %s",
        sorted(hard_exclude & set(feature_names)),
    )
    classes = resolve_classes(data.labels, args.classes, data.class_names)
    scores, sample_counts, global_feature_spans = compute_all_classes(
        data, classes, args.sample_per_class, hard_exclude, config
    )
    estimator_check: dict[str, Any] | None = None
    if args.estimator_check:
        check_class = classes[0]
        X_check, _ = deterministic_class_sample(
            data.X, data.labels, check_class, args.sample_per_class, config.seed
        )
        primary = scores[scores["class"] == check_class].copy()
        rho = run_estimator_check(
            X_check,
            check_class,
            primary,
            feature_names,
            global_feature_spans,
            hard_exclude,
            config,
        )
        estimator_check = {"class": check_class, "spearman_rho": rho}

    elapsed = time.perf_counter() - started
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.seed,
        "training_data_source": str(data.source),
        "label_source": data.label_source,
        "representation": data.representation,
        "train_only_contract": "Only X_train/y_train_cat or an explicitly train-named table is loaded",
        "classes": list(classes),
        "sample_counts": sample_counts,
        "cv_folds": config.cv_folds,
        "lightgbm_version": lightgbm_version,
        "lightgbm_parameters": {
            "objective": "regression",
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "min_child_samples": config.min_child_samples,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": config.seed,
            "n_jobs": -1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        },
        "near_constant_threshold": config.near_constant_threshold,
        "span_degeneracy_ratio": config.span_degeneracy_ratio,
        "min_absolute_span": config.min_absolute_span,
        "global_feature_spans": global_feature_spans,
        "epsilon": config.epsilon,
        "hard_exclusions": sorted(hard_exclude),
        "hard_exclusions_present": sorted(hard_exclude & set(feature_names)),
        "number_of_model_features": len(feature_names),
        "feature_order_source": data.feature_order_source,
        "feature_processing_parallelism": "sequential",
        "estimator_check": estimator_check,
        "runtime_seconds": elapsed,
    }
    save_results(
        scores,
        args.output_dir.resolve(),
        classes,
        feature_names,
        sample_counts,
        metadata,
        args.plots,
    )
    LOGGER.info("Saved CFF artifacts to %s", args.output_dir.resolve())
    LOGGER.info("Total runtime: %.2fs", elapsed)


if __name__ == "__main__":
    main()
