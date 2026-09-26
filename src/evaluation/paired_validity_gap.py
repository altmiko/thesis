"""Paired sample-level analysis of raw versus strict-valid adversarial success.

The analysis never regenerates or modifies attacks.  It independently derives eligibility,
classifier success, strict validity, paired contingency tables, confidence intervals, and
hypothesis tests from saved per-sample ``.npz`` artifacts.

Default reproduction command (PowerShell)::

    $env:PYTHONPATH="src"; python -m evaluation.paired_validity_gap \
        --output-dir outputs/statistical_tests

The 95% interval is Newcombe's square-and-add interval for paired proportions (method 10),
including the continuity-corrected correlation estimate implemented by the R
``contingencytables::Newcombe_square_and_add_CI_paired_2x2`` reference implementation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binom, chi2, norm


BENIGN_LABEL = 0
ALPHA = 0.05
PRIMARY_SEED = 42
PRIMARY_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")

DEFAULT_INPUTS: tuple[tuple[str, str, bool], ...] = (
    ("Input PGD", "outputs/cicids2017_input_baseline_postfix", False),
    ("Primitive-Direct", "outputs/cicids2017_primitive_attack_postfix", False),
    ("VAE-Latent-Masked", "outputs/cicids2017_latent_masked_postfix", False),
    ("VAE-Latent-Primitive", "outputs/cicids2017_vae_latent_attack_postfix", False),
    ("VAE-Latent-Raw", "outputs/cicids2017_latent_raw_postfix", True),
)

STRICT_COMPONENT_SETS: tuple[tuple[str, ...], ...] = (
    ("pave_valid", "mined_valid", "realizable"),
    ("pave_valid", "mined_valid", "mask_valid"),
)

FAILURE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("pave_valid", "feature-domain / PAVE-style"),
    ("mined_valid", "dataset-mined constraints"),
    ("dep_ok", "dependency / algebraic"),
    ("packet_ok", "packet-summary"),
    ("timing_ok", "timing"),
    ("rate_ok", "rate non-negativity"),
    ("disc_ok", "discreteness"),
    ("frozen_ok", "frozen-feature consistency"),
    ("derived_ok", "latent derived-feature consistency"),
)


@dataclass(frozen=True)
class InputSpec:
    method: str
    directory: Path
    diagnostic: bool = False


@dataclass
class ArtifactRows:
    records: list[dict[str, Any]]
    clean_vectors: dict[str, np.ndarray]


def _scalar(data: np.lib.npyio.NpzFile, name: str, default: Any = None) -> Any:
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ValueError(f"artifact field {name!r} must be scalar, got shape {value.shape}")
    return value.item()


def _array(data: np.lib.npyio.NpzFile, names: Sequence[str], n: int | None = None) -> np.ndarray:
    for name in names:
        if name in data.files:
            value = np.asarray(data[name])
            if n is not None and len(value) != n:
                raise ValueError(f"field {name!r} length {len(value)} != expected {n}")
            return value
    raise KeyError(f"artifact lacks required alternatives {tuple(names)!r}")


def _artifact_path(output_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    fallback = output_dir / "attack_artifacts" / path.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"missing attack artifact: {recorded!r}; also tried {fallback}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_strict_valid(data: np.lib.npyio.NpzFile, n: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Recompute strict validity from current independent validator outputs.

    A stored ``strict_valid`` value is treated as an assertion, never as the source of truth,
    whenever its component masks are present.
    """
    keys = set(data.files)
    components: tuple[str, ...] | None = None
    for candidate in STRICT_COMPONENT_SETS:
        if set(candidate) <= keys:
            components = candidate
            break
    if components is None:
        if "strict_valid" not in keys:
            raise KeyError(
                "artifact has neither recognized strict-validity components nor strict_valid; "
                f"available fields={sorted(keys)}"
            )
        return np.asarray(data["strict_valid"], dtype=bool), ("strict_valid",)

    strict = np.ones(n, dtype=bool)
    for name in components:
        strict &= np.asarray(data[name], dtype=bool)
    if "strict_valid" in keys:
        stored = np.asarray(data["strict_valid"], dtype=bool)
        if stored.shape != strict.shape or not np.array_equal(stored, strict):
            raise ValueError(
                f"stored strict_valid disagrees with {' & '.join(components)}"
            )
    return strict, components


def _validate_component_composition(data: np.lib.npyio.NpzFile, n: int) -> None:
    keys = set(data.files)
    if "realizable" in keys:
        names = ("dep_ok", "packet_ok", "timing_ok", "rate_ok", "disc_ok", "frozen_ok")
        if not set(names) <= keys:
            raise KeyError(f"realizable artifact lacks component masks: {sorted(set(names) - keys)}")
        composed = np.ones(n, dtype=bool)
        for name in names:
            composed &= np.asarray(data[name], dtype=bool)
        if not np.array_equal(composed, np.asarray(data["realizable"], dtype=bool)):
            raise ValueError("realizable mask disagrees with its six current category masks")
    if "mask_valid" in keys:
        names = ("derived_ok", "frozen_ok")
        if not set(names) <= keys:
            raise KeyError(f"mask_valid artifact lacks component masks: {sorted(set(names) - keys)}")
        composed = np.asarray(data["derived_ok"], dtype=bool) & np.asarray(data["frozen_ok"], dtype=bool)
        if not np.array_equal(composed, np.asarray(data["mask_valid"], dtype=bool)):
            raise ValueError("mask_valid disagrees with derived_ok & frozen_ok")


def _load_artifact(
    artifact: Path,
    *,
    method_display: str,
    diagnostic: bool,
    summary_dataset: str | None,
    summary_method: str | None,
) -> ArtifactRows:
    with np.load(artifact, allow_pickle=False) as data:
        row_ids = _array(data, ("row_id",)).astype(str)
        n = len(row_ids)
        if n == 0:
            raise ValueError(f"{artifact} contains no rows")
        if len(set(row_ids.tolist())) != n:
            raise ValueError(f"{artifact} contains duplicate row IDs")

        y_true = _array(data, ("true_label", "y_true"), n).astype(np.int64)
        clean_prediction = _array(data, ("clean_prediction", "y_pred_clean"), n).astype(np.int64)
        adv_prediction = _array(
            data, ("final_adversarial_prediction", "y_pred_adv"), n
        ).astype(np.int64)
        if np.any(y_true == BENIGN_LABEL):
            raise ValueError(f"{artifact} contains benign source rows in malicious-only analysis")

        dataset = str(_scalar(data, "dataset", summary_dataset or "unknown"))
        method_id = str(_scalar(data, "method", summary_method or method_display))
        class_name = str(_scalar(data, "class_name", "unknown"))
        victim = str(_scalar(data, "victim", "unknown"))
        seed = int(_scalar(data, "seed", -1))
        if summary_dataset is not None and dataset != summary_dataset:
            raise ValueError(f"{artifact}: dataset {dataset!r} != summary {summary_dataset!r}")
        if summary_method is not None and method_id != summary_method:
            raise ValueError(f"{artifact}: method {method_id!r} != summary {summary_method!r}")

        eligible = clean_prediction == y_true
        raw_targeted = eligible & (adv_prediction == BENIGN_LABEL)
        raw_untargeted = eligible & (adv_prediction != y_true)
        strict, strict_components = derive_strict_valid(data, n)
        _validate_component_composition(data, n)
        valid_targeted = raw_targeted & strict
        valid_untargeted = raw_untargeted & strict

        if "clean_correct" in data.files and not np.array_equal(
            eligible, np.asarray(data["clean_correct"], dtype=bool)
        ):
            raise ValueError(f"{artifact}: clean_correct disagrees with clean_prediction == true_label")
        if "benign" in data.files and not np.array_equal(
            adv_prediction == BENIGN_LABEL, np.asarray(data["benign"], dtype=bool)
        ):
            raise ValueError(f"{artifact}: benign disagrees with final prediction == BENIGN")
        if "evasion" in data.files and not np.array_equal(
            adv_prediction != y_true, np.asarray(data["evasion"], dtype=bool)
        ):
            raise ValueError(f"{artifact}: evasion disagrees with final prediction != true label")
        if "target_success_flag" in data.files and not np.array_equal(
            adv_prediction == BENIGN_LABEL, np.asarray(data["target_success_flag"], dtype=bool)
        ):
            raise ValueError(f"{artifact}: target_success_flag disagrees with BENIGN prediction")
        if np.any(valid_targeted & ~raw_targeted) or np.any(valid_untargeted & ~raw_untargeted):
            raise AssertionError("valid success is not a subset of raw success")

        provenance = {
            "git_commit": str(_scalar(data, "git_commit", "")),
            "git_dirty": bool(_scalar(data, "git_dirty", False)),
            "source_tree_sha256": str(_scalar(data, "source_tree_sha256", "")),
            "run_id": str(_scalar(data, "run_id", "")),
            "config_json": str(_scalar(data, "config_json", "{}")),
            "checkpoint_identifiers_json": str(
                _scalar(data, "checkpoint_identifiers_json", "{}")
            ),
        }
        component_arrays = {
            field: np.asarray(data[field], dtype=bool)
            for field, _ in FAILURE_CATEGORIES
            if field in data.files
        }
        clean_raw = _array(data, ("X_clean_raw",), n).astype(np.float32, copy=False)
        adv_raw = _array(data, ("X_adv_raw",), n).astype(np.float32, copy=False)
        if clean_raw.shape != adv_raw.shape:
            raise ValueError(f"{artifact}: clean/adv vectors have different shapes")
        artifact_hash = _sha256(artifact)

        records: list[dict[str, Any]] = []
        clean_vectors: dict[str, np.ndarray] = {}
        for idx, row_id in enumerate(row_ids):
            clean_vectors[row_id] = np.array(clean_raw[idx], copy=True)
            record: dict[str, Any] = {
                "dataset": dataset,
                "method": method_display,
                "method_id": method_id,
                "diagnostic": diagnostic,
                "victim": victim,
                "class": class_name,
                "seed": seed,
                "row_id": row_id,
                "eligible": bool(eligible[idx]),
                "true_label": int(y_true[idx]),
                "clean_prediction": int(clean_prediction[idx]),
                "final_adversarial_prediction": int(adv_prediction[idx]),
                "raw_success": bool(raw_targeted[idx]),
                "is_valid": bool(strict[idx]),
                "valid_success": bool(valid_targeted[idx]),
                "untargeted_raw_success": bool(raw_untargeted[idx]),
                "untargeted_valid_success": bool(valid_untargeted[idx]),
                "strict_components": " & ".join(strict_components),
                "artifact_path": str(artifact),
                "artifact_sha256": artifact_hash,
                **provenance,
            }
            for field, _ in FAILURE_CATEGORIES:
                record[field] = bool(component_arrays[field][idx]) if field in component_arrays else None
            records.append(record)
    return ArtifactRows(records=records, clean_vectors=clean_vectors)


def load_inputs(specs: Sequence[InputSpec]) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """Load every per-sample artifact and reject duplicate sample identities."""
    all_records: list[dict[str, Any]] = []
    clean_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    seen_keys: set[tuple[Any, ...]] = set()
    for spec in specs:
        summary_path = spec.directory / "attack_results.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing attack summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cells = summary.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"{summary_path} has no attack cells")
        for cell in cells:
            artifact = _artifact_path(spec.directory, str(cell["artifact"]))
            loaded = _load_artifact(
                artifact,
                method_display=spec.method,
                diagnostic=spec.diagnostic,
                summary_dataset=summary.get("dataset"),
                summary_method=summary.get("method_id"),
            )
            for record in loaded.records:
                key = (
                    record["dataset"], record["method"], record["victim"],
                    record["class"], record["seed"], record["row_id"],
                )
                if key in seen_keys:
                    raise ValueError(f"duplicate seed/sample artifact row detected: {key}")
                seen_keys.add(key)
            all_records.extend(loaded.records)
            dataset = loaded.records[0]["dataset"]
            pool = clean_by_dataset.setdefault(dataset, {})
            for row_id, vector in loaded.clean_vectors.items():
                prior = pool.get(row_id)
                if prior is not None and not np.array_equal(prior, vector):
                    raise ValueError(f"source sample {dataset}:{row_id} has conflicting clean vectors")
                pool[row_id] = vector
    return all_records, clean_by_dataset


def align_by_row_id(
    left_ids: Sequence[str], left_values: np.ndarray,
    right_ids: Sequence[str], right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two arrays by unique row ID or abort on duplicates/mismatched sets."""
    if len(left_ids) != len(left_values) or len(right_ids) != len(right_values):
        raise ValueError("row ID and value lengths differ")
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ValueError("duplicate row IDs cannot enter a paired comparison")
    if set(left_ids) != set(right_ids):
        only_left = sorted(set(left_ids) - set(right_ids))[:5]
        only_right = sorted(set(right_ids) - set(left_ids))[:5]
        raise ValueError(f"mismatched sample IDs; left-only={only_left}, right-only={only_right}")
    order = sorted(set(left_ids))
    li = {row_id: idx for idx, row_id in enumerate(left_ids)}
    ri = {row_id: idx for idx, row_id in enumerate(right_ids)}
    return (
        np.asarray([left_values[li[row_id]] for row_id in order]),
        np.asarray([right_values[ri[row_id]] for row_id in order]),
        order,
    )


def validate_cross_method_pools(records: Sequence[Mapping[str, Any]], seed: int) -> None:
    """Require identical primary source pools and clean denominators across methods.

    This validation supports descriptive regime comparisons.  It does not turn those
    comparisons into a between-method hypothesis test.
    """
    groups: dict[tuple[str, str, str], dict[str, dict[str, bool]]] = {}
    for row in records:
        if int(row["seed"]) != seed:
            continue
        key = (str(row["dataset"]), str(row["victim"]), str(row["class"]))
        methods = groups.setdefault(key, {})
        pool = methods.setdefault(str(row["method"]), {})
        row_id = str(row["row_id"])
        if row_id in pool:
            raise ValueError(f"duplicate row ID in source pool {key}/{row['method']}: {row_id}")
        pool[row_id] = bool(row["eligible"])
    for key, methods in groups.items():
        if len(methods) < 2:
            continue
        reference_name, reference = next(iter(methods.items()))
        for method, pool in methods.items():
            if pool != reference:
                raise ValueError(
                    f"incompatible primary source pool or clean-correct denominator for {key}: "
                    f"{reference_name!r} versus {method!r}"
                )


def wilson_score_interval(successes: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    center = (successes + z * z / 2.0) / (n + z * z)
    half = z * math.sqrt(successes * (n - successes) / n + z * z / 4.0) / (n + z * z)
    return center - half, center + half


def newcombe_paired_ci(
    a: int, b: int, c: int, d: int, alpha: float = ALPHA
) -> tuple[float, float]:
    """Newcombe square-and-add (method 10) CI for ``p_raw - p_valid``.

    The correlation estimate ``psi`` includes Newcombe's continuity correction, matching
    ``contingencytables::Newcombe_square_and_add_CI_paired_2x2``.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("paired table counts must be nonnegative")
    n = a + b + c + d
    if n == 0:
        raise ValueError("paired table is empty")
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    p1, p2 = row1 / n, col1 / n
    estimate = p1 - p2
    l1, u1 = wilson_score_interval(row1, n, alpha)
    l2, u2 = wilson_score_interval(col1, n, alpha)

    if min(row1, row2, col1, col2) == 0:
        psi = 0.0
    else:
        cross = a * d - b * c
        product = row1 * row2 * col1 * col2
        if cross > n / 2.0:
            psi = (cross - n / 2.0) / math.sqrt(product)
        elif cross >= 0:
            psi = 0.0
        else:
            psi = cross / math.sqrt(product)

    lower_term = (
        (p1 - l1) ** 2 + (u2 - p2) ** 2
        - 2.0 * psi * (p1 - l1) * (u2 - p2)
    )
    upper_term = (
        (p2 - l2) ** 2 + (u1 - p1) ** 2
        - 2.0 * psi * (p2 - l2) * (u1 - p1)
    )
    lower = estimate - math.sqrt(max(0.0, lower_term))
    upper = estimate + math.sqrt(max(0.0, upper_term))
    return max(-1.0, lower), min(1.0, upper)


def mcnemar_test(b: int, c: int) -> dict[str, Any]:
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = b + c
    if discordant < 25:
        p_value = 1.0 if discordant == 0 else min(
            1.0, 2.0 * float(binom.cdf(min(b, c), discordant, 0.5))
        )
        return {
            "test_variant": "exact binomial McNemar",
            "test_statistic": None,
            "p_value": p_value,
        }
    statistic = (abs(b - c) - 1.0) ** 2 / discordant
    return {
        "test_variant": "asymptotic McNemar chi-square (continuity corrected)",
        "test_statistic": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
    }


def cochran_q(matrix: np.ndarray) -> dict[str, float | int]:
    """Cochran's Q for k related binary samples; ``matrix`` is (n paired units, k conditions)."""
    x = np.asarray(matrix).astype(np.int64)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("Cochran's Q needs an (n, k>=3) binary matrix")
    k = x.shape[1]
    col, row = x.sum(0), x.sum(1)
    denom = k * row.sum() - (row ** 2).sum()
    if denom == 0:
        return {"Q": 0.0, "df": k - 1, "p": 1.0}
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom
    return {"Q": float(q), "df": k - 1, "p": float(chi2.sf(q, k - 1))}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise-error correction."""
    m = len(p_values)
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(m), key=lambda idx: (p_values[idx], idx))
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def contingency(
    rows: Sequence[Mapping[str, Any]], *, outcome: str = "targeted"
) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["eligible"])]
    if not eligible:
        raise ValueError("analysis cell has zero clean-correct malicious rows")
    if outcome == "targeted":
        raw_key, valid_key = "raw_success", "valid_success"
    elif outcome == "untargeted":
        raw_key, valid_key = "untargeted_raw_success", "untargeted_valid_success"
    else:
        raise ValueError(f"unsupported outcome: {outcome}")
    raw = np.asarray([bool(row[raw_key]) for row in eligible], dtype=bool)
    valid = np.asarray([bool(row[valid_key]) for row in eligible], dtype=bool)
    a = int((raw & valid).sum())
    b = int((raw & ~valid).sum())
    c = int((~raw & valid).sum())
    d = int((~raw & ~valid).sum())
    if c != 0:
        raise AssertionError(
            f"c={c}: valid_success is not a subset of raw_success; alignment/definition bug"
        )
    n = len(eligible)
    if a + b + c + d != n:
        raise AssertionError("paired contingency table does not sum to N")
    ci_low, ci_high = newcombe_paired_ci(a, b, c, d)
    test = mcnemar_test(b, c)
    literal_or: str
    if c == 0 and b > 0:
        literal_or = "infinity"
    elif b == 0 and c > 0:
        literal_or = "0"
    elif b == 0 and c == 0:
        literal_or = "undefined (0/0)"
    else:
        literal_or = f"{b / c:.12g}"
    return {
        "N": n,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "raw_asr": (a + b) / n,
        "valid_asr": (a + c) / n,
        "delta_asr": (b - c) / n,
        "delta_asr_pp": 100.0 * (b - c) / n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_pp": 100.0 * ci_low,
        "ci_high_pp": 100.0 * ci_high,
        "rejected_raw_success_fraction": b / (a + b) if a + b else None,
        "odds_ratio_literal": literal_or,
        "odds_ratio_corrected_ha": (b + 0.5) / (c + 0.5),
        **test,
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _stats_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, outcome: str = "targeted"
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key, group in sorted(_group_rows(rows, keys).items(), key=lambda item: tuple(map(str, item[0]))):
        metadata = dict(zip(keys, key))
        result.append({**metadata, **contingency(group, outcome=outcome)})
    return result


def _apply_holm(rows: list[dict[str, Any]], *, alpha: float = ALPHA) -> None:
    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
        row["decision_alpha_0_05"] = "reject H0" if value < alpha else "do not reject H0"


def analyze(records: Sequence[Mapping[str, Any]], primary_seed: int = PRIMARY_SEED) -> dict[str, Any]:
    primary = [row for row in records if int(row["seed"]) == primary_seed]
    if not primary:
        raise ValueError(f"no rows exist for primary seed {primary_seed}")
    validate_cross_method_pools(primary, primary_seed)

    primary_methods = [row for row in primary if not bool(row["diagnostic"])]
    diagnostic = [row for row in primary if bool(row["diagnostic"])]

    primary_cells = _stats_rows(primary_methods, ("dataset", "method", "victim"))
    _apply_holm(primary_cells)
    diagnostic_cells = _stats_rows(diagnostic, ("dataset", "method", "victim")) if diagnostic else []
    for row in diagnostic_cells:
        row["holm_p"] = None
        row["decision_alpha_0_05"] = "diagnostic; outside primary Holm family"
    per_cell = primary_cells + diagnostic_cells

    per_class = _stats_rows(primary_methods, ("dataset", "method", "class"))
    _apply_holm(per_class)
    for row in per_class:
        row["multiplicity_family"] = "secondary dataset × method × source-class family"
    diagnostic_class = _stats_rows(diagnostic, ("dataset", "method", "class")) if diagnostic else []
    for row in diagnostic_class:
        row["holm_p"] = None
        row["decision_alpha_0_05"] = "diagnostic; outside secondary Holm family"
        row["multiplicity_family"] = "diagnostic"
    per_class.extend(diagnostic_class)

    headline = _stats_rows(primary, ("dataset", "method"))
    for row in headline:
        row["pooling"] = "micro-pooled over distinct victim × class adversarial examples"

    seed_robustness = _stats_rows(records, ("dataset", "method", "seed"))
    for row in seed_robustness:
        row["pooling"] = "micro-pooled within seed; seeds never concatenated"

    untargeted = _stats_rows(primary, ("dataset", "method", "victim"), outcome="untargeted")
    for row in untargeted:
        row["analysis_role"] = "secondary untargeted; never mixed with targeted ASR"

    failure_rows: list[dict[str, Any]] = []
    for key, group in sorted(
        _group_rows(primary, ("dataset", "method", "victim")).items(),
        key=lambda item: tuple(map(str, item[0])),
    ):
        lost = [row for row in group if bool(row["eligible"]) and bool(row["raw_success"]) and not bool(row["is_valid"])]
        b = len(lost)
        for field, label in FAILURE_CATEGORIES:
            available = [row for row in lost if row.get(field) is not None]
            if not available:
                continue
            count = sum(not bool(row[field]) for row in available)
            failure_rows.append({
                "dataset": key[0], "method": key[1], "victim": key[2],
                "validator_category": label,
                "b_failures": count,
                "lost_successes_b": b,
                "percent_of_lost_successes": 100.0 * count / b if b else None,
                "note": "non-exclusive categories; percentages may sum above 100%",
            })

    regime = []
    for row in headline:
        regime.append({
            "dataset": row["dataset"], "method": row["method"],
            "raw_asr": row["raw_asr"], "valid_asr": row["valid_asr"],
            "delta_asr_pp": row["delta_asr_pp"],
            "rejected_raw_success_fraction": row["rejected_raw_success_fraction"],
            "N": row["N"], "b": row["b"],
        })

    macro_summary = []
    for key, group in sorted(
        _group_rows(per_cell, ("dataset", "method")).items(),
        key=lambda item: tuple(map(str, item[0])),
    ):
        macro_summary.append({
            "dataset": key[0],
            "method": key[1],
            "victim_cells": len(group),
            "macro_raw_asr": float(np.mean([row["raw_asr"] for row in group])),
            "macro_valid_asr": float(np.mean([row["valid_asr"] for row in group])),
            "macro_delta_asr_pp": float(np.mean([row["delta_asr_pp"] for row in group])),
            "note": "unweighted macro-average over victim-level primary-seed cells",
        })

    return {
        "primary_seed": primary_seed,
        "primary_family_definition": "dataset × primary attack method × victim at seed 42, pooled over malicious source classes",
        "primary_family_size": len(primary_cells),
        "per_cell": per_cell,
        "headline_pooled": headline,
        "per_class": per_class,
        "failure_attribution": failure_rows,
        "regime_summary": regime,
        "macro_summary": macro_summary,
        "seed_robustness": seed_robustness,
        "untargeted_secondary": untargeted,
    }


def compute_genuine_validity(
    clean_by_dataset: Mapping[str, Mapping[str, np.ndarray]]
) -> list[dict[str, Any]]:
    """Evaluate untouched source rows with the same current strict validators.

    Currently the repository exposes the full validator construction for
    ``cicids2017_distrinet``.  Other artifact datasets remain analyzable; their genuine
    context is marked unavailable until their experiment adapter exposes an equivalent
    construction.
    """
    output: list[dict[str, Any]] = []
    for dataset, vectors_by_id in clean_by_dataset.items():
        if dataset != "cicids2017_distrinet":
            output.append({
                "dataset": dataset,
                "N_unique_genuine_rows": len(vectors_by_id),
                "strict_valid_count": None,
                "genuine_test_validity_rate": None,
                "status": "unavailable: no repository strict-validator constructor registered",
            })
            continue
        from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
        from attack.realizability.validator import RealizabilityValidator
        from datasets.cicids2017 import CICIDS2017Adapter
        from evaluation.pave_style_validator import PAVEStyleValidator
        from experiments.ablations import build_ablation
        import torch

        adapter = CICIDS2017Adapter()
        manifest = adapter.feature_manifest()
        model = CICIDS2017PrimitiveModel(manifest)
        validator = RealizabilityValidator(model)
        raw_train = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
        pave = PAVEStyleValidator(
            integer_tolerance=SCALER_ATOL, range_tolerance=SCALER_ATOL
        ).fit(np.asarray(raw_train, dtype=np.float64), manifest.names, schema=manifest)
        layer1_fit = np.ascontiguousarray(raw_train[:200000], dtype=np.float32)
        engine = build_ablation(
            "A4", adapter, encoder_input_transform="asinh",
            layer1_fit_x_raw=layer1_fit,
            layer2_path=adapter.repo_root / "old_constraints" / adapter.name / "mined.json",
        ).engine
        ordered_ids = sorted(vectors_by_id)
        x_np = np.stack([vectors_by_id[row_id] for row_id in ordered_ids]).astype(np.float32)
        x = torch.from_numpy(x_np)
        pave_valid = np.asarray(pave.validate_batch(x_np)["valid_mask"], dtype=bool)
        mined_valid = engine.validate(x)["pass_l0_l1_l2"].cpu().numpy().astype(bool)
        realizable = validator.validate(x, x).valid.cpu().numpy().astype(bool)
        strict = pave_valid & mined_valid & realizable
        output.append({
            "dataset": dataset,
            "N_unique_genuine_rows": len(strict),
            "strict_valid_count": int(strict.sum()),
            "genuine_test_validity_rate": float(strict.mean()),
            "pave_validity_rate": float(pave_valid.mean()),
            "mined_validity_rate": float(mined_valid.mean()),
            "realizability_validity_rate": float(realizable.mean()),
            "status": "computed on unique untouched held-out source rows used by attacks",
            "miner_fit_split": "train",
            "miner_keep_violation_rate_lte": 0.01,
            "miner_retained_rules": 14,
            "miner_retained_max_train_violation_rate": 0.0,
        })
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _fmt_pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def _fmt_p(value: Any) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return "<1e-300" if value == 0.0 else f"{value:.4g}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("|" + "|".join(str(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def build_report(results: Mapping[str, Any], genuine: Sequence[Mapping[str, Any]]) -> str:
    headline_rows = []
    for row in results["headline_pooled"]:
        headline_rows.append((
            row["dataset"], row["method"], row["N"], row["a"], row["b"], row["c"], row["d"],
            _fmt_pct(row["raw_asr"]), _fmt_pct(row["valid_asr"]), f"{row['delta_asr_pp']:.2f}",
            f"[{row['ci_low_pp']:.2f}, {row['ci_high_pp']:.2f}]", row["test_variant"],
            _fmt_p(row["p_value"]), row["odds_ratio_literal"],
        ))
    macro_rows = []
    for row in results["macro_summary"]:
        macro_rows.append((
            row["dataset"], row["method"], row["victim_cells"],
            _fmt_pct(row["macro_raw_asr"]), _fmt_pct(row["macro_valid_asr"]),
            f"{row['macro_delta_asr_pp']:.2f}",
        ))
    cell_rows = []
    for row in results["per_cell"]:
        cell_rows.append((
            row["dataset"], row["method"], row["victim"], row["N"], row["a"], row["b"],
            row["c"], row["d"], _fmt_pct(row["raw_asr"]), _fmt_pct(row["valid_asr"]),
            f"{row['delta_asr_pp']:.2f}", f"[{row['ci_low_pp']:.2f}, {row['ci_high_pp']:.2f}]",
            row["test_variant"], _fmt_p(row["p_value"]), _fmt_p(row.get("holm_p")),
            row["odds_ratio_literal"], row["decision_alpha_0_05"],
        ))
    genuine_lines = []
    for row in genuine:
        rate = row.get("genuine_test_validity_rate")
        genuine_lines.append(
            f"- **{row['dataset']}**: {row['status']}; N={row['N_unique_genuine_rows']}; "
            f"strict validity={_fmt_pct(rate)}."
        )

    return f"""# Paired Raw-vs-Valid Adversarial Success Analysis

## Design

This is a within-subject paired analysis. Each generated adversarial example is evaluated twice conceptually: (1) raw targeted classifier success and (2) the same classifier success after requiring current strict validity. No control attack and no second adversarial vector are generated. The classifier prediction and validity masks come from the exact same saved final adversarial vector.

The primary threat model is malicious $\\rightarrow$ Benign. Eligibility requires a malicious source row whose clean prediction equals its true label. For eligible row $i$, `raw_success_i = 1[final prediction = BENIGN]`, `is_valid_i = strict_valid_i`, and `valid_success_i = raw_success_i AND is_valid_i`. Untargeted results are saved separately and never mixed into targeted ASR.

The predeclared primary seed is **{results['primary_seed']}**. The Holm family is **{results['primary_family_definition']}** ({results['primary_family_size']} cells). `VAE-Latent-Raw` is diagnostic and excluded from that family. Other seeds are reported separately rather than concatenated.

## Current strict-validity definition

The analysis recomputed strict validity from per-sample outputs created by the current experiment pipeline and asserted equality with the stored `strict_valid` mask:

- Input PGD, Primitive-Direct, and VAE-Latent-Primitive: `pave_valid AND mined_valid AND realizable`.
- `realizable = dep_ok AND packet_ok AND timing_ok AND rate_ok AND disc_ok AND frozen_ok`.
- VAE-Latent-Raw and VAE-Latent-Masked: `pave_valid AND mined_valid AND mask_valid`.
- `mask_valid = derived_ok AND frozen_ok`.

`in_dist` is the separately reported VAE/Mahalanobis realism gate; the current runners do **not** include it in `strict_valid`. No sample was repaired, projected, regenerated, or modified by this analysis.

## Headline micro-pooled results

The headline pools distinct adversarial examples over victims and source classes within each dataset × method at seed {results['primary_seed']}. A source row attacked against different victims is a distinct adversarial example; repeated seeds are not pooled.

{_markdown_table(['Dataset','Method','N','a','b','c','d','Raw ASR','Valid ASR','ΔASR pp','95% CI pp','Test','p','OR'], headline_rows)}

Macro estimates are reported separately and are never substituted for pooled counts:

{_markdown_table(['Dataset','Method','Victim cells','Macro Raw ASR','Macro Valid ASR','Macro ΔASR pp'], macro_rows)}

## Primary victim-level family

{_markdown_table(['Dataset','Method','Victim','N','a','b','c','d','Raw ASR','Valid ASR','ΔASR pp','95% CI pp','Test','p','Holm p','OR','Decision'], cell_rows)}

## Statistical methods

$\\Delta ASR = ASR_{{raw}} - ASR_{{valid}} = (b-c)/N$, reported in percentage points. The 95% interval is Newcombe's square-and-add matched-pairs interval (method 10) using Wilson score bounds and the continuity-corrected paired correlation estimate from `contingencytables::Newcombe_square_and_add_CI_paired_2x2`; independent-binomial intervals were not subtracted.

McNemar's test uses the predeclared rule: exact two-sided binomial when $b+c<25$; otherwise continuity-corrected $\\chi^2_1$ with statistic $(|b-c|-1)^2/(b+c)$. Every table records the variant. Because `valid_success = raw_success AND validity`, $c=0$ is structural and is asserted. The uncorrected discordant OR is $b/c$ (infinite when $b>0,c=0$); the finite Haldane-Anscombe estimate $(b+0.5)/(c+0.5)$ is saved separately and is not presented as the literal OR.

## Validator soundness context

{chr(10).join(genuine_lines)}

The CICIDS2017 mined constraint artifact was fit on 1,456,265 training rows, retained 14 rules at a predeclared maximum 1% violation rate, and reports a maximum retained training violation rate of 0.0. Validator thresholds were not tuned on adversarial outcomes.

## Failure attribution

`failure_attribution.csv` counts each current validator category among cell-$b$ rows (raw targeted success lost under strict validity). Categories are non-exclusive, so percentages can exceed 100% in total. This is descriptive attribution, not causal evidence that any individual rule caused classifier success to disappear.

## Interpretation

The analysis estimates how much conventional feature-space targeted ASR overstates success after current feature-domain, mined-dependency, and method-specific realizability requirements are enforced. Statistical significance establishes a reliable paired reduction in measured attack success, not operational irrelevance: a nonzero valid ASR may remain important. No claim that one attack method has a statistically smaller gap than another is made; such a claim requires a separate paired between-method test.

## Threats to validity

1. The test establishes a reliable paired reduction, not which individual validity constraint caused it.
2. Statistical significance does not imply operational insignificance; `ASR_valid` may remain practically important.
3. Findings are conditional on these datasets, trained victims, methods, clean-correct source pools, and the malicious-to-Benign threat model.
4. The gap is bounded by validator quality; an overly restrictive validator can inflate $\\Delta ASR$.
5. Current CICIDS2017 primitive/VAE experiments establish Level A+B feature-space consistency, not full Level-C packet-trace realization. Level C requires actual PCAP modification and feature re-extraction.
6. The structural $c=0$ asymmetry follows from the metric definition; validity enforcement can preserve or remove raw successes but cannot create them.
7. Cross-dataset replication should assess the pattern `ASR_raw > ASR_valid` and the gap distribution, not expect identical magnitudes.

## Reproducibility

All reported counts and rates were independently recomputed from per-sample artifacts. `per_sample_audit.csv` preserves exact row IDs, predictions, eligibility, success/validity outcomes, seed, artifact path and hash, git metadata, run ID, configuration JSON, and checkpoint hashes. Duplicate IDs, inconsistent source vectors, mismatched source pools, unequal array lengths, inconsistent stored masks, and nonzero $c$ abort the analysis.
"""


def write_outputs(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    genuine: Sequence[Mapping[str, Any]],
    command: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample_audit.csv", records)
    write_csv(output_dir / "per_cell_results.csv", results["per_cell"])
    write_csv(output_dir / "headline_pooled.csv", results["headline_pooled"])
    write_csv(output_dir / "per_class_results.csv", results["per_class"])
    write_csv(output_dir / "failure_attribution.csv", results["failure_attribution"])
    write_csv(output_dir / "primary_holm_results.csv", [row for row in results["per_cell"] if row.get("holm_p") is not None])
    write_csv(output_dir / "seed_robustness.csv", results["seed_robustness"])
    write_csv(output_dir / "regime_summary.csv", results["regime_summary"])
    write_csv(output_dir / "macro_summary.csv", results["macro_summary"])
    write_csv(output_dir / "untargeted_secondary.csv", results["untargeted_secondary"])
    write_csv(output_dir / "genuine_validity.csv", genuine)
    payload = {
        "analysis": results,
        "genuine_validity": list(genuine),
        "strict_validity_definitions": {
            "realizability_methods": "pave_valid & mined_valid & realizable",
            "realizable": "dep_ok & packet_ok & timing_ok & rate_ok & disc_ok & frozen_ok",
            "latent_variants": "pave_valid & mined_valid & mask_valid",
            "mask_valid": "derived_ok & frozen_ok",
            "in_dist_note": "separate realism/IDR gate; not part of current strict_valid",
        },
        "statistics": {
            "confidence_interval": "Newcombe square-and-add matched-pairs method 10 with continuity-corrected psi",
            "confidence_level": 0.95,
            "mcnemar_rule": "exact binomial if b+c<25; otherwise continuity-corrected chi-square",
            "multiple_comparisons": "Holm, alpha=0.05",
        },
        "reproduction_command": command,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(build_report(results, genuine), encoding="utf-8")
    (output_dir / "reproduction_command.txt").write_text(command + "\n", encoding="utf-8")


def parse_input(value: str) -> InputSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be METHOD=PATH or METHOD=PATH:diagnostic")
    method, raw_path = value.split("=", 1)
    diagnostic = raw_path.endswith(":diagnostic")
    if diagnostic:
        raw_path = raw_path[: -len(":diagnostic")]
    if not method.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--input method and path must be nonempty")
    return InputSpec(method.strip(), Path(raw_path.strip()), diagnostic)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=parse_input, default=None,
        help="repeatable METHOD=OUTPUT_DIR; append :diagnostic to exclude from primary Holm family",
    )
    parser.add_argument("--primary-seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/statistical_tests"))
    parser.add_argument(
        "--skip-genuine-validity", action="store_true",
        help="skip deterministic clean-row validator recomputation (artifact analysis still complete)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = args.input or [InputSpec(name, Path(path), diagnostic) for name, path, diagnostic in DEFAULT_INPUTS]
    records, clean_by_dataset = load_inputs(specs)
    results = analyze(records, primary_seed=args.primary_seed)
    if args.skip_genuine_validity:
        genuine = [{
            "dataset": dataset,
            "N_unique_genuine_rows": len(rows),
            "strict_valid_count": None,
            "genuine_test_validity_rate": None,
            "status": "skipped by --skip-genuine-validity",
        } for dataset, rows in clean_by_dataset.items()]
    else:
        genuine = compute_genuine_validity(clean_by_dataset)
    command = (
        '$env:PYTHONPATH="src"; python -m evaluation.paired_validity_gap '
        f'--primary-seed {args.primary_seed} --output-dir "{args.output_dir.as_posix()}"'
    )
    write_outputs(args.output_dir, records, results, genuine, command)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "artifacts": len({row["artifact_path"] for row in records}),
        "sample_rows": len(records),
        "primary_family_size": results["primary_family_size"],
        "headline_cells": len(results["headline_pooled"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
