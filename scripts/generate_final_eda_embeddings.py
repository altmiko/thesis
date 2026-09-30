#!/usr/bin/env python3
"""Generate before/after PCA and t-SNE EDA figures for both DistriNet datasets.

The pre-preprocessing view is sampled directly from the corrected raw CSVs. It
applies only label mapping and finite-row filtering required to construct an
embedding: no cleaning, deduplication, split, class-size reduction, or fitted
production transform is used. The post-preprocessing view samples the final
RobustScaler-transformed train/validation/test arrays.

Run from the repository root:
    python scripts/generate_final_eda_embeddings.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from preprocessing.preprocess_cicids2017_distrinet import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_ID,
    EXPECTED_FILES as CICIDS2017_FILES,
    SOURCE_TO_CATEGORY as CICIDS2017_SOURCE_TO_CATEGORY,
)
from preprocessing.preprocess_cicids2018_distrinet import (  # noqa: E402
    EXPECTED_FILES as CICIDS2018_FILES,
    SOURCE_TO_CATEGORY as CICIDS2018_SOURCE_TO_CATEGORY,
)
from evaluation.cicids2018_distrinet_eda import decode_label, read_header  # noqa: E402

SEED = 42
LABEL_COLUMN = "Label"
READ_BLOCK_SIZE = 2 << 20
CLASS_COLORS = {
    "Benign": "#2E8B57",
    "DoS": "#E67E22",
    "DDoS": "#C0392B",
    "Recon": "#2E86C1",
    "BruteForce": "#7D3C98",
}
DATASETS = {
    "cicids2017": {
        "display": "CICIDS2017-DistriNet",
        "raw_dir": REPO_ROOT / "data" / "raw" / "CICIDS_2017_Distrinet",
        "processed_dir": REPO_ROOT / "data" / "processed" / "CICIDS_2017_Distrinet",
        "files": tuple(CICIDS2017_FILES),
    },
    "cicids2018": {
        "display": "CSE-CIC-IDS-2018-DistriNet",
        "raw_dir": REPO_ROOT / "data" / "raw" / "CSECICIDS2018_Distrinet",
        "processed_dir": REPO_ROOT / "data" / "processed" / "CSECICIDS_2018_Distrinet",
        "files": tuple(CICIDS2018_FILES),
    },
}


@dataclass(frozen=True)
class RawSampleTask:
    dataset: str
    path: Path
    file_index: int
    raw_columns: tuple[str, ...]
    features: tuple[str, ...]
    sample_per_class: int
    seed: int


@dataclass
class Sample:
    priorities: dict[int, np.ndarray]
    features: dict[int, np.ndarray]
    mapped_counts: np.ndarray
    finite_counts: np.ndarray


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _priority(file_index: int, first_row: int, n: int, seed: int) -> np.ndarray:
    rows = np.arange(first_row, first_row + n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keys = rows ^ (np.uint64(file_index + 1) * np.uint64(0xD6E8FEB86659FD93))
        keys ^= np.uint64(seed) * np.uint64(0xA0761D6478BD642F)
        return _splitmix64(keys)


def _trim(
    old_priority: np.ndarray,
    old_x: np.ndarray,
    new_priority: np.ndarray,
    new_x: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    priority = np.concatenate((old_priority, new_priority))
    values = np.concatenate((old_x, new_x), axis=0)
    if len(priority) > limit:
        keep = np.argpartition(priority, limit - 1)[:limit]
        priority, values = priority[keep], values[keep]
    order = np.argsort(priority, kind="stable")
    return priority[order], values[order]


def _normalise_label(raw: bytes | None) -> str:
    label, _ = decode_label(raw)
    return label.replace("\x96", "-").replace("\u2013", "-").replace("\u2014", "-").strip()


def _category_for_label(dataset: str, raw: bytes | None) -> int:
    label = _normalise_label(raw)
    if label.endswith(" - Attempted"):
        label = "BENIGN"
    mapping = (
        CICIDS2017_SOURCE_TO_CATEGORY
        if dataset == "cicids2017"
        else CICIDS2018_SOURCE_TO_CATEGORY
    )
    category = mapping.get(label)
    return CATEGORY_TO_ID[category] if category is not None else -1


def _category_codes(dataset: str, column: pa.Array) -> np.ndarray:
    encoded = column.dictionary_encode()
    dictionary = encoded.dictionary.to_pylist()
    lookup = np.asarray(
        [_category_for_label(dataset, value) for value in dictionary] + [-1], dtype=np.int8
    )
    indices = pc.fill_null(encoded.indices, len(dictionary)).to_numpy(zero_copy_only=False)
    return lookup[indices]


def _float_matrix(batch: pa.RecordBatch, features: tuple[str, ...]) -> np.ndarray:
    matrix = np.empty((batch.num_rows, len(features)), dtype=np.float64)
    for j, name in enumerate(features):
        matrix[:, j] = batch.column(name).to_numpy(zero_copy_only=False)
    return matrix


def _sample_raw_file(task: RawSampleTask) -> Sample:
    include = [LABEL_COLUMN, *task.features]
    read_options = pacsv.ReadOptions(
        block_size=READ_BLOCK_SIZE,
        column_names=list(task.raw_columns),
        skip_rows=1,
    )
    convert_options = pacsv.ConvertOptions(
        column_types={LABEL_COLUMN: pa.binary()}
        | {name: pa.float64() for name in task.features},
        include_columns=include,
        null_values=["", "NaN", "nan", "NULL", "null"],
        strings_can_be_null=True,
    )
    priorities = {code: np.empty(0, dtype=np.uint64) for code in range(len(CATEGORY_NAMES))}
    samples = {
        code: np.empty((0, len(task.features)), dtype=np.float32)
        for code in range(len(CATEGORY_NAMES))
    }
    mapped_counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    finite_counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    offset = 0
    with task.path.open("rb") as handle:
        reader = pacsv.open_csv(handle, read_options=read_options, convert_options=convert_options)
        for batch in reader:
            n = batch.num_rows
            category = _category_codes(task.dataset, batch.column(LABEL_COLUMN))
            for code in range(len(CATEGORY_NAMES)):
                mapped_counts[code] += np.count_nonzero(category == code)
            x = _float_matrix(batch, task.features)
            finite = np.isfinite(x).all(axis=1)
            batch_priority = _priority(task.file_index, offset + 1, n, task.seed)
            for code in range(len(CATEGORY_NAMES)):
                selected = np.flatnonzero((category == code) & finite)
                finite_counts[code] += len(selected)
                if not len(selected):
                    continue
                candidate_priority = batch_priority[selected]
                if len(selected) > task.sample_per_class:
                    local = np.argpartition(candidate_priority, task.sample_per_class - 1)[
                        : task.sample_per_class
                    ]
                    selected = selected[local]
                    candidate_priority = candidate_priority[local]
                priorities[code], samples[code] = _trim(
                    priorities[code],
                    samples[code],
                    candidate_priority,
                    np.ascontiguousarray(x[selected], dtype=np.float32),
                    task.sample_per_class,
                )
            offset += n
    return Sample(priorities, samples, mapped_counts, finite_counts)


def _merge_samples(parts: list[Sample], sample_per_class: int) -> Sample:
    priorities = {code: np.empty(0, dtype=np.uint64) for code in range(len(CATEGORY_NAMES))}
    features = {
        code: np.empty((0, 0), dtype=np.float32) for code in range(len(CATEGORY_NAMES))
    }
    mapped_counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    finite_counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    for part in parts:
        mapped_counts += part.mapped_counts
        finite_counts += part.finite_counts
        for code in range(len(CATEGORY_NAMES)):
            if not len(part.priorities[code]):
                continue
            if features[code].shape[1] == 0:
                features[code] = part.features[code]
                priorities[code] = part.priorities[code]
            else:
                priorities[code], features[code] = _trim(
                    priorities[code],
                    features[code],
                    part.priorities[code],
                    part.features[code],
                    sample_per_class,
                )
    return Sample(priorities, features, mapped_counts, finite_counts)


def sample_raw(dataset: str, sample_per_class: int, workers: int, seed: int) -> Sample:
    config = DATASETS[dataset]
    manifest = json.loads((config["processed_dir"] / "preprocessing_manifest.json").read_text())
    features = tuple(manifest["modelling_feature_names"])
    tasks: list[RawSampleTask] = []
    for file_index, name in enumerate(config["files"]):
        path = config["raw_dir"] / name
        tasks.append(
            RawSampleTask(
                dataset=dataset,
                path=path,
                file_index=file_index,
                raw_columns=tuple(read_header(path)),
                features=features,
                sample_per_class=sample_per_class,
                seed=seed,
            )
        )
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        parts = list(pool.map(_sample_raw_file, tasks))
    return _merge_samples(parts, sample_per_class)


def sample_processed(dataset: str, sample_per_class: int, seed: int) -> Sample:
    directory = DATASETS[dataset]["processed_dir"]
    manifest = json.loads((directory / "preprocessing_manifest.json").read_text())
    n_features = len(manifest["modelling_feature_names"])
    priorities = {code: np.empty(0, dtype=np.uint64) for code in range(len(CATEGORY_NAMES))}
    features = {
        code: np.empty((0, n_features), dtype=np.float32) for code in range(len(CATEGORY_NAMES))
    }
    counts = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    for split_index, split in enumerate(("train", "val", "test")):
        x = np.load(directory / f"X_{split}.npy", mmap_mode="r")
        y = np.load(directory / f"y_{split}_cat.npy", mmap_mode="r")
        split_priority = _priority(split_index, 1, len(y), seed)
        for code in range(len(CATEGORY_NAMES)):
            members = np.flatnonzero(y == code)
            counts[code] += len(members)
            candidate_priority = split_priority[members]
            if len(members) > sample_per_class:
                keep = np.argpartition(candidate_priority, sample_per_class - 1)[
                    : sample_per_class
                ]
                members, candidate_priority = members[keep], candidate_priority[keep]
            priorities[code], features[code] = _trim(
                priorities[code],
                features[code],
                candidate_priority,
                np.asarray(x[members], dtype=np.float32),
                sample_per_class,
            )
    return Sample(priorities, features, counts.copy(), counts.copy())


def _prepare(
    sample: Sample, stage: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    priorities: list[np.ndarray] = []
    for code in range(len(CATEGORY_NAMES)):
        matrices.append(sample.features[code])
        labels.append(np.full(len(sample.features[code]), code, dtype=np.int8))
        priorities.append(sample.priorities[code])
    x = np.concatenate(matrices).astype(np.float64)
    y = np.concatenate(labels)
    priority = np.concatenate(priorities)
    if stage == "before_preprocessing":
        x = np.sign(x) * np.log1p(np.abs(x))
    else:
        x = np.arcsinh(x)
    varying = np.std(x, axis=0) > 0
    if np.count_nonzero(varying) < 2:
        raise ValueError(f"{stage}: fewer than two varying features")
    x = StandardScaler().fit_transform(x[:, varying])
    if not np.isfinite(x).all():
        raise ValueError(f"{stage}: visualization matrix contains NaN/Inf")
    return x, y, priority, varying


def _save_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    for code, name in enumerate(CATEGORY_NAMES):
        selected = labels == code
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=7,
            alpha=0.48,
            color=CLASS_COLORS[name],
            edgecolors="none",
            rasterized=True,
            label=f"{name} (n={selected.sum():,})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(markerscale=2.3, fontsize=9, frameon=True)
    ax.grid(alpha=0.16, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def generate_embeddings(
    dataset: str,
    stage: str,
    sample: Sample,
    feature_names: list[str],
    output_dir: Path,
    tsne_per_class: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, labels, priority, varying = _prepare(sample, stage)
    pca = PCA(n_components=2, svd_solver="full")
    pca_coordinates = pca.fit_transform(x)
    stage_label = "Before preprocessing" if stage == "before_preprocessing" else "After preprocessing"
    display = DATASETS[dataset]["display"]
    _save_scatter(
        pca_coordinates,
        labels,
        f"{display}: PCA — {stage_label}",
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        output_dir / "pca.png",
    )
    pd.DataFrame(
        pca.components_.T,
        index=np.asarray(feature_names)[varying],
        columns=["PC1", "PC2"],
    ).rename_axis("feature").to_csv(output_dir / "pca_loadings.csv")

    tsne_indices: list[np.ndarray] = []
    for code in range(len(CATEGORY_NAMES)):
        members = np.flatnonzero(labels == code)
        order = np.argsort(priority[members], kind="stable")
        tsne_indices.append(members[order[: min(tsne_per_class, len(members))]])
    selected = np.sort(np.concatenate(tsne_indices))
    n_components = min(30, x.shape[1], len(selected) - 1)
    reduced = PCA(n_components=n_components, svd_solver="full").fit_transform(x[selected])
    kwargs = {
        "n_components": 2,
        "perplexity": min(40.0, max(5.0, (len(selected) - 1) / 3.0)),
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "n_jobs": -1,
    }
    try:
        tsne = TSNE(max_iter=1000, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=1000, **kwargs)
    tsne_coordinates = tsne.fit_transform(reduced)
    _save_scatter(
        tsne_coordinates,
        labels[selected],
        f"{display}: t-SNE — {stage_label}",
        "t-SNE 1",
        "t-SNE 2",
        output_dir / "tsne.png",
    )
    np.savez_compressed(
        output_dir / "embedding_coordinates.npz",
        pca=pca_coordinates.astype(np.float32),
        pca_labels=labels,
        pca_priority=priority,
        tsne=tsne_coordinates.astype(np.float32),
        tsne_labels=labels[selected],
        tsne_priority=priority[selected],
    )
    return {
        "sample_counts": {
            name: int(np.count_nonzero(labels == code))
            for code, name in enumerate(CATEGORY_NAMES)
        },
        "population_counts": {
            name: int(sample.finite_counts[code]) for code, name in enumerate(CATEGORY_NAMES)
        },
        "mapped_before_finite_filter_counts": {
            name: int(sample.mapped_counts[code]) for code, name in enumerate(CATEGORY_NAMES)
        },
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tsne_rows": int(len(selected)),
        "tsne_kl_divergence": float(tsne.kl_divergence_),
        "visualization_transform": (
            "signed_log1p(raw), then sample-fitted StandardScaler"
            if stage == "before_preprocessing"
            else "asinh(production RobustScaler output), then sample-fitted StandardScaler"
        ),
    }


def plot_cicids2018_reduction(output_dir: Path) -> dict[str, Any]:
    source = DATASETS["cicids2018"]["processed_dir"] / "class_reduction_by_split.csv"
    frame = pd.read_csv(source)
    summary = (
        frame.groupby("category_label", as_index=False)[["before", "after", "removed"]]
        .sum()
        .set_index("category_label")
        .reindex(CATEGORY_NAMES)
    )
    summary["retained_pct"] = 100.0 * summary["after"] / summary["before"]
    summary.to_csv(output_dir / "cicids2018_reduction_counts.csv")

    x = np.arange(len(CATEGORY_NAMES))
    width = 0.37
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})
    before = summary["before"].to_numpy(dtype=float)
    after = summary["after"].to_numpy(dtype=float)
    axes[0].bar(x - width / 2, before, width, color="#7F8C8D", label="Before reduction")
    axes[0].bar(x + width / 2, after, width, color=[CLASS_COLORS[name] for name in CATEGORY_NAMES], label="After reduction")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, CATEGORY_NAMES, rotation=20, ha="right")
    axes[0].set_ylabel("Rows (log scale)")
    axes[0].set_title("Exact rows before vs after class-size reduction", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2, which="both")
    for i, (b, a) in enumerate(zip(before, after)):
        axes[0].text(i - width / 2, b * 1.08, f"{int(b):,}", ha="center", va="bottom", fontsize=8, rotation=25)
        axes[0].text(i + width / 2, a * 1.08, f"{int(a):,}", ha="center", va="bottom", fontsize=8, rotation=25)

    composition = np.vstack((before / before.sum(), after / after.sum())) * 100.0
    left = np.zeros(2)
    for code, name in enumerate(CATEGORY_NAMES):
        axes[1].barh([0, 1], composition[:, code], left=left, color=CLASS_COLORS[name], label=name)
        left += composition[:, code]
    axes[1].set_yticks([0, 1], ["Before reduction", "After reduction"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Class composition (%)")
    axes[1].set_title("Reduction changes class composition", fontweight="bold")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=8)
    axes[1].grid(axis="x", alpha=0.2)
    total_before, total_after = int(before.sum()), int(after.sum())
    fig.suptitle(
        f"CSE-CIC-IDS-2018-DistriNet reduction: {total_before:,} → {total_after:,} rows "
        f"({100 * total_after / total_before:.2f}% retained)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cicids2018_data_reduction_showcase.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "before": total_before,
        "after": total_after,
        "retained_pct": 100.0 * total_after / total_before,
        "by_class": summary.reset_index().to_dict(orient="records"),
    }


def write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    reduction = metadata["cicids2018_reduction"]
    lines = [
        "# PCA and t-SNE EDA",
        "",
        "Deterministic descriptive embeddings for the two active DistriNet datasets.",
        "",
        "## Stage definitions",
        "",
        "- **Before preprocessing:** sampled directly from the corrected raw CSVs after only the five-category label mapping and exclusion of non-finite rows required by PCA/t-SNE. No production cleaning, deduplication, split, class-size reduction, or production scaler is applied.",
        "- **After preprocessing:** sampled across the final train/validation/test `X_*.npy` arrays after production filtering, category mapping, float32 feature+category deduplication, chronological within-source-label splitting, train-fitted RobustScaler transformation, and (for 2018) class-size reduction.",
        "- Sampling is deterministic bottom-K sampling, stratified by category. PCA uses at most "
        f"{metadata['arguments']['pca_per_class']:,} rows per category; t-SNE uses at most "
        f"{metadata['arguments']['tsne_per_class']:,}. Class-stratified scatter density therefore does **not** encode natural prevalence.",
        "- Before and after embeddings are fitted independently. Their axes are not a shared coordinate system; compare class overlap/separation, not absolute coordinates.",
        "",
        "## Outputs",
        "",
        "Each `<dataset>/<stage>/` directory contains `pca.png`, `tsne.png`, PCA loadings, and compressed plotted coordinates.",
        "",
        "- `cicids2017/{before_preprocessing,after_preprocessing}/`",
        "- `cicids2018/{before_preprocessing,after_preprocessing}/`",
        f"- `cicids2018_data_reduction_showcase.png`: exact cleaned/deduplicated pre-reduction vs final counts ({reduction['before']:,} → {reduction['after']:,}; {reduction['retained_pct']:.2f}% retained).",
        "- `cicids2018_reduction_counts.csv`: count source for the reduction figure.",
        "- `metadata.json`: sample populations, PCA variance, t-SNE KL divergence, methods, and arguments.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$Env:PYTHONPATH = 'src'",
        "python scripts/generate_final_eda_embeddings.py",
        "```",
        "",
        "These figures are descriptive only. They do not fit or alter any classifier, constraint, validator, or attack artifact.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "FINAL_OUTPUTS" / "eda" / "new")
    parser.add_argument("--pca-per-class", type=int, default=5000)
    parser.add_argument("--tsne-per-class", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.pca_per_class < 2 or args.tsne_per_class < 2 or args.workers < 1:
        parser.error("sample sizes must be >= 2 and workers must be >= 1")
    if args.tsne_per_class > args.pca_per_class:
        parser.error("--tsne-per-class cannot exceed --pca-per-class")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "seed": args.seed,
        "arguments": {
            "pca_per_class": args.pca_per_class,
            "tsne_per_class": args.tsne_per_class,
            "workers": args.workers,
        },
        "stage_definitions": {
            "before_preprocessing": "corrected raw CSV; five-category mapping and finite rows only",
            "after_preprocessing": "final RobustScaler-transformed train+val+test arrays",
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        print(f"{dataset}: sampling corrected raw CSVs", flush=True)
        before = sample_raw(dataset, args.pca_per_class, args.workers, args.seed)
        print(f"{dataset}: sampling final processed arrays", flush=True)
        after = sample_processed(dataset, args.pca_per_class, args.seed)
        manifest = json.loads(
            (DATASETS[dataset]["processed_dir"] / "preprocessing_manifest.json").read_text()
        )
        feature_names = list(manifest["modelling_feature_names"])
        metadata["datasets"][dataset] = {}
        for stage, sample in (("before_preprocessing", before), ("after_preprocessing", after)):
            print(f"{dataset}: generating {stage} PCA and t-SNE", flush=True)
            metadata["datasets"][dataset][stage] = generate_embeddings(
                dataset,
                stage,
                sample,
                feature_names,
                output_dir / dataset / stage,
                args.tsne_per_class,
                args.seed,
            )
    metadata["cicids2018_reduction"] = plot_cicids2018_reduction(output_dir)
    metadata["elapsed_seconds"] = round(time.time() - started, 1)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(output_dir, metadata)
    print(f"Done in {metadata['elapsed_seconds']:.1f}s -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
