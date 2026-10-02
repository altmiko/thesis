"""Render verified before/after preprocessing class-distribution figures."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

REVISION = "class-distribution-v5"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "preprocessing_class_distribution"
CLASSES = ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
BEFORE_LABEL = "Before preprocessing"
AFTER_LABEL = "After preprocessing"
X_LABEL = "Flow count (log scale)"
Y_LABEL = "Class"

SOURCE_TO_CATEGORY = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Recon",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
}
ATTEMPTED_SUFFIX = " - Attempted"
EXPECTED_2017_BEFORE = (1_666_837, 171_779, 95_123, 159_151, 6_953)
EXPECTED_2017_AFTER = (1_647_759, 171_559, 95_098, 159_016, 6_947)
EXPECTED_2018_BEFORE = (59_659_723, 1_834_210, 1_374_148, 89_374, 94_197)
EXPECTED_2018_AFTER = (250_000, 200_000, 200_000, 89_355, 94_197)


@dataclass(frozen=True)
class FigureSpec:
    title: str
    footnote: str
    before: tuple[int, ...]
    after: tuple[int, ...]
    filename: str


def _normalize_label(value: str) -> str:
    label = value.strip().replace("\x96", "-").replace("\u2013", "-").replace("\u2014", "-")
    return "BENIGN" if label.upper() == "BENIGN" else label


def _mapped_label(value: str) -> str | None:
    label = _normalize_label(value)
    # The production manifest locks attempted_policy to "benign".
    source = "BENIGN" if label.endswith(ATTEMPTED_SUFFIX) else label
    return SOURCE_TO_CATEGORY.get(source)


def _read_2017_raw_counts() -> tuple[int, ...]:
    raw_dir = ROOT / "data" / "raw" / "CICIDS_2017_Distrinet"
    expected_files = (
        "Monday-WorkingHours.csv",
        "Tuesday-WorkingHours.csv",
        "Wednesday-WorkingHours.csv",
        "Thursday-WorkingHours.csv",
        "Friday-WorkingHours.csv",
    )
    counts: Counter[str] = Counter()
    for filename in expected_files:
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing authoritative raw input: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = [column.strip() for column in next(reader)]
            except StopIteration as exc:
                raise ValueError(f"Empty raw input: {path}") from exc
            try:
                label_index = header.index("Label")
            except ValueError as exc:
                raise ValueError(f"Label column absent from {path}") from exc
            for row_number, row in enumerate(reader, start=2):
                if label_index >= len(row):
                    raise ValueError(f"Malformed row {row_number} in {path}")
                category = _mapped_label(row[label_index])
                if category is not None:
                    counts[category] += 1
    return tuple(counts[name] for name in CLASSES)


def _read_distribution(path: Path, column: str) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing authoritative distribution: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["category_label"]: row for row in csv.DictReader(handle)}
    if set(rows) != set(CLASSES):
        raise ValueError(f"Unexpected classes in {path}: {sorted(rows)}")
    try:
        return tuple(int(rows[name][column]) for name in CLASSES)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column!r} values in {path}") from exc


def _require_exact(actual: Sequence[int], expected: Sequence[int], source: str) -> None:
    if tuple(actual) != tuple(expected):
        details = ", ".join(
            f"{name}: expected {wanted:,}, found {found:,}"
            for name, wanted, found in zip(CLASSES, expected, actual)
            if wanted != found
        )
        raise RuntimeError(f"Count drift in {source}: {details}")


def load_verified_specs() -> tuple[FigureSpec, FigureSpec]:
    manifest = ROOT / "data" / "processed" / "CICIDS_2017_Distrinet" / "preprocessing_manifest.json"
    if '"attempted_policy": "benign"' not in manifest.read_text(encoding="utf-8"):
        raise RuntimeError(f"CICIDS2017 attempted-label policy drift in {manifest}")

    before_2017 = _read_2017_raw_counts()
    after_2017 = _read_distribution(manifest.parent / "class_distribution.csv", "total")
    dist_2018 = ROOT / "data" / "processed" / "CSECICIDS_2018_Distrinet" / "class_distribution.csv"
    before_2018 = _read_distribution(dist_2018, "raw_retained")
    after_2018 = _read_distribution(dist_2018, "final_total")

    for actual, expected, source in (
        (before_2017, EXPECTED_2017_BEFORE, "CICIDS2017 raw Label rows"),
        (after_2017, EXPECTED_2017_AFTER, "CICIDS2017 class_distribution.csv total"),
        (before_2018, EXPECTED_2018_BEFORE, "CICIDS2018 class_distribution.csv raw_retained"),
        (after_2018, EXPECTED_2018_AFTER, "CICIDS2018 class_distribution.csv final_total"),
    ):
        _require_exact(actual, expected, source)

    return (
        FigureSpec(
            "Class Distribution in CICIDS2017-DistriNet",
            "Before: supported raw rows mapped to the five classes before cleaning. After: final retained flows.",
            before_2017,
            after_2017,
            "cicids2017_class_distribution_before_after.png",
        ),
        FigureSpec(
            "Class Distribution in CSE-CIC-IDS-2018-DistriNet",
            "Before: supported raw rows mapped to the five classes before cleaning. After: final retained flows, including class-size reduction.",
            before_2018,
            after_2018,
            "cicids2018_class_distribution_before_after.png",
        ),
    )


def render(spec: FigureSpec) -> Path:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(20 / 3, 10), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.26, right=0.93, top=0.76, bottom=0.17)
    y = np.arange(len(CLASSES))
    height = 0.32
    colors = ("#0072B2", "#D55E00")
    before_bars = ax.barh(
        y - height / 2, spec.before, height, label=BEFORE_LABEL, color=colors[0]
    )
    after_bars = ax.barh(
        y + height / 2, spec.after, height, label=AFTER_LABEL, color=colors[1]
    )
    ax.set_xscale("log")
    ax.set_xlim(2_500, max(max(spec.before), max(spec.after)) * 4.8)
    fig.suptitle(
        fill(spec.title, width=38, break_long_words=False, break_on_hyphens=False),
        y=0.93,
        fontsize=16,
        fontweight="bold",
        linespacing=1.15,
    )
    ax.set_xlabel(X_LABEL, labelpad=11)
    ax.set_ylabel(Y_LABEL, labelpad=11)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.tick_params(axis="y", pad=7)
    ax.margins(y=0.09)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", which="major", color="#AAB2B8", linewidth=0.8, alpha=0.65)
    ax.grid(axis="x", which="minor", color="#D5DADD", linewidth=0.45, alpha=0.50)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        columnspacing=2.0,
        handletextpad=0.7,
    )

    for bars in (before_bars, after_bars):
        for bar in bars:
            value = int(bar.get_width())
            ax.annotate(
                f"{value:,}",
                (value, bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=0,
                fontsize=9,
                color="#202124",
            )

    fig.text(
        0.5,
        0.06,
        fill(spec.footnote, width=60, break_long_words=False, break_on_hyphens=False),
        ha="center",
        va="center",
        fontsize=9.5,
        linespacing=1.35,
        color="#3C4043",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / spec.filename
    fig.savefig(
        destination,
        dpi=180,
        facecolor="white",
        metadata={"Revision": REVISION, "Title": spec.title},
    )
    plt.close(fig)
    return destination


def main() -> None:
    for spec in load_verified_specs():
        print(render(spec))


if __name__ == "__main__":
    main()
