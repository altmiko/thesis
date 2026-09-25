#!/usr/bin/env python3
"""Preprocess the corrected DistriNet CSE-CIC-IDS-2018 release (secondary dataset).

Mirrors ``preprocess_cicids2017_distrinet.py``: the 79 CICIDS2017 modelling features, five
categories (Benign, DoS, DDoS, Recon, BruteForce), global float32 feature+category
deduplication keeping the earliest occurrence, a chronological 70/15/15 split within each
retained source label, and a RobustScaler fitted on the final training rows only.

Two steps are specific to CICIDS2018:

* negative ``Fwd/Bwd Header Length`` values (a 16-bit extractor artefact) are kept and
  flagged instead of dropping the row, because dropping them removes 47.8% of DDoS-LOIC-UDP;
* after the chronological split, the classes named in ``--class-row-targets`` (default
  Benign=250000, DoS=200000, DDoS=200000) are undersampled to that TOTAL row count. Each total is
  apportioned 70/15/15 over train/validation/test and sampled independently inside each split,
  stratified by source label x source file x fixed-width UTC time bin. Unlisted classes (Recon,
  BruteForce) keep every row. Validation/test are controlled, not natural-prevalence, sets.

The 36 GB input never sits in memory. Pass 1 streams every CSV and keeps compact per-row
keys (file, id, timestamp, label, 128-bit hash of the float32 feature vector).
Deduplication, splitting and sampling run on these keys. Pass 2 streams the CSVs again and
writes only the selected rows into memory-mapped float32 arrays.

Run from the repository root:
    python src/preprocessing/preprocess_cicids2018_distrinet.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from config.paths import SEED  # noqa: E402
from evaluation.cicids2018_distrinet_eda import (  # noqa: E402
    EXPECTED_FILES,
    MISSING_LABEL,
    PROCESS_ROWS,
    READ_BLOCK_SIZE,
    HashingReader,
    _json_default,
    _rebatched,
    decode_label,
    md_table,
    read_header,
)
from preprocessing.preprocess_cicids2017_distrinet import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_ID,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    allocate_class_counts,
    balanced_class_weights,
)

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw" / "CSECICIDS2018_Distrinet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "CSECICIDS_2018_Distrinet"
DEFAULT_REPORT_DIR = REPO_ROOT / "outputs" / "cicids2018distrinet" / "preprocessing"
DEFAULT_REFERENCE_MANIFEST = (
    REPO_ROOT / "data" / "processed" / "CICIDS_2017_Distrinet" / "preprocessing_manifest.json"
)
DEFAULT_CLASS_ROW_TARGETS = "Benign=250000,DoS=200000,DDoS=200000"
DEFAULT_STRATUM_HOURS = 1
DEFAULT_WORKERS = 8
DEFAULT_MIN_PER_SPLIT_WARNING = 10
DEFAULT_SPEARMAN_SAMPLE = 100_000
DEFAULT_PCA_PER_CLASS = 5_000
EXPECTED_MODELLING_FEATURE_COUNT = 79
# Timestamp parsing "fails materially" above this fraction of rows; below it the
# unparseable rows are dropped and counted.
MAX_TIMESTAMP_FAILURE_FRACTION = 1e-5
OUTPUT_CHUNK_ROWS = 1_000_000

ID_COLUMN = "id"
FLOW_ID_COLUMN = "Flow ID"
SRC_IP_COLUMN = "Src IP"
DST_IP_COLUMN = "Dst IP"
TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"
ATTEMPTED_COLUMN = "Attempted Category"
NON_FEATURE_COLUMNS = (
    ID_COLUMN,
    FLOW_ID_COLUMN,
    SRC_IP_COLUMN,
    DST_IP_COLUMN,
    TIMESTAMP_COLUMN,
    LABEL_COLUMN,
    ATTEMPTED_COLUMN,
)
# Present in the 2018 export only; excluded so the schema equals the CICIDS2017 one.
CICIDS2018_ONLY_COLUMNS = (
    "Fwd RST Flags",
    "Bwd RST Flags",
    "ICMP Code",
    "ICMP Type",
    "Total TCP Flow Time",
)
PROTOCOL_COLUMN = "Protocol"
# Header-length column -> packet-count column of the same direction.
HEADER_LENGTH_COLUMNS = {
    "Fwd Header Length": "Total Fwd Packet",
    "Bwd Header Length": "Total Bwd packets",
}
HEADER_SHORT = {"Fwd Header Length": "fwd_header", "Bwd Header Length": "bwd_header"}
# Smallest transport header per packet (TCP without options, UDP). Used only for the
# descriptive overflow audit, never to alter values.
MIN_HEADER_BYTES_PER_PACKET = {6: 20, 17: 8}
INT16_MAX = 32767
# header_flags bits: 0/1 = Fwd/Bwd Header Length negative; 2/3 = Fwd/Bwd overflow, i.e.
# packets x minimum transport header > INT16_MAX, so the stored value is certainly wrapped
# whatever its sign.
HEADER_NEGATIVE_BITS = 0b0011
HEADER_OVERFLOW_BITS = 0b1100
US_PER_SECOND = 1_000_000
US_PER_HOUR = 3_600 * US_PER_SECOND

BENIGN_SOURCE_LABEL = "BENIGN"
ATTEMPTED_SUFFIX = " - Attempted"
SOURCE_TO_CATEGORY: dict[str, str] = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowloris": "DoS",
    "DDoS-HOIC": "DDoS",
    "DDoS-LOIC-HTTP": "DDoS",
    "DDoS-LOIC-UDP": "DDoS",
    "Infiltration - NMAP Portscan": "Recon",
    "SSH-BruteForce": "BruteForce",
}
SOURCE_LABELS = tuple(SOURCE_TO_CATEGORY)
# Every Attempted label observed by the EDA. Policy: Attempted -> BENIGN (DistriNet guidance).
ATTEMPTED_LABELS = (
    "FTP-BruteForce - Attempted",
    "DoS GoldenEye - Attempted",
    "DoS Slowloris - Attempted",
    "Botnet Ares - Attempted",
    "DDoS-LOIC-UDP - Attempted",
    "Web Attack - Brute Force - Attempted",
    "DoS Hulk - Attempted",
    "Infiltration - Dropbox Download - Attempted",
    "Web Attack - SQL - Attempted",
    "Web Attack - XSS - Attempted",
)
# Out-of-scope attacks, dropped as Bot/Web/Infiltration were for CICIDS2017.
DROPPED_LABELS = (
    "Botnet Ares",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - SQL",
    "Infiltration - Dropbox Download",
    "Infiltration - Communication Victim Attacker",
)
UNKNOWN_LABEL = "<unknown>"
# Slot 0 = unknown label; any row landing there aborts the run.
LABEL_VOCAB = (UNKNOWN_LABEL, *SOURCE_LABELS, *ATTEMPTED_LABELS, *DROPPED_LABELS)
LABEL_SLOT = {label: slot for slot, label in enumerate(LABEL_VOCAB)}


def _slot_tables() -> tuple[np.ndarray, np.ndarray]:
    category = np.full(len(LABEL_VOCAB), -1, dtype=np.int8)
    source = np.full(len(LABEL_VOCAB), -1, dtype=np.int8)
    for slot, label in enumerate(LABEL_VOCAB):
        source_label = BENIGN_SOURCE_LABEL if label.endswith(ATTEMPTED_SUFFIX) else label
        if source_label in SOURCE_TO_CATEGORY:
            source[slot] = SOURCE_LABELS.index(source_label)
            category[slot] = CATEGORY_TO_ID[SOURCE_TO_CATEGORY[source_label]]
    return category, source


SLOT_CATEGORY, SLOT_SOURCE = _slot_tables()
SLOT_IS_ATTEMPTED = np.asarray([label.endswith(ATTEMPTED_SUFFIX) for label in LABEL_VOCAB])
FILE_NAMES = tuple(EXPECTED_FILES)
SPLIT_CODE = {name: code for code, name in enumerate(SPLIT_NAMES)}

# Cleaning rules, applied in this order; each row is charged to its first failing rule.
CLEANING_RULES = (
    ("nonfinite", "drop rows with NaN/null/+-inf in any of the 79 modelling features"),
    ("bad_timestamp", "drop rows whose Timestamp is missing or unparseable"),
    (
        "negative_non_header",
        "drop rows with a negative value in a modelling feature other than Fwd/Bwd Header Length "
        "(the CICIDS2017 negative-value rule, minus the documented header-length exemption)",
    ),
    ("unsupported_label", "drop rows whose label maps to none of the five categories"),
)


# --------------------------------------------------------------------------- #
# Row hashing (128-bit, exact up to hash collisions)
# --------------------------------------------------------------------------- #
_HASH_SEEDS = (np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707344))


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _fmix64(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
    z = (z ^ (z >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53)
    return z ^ (z >> np.uint64(33))


def feature_hashes(x32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two independent 64-bit hashes of each float32 row's exact bit pattern."""
    if x32.dtype != np.float32 or not x32.flags.c_contiguous:
        raise TypeError("feature_hashes expects a C-contiguous float32 matrix")
    bits = x32.view(np.uint32)
    h1 = np.full(len(x32), _HASH_SEEDS[0], dtype=np.uint64)
    h2 = np.full(len(x32), _HASH_SEEDS[1], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(bits.shape[1]):
            column = bits[:, j].astype(np.uint64)
            h1 = _splitmix64(h1 ^ column)
            h2 = _fmix64(h2 ^ (column | (np.uint64(j + 1) << np.uint64(32))))
    return h1, h2


def labelled_hashes(
    hash_1: np.ndarray, hash_2: np.ndarray, category: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mix the mapped category into the feature hash: the deduplication key."""
    code = category.astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        return _splitmix64(hash_1 ^ code), _fmix64(hash_2 ^ (code << np.uint64(40)))


def canonical_float32(x: np.ndarray) -> np.ndarray:
    """Float32 model representation; -0.0 is folded into +0.0 so bit hashes equal value equality."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    x32 += np.float32(0.0)
    return x32


# --------------------------------------------------------------------------- #
# Inventory and schema
# --------------------------------------------------------------------------- #
def load_reference_features(path: Path, expected_count: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"CICIDS2017 reference manifest missing: {path}; it defines the 79 modelling features"
        )
    names = json.loads(path.read_text(encoding="utf-8")).get("modelling_feature_names")
    if not names:
        raise ValueError(f"{path} lacks modelling_feature_names")
    if len(names) != expected_count or len(set(names)) != len(names):
        raise ValueError(
            f"reference feature list has {len(names)} entries ({len(set(names))} unique); "
            f"expected {expected_count}"
        )
    return list(names)


def validate_inventory(input_dir: Path, features: list[str]) -> tuple[list[Path], list[str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    present = sorted(path.name for path in input_dir.glob("*.csv"))
    expected = sorted(FILE_NAMES)
    if present != expected:
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        raise ValueError(f"CSV inventory mismatch; missing={missing}, extra={extra}")
    paths = [input_dir / name for name in FILE_NAMES]
    headers = [read_header(path) for path in paths]
    if any(header != headers[0] for header in headers[1:]):
        raise ValueError("CSV headers differ across DistriNet CSE-CIC-IDS-2018 files")
    header = headers[0]
    if len(header) != len(set(header)):
        raise ValueError("duplicate normalized column names are not supported")
    missing_meta = sorted(set(NON_FEATURE_COLUMNS) - set(header))
    if missing_meta:
        raise ValueError(f"required metadata columns absent: {missing_meta}")
    missing_features = [name for name in features if name not in header]
    if missing_features:
        raise ValueError(f"required modelling features absent: {missing_features}")
    positions = [header.index(name) for name in features]
    if positions != sorted(positions):
        raise ValueError("modelling features appear in a different relative order than CICIDS2017")
    extras = sorted(set(header) - set(NON_FEATURE_COLUMNS) - set(features))
    if extras != sorted(CICIDS2018_ONLY_COLUMNS):
        raise ValueError(
            f"unexpected non-modelling columns {extras}; expected {sorted(CICIDS2018_ONLY_COLUMNS)}"
        )
    for name in (PROTOCOL_COLUMN, *HEADER_LENGTH_COLUMNS, *HEADER_LENGTH_COLUMNS.values()):
        if name not in features:
            raise ValueError(f"cleaning audit needs modelling feature {name!r}")
    return paths, header


def _read_options(raw_columns: list[str]) -> pacsv.ReadOptions:
    return pacsv.ReadOptions(block_size=READ_BLOCK_SIZE, column_names=raw_columns, skip_rows=1)


def _convert_options(features: list[str], include: list[str]) -> pacsv.ConvertOptions:
    types: dict[str, pa.DataType] = {name: pa.float64() for name in features}
    types.update(
        {
            ID_COLUMN: pa.int64(),
            LABEL_COLUMN: pa.binary(),
            ATTEMPTED_COLUMN: pa.float64(),
            TIMESTAMP_COLUMN: pa.string(),
            FLOW_ID_COLUMN: pa.string(),
            SRC_IP_COLUMN: pa.string(),
            DST_IP_COLUMN: pa.string(),
        }
    )
    return pacsv.ConvertOptions(
        column_types={name: types[name] for name in include},
        include_columns=include,
        null_values=["", "NaN", "nan", "NULL", "null"],
        strings_can_be_null=True,
    )


def _float_matrix(batch: pa.RecordBatch, features: list[str]) -> np.ndarray:
    x = np.empty((batch.num_rows, len(features)), dtype=np.float64)
    for j, name in enumerate(features):
        x[:, j] = batch.column(name).to_numpy(zero_copy_only=False)
    return x


# --------------------------------------------------------------------------- #
# Pass 1: stream, clean, map, key (runs in worker processes)
# --------------------------------------------------------------------------- #
@dataclass
class ScanTask:
    path: Path
    file_index: int
    raw_columns: list[str]
    features: list[str]
    max_rows: int | None
    hash_input: bool


@dataclass
class ScanResult:
    name: str
    file_index: int
    size_bytes: int
    sha256: str | None
    raw_rows: int
    complete: bool
    source_id: np.ndarray
    timestamp_us: np.ndarray
    label_slot: np.ndarray
    attempted_category: np.ndarray
    header_flags: np.ndarray
    feature_hash_1: np.ndarray
    feature_hash_2: np.ndarray
    rule_counts: dict[str, np.ndarray]
    unknown_labels: Counter
    label_decode_fallback_rows: int
    nonfinite_columns: dict[str, int]
    nonfinite_examples: list[dict[str, Any]]
    timestamp: dict[str, Any]
    header_extrema: dict[str, list[float]]


def _label_slots(column: pa.Array, unknown: Counter) -> tuple[np.ndarray, int]:
    encoded = column.dictionary_encode()
    dictionary = encoded.dictionary.to_pylist()
    null_index = len(dictionary)
    slot_of = np.zeros(null_index + 1, dtype=np.int16)
    fallback = np.zeros(null_index + 1, dtype=bool)
    names = []
    for j, raw in enumerate(dictionary):
        label, used_fallback = decode_label(raw)
        names.append(label)
        slot_of[j] = LABEL_SLOT.get(label, 0)
        fallback[j] = used_fallback
    names.append(MISSING_LABEL)
    indices = pc.fill_null(encoded.indices, null_index).to_numpy(zero_copy_only=False)
    per_entry = np.bincount(indices, minlength=null_index + 1)
    for j in np.flatnonzero((slot_of == 0) & (per_entry > 0)):
        unknown[names[j]] += int(per_entry[j])
    return slot_of[indices], int(per_entry[fallback].sum())


def scan_file(task: ScanTask) -> ScanResult:
    features = task.features
    feature_index = {name: j for j, name in enumerate(features)}
    header_idx = [feature_index[name] for name in HEADER_LENGTH_COLUMNS]
    other_idx = np.asarray([j for j in range(len(features)) if j not in header_idx])
    protocol_i = feature_index[PROTOCOL_COLUMN]
    include = [ID_COLUMN, TIMESTAMP_COLUMN, LABEL_COLUMN, ATTEMPTED_COLUMN, *features]
    read_options = _read_options(task.raw_columns)
    convert_options = _convert_options(features, include)
    expected_day = int(np.datetime64(EXPECTED_FILES[task.path.name], "D").astype(np.int64))

    vocab_size = len(LABEL_VOCAB)
    rule_counts: dict[str, np.ndarray] = {}

    def add(rule: str, slots: np.ndarray) -> None:
        counts = np.bincount(slots, minlength=vocab_size).astype(np.int64)
        rule_counts[rule] = rule_counts.get(rule, np.zeros(vocab_size, np.int64)) + counts

    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("id", "ts", "slot", "attempted", "flags", "h1", "h2")
    }
    unknown: Counter = Counter()
    fallback_rows = 0
    nonfinite_columns = np.zeros(len(features), dtype=np.int64)
    nonfinite_examples: list[dict[str, Any]] = []
    ts_info: dict[str, Any] = {
        "missing": 0,
        "unparseable": 0,
        "date_differs_from_file_name": 0,
        "min_us": None,
        "max_us": None,
    }
    header_extrema = {name: [np.inf, -np.inf] for name in HEADER_LENGTH_COLUMNS}
    rows = 0
    state = {"truncated": False}
    size_bytes = task.path.stat().st_size

    with task.path.open("rb") as raw_handle:
        hashing = HashingReader(raw_handle) if task.hash_input else None
        source: Any = hashing if hashing is not None else raw_handle
        reader = pacsv.open_csv(source, read_options=read_options, convert_options=convert_options)
        for batch in _rebatched(reader, PROCESS_ROWS, task.max_rows, state):
            n = batch.num_rows
            ids = batch.column(ID_COLUMN).to_numpy(zero_copy_only=False)
            if not np.array_equal(ids, np.arange(rows + 1, rows + n + 1)):
                raise ValueError(
                    f"{task.path.name}: 'id' is not the 1-based row number near row {rows + 1}; "
                    "file:id would not be a stable key"
                )
            slots, fallbacks = _label_slots(batch.column(LABEL_COLUMN), unknown)
            fallback_rows += fallbacks

            x = _float_matrix(batch, features)
            finite_cells = np.isfinite(x)
            finite = finite_cells.all(axis=1)
            nonfinite_columns += (~finite_cells).sum(axis=0)
            if not finite.all() and len(nonfinite_examples) < 500:
                for i in np.flatnonzero(~finite)[: 500 - len(nonfinite_examples)]:
                    nonfinite_examples.append(
                        {
                            "source_file": task.path.name,
                            "source_id": int(ids[i]),
                            "label": LABEL_VOCAB[slots[i]],
                            "nonfinite_columns": [
                                features[j] for j in np.flatnonzero(~finite_cells[i])
                            ],
                            "values": [
                                str(x[i, j]) for j in np.flatnonzero(~finite_cells[i])
                            ],
                        }
                    )

            ts_series = batch.column(TIMESTAMP_COLUMN).to_pandas()
            missing_ts = ts_series.isna().to_numpy()
            parsed = pd.to_datetime(ts_series, format="ISO8601", errors="coerce")
            valid_ts = parsed.notna().to_numpy()
            ts_us = np.zeros(n, dtype=np.int64)
            ts_us[valid_ts] = (
                parsed[valid_ts].to_numpy().astype("datetime64[us]").astype(np.int64)
            )
            ts_info["missing"] += int(missing_ts.sum())
            ts_info["unparseable"] += int((~valid_ts & ~missing_ts).sum())
            if valid_ts.any():
                valid_us = ts_us[valid_ts]
                ts_info["date_differs_from_file_name"] += int(
                    (valid_us // (86_400 * US_PER_SECOND) != expected_day).sum()
                )
                lo, hi = int(valid_us.min()), int(valid_us.max())
                ts_info["min_us"] = lo if ts_info["min_us"] is None else min(ts_info["min_us"], lo)
                ts_info["max_us"] = hi if ts_info["max_us"] is None else max(ts_info["max_us"], hi)

            with np.errstate(invalid="ignore"):
                negative_other = (x[:, other_idx] < 0).any(axis=1)
                header_negative = x[:, header_idx] < 0
            mapped = SLOT_CATEGORY[slots] >= 0
            clean = finite & valid_ts & ~negative_other
            keep = clean & mapped

            add("raw", slots)
            add("nonfinite", slots[~finite])
            add("bad_timestamp", slots[finite & ~valid_ts])
            add("negative_non_header", slots[finite & valid_ts & negative_other])
            add("unsupported_label", slots[clean & ~mapped])
            add("kept", slots[keep])
            add("header_negative_any_clean_rows", slots[clean & header_negative.any(axis=1)])

            kept_slots = slots[keep]
            protocol = x[keep, protocol_i]
            min_bytes = np.zeros(len(protocol), dtype=np.float64)
            for proto, per_packet in MIN_HEADER_BYTES_PER_PACKET.items():
                min_bytes[protocol == proto] = per_packet
            flags = np.zeros(int(keep.sum()), dtype=np.uint8)
            for bit, (name, packet_name) in enumerate(HEADER_LENGTH_COLUMNS.items()):
                short = HEADER_SHORT[name]
                values = x[keep, feature_index[name]]
                bound = x[keep, feature_index[packet_name]] * min_bytes
                negative = values < 0
                overflow = bound > INT16_MAX
                flags |= negative.astype(np.uint8) << bit
                flags |= overflow.astype(np.uint8) << (bit + 2)
                add(f"{short}_negative", kept_slots[negative])
                add(
                    f"{short}_nonnegative_below_protocol_minimum",
                    kept_slots[(values >= 0) & (values < bound)],
                )
                add(f"{short}_protocol_minimum_exceeds_int16", kept_slots[overflow])
                if values.size:
                    header_extrema[name][0] = min(header_extrema[name][0], float(values.min()))
                    header_extrema[name][1] = max(header_extrema[name][1], float(values.max()))
            add("header_negative_any", kept_slots[(flags & HEADER_NEGATIVE_BITS) > 0])
            add("header_overflow_any", kept_slots[(flags & HEADER_OVERFLOW_BITS) > 0])

            x32 = canonical_float32(x[keep])
            h1, h2 = feature_hashes(x32)
            attempted = batch.column(ATTEMPTED_COLUMN).to_numpy(zero_copy_only=False)[keep]
            attempted = np.where(np.isfinite(attempted), attempted, -128).astype(np.int8)
            parts["id"].append(ids[keep].astype(np.int32))
            parts["ts"].append(ts_us[keep])
            parts["slot"].append(kept_slots.astype(np.int16))
            parts["attempted"].append(attempted)
            parts["flags"].append(flags)
            parts["h1"].append(h1)
            parts["h2"].append(h2)
            rows += n
        complete = not state["truncated"]
        sha256 = None
        if hashing is not None and complete:
            while hashing.read(1 << 20):
                pass
            if hashing.bytes_read != size_bytes:
                raise AssertionError(
                    f"{task.path.name}: hashed {hashing.bytes_read} bytes, file has {size_bytes}"
                )
            sha256 = hashing.digest.hexdigest()

    def cat(key: str, dtype: Any) -> np.ndarray:
        return np.concatenate(parts[key]).astype(dtype, copy=False) if parts[key] else np.empty(0, dtype)

    return ScanResult(
        name=task.path.name,
        file_index=task.file_index,
        size_bytes=size_bytes,
        sha256=sha256,
        raw_rows=rows,
        complete=complete,
        source_id=cat("id", np.int32),
        timestamp_us=cat("ts", np.int64),
        label_slot=cat("slot", np.int16),
        attempted_category=cat("attempted", np.int8),
        header_flags=cat("flags", np.uint8),
        feature_hash_1=cat("h1", np.uint64),
        feature_hash_2=cat("h2", np.uint64),
        rule_counts=rule_counts,
        unknown_labels=unknown,
        label_decode_fallback_rows=fallback_rows,
        nonfinite_columns={
            features[j]: int(v) for j, v in enumerate(nonfinite_columns) if v
        },
        nonfinite_examples=nonfinite_examples,
        timestamp=ts_info,
        header_extrema=header_extrema,
    )


# --------------------------------------------------------------------------- #
# Pass 2: extract the selected rows into memory-mapped arrays (worker processes)
# --------------------------------------------------------------------------- #
@dataclass
class ExtractTask:
    path: Path
    file_index: int
    raw_columns: list[str]
    features: list[str]
    max_rows: int | None
    source_id: np.ndarray  # sorted ascending
    split_code: np.ndarray
    position: np.ndarray
    feature_hash_1: np.ndarray
    feature_hash_2: np.ndarray
    pristine_paths: dict[int, str]


@dataclass
class ExtractResult:
    file_index: int
    rows_written: int
    strings: dict[int, pa.Table]


def extract_file(task: ExtractTask) -> ExtractResult:
    include = [ID_COLUMN, FLOW_ID_COLUMN, SRC_IP_COLUMN, DST_IP_COLUMN, *task.features]
    read_options = _read_options(task.raw_columns)
    convert_options = _convert_options(task.features, include)
    outputs = {code: np.load(path, mmap_mode="r+") for code, path in task.pristine_paths.items()}
    wanted = task.source_id.astype(np.int64)
    string_parts: dict[int, list[pa.Table]] = {code: [] for code in task.pristine_paths}
    written = 0
    rows = 0
    state = {"truncated": False}
    with task.path.open("rb") as handle:
        reader = pacsv.open_csv(handle, read_options=read_options, convert_options=convert_options)
        for batch in _rebatched(reader, PROCESS_ROWS, task.max_rows, state):
            n = batch.num_rows
            lo = int(np.searchsorted(wanted, rows + 1))
            hi = int(np.searchsorted(wanted, rows + n + 1))
            if hi > lo:
                local = wanted[lo:hi] - (rows + 1)
                sub = batch.take(pa.array(local))
                ids = sub.column(ID_COLUMN).to_numpy(zero_copy_only=False)
                if not np.array_equal(ids, wanted[lo:hi]):
                    raise AssertionError(f"{task.path.name}: pass-2 row ids drifted from pass 1")
                x32 = canonical_float32(_float_matrix(sub, task.features))
                h1, h2 = feature_hashes(x32)
                if not (
                    np.array_equal(h1, task.feature_hash_1[lo:hi])
                    and np.array_equal(h2, task.feature_hash_2[lo:hi])
                ):
                    raise AssertionError(
                        f"{task.path.name}: pass-2 feature values differ from pass 1"
                    )
                split = task.split_code[lo:hi]
                position = task.position[lo:hi]
                for code, output in outputs.items():
                    mask = split == code
                    if not mask.any():
                        continue
                    output[position[mask]] = x32[mask]
                    selected = pa.array(np.flatnonzero(mask))
                    string_parts[code].append(
                        pa.table(
                            {
                                "position": pa.array(position[mask]),
                                FLOW_ID_COLUMN: sub.column(FLOW_ID_COLUMN).take(selected),
                                SRC_IP_COLUMN: sub.column(SRC_IP_COLUMN).take(selected),
                                DST_IP_COLUMN: sub.column(DST_IP_COLUMN).take(selected),
                            }
                        )
                    )
                written += hi - lo
            rows += n
    for output in outputs.values():
        output.flush()
    del outputs
    if written != len(wanted):
        raise AssertionError(f"{task.path.name}: extracted {written} of {len(wanted)} rows")
    strings = {
        code: pa.concat_tables(tables).combine_chunks()
        for code, tables in string_parts.items()
        if tables
    }
    return ExtractResult(task.file_index, written, strings)


def run_parallel(function: Callable[[Any], Any], tasks: list[Any], workers: int) -> list[Any]:
    results: list[Any] = []
    if workers == 1:
        for task in tasks:
            results.append(function(task))
            print(f"  done: {task.path.name}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for task, result in zip(tasks, pool.map(function, tasks)):
            results.append(result)
            print(f"  done: {task.path.name}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Key table: chronology, deduplication, splitting, Benign sampling
# --------------------------------------------------------------------------- #
@dataclass
class Keys:
    """Every cleaned, label-retained row in chronological (timestamp, file, id) order."""

    file_index: np.ndarray
    source_id: np.ndarray
    timestamp_us: np.ndarray
    label_slot: np.ndarray
    attempted_category: np.ndarray
    header_flags: np.ndarray
    feature_hash_1: np.ndarray
    feature_hash_2: np.ndarray

    def __len__(self) -> int:
        return len(self.source_id)

    @property
    def category(self) -> np.ndarray:
        return SLOT_CATEGORY[self.label_slot]

    @property
    def source(self) -> np.ndarray:
        return SLOT_SOURCE[self.label_slot]


def assemble_keys(scans: list[ScanResult]) -> Keys:
    file_index = np.concatenate(
        [np.full(len(scan.source_id), scan.file_index, dtype=np.int8) for scan in scans]
    )
    source_id = np.concatenate([scan.source_id for scan in scans])
    timestamp_us = np.concatenate([scan.timestamp_us for scan in scans])
    # Chronology comes from the parsed timestamp only; file order and id break ties.
    order = np.lexsort((source_id, file_index, timestamp_us))

    def take(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts)[order]

    return Keys(
        file_index=file_index[order],
        source_id=source_id[order],
        timestamp_us=timestamp_us[order],
        label_slot=take([scan.label_slot for scan in scans]),
        attempted_category=take([scan.attempted_category for scan in scans]),
        header_flags=take([scan.header_flags for scan in scans]),
        feature_hash_1=take([scan.feature_hash_1 for scan in scans]),
        feature_hash_2=take([scan.feature_hash_2 for scan in scans]),
    )


def assert_chronological(keys: Keys) -> None:
    ts, fi, sid = keys.timestamp_us, keys.file_index, keys.source_id
    later = (ts[1:] > ts[:-1]) | (
        (ts[1:] == ts[:-1])
        & ((fi[1:] > fi[:-1]) | ((fi[1:] == fi[:-1]) & (sid[1:] > sid[:-1])))
    )
    if not later.all():
        raise AssertionError("key table is not strictly ordered by (timestamp, file, id)")


def deduplicate(keys: Keys) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_duplicate, duplicate_of) with the earliest chronological row kept.

    ``np.lexsort`` is stable, so within one (hash_1, hash_2) group rows stay in
    chronological order and the first one is the survivor.
    """
    h1, h2 = labelled_hashes(keys.feature_hash_1, keys.feature_hash_2, keys.category)
    order = np.lexsort((h2, h1))
    s1, s2 = h1[order], h2[order]
    starts = np.ones(len(order), dtype=bool)
    starts[1:] = (s1[1:] != s1[:-1]) | (s2[1:] != s2[:-1])
    group_start = np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0))
    first = order[group_start]
    is_duplicate = np.zeros(len(order), dtype=bool)
    is_duplicate[order[~starts]] = True
    duplicate_of = np.full(len(order), -1, dtype=np.int64)
    duplicate_of[order[~starts]] = first[~starts]
    if is_duplicate.any() and (duplicate_of[is_duplicate] >= np.flatnonzero(is_duplicate)).any():
        raise AssertionError("a duplicate survived instead of its earliest occurrence")
    return is_duplicate, duplicate_of


def label_conflicts(keys: Keys, kept: np.ndarray) -> dict[str, Any]:
    """Identical float32 feature vectors that carry different categories after dedup."""
    idx = np.flatnonzero(kept)
    order = idx[np.lexsort((keys.feature_hash_2[idx], keys.feature_hash_1[idx]))]
    h1, h2 = keys.feature_hash_1[order], keys.feature_hash_2[order]
    same = (h1[1:] == h1[:-1]) & (h2[1:] == h2[:-1])
    in_group = np.zeros(len(order), dtype=bool)
    in_group[1:] |= same
    in_group[:-1] |= same
    starts = np.ones(len(order), dtype=bool)
    starts[1:] = ~same
    group_id = np.cumsum(starts) - 1
    rows = order[in_group]
    group_of_row = group_id[in_group]
    groups = np.split(rows, np.flatnonzero(np.diff(group_of_row)) + 1) if len(rows) else []
    pairs: Counter = Counter()
    label_pairs: Counter = Counter()
    categories = keys.category
    for members in groups:
        pairs[" | ".join(sorted({CATEGORY_NAMES[c] for c in categories[members]}))] += 1
        label_pairs[" | ".join(sorted({LABEL_VOCAB[s] for s in keys.label_slot[members]}))] += 1
    return {
        "definition": "identical float32 79-feature vector with more than one mapped category",
        "conflicting_feature_vectors": len(groups),
        "rows_involved": int(len(rows)),
        "category_sets": dict(pairs.most_common()),
        "original_label_sets": dict(label_pairs.most_common()),
        "policy": "kept: the duplicate key is features plus category, so these are distinct samples",
    }


def split_within_source_label(
    keys: Keys, kept: np.ndarray, warning_floor: int
) -> tuple[np.ndarray, list[str]]:
    """Chronological 70/15/15 within each retained source label (CICIDS2017 protocol)."""
    split_code = np.full(len(keys), -1, dtype=np.int8)
    source = keys.source
    warnings_list: list[str] = []
    for source_code, source_label in enumerate(SOURCE_LABELS):
        indices = np.flatnonzero(kept & (source == source_code))
        counts = allocate_class_counts(len(indices))
        boundaries = np.cumsum(counts)[:-1]
        for code, part in enumerate(np.split(indices, boundaries)):
            split_code[part] = code
        small = [f"{name}={int(c)}" for name, c in zip(SPLIT_NAMES, counts) if c < warning_floor]
        if small:
            message = f"{source_label}: small partition(s) {', '.join(small)}"
            warnings_list.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    if (split_code[kept] < 0).any() or (split_code[~kept] >= 0).any():
        raise AssertionError("split assignment does not partition exactly the deduplicated rows")
    return split_code, warnings_list


def parse_class_row_targets(value: str) -> dict[str, int]:
    """Parse ``Class=rows,...``; every class must be a final category, rows >= 3 (one per split)."""
    targets: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, _, rows = item.partition("=")
        name = name.strip()
        if name not in CATEGORY_TO_ID:
            raise ValueError(f"unknown class {name!r} in class row targets; valid={list(CATEGORY_NAMES)}")
        if name in targets:
            raise ValueError(f"class {name!r} listed twice in class row targets")
        total = int(rows)
        if total < len(SPLIT_NAMES):
            raise ValueError(f"{name}: target {total} must be >= {len(SPLIT_NAMES)} (one row per split)")
        targets[name] = total
    if not targets:
        raise ValueError("class row targets must name at least one class")
    return targets


def split_row_targets(total: int) -> dict[str, int]:
    """Apportion a class's total row target 70/15/15 (largest remainder, >= 1 per split)."""
    return dict(zip(SPLIT_NAMES, (int(v) for v in allocate_class_counts(total))))


def allocate_stratum_quotas(sizes: np.ndarray, target: int, *, minimum_one: bool = True) -> np.ndarray:
    """Proportional quotas summing exactly to ``min(target, sizes.sum())``.

    1. Proportional target ``t_h = target * n_h / N`` per stratum; floor it and hand the leftover
       rows to the largest fractional remainders (ties: earlier stratum). This is Hamilton
       apportionment.
    2. When ``target >= #strata``, a stratum whose quota rounded to zero receives one row, taken
       from the stratum with the largest surplus ``quota - t_h`` that keeps at least one row
       (ties: larger quota, then earlier stratum). Short time bins are therefore never silently
       dropped, and allocations stay exactly proportional when no floor is needed.

    Capacity: when ``target < N`` every ``t_h < n_h``, so ``ceil(t_h) <= n_h`` and no stratum is
    ever asked for more rows than it has; step 2 only moves rows from strata keeping >= 1 row to
    strata holding >= 1 row. No quota is therefore ever left unused, and no capacity
    redistribution is needed. The invariant ``quota <= n_h`` is asserted. When ``target >= N``
    every stratum is kept whole.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if (sizes <= 0).any():
        raise ValueError("strata must be non-empty")
    if target < 0:
        raise ValueError("target must be non-negative")
    total = int(sizes.sum())
    if total <= target:
        return sizes.copy()
    proportional = target * sizes.astype(np.float64) / total
    quotas = np.floor(proportional).astype(np.int64)
    leftover = target - int(quotas.sum())
    priority = np.argsort(-(proportional - quotas), kind="stable")
    quotas[priority[:leftover]] += 1
    if minimum_one and target >= len(sizes):
        for starved in np.flatnonzero(quotas == 0):
            surplus = np.where(quotas > 1, quotas - proportional, -np.inf)
            best = surplus.max()
            donor = np.flatnonzero(surplus == best)
            donor = donor[np.argmax(quotas[donor])]
            quotas[donor] -= 1
            quotas[starved] += 1
    if int(quotas.sum()) != target or (quotas > sizes).any() or (quotas < 0).any():
        raise AssertionError("stratum quota allocation is inconsistent")
    return quotas


@dataclass
class StratifiedSample:
    selected: np.ndarray  # key-table indices, chronological
    stratum_of_candidate: np.ndarray  # aligned with candidates
    strata: pd.DataFrame
    audit: dict[str, Any]


def time_stratified_sample(
    keys: Keys,
    candidates: np.ndarray,
    target: int,
    stratum_hours: int,
    seed: int,
    split_name: str,
    category_label: str,
) -> StratifiedSample:
    """Deterministic time-stratified undersampling of one class inside one partition.

    Stratum = (source label, source file, fixed-width UTC bin), so the sample keeps each
    source label's share (e.g. DoS Hulk vs GoldenEye vs Slowloris) and its temporal spread.
    """
    split_index = SPLIT_CODE[split_name]
    class_index = CATEGORY_TO_ID[category_label]
    width_us = stratum_hours * US_PER_HOUR
    bins = keys.timestamp_us[candidates] // width_us
    files = keys.file_index[candidates].astype(np.int64)
    sources = keys.source[candidates].astype(np.int64)
    stratum_key = (sources << 48) + (files << 40) + bins
    unique_keys, inverse, sizes = np.unique(stratum_key, return_inverse=True, return_counts=True)
    quotas = allocate_stratum_quotas(sizes, target)
    hamilton = allocate_stratum_quotas(sizes, target, minimum_one=False)
    grouped = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    selected_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    population = int(sizes.sum())
    kept_total = min(target, population)
    for stratum, key in enumerate(unique_keys):
        source_index = int(key >> 48)
        file_index = int((key >> 40) & 0xFF)
        time_bin = int(key & ((1 << 40) - 1))
        members = candidates[grouped[starts[stratum] : starts[stratum] + sizes[stratum]]]
        quota = int(quotas[stratum])
        if quota == len(members):
            chosen = members
        else:
            rng = np.random.default_rng([seed, split_index, class_index, source_index, file_index, time_bin])
            chosen = members[np.sort(rng.choice(len(members), size=quota, replace=False))]
        selected_parts.append(chosen)
        proportional = target * sizes[stratum] / population if population > target else float(sizes[stratum])
        rows.append(
            {
                "split": split_name,
                "category_label": category_label,
                "source_label": SOURCE_LABELS[source_index],
                "stratum_id": stratum,
                "source_file": FILE_NAMES[file_index],
                "bin_start_utc": pd.Timestamp(time_bin * width_us, unit="us").isoformat() + "Z",
                "population": int(sizes[stratum]),
                "population_share": sizes[stratum] / population,
                "proportional_target": float(proportional),
                "quota": quota,
                "selected": int(len(chosen)),
                "selected_share": len(chosen) / max(kept_total, 1),
                "sampling_fraction": len(chosen) / sizes[stratum],
                "hamilton_quota": int(hamilton[stratum]),
                "minimum_quota_applied": bool(hamilton[stratum] == 0 and quota >= 1),
                "donated_to_minimum": int(hamilton[stratum] - quota) if hamilton[stratum] > quota else 0,
            }
        )
    selected = np.sort(np.concatenate(selected_parts)) if selected_parts else np.empty(0, np.int64)
    strata = pd.DataFrame(rows)
    if len(selected) != kept_total or len(np.unique(selected)) != len(selected):
        raise AssertionError(
            f"{split_name}/{category_label}: sample has {len(selected)} rows; expected {kept_total}"
        )
    if not np.isin(selected, candidates).all():
        raise AssertionError(f"{split_name}/{category_label}: sample contains a row outside its own pool")
    represented = int((strata["selected"] > 0).sum()) if len(strata) else 0
    if target >= len(strata) and represented != len(strata):
        raise AssertionError(
            f"{split_name}/{category_label}: a stratum vanished although the target allows one row each"
        )
    abs_share_error = (
        (strata["selected_share"] - strata["population_share"]).abs() if len(strata) else pd.Series(dtype=float)
    )
    audit = {
        "split": split_name,
        "category_label": category_label,
        "population": population,
        "target": target,
        "target_reached": population >= target,
        "selected": int(len(selected)),
        "removed": population - int(len(selected)),
        "sampling_fraction": len(selected) / population if population else 0.0,
        "strata_before": int(len(strata)),
        "strata_represented_after": represented,
        "strata_with_minimum_quota": int(strata["minimum_quota_applied"].sum()) if len(strata) else 0,
        "rows_moved_by_minimum_quota": int(strata["donated_to_minimum"].sum()) if len(strata) else 0,
        "fully_retained_strata": int((strata["selected"] == strata["population"]).sum()) if len(strata) else 0,
        "max_abs_share_deviation": float(abs_share_error.max()) if len(strata) else 0.0,
        "first_bin_utc": strata["bin_start_utc"].min() if len(strata) else None,
        "last_bin_utc": strata["bin_start_utc"].max() if len(strata) else None,
    }
    return StratifiedSample(selected, inverse.astype(np.int32), strata, audit)


# --------------------------------------------------------------------------- #
# Count tables
# --------------------------------------------------------------------------- #
def slot_counts_to_frame(rule_counts: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({rule: counts for rule, counts in rule_counts.items()})
    frame.insert(0, "original_label", LABEL_VOCAB)
    frame.insert(
        1,
        "source_label",
        [SOURCE_LABELS[s] if s >= 0 else "" for s in SLOT_SOURCE],
    )
    frame.insert(
        2,
        "category_label",
        [CATEGORY_NAMES[c] if c >= 0 else "(dropped)" for c in SLOT_CATEGORY],
    )
    frame = frame[frame["raw"] > 0].reset_index(drop=True)
    return frame


def count_by(values: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(values.astype(np.int64), minlength=size)[:size]


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def header_flag_columns(flags: np.ndarray) -> dict[str, pa.Array]:
    return {
        "fwd_header_length_negative": pa.array((flags & 1) > 0),
        "bwd_header_length_negative": pa.array((flags & 2) > 0),
        "fwd_header_length_overflow": pa.array((flags & 4) > 0),
        "bwd_header_length_overflow": pa.array((flags & 8) > 0),
    }


def dictionary_array(codes: np.ndarray, names: Iterable[str]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(codes.astype(np.int32)), pa.array(list(names), type=pa.string())
    )


def write_row_index(
    path: Path,
    keys: Keys,
    is_duplicate: np.ndarray,
    duplicate_of: np.ndarray,
    record_id: np.ndarray,
    split_code: np.ndarray,
    stratum: np.ndarray,
    in_output: np.ndarray,
    position: np.ndarray,
) -> None:
    split_names = (*SPLIT_NAMES, "duplicate_removed")
    split_index = np.where(split_code >= 0, split_code, len(SPLIT_NAMES))
    dup_rows = np.where(duplicate_of >= 0, duplicate_of, 0)
    table = pa.table(
        {
            "source_file": dictionary_array(keys.file_index, FILE_NAMES),
            "source_id": pa.array(keys.source_id),
            "timestamp_epoch_us": pa.array(keys.timestamp_us),
            "original_label": dictionary_array(keys.label_slot, LABEL_VOCAB),
            "attempted_category": pa.array(keys.attempted_category),
            "source_label": dictionary_array(keys.source, SOURCE_LABELS),
            "category_label": dictionary_array(keys.category, CATEGORY_NAMES),
            **header_flag_columns(keys.header_flags),
            "feature_hash_1": pa.array(keys.feature_hash_1),
            "feature_hash_2": pa.array(keys.feature_hash_2),
            "record_id": pa.array(record_id),
            "is_duplicate": pa.array(is_duplicate),
            "duplicate_of_source_file": pa.DictionaryArray.from_arrays(
                pa.array(keys.file_index[dup_rows].astype(np.int32), mask=~is_duplicate),
                pa.array(FILE_NAMES, type=pa.string()),
            ),
            "duplicate_of_source_id": pa.array(
                np.where(is_duplicate, keys.source_id[dup_rows], -1).astype(np.int32)
            ),
            "split": dictionary_array(split_index, split_names),
            "sampling_stratum": pa.array(stratum),
            "in_final_output": pa.array(in_output),
            "output_position": pa.array(position),
        }
    )
    pq.write_table(table, path, row_group_size=4_000_000, compression="zstd")


def write_split_parquet(
    path: Path,
    keys: Keys,
    indices: np.ndarray,
    strings: pa.Table,
    stratum: np.ndarray,
    record_id: np.ndarray,
    pristine: np.ndarray,
    features: list[str],
) -> None:
    writer: pq.ParquetWriter | None = None
    day_names = [name.split("-")[0] for name in FILE_NAMES]
    try:
        for start in range(0, len(indices), OUTPUT_CHUNK_ROWS):
            stop = min(start + OUTPUT_CHUNK_ROWS, len(indices))
            rows = indices[start:stop]
            file_index = keys.file_index[rows]
            source_id = keys.source_id[rows]
            ts = keys.timestamp_us[rows]
            category = keys.category[rows]
            file_names = dictionary_array(file_index, FILE_NAMES).cast(pa.string())
            columns: dict[str, Any] = {
                "sample_id": pc.binary_join_element_wise(
                    file_names, pc.cast(pa.array(source_id), pa.string()), ":"
                ),
                "record_id": pa.array(record_id[rows]),
                "source_file": file_names,
                "source_day": dictionary_array(file_index, day_names).cast(pa.string()),
                "source_day_order": pa.array(file_index.astype(np.int8)),
                "source_id": pa.array(source_id),
                "source_row": pa.array(source_id.astype(np.int64) + 1),
                FLOW_ID_COLUMN: strings.column(FLOW_ID_COLUMN).slice(start, stop - start),
                SRC_IP_COLUMN: strings.column(SRC_IP_COLUMN).slice(start, stop - start),
                DST_IP_COLUMN: strings.column(DST_IP_COLUMN).slice(start, stop - start),
                TIMESTAMP_COLUMN: pa.array(ts, type=pa.timestamp("us", tz="UTC")),
                "timestamp_epoch_seconds": pa.array(ts // US_PER_SECOND),
                "timestamp_epoch_us": pa.array(ts),
                "original_label": dictionary_array(keys.label_slot[rows], LABEL_VOCAB).cast(pa.string()),
                "attempted_category": pa.array(keys.attempted_category[rows]),
                "is_attempted": pa.array(SLOT_IS_ATTEMPTED[keys.label_slot[rows]].astype(np.uint8)),
                "source_label": dictionary_array(keys.source[rows], SOURCE_LABELS).cast(pa.string()),
                "category_label": dictionary_array(category, CATEGORY_NAMES).cast(pa.string()),
                "binary_label": pa.array((category != CATEGORY_TO_ID["Benign"]).astype(np.int8)),
                **header_flag_columns(keys.header_flags[rows]),
                "sampling_stratum": pa.array(stratum[rows]),
            }
            block = np.asarray(pristine[start:stop])
            for j, name in enumerate(features):
                columns[name] = pa.array(block[:, j])
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def output_fingerprints(
    pristine: np.ndarray, category: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent (pandas) 64-bit fingerprints of the saved float32 rows, with/without label."""
    feature_parts, labelled_parts = [], []
    for start in range(0, len(pristine), OUTPUT_CHUNK_ROWS):
        block = pd.DataFrame(np.asarray(pristine[start : start + OUTPUT_CHUNK_ROWS]))
        fingerprint = hash_pandas_object(block, index=False).to_numpy(dtype=np.uint64)
        feature_parts.append(fingerprint)
        labelled_parts.append(
            hash_pandas_object(
                pd.DataFrame({"f": fingerprint, "y": category[start : start + len(block)]}),
                index=False,
            ).to_numpy(dtype=np.uint64)
        )
    return np.concatenate(feature_parts), np.concatenate(labelled_parts)


def leakage_audit(
    keys: Keys,
    final_indices: dict[str, np.ndarray],
    split_code: np.ndarray,
    pristine: dict[str, np.ndarray],
) -> dict[str, Any]:
    category = keys.category
    fingerprints = {
        name: output_fingerprints(pristine[name], category[final_indices[name]])
        for name in SPLIT_NAMES
    }
    sample_key = {
        name: (keys.file_index[idx].astype(np.int64) << 32) | keys.source_id[idx].astype(np.int64)
        for name, idx in final_indices.items()
    }
    # Class-size reduction only removes rows, so it cannot create duplicates; this re-proves it on
    # the saved arrays (hash collisions are resolved byte-exactly).
    within_split: dict[str, int] = {}
    for name in SPLIT_NAMES:
        labelled = fingerprints[name][1]
        values, counts = np.unique(labelled, return_counts=True)
        exact = 0
        for fingerprint in values[counts > 1]:
            rows = np.flatnonzero(labelled == fingerprint)
            block = np.asarray(pristine[name][rows])
            labels = category[final_indices[name][rows]]
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    exact += int(np.array_equal(block[a], block[b]) and labels[a] == labels[b])
        within_split[name] = exact
        if exact:
            raise AssertionError(f"{name}: {exact} feature+label duplicate pairs inside the split")
    pairwise: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_ids = int(np.intersect1d(sample_key[left], sample_key[right]).size)
        (lf, ll), (rf, rl) = fingerprints[left], fingerprints[right]
        shared_labelled = np.intersect1d(np.unique(ll), np.unique(rl), assume_unique=True)
        exact_labelled = 0
        for fingerprint in shared_labelled:
            i = np.flatnonzero(ll == fingerprint)
            j = np.flatnonzero(rl == fingerprint)
            for a in i:
                for b in j:
                    same_x = np.array_equal(pristine[left][a], pristine[right][b])
                    same_y = category[final_indices[left][a]] == category[final_indices[right][b]]
                    exact_labelled += int(same_x and same_y)
        shared_features = np.intersect1d(np.unique(lf), np.unique(rf), assume_unique=True)
        examples = []
        for fingerprint in shared_features[:20]:
            a = int(np.flatnonzero(lf == fingerprint)[0])
            b = int(np.flatnonzero(rf == fingerprint)[0])
            ka, kb = final_indices[left][a], final_indices[right][b]
            examples.append(
                {
                    "exact_feature_match": bool(np.array_equal(pristine[left][a], pristine[right][b])),
                    left: {
                        "sample_id": f"{FILE_NAMES[keys.file_index[ka]]}:{keys.source_id[ka]}",
                        "original_label": LABEL_VOCAB[keys.label_slot[ka]],
                        "category_label": CATEGORY_NAMES[category[ka]],
                    },
                    right: {
                        "sample_id": f"{FILE_NAMES[keys.file_index[kb]]}:{keys.source_id[kb]}",
                        "original_label": LABEL_VOCAB[keys.label_slot[kb]],
                        "category_label": CATEGORY_NAMES[category[kb]],
                    },
                }
            )
        pairwise[f"{left}_vs_{right}"] = {
            "shared_sample_ids": shared_ids,
            "shared_feature_plus_label_fingerprints": int(len(shared_labelled)),
            "exact_feature_plus_label_duplicates": exact_labelled,
            "shared_feature_only_fingerprints": int(len(shared_features)),
            "feature_only_examples_first_20": examples,
        }
        if shared_ids:
            raise AssertionError(f"{left}/{right}: shared sample ids")
        if exact_labelled:
            raise AssertionError(f"{left}/{right}: feature+label duplicate crosses partitions")

    # Chronology within every source label, on the full (pre-cap) partition membership:
    # every train row precedes every val row, which precedes every test row, in the
    # (timestamp, file, id) order of the key table.
    source = keys.source
    chronology: dict[str, Any] = {}
    coverage: dict[str, dict[str, int]] = {}
    for code, label in enumerate(SOURCE_LABELS):
        members = {name: np.flatnonzero((split_code == SPLIT_CODE[name]) & (source == code)) for name in SPLIT_NAMES}
        in_output = {
            name: int(np.count_nonzero(source[final_indices[name]] == code)) for name in SPLIT_NAMES
        }
        coverage[label] = in_output
        if any(count == 0 for count in in_output.values()):
            raise AssertionError(f"source label {label} missing from a final partition: {in_output}")
        if not (members["train"].max() < members["val"].min() and members["val"].max() < members["test"].min()):
            raise AssertionError(f"within-source chronology failure for {label}")
        chronology[label] = {
            name: {
                "first_utc": pd.Timestamp(int(keys.timestamp_us[idx.min()]), unit="us").isoformat() + "Z",
                "last_utc": pd.Timestamp(int(keys.timestamp_us[idx.max()]), unit="us").isoformat() + "Z",
            }
            for name, idx in members.items()
        }
    return {
        "membership_is_disjoint": True,
        "exact_feature_plus_label_duplicates_within_split": within_split,
        "pairwise": pairwise,
        "source_label_coverage_final_outputs": coverage,
        "source_label_chronology_asserted": True,
        "chronology_order": "(timestamp_us, source_day_order, source_id); train < val < test strictly per source label",
        "source_label_chronology_utc": chronology,
        "fingerprint_method": (
            "pandas hash_pandas_object over the saved float32 rows (independent of the pass-1 "
            "hash); shared feature+label fingerprints are compared byte-exactly"
        ),
        "feature_only_overlap_note": (
            "Feature-only overlap can remain when an identical vector carries different categories; "
            "feature+category duplicates were removed globally before splitting."
        ),
    }


def scaler_roundtrip(scaler: RobustScaler, pristine: np.ndarray, scaled: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pristine), size=min(10_000, len(pristine)), replace=False))
    raw = np.asarray(pristine[idx])
    roundtrip = scaler.inverse_transform(np.asarray(scaled[idx]))
    error = np.abs(roundtrip - raw)
    tolerance = 1e-3 + 1e-5 * np.maximum(np.abs(raw), np.asarray(scaler.scale_, dtype=np.float32))
    if np.any(error > tolerance):
        raise AssertionError("scaler round-trip failed")
    return float(error.max())


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
CLASS_COLORS = {
    "Benign": "#7f7f7f",
    "DoS": "#d62728",
    "DDoS": "#ff7f0e",
    "Recon": "#1f77b4",
    "BruteForce": "#9467bd",
}


SPLIT_TITLES = {"train": "Train", "val": "Validation", "test": "Test"}
DESIGN_CAPTION = (
    "Class counts are an experimental design choice (fixed per-class row targets), "
    "not the natural CICIDS2018 class prevalence."
)


def plot_class_distribution_before_reduction(reduction: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.26
    x = np.arange(len(CATEGORY_NAMES))
    for k, name in enumerate(SPLIT_NAMES):
        values = reduction[reduction["split"] == name].set_index("category_label").loc[list(CATEGORY_NAMES), "before"]
        bars = ax.bar(x + (k - 1) * width, values, width, label=SPLIT_TITLES[name])
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], fontsize=6, rotation=90, padding=2)
    ax.set_xticks(x, CATEGORY_NAMES)
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    ax.set_title("Original retained class distribution per chronological split, BEFORE class-size reduction")
    ax.legend()
    ax.margins(y=0.25)
    natural = 100 * reduction.loc[reduction["category_label"] == "Benign", "before"].sum() / reduction["before"].sum()
    fig.text(
        0.5, 0.01,
        f"Natural post-cleaning, post-deduplication counts (CICIDS2018 prevalence: {natural:.1f}% Benign).",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_split_after_reduction(reduction: pd.DataFrame, split: str, output: Path) -> None:
    frame = reduction[reduction["split"] == split].set_index("category_label").loc[list(CATEGORY_NAMES)]
    colors = [CLASS_COLORS[name] for name in CATEGORY_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    counts = frame["after"]
    shares = 100 * counts / counts.sum()
    bars = axes[0].bar(CATEGORY_NAMES, counts, color=colors)
    axes[0].bar_label(bars, labels=[f"{v:,}" for v in counts], fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_title("rows (log scale)")
    bars = axes[1].bar(CATEGORY_NAMES, shares, color=colors)
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in shares], fontsize=7)
    axes[1].set_title("share of split (%)")
    for ax in axes:
        ax.margins(y=0.15)
    reduced = [label for label in CATEGORY_NAMES if frame.loc[label, "status"] == "reduced"]
    fig.suptitle(
        f"{SPLIT_TITLES[split]} class distribution AFTER class-size reduction "
        f"({int(counts.sum()):,} rows; reduced: {', '.join(reduced) or 'none'})"
    )
    fig.text(0.5, 0.01, DESIGN_CAPTION, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_sampling_strata(strata: pd.DataFrame, label: str, output: Path) -> None:
    strata = strata[strata["category_label"] == label]
    fig, axes = plt.subplots(len(SPLIT_NAMES), 1, figsize=(12, 9))
    for ax, name in zip(axes, SPLIT_NAMES):
        part = strata[strata["split"] == name]
        times = pd.to_datetime(part["bin_start_utc"].str.rstrip("Z"))
        ax.scatter(times, part["population"], s=8, label=f"{label} rows in split", color="#7f7f7f")
        ax.scatter(times, part["selected"], s=8, label="selected", color=CLASS_COLORS[label])
        ax.set_yscale("log")
        ax.set_ylabel("rows / stratum")
        represented = int((part["selected"] > 0).sum())
        ax.set_title(
            f"{SPLIT_TITLES[name]}: {len(part):,} strata (source label x source file x UTC hour), "
            f"{represented:,} represented after sampling",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(f"{label} temporal strata per split (own date axis per panel): population vs selected")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_spearman(sample: np.ndarray, features: list[str], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(sample, columns=features)
    corr = frame.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(15, 13))
    cmap = plt.get_cmap("coolwarm").with_extremes(bad="lightgrey")
    image = ax.imshow(np.ma.masked_invalid(corr.to_numpy()), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=6)
    ax.set_yticks(range(len(features)), features, fontsize=6)
    ax.set_title(
        f"Spearman correlation, uniform sample of {len(sample):,} final training rows "
        "(grey = constant in sample; descriptive only)"
    )
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return corr


def plot_pca(x_scaled: np.ndarray, y: np.ndarray, output: Path, seed: int) -> dict[str, Any]:
    features = np.arcsinh(x_scaled.astype(np.float64))
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 7))
    for code, name in enumerate(CATEGORY_NAMES):
        mask = y == code
        ax.scatter(projected[mask, 0], projected[mask, 1], s=3, alpha=0.4, color=CLASS_COLORS[name], label=f"{name} ({mask.sum():,})")
    ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)")
    ax.set_title("PCA of asinh(RobustScaler(x)) on a stratified training sample (descriptive only)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {"explained_variance_ratio": ratio.tolist(), "rows": int(len(y))}


# --------------------------------------------------------------------------- #
# CLI and orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
        help="CICIDS2017 preprocessing manifest whose modelling_feature_names fix the feature order.",
    )
    parser.add_argument("--expected-feature-count", type=int, default=EXPECTED_MODELLING_FEATURE_COUNT)
    parser.add_argument(
        "--class-row-targets",
        default=DEFAULT_CLASS_ROW_TARGETS,
        help=(
            "Comma-separated Class=TOTAL_ROWS. Each total is split 70/15/15 over train/val/test and "
            "sampled inside each split; classes not listed keep every row."
        ),
    )
    parser.add_argument(
        "--stratum-hours",
        type=int,
        default=DEFAULT_STRATUM_HOURS,
        help="Width of the UTC time bins that, crossed with source label and file, define sampling strata.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="1 = in-process.")
    parser.add_argument("--min-per-split-warning", type=int, default=DEFAULT_MIN_PER_SPLIT_WARNING)
    parser.add_argument("--spearman-sample", type=int, default=DEFAULT_SPEARMAN_SAMPLE)
    parser.add_argument("--pca-per-class", type=int, default=DEFAULT_PCA_PER_CLASS)
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Debug/smoke-test limit. Limited runs are marked non-production.",
    )
    parser.add_argument("--skip-input-hashes", action="store_true")
    args = parser.parse_args(argv)
    for name in ("stratum_hours", "workers", "min_per_split_warning", "spearman_sample", "pca_per_class"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1, got {getattr(args, name)}")
    args.class_row_targets = parse_class_row_targets(args.class_row_targets)
    return args


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    started = time.time()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    figures_dir = report_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features = load_reference_features(args.reference_manifest.resolve(), args.expected_feature_count)
    paths, raw_columns = validate_inventory(input_dir, features)
    dropped_columns = [c for c in raw_columns if c not in features]
    print(f"Validated {len(paths)} files: {len(raw_columns)} columns, {len(features)} modelling features")

    # ---- pass 1 -------------------------------------------------------------
    print("Pass 1: stream, clean, map labels, hash float32 feature vectors")
    t0 = time.time()
    scans: list[ScanResult] = run_parallel(
        scan_file,
        [
            ScanTask(path, index, raw_columns, features, args.max_rows_per_file, not args.skip_input_hashes)
            for index, path in enumerate(paths)
        ],
        args.workers,
    )
    pass1_seconds = time.time() - t0
    unknown = sum((scan.unknown_labels for scan in scans), Counter())
    if unknown:
        raise ValueError(f"labels without an explicit mapping/drop policy: {dict(unknown)}")
    rule_counts = {
        rule: sum(scan.rule_counts[rule] for scan in scans) for rule in scans[0].rule_counts
    }
    raw_rows = sum(scan.raw_rows for scan in scans)
    ts_failures = sum(scan.timestamp["missing"] + scan.timestamp["unparseable"] for scan in scans)
    if ts_failures > MAX_TIMESTAMP_FAILURE_FRACTION * raw_rows:
        raise ValueError(f"timestamp parsing failed on {ts_failures:,} of {raw_rows:,} rows")
    by_label = slot_counts_to_frame(rule_counts)
    removed_total = {rule: int(rule_counts[rule].sum()) for rule, _ in CLEANING_RULES}
    if int(rule_counts["raw"].sum()) != int(rule_counts["kept"].sum()) + sum(removed_total.values()):
        raise AssertionError("cleaning row counts do not reconcile")

    keys = assemble_keys(scans)
    assert_chronological(keys)
    n_clean = len(keys)
    print(f"  {raw_rows:,} raw rows -> {n_clean:,} cleaned label-retained rows ({pass1_seconds:,.0f} s)")

    # ---- deduplication --------------------------------------------------------
    is_duplicate, duplicate_of = deduplicate(keys)
    kept = ~is_duplicate
    record_id = np.where(kept, np.cumsum(kept) - 1, -1).astype(np.int64)
    conflicts = label_conflicts(keys, kept)
    n_dedup = int(kept.sum())
    print(f"  duplicates removed: {int(is_duplicate.sum()):,}; rows after dedup: {n_dedup:,}")

    # ---- chronological split within source label -----------------------------
    split_code, small_warnings = split_within_source_label(keys, kept, args.min_per_split_warning)

    # ---- class-size reduction, independently inside each split ---------------
    # Split membership is final at this point; sampling only removes rows of a targeted class
    # inside one split, using that split's own rows, so no row can move between partitions.
    category = keys.category
    source = keys.source
    benign_id = CATEGORY_TO_ID["Benign"]
    class_targets = args.class_row_targets
    per_split_targets = {label: split_row_targets(total) for label, total in class_targets.items()}
    stratum = np.full(n_clean, -1, dtype=np.int32)
    in_output = split_code >= 0
    samples: dict[tuple[str, str], StratifiedSample] = {}
    for name, code in SPLIT_CODE.items():
        members = split_code == code
        for label, targets in per_split_targets.items():
            candidates = np.flatnonzero(members & (category == CATEGORY_TO_ID[label]))
            target = targets[name]
            sample = time_stratified_sample(
                keys, candidates, target, args.stratum_hours, args.seed, name, label
            )
            samples[(name, label)] = sample
            stratum[candidates] = sample.stratum_of_candidate
            in_output[candidates] = False
            in_output[sample.selected] = True
            if not sample.audit["target_reached"]:
                message = (
                    f"{name}/{label}: only {len(candidates):,} rows available for a target of "
                    f"{target:,}; all kept"
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                sample.audit["shortfall_warning"] = message
    final_indices = {
        name: np.flatnonzero(in_output & (split_code == SPLIT_CODE[name])) for name in SPLIT_NAMES
    }
    position = np.full(n_clean, -1, dtype=np.int64)
    for name, idx in final_indices.items():
        position[idx] = np.arange(len(idx))

    # Invariants of the reduction stage.
    targeted_ids = {CATEGORY_TO_ID[label] for label in class_targets}
    reduction_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    source_reduction_rows: list[dict[str, Any]] = []
    for name, code in SPLIT_CODE.items():
        before = count_by(category[split_code == code], len(CATEGORY_NAMES))
        after = count_by(category[final_indices[name]], len(CATEGORY_NAMES))
        for c, label in enumerate(CATEGORY_NAMES):
            if c not in targeted_ids and before[c] != after[c]:
                raise AssertionError(f"{name}/{label}: {before[c] - after[c]:,} rows of an untargeted class removed")
            if c in targeted_ids:
                audit = samples[(name, label)].audit
                if after[c] != min(audit["population"], audit["target"]):
                    raise AssertionError(f"{name}/{label}: {after[c]:,} rows != min(available, target)")
            reduction_rows.append(
                {
                    "split": name,
                    "category_label": label,
                    "before": int(before[c]),
                    "target": per_split_targets[label][name] if label in per_split_targets else None,
                    "after": int(after[c]),
                    "removed": int(before[c] - after[c]),
                    "status": "reduced" if after[c] != before[c] else "unchanged",
                }
            )
        if (split_code[final_indices[name]] != code).any() or not in_output[final_indices[name]].all():
            raise AssertionError(f"{name}: a row moved between partitions")
        total = int(after.sum())
        composition_rows.append(
            {
                "split": name,
                "total_rows": total,
                "benign_rows": int(after[benign_id]),
                "attack_rows": int(total - after[benign_id]),
                **{f"{label}_rows": int(after[c]) for c, label in enumerate(CATEGORY_NAMES)},
                **{f"{label}_pct": 100.0 * after[c] / total for c, label in enumerate(CATEGORY_NAMES)},
            }
        )
        for s, source_label in enumerate(SOURCE_LABELS):
            source_reduction_rows.append(
                {
                    "split": name,
                    "source_label": source_label,
                    "category_label": SOURCE_TO_CATEGORY[source_label],
                    "before": int(np.count_nonzero((split_code == code) & (source == s))),
                    "after": int(np.count_nonzero(source[final_indices[name]] == s)),
                }
            )
    for label, total in class_targets.items():
        achieved = sum(
            int(np.count_nonzero(category[final_indices[name]] == CATEGORY_TO_ID[label])) for name in SPLIT_NAMES
        )
        available = int(np.count_nonzero(kept & (category == CATEGORY_TO_ID[label])))
        if achieved != min(total, available) and all(
            samples[(name, label)].audit["target_reached"] for name in SPLIT_NAMES
        ):
            raise AssertionError(f"{label}: total {achieved:,} != target {total:,}")
    reduction = pd.DataFrame(reduction_rows)
    composition = pd.DataFrame(composition_rows)
    source_reduction = pd.DataFrame(source_reduction_rows)
    untargeted = np.isin(category, list(targeted_ids), invert=True)
    if not in_output[(split_code >= 0) & untargeted].all():
        raise AssertionError("an individual row of an untargeted class was dropped")
    for (name, label), sample in samples.items():
        members = split_code == SPLIT_CODE[name]
        if (source[sample.selected] < 0).any():
            raise AssertionError(f"{name}/{label}: sampled row without a source label")
        for s, source_label in enumerate(SOURCE_LABELS):
            if SOURCE_TO_CATEGORY[source_label] != label:
                continue
            available = np.count_nonzero(members & (source == s))
            selected = np.count_nonzero(source[sample.selected] == s)
            if available and not selected and sample.audit["target"] >= sample.audit["strata_before"]:
                raise AssertionError(f"{name}/{source_label}: source label vanished during sampling")
    if (category[in_output] < 0).any():
        raise AssertionError("a row outside the five final categories reached the outputs")
    attempted_rows: list[dict[str, Any]] = []
    for name, code in SPLIT_CODE.items():
        in_split = split_code == code
        for slot in np.flatnonzero(SLOT_IS_ATTEMPTED):
            before_n = int(np.count_nonzero(in_split & (keys.label_slot == slot)))
            if before_n:
                attempted_rows.append(
                    {
                        "split": name,
                        "original_label": LABEL_VOCAB[slot],
                        "before": before_n,
                        "after": int(np.count_nonzero(keys.label_slot[final_indices[name]] == slot)),
                    }
                )
    attempted = pd.DataFrame(attempted_rows, columns=["split", "original_label", "before", "after"])
    print(composition[["split", "total_rows", *(f"{c}_rows" for c in CATEGORY_NAMES)]].to_string(index=False))

    # ---- pass 2 ---------------------------------------------------------------
    print("Pass 2: extract selected rows into float32 arrays")
    pristine_paths = {code: output_dir / f"X_{name}_pristine.npy" for name, code in SPLIT_CODE.items()}
    for name, code in SPLIT_CODE.items():
        array = np.lib.format.open_memmap(
            pristine_paths[code], mode="w+", dtype=np.float32, shape=(len(final_indices[name]), len(features))
        )
        array[:] = np.nan
        array.flush()
        del array
    extract_tasks = []
    for file_index, path in enumerate(paths):
        rows = np.flatnonzero(in_output & (keys.file_index == file_index))
        rows = rows[np.argsort(keys.source_id[rows], kind="stable")]
        extract_tasks.append(
            ExtractTask(
                path=path,
                file_index=file_index,
                raw_columns=raw_columns,
                features=features,
                max_rows=args.max_rows_per_file,
                source_id=keys.source_id[rows],
                split_code=split_code[rows],
                position=position[rows],
                feature_hash_1=keys.feature_hash_1[rows],
                feature_hash_2=keys.feature_hash_2[rows],
                pristine_paths={code: str(p) for code, p in pristine_paths.items()},
            )
        )
    t0 = time.time()
    extracted: list[ExtractResult] = run_parallel(extract_file, extract_tasks, args.workers)
    pass2_seconds = time.time() - t0
    strings: dict[str, pa.Table] = {}
    for name, code in SPLIT_CODE.items():
        table = pa.concat_tables([r.strings[code] for r in extracted if code in r.strings])
        order = pc.sort_indices(table, sort_keys=[("position", "ascending")])
        table = table.take(order).combine_chunks()
        if not np.array_equal(
            table.column("position").to_numpy(), np.arange(len(final_indices[name]))
        ):
            raise AssertionError(f"{name}: extracted positions are not a complete permutation")
        strings[name] = table
    pristine = {
        name: np.load(pristine_paths[code], mmap_mode="r") for name, code in SPLIT_CODE.items()
    }
    header_positions = [features.index(name) for name in HEADER_LENGTH_COLUMNS]
    other_positions = [j for j in range(len(features)) if j not in header_positions]
    for name, array in pristine.items():
        for start in range(0, len(array), OUTPUT_CHUNK_ROWS):
            block = np.asarray(array[start : start + OUTPUT_CHUNK_ROWS])
            if not np.isfinite(block).all():
                raise AssertionError(f"{name}: NaN/Inf in pristine output (unfilled row?)")
            if (block[:, other_positions] < 0).any():
                raise AssertionError(f"{name}: negative value outside the header-length exemption")
    print(f"  extracted in {pass2_seconds:,.0f} s")

    # ---- train-only transform -------------------------------------------------
    train_matrix = np.ascontiguousarray(np.load(pristine_paths[SPLIT_CODE["train"]]))
    train_constant = [
        name for j, name in enumerate(features) if train_matrix[:, j].min() == train_matrix[:, j].max()
    ]
    scaler = RobustScaler(copy=True).fit(train_matrix)
    del train_matrix
    with (output_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(
        output_dir / "scaler_parameters.json",
        {
            "class": "sklearn.preprocessing.RobustScaler",
            "params": scaler.get_params(),
            "fit_split": "train (after per-split Benign reduction)",
            "fit_rows": int(len(final_indices["train"])),
            "feature_names": features,
            "center_median": dict(zip(features, scaler.center_.tolist())),
            "scale_iqr": dict(zip(features, scaler.scale_.tolist())),
            "model_side_transform": "asinh applied inside victims/VAE (input_transform='asinh'), not here",
        },
    )
    split_reports: dict[str, Any] = {}
    for name, code in SPLIT_CODE.items():
        scaled = np.lib.format.open_memmap(
            output_dir / f"X_{name}.npy", mode="w+", dtype=np.float32, shape=pristine[name].shape
        )
        for start in range(0, len(scaled), OUTPUT_CHUNK_ROWS):
            block = scaler.transform(np.asarray(pristine[name][start : start + OUTPUT_CHUNK_ROWS]))
            if not np.isfinite(block).all():
                raise AssertionError(f"{name}: scaled output contains NaN/Inf")
            scaled[start : start + len(block)] = block
        scaled.flush()
        idx = final_indices[name]
        y_category = category[idx].astype(np.int8)
        y_binary = (y_category != benign_id).astype(np.int8)
        np.save(output_dir / f"y_{name}_cat.npy", y_category)
        np.save(output_dir / f"y_{name}_bin.npy", y_binary)
        np.save(output_dir / f"timestamp_epoch_seconds_{name}.npy", keys.timestamp_us[idx] // US_PER_SECOND)
        np.save(output_dir / f"timestamp_epoch_us_{name}.npy", keys.timestamp_us[idx])
        max_error = scaler_roundtrip(scaler, pristine[name], scaled, args.seed)
        del scaled
        write_split_parquet(
            output_dir / f"{name}.parquet", keys, idx, strings[name], stratum, record_id,
            pristine[name], features,
        )
        split_reports[name] = {
            "rows": int(len(idx)),
            "shape": [int(len(idx)), len(features)],
            "timestamp_min_utc": pd.Timestamp(int(keys.timestamp_us[idx].min()), unit="us").isoformat() + "Z",
            "timestamp_max_utc": pd.Timestamp(int(keys.timestamp_us[idx].max()), unit="us").isoformat() + "Z",
            "category_label_counts": {
                CATEGORY_NAMES[c]: int(v) for c, v in enumerate(count_by(y_category, len(CATEGORY_NAMES)))
            },
            "binary_counts": {"benign": int((y_binary == 0).sum()), "attack": int((y_binary == 1).sum())},
            "header_length_negative_rows": int(((keys.header_flags[idx] & HEADER_NEGATIVE_BITS) > 0).sum()),
            "header_length_overflow_rows": int(((keys.header_flags[idx] & HEADER_OVERFLOW_BITS) > 0).sum()),
            "max_scaler_roundtrip_abs_error": max_error,
        }
        print(f"  wrote {name}: {split_reports[name]['shape']}")

    y_train = category[final_indices["train"]].astype(np.int64)
    category_weights = balanced_class_weights(y_train, len(CATEGORY_NAMES))
    binary_weights = balanced_class_weights((y_train != benign_id).astype(np.int64), 2)
    np.save(output_dir / "class_weights_5.npy", category_weights)
    np.save(output_dir / "class_weights_2.npy", binary_weights)
    label_encoders = {"binary": {"Benign": 0, "Attack": 1}, "category": CATEGORY_TO_ID}
    _write_json(output_dir / "label_encoders.json", label_encoders)

    # ---- audits ---------------------------------------------------------------
    print("Auditing leakage, chronology and coverage")
    leakage = leakage_audit(keys, final_indices, split_code, pristine)
    del pristine
    _write_json(output_dir / "leakage_audit.json", leakage)

    n_cat = len(CATEGORY_NAMES)
    stage_counts: dict[str, np.ndarray] = {}
    for rule in ("raw", "nonfinite", "bad_timestamp", "negative_non_header", "kept", "header_negative_any",
                 "header_overflow_any"):
        vec = np.zeros(n_cat, np.int64)
        mapped_slots = np.flatnonzero(SLOT_CATEGORY >= 0)
        np.add.at(vec, SLOT_CATEGORY[mapped_slots], rule_counts[rule][mapped_slots])
        stage_counts[rule] = vec
    raw_by_cat = stage_counts["raw"]
    duplicates_by_cat = count_by(category[is_duplicate], n_cat)
    class_rows = []
    for c, name in enumerate(CATEGORY_NAMES):
        before = {
            name: int(np.count_nonzero((split_code == code) & (category == c))) for name, code in SPLIT_CODE.items()
        }
        row = {
            "category_label": name,
            "raw_retained": int(raw_by_cat[c]),
            "removed_nonfinite": int(stage_counts["nonfinite"][c]),
            "removed_bad_timestamp": int(stage_counts["bad_timestamp"][c]),
            "removed_negative_non_header": int(stage_counts["negative_non_header"][c]),
            "after_cleaning": int(stage_counts["kept"][c]),
            "header_length_negative_kept": int(stage_counts["header_negative_any"][c]),
            "header_length_overflow_kept": int(stage_counts["header_overflow_any"][c]),
            "duplicates_removed": int(duplicates_by_cat[c]),
            "total": int(np.count_nonzero(kept & (category == c))),
            **{f"{split}_before_reduction": before[split] for split in SPLIT_NAMES},
            "target_total": class_targets.get(name),
            "train": int(np.count_nonzero(category[final_indices["train"]] == c)),
            "val": int(np.count_nonzero(category[final_indices["val"]] == c)),
            "test": int(np.count_nonzero(category[final_indices["test"]] == c)),
        }
        row["final_total"] = row["train"] + row["val"] + row["test"]
        for split in SPLIT_NAMES:
            row[f"{split}_removed_by_reduction"] = row[f"{split}_before_reduction"] - row[split]
        class_rows.append(row)
    class_df = pd.DataFrame(class_rows)
    for split in SPLIT_NAMES:
        for column in (split, f"{split}_before_reduction"):
            class_df[f"{column}_pct_of_split"] = 100.0 * class_df[column] / class_df[column].sum()
    removed_cols = [f"{split}_removed_by_reduction" for split in SPLIT_NAMES]
    untargeted_rows = ~class_df["category_label"].isin(list(class_targets))
    if (class_df.loc[untargeted_rows, removed_cols] != 0).any().any():
        raise AssertionError("rows of an untargeted class removed by class-size reduction")
    if (class_df.loc[untargeted_rows, "final_total"] != class_df.loc[untargeted_rows, "total"]).any():
        raise AssertionError("an untargeted class lost cleaned, deduplicated rows")

    source = keys.source
    source_rows = []
    for s, label in enumerate(SOURCE_LABELS):
        total = int(np.count_nonzero(kept & (source == s)))
        row = {
            "source_label": label,
            "category_label": SOURCE_TO_CATEGORY[label],
            "after_cleaning": int(np.count_nonzero(source == s)),
            "duplicates_removed": int(np.count_nonzero(is_duplicate & (source == s))),
            "total": total,
            **{
                f"{split}_before_reduction": int(np.count_nonzero((split_code == code) & (source == s)))
                for split, code in SPLIT_CODE.items()
            },
            "train": int(np.count_nonzero(source[final_indices["train"]] == s)),
            "val": int(np.count_nonzero(source[final_indices["val"]] == s)),
            "test": int(np.count_nonzero(source[final_indices["test"]] == s)),
        }
        for split in SPLIT_NAMES:
            row[f"{split}_before_reduction_pct_of_source"] = (
                100.0 * row[f"{split}_before_reduction"] / total
            )
        row["small_partition_warning"] = "; ".join(
            f"{split}={row[split]}" for split in SPLIT_NAMES if row[split] < args.min_per_split_warning
        )
        source_rows.append(row)
    source_df = pd.DataFrame(source_rows)
    class_df.to_csv(output_dir / "class_distribution.csv", index=False)
    source_df.to_csv(output_dir / "source_label_distribution.csv", index=False)
    by_label.to_csv(output_dir / "cleaning_by_original_label.csv", index=False)
    reduction.to_csv(output_dir / "class_reduction_by_split.csv", index=False)
    source_reduction.to_csv(output_dir / "source_label_reduction_by_split.csv", index=False)
    composition.to_csv(output_dir / "split_composition.csv", index=False)
    attempted.to_csv(output_dir / "attempted_benign_by_split.csv", index=False)

    header_cols = [c for c in by_label.columns if c.startswith(("fwd_header", "bwd_header", "header_"))]
    header_by_label = by_label[["original_label", "category_label", "kept", *header_cols]]
    header_by_label.to_csv(output_dir / "header_length_audit_by_label.csv", index=False)
    class_header_cols = [c for c in header_cols if c != "header_negative_any_clean_rows"]
    header_by_class = header_by_label[header_by_label["category_label"] != "(dropped)"].groupby(
        "category_label", sort=False
    )[["kept", *class_header_cols]].sum().reindex(list(CATEGORY_NAMES)).reset_index()
    for flag in ("header_negative_any", "header_overflow_any"):
        header_by_class[f"{flag}_pct"] = 100 * header_by_class[flag] / header_by_class["kept"]
    header_by_class.to_csv(output_dir / "header_length_audit_by_class.csv", index=False)
    header_by_split = {
        flag_name: {
            name: {
                CATEGORY_NAMES[c]: int(
                    np.count_nonzero(((keys.header_flags[idx] & bits) > 0) & (category[idx] == c))
                )
                for c in range(n_cat)
            }
            for name, idx in final_indices.items()
        }
        for flag_name, bits in (("negative", HEADER_NEGATIVE_BITS), ("overflow", HEADER_OVERFLOW_BITS))
    }

    dup_by_label = pd.DataFrame(
        {
            "original_label": [LABEL_VOCAB[s] for s in range(len(LABEL_VOCAB))],
            "after_cleaning": count_by(keys.label_slot, len(LABEL_VOCAB)),
            "duplicates_removed": count_by(keys.label_slot[is_duplicate], len(LABEL_VOCAB)),
        }
    )
    dup_by_label = dup_by_label[dup_by_label["after_cleaning"] > 0].reset_index(drop=True)
    dup_by_label["duplicate_pct"] = 100 * dup_by_label["duplicates_removed"] / dup_by_label["after_cleaning"]
    dup_by_label.to_csv(output_dir / "duplicates_by_original_label.csv", index=False)
    cross_file_dups = int(
        np.count_nonzero(is_duplicate & (keys.file_index != keys.file_index[np.maximum(duplicate_of, 0)]))
    )
    duplicate_audit = {
        "definition": "exact equality of every float32 modelling feature plus mapped category_label",
        "implementation": (
            "128-bit hash (two independent 64-bit mixers over the float32 bit patterns, -0.0 folded "
            "to +0.0) of features+category; expected false merges ~ n^2 / 2^129 (negligible)"
        ),
        "survivor": "earliest row in (timestamp, source_day_order, source_id) order",
        "stage": "global, after cleaning and label mapping, before splitting",
        "rows_before_duplicate_removal": n_clean,
        "exact_duplicates_removed": int(is_duplicate.sum()),
        "duplicate_percentage": 100.0 * int(is_duplicate.sum()) / n_clean,
        "rows_after_duplicate_removal": n_dedup,
        "duplicates_whose_survivor_is_in_another_file": cross_file_dups,
        "by_category": {CATEGORY_NAMES[c]: int(v) for c, v in enumerate(duplicates_by_cat)},
        "by_original_label": dict(zip(dup_by_label["original_label"], dup_by_label["duplicates_removed"].astype(int))),
        "provenance": "row_index.parquet: is_duplicate, duplicate_of_source_file, duplicate_of_source_id",
        "label_conflicts_after_dedup": conflicts,
    }
    _write_json(output_dir / "duplicate_audit.json", duplicate_audit)

    cleaning_audit = {
        "rules_in_order": [{"rule": rule, "description": text, "rows_removed": removed_total[rule]} for rule, text in CLEANING_RULES],
        "raw_rows": raw_rows,
        "rows_after_cleaning_and_label_filter": n_clean,
        "per_final_class": {
            row["category_label"]: {
                "raw_retained": row["raw_retained"],
                "removed_nonfinite": row["removed_nonfinite"],
                "removed_bad_timestamp": row["removed_bad_timestamp"],
                "removed_negative_non_header": row["removed_negative_non_header"],
                "after_cleaning": row["after_cleaning"],
                "header_length_negative_kept": row["header_length_negative_kept"],
                "header_length_overflow_kept": row["header_length_overflow_kept"],
            }
            for row in class_rows
        },
        "nonfinite_columns": dict(sum((Counter(scan.nonfinite_columns) for scan in scans), Counter())),
        "nonfinite_rows_first_500": [ex for scan in scans for ex in scan.nonfinite_examples][:500],
        "timestamp": {
            scan.name: {
                **{k: v for k, v in scan.timestamp.items() if not k.endswith("_us")},
                "min_utc": None if scan.timestamp["min_us"] is None else pd.Timestamp(scan.timestamp["min_us"], unit="us").isoformat() + "Z",
                "max_utc": None if scan.timestamp["max_us"] is None else pd.Timestamp(scan.timestamp["max_us"], unit="us").isoformat() + "Z",
            }
            for scan in scans
        },
        "label_decode_cp1252_fallback_rows": sum(scan.label_decode_fallback_rows for scan in scans),
        "header_length_policy": {
            "columns": list(HEADER_LENGTH_COLUMNS),
            "policy": (
                "retain rows and raw values; flag fwd/bwd_header_length_negative and "
                "fwd/bwd_header_length_overflow in metadata"
            ),
            "rationale": (
                "values lie in [-32768, 32767]: the extractor stores header bytes in a signed 16-bit "
                "field, so long flows wrap. The wrap count is unknown and cannot be undone. Sign is "
                "not a validity test: wrapped values land positive as often as negative, so dropping "
                "negatives would remove about half of DDoS-LOIC-UDP and still leave the other "
                "wrapped rows in place"
            ),
            "overflow_definition": (
                "packets x minimum transport header per packet (TCP 20 B, UDP 8 B) > 32767: the "
                "true header byte count cannot be represented, so the stored value is wrapped"
            ),
            "observed_range_kept_rows": {name: v for name, v in _merge_extrema(scans).items()},
            "by_class": header_by_class.to_dict(orient="records"),
            "rows_flagged_per_final_split": header_by_split,
            "below_protocol_minimum_note": (
                "nonnegative values below packets x minimum transport header are wrapped values "
                "that happen to land positive; descriptive only"
            ),
        },
        "icmp_sentinel_columns": "excluded with the other CICIDS2018-only columns",
        "imputation": None,
        "winsorization": None,
        "clipping": None,
    }
    _write_json(output_dir / "cleaning_audit.json", cleaning_audit)
    strata_all = pd.concat([sample.strata for sample in samples.values()], ignore_index=True)
    strata_all.to_csv(output_dir / "sampling_strata.csv", index=False)
    per_file = strata_all.groupby(["split", "category_label", "source_file"], sort=False)[
        ["population", "quota", "selected"]
    ].sum().reset_index()
    group = per_file.groupby(["split", "category_label"])
    per_file["population_share"] = per_file["population"] / group["population"].transform("sum")
    per_file["selected_share"] = per_file["selected"] / group["selected"].transform("sum")
    width_us = args.stratum_hours * US_PER_HOUR
    sampling_audit = {
        "method": "deterministic time-stratified undersampling without replacement, per split and class",
        "applies_to": "classes named in class_row_targets, inside each chronological partition (after the split)",
        "class_row_targets_total": class_targets,
        "class_row_targets_per_split": per_split_targets,
        "target_allocation": "total apportioned 70/15/15 by largest remainder (CICIDS2017 allocate_class_counts)",
        "untargeted_classes": [label for label in CATEGORY_NAMES if label not in class_targets],
        "shortfall_policy": "if a split holds fewer rows than its target, keep them all and report target_reached = false",
        "seed": args.seed,
        "stratum_definition": (
            f"split x class x source_label x source_file x {args.stratum_hours}-hour UTC bin "
            f"(floor(timestamp_us / {width_us}))"
        ),
        "per_stratum_rng": (
            "numpy default_rng([seed, split_index, class_index, source_index, file_index, time_bin]); "
            "Generator.choice(replace=False)"
        ),
        "allocation": (
            "Hamilton (largest-remainder) apportionment of the split target proportional to stratum "
            "population; if target >= #strata, each stratum rounded to zero gets one row, taken from the "
            "stratum with the largest surplus over its exact proportional target; quota <= stratum size is "
            "guaranteed (proportional targets are below stratum sizes) and asserted"
        ),
        "output_order": "chronological (timestamp, file order, id)",
        "samples": {f"{name}/{label}": sample.audit for (name, label), sample in samples.items()},
        "per_file": per_file.to_dict(orient="records"),
    }
    _write_json(output_dir / "sampling_audit.json", sampling_audit)

    print("Writing row index")
    write_row_index(
        output_dir / "row_index.parquet", keys, is_duplicate, duplicate_of, record_id,
        split_code, stratum, in_output, position,
    )

    # ---- figures (descriptive only; nothing here feeds a fitted step) --------
    print("Figures")
    for stale in figures_dir.glob("*.png"):
        stale.unlink()
    plot_class_distribution_before_reduction(reduction, figures_dir / "class_distribution_before_reduction.png")
    for name in SPLIT_NAMES:
        plot_split_after_reduction(reduction, name, figures_dir / f"class_distribution_{name}_after_reduction.png")
    for label in class_targets:
        plot_sampling_strata(strata_all, label, figures_dir / f"sampling_strata_{label}.png")
    rng = np.random.default_rng(args.seed)
    x_train_pristine = np.load(output_dir / "X_train_pristine.npy", mmap_mode="r")
    spearman_idx = np.sort(rng.choice(len(x_train_pristine), size=min(args.spearman_sample, len(x_train_pristine)), replace=False))
    corr = plot_spearman(np.asarray(x_train_pristine[spearman_idx]), features, figures_dir / "spearman_train_sample.png")
    corr.to_csv(report_dir / "spearman_train_sample.csv")
    x_train_scaled = np.load(output_dir / "X_train.npy", mmap_mode="r")
    y_train_small = category[final_indices["train"]]
    pca_idx = []
    for c in range(n_cat):
        members = np.flatnonzero(y_train_small == c)
        take = min(args.pca_per_class, len(members))
        pca_idx.append(np.sort(np.random.default_rng([args.seed, c]).choice(members, size=take, replace=False)))
    pca_idx = np.sort(np.concatenate(pca_idx))
    pca_info = plot_pca(np.asarray(x_train_scaled[pca_idx]), y_train_small[pca_idx], figures_dir / "pca_train_stratified.png", args.seed)
    del x_train_pristine, x_train_scaled

    input_hashes = None if args.skip_input_hashes else {scan.name: scan.sha256 for scan in scans}
    outputs = {
        "processed_dir": str(output_dir),
        "report_dir": str(report_dir),
        "arrays": {
            name: {
                "X": str(output_dir / f"X_{name}.npy"),
                "X_pristine": str(output_dir / f"X_{name}_pristine.npy"),
                "y_cat": str(output_dir / f"y_{name}_cat.npy"),
                "y_bin": str(output_dir / f"y_{name}_bin.npy"),
                "timestamp_epoch_seconds": str(output_dir / f"timestamp_epoch_seconds_{name}.npy"),
                "timestamp_epoch_us": str(output_dir / f"timestamp_epoch_us_{name}.npy"),
                "parquet": str(output_dir / f"{name}.parquet"),
            }
            for name in SPLIT_NAMES
        },
        "row_index": str(output_dir / "row_index.parquet"),
        "scaler": str(output_dir / "scaler.pkl"),
        "scaler_parameters": str(output_dir / "scaler_parameters.json"),
        "label_encoders": str(output_dir / "label_encoders.json"),
        "class_weights": [str(output_dir / "class_weights_5.npy"), str(output_dir / "class_weights_2.npy")],
        "tables": [
            str(output_dir / name)
            for name in (
                "class_distribution.csv", "source_label_distribution.csv", "cleaning_by_original_label.csv",
                "header_length_audit_by_label.csv", "header_length_audit_by_class.csv",
                "duplicates_by_original_label.csv", "sampling_strata.csv", "class_reduction_by_split.csv",
                "source_label_reduction_by_split.csv", "split_composition.csv", "attempted_benign_by_split.csv",
            )
        ],
        "audits": [
            str(output_dir / name)
            for name in ("cleaning_audit.json", "duplicate_audit.json", "sampling_audit.json", "leakage_audit.json")
        ],
        "figures": sorted(str(p) for p in figures_dir.glob("*.png")),
        "report": str(report_dir / "cicids2018_distrinet_preprocessing_report.md"),
    }
    manifest = {
        "dataset": "corrected/relabelled DistriNet CSE-CIC-IDS-2018 ten-file release",
        "methodological_description": (
            "Rows are mapped to five categories and filtered before a leakage-controlled chronological "
            "70/15/15 split within each retained source label. Exact float32 feature+category duplicates "
            "are removed globally before splitting. After the split, "
            + ", ".join(f"{label} to {total:,}" for label, total in class_targets.items())
            + " total rows were deterministically undersampled (each total apportioned 70/15/15 and "
            "sampled independently inside train, validation and test, stratified by source label, source "
            "file and UTC hour). "
            + ", ".join(label for label in CATEGORY_NAMES if label not in class_targets)
            + " keep every row. Every fitted statistic uses the final training rows only."
        ),
        "evaluation_design": (
            "controlled: the evaluation partitions do not reproduce the original CICIDS2018 class prior "
            "and should be interpreted as controlled experimental datasets rather than estimates of "
            "real-world attack prevalence"
        ),
        "not_claimed": [
            "global forward-time or independent attack-campaign generalization",
            "that the undersampled classes preserve their complete original distributions",
            "natural CICIDS2018 class prevalence in any partition",
            "calibrated real-world probabilities from softmax outputs",
        ],
        "research_target": "binary and five-category closed-set classification",
        "compatible_with": "CICIDS2017 DistriNet processed layout (data/processed/CICIDS_2017_Distrinet)",
        "seed": args.seed,
        "non_production": args.max_rows_per_file is not None,
        "max_rows_per_file": args.max_rows_per_file,
        "input_dir": str(input_dir),
        "input_files": [p.name for p in paths],
        "input_sha256": input_hashes,
        "input_raw_rows": {scan.name: scan.raw_rows for scan in scans},
        "raw_columns": raw_columns,
        "raw_column_count": len(raw_columns),
        "reference_feature_manifest": str(args.reference_manifest.resolve()),
        "modelling_feature_names": features,
        "modelling_feature_count": len(features),
        "dropped_columns": {
            "non_feature_metadata": list(NON_FEATURE_COLUMNS),
            "cicids2018_only_features": list(CICIDS2018_ONLY_COLUMNS),
            "all": dropped_columns,
        },
        "metadata_excluded_from_X": [
            "sample_id", "record_id", "source_file", "source_day", "source_day_order", "source_id",
            "source_row", "Flow ID", "Src IP", "Dst IP", "Timestamp", "timestamp_epoch_seconds",
            "timestamp_epoch_us", "original_label", "attempted_category", "is_attempted", "source_label",
            "category_label", "binary_label", "fwd_header_length_negative", "bwd_header_length_negative",
            "fwd_header_length_overflow", "bwd_header_length_overflow",
            "sampling_stratum",
        ],
        "timestamp_policy": {
            "format": "ISO-8601 'YYYY-MM-DD HH:MM:SS.ffffff'",
            "timezone": "UTC per DistriNet documentation; strings are timezone-naive",
            "sort_key": ["timestamp_us", "source_day_order", "source_id"],
            "file_name_date_used_for_chronology": False,
            "classifier_input": False,
            "max_failure_fraction": MAX_TIMESTAMP_FAILURE_FRACTION,
        },
        "label_policy": {
            "heads": ["binary", "category"],
            "source_to_category": SOURCE_TO_CATEGORY,
            "attempted_policy": "benign: every '<attack> - Attempted' label -> source_label BENIGN",
            "attempted_labels": list(ATTEMPTED_LABELS),
            "dropped_labels": list(DROPPED_LABELS),
            "unknown_label_policy": "abort",
            "retained_categories": list(CATEGORY_NAMES),
            "category_to_id": CATEGORY_TO_ID,
            "binary_to_id": {"Benign": 0, "Attack": 1},
            "mapping_and_filter_stage": "before split membership assignment",
        },
        "cleaning_policy": {
            "rules_in_order": cleaning_audit["rules_in_order"],
            "header_length_negative": "kept and flagged (see cleaning_audit.json)",
            "fitted_statistics_used_before_split": False,
            "imputation": None,
            "winsorization": None,
        },
        "duplicate_audit": {k: v for k, v in duplicate_audit.items() if k != "label_conflicts_after_dedup"},
        "split_policy": {
            "protocol": "chronological within each retained source label (after Attempted->BENIGN mapping)",
            "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS.tolist())),
            "allocation": "largest remainder with at least one row per split (CICIDS2017 allocate_class_counts)",
            "shuffle_before_assignment": False,
            "retained_source_labels": list(SOURCE_LABELS),
            "small_partition_warning_floor": args.min_per_split_warning,
            "small_source_label_warnings": small_warnings,
        },
        "class_size_reduction": sampling_audit,
        "class_reduction_by_split": reduction.to_dict(orient="records"),
        "source_label_reduction_by_split": source_reduction.to_dict(orient="records"),
        "split_composition": composition.to_dict(orient="records"),
        "attempted_benign_by_split": attempted.to_dict(orient="records"),
        "class_distribution": class_df.to_dict(orient="records"),
        "source_label_distribution": source_df.to_dict(orient="records"),
        "leakage_audit_summary": {
            k: v for k, v in leakage.items() if k != "pairwise"
        } | {"pairwise": {k: {kk: vv for kk, vv in v.items() if kk != "feature_only_examples_first_20"} for k, v in leakage["pairwise"].items()}},
        "train_only_preprocessing": {
            "scaler": "sklearn.preprocessing.RobustScaler (median/IQR, quantile_range=(25, 75))",
            "scaler_fit_rows": int(len(final_indices["train"])),
            "scaler_fit_split": "train after per-split class-size reduction (validation/test never used)",
            "model_side_input_transform": "asinh inside victims/VAE, as for CICIDS2017",
            "train_constant_columns_reported_not_removed": train_constant,
            "class_weights": {
                "binary": binary_weights.tolist(),
                "category": category_weights.tolist(),
                "formula": "n_samples / (n_classes * class_count), final train labels only",
            },
        },
        "descriptive_only": {"pca": pca_info, "spearman_sample_rows": int(len(spearman_idx))},
        "split_reports": split_reports,
        "outputs": outputs,
        "timing_seconds": {"pass1": round(pass1_seconds, 1), "pass2": round(pass2_seconds, 1)},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(output_dir / "preprocessing_manifest.json", manifest)
    write_report(report_dir / "cicids2018_distrinet_preprocessing_report.md", manifest, class_df, source_df,
                 header_by_class, per_file, reduction, source_reduction, composition, attempted,
                 sampling_audit, cleaning_audit, duplicate_audit, leakage)

    print(f"\nDone in {manifest['elapsed_seconds']:,.0f} s -> {output_dir}")
    print(composition.to_string(index=False))
    return manifest


def _merge_extrema(scans: list[ScanResult]) -> dict[str, list[float]]:
    merged = {name: [np.inf, -np.inf] for name in HEADER_LENGTH_COLUMNS}
    for scan in scans:
        for name, (lo, hi) in scan.header_extrema.items():
            merged[name] = [min(merged[name][0], lo), max(merged[name][1], hi)]
    return merged


def write_report(
    path: Path,
    manifest: dict[str, Any],
    class_df: pd.DataFrame,
    source_df: pd.DataFrame,
    header_by_class: pd.DataFrame,
    per_file: pd.DataFrame,
    reduction: pd.DataFrame,
    source_reduction: pd.DataFrame,
    composition: pd.DataFrame,
    attempted: pd.DataFrame,
    sampling: dict[str, Any],
    cleaning: dict[str, Any],
    duplicates: dict[str, Any],
    leakage: dict[str, Any],
) -> None:
    sample_audits = pd.DataFrame(sampling["samples"].values())
    targets = ", ".join(f"{label}={total:,}" for label, total in sampling["class_row_targets_total"].items())
    lines = [
        "# CSE-CIC-IDS-2018 DistriNet — preprocessing report",
        "",
        f"Generated by `src/preprocessing/preprocess_cicids2018_distrinet.py` in {manifest['elapsed_seconds']:,.0f} s. "
        f"Seed {manifest['seed']}. Class row targets (total over all splits): {targets}. "
        + ("**Limited (non-production) run.**" if manifest["non_production"] else "Full production run."),
        "",
        f"> {manifest['methodological_description']}",
        "",
        "> The resulting evaluation partitions do not reproduce the original CICIDS2018 class prior and should "
        "therefore be interpreted as controlled experimental datasets rather than estimates of real-world attack "
        "prevalence.",
        "",
        "## 1. Inputs",
        f"{len(manifest['input_files'])} files, {sum(manifest['input_raw_rows'].values()):,} raw rows, "
        f"{manifest['raw_column_count']} columns; {manifest['modelling_feature_count']} modelling features "
        "(CICIDS2017 order).",
        "",
        md_table(pd.DataFrame({
            "file": list(manifest["input_raw_rows"]),
            "raw rows": list(manifest["input_raw_rows"].values()),
            "sha256": [(manifest["input_sha256"] or {}).get(f) or "—" for f in manifest["input_raw_rows"]],
        })),
        "## 2. Cleaning (rules applied in order)",
        md_table(pd.DataFrame(cleaning["rules_in_order"])[["rule", "rows_removed", "description"]]),
        "Per final class (rows whose label maps to one of the five classes):",
        "",
        md_table(class_df[["category_label", "raw_retained", "removed_nonfinite", "removed_bad_timestamp",
                           "removed_negative_non_header", "after_cleaning", "header_length_negative_kept",
                           "header_length_overflow_kept"]]),
        "### Fwd/Bwd Header Length artefact (rows kept, flagged)",
        md_table(header_by_class),
        "## 3. Deduplication",
        f"{duplicates['exact_duplicates_removed']:,} duplicates removed "
        f"({duplicates['duplicate_percentage']:.3f}%) from {duplicates['rows_before_duplicate_removal']:,} rows; "
        f"{duplicates['rows_after_duplicate_removal']:,} remain. Label conflicts after dedup: "
        f"{duplicates['label_conflicts_after_dedup']['conflicting_feature_vectors']:,} vectors "
        f"({duplicates['label_conflicts_after_dedup']['rows_involved']:,} rows).",
        "",
        md_table(pd.DataFrame({"category": list(duplicates["by_category"]), "duplicates removed": list(duplicates["by_category"].values())})),
        "## 4. Class counts through the pipeline",
        md_table(class_df[["category_label", "raw_retained", "after_cleaning", "duplicates_removed", "total",
                           "train_before_reduction", "train", "val_before_reduction", "val",
                           "test_before_reduction", "test", "target_total", "final_total"]]),
        "## 5. Source-label split (chronological 70/15/15 within each label, before class-size reduction)",
        md_table(source_df[["source_label", "category_label", "total", "train_before_reduction",
                            "val_before_reduction", "test_before_reduction", "train", "val", "test"]]),
        "## 6. Class-size reduction inside each split",
        f"Each total target is apportioned 70/15/15 and sampled inside its split. "
        f"Untargeted classes keep every row: {', '.join(sampling['untargeted_classes']) or 'none'}. "
        f"Strata: {sampling['stratum_definition']}.",
        "",
        md_table(reduction),
        "Source labels (before → after):",
        "",
        md_table(source_reduction),
        "Final split composition:",
        "",
        md_table(composition.round(4)),
        "Temporal strata coverage:",
        "",
        md_table(sample_audits[["split", "category_label", "population", "target", "target_reached", "selected",
                                "strata_before", "strata_represented_after", "strata_with_minimum_quota",
                                "max_abs_share_deviation", "first_bin_utc", "last_bin_utc"]]),
        "Per source file:",
        "",
        md_table(per_file.round(6)),
        "Attempted-derived Benign rows (before → after reduction):",
        "",
        md_table(attempted),
        "## 7. Leakage and invariant checks (all asserted)",
        md_table(pd.DataFrame([
            {"pair": pair, **{k: v for k, v in rec.items() if k != "feature_only_examples_first_20"}}
            for pair, rec in leakage["pairwise"].items()
        ])),
        "- train/val/test are disjoint by `source_file:id`; every source label is present in every final split;",
        "- within each source label every train row precedes every validation row, which precedes every test row;",
        "- rows of untargeted classes are never removed (asserted per class and split, and per row);",
        "- no row moves between partitions; each split's sample is drawn from its own rows only;",
        "- every targeted class reaches exactly min(available, target) rows in every split.",
        "",
        "## 8. Transformation",
        f"RobustScaler fitted on {manifest['train_only_preprocessing']['scaler_fit_rows']:,} final training rows; "
        f"{manifest['modelling_feature_count']} features; train-constant columns (kept): "
        f"{', '.join(manifest['train_only_preprocessing']['train_constant_columns_reported_not_removed']) or 'none'}. "
        "`asinh` is applied inside the models, as for CICIDS2017.",
        "",
        "## 9. Figures",
        *[f"- `{Path(p).name}`" for p in manifest["outputs"]["figures"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
