"""Raw-data exploratory analysis of the corrected DistriNet CSE-CIC-IDS-2018 release.

Streams each of the ten corrected DistriNet day CSVs exactly once (pyarrow block
reader, SHA-256 computed in the same pass) and produces:

* exact counts: inventory, schema, labels, Attempted flows, per-day attack schedule,
  protocol/endpoint cardinality, NaN/Inf, negative values and sentinels, per-column
  min/max/mean/zero/integer statistics;
* 64-bit row-hash analyses: exact duplicates (raw row, float32 features + label),
  feature-vector label conflicts, ``Flow ID`` reuse;
* seeded bottom-K samples (uniform and per-label stratified) for quantiles, Spearman
  correlation, identical-column detection, PCA, and histograms.

The EDA runs before any split exists. Every statistic here is descriptive: nothing
computed by this script may feed the scaler, masks, mined rules, or thresholds (those
are fitted on the training split only).

Run from the repository root:
    python src/evaluation/cicids2018_distrinet_eda.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from pandas.util import hash_pandas_object

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from config.paths import SEED  # noqa: E402

# Chronological capture order; value = capture date encoded in the file name.
EXPECTED_FILES: dict[str, str] = {
    "Wednesday-14-02-2018.csv": "2018-02-14",
    "Thursday-15-02-2018.csv": "2018-02-15",
    "Friday-16-02-2018.csv": "2018-02-16",
    "Tuesday-20-02-2018.csv": "2018-02-20",
    "Wednesday-21-02-2018.csv": "2018-02-21",
    "Thursday-22-02-2018.csv": "2018-02-22",
    "Friday-23-02-2018.csv": "2018-02-23",
    "Wednesday-28-02-2018.csv": "2018-02-28",
    "Thursday-01-03-2018.csv": "2018-03-01",
    "Friday-02-03-2018.csv": "2018-03-02",
}
ID_COLUMN = "id"
FLOW_ID_COLUMN = "Flow ID"
SRC_IP_COLUMN = "Src IP"
DST_IP_COLUMN = "Dst IP"
TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"
ATTEMPTED_COLUMN = "Attempted Category"
STRING_COLUMNS = (FLOW_ID_COLUMN, SRC_IP_COLUMN, DST_IP_COLUMN, TIMESTAMP_COLUMN)
NON_FEATURE_COLUMNS = {
    ID_COLUMN,
    *STRING_COLUMNS,
    LABEL_COLUMN,
    ATTEMPTED_COLUMN,
}
PROTOCOL_COLUMN = "Protocol"
SRC_PORT_COLUMN = "Src Port"
DST_PORT_COLUMN = "Dst Port"
ICMP_SENTINEL_COLUMNS = ("ICMP Code", "ICMP Type")
TCP_FLOW_TIME_COLUMN = "Total TCP Flow Time"
SENTINEL_VALUE = -1.0
ICMP_PROTOCOL = 1
TCP_PROTOCOL = 6
BENIGN_LABEL = "BENIGN"
ATTEMPTED_MARKER = "Attempted"
MISSING_LABEL = "<missing>"
# pyarrow's streaming CSV reader slows super-linearly with block size on these files
# (measured: 2 MiB ~300k rows/s vs 64 MiB ~25k rows/s), so read small blocks and
# re-batch before the per-batch numpy/pandas work.
READ_BLOCK_SIZE = 2 << 20
PROCESS_ROWS = 250_000
BUCKET_SECONDS = 600
TIMELINE_WINDOW_HOURS = 30
QUANTILES = (0.5, 0.95, 0.99, 0.999)
IDENTICAL_RTOL = 1e-9
SELECTED_FEATURES = (
    "Flow Duration",
    "Total Fwd Packet",
    "Total Length of Fwd Packet",
    "Flow Bytes/s",
    "Flow IAT Mean",
    "Packet Length Mean",
    "FWD Init Win Bytes",
    "Fwd Seg Size Min",
)
CICIDS2017_MANIFEST = (
    REPO_ROOT / "data" / "processed" / "CICIDS_2017_Distrinet" / "preprocessing_manifest.json"
)
US_PER_SECOND = 1_000_000
US_PER_DAY = 86_400 * US_PER_SECOND


# --------------------------------------------------------------------------- #
# Streaming helpers
# --------------------------------------------------------------------------- #
class HashingReader(io.RawIOBase):
    """Sequential file wrapper that SHA-256-hashes exactly the bytes pyarrow reads."""

    def __init__(self, handle: io.BufferedReader) -> None:
        super().__init__()
        self._handle = handle
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.bytes_read

    def readinto(self, buffer: Any) -> int:
        count = self._handle.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
            self.bytes_read += count
        return count


class BottomK:
    """Deterministic uniform sample: keep the k rows with the smallest random keys.

    Mergeable across files (bottom-k of a union = bottom-k of the per-part bottom-ks),
    so the result is independent of worker scheduling.
    """

    def __init__(self, k: int, n_features: int) -> None:
        self.k = k
        self.keys = np.empty(0, dtype=np.float64)
        self.x = np.empty((0, n_features), dtype=np.float64)
        self.labels = np.empty(0, dtype=object)
        self.files = np.empty(0, dtype=np.int16)
        self.ts_us = np.empty(0, dtype=np.int64)
        self.threshold = np.inf
        self._pending: list[tuple[np.ndarray, ...]] = []
        self._pending_rows = 0

    def offer(
        self,
        keys: np.ndarray,
        x: np.ndarray,
        labels: np.ndarray,
        files: np.ndarray,
        ts_us: np.ndarray,
    ) -> None:
        keep = keys < self.threshold
        if not keep.any():
            return
        self._pending.append((keys[keep], x[keep], labels[keep], files[keep], ts_us[keep]))
        self._pending_rows += int(keep.sum())
        if self._pending_rows >= self.k:
            self._compact()

    def _compact(self) -> None:
        if not self._pending:
            return
        parts = [(self.keys, self.x, self.labels, self.files, self.ts_us), *self._pending]
        keys, x, labels, files, ts_us = (
            np.concatenate([part[i] for part in parts]) for i in range(5)
        )
        if len(keys) > self.k:
            chosen = np.argpartition(keys, self.k - 1)[: self.k]
            keys, x, labels, files, ts_us = (
                keys[chosen],
                x[chosen],
                labels[chosen],
                files[chosen],
                ts_us[chosen],
            )
        if len(keys) >= self.k:
            self.threshold = float(keys.max())
        self.keys, self.x, self.labels, self.files, self.ts_us = keys, x, labels, files, ts_us
        self._pending = []
        self._pending_rows = 0

    def finalize(self) -> BottomK:
        self._compact()
        order = np.argsort(self.keys, kind="stable")
        self.keys, self.x, self.labels, self.files, self.ts_us = (
            self.keys[order],
            self.x[order],
            self.labels[order],
            self.files[order],
            self.ts_us[order],
        )
        return self

    @classmethod
    def merge(cls, parts: list[BottomK], k: int, n_features: int) -> BottomK:
        merged = cls(k, n_features)
        for part in parts:
            merged.offer(part.keys, part.x, part.labels, part.files, part.ts_us)
        return merged.finalize()


@dataclass
class ScanTask:
    path: Path
    file_index: int
    expected_date: str
    raw_columns: list[str]
    feature_columns: list[str]
    max_rows: int | None
    hash_input: bool
    uniform_k: int
    per_label_k: int
    seed: int


@dataclass
class FileScan:
    name: str
    file_index: int
    size_bytes: int
    sha256: str | None
    rows: int
    complete: bool
    id_is_row_number: bool
    label_counts: Counter = field(default_factory=Counter)
    label_decode_fallbacks: int = 0
    label_attempted: Counter = field(default_factory=Counter)
    label_protocol: Counter = field(default_factory=Counter)
    attack_src_ip: Counter = field(default_factory=Counter)
    attack_dst_ip: Counter = field(default_factory=Counter)
    attack_dst_port: Counter = field(default_factory=Counter)
    src_ips: set[str] = field(default_factory=set)
    dst_ips: set[str] = field(default_factory=set)
    src_port_counts: np.ndarray | None = None
    dst_port_counts: np.ndarray | None = None
    label_time: dict[str, list[int]] = field(default_factory=dict)
    buckets: Counter = field(default_factory=Counter)
    timestamp: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    rows_flags: dict[str, int] = field(default_factory=dict)
    label_nonfinite: Counter = field(default_factory=Counter)
    label_negative: Counter = field(default_factory=Counter)
    sentinel_checks: dict[str, int] = field(default_factory=dict)
    flow_hash: np.ndarray | None = None
    full_row_hash: np.ndarray | None = None
    feature_hash: np.ndarray | None = None
    feature_label_hash: np.ndarray | None = None
    clean_rows: np.ndarray | None = None
    label_codes: np.ndarray | None = None
    label_vocab: list[str] = field(default_factory=list)
    uniform: BottomK | None = None
    per_label: dict[str, BottomK] = field(default_factory=dict)


def normalize_column_name(value: Any) -> str:
    return str(value).lstrip("\ufeff").strip()


def decode_label(raw: bytes | None) -> tuple[str, bool]:
    """Decode a raw label byte string; returns (label, used_cp1252_fallback)."""
    if raw is None:
        return MISSING_LABEL, False
    try:
        text, fallback = raw.decode("utf-8"), False
    except UnicodeDecodeError:
        text, fallback = raw.decode("cp1252"), True
    text = text.strip()
    return (text if text else MISSING_LABEL), fallback


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [normalize_column_name(value) for value in handle.readline().rstrip("\r\n").split(",")]


def validate_inventory(input_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    present = sorted(path.name for path in input_dir.glob("*.csv"))
    expected = sorted(EXPECTED_FILES)
    if present != expected:
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        raise ValueError(f"CSV inventory mismatch; missing={missing}, extra={extra}")
    paths = [input_dir / name for name in EXPECTED_FILES]
    headers = [read_header(path) for path in paths]
    if any(header != headers[0] for header in headers[1:]):
        raise ValueError("CSV headers differ across DistriNet CSE-CIC-IDS-2018 days")
    header = headers[0]
    if len(header) != len(set(header)):
        raise ValueError("duplicate normalized column names are not supported")
    missing_required = sorted(NON_FEATURE_COLUMNS - set(header))
    if missing_required:
        raise ValueError(f"required columns absent: {missing_required}")
    feature_columns = [column for column in header if column not in NON_FEATURE_COLUMNS]
    for column in (PROTOCOL_COLUMN, SRC_PORT_COLUMN, DST_PORT_COLUMN, *ICMP_SENTINEL_COLUMNS):
        if column not in feature_columns:
            raise ValueError(f"expected numeric column absent: {column}")
    return paths, header, feature_columns


def _counter_from_pairs(codes: np.ndarray, values: np.ndarray) -> Counter:
    frame = pd.DataFrame({"code": codes, "value": values})
    return Counter(dict(frame.value_counts(sort=False, dropna=False).items()))


def _string_array(batch: pa.RecordBatch, name: str) -> np.ndarray:
    return batch.column(name).to_numpy(zero_copy_only=False)


def _rebatched(
    reader: pacsv.CSVStreamingReader,
    target_rows: int,
    max_rows: int | None,
    state: dict[str, bool],
) -> Iterator[pa.RecordBatch]:
    """Concatenate small reader batches into ~target_rows batches, honouring max_rows.

    Sets ``state["truncated"]`` when max_rows stopped the read before end of file.
    """
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    emitted = 0
    for batch in reader:
        if max_rows is not None:
            room = max_rows - emitted - pending_rows
            if room <= 0:
                state["truncated"] = True
                break
            if batch.num_rows > room:
                batch = batch.slice(0, room)
                state["truncated"] = True
        if batch.num_rows:
            pending.append(batch)
            pending_rows += batch.num_rows
        if pending_rows >= target_rows or state["truncated"]:
            if pending:
                yield pa.Table.from_batches(pending).combine_chunks().to_batches()[0]
            emitted += pending_rows
            pending, pending_rows = [], 0
        if state["truncated"]:
            return
    if pending:
        yield pa.Table.from_batches(pending).combine_chunks().to_batches()[0]



# --------------------------------------------------------------------------- #
# Per-file scan (runs in a worker process)
# --------------------------------------------------------------------------- #
def scan_file(task: ScanTask) -> FileScan:
    p = len(task.feature_columns)
    column_types: dict[str, pa.DataType] = {name: pa.float64() for name in task.feature_columns}
    column_types.update({name: pa.string() for name in STRING_COLUMNS})
    column_types[ID_COLUMN] = pa.int64()
    column_types[LABEL_COLUMN] = pa.binary()
    column_types[ATTEMPTED_COLUMN] = pa.float64()
    read_options = pacsv.ReadOptions(
        block_size=READ_BLOCK_SIZE, column_names=task.raw_columns, skip_rows=1
    )
    convert_options = pacsv.ConvertOptions(
        column_types=column_types,
        null_values=["", "NaN", "nan", "NULL", "null"],
        strings_can_be_null=True,
    )

    feature_index = {name: i for i, name in enumerate(task.feature_columns)}
    protocol_i = feature_index[PROTOCOL_COLUMN]
    src_port_i = feature_index[SRC_PORT_COLUMN]
    dst_port_i = feature_index[DST_PORT_COLUMN]
    icmp_i = [feature_index[name] for name in ICMP_SENTINEL_COLUMNS]
    non_icmp_mask = np.ones(p, dtype=bool)
    non_icmp_mask[icmp_i] = False
    tcp_time_i = feature_index.get(TCP_FLOW_TIME_COLUMN)
    expected_day = int(np.datetime64(task.expected_date, "D").astype(np.int64))

    rng = np.random.default_rng([task.seed, task.file_index])
    uniform = BottomK(task.uniform_k, p)
    per_label: dict[str, BottomK] = {}

    vocab: dict[str, int] = {}
    codes_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    feature_label_parts: list[np.ndarray] = []
    clean_parts: list[np.ndarray] = []

    stats = {
        "n_null": np.zeros(p, dtype=np.int64),
        "n_nan": np.zeros(p, dtype=np.int64),
        "n_posinf": np.zeros(p, dtype=np.int64),
        "n_neginf": np.zeros(p, dtype=np.int64),
        "n_finite": np.zeros(p, dtype=np.int64),
        "n_negative": np.zeros(p, dtype=np.int64),
        "n_minus1": np.zeros(p, dtype=np.int64),
        "n_zero": np.zeros(p, dtype=np.int64),
        "n_nonint": np.zeros(p, dtype=np.int64),
        "sum": np.zeros(p, dtype=np.float64),
        "sumsq": np.zeros(p, dtype=np.float64),
        "min": np.full(p, np.inf),
        "max": np.full(p, -np.inf),
    }
    row_flags = Counter()
    sentinel = Counter()
    ts_info = {
        "missing": 0,
        "unparseable": 0,
        "with_fraction": 0,
        "date_mismatch": 0,
        "reversals": 0,
        "min_us": None,
        "max_us": None,
    }
    last_ts: int | None = None
    label_time: dict[int, list[int]] = {}
    buckets: Counter = Counter()
    label_attempted: Counter = Counter()
    label_protocol: Counter = Counter()
    attack_src_ip: Counter = Counter()
    attack_dst_ip: Counter = Counter()
    attack_dst_port: Counter = Counter()
    label_nonfinite: Counter = Counter()
    label_negative: Counter = Counter()
    label_counts: Counter = Counter()
    src_ips: set[str] = set()
    dst_ips: set[str] = set()
    src_port_counts = np.zeros(65536, dtype=np.int64)
    dst_port_counts = np.zeros(65536, dtype=np.int64)
    decode_fallbacks = 0
    id_is_row_number = True
    rows = 0
    read_state = {"truncated": False}

    size_bytes = task.path.stat().st_size
    with task.path.open("rb") as raw_handle:
        hashing = HashingReader(raw_handle) if task.hash_input else None
        source: Any = hashing if hashing is not None else raw_handle
        reader = pacsv.open_csv(source, read_options=read_options, convert_options=convert_options)
        for batch in _rebatched(reader, PROCESS_ROWS, task.max_rows, read_state):
            n = batch.num_rows

            # --- labels (binary -> decoded dictionary -> per-file codes) ---
            encoded = batch.column(LABEL_COLUMN).dictionary_encode()
            dictionary = encoded.dictionary.to_pylist()
            dict_codes = np.empty(len(dictionary), dtype=np.int32)
            for j, raw in enumerate(dictionary):
                label, fallback = decode_label(raw)
                decode_fallbacks += int(fallback)
                dict_codes[j] = vocab.setdefault(label, len(vocab))
            indices = encoded.indices
            if indices.null_count:
                missing_code = vocab.setdefault(MISSING_LABEL, len(vocab))
                indices = pc.fill_null(indices, -1)
                idx = indices.to_numpy(zero_copy_only=False)
                codes = np.where(idx < 0, missing_code, dict_codes[np.maximum(idx, 0)])
            else:
                codes = dict_codes[indices.to_numpy(zero_copy_only=False)]
            codes = codes.astype(np.int32, copy=False)
            vocab_array = np.asarray(list(vocab), dtype=object)
            labels = vocab_array[codes]
            label_counts.update(Counter(dict(zip(*np.unique(codes, return_counts=True)))))
            benign_code = vocab.get(BENIGN_LABEL, -1)
            attack_rows = codes != benign_code

            # --- ids ---
            ids = batch.column(ID_COLUMN).to_numpy(zero_copy_only=False)
            if id_is_row_number and not np.array_equal(
                ids, np.arange(rows + 1, rows + n + 1, dtype=np.int64)
            ):
                id_is_row_number = False

            # --- numeric feature matrix ---
            columns = [batch.column(name) for name in task.feature_columns]
            stats["n_null"] += np.asarray([column.null_count for column in columns], dtype=np.int64)
            x = np.column_stack(
                [column.to_numpy(zero_copy_only=False) for column in columns]
            ).astype(np.float64, copy=False)
            finite = np.isfinite(x)
            nan = np.isnan(x)
            stats["n_nan"] += nan.sum(axis=0)
            stats["n_posinf"] += np.isposinf(x).sum(axis=0)
            stats["n_neginf"] += np.isneginf(x).sum(axis=0)
            stats["n_finite"] += finite.sum(axis=0)
            safe = np.where(finite, x, 0.0)
            negative = finite & (x < 0)
            stats["n_negative"] += negative.sum(axis=0)
            stats["n_minus1"] += (finite & (x == SENTINEL_VALUE)).sum(axis=0)
            stats["n_zero"] += (finite & (x == 0)).sum(axis=0)
            stats["n_nonint"] += (finite & (np.floor(safe) != safe)).sum(axis=0)
            stats["sum"] += safe.sum(axis=0)
            stats["sumsq"] += np.square(safe).sum(axis=0)
            stats["min"] = np.minimum(stats["min"], np.where(finite, x, np.inf).min(axis=0))
            stats["max"] = np.maximum(stats["max"], np.where(finite, x, -np.inf).max(axis=0))

            row_nonfinite = ~finite.all(axis=1)
            row_negative = negative.any(axis=1)
            icmp_non_sentinel = negative[:, icmp_i] & (x[:, icmp_i] != SENTINEL_VALUE)
            row_negative_excl_sentinel = negative[:, non_icmp_mask].any(axis=1) | icmp_non_sentinel.any(
                axis=1
            )
            row_flags["any_nonfinite"] += int(row_nonfinite.sum())
            row_flags["any_negative"] += int(row_negative.sum())
            row_flags["any_negative_excluding_icmp_sentinel"] += int(
                (row_negative_excl_sentinel & ~row_nonfinite).sum()
            )
            row_flags["dropped_by_nonfinite_or_negative_excluding_icmp_sentinel"] += int(
                (row_nonfinite | row_negative_excl_sentinel).sum()
            )
            label_nonfinite.update(
                Counter(dict(zip(*np.unique(codes[row_nonfinite], return_counts=True))))
            )
            label_negative.update(
                Counter(
                    dict(
                        zip(
                            *np.unique(
                                codes[row_negative_excl_sentinel & ~row_nonfinite],
                                return_counts=True,
                            )
                        )
                    )
                )
            )

            protocol = x[:, protocol_i]
            is_icmp = protocol == ICMP_PROTOCOL
            for name, i in zip(ICMP_SENTINEL_COLUMNS, icmp_i):
                sentinel[f"{name}: sentinel on ICMP rows"] += int((is_icmp & (x[:, i] == SENTINEL_VALUE)).sum())
                sentinel[f"{name}: non-sentinel on non-ICMP rows"] += int(
                    (~is_icmp & (x[:, i] != SENTINEL_VALUE)).sum()
                )
            sentinel["ICMP rows"] += int(is_icmp.sum())
            if tcp_time_i is not None:
                sentinel[f"{TCP_FLOW_TIME_COLUMN}: nonzero on non-TCP rows"] += int(
                    ((protocol != TCP_PROTOCOL) & finite[:, tcp_time_i] & (x[:, tcp_time_i] != 0)).sum()
                )

            # --- protocol / attempted / endpoints ---
            protocol_values = np.where(np.isfinite(protocol), protocol, -1).astype(np.int64)
            label_protocol.update(_counter_from_pairs(codes, protocol_values))
            attempted = batch.column(ATTEMPTED_COLUMN).to_numpy(zero_copy_only=False)
            attempted_values = np.where(np.isfinite(attempted), attempted, np.nan)
            label_attempted.update(
                _counter_from_pairs(codes, pd.Series(attempted_values).fillna(-999).astype(np.int64))
            )
            src_ip = _string_array(batch, SRC_IP_COLUMN)
            dst_ip = _string_array(batch, DST_IP_COLUMN)
            src_ips.update(pc.unique(batch.column(SRC_IP_COLUMN)).to_pylist())
            dst_ips.update(pc.unique(batch.column(DST_IP_COLUMN)).to_pylist())
            for port_i, target in ((src_port_i, src_port_counts), (dst_port_i, dst_port_counts)):
                port = x[:, port_i]
                valid_port = np.isfinite(port) & (port >= 0) & (port <= 65535)
                target += np.bincount(port[valid_port].astype(np.int64), minlength=65536)
            if attack_rows.any():
                attack_codes = codes[attack_rows]
                attack_src_ip.update(_counter_from_pairs(attack_codes, src_ip[attack_rows]))
                attack_dst_ip.update(_counter_from_pairs(attack_codes, dst_ip[attack_rows]))
                dst_port = x[attack_rows, dst_port_i]
                attack_dst_port.update(
                    _counter_from_pairs(
                        attack_codes, np.where(np.isfinite(dst_port), dst_port, -1).astype(np.int64)
                    )
                )

            # --- timestamps ---
            ts_raw = batch.column(TIMESTAMP_COLUMN)
            ts_series = ts_raw.to_pandas()
            # Empty strings are already null via ConvertOptions.null_values.
            missing_ts = ts_series.isna().to_numpy()
            parsed = pd.to_datetime(ts_series, format="ISO8601", errors="coerce")
            valid_ts = parsed.notna().to_numpy()
            ts_info["missing"] += int(missing_ts.sum())
            ts_info["unparseable"] += int((~valid_ts & ~missing_ts).sum())
            ts_info["with_fraction"] += int(
                pc.sum(pc.cast(pc.match_substring(ts_raw, "."), pa.int64())).as_py() or 0
            )
            ts_us = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
            ts_us[valid_ts] = (
                parsed[valid_ts].to_numpy().astype("datetime64[us]").astype(np.int64)
            )
            valid_us = ts_us[valid_ts]
            if valid_us.size:
                ts_info["date_mismatch"] += int(((valid_us // US_PER_DAY) != expected_day).sum())
                sequence = valid_us if last_ts is None else np.concatenate(([last_ts], valid_us))
                ts_info["reversals"] += int((np.diff(sequence) < 0).sum())
                last_ts = int(valid_us[-1])
                lo, hi = int(valid_us.min()), int(valid_us.max())
                ts_info["min_us"] = lo if ts_info["min_us"] is None else min(ts_info["min_us"], lo)
                ts_info["max_us"] = hi if ts_info["max_us"] is None else max(ts_info["max_us"], hi)
                valid_codes = codes[valid_ts]
                frame = pd.DataFrame({"code": valid_codes, "ts": valid_us})
                grouped = frame.groupby("code")["ts"].agg(["count", "min", "max"])
                for code, (count, t_min, t_max) in grouped.iterrows():
                    entry = label_time.setdefault(int(code), [0, int(t_min), int(t_max)])
                    entry[0] += int(count)
                    entry[1] = min(entry[1], int(t_min))
                    entry[2] = max(entry[2], int(t_max))
                buckets.update(
                    _counter_from_pairs(valid_codes, valid_us // (BUCKET_SECONDS * US_PER_SECOND))
                )

            # --- hashes ---
            x32 = pd.DataFrame(x.astype(np.float32))
            feature_hash = hash_pandas_object(x32, index=False).to_numpy(dtype=np.uint64)
            feature_label_hash = hash_pandas_object(
                pd.DataFrame({"f": feature_hash, "l": labels}), index=False
            ).to_numpy(dtype=np.uint64)
            raw_numeric_hash = hash_pandas_object(pd.DataFrame(x), index=False).to_numpy(
                dtype=np.uint64
            )
            full_row_hash = hash_pandas_object(
                pd.DataFrame(
                    {
                        "numeric": raw_numeric_hash,
                        "flow": _string_array(batch, FLOW_ID_COLUMN),
                        "src": src_ip,
                        "dst": dst_ip,
                        "ts": ts_raw.to_numpy(zero_copy_only=False),
                        "label": labels,
                        "attempted": attempted_values,
                    }
                ),
                index=False,
            ).to_numpy(dtype=np.uint64)
            flow_hash = hash_pandas_object(
                pd.Series(_string_array(batch, FLOW_ID_COLUMN)), index=False
            ).to_numpy(dtype=np.uint64)
            codes_parts.append(codes)
            flow_parts.append(flow_hash)
            full_parts.append(full_row_hash)
            feature_parts.append(feature_hash)
            feature_label_parts.append(feature_label_hash)
            # Rows surviving the 2017-style row-local cleaning rule (finite, no negative
            # value other than the ICMP -1 sentinel); duplicate analysis is repeated on them.
            clean_parts.append(~(row_nonfinite | row_negative_excl_sentinel))

            # --- samples ---
            keys = rng.random(n)
            file_ids = np.full(n, task.file_index, dtype=np.int16)
            uniform.offer(keys, x, labels, file_ids, ts_us)
            for code in np.unique(codes):
                selected = codes == code
                label = vocab_array[code]
                sampler = per_label.setdefault(label, BottomK(task.per_label_k, p))
                sampler.offer(
                    keys[selected], x[selected], labels[selected], file_ids[selected], ts_us[selected]
                )
            rows += n
        complete = not read_state["truncated"]

        sha256 = None
        if hashing is not None and complete:
            while hashing.read(1 << 20):
                pass
            if hashing.bytes_read != size_bytes:
                raise AssertionError(
                    f"{task.path.name}: hashed {hashing.bytes_read} bytes, file has {size_bytes}"
                )
            sha256 = hashing.digest.hexdigest()

    vocab_list = list(vocab)
    return FileScan(
        name=task.path.name,
        file_index=task.file_index,
        size_bytes=size_bytes,
        sha256=sha256,
        rows=rows,
        complete=complete,
        id_is_row_number=id_is_row_number,
        label_counts=Counter({vocab_list[c]: n for c, n in label_counts.items()}),
        label_decode_fallbacks=decode_fallbacks,
        label_attempted=_relabel(label_attempted, vocab_list),
        label_protocol=_relabel(label_protocol, vocab_list),
        attack_src_ip=_relabel(attack_src_ip, vocab_list),
        attack_dst_ip=_relabel(attack_dst_ip, vocab_list),
        attack_dst_port=_relabel(attack_dst_port, vocab_list),
        src_ips={ip for ip in src_ips if ip is not None},
        dst_ips={ip for ip in dst_ips if ip is not None},
        src_port_counts=src_port_counts,
        dst_port_counts=dst_port_counts,
        label_time={vocab_list[c]: v for c, v in label_time.items()},
        buckets=_relabel(buckets, vocab_list),
        timestamp=ts_info,
        columns=stats,
        rows_flags=dict(row_flags),
        label_nonfinite=Counter({vocab_list[c]: n for c, n in label_nonfinite.items()}),
        label_negative=Counter({vocab_list[c]: n for c, n in label_negative.items()}),
        sentinel_checks=dict(sentinel),
        flow_hash=_concat(flow_parts, np.uint64),
        full_row_hash=_concat(full_parts, np.uint64),
        feature_hash=_concat(feature_parts, np.uint64),
        feature_label_hash=_concat(feature_label_parts, np.uint64),
        clean_rows=_concat(clean_parts, bool),
        label_codes=_concat(codes_parts, np.int32),
        label_vocab=vocab_list,
        uniform=uniform.finalize(),
        per_label={label: sampler.finalize() for label, sampler in per_label.items()},
    )


def _relabel(counter: Counter, vocab: list[str]) -> Counter:
    return Counter({(vocab[code], value): n for (code, value), n in counter.items()})


def _concat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


# --------------------------------------------------------------------------- #
# Aggregation and analysis (parent process)
# --------------------------------------------------------------------------- #
def _iso(us: int | None) -> str | None:
    if us is None:
        return None
    return pd.Timestamp(int(us), unit="us").strftime("%Y-%m-%d %H:%M:%S.%f")


def _top(counter: Counter, label: str, k: int) -> tuple[int, str]:
    items = [(value, n) for (lab, value), n in counter.items() if lab == label]
    total = sum(n for _, n in items)
    items.sort(key=lambda item: (-item[1], str(item[0])))
    text = "; ".join(f"{value} ({100.0 * n / total:.1f}%)" for value, n in items[:k])
    return len(items), text


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def analyse(
    scans: list[FileScan],
    header: list[str],
    feature_columns: list[str],
    uniform_k: int,
    per_label_k: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], BottomK, dict[str, BottomK]]:
    p = len(feature_columns)
    day_names = [scan.name.removesuffix(".csv") for scan in scans]
    total_rows = sum(scan.rows for scan in scans)
    tables: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {}

    # --- inventory ---
    tables["file_inventory"] = pd.DataFrame(
        {
            "file": scan.name,
            "capture_date": EXPECTED_FILES[scan.name],
            "size_bytes": scan.size_bytes,
            "rows": scan.rows,
            "first_timestamp": _iso(scan.timestamp["min_us"]),
            "last_timestamp": _iso(scan.timestamp["max_us"]),
            "id_is_1_based_row_number": scan.id_is_row_number,
            "sha256": scan.sha256,
        }
        for scan in scans
    )
    report["inventory"] = {
        "files": len(scans),
        "total_rows": total_rows,
        "total_bytes": sum(scan.size_bytes for scan in scans),
        "complete_read": all(scan.complete for scan in scans),
    }

    # --- schema (+ comparison with the CICIDS2017 modelling features) ---
    schema: dict[str, Any] = {
        "raw_columns": header,
        "raw_column_count": len(header),
        "non_feature_columns": [c for c in header if c in NON_FEATURE_COLUMNS],
        "numeric_candidate_columns": feature_columns,
        "numeric_candidate_count": p,
    }
    if CICIDS2017_MANIFEST.is_file():
        names_2017 = json.loads(CICIDS2017_MANIFEST.read_text(encoding="utf-8"))[
            "modelling_feature_names"
        ]
        schema["cicids2017_comparison"] = {
            "manifest": str(CICIDS2017_MANIFEST.relative_to(REPO_ROOT)),
            "cicids2017_feature_count": len(names_2017),
            "shared": [c for c in names_2017 if c in feature_columns],
            "only_in_2018": [c for c in feature_columns if c not in names_2017],
            "only_in_2017": [c for c in names_2017 if c not in feature_columns],
            "shared_in_same_relative_order": [c for c in feature_columns if c in names_2017]
            == [c for c in names_2017 if c in feature_columns],
        }
    else:
        schema["cicids2017_comparison"] = {"manifest": None, "reason": f"{CICIDS2017_MANIFEST} not found"}
    report["schema"] = schema

    # --- labels ---
    label_counts: Counter = Counter()
    for scan in scans:
        label_counts.update(scan.label_counts)
    vocab = sorted(label_counts)
    code_of = {label: i for i, label in enumerate(vocab)}
    labels_frame = pd.DataFrame(
        {
            "label": label,
            "rows": label_counts[label],
            "share_pct": 100.0 * label_counts[label] / total_rows,
            "attempted": ATTEMPTED_MARKER in label,
            "days_present": sum(scan.label_counts.get(label, 0) > 0 for scan in scans),
        }
        for label in vocab
    ).sort_values(["rows", "label"], ascending=[False, True], ignore_index=True)
    tables["label_distribution"] = labels_frame
    tables["label_by_day"] = pd.DataFrame(
        {day: [scan.label_counts.get(label, 0) for label in labels_frame["label"]] for day, scan in zip(day_names, scans)},
        index=labels_frame["label"],
    ).reset_index()
    attempted_rows = int(labels_frame.loc[labels_frame["attempted"], "rows"].sum())
    benign_rows = int(label_counts.get(BENIGN_LABEL, 0))
    report["labels"] = {
        "distinct_labels": len(vocab),
        "benign_rows": benign_rows,
        "benign_share_pct": 100.0 * benign_rows / total_rows,
        "attempted_label_rows": attempted_rows,
        "attempted_label_share_pct": 100.0 * attempted_rows / total_rows,
        "label_decode_cp1252_fallbacks": sum(scan.label_decode_fallbacks for scan in scans),
        "labels_below_1000_rows": labels_frame.loc[labels_frame["rows"] < 1000, "label"].tolist(),
        "labels_below_100_rows": labels_frame.loc[labels_frame["rows"] < 100, "label"].tolist(),
    }

    attempted_counter: Counter = Counter()
    for scan in scans:
        attempted_counter.update(scan.label_attempted)
    tables["attempted_category_crosstab"] = (
        pd.DataFrame(
            [(label, value, n) for (label, value), n in attempted_counter.items()],
            columns=["label", "attempted_category", "rows"],
        )
        .sort_values(["label", "attempted_category"], ignore_index=True)
    )

    # --- per-day attack schedule ---
    schedule_rows = []
    for day, scan in zip(day_names, scans):
        for label, (count, t_min, t_max) in sorted(scan.label_time.items()):
            if label == BENIGN_LABEL:
                continue
            schedule_rows.append(
                {
                    "day": day,
                    "label": label,
                    "rows": count,
                    "first": _iso(t_min),
                    "last": _iso(t_max),
                    "span_minutes": (t_max - t_min) / (60 * US_PER_SECOND),
                }
            )
    tables["attack_schedule"] = pd.DataFrame(schedule_rows)

    # --- protocol and endpoints ---
    protocol_counter: Counter = Counter()
    for scan in scans:
        protocol_counter.update(scan.label_protocol)
    tables["protocol_by_label"] = (
        pd.DataFrame(
            [(label, proto, n) for (label, proto), n in protocol_counter.items()],
            columns=["label", "protocol", "rows"],
        )
        .pivot_table(index="label", columns="protocol", values="rows", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    tables["protocol_by_label"].columns = [str(c) for c in tables["protocol_by_label"].columns]
    src_ips: set[str] = set().union(*(scan.src_ips for scan in scans))
    dst_ips: set[str] = set().union(*(scan.dst_ips for scan in scans))
    src_ports = np.sum([scan.src_port_counts for scan in scans], axis=0)
    dst_ports = np.sum([scan.dst_port_counts for scan in scans], axis=0)
    protocol_totals: Counter = Counter()
    for (_, proto), n in protocol_counter.items():
        protocol_totals[int(proto)] += n
    report["cardinality"] = {
        "protocol_rows": dict(sorted(protocol_totals.items())),
        "distinct_src_ip": len(src_ips),
        "distinct_dst_ip": len(dst_ips),
        "distinct_src_port": int((src_ports > 0).sum()),
        "distinct_dst_port": int((dst_ports > 0).sum()),
    }
    endpoint_counters = {"src": Counter(), "dst": Counter(), "port": Counter()}
    for scan in scans:
        endpoint_counters["src"].update(scan.attack_src_ip)
        endpoint_counters["dst"].update(scan.attack_dst_ip)
        endpoint_counters["port"].update(scan.attack_dst_port)
    endpoint_rows = []
    for label in labels_frame["label"]:
        if label == BENIGN_LABEL:
            continue
        n_src, top_src = _top(endpoint_counters["src"], label, 3)
        n_dst, top_dst = _top(endpoint_counters["dst"], label, 3)
        n_port, top_port = _top(endpoint_counters["port"], label, 3)
        endpoint_rows.append(
            {
                "label": label,
                "distinct_src_ip": n_src,
                "top_src_ip": top_src,
                "distinct_dst_ip": n_dst,
                "top_dst_ip": top_dst,
                "distinct_dst_port": n_port,
                "top_dst_port": top_port,
            }
        )
    tables["attack_endpoints"] = pd.DataFrame(endpoint_rows)

    # --- row identity: id and Flow ID ---
    flow_hash = np.concatenate([scan.flow_hash for scan in scans])
    _, flow_counts = np.unique(flow_hash, return_counts=True)
    per_file_flows = np.concatenate([np.unique(scan.flow_hash) for scan in scans])
    _, file_multiplicity = np.unique(per_file_flows, return_counts=True)
    report["row_identity"] = {
        "id_is_1_based_row_number_in_every_file": all(scan.id_is_row_number for scan in scans),
        "flow_id_distinct": int(len(flow_counts)),
        "flow_id_rows_beyond_first": int(len(flow_hash) - len(flow_counts)),
        "flow_id_max_multiplicity": int(flow_counts.max()) if len(flow_counts) else 0,
        "flow_id_in_two_or_more_files": int((file_multiplicity > 1).sum()),
    }
    del flow_hash, per_file_flows

    # --- timestamps ---
    tables["timestamp_audit"] = pd.DataFrame(
        {
            "file": scan.name,
            "missing": scan.timestamp["missing"],
            "unparseable": scan.timestamp["unparseable"],
            "with_fractional_seconds": scan.timestamp["with_fraction"],
            "date_differs_from_filename": scan.timestamp["date_mismatch"],
            "adjacent_reversals": scan.timestamp["reversals"],
        }
        for scan in scans
    )
    report["timestamps"] = {
        key: int(tables["timestamp_audit"][column].sum())
        for key, column in (
            ("missing", "missing"),
            ("unparseable", "unparseable"),
            ("with_fractional_seconds", "with_fractional_seconds"),
            ("date_differs_from_filename", "date_differs_from_filename"),
            ("adjacent_reversals", "adjacent_reversals"),
        )
    }

    # --- per-column statistics ---
    stats = {key: np.sum([scan.columns[key] for scan in scans], axis=0) for key in scans[0].columns if key not in ("min", "max")}
    stats["min"] = np.min([scan.columns["min"] for scan in scans], axis=0)
    stats["max"] = np.max([scan.columns["max"] for scan in scans], axis=0)
    n_finite = np.maximum(stats["n_finite"], 1)
    mean = stats["sum"] / n_finite
    std = np.sqrt(np.maximum(stats["sumsq"] / n_finite - mean**2, 0.0))

    uniform = BottomK.merge([scan.uniform for scan in scans], uniform_k, p)
    per_label_parts: dict[str, list[BottomK]] = {}
    for scan in scans:
        for label, sample in scan.per_label.items():
            per_label_parts.setdefault(label, []).append(sample)
    per_label = {label: BottomK.merge(parts, per_label_k, p) for label, parts in per_label_parts.items()}
    sample_x = np.where(np.isfinite(uniform.x), uniform.x, np.nan)
    quantiles = (
        np.nanquantile(sample_x, QUANTILES, axis=0)
        if len(sample_x)
        else np.full((len(QUANTILES), p), np.nan)
    )

    constant = (stats["min"] == stats["max"]) & (stats["n_finite"] == total_rows)
    column_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "min": stats["min"],
            "max": stats["max"],
            "mean": mean,
            "std": std,
            **{f"q{q * 100:g}": quantiles[i] for i, q in enumerate(QUANTILES)},
            "zero_pct": 100.0 * stats["n_zero"] / n_finite,
            "integer_valued": stats["n_nonint"] == 0,
            "constant": constant,
            "n_null": stats["n_null"],
            "n_nan_incl_null": stats["n_nan"],
            "n_posinf": stats["n_posinf"],
            "n_neginf": stats["n_neginf"],
            "n_negative": stats["n_negative"],
            "n_minus1": stats["n_minus1"],
        }
    )
    tables["feature_summary"] = column_frame
    nonfinite_mask = (column_frame["n_nan_incl_null"] + column_frame["n_posinf"] + column_frame["n_neginf"]) > 0
    tables["nonfinite_columns"] = column_frame.loc[
        nonfinite_mask, ["feature", "n_null", "n_nan_incl_null", "n_posinf", "n_neginf"]
    ].reset_index(drop=True)
    tables["negative_columns"] = column_frame.loc[
        column_frame["n_negative"] > 0, ["feature", "n_negative", "n_minus1", "min"]
    ].reset_index(drop=True)
    row_flags: Counter = Counter()
    sentinel: Counter = Counter()
    label_nonfinite: Counter = Counter()
    label_negative: Counter = Counter()
    for scan in scans:
        row_flags.update(scan.rows_flags)
        sentinel.update(scan.sentinel_checks)
        label_nonfinite.update(scan.label_nonfinite)
        label_negative.update(scan.label_negative)
    tables["cleaning_impact_by_label"] = pd.DataFrame(
        {
            "label": label,
            "rows": label_counts[label],
            "nonfinite_rows": label_nonfinite.get(label, 0),
            "negative_rows_excl_icmp_sentinel": label_negative.get(label, 0),
            "removed_pct": 100.0
            * (label_nonfinite.get(label, 0) + label_negative.get(label, 0))
            / label_counts[label],
        }
        for label in labels_frame["label"]
    )
    report["cleaning"] = {
        "rows_any_nonfinite": row_flags["any_nonfinite"],
        "rows_any_negative_all_columns": row_flags["any_negative"],
        "rows_negative_excluding_icmp_sentinel_among_finite": row_flags[
            "any_negative_excluding_icmp_sentinel"
        ],
        "rows_removed_by_2017_style_rule_with_icmp_sentinel_exempt": row_flags[
            "dropped_by_nonfinite_or_negative_excluding_icmp_sentinel"
        ],
        "sentinel_checks": dict(sentinel),
        "icmp_negatives_are_only_minus1": {
            name: bool(
                column_frame.loc[column_frame["feature"] == name, "n_negative"].iloc[0]
                == column_frame.loc[column_frame["feature"] == name, "n_minus1"].iloc[0]
            )
            for name in ICMP_SENTINEL_COLUMNS
        },
    }

    # --- duplicates and label conflicts (64-bit row hashes) ---
    codes = np.concatenate(
        [np.asarray([code_of[l] for l in scan.label_vocab], dtype=np.int32)[scan.label_codes] for scan in scans]
    )
    full_row = np.concatenate([scan.full_row_hash for scan in scans])
    feature = np.concatenate([scan.feature_hash for scan in scans])
    feature_label = np.concatenate([scan.feature_label_hash for scan in scans])
    clean = np.concatenate([scan.clean_rows for scan in scans])
    duplicates: dict[str, Any] = {
        "hash": "pandas hash_pandas_object (64-bit); expected collisions ~ n^2 / 2^65",
        "expected_hash_collisions": float(total_rows**2 / 2.0**65),
        "full_raw_row_excluding_id": int(total_rows - np.unique(full_row).size),
    }
    del full_row
    for scope, mask in (("all_rows", slice(None)), ("clean_rows", clean)):
        fl = feature_label[mask]
        f = feature[mask]
        c = codes[mask]
        unique_fl, first = np.unique(fl, return_index=True)
        pair_feature = f[first]
        pair_label = c[first]
        vec, vec_counts = np.unique(pair_feature, return_counts=True)
        conflict_vectors = vec[vec_counts > 1]
        duplicates[scope] = {
            "rows": int(len(fl)),
            "float32_features_plus_label_duplicates": int(len(fl) - len(unique_fl)),
            "float32_features_plus_label_duplicate_pct": 100.0 * (len(fl) - len(unique_fl)) / max(len(fl), 1),
            "rows_after_dedup": int(len(unique_fl)),
            "float32_feature_only_duplicates": int(len(f) - len(vec)),
            "conflicting_feature_vectors": int(len(conflict_vectors)),
            "rows_with_conflicting_feature_vector": int(np.isin(f, conflict_vectors).sum()),
        }
        if scope == "clean_rows":
            rows_per_label = np.bincount(c, minlength=len(vocab))
            unique_per_label = np.bincount(pair_label, minlength=len(vocab))
            tables["duplicates_by_label"] = pd.DataFrame(
                {
                    "label": vocab,
                    "clean_rows": rows_per_label,
                    "duplicates": rows_per_label - unique_per_label,
                    "duplicate_pct": 100.0
                    * (rows_per_label - unique_per_label)
                    / np.maximum(rows_per_label, 1),
                    "rows_after_dedup": unique_per_label,
                }
            ).sort_values("clean_rows", ascending=False, ignore_index=True)
            in_conflict = np.isin(pair_feature, conflict_vectors)
            conflict_frame = pd.DataFrame(
                {"vector": pair_feature[in_conflict], "label": np.asarray(vocab, dtype=object)[pair_label[in_conflict]]}
            )
            label_sets = conflict_frame.groupby("vector")["label"].agg(lambda s: " | ".join(sorted(s)))
            tables["label_conflicts"] = (
                label_sets.value_counts().rename_axis("labels_sharing_a_feature_vector").reset_index(name="vectors")
            )
    report["duplicates"] = duplicates
    del feature, feature_label, clean, codes

    # --- sample-based structure: identical columns, Spearman, class medians ---
    finite_sample = uniform.x[np.isfinite(uniform.x).all(axis=1)]
    varying = finite_sample.std(axis=0) > 0 if len(finite_sample) else np.zeros(p, dtype=bool)
    varying_idx = np.flatnonzero(varying)
    identical_rows = []
    for a_pos, a in enumerate(varying_idx):
        others = varying_idx[a_pos + 1 :]
        if not len(others):
            continue
        # CSV decimals of the same quantity can differ in the last digits, so compare
        # with a relative tolerance rather than bitwise.
        equal = np.isclose(
            finite_sample[:, [a]], finite_sample[:, others], rtol=IDENTICAL_RTOL, atol=0.0
        ).mean(axis=0)
        for b, fraction in zip(others, equal):
            if fraction >= 0.999:
                identical_rows.append(
                    {"feature_a": feature_columns[a], "feature_b": feature_columns[b], "equal_fraction": fraction}
                )
    tables["identical_columns_sample"] = pd.DataFrame(
        identical_rows, columns=["feature_a", "feature_b", "equal_fraction"]
    )
    spearman = pd.DataFrame(np.nan, index=feature_columns, columns=feature_columns)
    if len(varying_idx) > 1:
        ranks = pd.DataFrame(finite_sample[:, varying_idx]).rank(method="average").to_numpy()
        rho = np.corrcoef(ranks, rowvar=False)
        names = [feature_columns[i] for i in varying_idx]
        spearman.loc[names, names] = rho
    tables["spearman_sample"] = spearman.reset_index(names="feature")
    upper = np.triu(np.ones(spearman.shape, dtype=bool), k=1)
    high = spearman.where(upper).stack()
    high = high[high.abs() >= 0.99].sort_values(key=np.abs, ascending=False)
    tables["high_spearman_pairs"] = high.rename_axis(["feature_a", "feature_b"]).reset_index(name="spearman_rho")

    selected = [name for name in SELECTED_FEATURES if name in feature_columns]
    selected_idx = [feature_columns.index(name) for name in selected]
    median_rows = []
    for label in labels_frame["label"]:
        sample = per_label[label].x[:, selected_idx]
        sample = np.where(np.isfinite(sample), sample, np.nan)
        medians = np.nanmedian(sample, axis=0) if len(sample) else np.full(len(selected), np.nan)
        median_rows.append({"label": label, "sampled_rows": len(sample), **dict(zip(selected, medians))})
    tables["label_medians_sample"] = pd.DataFrame(median_rows)
    report["structure"] = {
        "constant_columns": column_frame.loc[column_frame["constant"], "feature"].tolist(),
        "integer_valued_columns": column_frame.loc[column_frame["integer_valued"], "feature"].tolist(),
        "non_integer_columns": column_frame.loc[~column_frame["integer_valued"], "feature"].tolist(),
        "uniform_sample_rows": int(len(uniform.keys)),
        "uniform_sample_all_finite_rows": int(len(finite_sample)),
        "per_label_sample_cap": per_label_k,
    }

    # --- imbalance ---
    attack = labels_frame[labels_frame["label"] != BENIGN_LABEL]
    report["imbalance"] = {
        "largest_attack_label": attack.iloc[0]["label"] if len(attack) else None,
        "largest_attack_rows": int(attack.iloc[0]["rows"]) if len(attack) else 0,
        "smallest_label": labels_frame.iloc[-1]["label"],
        "smallest_label_rows": int(labels_frame.iloc[-1]["rows"]),
        "largest_to_smallest_ratio": float(labels_frame.iloc[0]["rows"] / labels_frame.iloc[-1]["rows"]),
    }
    return report, tables, uniform, per_label


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _label_colors(labels: list[str]) -> dict[str, Any]:
    # Saturated qualitative colours only; grey is reserved for BENIGN.
    palette = [
        *[c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7],
        *[c for i, c in enumerate(plt.get_cmap("Dark2").colors) if i != 7],
        *[c for i, c in enumerate(plt.get_cmap("Set1").colors) if i != 8],
    ]
    colors: dict[str, Any] = {BENIGN_LABEL: "0.55"}
    others = [label for label in labels if label != BENIGN_LABEL]
    for i, label in enumerate(others):
        colors[label] = palette[i % len(palette)]
    return colors


def plot_label_distribution(frame: pd.DataFrame, output: Path) -> None:
    ordered = frame.sort_values("rows")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(ordered))))
    colors = ["tab:orange" if attempted else "tab:blue" for attempted in ordered["attempted"]]
    colors = ["0.55" if label == BENIGN_LABEL else c for label, c in zip(ordered["label"], colors)]
    ax.barh(ordered["label"], ordered["rows"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title("Raw label distribution (orange = Attempted, grey = BENIGN)")
    for y, value in enumerate(ordered["rows"]):
        ax.text(value, y, f" {value:,}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_attack_timeline(scans: list[FileScan], colors: dict[str, Any], output: Path) -> None:
    """Flows per 10 min by label, one panel per file, restricted to the capture window
    [capture date 00:00, +TIMELINE_WINDOW_HOURS); rows outside it are counted in the title."""
    fig, axes = plt.subplots(len(scans), 1, figsize=(14, 2.6 * len(scans)), sharex=True)
    axes = np.atleast_1d(axes)
    span = [np.inf, -np.inf]
    for ax, scan in zip(axes, scans):
        day_start = int(np.datetime64(EXPECTED_FILES[scan.name], "D").astype("datetime64[us]").astype(np.int64))
        per_label: dict[str, list[tuple[float, int]]] = {}
        outside = 0
        for (label, bucket), n in scan.buckets.items():
            hour = (int(bucket) * BUCKET_SECONDS * US_PER_SECOND - day_start) / (3600 * US_PER_SECOND)
            if 0 <= hour < TIMELINE_WINDOW_HOURS:
                per_label.setdefault(label, []).append((hour, n))
                span[0], span[1] = min(span[0], hour), max(span[1], hour)
            else:
                outside += n
        for label, points in sorted(per_label.items(), key=lambda item: item[0] != BENIGN_LABEL):
            points.sort()
            hours, counts = zip(*points)
            ax.plot(
                hours,
                counts,
                drawstyle="steps-post",
                color=colors.get(label, "black"),
                lw=1.0 if label == BENIGN_LABEL else 1.4,
                label=label,
            )
        ax.set_yscale("log")
        ax.set_ylabel("flows / 10 min")
        title = scan.name.removesuffix(".csv")
        if outside:
            title += f"  ({outside:,} rows outside the plotted window)"
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xlabel("hours after 00:00 of the file's capture date (timestamp as recorded)")
    if np.isfinite(span[0]):
        axes[-1].set_xlim(np.floor(span[0]) - 0.5, np.ceil(span[1]) + 0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_feature_histograms(
    per_label: dict[str, BottomK],
    label_order: list[str],
    feature_columns: list[str],
    colors: dict[str, Any],
    output: Path,
) -> None:
    selected = [name for name in SELECTED_FEATURES if name in feature_columns]
    attacks = [label for label in label_order if label != BENIGN_LABEL and ATTEMPTED_MARKER not in label][:5]
    groups = [label for label in (BENIGN_LABEL, *attacks) if label in per_label]
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    for ax, name in zip(np.ravel(axes), selected):
        i = feature_columns.index(name)
        for label in groups:
            values = per_label[label].x[:, i]
            values = _signed_log1p(values[np.isfinite(values)])
            if len(values):
                ax.hist(values, bins=60, density=True, histtype="step", color=colors.get(label), label=label)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("sign(x)·log1p(|x|)", fontsize=8)
        # Tool-generated attack flows are near-constant spikes; log density keeps BENIGN visible.
        ax.set_yscale("log")
    for ax in np.ravel(axes)[len(selected) :]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=6)
    fig.suptitle("Selected features: BENIGN vs five largest attack labels (per-label sample)")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_spearman(spearman: pd.DataFrame, output: Path) -> None:
    matrix = spearman.set_index("feature")
    fig, ax = plt.subplots(figsize=(17, 15))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=5)
    fig.colorbar(image, ax=ax, fraction=0.03, label="Spearman rho (uniform sample, finite rows)")
    ax.set_title("Spearman rank correlation (blank = constant in sample)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_pca(
    per_label: dict[str, BottomK],
    label_order: list[str],
    colors: dict[str, Any],
    seed: int,
    output: Path,
) -> dict[str, Any]:
    x = np.concatenate([per_label[label].x for label in label_order])
    labels = np.concatenate([per_label[label].labels for label in label_order])
    keep = np.isfinite(x).all(axis=1)
    x, labels = _signed_log1p(x[keep]), labels[keep]
    varying = x.std(axis=0) > 0
    x = x[:, varying]
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in label_order:
        selected = labels == label
        if selected.any():
            ax.scatter(
                coords[selected, 0],
                coords[selected, 1],
                s=3,
                alpha=0.35,
                color=colors.get(label),
                label=label,
                rasterized=True,
            )
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("PCA of per-label stratified sample (signed log1p, standardized; descriptive only)")
    ax.legend(markerscale=4, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return {
        "rows": int(len(coords)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—" if np.isnan(value) else str(value)
        if float(value).is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.6g}"
    return str(value).replace("|", "\\|")


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_\n"
    header = "| " + " | ".join(str(c).replace("|", "\\|") for c in frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator, *body]) + "\n"


def write_markdown(
    output: Path,
    report: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    inv = report["inventory"]
    schema = report["schema"]
    comparison = schema["cicids2017_comparison"]
    labels = report["labels"]
    card = report["cardinality"]
    identity = report["row_identity"]
    ts = report["timestamps"]
    cleaning = report["cleaning"]
    dup = report["duplicates"]
    structure = report["structure"]
    imbalance = report["imbalance"]
    summary = tables["feature_summary"]
    quantile_columns = ["feature", "min", *[f"q{q * 100:g}" for q in QUANTILES], "max", "zero_pct"]
    selected_summary = summary[summary["feature"].isin(SELECTED_FEATURES)][quantile_columns]

    lines = [
        "# CSE-CIC-IDS-2018 DistriNet — raw-data EDA report",
        "",
        f"Generated by `src/evaluation/cicids2018_distrinet_eda.py` in {report['elapsed_seconds']:,.0f} s "
        f"(seed {args.seed}). Input: `{report['input_dir']}`.",
        "",
        "All statistics describe the **raw release before any split**. They are descriptive only and are "
        "never used to fit preprocessing, masks, rules, or thresholds (train-only by thesis policy). "
        "Counts are exact over every row unless marked *sample*; duplicate/`Flow ID` analyses use 64-bit row hashes "
        f"(expected collisions ≈ {dup['expected_hash_collisions']:.2g}).",
        "",
    ]
    if report["non_production"]:
        lines += [f"> **Limited run:** `--max-rows-per-file {args.max_rows_per_file}`; numbers are not dataset totals.", ""]
    lines += [
        "Guide to every section: `docs/data/cicids2018distrinet/eda_explained.md`.",
        "",
        "## 1. File inventory",
        "",
        f"{inv['files']} files, {inv['total_rows']:,} rows, {inv['total_bytes']:,} bytes.",
        "",
        md_table(tables["file_inventory"]),
        "## 2. Schema",
        "",
        f"All headers identical: **{schema['raw_column_count']} columns**; "
        f"{len(schema['non_feature_columns'])} non-feature columns "
        f"(`{'`, `'.join(schema['non_feature_columns'])}`) and "
        f"**{schema['numeric_candidate_count']} numeric candidate columns**.",
        "",
    ]
    if comparison.get("manifest"):
        lines += [
            f"Against the CICIDS2017 modelling features (`{comparison['manifest']}`, "
            f"{comparison['cicids2017_feature_count']} features): {len(comparison['shared'])} shared "
            f"(same relative order: {_fmt(comparison['shared_in_same_relative_order'])}).",
            "",
            f"- only in 2018: {', '.join(f'`{c}`' for c in comparison['only_in_2018']) or 'none'}",
            f"- only in 2017: {', '.join(f'`{c}`' for c in comparison['only_in_2017']) or 'none'}",
            "",
        ]
    else:
        lines += [f"CICIDS2017 comparison unavailable: {comparison['reason']}.", ""]
    lines += [
        "## 3. Raw label distribution",
        "",
        f"{labels['distinct_labels']} literal labels. BENIGN: {labels['benign_rows']:,} rows "
        f"({labels['benign_share_pct']:.4f}%). Labels containing `Attempted`: "
        f"{labels['attempted_label_rows']:,} rows ({labels['attempted_label_share_pct']:.4f}%). "
        f"cp1252 label-decoding fallbacks: {labels['label_decode_cp1252_fallbacks']}.",
        "",
        md_table(tables["label_distribution"]),
        "![label distribution](figures/label_distribution.png)",
        "",
        "### 3.1 Labels by capture day",
        "",
        md_table(tables["label_by_day"]),
        "### 3.2 `Attempted Category` vs `Label`",
        "",
        md_table(tables["attempted_category_crosstab"]),
        "## 4. Attack schedule by day",
        "",
        "First/last timestamp of each non-BENIGN label per file (timestamp as recorded).",
        "",
        md_table(tables["attack_schedule"]),
        "![attack timeline](figures/attack_timeline.png)",
        "",
        "## 5. Protocol and endpoint cardinality",
        "",
        f"Protocol rows: {', '.join(f'{k}: {v:,}' for k, v in card['protocol_rows'].items())}. "
        f"Distinct Src IP {card['distinct_src_ip']:,}; Dst IP {card['distinct_dst_ip']:,}; "
        f"Src Port {card['distinct_src_port']:,}; Dst Port {card['distinct_dst_port']:,}.",
        "",
        "### 5.1 Protocol by label",
        "",
        md_table(tables["protocol_by_label"]),
        "### 5.2 Attack endpoints (top 3 by share)",
        "",
        md_table(tables["attack_endpoints"]),
        "## 6. Row identity: `id` and `Flow ID`",
        "",
        f"- `id` equals the 1-based row number in every file: {_fmt(identity['id_is_1_based_row_number_in_every_file'])}",
        f"- distinct `Flow ID`: {identity['flow_id_distinct']:,}; rows beyond first occurrence: "
        f"{identity['flow_id_rows_beyond_first']:,}; max multiplicity: {identity['flow_id_max_multiplicity']:,}; "
        f"`Flow ID`s present in ≥2 files: {identity['flow_id_in_two_or_more_files']:,}",
        "",
        "## 7. Timestamps",
        "",
        f"Missing {ts['missing']:,}; unparseable {ts['unparseable']:,}; with fractional seconds "
        f"{ts['with_fractional_seconds']:,}; date differs from file name {ts['date_differs_from_filename']:,}; "
        f"adjacent reversals in file order {ts['adjacent_reversals']:,}.",
        "",
        md_table(tables["timestamp_audit"]),
        "## 8. Missing and infinite values",
        "",
        f"Rows with at least one non-finite numeric value: **{cleaning['rows_any_nonfinite']:,}**.",
        "",
        md_table(tables["nonfinite_columns"]),
        "## 9. Negative values and sentinels",
        "",
        f"Rows with any negative value (all numeric columns): {cleaning['rows_any_negative_all_columns']:,}. "
        f"Finite rows with a negative value other than the ICMP `-1` sentinel: "
        f"**{cleaning['rows_negative_excluding_icmp_sentinel_among_finite']:,}**. "
        f"ICMP negatives are exclusively `-1`: "
        f"{', '.join(f'`{k}` {_fmt(v)}' for k, v in cleaning['icmp_negatives_are_only_minus1'].items())}.",
        "",
        md_table(tables["negative_columns"]),
        "Sentinel consistency checks:",
        "",
        *[f"- {name}: {value:,}" for name, value in cleaning["sentinel_checks"].items()],
        "",
        "### 9.1 Row-local cleaning impact by label",
        "",
        f"Applying the CICIDS2017 rule (drop non-finite rows and rows with a negative value, exempting the "
        f"ICMP `-1` sentinel) removes **{cleaning['rows_removed_by_2017_style_rule_with_icmp_sentinel_exempt']:,}** rows.",
        "",
        md_table(tables["cleaning_impact_by_label"]),
        "## 10. Duplicates and label conflicts",
        "",
        f"- identical raw rows (all columns except `id`): {dup['full_raw_row_excluding_id']:,}",
    ]
    for scope, title in (("all_rows", "all rows"), ("clean_rows", "rows surviving §9.1 cleaning")):
        d = dup[scope]
        lines.append(
            f"- {title}: {d['rows']:,} rows; float32 features + label duplicates "
            f"{d['float32_features_plus_label_duplicates']:,} ({d['float32_features_plus_label_duplicate_pct']:.4f}%) "
            f"→ {d['rows_after_dedup']:,} rows; feature-only duplicates {d['float32_feature_only_duplicates']:,}; "
            f"feature vectors carrying >1 label {d['conflicting_feature_vectors']:,} "
            f"({d['rows_with_conflicting_feature_vector']:,} rows)"
        )
    lines += [
        "",
        "### 10.1 Duplicates by label (clean rows)",
        "",
        md_table(tables["duplicates_by_label"]),
        "### 10.2 Label sets sharing an identical feature vector (clean rows)",
        "",
        md_table(tables["label_conflicts"].head(25)),
        "## 11. Feature scale and heavy tails",
        "",
        f"Exact min/max; quantiles from a uniform *sample* of {structure['uniform_sample_rows']:,} rows "
        "(non-finite values ignored). Full table: `tables/feature_summary.csv`.",
        "",
        md_table(selected_summary),
        "![feature histograms](figures/feature_histograms.png)",
        "",
        "### 11.1 Class-conditional medians (*sample*, ≤"
        f"{structure['per_label_sample_cap']:,} rows per label)",
        "",
        md_table(tables["label_medians_sample"]),
        "## 12. Column structure",
        "",
        f"- constant columns (exact): {', '.join(f'`{c}`' for c in structure['constant_columns']) or 'none'}",
        f"- non-integer columns (exact): {', '.join(f'`{c}`' for c in structure['non_integer_columns']) or 'none'}",
        f"- integer-valued columns: {len(structure['integer_valued_columns'])} of {schema['numeric_candidate_count']}",
        "",
        f"### 12.1 Near-identical column pairs (*sample*, ≥99.9% of rows equal within rtol {IDENTICAL_RTOL:g}, finite rows)",
        "",
        md_table(tables["identical_columns_sample"]),
        "### 12.2 Spearman |rho| ≥ 0.99 (*sample*)",
        "",
        md_table(tables["high_spearman_pairs"]),
        "![spearman](figures/spearman_heatmap.png)",
        "",
        "## 13. Class imbalance",
        "",
        f"Largest attack label: {imbalance['largest_attack_label']} ({imbalance['largest_attack_rows']:,}). "
        f"Smallest label: {imbalance['smallest_label']} ({imbalance['smallest_label_rows']:,}). "
        f"Largest/smallest ratio: {imbalance['largest_to_smallest_ratio']:,.1f}. "
        f"Labels with <1,000 rows: {', '.join(labels['labels_below_1000_rows']) or 'none'}. "
        f"Labels with <100 rows: {', '.join(labels['labels_below_100_rows']) or 'none'}.",
        "",
        "## 14. PCA (*per-label sample*, descriptive only)",
        "",
        f"{report['pca']['rows']:,} finite rows; explained variance PC1/PC2 = "
        f"{', '.join(f'{100 * v:.1f}%' for v in report['pca']['explained_variance_ratio'])}.",
        "",
        "![pca](figures/pca_stratified.png)",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "CSECICIDS2018_Distrinet")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cicids2018distrinet" / "eda")
    parser.add_argument("--workers", type=int, default=4, help="Files scanned in parallel (1 = in-process).")
    parser.add_argument("--uniform-sample", type=int, default=250_000)
    parser.add_argument("--per-label-sample", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-rows-per-file", type=int, default=None, help="Smoke-test limit; report is marked limited."
    )
    parser.add_argument("--skip-input-hashes", action="store_true", help="Skip SHA-256 of the raw CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.uniform_sample < 1 or args.per_label_sample < 1:
        raise ValueError("--workers, --uniform-sample and --per-label-sample must be >= 1")
    started = time.time()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    paths, header, feature_columns = validate_inventory(input_dir)
    print(f"Validated {len(paths)} files: {len(header)} columns, {len(feature_columns)} numeric candidates")
    tasks = [
        ScanTask(
            path=path,
            file_index=i,
            expected_date=EXPECTED_FILES[path.name],
            raw_columns=header,
            feature_columns=feature_columns,
            max_rows=args.max_rows_per_file,
            hash_input=not args.skip_input_hashes and args.max_rows_per_file is None,
            uniform_k=args.uniform_sample,
            per_label_k=args.per_label_sample,
            seed=args.seed,
        )
        for i, path in enumerate(paths)
    ]
    scans: list[FileScan] = []
    if args.workers == 1:
        for task in tasks:
            scans.append(scan_file(task))
            print(f"  {task.path.name}: {scans[-1].rows:,} rows ({time.time() - started:,.0f} s)")
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            for scan in pool.map(scan_file, tasks):
                scans.append(scan)
                print(f"  {scan.name}: {scan.rows:,} rows ({time.time() - started:,.0f} s)")
    scans.sort(key=lambda scan: scan.file_index)

    report, tables, _, per_label = analyse(
        scans, header, feature_columns, args.uniform_sample, args.per_label_sample
    )
    label_order = tables["label_distribution"]["label"].tolist()
    colors = _label_colors(label_order)
    figures = output_dir / "figures"
    plot_label_distribution(tables["label_distribution"], figures / "label_distribution.png")
    plot_attack_timeline(scans, colors, figures / "attack_timeline.png")
    plot_feature_histograms(per_label, label_order, feature_columns, colors, figures / "feature_histograms.png")
    plot_spearman(tables["spearman_sample"], figures / "spearman_heatmap.png")
    report["pca"] = plot_pca(per_label, label_order, colors, args.seed, figures / "pca_stratified.png")

    report["input_dir"] = str(input_dir)
    report["non_production"] = args.max_rows_per_file is not None
    report["arguments"] = {
        "max_rows_per_file": args.max_rows_per_file,
        "uniform_sample": args.uniform_sample,
        "per_label_sample": args.per_label_sample,
        "seed": args.seed,
        "input_hashes": tasks[0].hash_input,
    }
    report["elapsed_seconds"] = round(time.time() - started, 1)
    for name, frame in tables.items():
        frame.to_csv(output_dir / "tables" / f"{name}.csv", index=False)
    with (output_dir / "eda_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
        handle.write("\n")
    write_markdown(output_dir / "cicids2018_distrinet_eda.md", report, tables, args)
    print(f"EDA complete in {report['elapsed_seconds']:,.0f} s: {output_dir}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    main()
