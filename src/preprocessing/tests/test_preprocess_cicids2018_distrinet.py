"""CICIDS2018 DistriNet preprocessing: Benign quota allocation and end-to-end invariants.

The end-to-end test runs the real pipeline on ten small synthetic CSVs that reproduce the
DistriNet schema (91 -> reduced columns, same metadata/extra-column structure).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from preprocessing import preprocess_cicids2018_distrinet as prep

FEATURES = [
    "Src Port",
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Fwd Header Length",
    "Bwd Header Length",
]
HEADER = [
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Fwd RST Flags", "Bwd RST Flags",
    "Fwd Header Length", "Bwd Header Length", "ICMP Code", "ICMP Type", "Total TCP Flow Time",
    "Label", "Attempted Category",
]
ATTACKS = [label for label in prep.SOURCE_LABELS if label != "BENIGN"]


# --------------------------------------------------------------------------- #
# Quota allocation
# --------------------------------------------------------------------------- #
def test_quotas_sum_to_cap_and_follow_population():
    sizes = np.array([10_000, 5_000, 2_500, 2_500])
    quotas = prep.allocate_stratum_quotas(sizes, 1_000)
    assert quotas.sum() == 1_000
    assert quotas.tolist() == [500, 250, 125, 125]


def test_tiny_strata_keep_one_row_and_never_exceed_size():
    sizes = np.array([1_000_000, 3, 1, 2])
    quotas = prep.allocate_stratum_quotas(sizes, 10)
    assert quotas.sum() == 10
    assert (quotas[1:] >= 1).all()
    assert (quotas <= sizes).all()


def test_quotas_never_exceed_stratum_size():
    sizes = np.array([2, 2, 100])
    quotas = prep.allocate_stratum_quotas(sizes, 90)
    assert quotas.sum() == 90
    assert (quotas <= sizes).all()


def test_population_below_cap_is_kept_whole():
    sizes = np.array([3, 4])
    assert prep.allocate_stratum_quotas(sizes, 100).tolist() == [3, 4]


# --------------------------------------------------------------------------- #
# End-to-end on synthetic DistriNet-shaped files
# --------------------------------------------------------------------------- #
def _row(i, label, ts, rng, *, fwd_header=None, duration=None, protocol=6):
    fwd_packets = int(rng.integers(1, 50))
    return {
        "Flow ID": f"f{i}", "Src IP": "10.0.0.1", "Src Port": int(rng.integers(1024, 65535)),
        "Dst IP": "10.0.0.2", "Dst Port": 80, "Protocol": protocol,
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "Flow Duration": float(rng.integers(1, 10**9)) if duration is None else duration,
        "Total Fwd Packet": fwd_packets, "Total Bwd packets": int(rng.integers(0, 50)),
        "Fwd RST Flags": 0, "Bwd RST Flags": 0,
        "Fwd Header Length": 20 * fwd_packets if fwd_header is None else fwd_header,
        "Bwd Header Length": 0, "ICMP Code": -1, "ICMP Type": -1, "Total TCP Flow Time": 0,
        "Label": label, "Attempted Category": 0 if label.endswith(" - Attempted") else -1,
    }


def _write_dataset(root, extra_labels=()):
    rng = np.random.default_rng(0)
    raw = root / "raw"
    raw.mkdir()
    counter = 0
    special = {}
    for f, (name, date) in enumerate(prep.EXPECTED_FILES.items()):
        start = pd.Timestamp(date) + pd.Timedelta(hours=8)
        rows = []
        for k in range(300):
            rows.append(_row(counter, "BENIGN", start + pd.Timedelta(seconds=int(rng.integers(0, 6 * 3600))), rng))
            counter += 1
        attack = ATTACKS[f % len(ATTACKS)]
        for k in range(40):
            rows.append(_row(counter, attack, start + pd.Timedelta(minutes=30, seconds=k), rng))
            counter += 1
        for label in (*prep.DROPPED_LABELS, *prep.ATTEMPTED_LABELS[:2], *extra_labels):
            rows.append(_row(counter, label, start + pd.Timedelta(minutes=45), rng))
            counter += 1
        if f == 0:
            inf_row = _row(counter, "BENIGN", start, rng, duration=float("inf"))
            first = _row(counter + 1, "BENIGN", start + pd.Timedelta(minutes=1), rng)
            later = dict(first, **{"Flow ID": "dup", "Timestamp": (start + pd.Timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S.%f")})
            wrapped = _row(counter + 2, "DDoS-LOIC-UDP", start + pd.Timedelta(minutes=31), rng, fwd_header=-32000, protocol=17)
            rows = [later, *rows, inf_row, first, wrapped]  # file order deliberately not chronological
            special = {"inf": inf_row["Flow ID"], "dup_first": first["Flow ID"], "wrapped": wrapped["Flow ID"]}
            counter += 3
        frame = pd.DataFrame(rows)
        frame.insert(0, "id", np.arange(1, len(frame) + 1))
        frame[HEADER].to_csv(raw / name, index=False)
    manifest = root / "reference_manifest.json"
    manifest.write_text(json.dumps({"modelling_feature_names": FEATURES}), encoding="utf-8")
    return raw, manifest, special


TARGETS = "Benign=600,DoS=100,DDoS=60"
TARGET_TOTALS = {"Benign": 600, "DoS": 100, "DDoS": 60}
UNTARGETED = ("Recon", "BruteForce")


def _run(root, raw, manifest, *, seed=42, targets=TARGETS, tag="out"):
    return prep.main([
        "--input-dir", str(raw), "--output-dir", str(root / tag), "--report-dir", str(root / f"{tag}_rep"),
        "--reference-manifest", str(manifest), "--expected-feature-count", str(len(FEATURES)),
        "--class-row-targets", targets, "--seed", str(seed),
        "--workers", "1", "--spearman-sample", "200", "--pca-per-class", "20",
    ])


def _splits(out):
    return {name: pd.read_parquet(out / f"{name}.parquet") for name in prep.SPLIT_NAMES}


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cicids2018")
    raw, manifest, special = _write_dataset(root)
    result = _run(root, raw, manifest)
    _run(root, raw, manifest, tag="repeat")
    _run(root, raw, manifest, seed=7, tag="seed7")
    return result, root, _splits(root / "out"), special


def test_class_row_targets_are_parsed_and_apportioned_70_15_15():
    assert prep.parse_class_row_targets("Benign=250000, DoS=200000") == {"Benign": 250_000, "DoS": 200_000}
    assert prep.split_row_targets(250_000) == {"train": 175_000, "val": 37_500, "test": 37_500}
    assert sum(prep.split_row_targets(200_001).values()) == 200_001
    with pytest.raises(ValueError):
        prep.parse_class_row_targets("Botnet=10")
    with pytest.raises(ValueError):
        prep.parse_class_row_targets("DoS=2")


def test_only_targeted_classes_are_reduced_to_their_totals(pipeline_run):
    _, root, splits, _ = pipeline_run
    out = root / "out"
    reduction = pd.read_csv(out / "class_reduction_by_split.csv")
    untouched = reduction[reduction["category_label"].isin(UNTARGETED)]
    assert (untouched["before"] == untouched["after"]).all() and (untouched["after"] > 0).all()
    for label, total in TARGET_TOTALS.items():
        rows = reduction[reduction["category_label"] == label].set_index("split")
        expected = prep.split_row_targets(total)
        assert rows["after"].to_dict() == expected
        assert (rows["after"] < rows["before"]).all()
        assert sum((frame["category_label"] == label).sum() for frame in splits.values()) == total
    classes = pd.read_csv(out / "class_distribution.csv").set_index("category_label").loc[list(UNTARGETED)]
    assert (classes["train"] + classes["val"] + classes["test"] == classes["total"]).all()


def test_seed_changes_only_the_sampled_subsets_and_is_reproducible(pipeline_run):
    _, root, splits, _ = pipeline_run
    repeat, other = _splits(root / "repeat"), _splits(root / "seed7")
    index = {tag: pd.read_parquet(root / tag / "row_index.parquet") for tag in ("out", "seed7")}
    assert index["out"]["split"].astype(str).equals(index["seed7"]["split"].astype(str))
    changed = False
    for name, frame in splits.items():
        assert frame["sample_id"].equals(repeat[name]["sample_id"])
        assert np.array_equal(np.load(root / "out" / f"X_{name}.npy"), np.load(root / "repeat" / f"X_{name}.npy"))
        kept_whole = frame["category_label"].isin(UNTARGETED)
        other_whole = other[name]["category_label"].isin(UNTARGETED)
        assert frame.loc[kept_whole, "sample_id"].tolist() == other[name].loc[other_whole, "sample_id"].tolist()
        assert frame["category_label"].value_counts().equals(other[name]["category_label"].value_counts())
        changed |= set(frame.loc[~kept_whole, "sample_id"]) != set(other[name].loc[~other_whole, "sample_id"])
    assert changed


def test_temporal_strata_stay_represented_when_the_target_allows(pipeline_run):
    _, root, _, _ = pipeline_run
    strata = pd.read_csv(root / "out" / "sampling_strata.csv")
    audit = json.loads((root / "out" / "sampling_audit.json").read_text())["samples"]
    for (name, label), part in strata.groupby(["split", "category_label"]):
        assert part["quota"].le(part["population"]).all()
        if audit[f"{name}/{label}"]["target"] >= len(part):
            assert (part["selected"] >= 1).all(), (name, label)


def test_scaler_is_fitted_on_final_training_rows_only(pipeline_run):
    _, root, _, _ = pipeline_run
    with (root / "out" / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    train = np.load(root / "out" / "X_train_pristine.npy")
    assert np.allclose(scaler.center_, np.median(train, axis=0))
    everything = np.concatenate([train, *(np.load(root / "out" / f"X_{s}_pristine.npy") for s in ("val", "test"))])
    assert not np.allclose(scaler.center_, np.median(everything, axis=0))


def test_no_feature_label_duplicates_in_outputs(pipeline_run):
    _, _, splits, _ = pipeline_run
    rows = pd.concat(splits.values())
    assert not rows.duplicated(subset=[*FEATURES, "category_label"]).any()


def test_cleaning_dedup_and_label_policy(pipeline_run):
    _, root, splits, special = pipeline_run
    out = root / "out"
    rows = pd.concat(splits.values())
    assert special["inf"] not in set(rows["Flow ID"])
    assert set(rows["category_label"]) <= set(prep.CATEGORY_NAMES)
    assert not set(rows["original_label"]) & set(prep.DROPPED_LABELS)
    index = pd.read_parquet(out / "row_index.parquet")
    attempted = index[index["original_label"].astype(str).str.endswith(" - Attempted")]
    assert len(attempted) and (attempted["category_label"] == "Benign").all()
    flagged = index[index["fwd_header_length_negative"]]
    assert len(flagged) == 1 and flagged["category_label"].iloc[0] == "DDoS"
    kept_flagged = rows[rows["fwd_header_length_negative"]]
    assert (kept_flagged["Fwd Header Length"] == -32000).all()
    duplicates = index[index["is_duplicate"]]
    assert len(duplicates) == 1
    survivor = index[(index["source_file"] == duplicates["duplicate_of_source_file"].iloc[0])
                     & (index["source_id"] == duplicates["duplicate_of_source_id"].iloc[0])]
    assert survivor["timestamp_epoch_us"].iloc[0] < duplicates["timestamp_epoch_us"].iloc[0]


def test_splits_are_disjoint_chronological_and_aligned(pipeline_run):
    _, root, splits, _ = pipeline_run
    out = root / "out"
    ids = {name: set(frame["sample_id"]) for name, frame in splits.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    for label in prep.SOURCE_LABELS:
        times = {name: frame.loc[frame["source_label"] == label, "timestamp_epoch_us"] for name, frame in splits.items()}
        assert all(len(t) for t in times.values())
        assert times["val"].min() >= times["train"].max()
        assert times["test"].min() >= times["val"].max()
    for name, frame in splits.items():
        x = np.load(out / f"X_{name}_pristine.npy")
        y = np.load(out / f"y_{name}_cat.npy")
        assert np.array_equal(x, frame[FEATURES].to_numpy(np.float32))
        assert np.array_equal(y, frame["category_label"].map(prep.CATEGORY_TO_ID).to_numpy())
        assert np.isfinite(np.load(out / f"X_{name}.npy")).all()


def test_unknown_label_aborts(tmp_path):
    raw, manifest, _ = _write_dataset(tmp_path, extra_labels=("Brand New Attack",))
    with pytest.raises(ValueError, match="Brand New Attack"):
        _run(tmp_path, raw, manifest)
