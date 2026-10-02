"""Audit fresh clean-correct cohorts and summarize untargeted Prim-PGD p75/unbounded.

Only seed 42 has a prespecified paired inferential comparison; different seeds
sample different test flows and are never treated as repeated measurements of rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from datasets import get_adapter  # noqa: E402
from evaluation.paired_validity_gap import (  # noqa: E402
    holm_adjust, mcnemar_test, newcombe_paired_ci, wilson_score_interval,
)
from run_full_adversarial_eval import file_sha256  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from validation.attack_interface import get_validator  # noqa: E402

SEEDS = (42, 2024, 2026)
CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
VICTIMS = {
    "cicids2017_distrinet": ("mlp", "cnn", "ft_transformer"),
    "cicids2018_distrinet": ("mlp-s42", "cnn-s42", "ft_transformer-s42"),
}
BUDGETS = ("p75", "unbounded")
FILE_BUDGET = {"p75": "p75", "unbounded": "unb"}
N_PER_CLASS = 800
BOOT_REPS = 10000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def id_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest()


def raw_hash(raw: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()


def same(a: np.ndarray, b: np.ndarray, context: str) -> None:
    require(np.array_equal(a, b), context)


def scalar(data: dict, field: str, expected: object, where: str) -> None:
    require(field in data and np.asarray(data[field]).ndim == 0 and
            np.asarray(data[field]).item() == expected,
            f"{where}: {field} differs from expected {expected!r}")


def mask(data: dict, field: str, expected: np.ndarray, where: str) -> None:
    require(field in data and np.asarray(data[field]).dtype == np.bool_,
            f"{where}: missing/non-boolean mask {field}")
    same(data[field], expected, f"{where}: {field} disagrees with recomputation")


def predict(victim, raw: np.ndarray, center: torch.Tensor, scale: torch.Tensor,
            device: str) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(raw), 1024):
            x = torch.as_tensor(np.asarray(raw[offset:offset + 1024]), device=device)
            outputs.append(victim((x - center) / scale).argmax(1).cpu().numpy())
    return np.concatenate(outputs)


def cp_interval(k: int, n: int) -> tuple[float, float]:
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def bootstrap(values: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI resampling rows within each fixed class (and seed) stratum."""
    total = sum(len(x) for x in values)
    estimates = np.zeros(BOOT_REPS, dtype=np.float64)
    for x in values:
        x = np.asarray(x, dtype=np.float64)
        # Chunk to bound allocation while preserving a deterministic RNG stream.
        for start in range(0, BOOT_REPS, 500):
            end = min(start + 500, BOOT_REPS)
            draws = rng.integers(0, len(x), size=(end - start, len(x)))
            estimates[start:end] += x[draws].sum(axis=1) / total
    return tuple(float(z) for z in np.quantile(estimates, [.025, .975]))


def paired(a: np.ndarray, b: np.ndarray, **labels) -> dict:
    require(a.shape == b.shape and a.ndim == 1, "paired outcomes require aligned vectors")
    both = int(np.count_nonzero(a & b))
    a_only = int(np.count_nonzero(a & ~b))
    b_only = int(np.count_nonzero(~a & b))
    neither = int(np.count_nonzero(~a & ~b))
    test = mcnemar_test(a_only, b_only)
    low, high = newcombe_paired_ci(both, a_only, b_only, neither)
    return {**labels, "seed": 42, "n": len(a), "rate_a": float(a.mean()),
            "rate_b": float(b.mean()), "difference_pp": 100 * float(a.mean() - b.mean()),
            "difference_ci_low_pp": 100 * low, "difference_ci_high_pp": 100 * high,
            "both": both, "a_only": a_only, "b_only": b_only, "neither": neither,
            "variant": test["test_variant"], "statistic": test["test_statistic"],
            "p_value": test["p_value"], "p_holm": None, "decision": "descriptive only"}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_dir: Path, device: str) -> None:
    expected = [results_dir / dataset / f"seed{seed}" / "primattack" / "artifacts" /
                f"{victim}__{cls}__{FILE_BUDGET[budget]}__pgd__seed{seed}.npz"
                for dataset, victims in VICTIMS.items() for seed in SEEDS
                for victim in victims for cls in CLASSES for budget in BUDGETS]
    missing = [str(p) for p in expected if not p.is_file()]
    require(not missing, f"missing {len(missing)}/144 required artifacts: " + "; ".join(missing))
    require(len(expected) == 144, "incorrect expected artifact grid")
    historical_path = ROOT / "outputs/expA_primattack_full_pool/row_resampling.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    selected: dict[tuple, dict] = {}
    vectors: dict[tuple, dict] = {}
    checkpoint_hashes: dict[str, str] = {}
    audit = {"artifact_count": 0, "audited_rows": 0, "validator_rechecked_rows": 0,
             "prediction_rechecked_rows": 0, "seed42_exact_final_selection": True,
             "cross_seed_overlap": [], "selection": {},
             "checkpoint_sha256": checkpoint_hashes, "historical_source": str(historical_path)}
    for dataset, victims in VICTIMS.items():
        adapter = get_adapter(dataset)
        processed = adapter._processed
        raw_all = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
        labels = np.load(processed / "y_test_cat.npy", mmap_mode="r")
        ids_all = pd.read_parquet(processed / "test.parquet", columns=["sample_id"])["sample_id"].astype(str).to_numpy()
        require(len(raw_all) == len(labels) == len(ids_all), f"{dataset}: test arrays not aligned")
        canonical_dir = ROOT / "FINAL_OUTPUTS/runs" / dataset / "baselines_untargeted"
        canonical = json.loads((canonical_dir / "selection.json").read_text(encoding="utf-8"))
        victim_info = json.loads((canonical_dir / "config.json").read_text(encoding="utf-8"))["victims"]
        mapping = adapter.class_mapping()
        transform = adapter.feature_transform()
        center = torch.as_tensor(transform.center, dtype=torch.float32, device=device)
        scale = torch.as_tensor(transform.scale, dtype=torch.float32, device=device)
        validator = get_validator(dataset)
        for victim_name in victims:
            info = victim_info[victim_name]
            ckpt = Path(info["checkpoint"])
            if not ckpt.is_absolute():
                ckpt = ROOT / ckpt
            ck_hash = file_sha256(ckpt)
            require(ck_hash == info["checkpoint_sha256"], f"{dataset}/{victim_name}: checkpoint hash changed")
            checkpoint_hashes[f"{dataset}/{victim_name}"] = ck_hash
            model = load_category_victim(ckpt, adapter=adapter,
                                         expected_model_type=info["arch"], device=device)
            for seed in SEEDS:
                selection_path = results_dir / dataset / f"seed{seed}" / "selection.json"
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                for cls in CLASSES:
                    where = f"{dataset}/{victim_name}/{cls}/seed{seed}"
                    entry = selection[victim_name][cls]
                    idx = np.asarray(entry["positional_idx"], dtype=np.int64)
                    ids = np.asarray(entry["sample_ids"], dtype=str)
                    cid = int(mapping.name_to_id[cls])
                    require(len(idx) == len(ids) == entry["n_used"] == N_PER_CLASS, f"{where}: denominator != 800")
                    require(np.all((idx >= 0) & (idx < len(raw_all))) and
                            np.all(idx[:-1] < idx[1:]), f"{where}: invalid/duplicate/unsorted indices")
                    require(len(np.unique(ids)) == len(ids), f"{where}: duplicate sample IDs")
                    same(ids_all[idx], ids, f"{where}: index-to-ID mapping differs")
                    require(id_hash(ids) == entry["sha256_sample_ids"], f"{where}: sample-ID hash differs")
                    require(entry["class_id"] == cid and np.all(labels[idx] == cid), f"{where}: source class differs")
                    require(N_PER_CLASS <= int(entry["n_eligible_total"]) <= int(np.count_nonzero(labels == cid)),
                            f"{where}: impossible eligible count")
                    raw = np.ascontiguousarray(raw_all[idx], dtype=np.float32)
                    require(raw_hash(raw) == entry["clean_raw_sha256"], f"{where}: clean-input hash differs")
                    clean_pred = predict(model, raw, center, scale, device)
                    require(np.all(clean_pred == cid), f"{where}: selected source not clean-correct")
                    if seed == 42:
                        # Exact canonical cohort and metadata, not just an overlapping random draw.
                        old = canonical[victim_name][cls]
                        for key in ("class_id", "n_used", "n_eligible_total", "sha256_sample_ids", "clean_raw_sha256"):
                            require(entry[key] == old[key], f"{where}: seed42 differs from FINAL {key}")
                        same(idx, np.asarray(old["positional_idx"]), f"{where}: seed42 indices differ from FINAL")
                        same(ids, np.asarray(old["sample_ids"]), f"{where}: seed42 IDs differ from FINAL")
                    selected[(dataset, victim_name, cls, seed)] = {"ids": ids, "idx": idx}
                    audit["selection"][where] = {"n": len(idx), "n_eligible_total": entry["n_eligible_total"],
                                                  "sha256_sample_ids": entry["sha256_sample_ids"],
                                                  "clean_raw_sha256": entry["clean_raw_sha256"]}
                    for budget in BUDGETS:
                        path = (results_dir / dataset / f"seed{seed}" / "primattack" / "artifacts" /
                                f"{victim_name}__{cls}__{FILE_BUDGET[budget]}__pgd__seed{seed}.npz")
                        with np.load(path, allow_pickle=False) as npz:
                            data = {key: npz[key] for key in npz.files}
                        context = str(path)
                        same(data["sample_id"].astype(str), ids, f"{context}: artifact IDs differ")
                        same(np.asarray(data["positional_idx"], dtype=np.int64), idx,
                             f"{context}: artifact indices differ")
                        for field, value in (("dataset", dataset), ("victim", victim_name),
                                             ("attack_class", cls), ("seed", seed),
                                             ("budget", "maximum-evaluated" if budget == "p75" else budget),
                                             ("objective", "untargeted"),
                                             ("method", "pgd"), ("primitive_mode", "joint"),
                                             ("clean_raw_sha256", entry["clean_raw_sha256"]),
                                             ("checkpoint_sha256", ck_hash)):
                            scalar(data, field, value, context)
                        same(data["true_class"], np.full(N_PER_CLASS, cid), f"{context}: source labels differ")
                        same(data["clean_pred"], clean_pred, f"{context}: clean predictions differ")
                        adv = np.asarray(data["adv_raw"], dtype=np.float32)
                        require(adv.shape == raw.shape and np.isfinite(adv).all(), f"{context}: malformed adv_raw")
                        adv_pred = predict(model, adv, center, scale, device)
                        same(data["adv_pred"], adv_pred, f"{context}: adversarial predictions differ")
                        hit = adv_pred != cid
                        valid = validator.validate_batch(adv, raw).hybrid_valid
                        mask(data, "raw_success", hit, context)
                        mask(data, "untargeted_success", hit, context)
                        mask(data, "validator_pass", valid, context)
                        mask(data, "valid_success", hit & valid, context)
                        mask(data, "final_success", hit & valid, context)
                        mask(data, "targeted_success", adv_pred == 0, context)
                        require(np.count_nonzero(data["n_modified_outside_primattack_mask"]) == 0,
                                f"{context}: modified outside PrimAttack support")
                        require(not np.any(data["empty_fwd_packet_filled"]),
                                f"{context}: empty forward packet filled")
                        require(not np.any(np.asarray(data["p"])[~np.asarray(data["pad_allowed"], bool)] > 0),
                                f"{context}: padding on forbidden rows")
                        vectors[(dataset, victim_name, cls, seed, budget)] = {
                            "raw": hit, "valid": hit & valid, "validator": valid,
                        }
                        audit["artifact_count"] += 1
                        audit["audited_rows"] += len(ids)
                        audit["prediction_rechecked_rows"] += len(ids)
                        audit["validator_rechecked_rows"] += len(ids)
            for cls in CLASSES:
                for left, right in combinations(SEEDS, 2):
                    a = selected[(dataset, victim_name, cls, left)]["ids"]
                    b = selected[(dataset, victim_name, cls, right)]["ids"]
                    overlap = len(set(a) & set(b))
                    audit["cross_seed_overlap"].append({"dataset": dataset, "victim": victim_name,
                                                          "source_class": cls, "seed_a": left,
                                                          "seed_b": right, "n_each": N_PER_CLASS,
                                                          "intersection": overlap,
                                                          "jaccard": overlap / (2 * N_PER_CLASS - overlap)})
    rng = np.random.default_rng(20261002)
    summaries: list[dict] = []
    tests: list[dict] = []
    for dataset, victims in VICTIMS.items():
        for victim_name in victims:
            for budget in BUDGETS:
                for cls in (*CLASSES, "ALL"):
                    strata = CLASSES if cls == "ALL" else (cls,)
                    for outcome in ("raw", "valid"):
                        per_seed = [np.concatenate([vectors[(dataset, victim_name, c, seed, budget)][outcome]
                                                    for c in strata]) for seed in SEEDS]
                        rates = np.asarray([v.mean() for v in per_seed])
                        pooled = np.concatenate(per_seed)
                        k, n = int(per_seed[0].sum()), len(per_seed[0])
                        cp = cp_interval(k, n)
                        wilson = wilson_score_interval(k, n)
                        bs_strata = [vectors[(dataset, victim_name, c, 42, budget)][outcome]
                                     for c in strata]
                        bs = bootstrap(bs_strata, rng)
                        summaries.append({"dataset": dataset, "victim": victim_name,
                                          "source_class": cls, "budget": budget, "outcome": outcome,
                                          "n_per_seed": n, "n_total": len(pooled),
                                          "successes_total": int(pooled.sum()),
                                          "ci_reference_seed": 42,
                                          "successes_reference_seed": k,
                                          **{f"seed{seed}_successes": int(v.sum())
                                             for seed, v in zip(SEEDS, per_seed)},
                                          **{f"seed{seed}_rate": float(v.mean())
                                             for seed, v in zip(SEEDS, per_seed)},
                                          "mean_seed_rate": float(rates.mean()),
                                          "sd_seed_rate_ddof1": float(rates.std(ddof=1)),
                                          "pooled_rate": float(pooled.mean()),
                                          "cp_low": cp[0], "cp_high": cp[1],
                                          "wilson_low": wilson[0], "wilson_high": wilson[1],
                                          "strat_boot_low": bs[0], "strat_boot_high": bs[1]})
            for budget in BUDGETS:
                raw = np.concatenate([vectors[(dataset, victim_name, cls, 42, budget)]["raw"]
                                      for cls in CLASSES])
                valid = np.concatenate([vectors[(dataset, victim_name, cls, 42, budget)]["valid"]
                                        for cls in CLASSES])
                require(np.all(~valid | raw), "raw/valid nesting violated")
                tests.append(paired(raw, valid, dataset=dataset, victim=victim_name,
                                    comparison="raw vs valid", budget=budget, outcome="untargeted",
                                    inferential_family="descriptive diagnostic; unadjusted"))
            for outcome in ("raw", "valid"):
                a = np.concatenate([vectors[(dataset, victim_name, cls, 42, "p75")][outcome]
                                    for cls in CLASSES])
                b = np.concatenate([vectors[(dataset, victim_name, cls, 42, "unbounded")][outcome]
                                    for cls in CLASSES])
                tests.append(paired(a, b, dataset=dataset, victim=victim_name,
                                    comparison="p75 vs unbounded", budget="both", outcome=outcome,
                                    inferential_family=("primary six, Holm" if outcome == "valid"
                                                       else "descriptive diagnostic; unadjusted")))
    primary = [row for row in tests if row["inferential_family"] == "primary six, Holm"]
    require(len(primary) == 6, "primary Holm family must include exactly six victims")
    for row, adjusted in zip(primary, holm_adjust([r["p_value"] for r in primary])):
        row["p_holm"] = adjusted
        row["decision"] = "reject H0" if adjusted < .05 else "do not reject H0"
    # Complete audit before writing anything; failure never leaves a partial summary/report.
    require(audit["artifact_count"] == 144 and audit["audited_rows"] == 144 * N_PER_CLASS,
            "incomplete audit")
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "summary.csv", summaries)
    write_csv(results_dir / "tests.csv", tests)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = ["# Fresh-row Prim-PGD untargeted joint comparison", "",
              "Each attack seed (42, 2024, 2026) selects a fresh, random, clean-correct "
              "800 flows per attack class per victim. Each budget uses identical rows within "
              "a seed; seed 42 exactly matches FINAL selection. Other seeds have partly "
              "overlapping, **not paired**, cohorts. Victims are reported separately.", "",
              "## Three-seed rates", "",
              "Rates are class-balanced within each victim (four classes × 800). SD is the "
              "sample SD across three seed rates (ddof=1), not an uncertainty interval. "
              "Clopper–Pearson (CP), Wilson and class-stratified bootstrap 95% "
              "intervals refer to the reference seed 42 only (n=3,200 per victim). "
              "The 10,000-replicate bootstrap resamples flows within each class "
              "at that seed, conditional on this test split and frozen victim. "
              "These intervals do not measure variation across attack seeds or "
              "new campaigns; the binomial intervals assume independent flows. "
              "Cohorts overlap, so three-seed pooled-flow intervals would "
              "overstate precision.", "",
              "|Dataset|Victim|Budget|Outcome|Seed 42 / 2024 / 2026|Mean ± SD|CP 95%|Wilson 95%|Stratified bootstrap 95%|",
              "|---|---|---|---|---|---|---|---|---|"]
    pct = lambda x: f"{100*x:.2f}%"
    def fmt_p(value: float) -> str:
        return "<1e-300 (underflow)" if value == 0 else f"{value:.4g}"
    for row in summaries:
        if row["source_class"] != "ALL":
            continue
        vals = " / ".join(pct(row[f"seed{s}_rate"]) for s in SEEDS)
        report.append(f"|{row['dataset']}|{row['victim']}|{row['budget']}|{row['outcome']}|{vals}|"
                      f"{pct(row['mean_seed_rate'])} ± {pct(row['sd_seed_rate_ddof1'])}|"
                      f"{pct(row['cp_low'])}–{pct(row['cp_high'])}|"
                      f"{pct(row['wilson_low'])}–{pct(row['wilson_high'])}|"
                      f"{pct(row['strat_boot_low'])}–{pct(row['strat_boot_high'])}|")
    report += ["", "Per-class three-seed rates and intervals are in `summary.csv`.", "",
               "## Reference seed 42: paired tests", "",
               "Only valid-ASR p75 vs unbounded has a prespecified six-victim "
               "Holm family (α=0.05). McNemar is exact binomial below 25 discordants, "
               "otherwise continuity-corrected chi-square; differences have 95% "
               "Newcombe paired intervals. Raw-vs-valid within each budget and the "
               "raw p75-vs-unbounded comparisons are **descriptive, unadjusted diagnostics**; "
               "their p-values are not multiplicity-controlled findings. No Cochran Q "
               "or cross-seed paired test is performed.", "",
               "|Dataset|Victim|Comparison|Budget|Outcome|Difference pp [95% CI]|A-only / B-only|p|Holm p|Decision|",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for row in tests:
        adjusted = fmt_p(row["p_holm"]) if row["p_holm"] is not None else "not adjusted"
        report.append(f"|{row['dataset']}|{row['victim']}|{row['comparison']}|{row['budget']}|"
                      f"{row['outcome']}|{row['difference_pp']:.2f} "
                      f"[{row['difference_ci_low_pp']:.2f}, {row['difference_ci_high_pp']:.2f}]|"
                      f"{row['a_only']} / {row['b_only']}|{fmt_p(row['p_value'])}|"
                      f"{adjusted}|{row['decision']}|")
    significant = sum(r["decision"] == "reject H0" for r in primary)
    varying = sum(r["sd_seed_rate_ddof1"] > 0 for r in summaries
                  if r["source_class"] == "ALL" and r["outcome"] == "valid")
    report += ["", f"Observed seed-cohort rates differ in {varying}/12 victim–budget "
               "conditions; the three rates supply descriptive variance, not a valid "
               "cross-seed paired hypothesis test. Seed-42 p75 versus unbounded Valid "
               f"ASR differs after Holm in {significant}/6 victims. These tests compare "
               "budgets, not attack methods or independent row-selection effects. "
               "The existing FINAL Experiment A baseline tests remain applicable to "
               "the seed-42 cohort only.", ""]
    report += ["", "## Cohort overlap and historical context", "",
               "`audit.json` records exact pairwise sample-ID intersections and Jaccard "
               "overlap per dataset × victim × class. Overlap alone does not turn distinct "
               "seed cohorts into a fully paired repeated-measures panel.", "",
               "|Dataset|Victim|Seed pair|Shared IDs (out of 3200)|Jaccard|",
               "|---|---|---|---:|---:|"]
    for dataset, victims in VICTIMS.items():
        for victim_name in victims:
            for left, right in combinations(SEEDS, 2):
                cells = [r for r in audit["cross_seed_overlap"] if r["dataset"] == dataset
                         and r["victim"] == victim_name and r["seed_a"] == left
                         and r["seed_b"] == right]
                require(len(cells) == len(CLASSES), "incomplete overlap audit")
                shared = sum(r["intersection"] for r in cells)
                report.append(f"|{dataset}|{victim_name}|{left}/{right}|{shared}|"
                              f"{shared / (6400 - shared):.4f}|")
    report += ["", "The earlier `outputs/expA_primattack_full_pool/row_resampling.json` "
               "resamples rows from an already-attacked full pool at p75; it measures "
               "row-selection variation **conditional on fixed attack outcomes**, not "
               "independent attack-seed variability, and cannot replace these three "
               "fresh attack runs. Its historical valid p75 figures (canonical / "
               "resample mean / resample SD / 2.5–97.5%):", "",
               "|Dataset|Victim|Canonical|Resample mean ± SD|Resample 95% range|",
               "|---|---|---|---|---|"]
    for dataset, victims in VICTIMS.items():
        for victim_name in victims:
            old = historical[dataset][victim_name]["valid"]
            report.append(f"|{dataset}|{victim_name}|{pct(old['canonical_sample'])}|"
                          f"{pct(old['resample_mean'])} ± {pct(old['resample_sd'])}|"
                          f"{pct(old['resample_p2_5'])}–{pct(old['resample_p97_5'])}|")
    report += ["", "All successes are offline feature-space proxies on held-out "
               "chronological-within-label test splits. No PCAP replay, packet-level "
               "realizability, new-campaign generalization, or unconditional seed robustness "
               "is established. Non-rejection by McNemar does not prove equivalence.", ""]
    (results_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Audited {audit['artifact_count']} artifacts, {audit['audited_rows']} rows; "
          f"wrote {results_dir / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "ablations/seeded_rows/results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    analyze(args.results_dir.resolve(), args.device)


if __name__ == "__main__":
    main()
