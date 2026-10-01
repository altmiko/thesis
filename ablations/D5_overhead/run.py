"""D5 - Attacker overhead of PrimAttack's valid successes (analysis only, no new attack run).

Design sources:
* Amoeba (CoNEXT 2023, Table 1): data overhead DO = padding / (payload + padding) and time
  overhead TO = added time / (added time + duration) of every success;
* Tamaraw (CCS 2014, Fig. 4) and FRONT (USENIX Security 2020, Fig. 7): success as a function of
  the bandwidth and time the attacker is willing to spend (cost curves / 2-D frontier);
* DeTorrent (PETS 2024, Fig. 6): where a larger budget stops paying off (diminishing returns);
* PLAA (2026, Table VI): per-success relative change of packet length, inter-arrival time and
  rate between the adversarial and the original flow.

Inputs are the canonical FINAL-suite per-row artifacts (never modified):
* ``untargeted_pgd`` - Exp A PrimAttack (Prim-PGD, untargeted, joint, capability-aware), p75 and
  unbounded (``primattack_untargeted``);
* ``targeted_pgd`` / ``targeted_hybrid`` - Exp B/C targeted->Benign at p50, p75 and unbounded
  (``primattack_targeted_budgets`` + ``primattack_targeted_optimizers``).

Overheads are computed from the realized adversarial flow and its source flow (test split,
``X_test_pristine``): added bytes = Δ(Total Length of Fwd Packet); total bytes = forward +
backward bytes; added time = Δ(Flow Duration). All rates use the attempted denominator
(800 flows per class, 3,200 per victim and seed); overhead statistics are over valid successes,
all attack seeds pooled per (dataset, victim, arm, budget).

    python ablations/D5_overhead/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

from ablations.common.runner import CLASSES, DATASETS, FINAL_RUNS, REPO_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from datasets import get_adapter  # noqa: E402

ARMS = {
    "untargeted_pgd": [("primattack_untargeted", "pgd", "p75"),
                       ("primattack_untargeted", "pgd", "unb")],
    "targeted_pgd": [("primattack_targeted_budgets", "pgd", "p50"),
                     ("primattack_targeted_optimizers", "pgd", "p75"),
                     ("primattack_targeted_budgets", "pgd", "unb")],
    "targeted_hybrid": [("primattack_targeted_budgets", "hybrid", "p50"),
                        ("primattack_targeted_optimizers", "hybrid", "p75"),
                        ("primattack_targeted_budgets", "hybrid", "unb")],
}
SEEDS = (42, 2024, 2026)
TO_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
BUDGET_ORDER = {"p50": 0, "p75": 1, "unb": 2}


def _overheads(raw: np.ndarray, adv: np.ndarray, i: dict[str, int]) -> dict[str, np.ndarray]:
    def rel(name: str) -> np.ndarray:
        a, b = adv[:, i[name]].astype(np.float64), raw[:, i[name]].astype(np.float64)
        return np.divide(a - b, np.abs(b), out=np.full(len(a), np.nan), where=b != 0)

    fwd = raw[:, i["Total Length of Fwd Packet"]].astype(np.float64)
    bwd = raw[:, i["Total Length of Bwd Packet"]].astype(np.float64)
    added_bytes = adv[:, i["Total Length of Fwd Packet"]].astype(np.float64) - fwd
    dur0 = raw[:, i["Flow Duration"]].astype(np.float64)
    added_time = adv[:, i["Flow Duration"]].astype(np.float64) - dur0
    total = fwd + bwd + added_bytes
    return {
        "added_bytes": added_bytes, "added_time_us": added_time,
        "data_overhead": np.divide(added_bytes, total, out=np.zeros_like(total), where=total > 0),
        "time_overhead": np.divide(added_time, added_time + dur0,
                                   out=np.zeros_like(dur0), where=(added_time + dur0) > 0),
        "rel_packet_length": rel("Average Packet Size"),
        "rel_flow_iat_mean": rel("Flow IAT Mean"),
        "rel_flow_bytes_rate": rel("Flow Bytes/s"),
        "rel_flow_packets_rate": rel("Flow Packets/s"),
    }


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, curves = [], []
    for dataset, spec in DATASETS.items():
        adapter = get_adapter(spec["cli"])
        names = list(adapter.feature_manifest().names)
        i = {n: k for k, n in enumerate(names)}
        raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
        for victim in spec["victims"]:
            for arm, stages in ARMS.items():
                for stage, method, budget in stages:
                    art = FINAL_RUNS / dataset / stage / "artifacts"
                    per_seed_asr, pooled = [], []
                    for seed in SEEDS:
                        parts = []
                        for cname in CLASSES:
                            path = art / f"{victim}__{cname}__{budget}__{method}__seed{seed}.npz"
                            if not path.exists():
                                raise FileNotFoundError(path)
                            with np.load(path, allow_pickle=True) as z:
                                raw = np.asarray(raw_all[z["positional_idx"]])
                                ov = _overheads(raw, z["adv_raw"], i)
                                ov["valid_success"] = z["valid_success"].astype(bool)
                                ov["p"] = z["p"]
                                parts.append(ov)
                        r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                        per_seed_asr.append(r["valid_success"].mean())
                        pooled.append(r)
                    r = {k: np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
                    s = r["valid_success"]
                    attempts = len(s)
                    med = lambda k: float(np.nanmedian(r[k][s])) if s.any() else float("nan")  # noqa: E731
                    q90 = lambda k: float(np.nanquantile(r[k][s], 0.9)) if s.any() else float("nan")  # noqa: E731
                    summary.append({
                        "dataset": dataset, "victim": victim, "arm": arm, "budget": budget,
                        "attempts_per_seed": attempts // len(SEEDS),
                        "valid_asr_mean": float(np.mean(per_seed_asr)),
                        "valid_asr_min": float(np.min(per_seed_asr)),
                        "valid_asr_max": float(np.max(per_seed_asr)),
                        "successes_pooled": int(s.sum()),
                        "frac_success_with_padding": float((r["p"][s] > 0).mean()) if s.any() else float("nan"),
                        "median_data_overhead": med("data_overhead"),
                        "median_time_overhead": med("time_overhead"),
                        "p90_time_overhead": q90("time_overhead"),
                        "median_added_bytes": med("added_bytes"),
                        "median_added_time_us": med("added_time_us"),
                        "median_rel_packet_length": med("rel_packet_length"),
                        "median_rel_flow_iat_mean": med("rel_flow_iat_mean"),
                        "median_rel_flow_bytes_rate": med("rel_flow_bytes_rate"),
                        "median_rel_flow_packets_rate": med("rel_flow_packets_rate"),
                    })
                    for t in TO_THRESHOLDS:
                        ok = s & (r["time_overhead"] <= t)
                        curves.append({"dataset": dataset, "victim": victim, "arm": arm,
                                       "budget": budget, "max_time_overhead": t,
                                       "valid_asr": float(ok.sum() / attempts)})
    return pd.DataFrame(summary), pd.DataFrame(curves)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def _num(x: float) -> str:
    return "n/a" if x != x else f"{x:.3g}"


def report(summary: pd.DataFrame, curves: pd.DataFrame, out: Path) -> None:
    summary = summary.assign(_b=summary["budget"].map(BUDGET_ORDER)).sort_values(
        ["dataset", "victim", "arm", "_b"]).drop(columns="_b")
    summary.to_csv(out / "overhead_summary.csv", index=False)
    curves.to_csv(out / "cost_curves.csv", index=False)
    L = ["# D5 - Attacker overhead of PrimAttack's valid successes", "",
         "FINAL-suite artifacts; overheads over valid successes (all attack seeds pooled), "
         "Valid ASR = mean over seeds 42/2024/2026 of successes / 3,200 attempted flows.", "",
         "## Overhead per success (Amoeba DO/TO, PLAA relative changes)", "",
         "| dataset | victim | arm | budget | Valid ASR (min-max) | successes | with padding | "
         "median DO | median TO | p90 TO | median added µs | Δ pkt length | Δ Flow IAT Mean | "
         "Δ Flow Bytes/s | Δ Flow Pkts/s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples(index=False):
        L.append(f"| {r.dataset} | {r.victim} | {r.arm} | {r.budget} | "
                 f"{_pct(r.valid_asr_mean)} ({_pct(r.valid_asr_min)}-{_pct(r.valid_asr_max)}) | "
                 f"{r.successes_pooled} | {_num(r.frac_success_with_padding)} | "
                 f"{_num(r.median_data_overhead)} | {_num(r.median_time_overhead)} | "
                 f"{_num(r.p90_time_overhead)} | {_num(r.median_added_time_us)} | "
                 f"{_num(r.median_rel_packet_length)} | {_num(r.median_rel_flow_iat_mean)} | "
                 f"{_num(r.median_rel_flow_bytes_rate)} | {_num(r.median_rel_flow_packets_rate)} |")
    L += ["", "DO = added bytes / (forward + backward bytes + added bytes); TO = added flow time / "
          "(added time + original duration); Δ = (adversarial - original) / original.", "",
          "## Diminishing returns over the budget ladder (targeted arms: p50 -> p75 -> unbounded)",
          "", "| dataset | victim | arm | p50 | p75 | unbounded | gain p50->p75 (pp) | "
          "gain p75->unb (pp) | median TO p50 / p75 / unb |", "|---|---|---|---|---|---|---|---|---|"]
    for (dataset, victim, arm), df in summary.groupby(["dataset", "victim", "arm"], sort=False):
        b = {r.budget: r for r in df.itertuples(index=False)}
        if not {"p50", "p75", "unb"} <= set(b):
            continue
        g1 = 100 * (b["p75"].valid_asr_mean - b["p50"].valid_asr_mean)
        g2 = 100 * (b["unb"].valid_asr_mean - b["p75"].valid_asr_mean)
        L.append(f"| {dataset} | {victim} | {arm} | {_pct(b['p50'].valid_asr_mean)} | "
                 f"{_pct(b['p75'].valid_asr_mean)} | {_pct(b['unb'].valid_asr_mean)} | {g1:+.2f} | "
                 f"{g2:+.2f} | {_num(b['p50'].median_time_overhead)} / "
                 f"{_num(b['p75'].median_time_overhead)} / {_num(b['unb'].median_time_overhead)} |")
    L += ["", "## Cost curve: Valid ASR when the attacker caps the time overhead TO", "",
          "| dataset | victim | arm | budget | " + " | ".join(
              f"TO≤{t:g}" for t in TO_THRESHOLDS) + " |", "|---|---|---|---|" + "---|" * len(TO_THRESHOLDS)]
    for (dataset, victim, arm, budget), d in curves.groupby(
            ["dataset", "victim", "arm", "budget"], sort=False):
        vals = d.sort_values("max_time_overhead")["valid_asr"].tolist()
        L.append(f"| {dataset} | {victim} | {arm} | {budget} | "
                 + " | ".join(f"{100 * v:.2f}" for v in vals) + " |")
    L += ["", "Cost-curve values are Valid ASR in % of attempted flows, pooled over seeds. No "
          "valid success adds bytes (column `with padding` above), so no data-overhead cap is "
          "needed."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "inputs.json").write_text(json.dumps({
        "final_runs": FINAL_RUNS.relative_to(REPO_ROOT).as_posix(),
        "arms": {k: [list(v) for v in vs] for k, vs in ARMS.items()},
        "seeds": list(SEEDS), "classes": list(CLASSES)}, indent=2), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    out = EXP_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary, curves = collect()
    report(summary, curves, out)


if __name__ == "__main__":
    main()
