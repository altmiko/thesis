"""Baseline (unconstrained input-space) PGD and C&W attacks on the CICIDS2017-DistriNet
NIDS victims (MLP, CNN, FT-Transformer), with domain-validity gating and
multi-seed variance.

These are the *classical* adversarial baselines -- they perturb all 79 RobustScaler-space
features freely with NO realizability / domain model. Success is untargeted misclassification
(pred != true label) on clean-correct malicious test flows.

Validity is scored with the repository's canonical gate (``evaluate_cell`` from
``attack.run_cicids2017_primitive_attack``), so no second validity convention is invented:

* ``ASR``              -- untargeted evasion rate (misclassified).
* ``domain_valid``     -- validator_v2 ``hybrid_valid`` (PAVE feature-domain + mined density).
* ``realizable``       -- internal realizability all-pass (dependency / packet-summary /
                          timing / rate / discreteness / frozen categories).
* ``valid_evasion``    -- evasion AND domain_valid.

All rates use the clean-correct denominator (repo convention: (mask & clean_correct)/clean_correct).
Runs every seed and reports mean +/- sample-std across seeds.

Run (thesis mamba env, CUDA):
    python scripts/run_baseline_attacks_nids.py --device cuda --seeds 42,123,2024,7,99
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from attack.run_cicids2017_primitive_attack import (  # noqa: E402
    evaluate_cell, _LENGTH_COLS, _TIMING_COLS, _RATE_COLS,
)
from datasets.cicids2017 import CICIDS2017Adapter  # noqa: E402
from experiments.provenance import deterministic_runtime  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from vae.cicids2017_stage_a import ATTACK_CLASSES  # noqa: E402

# victim tag -> (checkpoint path relative to repo root, expected model_type)
VICTIMS = {
    "mlp": ("outputs/cicids2017distrinet/models/mlp_category.pt", "mlp"),
    "cnn": ("outputs/cicids2017distrinet/models/cnn_category.pt", "cnn"),
    "ft_transformer": (
        "outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt",
        "ft_transformer",
    ),
}

# metric key -> mask expression over the evaluate_cell masks (all gated by clean_correct later)
METRICS = ("asr", "domain_valid", "realizable", "valid_evasion")


def _select_clean_correct(victim, x_scaled, y, class_id, per_class, device, bs=8192):
    """First ``per_class`` rows of ``class_id`` the victim classifies correctly."""
    keep: list[int] = []
    rows = np.flatnonzero(y == class_id)
    for start in range(0, len(rows), bs):
        idx = rows[start : start + bs]
        chunk = torch.tensor(
            np.ascontiguousarray(x_scaled[idx]), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred = chunk if False else victim(chunk).argmax(1).cpu().numpy()
        keep.extend(int(j) for j in idx[pred == class_id])
        if len(keep) >= per_class:
            break
    return np.asarray(sorted(keep[:per_class]), dtype=np.int64)


def _counts(masks, extra_masks, cc):
    """Numerator counts (mask & clean_correct) for each reported metric."""
    ev = masks["evasion"]
    dv = masks["domain_valid"]
    rz = masks["primitive_transform_consistent"]
    m = {
        "asr": ev,
        "domain_valid": dv,
        "realizable": rz,
        "valid_evasion": ev & dv,
    }
    return {k: int((v & cc).sum().item()) for k, v in m.items()}


def _mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="42,123,2024,7,99")
    ap.add_argument("--samples-per-class", type=int, default=200)
    ap.add_argument("--victims", default="mlp,cnn,ft_transformer")
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--pgd-epsilon", type=float, default=0.5)
    ap.add_argument("--pgd-alpha", type=float, default=0.05)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--cw-lambda", type=float, default=1.0)
    ap.add_argument("--cw-kappa", type=float, default=0.0)
    ap.add_argument("--cw-iters", type=int, default=200)
    ap.add_argument("--cw-lr", type=float, default=0.01)
    ap.add_argument("--cw-conv", type=float, default=1e-5)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cicids2017_baseline_pgd_cw")
    args = ap.parse_args()

    device = args.device
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    victim_tags = [v.strip() for v in args.victims.split(",") if v.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    adapter = CICIDS2017Adapter()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)

    prim = CICIDS2017PrimitiveModel(manifest)
    validator = RealizabilityValidator(prim)
    groups_idx = {
        "padding": torch.tensor([prim.i[n] for n in _LENGTH_COLS], device=device),
        "timing": torch.tensor([prim.i[n] for n in _TIMING_COLS], device=device),
        "rate": torch.tensor([prim.i[n] for n in _RATE_COLS], device=device),
    }

    x_scaled = np.asarray(np.load(adapter._processed / "X_test.npy", mmap_mode="r"))
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    y = np.load(adapter._processed / "y_test_cat.npy").astype(np.int64)

    # records[(victim, cls, seed, attack)] = {"cc": int, "n": int, counts..., "l2_sum", "linf_sum"}
    records: dict = {}
    clean_acc = {}

    for vname in victim_tags:
        ckpt_rel, model_type = VICTIMS[vname]
        victim = load_category_victim(
            REPO_ROOT / ckpt_rel, adapter=adapter,
            expected_model_type=model_type, device=device,
        )
        victim.eval()

        # full-test clean accuracy for context
        with torch.no_grad():
            preds = []
            for start in range(0, len(y), 16384):
                xb = torch.tensor(
                    np.ascontiguousarray(x_scaled[start : start + 16384]),
                    dtype=torch.float32, device=device,
                )
                preds.append(victim(xb).argmax(1).cpu().numpy())
            clean_acc[vname] = float((np.concatenate(preds) == y).mean())

        for cname in classes:
            cid = int(mapping.name_to_id[cname])
            idx = _select_clean_correct(victim, x_scaled, y, cid, args.samples_per_class, device)
            raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
            raw = torch.tensor(raw_np, device=device)
            x0 = (raw - center) / scale

            for seed in seeds:
                for attack in ("input-pgd", "input-cw"):
                    deterministic_runtime(seed)
                    if attack == "input-pgd":
                        x_adv, _ = input_pgd_attack(
                            classifier=victim, x_original=x0,
                            y_true=torch.full((len(idx),), cid, dtype=torch.long, device=device),
                            epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                            num_steps=args.pgd_steps, random_start=True, device=device,
                        )
                    else:
                        x_adv, _ = input_cw_attack(
                            classifier=victim, x_original=x0,
                            y_true=torch.full((len(idx),), cid, dtype=torch.long, device=device),
                            lambda_conf=args.cw_lambda, kappa=args.cw_kappa,
                            num_iterations=args.cw_iters, learning_rate=args.cw_lr,
                            convergence_threshold=args.cw_conv, device=device,
                        )
                    adv_raw = (x_adv * scale + center).detach()
                    masks, _cost, _cp, _ap = evaluate_cell(
                        prim, validator, victim, raw, adv_raw,
                        center, scale, cid, groups_idx,
                    )
                    cc = masks["clean_correct"]
                    delta = (x_adv - x0).reshape(len(idx), -1)
                    l2 = torch.linalg.norm(delta, dim=1)
                    linf = delta.abs().amax(dim=1)
                    rec = {
                        "cc": int(cc.sum().item()),
                        "n": int(len(idx)),
                        "l2_sum": float(l2.sum().item()),
                        "linf_sum": float(linf.sum().item()),
                    }
                    rec.update(_counts(masks, None, cc))
                    records[(vname, cname, seed, attack)] = rec
            print(f"[baseline] {vname}/{cname}: {len(idx)} clean-correct rows x "
                  f"{len(seeds)} seeds done", flush=True)

    # ---- aggregation ----
    def seed_rate(vname, attack, metric, seed, cls_list):
        num = sum(records[(vname, c, seed, attack)][metric] for c in cls_list)
        den = sum(records[(vname, c, seed, attack)]["cc"] for c in cls_list)
        return (num / den) if den else float("nan")

    def seed_pert(vname, attack, seed, cls_list, key):
        s = sum(records[(vname, c, seed, attack)][f"{key}_sum"] for c in cls_list)
        n = sum(records[(vname, c, seed, attack)]["n"] for c in cls_list)
        return (s / n) if n else float("nan")

    summary = []
    for vname in victim_tags:
        for attack in ("input-pgd", "input-cw"):
            row = {"victim": vname, "attack": attack, "clean_test_accuracy": clean_acc[vname],
                   "n_per_seed": int(sum(records[(vname, c, seeds[0], attack)]["cc"] for c in classes))}
            for metric in METRICS:
                mean, std = _mean_std([seed_rate(vname, attack, metric, s, classes) for s in seeds])
                row[metric] = {"mean": mean, "std": std}
            for key in ("l2", "linf"):
                mean, std = _mean_std([seed_pert(vname, attack, s, classes, key) for s in seeds])
                row[f"mean_{key}"] = {"mean": mean, "std": std}
            row["per_class"] = {}
            for cname in classes:
                pc = {}
                for metric in ("asr", "valid_evasion"):
                    mean, std = _mean_std([seed_rate(vname, attack, metric, s, [cname]) for s in seeds])
                    pc[metric] = {"mean": mean, "std": std}
                row["per_class"][cname] = pc
            summary.append(row)

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": device,
            "seeds": seeds,
            "dataset": "cicids2017_distrinet",
            "n_features": int(manifest.n_features),
            "n_test": int(len(y)),
            "class_names": list(mapping.names),
            "attacked_classes": classes,
            "samples_per_class": args.samples_per_class,
            "pgd_params": {"epsilon": args.pgd_epsilon, "alpha": args.pgd_alpha,
                           "num_steps": args.pgd_steps, "random_start": True},
            "cw_params": {"lambda_conf": args.cw_lambda, "kappa": args.cw_kappa,
                          "num_iterations": args.cw_iters, "learning_rate": args.cw_lr,
                          "convergence_threshold": args.cw_conv},
            "validity_source": "attack.run_cicids2017_primitive_attack.evaluate_cell "
                               "(validator_v2 hybrid_valid domain gate + internal realizability)",
        },
        "summary": summary,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_pgd_cw_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out / "baseline_pgd_cw_results.md", payload)
    print(f"[baseline] wrote {out/'baseline_pgd_cw_results.md'}", flush=True)


def _pm(d, pct=True):
    if pct:
        return f"{d['mean']*100:.2f}\u00b1{d['std']*100:.2f}%"
    return f"{d['mean']:.3f}\u00b1{d['std']:.3f}"


def _write_md(path: Path, payload: dict) -> None:
    m = payload["meta"]
    L = ["# Baseline PGD & C&W Attacks on CICIDS2017-DistriNet NIDS Victims (validity-gated, seeded)\n"]
    L.append(f"- Generated: {m['generated_utc']}")
    L.append(f"- Device: `{m['device']}`  |  Seeds: {m['seeds']} (mean +/- sample std across seeds)")
    L.append(f"- Dataset: CICIDS2017-DistriNet, {m['n_features']} RobustScaler-space features, "
             f"test split ({m['n_test']} flows)")
    L.append(f"- Classes: {', '.join(m['class_names'])}; attacked (malicious): {', '.join(m['attacked_classes'])}")
    L.append(f"- Selection: {m['samples_per_class']} clean-correct flows per malicious class "
             "(fixed across seeds); untargeted evasion (pred != true).")
    L.append("- **Unconstrained input-space baselines**: all 79 features editable, NO realizability/domain model.")
    L.append(f"- Validity gate: {m['validity_source']}\n")

    pg, cw = m["pgd_params"], m["cw_params"]
    L.append("## Attack hyperparameters\n")
    L.append(f"- **Input PGD** (L-inf): epsilon={pg['epsilon']}, alpha={pg['alpha']}, "
             f"steps={pg['num_steps']}, random_start={pg['random_start']}")
    L.append(f"- **Input C&W** (L2): lambda_conf={cw['lambda_conf']}, kappa={cw['kappa']}, "
             f"iters={cw['num_iterations']}, lr={cw['learning_rate']}, conv_thresh={cw['convergence_threshold']}")
    L.append("- C&W uses a deterministic Adam init (delta=0), so its across-seed std is ~0.\n")

    L.append("## Metric definitions\n")
    L.append("- **ASR**: untargeted evasion rate (misclassified).")
    L.append("- **Domain-valid**: validator_v2 `hybrid_valid` (PAVE feature-domain + mined density).")
    L.append("- **Realizable**: internal realizability all-pass (dependency/packet/timing/rate/discreteness/frozen).")
    L.append("- **Valid-evasion**: evasion AND domain-valid.")
    L.append("- All rates over the clean-correct denominator.\n")

    L.append("## Summary (mean +/- std across seeds)\n")
    L.append("| Victim | Clean acc | n | Attack | ASR | Valid-evasion | "
             "Domain-valid | Realizable | Mean L2 | Mean Linf |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["summary"]:
        L.append(
            f"| {r['victim']} | {r['clean_test_accuracy']*100:.2f}% | {r['n_per_seed']} | {r['attack']} | "
            f"{_pm(r['asr'])} | {_pm(r['valid_evasion'])} | "
            f"{_pm(r['domain_valid'])} | {_pm(r['realizable'])} | "
            f"{_pm(r['mean_l2'], pct=False)} | {_pm(r['mean_linf'], pct=False)} |"
        )
    L.append("")

    L.append("## Per-class (mean +/- std across seeds)\n")
    for r in payload["summary"]:
        L.append(f"### {r['victim']} / {r['attack']}\n")
        L.append("| Class | ASR | Valid-evasion |")
        L.append("|---|---|---|")
        for cname, pc in r["per_class"].items():
            L.append(f"| {cname} | {_pm(pc['asr'])} | {_pm(pc['valid_evasion'])} |")
        L.append("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
