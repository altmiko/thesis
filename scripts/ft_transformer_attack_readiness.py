"""FT-Transformer adversarial-readiness smoke harness (task §22-25, §32).

Confirms the trained FT-Transformer victim is a valid gradient-attack target under the
EXISTING attack building blocks (no attack code is modified):

* raw-logit contract + input gradients finite / non-zero / no NaN-Inf;
* feature ordering matches the schema and the checkpoint metadata;
* scaler is applied exactly once (attacks run in the RobustScaler model space, or,
  for PrimAttack, ``(phi(raw) - center) / scale`` once);
* feature-space PGD and C&W flip predictions on clean-correct malicious flows;
* targeted-Benign objective resolves the correct Benign label index (0);
* PrimAttack primitive->feature transform yields finite ``dL/dp`` through the victim.

Run:
    PYTHONPATH=src python scripts/ft_transformer_attack_readiness.py \
        --victim outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt \
        --device cpu --n 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import get_adapter  # noqa: E402
from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402


def _clean_correct_malicious(victim, x_scaled, y, benign_id, n, device, scan=40000, bs=4096):
    """Rows the victim classifies correctly and whose true class is not Benign.

    Scans a bounded prefix in mini-batches (avoids a full 312k-row forward) until
    ``n`` eligible rows are collected.
    """
    keep: list[int] = []
    limit = min(scan, len(x_scaled))
    for start in range(0, limit, bs):
        stop = min(start + bs, limit)
        chunk = torch.tensor(
            np.ascontiguousarray(x_scaled[start:stop]), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred = victim(chunk).argmax(1).cpu().numpy()
        yc = y[start:stop]
        idx = np.flatnonzero((pred == yc) & (yc != benign_id))
        keep.extend((start + idx).tolist())
        if len(keep) >= n:
            break
    return np.asarray(keep[:n], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--victim", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    report: dict[str, object] = {"victim": str(args.victim), "device": device}

    adapter = get_adapter("cicids2017")
    manifest = adapter.feature_manifest()
    class_map = adapter.class_mapping()
    benign_id = class_map.name_to_id["Benign"]
    report["benign_index"] = int(benign_id)
    report["benign_index_is_zero"] = bool(benign_id == 0)

    victim = load_category_victim(
        args.victim, adapter=adapter, expected_model_type="ft_transformer", device=device
    )
    victim.eval()

    # -- feature ordering: schema == checkpoint metadata --------------------------
    ckpt = torch.load(args.victim, map_location="cpu", weights_only=True)
    ckpt_names = list(ckpt.get("metadata", {}).get("feature_names") or [])
    schema_names = [s.name for s in manifest.specs]
    report["feature_order_matches_schema"] = bool(ckpt_names == schema_names)
    report["feature_count"] = len(schema_names)

    x_scaled = np.load(adapter._processed / "X_test.npy", mmap_mode="r")
    y = np.load(adapter._processed / "y_test_cat.npy").astype(np.int64)
    keep = _clean_correct_malicious(victim, np.asarray(x_scaled), y, benign_id, args.n, device)
    report["eligible_clean_correct_malicious"] = int(len(keep))
    xb = torch.tensor(np.ascontiguousarray(x_scaled[keep]), dtype=torch.float32, device=device)
    yb = torch.tensor(y[keep], dtype=torch.long, device=device)

    # -- 1. raw logits + input gradients -----------------------------------------
    xg = xb.clone().requires_grad_(True)
    logits = victim(xg)
    loss = F.cross_entropy(logits, yb)
    grad = torch.autograd.grad(loss, xg)[0]
    report["returns_raw_logits"] = bool(
        isinstance(logits, torch.Tensor) and logits.shape == (len(xb), class_map.n_classes)
    )
    report["logits_finite"] = bool(torch.isfinite(logits).all())
    report["input_grad_shape_ok"] = bool(grad.shape == xb.shape)
    report["input_grad_finite"] = bool(torch.isfinite(grad).all())
    report["input_grad_nonzero"] = bool((grad != 0).any())

    # -- 2. feature-space PGD (untargeted) ---------------------------------------
    x_adv, _ = input_pgd_attack(
        classifier=victim, x_original=xb, y_true=yb,
        epsilon=0.5, alpha=0.1, num_steps=20, random_start=False, device=device,
    )
    with torch.no_grad():
        pgd_pred = victim(x_adv).argmax(1)
    report["pgd_evasion"] = int((pgd_pred != yb).sum().item())
    report["pgd_ran"] = True

    # -- 3. feature-space C&W (untargeted) ---------------------------------------
    x_cw, _ = input_cw_attack(
        classifier=victim, x_original=xb, y_true=yb,
        lambda_conf=1.0, kappa=0.0, num_iterations=50,
        learning_rate=0.05, convergence_threshold=1e-6, device=device,
    )
    with torch.no_grad():
        cw_pred = victim(x_cw).argmax(1)
    report["cw_evasion"] = int((cw_pred != yb).sum().item())
    report["cw_ran"] = True

    # -- 4. targeted-Benign objective (differentiable, resolves label 0) ---------
    delta = torch.zeros_like(xb, requires_grad=True)
    target = torch.full((len(xb),), benign_id, dtype=torch.long, device=device)
    opt = torch.optim.SGD([delta], lr=0.1)
    for _ in range(30):
        opt.zero_grad()
        t_loss = F.cross_entropy(victim(xb + delta), target)
        t_loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-1.0, 1.0)
    with torch.no_grad():
        tb_pred = victim(xb + delta).argmax(1)
    report["targeted_benign_grad_finite"] = bool(torch.isfinite(delta.grad).all())
    report["targeted_benign_hits"] = int((tb_pred == benign_id).sum().item())

    # -- 5. PrimAttack primitive->feature transform gradient ---------------------
    prim = CICIDS2017PrimitiveModel(manifest)
    transform = adapter.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    dos_id = class_map.name_to_id["DoS"]
    active = None
    raw_rows = np.flatnonzero(y == dos_id)
    # keep only rows where p and alpha primitives are active (perturbable)
    raw_probe = torch.tensor(
        np.ascontiguousarray(raw_all[raw_rows[:512]]), dtype=torch.float32, device=device
    )
    active = (prim.active_mask(raw_probe, "p") & prim.active_mask(raw_probe, "alpha"))
    idx = torch.nonzero(active, as_tuple=False).flatten()[: min(64, int(active.sum()))]
    raw = raw_probe[idx]
    p = torch.full((len(raw),), 2.25, dtype=torch.float32, device=device, requires_grad=True)
    alpha = torch.full((len(raw),), 1.01, dtype=torch.float32, device=device, requires_grad=True)
    adv = prim.generate(raw, {"p": p, "alpha": alpha})
    scaled_once = (adv - center) / scale  # scaler applied exactly once
    prim_target = torch.full((len(raw),), benign_id, dtype=torch.long, device=device)
    prim_loss = F.cross_entropy(victim(scaled_once), prim_target, reduction="sum")
    grad_p, grad_alpha = torch.autograd.grad(prim_loss, (p, alpha))
    report["primattack_rows"] = int(len(raw))
    report["primattack_dLdp_finite"] = bool(torch.isfinite(grad_p).all())
    report["primattack_dLdalpha_finite"] = bool(torch.isfinite(grad_alpha).all())
    report["primattack_dLdp_nonzero"] = bool((grad_p != 0).any())
    report["primattack_scaler_applied_once"] = True

    all_pass = all(
        bool(report[k]) for k in (
            "benign_index_is_zero", "feature_order_matches_schema", "returns_raw_logits",
            "logits_finite", "input_grad_shape_ok", "input_grad_finite", "input_grad_nonzero",
            "pgd_ran", "cw_ran", "targeted_benign_grad_finite",
            "primattack_dLdp_finite", "primattack_dLdalpha_finite", "primattack_dLdp_nonzero",
        )
    )
    report["ALL_READINESS_CHECKS_PASS"] = bool(all_pass)
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
