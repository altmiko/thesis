"""Regenerate CICIDS2017_adversarial_examples.md from the primitive-control attack artifacts.

Same layout as the previous examples file (all 79 features, raw units, role, delta table),
but for the realizability-aware primitive-control attack. One example per class: the
smallest-cost successful Target->Benign strict-valid sample (seed 42).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from datasets import get_adapter

ROLE_TAG = {
    "primitive_controlled": "P", "direct_derived": "D", "conditional_derived": "C",
    "rate": "R", "frozen": "F", "level_c_frozen": "Fᶜ",
}
ART = "outputs/cicids2017_primitive_attack/attack_artifacts"


def fnum(v: float) -> str:
    a = abs(v)
    if v == 0:
        return "0"
    if a >= 1e6 or a < 1e-3:
        return f"{v:.4e}"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.4f}"


def main() -> None:
    ad = get_adapter("cicids2017"); man = ad.feature_manifest()
    model = CICIDS2017PrimitiveModel(man)
    names = list(man.names)
    roles = model.roles()
    idn = ad.class_mapping().id_to_name
    classes = ["DoS", "DDoS", "Recon", "BruteForce"]
    victims = ["mlp", "cnn", "lstm", "serial"]

    L = []
    L.append("# CICIDS2017-DistriNet — realizability-aware primitive-control adversarial examples\n")
    L.append("Successful adversarial network flows from the **primitive-control** attack: the "
             "attacker optimizes only two primitives per flow — forward packet-length augmentation "
             "`p` (bytes/fwd packet) and forward timing dilation `α` (delay factor) — then applies a "
             "discrete realizability projection (round `p` to integer bytes; µs-quantize timing) and "
             "recomputes every dependent CICFlowMeter feature.\n")
    L.append("- **Artifacts:** `outputs/cicids2017_primitive_attack/attack_artifacts/<Class>_<victim>_seed42.npz`")
    L.append("- **Class ids:** `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`")
    L.append("- **Feature role:** `P` = primitive-controlled, `D` = direct-derived (exact identity), "
             "`C` = conditional-derived (structure-dependent / conservative projection), `R` = rate "
             "(count·byte / projected duration), `F` = frozen (copied verbatim), `Fᶜ` = Level-C frozen "
             "(would change under real packet edits but not reconstructable from the aggregate flow).")
    L.append("- Values are the **pristine raw** original CICFlowMeter units; the example per class is the "
             "smallest normalized-cost row that was clean-correct and evaded to **Benign** while passing "
             "PAVE ∧ mined ∧ internal-realizability (strict valid).\n")
    L.append("---\n")

    for cl in classes:
        best = None
        for v in victims:
            f = Path(ART) / f"{cl}_{v}_seed42.npz"
            if not f.exists():
                continue
            d = np.load(f)
            strict = d["pave_valid"].astype(bool) & d["mined_valid"].astype(bool) & d["realizable"].astype(bool)
            ok = d["clean_correct"].astype(bool) & d["benign"].astype(bool) & strict
            idxs = np.flatnonzero(ok)
            if idxs.size == 0:
                continue
            r = int(idxs[np.argmin(d["cost_total"][idxs])])
            cost = float(d["cost_total"][r])
            if best is None or cost < best[0]:
                best = (cost, v, r, d)
        if best is None:
            L.append(f"## {cl} → (no successful strict-valid → Benign example)\n\n---\n")
            continue
        cost, v, r, d = best
        o = d["X_clean_raw"][r].astype(float); adv = d["X_adv_raw"][r].astype(float)
        L.append(f"## {cl} → Benign  (victim = {v}, row = {r})\n")
        L.append(f"- primitives: forward packet-length augmentation `p = {d['p_cont'][r]:.3f}` → "
                 f"`p_real = {int(d['p'][r])}` bytes;  timing dilation `α = {d['alpha_cont'][r]:.4f}` → "
                 f"`α_real = {d['alpha'][r]:.4f}`  |  normalized cost = {cost:.4f}")
        L.append(f"- **true class = {cl}  →  adversarial prediction = {idn[int(d['y_pred_adv'][r])]}**\n")
        L.append("### Base vs. adversarial (all 79 features, raw units)\n")
        L.append("| idx | feature | role | base (raw) | adversarial (raw) |")
        L.append("|---:|---|:--:|---:|---:|")
        for j, nm in enumerate(names):
            L.append(f"| {j} | {nm} | {ROLE_TAG[roles[nm][0]]} | {fnum(o[j])} | {fnum(adv[j])} |")
        L.append("\n### Delta — changed features only\n")
        L.append("| idx | feature | role | from | → | to |")
        L.append("|---:|---|:--:|---:|:--:|---:|")
        for j, nm in enumerate(names):
            if abs(adv[j] - o[j]) > 1e-5 + 1e-4 * abs(o[j]):
                L.append(f"| {j} | {nm} | {ROLE_TAG[roles[nm][0]]} | {fnum(o[j])} | → | {fnum(adv[j])} |")
        c = lambda n: adv[names.index(n)]
        checks = {
            "fwd mean<min": c("Fwd Packet Length Mean") < c("Fwd Packet Length Min") - 1e-6,
            "fwd max>total": c("Fwd Packet Length Max") > c("Total Length of Fwd Packet") + 1e-6,
            "pkt mean<min": c("Packet Length Mean") < c("Packet Length Min") - 1e-6,
            "fwd IAT max>total": c("Fwd IAT Max") > c("Fwd IAT Total") + 1e-3,
            "fwd IAT total>dur": c("Fwd IAT Total") > c("Flow Duration") + 1e-3,
            "flow IAT mean>max": c("Flow IAT Mean") > c("Flow IAT Max") + 1e-3,
            "any rate<0": min(c("Flow Bytes/s"), c("Flow Packets/s"), c("Fwd Packets/s"), c("Bwd Packets/s")) < 0,
            "duration<=0": c("Flow Duration") <= 0,
        }
        L.append("\n**Impossible-case checks (all must be False):** " +
                 ", ".join(f"{k}={bool(x)}" for k, x in checks.items()))
        L.append("\n---\n")

    out = Path("CICIDS2017_adversarial_examples.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
