"""CICIDS2017-DistriNet realization of the primitive-control attack.

Two attacker-controlled primitives per flow, both differentiable:

* ``p >= 0``  -- **forward packet-length augmentation** (bytes added to every forward
  packet's *length*). This is a feature-level model of forward padding (e.g. TCP/MSS
  padding or filler segments). It is NOT asserted to insert application-layer *data*: the
  data-bearing-packet count (``Fwd Act Data Pkts``) is therefore held frozen and flagged as
  a Level-C (packet-trace) limitation, not overclaimed. See ``roles``.
* ``alpha >= 1`` -- **forward timing dilation**: stretch every forward inter-arrival gap by
  ``alpha`` (delay only; can never compress a flow, so duration stays positive and rates
  stay finite). Masked to ``1`` for single-packet flows (no forward IAT sequence).

Dependency semantics were mined on the pristine TRAIN split (leakage-safe); every identity
below had 0% violation across 400k-600k train rows:

    Fwd Packet Length Mean = Total Length of Fwd Packet / Total Fwd Packet         (exact)
    Fwd Segment Size Avg   = Fwd Packet Length Mean                                (exact)
    Packet Length Mean     = (TL_fwd + TL_bwd) / (Nf + Nb)                         (exact)
    Average Packet Size    = Packet Length Mean                                    (exact)
    Packet Length Max/Min  = ext(fwd, bwd) with direction-presence branch          (exact)
    Packet Length Variance = pooled sample variance(nf,mf,sf, nb,mb,sb)            (exact)
    Fwd IAT Mean           = Fwd IAT Total / (Nf - 1)                              (exact)
    Flow IAT Mean          = Flow Duration / (Nf + Nb - 1)                         (exact)
    <dir> Packets/s        = count / (duration_us / 1e6)                           (exact)
    Flow Bytes/s           = (TL_fwd + TL_bwd) / (duration_us / 1e6)               (exact)

Padding shift identities (fwd min/max/mean += p, fwd std invariant, TL_fwd += Nf*p) follow
from adding a constant to every forward packet length -- a *threat-model definition*, not an
extractor identity. Flow Duration under dilation is a conservative packet-sequence
projection (all added forward delay extends the flow); Flow IAT Max is grown by the same
delay so ``mean <= max <= duration`` stays exact -- both are labelled CONDITIONAL_DERIVED /
Level-C-approximate. Subflow bytes, bulk stats, Flow IAT Std/Min are NOT reconstructable
from the aggregate flow and are held frozen (Level-C limitation), never fabricated.
"""
from __future__ import annotations

from typing import Mapping

import torch

from attack.realizability.base import (
    FeatureRole,
    IdentityCheck,
    PrimitiveSpec,
)
from datasets.feature_manifest import FeatureManifest

# RobustScaler float64 round-trip residue is <= 1.5e-8 (audited); frozen-preservation slack.
SCALER_ATOL = 1e-6
_DUR_FLOOR_US = 1.0  # 1 microsecond timestamp resolution (CICFlowMeter emits integer us)

# Integer-valued features whose realizable values must be integral after projection
# (data-mined: >= 99.999% integral on 600k train rows).
_INTEGER_FEATURES = frozenset({
    "Src Port", "Dst Port", "Protocol", "Flow Duration", "Total Fwd Packet",
    "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Packet Length Min", "Packet Length Max",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Max", "Active Min",
    "Idle Max", "Idle Min",
})

# Integer features our transform WRITES (must be rounded on projection).
_QUANTIZE_LENGTH = ("Total Length of Fwd Packet", "Fwd Packet Length Max",
                    "Fwd Packet Length Min", "Packet Length Max", "Packet Length Min")
_QUANTIZE_TIMING = ("Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min", "Flow Duration",
                    "Flow IAT Max")


class CICIDS2017PrimitiveModel:
    """Differentiable (p, alpha) -> 79-feature map with complete dependency recomputation."""

    dataset = "cicids2017_distrinet"

    def __init__(self, manifest: FeatureManifest, *, dur_floor_us: float = _DUR_FLOOR_US) -> None:
        self.manifest = manifest
        self.dur_floor_us = float(dur_floor_us)
        self._names = tuple(manifest.names)
        self.i = {n: manifest.index_by_name(n) for n in self._names}
        roles = self.roles()
        self.controlled_idx = sorted(
            self.i[n] for n, (r, _) in roles.items()
            if r in (FeatureRole.DERIVED_P, FeatureRole.DERIVED_T, FeatureRole.DERIVED,
                     FeatureRole.CONDITIONAL, FeatureRole.RATE)
        )
        # Preserved exactly = everything the transform does not write: genuinely frozen (F),
        # proven-invariant (I), and Level-C held-constant (Fᶜ).
        self.frozen_idx = [j for j in range(manifest.n_features) if j not in set(self.controlled_idx)]

    # -- interface -------------------------------------------------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def primitives(self) -> tuple[PrimitiveSpec, ...]:
        return (
            PrimitiveSpec(
                name="p", identity=0.0, lower=0.0, upper=None, integer=True,
                description="forward packet-length augmentation (bytes added per fwd packet)",
                units="bytes",
            ),
            PrimitiveSpec(
                name="alpha", identity=1.0, lower=1.0, upper=None, integer=False,
                description="forward inter-arrival timing dilation (delay factor)",
                units="ratio",
            ),
        )

    def integer_features(self) -> frozenset[str]:
        return _INTEGER_FEATURES

    def roles(self) -> dict[str, tuple[FeatureRole, str]]:
        """Precise role of every feature w.r.t. the two primitives.

        Reminder: no CICFlowMeter feature is optimized directly -- only ``p`` and ``alpha``
        are attack variables. ``Dp/Dt`` = exactly derived from p / alpha; ``D`` = other exact
        derived; ``C`` = conditional/conservative reconstruction; ``R`` = rate; ``I`` = PROVEN
        invariant under the relevant primitive; ``F`` = genuinely unaffected; ``Fᶜ`` = UNRESOLVED
        and held constant (would change under real packet edits, not aggregate-reconstructable).
        """
        R = FeatureRole
        roles: dict[str, tuple[FeatureRole, str]] = {n: (R.FROZEN, "genuinely unaffected by p or alpha")
                                                      for n in self._names}
        # --- derived from packet-length primitive p (Dp) ---
        roles["Total Length of Fwd Packet"] = (R.DERIVED_P, "= TL_fwd0 + Nf*p (padding accumulator)")
        roles["Fwd Packet Length Min"] = (R.DERIVED_P, "= min0 + p (uniform fwd length shift)")
        roles["Fwd Packet Length Max"] = (R.DERIVED_P, "= max0 + p (uniform fwd length shift)")
        roles["Fwd Packet Length Mean"] = (R.DERIVED_P, "= TL_fwd / Nf (exact)")
        roles["Fwd Segment Size Avg"] = (R.DERIVED_P, "= Fwd Packet Length Mean (exact)")
        roles["Fwd Packet Length Std"] = (R.INVARIANT, "PROVEN invariant: uniform shift preserves fwd std")
        # combined length stats depend on p (via fwd block)
        roles["Packet Length Mean"] = (R.DERIVED_P, "= (TL_fwd+TL_bwd)/(Nf+Nb) (exact)")
        roles["Average Packet Size"] = (R.DERIVED_P, "= Packet Length Mean (exact)")
        roles["Packet Length Variance"] = (R.DERIVED_P, "pooled sample variance (exact)")
        roles["Packet Length Std"] = (R.DERIVED_P, "= sqrt(Packet Length Variance)")
        roles["Packet Length Max"] = (R.CONDITIONAL, "= ext(fwd_max+p, bwd_max), branch on direction presence")
        roles["Packet Length Min"] = (R.CONDITIONAL, "= ext(fwd_min+p, bwd_min), branch on direction presence")
        # --- derived from timing primitive alpha (Dt) ---
        roles["Fwd IAT Total"] = (R.DERIVED_T, "= alpha * Fwd IAT Total0 (uniform fwd IAT dilation)")
        roles["Fwd IAT Max"] = (R.DERIVED_T, "= alpha * Fwd IAT Max0")
        roles["Fwd IAT Min"] = (R.DERIVED_T, "= alpha * Fwd IAT Min0")
        roles["Fwd IAT Std"] = (R.DERIVED_T, "= alpha * Fwd IAT Std0 (scale-equivariant)")
        roles["Fwd IAT Mean"] = (R.DERIVED_T, "= Fwd IAT Total/(Nf-1) (exact)")
        roles["Flow Duration"] = (R.CONDITIONAL, "conservative delay projection (Level-C approximate)")
        roles["Flow IAT Mean"] = (R.DERIVED_T, "= Flow Duration/(Nf+Nb-1) (exact)")
        roles["Flow IAT Max"] = (R.CONDITIONAL, "= Flow IAT Max0 + added delay (preserves mean<=max<=dur)")
        # --- rates (depend on both primitives via totals/duration) ---
        for r in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
            roles[r] = (R.RATE, "count/byte over projected duration")
        # --- Fᶜ: UNRESOLVED, held constant (would change under real packet edits) ---
        roles["Fwd Act Data Pkts"] = (R.LEVEL_C, "p is length augmentation, not asserted payload insertion")
        roles["Subflow Fwd Bytes"] = (R.LEVEL_C, "subflow decomposition unknown from aggregate flow")
        roles["Fwd Bytes/Bulk Avg"] = (R.LEVEL_C, "bulk detection needs packet size/timing sequence")
        roles["Fwd Bulk Rate Avg"] = (R.LEVEL_C, "bulk detection needs packet size/timing sequence")
        roles["Fwd Packet/Bulk Avg"] = (R.LEVEL_C, "bulk detection needs packet size/timing sequence")
        roles["Flow IAT Std"] = (R.LEVEL_C, "merged fwd+bwd gap sequence reshuffles under fwd dilation")
        roles["Flow IAT Min"] = (R.LEVEL_C, "smallest merged gap not reconstructable; held constant")
        for f in ("Active Mean", "Active Std", "Active Max", "Active Min",
                  "Idle Mean", "Idle Std", "Idle Max", "Idle Min"):
            roles[f] = (R.LEVEL_C, "active/idle bursts depend on packet-level timing; not "
                                   "reconstructable from aggregates -- held constant, NOT invariant")
        return roles

    def algebraic_identities(self) -> tuple[IdentityCheck, ...]:
        """Exact identities (0% train violation) checked independently by the validator."""
        def nf(c): return c("Total Fwd Packet").clamp(min=1.0)
        def nfm1(c): return (c("Total Fwd Packet") - 1.0).clamp(min=1.0)
        def nb(c): return c("Total Bwd packets")
        def nall(c): return (c("Total Fwd Packet") + c("Total Bwd packets")).clamp(min=1.0)
        def nallm1(c): return (c("Total Fwd Packet") + c("Total Bwd packets") - 1.0).clamp(min=1.0)
        def durs(c): return (c("Flow Duration") / 1e6).clamp(min=1e-12)

        def cond_max(c):
            fm = c("Fwd Packet Length Max"); bm = c("Bwd Packet Length Max")
            has_b = c("Total Bwd packets") > 0; has_f = c("Total Fwd Packet") > 0
            v = torch.where(has_b, torch.maximum(fm, bm), fm)
            return torch.where(has_f, v, bm)

        def cond_min(c):
            fm = c("Fwd Packet Length Min"); bm = c("Bwd Packet Length Min")
            has_b = c("Total Bwd packets") > 0; has_f = c("Total Fwd Packet") > 0
            v = torch.where(has_b, torch.minimum(fm, bm), fm)
            return torch.where(has_f, v, bm)

        return (
            IdentityCheck("Fwd Packet Length Mean", ("Total Length of Fwd Packet", "Total Fwd Packet"),
                          lambda c: c("Total Length of Fwd Packet") / nf(c)),
            IdentityCheck("Fwd Segment Size Avg", ("Fwd Packet Length Mean",),
                          lambda c: c("Fwd Packet Length Mean")),
            IdentityCheck("Bwd Packet Length Mean", ("Total Length of Bwd Packet", "Total Bwd packets"),
                          lambda c: c("Total Length of Bwd Packet") / nb(c).clamp(min=1.0)),
            IdentityCheck("Bwd Segment Size Avg", ("Bwd Packet Length Mean",),
                          lambda c: c("Bwd Packet Length Mean")),
            IdentityCheck("Packet Length Mean", ("Total Length of Fwd Packet", "Total Length of Bwd Packet"),
                          lambda c: (c("Total Length of Fwd Packet") + c("Total Length of Bwd Packet")) / nall(c)),
            IdentityCheck("Average Packet Size", ("Packet Length Mean",),
                          lambda c: c("Packet Length Mean")),
            IdentityCheck("Packet Length Variance", ("Packet Length Std",),
                          lambda c: c("Packet Length Std") ** 2, atol=1.0, rtol=1e-3),
            IdentityCheck("Packet Length Max", ("Fwd Packet Length Max", "Bwd Packet Length Max"), cond_max),
            IdentityCheck("Packet Length Min", ("Fwd Packet Length Min", "Bwd Packet Length Min"), cond_min),
            IdentityCheck("Fwd IAT Mean", ("Fwd IAT Total", "Total Fwd Packet"),
                          lambda c: c("Fwd IAT Total") / nfm1(c), atol=1.0, rtol=1e-4),
            IdentityCheck("Flow IAT Mean", ("Flow Duration", "Total Fwd Packet", "Total Bwd packets"),
                          lambda c: c("Flow Duration") / nallm1(c), atol=1.0, rtol=1e-4),
            IdentityCheck("Fwd Packets/s", ("Total Fwd Packet", "Flow Duration"),
                          lambda c: c("Total Fwd Packet") / durs(c), atol=1e-3, rtol=1e-3),
            IdentityCheck("Bwd Packets/s", ("Total Bwd packets", "Flow Duration"),
                          lambda c: c("Total Bwd packets") / durs(c), atol=1e-3, rtol=1e-3),
            IdentityCheck("Flow Packets/s", ("Total Fwd Packet", "Total Bwd packets", "Flow Duration"),
                          lambda c: nall(c) / durs(c) * (nall(c) > 0), atol=1e-3, rtol=1e-3),
            IdentityCheck("Flow Bytes/s", ("Total Length of Fwd Packet", "Total Length of Bwd Packet", "Flow Duration"),
                          lambda c: (c("Total Length of Fwd Packet") + c("Total Length of Bwd Packet")) / durs(c),
                          atol=1e-2, rtol=1e-3),
        )

    # -- primitive activity & bounds ------------------------------------------
    def active_mask(self, raw: torch.Tensor, primitive: str) -> torch.Tensor:
        Nf = raw[:, self.i["Total Fwd Packet"]]
        if primitive == "p":
            return Nf >= 1.0
        if primitive == "alpha":
            return Nf >= 2.0  # need >=2 fwd packets for a forward IAT sequence
        raise KeyError(primitive)

    def per_flow_bounds(self, raw: torch.Tensor, config: Mapping[str, float]) -> dict[str, torch.Tensor]:
        """Per-flow feasible upper bounds for (p, alpha).

        ``p`` headroom is data-mined: the padding must keep every controlled forward-length
        feature within the TRAIN envelope (``config['env_*']``) AND within an optional
        per-packet MTU-style cap (``config['mtu_cap']``, applied to the resulting per-packet
        max length). ``alpha`` headroom keeps the dilated forward-IAT features and the
        projected duration within the train timing envelope. Both are clamped to the
        absolute ``p_max`` / ``alpha_max`` ceilings.
        """
        i = self.i; col = lambda n: raw[:, i[n]]
        eps = 1e-9
        Nf = col("Total Fwd Packet").clamp(min=1.0)
        p_max = float(config["p_max"]); alpha_max = float(config["alpha_max"])
        mtu_cap = float(config.get("mtu_cap", 0.0))  # 0 => disabled

        # --- p headroom (data-mined envelope on fwd length features) ---
        p_list = [
            config["env_Fwd Packet Length Max"] - col("Fwd Packet Length Max"),
            config["env_Fwd Packet Length Min"] - col("Fwd Packet Length Min"),
            config["env_Fwd Packet Length Mean"] - col("Fwd Packet Length Mean"),
            (config["env_Total Length of Fwd Packet"] - col("Total Length of Fwd Packet")) / Nf,
        ]
        if mtu_cap > 0.0:
            # resulting per-packet max length <= mtu_cap  ->  p <= mtu_cap - fwd_max
            p_list.append(mtu_cap - col("Fwd Packet Length Max"))
        p_hi = torch.stack(p_list, 0).amin(0).clamp(min=0.0, max=p_max)

        # --- alpha headroom (timing envelope) ---
        a_list = []
        for n in ("Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std", "Fwd IAT Mean"):
            cc = col(n)
            a_list.append(torch.where(cc > 0, config[f"env_{n}"] / cc.clamp(min=eps),
                                      torch.full_like(cc, alpha_max)))
        fit = col("Fwd IAT Total")
        a_list.append(torch.where(
            fit > 0,
            (config["env_Flow Duration"] - col("Flow Duration")) / fit.clamp(min=eps) + 1.0,
            torch.full_like(fit, alpha_max),
        ))
        alpha_hi = torch.stack(a_list, 0).amin(0).clamp(min=1.0, max=alpha_max)
        return {"p": p_hi, "alpha": alpha_hi}

    # -- decoder -> primitive inference (VAE latent attack) --------------------
    def infer_primitives_from_decoded(
        self,
        raw0: torch.Tensor,
        decoded_adv_raw: torch.Tensor,
        decoded_base_raw: torch.Tensor,
        bounds: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Deterministic, differentiable map: VAE decoder proposal -> (p, alpha).

        The decoder proposes a *direction* of traffic modification; we read that direction
        only through the reconstructable forward-length / forward-timing signals and collapse
        it into the two realizable primitives. Movement is measured relative to the decoded
        source ``decode(z0)`` (not raw0) so the constant VAE reconstruction bias cancels and
        the primitive is driven purely by the latent displacement ``z_adv - z0``.

        p is a least-squares-consistent non-negative padding from the fwd-length signals;
        alpha is a non-negative-log dilation from the fwd-timing ratios. Both are clamped to
        the per-flow feasible caps ``bounds``. For single-forward-packet flows alpha is forced
        to the identity 1 *inside the graph* (no forward IAT sequence exists).
        """
        i = self.i
        da = lambda n: decoded_adv_raw[:, i[n]]
        db = lambda n: decoded_base_raw[:, i[n]]
        r0 = lambda n: raw0[:, i[n]]
        eps = 1e-6
        Nf_raw = r0("Total Fwd Packet"); Nf = Nf_raw.clamp(min=1.0)

        # --- p_hat: average of the four fwd-length movement signals (each estimates p) ---
        p_signals = torch.stack([
            da("Fwd Packet Length Mean") - db("Fwd Packet Length Mean"),
            da("Fwd Packet Length Max") - db("Fwd Packet Length Max"),
            da("Fwd Packet Length Min") - db("Fwd Packet Length Min"),
            (da("Total Length of Fwd Packet") - db("Total Length of Fwd Packet")) / Nf,
        ], 0).mean(0)
        # Differentiable projection to the feasible (non-negative) padding direction:
        # identity (no decoder movement) maps exactly to p=0, while positive proposed
        # length increases pass through with full gradient. Padding cannot shorten packets,
        # so clamping the negative direction to 0 is the correct (not merely convenient) map.
        p_hat = torch.relu(p_signals)
        p_hat = torch.minimum(p_hat, bounds["p"].clamp(min=0.0))
        p_hat = torch.where(self.active_mask(raw0, "p"), p_hat, torch.zeros_like(p_hat))

        # Geometric mean of timing ratios. Averaging log ratios and exponentiating is
        # the exact conversion back to a multiplicative delay factor.
        def ratio(n):
            base = db(n)
            return torch.where(base.abs() > eps, da(n) / base.clamp(min=eps), torch.ones_like(base))
        mean_log_ratio = torch.stack([
            torch.log(ratio("Fwd IAT Total").clamp(min=eps)),
            torch.log(ratio("Fwd IAT Mean").clamp(min=eps)),
            torch.log(ratio("Flow Duration").clamp(min=eps)),
        ], 0).mean(0)
        alpha_hat = torch.exp(torch.relu(mean_log_ratio))
        alpha_hat = torch.minimum(alpha_hat, bounds["alpha"].clamp(min=1.0))
        alpha_hat = torch.where(self.active_mask(raw0, "alpha"), alpha_hat, torch.ones_like(alpha_hat))
        return {"p": p_hat, "alpha": alpha_hat}

    # -- projection ------------------------------------------------------------
    def project_controls(self, raw: torch.Tensor, controls: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Realizable control values: p rounded to integer bytes; alpha kept continuous
        (its realizability is enforced by integer-microsecond quantization of the derived
        timing features inside :meth:`generate` with ``quantize=True``)."""
        p = controls["p"].clamp(min=0.0)
        alpha = controls["alpha"].clamp(min=1.0)
        p = torch.where(self.active_mask(raw, "p"), p, torch.zeros_like(p))
        alpha = torch.where(self.active_mask(raw, "alpha"), alpha, torch.ones_like(alpha))
        return {"p": torch.round(p), "alpha": alpha}

    # -- generation ------------------------------------------------------------
    def generate(self, raw: torch.Tensor, controls: Mapping[str, torch.Tensor], *, quantize: bool = False) -> torch.Tensor:
        i = self.i
        x = raw.clone()
        col = lambda n: raw[:, i[n]]
        p = controls["p"]; alpha = controls["alpha"]
        # active-mask primitives (single-packet flows cannot be timing-dilated; empty fwd cannot be padded)
        p = torch.where(self.active_mask(raw, "p"), p, torch.zeros_like(p)).clamp(min=0.0)
        alpha = torch.where(self.active_mask(raw, "alpha"), alpha, torch.ones_like(alpha)).clamp(min=1.0)
        if quantize:
            p = torch.round(p)

        Nf_raw = col("Total Fwd Packet"); Nf = Nf_raw.clamp(min=1.0)
        Nb = col("Total Bwd packets"); N = Nf_raw + Nb

        # ---- forward packet-length augmentation ----
        tl_fwd = col("Total Length of Fwd Packet") + Nf_raw * p
        fmin = col("Fwd Packet Length Min") + p
        fmax = col("Fwd Packet Length Max") + p
        if quantize:
            tl_fwd = torch.round(tl_fwd); fmin = torch.round(fmin); fmax = torch.round(fmax)
        fmean = tl_fwd / Nf
        fstd = col("Fwd Packet Length Std")  # shift-invariant
        x[:, i["Total Length of Fwd Packet"]] = tl_fwd
        x[:, i["Fwd Packet Length Min"]] = fmin
        x[:, i["Fwd Packet Length Max"]] = fmax
        x[:, i["Fwd Packet Length Mean"]] = fmean
        x[:, i["Fwd Segment Size Avg"]] = fmean

        # ---- forward timing dilation ----
        fit0 = col("Fwd IAT Total")
        fit = alpha * fit0
        fimax = alpha * col("Fwd IAT Max"); fimin = alpha * col("Fwd IAT Min")
        fistd = alpha * col("Fwd IAT Std")
        bwd_iat_total = col("Bwd IAT Total")
        dur = col("Flow Duration") + (fit - fit0)
        dur = torch.maximum(dur, fit)
        dur = torch.maximum(dur, bwd_iat_total).clamp(min=self.dur_floor_us)
        d_dur = dur - col("Flow Duration")
        flow_iat_max = col("Flow IAT Max") + d_dur.clamp(min=0.0)  # added delay lands in largest gap
        if quantize:
            fit = torch.round(fit); fimax = torch.round(fimax); fimin = torch.round(fimin)
            dur = torch.round(dur).clamp(min=self.dur_floor_us)
            flow_iat_max = torch.round(flow_iat_max)
        fimean = fit / (Nf - 1.0).clamp(min=1.0)
        flow_iat_mean = dur / (N - 1.0).clamp(min=1.0)
        x[:, i["Fwd IAT Total"]] = fit
        x[:, i["Fwd IAT Mean"]] = fimean
        x[:, i["Fwd IAT Std"]] = fistd
        x[:, i["Fwd IAT Max"]] = fimax
        x[:, i["Fwd IAT Min"]] = fimin
        x[:, i["Flow Duration"]] = dur
        x[:, i["Flow IAT Mean"]] = flow_iat_mean
        x[:, i["Flow IAT Max"]] = flow_iat_max

        # ---- combined (fwd+bwd) packet-length statistics ----
        bmin = col("Bwd Packet Length Min"); bmax = col("Bwd Packet Length Max")
        mb = col("Bwd Packet Length Mean"); sb = col("Bwd Packet Length Std")
        has_b = Nb > 0; has_f = Nf_raw > 0
        cmax = torch.where(has_b, torch.maximum(fmax, bmax), fmax)
        cmax = torch.where(has_f, cmax, bmax)
        cmin = torch.where(has_b, torch.minimum(fmin, bmin), fmin)
        cmin = torch.where(has_f, cmin, bmin)
        tl_bwd = col("Total Length of Bwd Packet")
        Nc = N.clamp(min=1.0)
        pmean = (tl_fwd + tl_bwd) / Nc
        cm = (Nf_raw * fmean + Nb * mb) / Nc
        SSf = (Nf_raw - 1.0).clamp(min=0.0) * fstd ** 2
        SSb = (Nb - 1.0).clamp(min=0.0) * sb ** 2
        between = Nf_raw * (fmean - cm) ** 2 + Nb * (mb - cm) ** 2
        pvar = torch.where(N >= 2.0, (SSf + SSb + between) / (N - 1.0).clamp(min=1.0),
                           torch.zeros_like(N)).clamp(min=0.0)
        pstd = torch.sqrt(pvar)
        x[:, i["Packet Length Max"]] = cmax
        x[:, i["Packet Length Min"]] = cmin
        x[:, i["Packet Length Mean"]] = pmean
        x[:, i["Average Packet Size"]] = pmean
        x[:, i["Packet Length Variance"]] = pvar
        x[:, i["Packet Length Std"]] = pstd

        # ---- rates (recomputed from new totals + projected duration) ----
        dur_s = (dur / 1.0e6).clamp(min=1e-12)
        x[:, i["Fwd Packets/s"]] = Nf_raw / dur_s
        x[:, i["Bwd Packets/s"]] = Nb / dur_s
        x[:, i["Flow Packets/s"]] = N / dur_s
        x[:, i["Flow Bytes/s"]] = (tl_fwd + tl_bwd) / dur_s
        return x
