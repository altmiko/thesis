"""Code-derived dependency map of PrimAttack's canonical recomputation φ and the P2
direct-only reduction of it.

``derive_phi_mapping`` reads the source of ``CICIDS2017PrimitiveModel.generate`` (the canonical
φ, not modified) and traces its local data flow on the AST. Every
``write_when(<feature>, <value>, <rows>)`` call is a φ write site. The backward slice of each
written value over the local assignments of ``generate`` gives

* the primitives it depends on (``p``, ``delay``, ``shape``);
* the source columns it reads (``col("...")``);
* the OTHER written quantities it reads (its parents).

Classification rule (mechanical; no feature name is consulted):

* DIRECT  - the value is a function of the primitive controls and source columns only; its
  slice reads no value that ``generate`` writes to another feature.
* DERIVED - the slice reads at least one value written to another feature, i.e. φ recomputes
  it from a quantity it has already changed.

The P2 reduced path (:class:`DirectOnlyPrimitiveModel`) keeps φ's direct writes and resets every
derived write to the source value. Because a direct value depends on nothing that φ changes, the
reduced vector carries exactly the values φ would write on the direct coordinates.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import re
import textwrap
from dataclasses import asdict, dataclass

import torch

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel

PRIMITIVES = ("p", "delay", "shape")
_SECTION = re.compile(r"#\s*-{2,}\s*(.+?)\s*-{2,}")


@dataclass(frozen=True)
class WriteSite:
    feature: str
    index: int
    line: int                      # line number inside generate()
    section: str                   # code block of generate() holding the write
    value: str                     # source text of the written value
    kind: str                      # "direct" | "derived"
    primitives: tuple[str, ...]    # primitives in the backward slice of the value
    gated_by: tuple[str, ...]      # primitives in the slice of the write's row mask
    parents: tuple[str, ...]       # features whose written value the slice reads first
    direct_bases: tuple[str, ...]  # direct features the value ultimately depends on
    source_reads: tuple[str, ...]  # source columns read anywhere in the slice


def _names(node: ast.AST) -> set[str]:
    """Local names and source columns read by an expression (``src:<feature>`` for col())."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "col"
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add(f"src:{n.args[0].value}")
    out.discard("col")
    return out


def _generate_source() -> tuple[str, int]:
    src, start = inspect.getsourcelines(CICIDS2017PrimitiveModel.generate)
    return textwrap.dedent("".join(src)), start


def derive_phi_mapping(manifest) -> dict:
    """Trace ``CICIDS2017PrimitiveModel.generate`` and classify each φ write site."""
    code, first_line = _generate_source()
    tree = ast.parse(code)
    fn = tree.body[0]
    lines = code.splitlines()
    sections = [(k + 1, m.group(1)) for k, line in enumerate(lines)
                if (m := _SECTION.search(line))]

    defs: dict[str, set[str]] = {}
    writes: list[tuple[str, ast.AST, ast.AST, int]] = []

    def visit(stmts) -> None:
        for st in stmts:
            if isinstance(st, ast.FunctionDef):
                continue  # the write_when helper itself
            if isinstance(st, ast.If):
                visit(st.body)
                visit(st.orelse)
                continue
            if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                reads = _names(st.value) if st.value is not None else set()
                if isinstance(st, ast.AugAssign):
                    reads |= _names(st.target)
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                            defs.setdefault(n.id, set()).update(reads)
                continue
            if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                    and isinstance(st.value.func, ast.Name) and st.value.func.id == "write_when"):
                feature_node, value, rows = st.value.args
                writes.append((feature_node.value, value, rows, st.lineno))

    visit(fn.body)
    if not writes:
        raise AssertionError("no write_when sites found in CICIDS2017PrimitiveModel.generate")

    written_var: dict[str, list[str]] = {}
    for feature, value, _rows, _ln in writes:
        if isinstance(value, ast.Name):
            written_var.setdefault(value.id, []).append(feature)

    def slice_of(start: set[str], own: str | None):
        """Full backward slice: primitives, source columns, every other written value read."""
        prims, srcs, written, seen = set(), set(), set(), set()
        frontier = list(start)
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in PRIMITIVES:
                prims.add(name)
                continue
            if name.startswith("src:"):
                srcs.add(name[4:])
                continue
            if name in written_var and name != own:
                written.add(name)
            frontier.extend(defs.get(name, ()))
        return prims, srcs, written

    def nearest_parents(start: set[str], own: str | None) -> set[str]:
        """Written quantities reached before any other written quantity on each path."""
        out, seen, frontier = set(), set(), list(start)
        while frontier:
            nxt = []
            for name in frontier:
                if name in seen:
                    continue
                seen.add(name)
                if name in written_var and name != own:
                    out.add(name)
                    continue
                if name in defs and name not in PRIMITIVES:
                    nxt.extend(defs[name])
            frontier = nxt
        return out

    names = list(manifest.names)
    sites: list[WriteSite] = []
    for feature, value, rows, ln in writes:
        own = value.id if isinstance(value, ast.Name) else None
        start = _names(value)
        prims, srcs, all_parents = slice_of(start, own)
        gated, _, _ = slice_of(_names(rows), None)
        near = nearest_parents(start, own)
        section = next((s for l0, s in reversed(sections) if l0 <= ln), "")
        sites.append(WriteSite(
            feature=feature, index=names.index(feature), line=first_line + ln - 1,
            section=section, value=ast.unparse(value),
            kind="derived" if all_parents else "direct",
            primitives=tuple(p for p in PRIMITIVES if p in prims),
            gated_by=tuple(p for p in PRIMITIVES if p in gated),
            parents=tuple(sorted({f for v in near for f in written_var[v]} - {feature})),
            direct_bases=(), source_reads=tuple(sorted(srcs))))

    if len({s.feature for s in sites}) != len(sites):
        raise AssertionError("a feature is written by more than one write site")
    direct = {s.feature for s in sites if s.kind == "direct"}
    by_feature = {s.feature: s for s in sites}

    def bases(feature: str, trail=()) -> set[str]:
        if feature in direct:
            return {feature}
        out = set()
        for parent in by_feature[feature].parents:
            if parent not in trail:
                out |= bases(parent, trail + (feature,))
        return out

    sites = [WriteSite(**{**asdict(s), "direct_bases": tuple(sorted(bases(s.feature)))})
             for s in sites]
    derived_groups: dict[str, list[str]] = {}
    for s in sites:
        if s.kind == "derived":
            derived_groups.setdefault(s.section, []).append(s.feature)
    per_primitive = {
        p: {"direct": [s.feature for s in sites if s.kind == "direct" and p in s.primitives],
            "derived": [s.feature for s in sites if s.kind == "derived" and p in s.primitives]}
        for p in PRIMITIVES}
    return {
        "source": "src/attack/realizability/cicids2017.py:CICIDS2017PrimitiveModel.generate",
        "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "dataset": manifest.dataset_name,
        "rule": ("direct = the written value's backward slice (local assignments of generate) "
                 "reaches the primitive controls and source columns only; derived = the slice "
                 "reads a value that generate writes to another feature"),
        "write_sites": [asdict(s) for s in sites],
        "direct_features": [s.feature for s in sites if s.kind == "direct"],
        "derived_features": [s.feature for s in sites if s.kind == "derived"],
        "derived_groups": derived_groups,
        "per_primitive": per_primitive,
        "not_written": [n for n in names if n not in by_feature],
    }


def feature_masks(mapping: dict, n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean ``[F]`` masks of the direct and derived φ write sites."""
    direct = torch.zeros(n_features, dtype=torch.bool)
    derived = torch.zeros(n_features, dtype=torch.bool)
    for s in mapping["write_sites"]:
        (direct if s["kind"] == "direct" else derived)[s["index"]] = True
    return direct, derived


class DirectOnlyPrimitiveModel:
    """P2 reduced recomputation ``recompute_mode="direct_only"``.

    Projection, capabilities, bounds and quantization are the canonical model's. ``generate``
    runs canonical φ and keeps only its DIRECT writes; every DERIVED write is reset to the source
    flow's value, so dependent statistics are not propagated.
    """

    recompute_mode = "direct_only"

    def __init__(self, model: CICIDS2017PrimitiveModel, mapping: dict) -> None:
        self.full = model
        self.i = model.i
        self.direct_mask, self.derived_mask = feature_masks(mapping, len(model.feature_names))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.full.feature_names

    def project_controls(self, raw, controls, bounds, *, capabilities=None):
        return self.full.project_controls(raw, controls, bounds, capabilities=capabilities)

    def generate(self, raw: torch.Tensor, controls, *, quantize: bool = False,
                 capabilities=None) -> torch.Tensor:
        full = self.full.generate(raw, controls, quantize=quantize, capabilities=capabilities)
        return torch.where(self.direct_mask.to(raw.device)[None, :], full, raw)


@torch.no_grad()
def realize_full_phi(reduced_model: DirectOnlyPrimitiveModel, res, *, victim, raw, center, scale,
                     bounds, caps, objective):
    """Pass the primitives a direct-only search returned through canonical φ.

    Returns ``(result, extra)``. ``result`` is ``res`` with the realized flow, logits, margin and
    success replaced by those of ``φ(x, u)`` for the SAME projected primitives ``u`` (success =
    objective hit on the full-φ flow; validator_v2 is applied afterwards by the runner). The
    search's own view is returned in ``extra`` as ``reduced_*``. Evaluation bookkeeping
    (queries, first-success indices, candidate source) still describes the search. This
    re-evaluation is an outcome measurement, not a search query, and is not charged to the
    256-evaluation budget.
    """
    model = reduced_model.full
    u = res.projected
    reprojected = model.project_controls(raw, u, bounds, capabilities=caps)
    for name in ("p", "delay", "shape"):
        if not torch.equal(reprojected[name], u[name]):
            raise AssertionError(f"returned primitive {name} is not a fixed point of projection")
    reduced = res.adversarial_raw
    if not torch.equal(reduced, reduced_model.generate(raw, u, quantize=True, capabilities=caps)):
        raise AssertionError("returned reduced flow is not the direct-only map of its primitives")
    full = model.generate(raw, u, quantize=True, capabilities=caps)
    derived = reduced_model.derived_mask.to(raw.device)
    if bool((full != reduced)[:, ~derived].any()):
        raise AssertionError("full φ and direct-only flows differ outside the derived features")
    logits = victim((full - center) / scale)
    hit = objective.hit(logits)
    result = dataclasses.replace(
        res, adversarial_raw=full, logits=logits, objective_margin=objective.margin(logits),
        valid=torch.ones_like(hit), success=hit)
    extra = {
        "reduced_adv_raw": reduced,
        "reduced_logits": res.logits,
        "reduced_pred": res.logits.argmax(1),
        "reduced_margin": res.objective_margin,
        "reduced_success": res.success,
        "full_logits": logits,
    }
    return result, extra


def empirical_check(model: CICIDS2017PrimitiveModel, mapping: dict, raw: torch.Tensor) -> dict:
    """Which features φ actually changes per primitive on real flows, vs the traced slices.

    ``raw``: source flows; only flows admitting both padding and timing are used. Each primitive
    is switched on alone (p = 1 byte; delay = 1e3 µs at shape 0 and at shape 1); a feature is
    "changed by" a primitive if any flow's value moves.
    """
    caps = model.infer_capabilities(raw)
    keep = caps.pad_allowed & caps.timing_allowed
    raw = raw[keep]
    caps = model.infer_capabilities(raw)
    n = raw.shape[0]
    z = torch.zeros(n, dtype=raw.dtype, device=raw.device)
    one = torch.ones_like(z)
    names = model.feature_names

    def changed(a, b) -> set[str]:
        return {names[j] for j in torch.nonzero((a != b).any(0)).flatten().tolist()}

    pad = model.generate(raw, {"p": one, "delay": z, "shape": z}, quantize=True, capabilities=caps)
    d0 = model.generate(raw, {"p": z, "delay": 1e3 * one, "shape": z}, quantize=True,
                        capabilities=caps)
    d1 = model.generate(raw, {"p": z, "delay": 1e3 * one, "shape": one}, quantize=True,
                        capabilities=caps)
    observed = {"p": changed(pad, raw), "delay": changed(d0, raw) | changed(d1, raw),
                "shape": changed(d0, d1)}
    traced = {p: set(mapping["per_primitive"][p]["direct"])
              | set(mapping["per_primitive"][p]["derived"]) for p in PRIMITIVES}
    reduced = DirectOnlyPrimitiveModel(model, mapping)
    ctrl = {"p": one, "delay": 1e3 * one, "shape": 0.5 * one}
    full = model.generate(raw, ctrl, quantize=True, capabilities=caps)
    red = reduced.generate(raw, ctrl, quantize=True, capabilities=caps)
    direct = set(mapping["direct_features"])
    derived = set(mapping["derived_features"])
    return {
        "n_flows": int(n),
        "observed_changed": {p: sorted(v) for p, v in observed.items()},
        "observed_subset_of_traced": {p: observed[p] <= traced[p] for p in PRIMITIVES},
        "traced_not_observed": {p: sorted(traced[p] - observed[p]) for p in PRIMITIVES},
        "reduced_changes_only_direct": changed(red, raw) <= direct,
        "reduced_equals_full_on_direct": bool(
            (red[:, reduced.direct_mask] == full[:, reduced.direct_mask]).all()),
        "full_minus_reduced_only_derived": changed(full, red) <= derived,
        "derived_changed_by_full": sorted(changed(full, red)),
    }


def mapping_markdown(mapping: dict, check: dict | None = None) -> list[str]:
    lines = [
        "## φ dependency map (traced from code)", "",
        f"Source: `{mapping['source']}` (sha256 of the function source "
        f"`{mapping['source_sha256'][:16]}…`). Rule: {mapping['rule']}.", "",
        "### primitive → direct (base) quantities → recomputed dependent features", "",
        "| primitive | direct / base features (kept in P2) | dependent features recomputed by φ "
        "(frozen at source in P2) |", "|---|---|---|"]
    for p, d in mapping["per_primitive"].items():
        lines.append(f"| `{p}` | {', '.join(d['direct']) or '-'} | {', '.join(d['derived']) or '-'} |")
    lines += ["", "### Write sites", "",
              "| feature | kind | φ code block | primitives | gated by | parents (written "
              "values read) | direct bases | written value |",
              "|---|---|---|---|---|---|---|---|"]
    for s in mapping["write_sites"]:
        lines.append(
            f"| {s['feature']} | {s['kind']} | {s['section']} | {', '.join(s['primitives'])} | "
            f"{', '.join(s['gated_by'])} | {', '.join(s['parents']) or '-'} | "
            f"{', '.join(s['direct_bases'])} | `{s['value']}` |")
    lines += ["", f"{len(mapping['not_written'])} of "
              f"{len(mapping['not_written']) + len(mapping['write_sites'])} features are never "
              "written by φ and stay at the source value in both arms.", ""]
    if check is not None:
        lines += ["### Empirical check on real flows", "",
                  f"{check['n_flows']} source flows admitting padding and timing; each primitive "
                  "switched on alone.", ""]
        for p in PRIMITIVES:
            lines.append(f"* `{p}` changes {len(check['observed_changed'][p])} features; all "
                         f"inside the traced slice: {check['observed_subset_of_traced'][p]}; "
                         f"traced but unchanged here: "
                         f"{', '.join(check['traced_not_observed'][p]) or 'none'}.")
        lines += [f"* reduced vector changes only direct features: "
                  f"{check['reduced_changes_only_direct']}; equals φ on direct features: "
                  f"{check['reduced_equals_full_on_direct']}; φ − reduced differ only on derived "
                  f"features: {check['full_minus_reduced_only_derived']}.", ""]
    return lines
