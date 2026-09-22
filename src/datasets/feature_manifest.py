"""Dataset-agnostic feature description and ordering contract.

The manifest makes feature *semantics* independent of feature *positions*. Generic
model / constraint / attack code resolves features through this object
(``index_by_name`` / ``indices_of_value_type`` / ``index_by_semantic``) instead of
hard-coded integer columns. It is the single runtime contract that ties together
dataset arrays, the fitted transform, and saved checkpoints; any disagreement in
width / ordering / content raises loudly rather than silently mis-indexing.

Nothing here is CICIoT-specific. Dataset adapters (``datasets.ciciot2023`` etc.)
build a manifest; the VAE/constraint code only ever consumes one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

# Canonical value-type vocabulary. These describe how a feature must be *generated*
# and *bounded*, not what it is named. A field named "TCP" may be a [0,1] window
# average (``probability``), not a ``binary``.
VALUE_TYPES: frozenset[str] = frozenset(
    {
        "real",  # unrestricted continuous
        "positive_continuous",  # >= 0, no finite upper bound
        "bounded_continuous",  # [lower, upper], both finite
        "probability",  # [0,1] aggregate / occurrence frequency
        "integer_count",  # true non-negative integer count
        "categorical",  # finite unordered code with cardinality
        "binary",  # true {0,1}
        "derived",  # exactly determined by parents; not freely generated
    }
)

MANIFEST_SCHEMA_VERSION = "1.0"


class ManifestError(ValueError):
    """Raised on any manifest construction or compatibility violation."""


@dataclass(frozen=True)
class FeatureSpec:
    """Semantics of one feature, decoupled from its column position.

    ``model_index`` is the authoritative position in the frozen feature order; it
    must equal the feature's index within the manifest. ``value_type`` drives typed
    decoding and Layer-0 projection; ``semantic_type`` allows role-based lookup
    (e.g. all ``packet_size`` stats) without referencing column numbers.
    """

    name: str
    model_index: int
    value_type: str
    semantic_type: str

    lower: float | None = None
    upper: float | None = None

    # ``None`` = perturbability not yet mined (see PerturbabilityScorer stage).
    # It is intentionally not hand-authored here.
    mutable: bool | None = None

    primitive_or_derived: str = "primitive"  # "primitive" | "derived"
    parents: tuple[str, ...] = ()
    derivation: str | None = None  # symbolic tag resolved by the constraint layer

    scaling: str = "robust"  # transform family key understood by FeatureTransform
    aggregation: str | None = None  # e.g. "mean", "sum", "min", "max", None
    protocol_scope: str | None = None  # protocol this feature applies to, if any
    cardinality: int | None = None  # for value_type == "categorical"

    def __post_init__(self) -> None:
        if self.value_type not in VALUE_TYPES:
            raise ManifestError(
                f"{self.name}: unknown value_type {self.value_type!r}; "
                f"allowed={sorted(VALUE_TYPES)}"
            )
        if self.primitive_or_derived not in {"primitive", "derived"}:
            raise ManifestError(
                f"{self.name}: primitive_or_derived must be 'primitive' or 'derived'"
            )
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ManifestError(
                f"{self.name}: lower {self.lower} > upper {self.upper}"
            )
        if self.value_type == "bounded_continuous" and (
            self.lower is None or self.upper is None
        ):
            raise ManifestError(
                f"{self.name}: bounded_continuous requires finite lower and upper"
            )
        if self.value_type == "categorical" and not self.cardinality:
            raise ManifestError(f"{self.name}: categorical requires cardinality>=2")
        if self.primitive_or_derived == "derived" and not self.parents:
            raise ManifestError(
                f"{self.name}: derived feature must declare parents"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureSpec":
        payload = dict(payload)
        payload["parents"] = tuple(payload.get("parents", ()) or ())
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in allowed})


class FeatureManifest:
    """Ordered, validated collection of :class:`FeatureSpec`.

    The order of ``specs`` *is* the feature-vector column order and is a hard
    contract. Construction validates uniqueness, contiguous ``model_index``, and
    bound sanity, then precomputes lookup maps.
    """

    def __init__(
        self,
        specs: Iterable[FeatureSpec],
        *,
        dataset_name: str,
        dataset_version: str = "1.0",
        schema_version: str = MANIFEST_SCHEMA_VERSION,
    ) -> None:
        specs = tuple(specs)
        if not specs:
            raise ManifestError("manifest requires at least one feature")
        self.dataset_name = str(dataset_name)
        self.dataset_version = str(dataset_version)
        self.schema_version = str(schema_version)
        self._specs = specs
        self._validate_order()
        self._by_name = {s.name: s for s in specs}
        if len(self._by_name) != len(specs):
            raise ManifestError("duplicate feature names in manifest")
        self._index_by_name = {s.name: s.model_index for s in specs}

    # -- construction invariants ------------------------------------------------
    def _validate_order(self) -> None:
        for position, spec in enumerate(self._specs):
            if spec.model_index != position:
                raise ManifestError(
                    f"feature {spec.name!r} has model_index {spec.model_index} "
                    f"but appears at position {position}; order is a hard contract"
                )
            for parent in spec.parents:
                # parents must be declared features (dependency closure check is
                # left to the constraint layer, which may allow forward refs).
                pass

    # -- basic accessors --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._specs)

    @property
    def n_features(self) -> int:
        return len(self._specs)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return self._specs

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def __iter__(self):
        return iter(self._specs)

    def __getitem__(self, key: int | str) -> FeatureSpec:
        if isinstance(key, str):
            if key not in self._by_name:
                raise KeyError(f"no feature named {key!r}")
            return self._by_name[key]
        return self._specs[key]

    # -- semantic / positional resolution --------------------------------------
    def index_by_name(self, name: str) -> int:
        try:
            return self._index_by_name[name]
        except KeyError:
            raise KeyError(f"no feature named {name!r}") from None

    def has(self, name: str) -> bool:
        return name in self._by_name

    def index_by_semantic(self, semantic_type: str) -> list[int]:
        return [s.model_index for s in self._specs if s.semantic_type == semantic_type]

    def indices_of_value_type(self, value_type: str) -> list[int]:
        if value_type not in VALUE_TYPES:
            raise ManifestError(f"unknown value_type {value_type!r}")
        return [s.model_index for s in self._specs if s.value_type == value_type]

    def derived_indices(self) -> list[int]:
        return [s.model_index for s in self._specs if s.primitive_or_derived == "derived"]

    def primitive_indices(self) -> list[int]:
        return [s.model_index for s in self._specs if s.primitive_or_derived == "primitive"]

    def lower_bounds(self) -> list[float | None]:
        return [s.lower for s in self._specs]

    def upper_bounds(self) -> list[float | None]:
        return [s.upper for s in self._specs]

    def mutable_mask(self) -> list[bool] | None:
        """Return per-feature mutability, or ``None`` if not yet mined.

        Returns ``None`` unless every feature has an explicit ``mutable`` flag, so
        callers cannot silently treat an un-mined manifest as fully mutable.
        """
        if any(s.mutable is None for s in self._specs):
            return None
        return [bool(s.mutable) for s in self._specs]

    # -- identity / versioning --------------------------------------------------
    @property
    def content_hash(self) -> str:
        """Stable sha256 over ordered spec content (excludes un-mined mutability).

        Used to bind a checkpoint / transform to the exact schema it was built
        with. Mutability is excluded so mining perturbability later does not
        invalidate existing base-VAE checkpoints.
        """
        payload = []
        for s in self._specs:
            d = s.to_dict()
            d.pop("mutable", None)
            payload.append(d)
        blob = json.dumps(
            {"dataset": self.dataset_name, "specs": payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- fail-loud compatibility checks ----------------------------------------
    def assert_matches_array(self, array_like) -> None:
        width = int(getattr(array_like, "shape", (0, 0))[1])
        if width != self.n_features:
            raise ManifestError(
                f"array width {width} != manifest n_features {self.n_features} "
                f"for dataset {self.dataset_name!r}"
            )

    def assert_matches_scaler(self, scaler) -> None:
        for attr in ("center_", "scale_"):
            if hasattr(scaler, attr):
                length = len(getattr(scaler, attr))
                if length != self.n_features:
                    raise ManifestError(
                        f"scaler.{attr} length {length} != manifest n_features "
                        f"{self.n_features}"
                    )

    def assert_compatible_hash(self, other_hash: str, *, context: str = "") -> None:
        if other_hash != self.content_hash:
            raise ManifestError(
                f"schema hash mismatch{f' ({context})' if context else ''}: "
                f"expected {self.content_hash[:12]}, got {str(other_hash)[:12]}"
            )

    def assert_names_match(self, names: list[str]) -> None:
        if list(names) != self.names:
            raise ManifestError(
                "feature-name/order mismatch with manifest "
                f"{self.dataset_name!r}: {self.n_features} expected"
            )

    # -- (de)serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "content_hash": self.content_hash,
            "features": [s.to_dict() for s in self._specs],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureManifest":
        specs = [FeatureSpec.from_dict(d) for d in payload["features"]]
        return cls(
            specs,
            dataset_name=payload["dataset_name"],
            dataset_version=payload.get("dataset_version", "1.0"),
            schema_version=payload.get("schema_version", MANIFEST_SCHEMA_VERSION),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_mutability(self, mutable_by_name: dict[str, bool]) -> "FeatureManifest":
        """Return a copy with mined perturbability applied (does not mutate self)."""
        new_specs = [
            replace(s, mutable=mutable_by_name.get(s.name, s.mutable))
            for s in self._specs
        ]
        return FeatureManifest(
            new_specs,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            schema_version=self.schema_version,
        )
