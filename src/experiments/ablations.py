"""Config-driven ablation ladder A0-A6.

One code path, selected by :class:`AblationConfig`, instead of six model copies:

    A0  legacy structured decoder (original approach)
    A1  generic typed VAE
    A2  + Layer 0 hard projection in the attack pipeline
    A3  + Layer 1 generic soft constraints
    A4  + Layer 2 dataset/mined constraints
    A5  + masked residual attack head
    A6  + victim-guided Stage-B training

Jacobian / JSMA / tangent-space guidance is intentionally NOT part of this ladder
(deferred to a later stage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constraints import ConstraintEngine, RobustTailBound, load_layer2
from constraints.layer0 import Layer0Projector
from datasets.base import DatasetAdapter
from vae.model import MixedInputBetaVAE

_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    decoder_kind: str          # "legacy" | "typed"
    use_layer0: bool           # apply Layer-0 projection in the attack path
    use_layer1: bool           # generic soft constraints
    use_layer2: bool           # dataset-specific / mined constraints
    use_residual_head: bool    # masked residual adversarial head
    residual_mode: str         # "latent" | "residual" | "both"
    victim_guided_stage_b: bool


PRESETS: dict[str, AblationConfig] = {
    "A0": AblationConfig("A0", "legacy", False, False, False, False, "latent", False),
    "A1": AblationConfig("A1", "typed", False, False, False, False, "latent", False),
    "A2": AblationConfig("A2", "typed", True, False, False, False, "latent", False),
    "A3": AblationConfig("A3", "typed", True, True, False, False, "latent", False),
    "A4": AblationConfig("A4", "typed", True, True, True, False, "latent", False),
    "A5": AblationConfig("A5", "typed", True, True, True, True, "both", False),
    "A6": AblationConfig("A6", "typed", True, True, True, True, "both", True),
}


@dataclass
class AblationBundle:
    config: AblationConfig
    manifest: object
    transform: object
    vae: object
    engine: object
    generator: object | None


def build_ablation(
    config: AblationConfig | str,
    adapter: DatasetAdapter,
    *,
    latent_dim: int = 16,
    encoder_input_transform: str = "none",
    layer1_fit_x_raw: np.ndarray | None = None,
    mutable_mask=None,
    layer2_path: str | Path | None = None,
    active_layers=None,
) -> AblationBundle:
    """Assemble the components for one ablation. Train-only inputs stay train-only.

    ``layer1_fit_x_raw`` (TRAIN raw features) is REQUIRED when ``use_layer1`` so the
    robust-tail parameters are never fit on val/test.
    """
    import torch

    from attack.residual_head import ResidualAttackGenerator, ResidualHead

    cfg = PRESETS[config] if isinstance(config, str) else config
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()

    if cfg.decoder_kind == "legacy":
        from vae.schema import get_partition

        vae = MixedInputBetaVAE(
            partition=get_partition(),
            latent_dim=latent_dim,
            use_structured_continuous_decoder=True,
            structured_std_floor=0.01,
            encoder_input_transform=encoder_input_transform,
        )
    else:
        vae = MixedInputBetaVAE(
            manifest=manifest,
            latent_dim=latent_dim,
            encoder_input_transform=encoder_input_transform,
        )
    vae.register_feature_transform(transform)

    # Which layers this run uses. Explicit ``active_layers`` overrides the preset
    # flags, letting you request any combo ("0", "01", "012", {0, 2}, ...).
    from constraints.engine import parse_layers

    if active_layers is None:
        want = ({0} if cfg.use_layer0 else set()) \
            | ({1} if cfg.use_layer1 else set()) \
            | ({2} if cfg.use_layer2 else set())
    else:
        want = parse_layers(active_layers, {0, 1, 2})

    layer1 = []
    if 1 in want:
        if layer1_fit_x_raw is None:
            raise ValueError("layer 1 requested but layer1_fit_x_raw (TRAIN) not provided")
        layer1 = [
            RobustTailBound.fit(
                manifest, layer1_fit_x_raw, tau=None, coverage=0.99
            )
        ]

    layer2 = []
    if 2 in want:
        path = layer2_path or (_REPO / "constraints" / adapter.name / "mined.json")
        layer2 = load_layer2(path, manifest)

    engine = ConstraintEngine(
        manifest, layer0=Layer0Projector(manifest), layer1=layer1, layer2=layer2,
        active_layers=want,
    )

    generator = None
    if cfg.use_residual_head:
        if mutable_mask is None:
            mutable_mask = torch.ones(manifest.n_features, dtype=torch.bool)
        head = ResidualHead(latent_dim, manifest.n_features) if cfg.residual_mode != "latent" else None
        generator = ResidualAttackGenerator(
            vae, engine.layer0, mutable_mask, head=head, mode=cfg.residual_mode,
            apply_layer0=(0 in want),
        )

    return AblationBundle(cfg, manifest, transform, vae, engine, generator)
