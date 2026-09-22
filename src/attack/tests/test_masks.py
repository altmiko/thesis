"""Dataset perturbation-mask contract.

Run (thesis env):
    PYTHONPATH=".;src" python -m pytest src/attack/tests/test_masks.py -q -p no:faulthandler
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from attack.masks import get_dataset_mask
from attack.masks.base import DatasetMask
from datasets import get_adapter
from datasets.feature_manifest import ManifestError


@pytest.mark.parametrize(
    "dataset,n,n_pert,n_deriv",
    [("cicids2017", 79, 9, 7), ("ciciot2023", 39, 5, 3)],
)
def test_resolve_partitions_feature_vector(dataset, n, n_pert, n_deriv):
    manifest = get_adapter(dataset).feature_manifest()
    resolved = get_dataset_mask(dataset).resolve(manifest)
    p, d, f = set(resolved.perturbable_idx), set(resolved.derived_idx), set(resolved.frozen_idx)
    assert len(p) == n_pert and len(d) == n_deriv
    assert p.isdisjoint(d) and p.isdisjoint(f) and d.isdisjoint(f)
    assert p | d | f == set(range(n))
    # perturbable mask selects exactly the perturbable columns
    assert resolved.perturbable_mask().sum().item() == n_pert


def test_resolve_fails_loud_on_wrong_expected_index():
    manifest = get_adapter("cicids2017").feature_manifest()
    base = get_dataset_mask("cicids2017")
    bad = dataclasses.replace(base, expected_perturbable_index1=(1,) + base.expected_perturbable_index1[1:])
    with pytest.raises(ManifestError):
        bad.resolve(manifest)


def test_resolve_fails_loud_on_unknown_feature():
    manifest = get_adapter("cicids2017").feature_manifest()
    base = get_dataset_mask("cicids2017")
    bad = dataclasses.replace(base, perturbable=("Not A Feature",) + base.perturbable[1:])
    with pytest.raises(ManifestError):
        bad.resolve(manifest)


def test_frozen_preserved_and_derived_recomputed():
    manifest = get_adapter("cicids2017").feature_manifest()
    resolved = get_dataset_mask("cicids2017").resolve(manifest)
    i = resolved.name_to_idx
    orig = torch.zeros(3, manifest.n_features, dtype=torch.float64)
    orig[:, i["Total Fwd Packet"]] = 10.0
    orig[:, i["Total Length of Fwd Packet"]] = 5000.0
    orig[:, i["Dst Port"]] = 443.0  # frozen
    gen = orig.clone()
    gen[:, i["Total Length of Fwd Packet"]] = 6000.0  # attacker moves a perturbable parent
    gen[:, i["Dst Port"]] = 1.0                        # a stray move on a frozen feature
    gen[:, i["Fwd Packet Length Mean"]] = 999.0        # stray move on a derived feature

    out = resolved.apply(gen, orig)
    # frozen restored exactly from the source
    assert torch.equal(out[:, i["Dst Port"]], orig[:, i["Dst Port"]])
    # derived recomputed from the perturbed parent (6000/10), not the stray 999
    assert torch.allclose(out[:, i["Fwd Packet Length Mean"]], torch.full((3,), 600.0, dtype=torch.float64))
    # perturbable parent itself is untouched by the dependency stage
    assert torch.equal(out[:, i["Total Length of Fwd Packet"]], gen[:, i["Total Length of Fwd Packet"]])
    # the sanity gates agree
    assert int(resolved.frozen_violation_mask(out, orig).sum()) == 0
    assert int(resolved.derived_consistency_mask(out).sum()) == 0


def test_classifier_gradient_flows_through_derived_to_perturbable():
    manifest = get_adapter("cicids2017").feature_manifest()
    resolved = get_dataset_mask("cicids2017").resolve(manifest)
    i = resolved.name_to_idx
    x = torch.zeros(2, manifest.n_features, dtype=torch.float64)
    x[:, i["Total Fwd Packet"]] = 8.0
    x[:, i["Total Length of Fwd Packet"]] = 4000.0
    x[:, i["Flow Duration"]] = 5000.0
    x[:, i["Total Length of Bwd Packet"]] = 1000.0
    x.requires_grad_(True)
    rec = resolved.recompute(x)
    # a "classifier" reading only DERIVED features
    (rec[:, i["Fwd Packet Length Mean"]].sum() + rec[:, i["Flow Bytes/s"]].sum()).backward()
    g = x.grad
    # gradient reaches the perturbable parents ...
    assert abs(float(g[0, i["Total Length of Fwd Packet"]])) > 0
    assert abs(float(g[0, i["Flow Duration"]])) > 0
    # ... and is exactly zero on a frozen feature no derived depends on
    assert float(g[0, i["Dst Port"]]) == 0.0
