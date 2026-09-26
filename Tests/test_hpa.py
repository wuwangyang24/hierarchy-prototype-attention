"""Sanity checks for Hierarchical Prototype Attention.

Run with ``pytest Tests/test_hpa.py``. These tests exercise the hierarchy,
the prototype bank, the attention block and the multi-granularity consistency
loss without touching timm/ImageNet weights, so they are fast and offline.
"""

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from Hierarchy import HierarchyManager, PrototypeBank
from Loss import hpa_consistency_loss
from Models.hpa import HierarchicalPrototypeAttention

DIM = 16
NUM_LEVELS = 3


def _toy_data(num_coarse: int = 3, per_coarse: int = 40, seed: int = 0):
    """Coarse classes of Gaussian blobs, each split into two hidden sub-blobs."""
    rng = np.random.default_rng(seed)
    embeddings, coarse, fine = [], [], []
    for c in range(num_coarse):
        for sub in range(2):
            center = rng.normal(size=DIM) * 3.0
            x = center + rng.normal(scale=0.2, size=(per_coarse // 2, DIM))
            embeddings.append(x)
            coarse.append(np.full(per_coarse // 2, c))
            fine.append(np.full(per_coarse // 2, c * 2 + sub))
    x = np.concatenate(embeddings).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.concatenate(coarse), np.concatenate(fine)


@pytest.fixture(scope="module")
def snapshot():
    x, coarse, _ = _toy_data()
    return HierarchyManager(num_levels=NUM_LEVELS).build(x, coarse)


# ── Hierarchy construction ───────────────────────────────────────────────────

def test_hierarchy_never_mixes_coarse_classes(snapshot):
    """Every node's samples come from exactly one coarse class."""
    for level in range(snapshot.num_levels):
        rows = snapshot.level_proto_index[:, level]
        for row in np.unique(rows):
            members = snapshot.coarse_labels[rows == row]
            assert len(np.unique(members)) == 1
    # Node bookkeeping agrees with the per-class assignment.
    assert set(snapshot.nodes_per_coarse) == set(np.unique(snapshot.coarse_labels))


def test_sample_is_a_descendant_of_each_selected_node(snapshot):
    """A selected node's descendant count covers the samples routed to it."""
    for level in range(snapshot.num_levels):
        rows = snapshot.level_proto_index[:, level]
        for row in np.unique(rows):
            assert snapshot.node_sizes[row] >= int((rows == row).sum())
            assert snapshot.node_sizes[row] >= 1


def test_selected_ancestors_are_ordered_local_to_broad(snapshot):
    sizes = snapshot.node_sizes[snapshot.level_proto_index]  # (N, L)
    assert np.all(np.diff(sizes, axis=1) >= 0), "node sizes must grow toward the root"


def test_prototypes_match_embedding_dim_and_are_unit_norm(snapshot):
    assert snapshot.prototypes.shape[1] == DIM
    norms = np.linalg.norm(snapshot.prototypes, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hierarchy_build_accepts_no_fine_labels():
    """The public API must not expose any channel for fine-grained labels."""
    params = set(inspect.signature(HierarchyManager.build).parameters)
    assert params == {"self", "embeddings", "coarse_labels", "sample_ids"}

    x, coarse, fine = _toy_data()
    with pytest.raises(TypeError):
        HierarchyManager(num_levels=NUM_LEVELS).build(x, coarse, fine_labels=fine)


def test_degenerate_coarse_classes_are_handled():
    x, coarse, _ = _toy_data()
    coarse = coarse.copy()
    coarse[:1] = 99  # a coarse class with a single sample
    snap = HierarchyManager(num_levels=NUM_LEVELS).build(x, coarse)
    assert (snap.level_proto_index >= 0).all()


# ── Prototype bank ───────────────────────────────────────────────────────────

def test_stored_prototypes_are_detached(snapshot):
    bank = PrototypeBank()
    bank.load(snapshot, torch.device("cpu"))
    assert not bank.prototypes.requires_grad
    protos = bank.lookup(torch.arange(8))
    assert not protos.requires_grad and protos.grad_fn is None
    assert protos.shape == (8, NUM_LEVELS, DIM)


def test_nearest_lookup_needs_no_labels(snapshot):
    bank = PrototypeBank()
    bank.load(snapshot, torch.device("cpu"))
    protos = bank.lookup_nearest(torch.randn(5, DIM))
    assert protos.shape == (5, NUM_LEVELS, DIM)
    assert not protos.requires_grad


# ── Attention block ──────────────────────────────────────────────────────────

def test_attention_weights_sum_to_one():
    hpa = HierarchicalPrototypeAttention(dim=DIM, num_heads=2)
    z = F.normalize(torch.randn(4, DIM), dim=1)
    protos = F.normalize(torch.randn(4, NUM_LEVELS, DIM), dim=-1)
    _, diag = hpa(z, protos)
    assert torch.allclose(diag["attn_per_level"].sum(), torch.tensor(1.0), atol=1e-5)


def test_zero_gamma_reproduces_the_baseline_embedding():
    hpa = HierarchicalPrototypeAttention(dim=DIM, gamma=0.0, learn_gamma=True)
    z = F.normalize(torch.randn(4, DIM), dim=1)
    protos = F.normalize(torch.randn(4, NUM_LEVELS, DIM), dim=-1)
    refined, _ = hpa(z, protos)
    assert torch.allclose(refined, z, atol=1e-6)


def test_gradients_reach_encoder_projections_and_gamma():
    encoder = nn.Linear(DIM, DIM)
    hpa = HierarchicalPrototypeAttention(dim=DIM, gamma=0.1)
    protos = F.normalize(torch.randn(4, NUM_LEVELS, DIM), dim=-1).requires_grad_(True)

    z = F.normalize(encoder(torch.randn(4, DIM)), dim=1)
    refined, _ = hpa(z, protos)
    refined.sum().backward()

    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    for proj in (hpa.q_proj, hpa.k_proj, hpa.v_proj, hpa.out_proj):
        assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0
    assert hpa.gamma.grad is not None and hpa.gamma.grad.abs() > 0
    # Prototypes are dataset-level statistics: no gradient may reach them.
    assert protos.grad is None


def test_hpa_disabled_leaves_the_encoder_untouched():
    """Without an HPA block the encode path is untouched and z~ == z."""
    class TinyBackbone(nn.Module):
        def __init__(self, use_hpa):
            super().__init__()
            self.trunk = nn.Linear(DIM, DIM)
            self.hpa = HierarchicalPrototypeAttention(dim=DIM) if use_hpa else None

        def encode(self, x):
            return F.normalize(self.trunk(x), dim=1)

    torch.manual_seed(0)
    baseline = TinyBackbone(use_hpa=False)
    torch.manual_seed(0)
    with_hpa = TinyBackbone(use_hpa=True)

    x = torch.randn(4, DIM)
    protos = F.normalize(torch.randn(4, NUM_LEVELS, DIM), dim=-1)
    refined, _ = with_hpa.hpa(with_hpa.encode(x), protos)
    assert torch.allclose(baseline.encode(x), refined, atol=1e-6)


# ── Multi-granularity consistency loss ───────────────────────────────────────

def _level_prototypes(snapshot):
    protos = torch.as_tensor(snapshot.prototypes)
    return [protos[torch.as_tensor(rows.copy())] for rows in snapshot.slot_rows]


def test_consistency_loss_is_lower_when_views_agree(snapshot):
    levels = _level_prototypes(snapshot)
    torch.manual_seed(0)
    z = F.normalize(torch.randn(8, DIM), dim=1)

    agree = torch.stack([z, z], dim=1)
    disagree = torch.stack([z, z.flip(0)], dim=1)
    same = hpa_consistency_loss(agree, agree, levels)["loss"]
    different = hpa_consistency_loss(disagree, disagree, levels)["loss"]
    assert same < different


def test_consistency_loss_uses_no_labels():
    params = set(inspect.signature(hpa_consistency_loss).parameters)
    assert not any("label" in p for p in params)


def test_consistency_gradients_reach_the_student_only(snapshot):
    levels = _level_prototypes(snapshot)
    student = F.normalize(torch.randn(6, 2, DIM), dim=-1).requires_grad_(True)
    teacher = F.normalize(torch.randn(6, 2, DIM), dim=-1).requires_grad_(True)

    hpa_consistency_loss(student, teacher, levels)["loss"].backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
