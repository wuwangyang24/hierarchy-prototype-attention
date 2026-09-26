"""Multi-granularity prototype-assignment consistency.

Every view of an image is assigned softly to the internal nodes of the latent
hierarchy, one distribution per level (local -> broad). The objective asks the
views to agree at *every* granularity: the student's assignment of one view must
match the momentum teacher's assignment of another.

Information-leakage contract
----------------------------
No label of any kind enters this loss. Coarse labels only shaped the prototypes
upstream (:mod:`Hierarchy.manager`); here the candidate set is the full set of
prototypes at a level, so the normalizer carries no class information and train
and inference use the exact same assignment rule.
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import sinkhorn_normalize


def hpa_consistency_loss(
    student_views: Tensor,
    teacher_views: Tensor,
    level_prototypes: Sequence[Tensor],
    student_tau: float = 0.1,
    target_tau: float = 0.04,
    sinkhorn_iters: int = 3,
    level_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Tensor]:
    """Cross-view assignment consistency over all hierarchy levels.

    Args:
        student_views: ``(B, V, D)`` L2-normalized embeddings of the online
            encoder (HPA-refined), carrying gradients.
        teacher_views: ``(B, V, D)`` L2-normalized embeddings of the momentum
            encoder; used as targets and always detached.
        level_prototypes: one ``(M_l, D)`` tensor per level, the distinct nodes
            selected at that level across *all* coarse classes.
        student_tau: temperature of the student assignment.
        target_tau: temperature of the teacher assignment; smaller than
            ``student_tau`` so the target is the sharper of the two.
        sinkhorn_iters: Sinkhorn-Knopp iterations balancing the targets over the
            batch. Without them every image may drift onto the same node.
        level_weights: per-level weight, defaults to uniform.

    Returns the loss plus per-level agreement and uniformity diagnostics.
    """
    if student_views.ndim != 3 or teacher_views.shape != student_views.shape:
        raise ValueError("student_views and teacher_views must both be (B, V, D)")
    if not level_prototypes:
        raise ValueError("level_prototypes must hold at least one level")

    num_levels = len(level_prototypes)
    if level_weights is None:
        weights = [1.0 / num_levels] * num_levels
    else:
        if len(level_weights) != num_levels:
            raise ValueError("level_weights must have one entry per level")
        total = float(sum(level_weights))
        if total <= 0:
            raise ValueError("level_weights must sum to a positive value")
        weights = [float(w) / total for w in level_weights]

    b, v, _ = student_views.shape
    teacher_views = teacher_views.detach()
    # Single-view batches (validation) compare the student and the teacher on
    # the same view; the loss then measures student-teacher agreement only.
    pairs = [(i, j) for i in range(v) for j in range(v) if i != j] or [(0, 0)]

    loss = student_views.new_zeros(())
    metrics: Dict[str, Tensor] = {}
    for level, (protos, weight) in enumerate(zip(level_prototypes, weights)):
        protos = protos.detach().to(student_views.dtype)
        student_logits = student_views @ protos.t() / student_tau     # (B, V, M)
        target = _balanced_targets(teacher_views, protos, target_tau,
                                   sinkhorn_iters)                    # (B, V, M)

        log_p = student_logits.log_softmax(dim=-1)
        level_loss = student_views.new_zeros(())
        for src, dst in pairs:
            level_loss = level_loss - (target[:, dst] * log_p[:, src]).sum(dim=-1).mean()
        level_loss = level_loss / len(pairs)
        loss = loss + weight * level_loss

        with torch.no_grad():
            metrics[f"hpa_level{level}_loss"] = level_loss.detach()
            metrics[f"hpa_level{level}_agree"] = _agreement(student_logits, target)
            metrics[f"hpa_level{level}_uniformity"] = _uniformity(log_p)

    metrics["loss"] = loss
    metrics["hpa_assign_agree"] = torch.stack(
        [metrics[f"hpa_level{l}_agree"] for l in range(num_levels)]).mean()
    metrics["hpa_assign_uniformity"] = torch.stack(
        [metrics[f"hpa_level{l}_uniformity"] for l in range(num_levels)]).mean()
    return metrics


@torch.no_grad()
def _balanced_targets(teacher_views: Tensor, protos: Tensor, tau: float,
                      sinkhorn_iters: int) -> Tensor:
    """Teacher assignments, equipartitioned over the batch."""
    b, v, _ = teacher_views.shape
    scores = (teacher_views @ protos.t() / tau).flatten(0, 1)  # (B*V, M)
    q = scores.softmax(dim=-1)
    if sinkhorn_iters > 0:
        q = sinkhorn_normalize(q.float(), n_iters=sinkhorn_iters)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return q.to(teacher_views.dtype).view(b, v, -1)


@torch.no_grad()
def _agreement(student_logits: Tensor, target: Tensor) -> Tensor:
    """Fraction of views whose top-1 node matches the teacher's."""
    return (student_logits.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()


@torch.no_grad()
def _uniformity(log_p: Tensor) -> Tensor:
    """Entropy of the batch-mean assignment, in [0, 1]; 0 means collapsed."""
    mean_p = log_p.exp().flatten(0, 1).mean(dim=0)
    entropy = -(mean_p * mean_p.clamp_min(1e-12).log()).sum()
    return entropy / torch.log(torch.tensor(float(mean_p.numel()),
                                            device=mean_p.device))
