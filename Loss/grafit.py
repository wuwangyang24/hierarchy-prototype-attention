"""Grafit loss (Touvron et al., 2020).

*Grafit: Learning fine-grained image representations with coarse labels*
sums a coarse-label kNN (NCA) loss and an instance-level loss (Eq. 4)::

    L_tot(x) = L_knn(g(x), y) + lam * L_inst(x)

The paper uses ``lam = 1`` (Appendix B.1: "we just sum up the two losses").

``L_knn`` is the NCA loss of Wu et al. (2018): with
``p_ij ~ exp(cos(g_i, m_j) / sigma)`` normalized over all ``j != i``,
``L_knn(x_i, y_i) = -log sum_{y_j = y_i, j != i} p_ij`` and ``sigma = 0.05``.
The candidates ``m_j`` come from a memory bank holding one embedding per
training image, refreshed as ``m_i <- 1/2 (m_i + g(x_i))``
(:class:`GrafitMemoryBank`). Without a bank the batch plays that role, so
batches must be large enough to contain same-class candidates.

``L_inst`` keeps the fine-grained information the coarse labels cannot express.
It is BYOL-like (Eq. 1): a predictor ``q`` on the online branch regresses the
embeddings of an EMA target network over ``T`` views of the same image, with no
negatives::

    L_inst(x) = - sum_{i != j} cos(q(g(t_i(x))), g_xi(t_j(x))) / (T (T - 1))
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GrafitMemoryBank(nn.Module):
    """One embedding per training image, refreshed as ``m_i <- 1/2 (m_i + g_i)``.

    Slots are addressed by dataset index, so the batches must carry it. Entries
    are L2-normalized; never-written slots are excluded from the loss.
    """

    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.register_buffer("bank", torch.zeros(size, dim))
        self.register_buffer("bank_labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(size, dtype=torch.bool))

    @torch.no_grad()
    def update(self, index: Tensor, embeddings: Tensor, labels: Tensor) -> None:
        index = index.view(-1).to(self.bank.device)
        embeddings = embeddings.detach().to(self.bank.dtype).to(self.bank.device)
        updated = 0.5 * (self.bank[index] + embeddings)
        fresh = ~self.filled[index]
        updated[fresh] = embeddings[fresh]
        self.bank[index] = F.normalize(updated, dim=1)
        self.bank_labels[index] = labels.view(-1).to(self.bank_labels.device)
        self.filled[index] = True


def _softmax_term(logits: Tensor, pos_mask: Tensor,
                  denom_mask: Tensor) -> tuple:
    """``-log(sum_pos exp / sum_all exp)`` averaged over anchors with positives."""
    exp_logits = torch.exp(logits)
    pos_sum = (exp_logits * pos_mask).sum(dim=1)
    all_sum = (exp_logits * denom_mask).sum(dim=1)
    valid = pos_mask.sum(dim=1) > 0
    per_anchor = -torch.log((pos_sum + 1e-12) / (all_sum + 1e-12))
    if valid.any():
        loss = per_anchor[valid].mean()
    else:
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return loss, valid


@torch.no_grad()
def _knn_accuracy(logits: Tensor, labels: Tensor, cand_labels: Tensor,
                  cand_mask: Tensor) -> Dict[str, Tensor]:
    """Top-1/3/5 label agreement of each anchor with its nearest candidates."""
    masked = logits.masked_fill(cand_mask == 0, float("-inf"))
    labels = labels.view(-1, 1)
    n_cand = int(cand_mask.sum(dim=1).min().item())
    result = {}
    for k, suffix in ((1, "batch_knn_acc"), (3, "batch_knn_top3_acc"),
                      (5, "batch_knn_top5_acc")):
        if k > n_cand:
            result[suffix] = torch.ones((), device=logits.device)
            continue
        topk_idx = masked.topk(k, dim=1).indices
        hits = (cand_labels[topk_idx] == labels).any(dim=1)
        result[suffix] = hits.float().mean()
    return result


def byol_instance_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """BYOL-style cosine loss between every ordered pair of distinct views.

    ``- sum_{i != j} cos(q(g(t_i(x))), g_xi(t_j(x))) / (T (T - 1))`` over the
    ``(T, N, D)`` normalized predictor outputs and EMA-target embeddings.
    """
    v = predictions.size(0)
    sim = torch.einsum("ind,jnd->ijn", predictions, targets)      # (V, V, N)
    off_diag = 1.0 - torch.eye(v, device=sim.device, dtype=sim.dtype)
    return -(sim * off_diag.unsqueeze(-1)).sum(dim=(0, 1)).div(v * (v - 1)).mean()


def grafit_loss(embeddings: Tensor, labels: Tensor,
                lam: float = 1.0, temperature: float = 0.05,
                predictions: Optional[Tensor] = None,
                targets: Optional[Tensor] = None,
                bank: Optional[GrafitMemoryBank] = None,
                sample_idx: Optional[Tensor] = None,
                **kwargs) -> Dict[str, Tensor]:
    """Grafit joint kNN / instance loss on L2-normalized embeddings.

    Args:
        embeddings: ``(N, D)`` online embeddings of one view per image.
        labels: ``(N,)`` coarse integer labels.
        lam: weight of the instance term in ``L_knn + lam * L_inst``.
        temperature: NCA temperature ``sigma`` (0.05 in the paper).
        predictions: ``(V, N, D)`` normalized predictor outputs ``q(g(t_v(x)))``.
        targets: ``(V, N, D)`` normalized EMA-target embeddings ``g_xi(t_v(x))``.
            Together with ``predictions`` (and ``V > 1``) these give ``L_inst``;
            when either is missing the instance term is zero.
        bank: optional :class:`GrafitMemoryBank` supplying the kNN candidates.
        sample_idx: ``(N,)`` dataset indices of the batch, required to use
            ``bank`` (the bank is refreshed with ``embeddings`` first, and each
            anchor's own slot is excluded).

    Returns a dict with the scalar ``loss`` and monitoring metrics.
    """
    lam = kwargs.get("lam", lam)
    temperature = kwargs.get("temperature", temperature)

    device = embeddings.device
    n = embeddings.size(0)
    labels = labels.view(-1)

    if bank is not None and sample_idx is not None:
        bank.update(sample_idx, embeddings, labels)
        cand = bank.bank.to(device=device, dtype=embeddings.dtype)
        cand_labels = bank.bank_labels.to(device)
        denom_mask = bank.filled.to(device).to(embeddings.dtype).expand(n, -1).clone()
        denom_mask.scatter_(1, sample_idx.view(-1, 1).to(device), 0.0)
        logits = embeddings @ cand.t() / temperature
    else:
        cand_labels = labels
        denom_mask = 1.0 - torch.eye(n, device=device, dtype=embeddings.dtype)
        logits = embeddings @ embeddings.t() / temperature

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    pos_mask = torch.eq(labels.view(-1, 1), cand_labels.view(1, -1)).to(
        embeddings.dtype) * denom_mask

    knn_loss, knn_valid = _softmax_term(logits, pos_mask, denom_mask)

    if predictions is not None and targets is not None and predictions.size(0) > 1:
        inst_loss = byol_instance_loss(predictions, targets)
    else:
        inst_loss = torch.zeros((), device=device, dtype=embeddings.dtype)

    loss = knn_loss + lam * inst_loss

    with torch.no_grad():
        metrics = {
            "Grafit": loss.detach(),
            "grafit_instance": inst_loss.detach(),
            "grafit_knn": knn_loss.detach(),
            "pos_fraction": knn_valid.float().mean(),
            **_knn_accuracy(logits, labels, cand_labels, denom_mask),
        }
        if bank is not None:
            metrics["grafit_bank_fill"] = bank.filled.float().mean()
    return {"loss": loss, **metrics}
