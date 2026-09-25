"""Storage and fast lookup of internal-node prototypes."""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .manager import HierarchySnapshot


class PrototypeBank:
    """Holds the prototypes of the currently selected hierarchy nodes.

    Prototypes are dataset-level statistics recomputed at every hierarchy
    refresh, so they are stored **detached** and never receive gradients.

    Two lookup modes are provided:

    * :meth:`lookup` — training. The sample's own ancestor prototypes, gathered
      by dataset index in O(1).
    * :meth:`lookup_nearest` — inference. No hierarchy membership is known for
      held-out images, so each level contributes its nearest prototype by cosine
      similarity. This uses no labels of any kind.
    """

    def __init__(self) -> None:
        self.prototypes: Optional[Tensor] = None       # (K, D)
        self.level_index: Optional[Tensor] = None      # (N, L)
        self.slot_prototypes: list = []                # L tensors of (M_l, D)
        self.num_levels: int = 0

    @property
    def is_ready(self) -> bool:
        return self.prototypes is not None and self.prototypes.numel() > 0

    def load(self, snapshot: HierarchySnapshot, device: torch.device) -> None:
        """Move a freshly built snapshot onto ``device`` as detached tensors."""
        self.prototypes = torch.as_tensor(
            snapshot.prototypes, dtype=torch.float32, device=device).detach()
        self.level_index = torch.as_tensor(
            snapshot.level_proto_index, dtype=torch.long, device=device)
        self.slot_prototypes = [
            self.prototypes.index_select(
                0, torch.as_tensor(rows, dtype=torch.long, device=device))
            for rows in snapshot.slot_rows
        ]
        self.num_levels = snapshot.num_levels

    def reset(self) -> None:
        self.prototypes = None
        self.level_index = None
        self.slot_prototypes = []
        self.num_levels = 0

    def lookup(self, sample_idx: Tensor) -> Tensor:
        """Ancestor prototypes ``(B, L, D)`` of the given training samples."""
        if not self.is_ready:
            raise RuntimeError("PrototypeBank has no prototypes loaded.")
        rows = self.level_index.index_select(
            0, sample_idx.reshape(-1).to(self.level_index.device))
        return self.prototypes[rows].detach()

    def lookup_nearest(self, embeddings: Tensor) -> Tensor:
        """Nearest prototype per level ``(B, L, D)`` for unseen images."""
        if not self.is_ready:
            raise RuntimeError("PrototypeBank has no prototypes loaded.")
        z = F.normalize(embeddings.detach().float(), dim=1)
        per_level = []
        for slot in self.slot_prototypes:
            nearest = (z @ slot.t()).argmax(dim=1)
            per_level.append(slot.index_select(0, nearest))
        return torch.stack(per_level, dim=1).to(embeddings.dtype).detach()
