"""Hierarchical Prototype Attention (HPA).

Each image embedding attends over the prototypes of its selected ancestor nodes
in the latent hierarchy (local -> intermediate -> broad) and is refined by a
gated residual update:

    q_i = W_Q z_i,  k_il = W_K p_il,  v_il = W_V p_il
    a_il = softmax_l(q_i . k_il / sqrt(d))
    c_i  = sum_l a_il v_il
    z~_i = normalize(z_i + gamma * c_i)

The prototypes arrive detached: they are dataset-level statistics rebuilt
periodically, so gradients flow into the encoder and into W_Q/W_K/W_V/gamma but
never back into the prototype tensors.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class HierarchicalPrototypeAttention(nn.Module):
    """Image-to-ancestor-prototype attention with a gated residual.

    Args:
        dim: embedding dimension ``D`` of both images and prototypes.
        num_heads: attention heads; ``dim`` must be divisible by it.
        learn_gamma: learn the residual scale as a scalar parameter instead of
            using the fixed ``gamma`` value.
        gamma: residual scale. Used directly when ``learn_gamma`` is False, and
            as the initializer of the learned scalar otherwise (default 0.0, so
            HPA starts as an exact no-op and cannot overwhelm the pretrained
            representation).
        layernorm: LayerNorm the prototype context before the residual. Applied
            to the context rather than to ``z~`` so that ``gamma = 0`` still
            reproduces the baseline embedding bit-for-bit.
        out_proj: add an output projection on the attention context.
        renormalize: L2-normalize the refined embedding, matching the
            normalized space the prototypes and the classifier live in.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 1,
                 learn_gamma: bool = True,
                 gamma: float = 0.0,
                 layernorm: bool = True,
                 out_proj: bool = True,
                 renormalize: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f"HPA dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.renormalize = renormalize

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False) if out_proj else None
        self.norm = nn.LayerNorm(dim) if layernorm else None

        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(self, z: Tensor, prototypes: Tensor
                ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Refine embeddings ``z`` (B, D) with ancestor prototypes (B, L, D).

        Returns the refined embeddings and attention diagnostics.
        """
        if prototypes.ndim != 3:
            raise ValueError("prototypes must be a (B, L, D) tensor")
        if prototypes.size(0) != z.size(0) or prototypes.size(-1) != z.size(-1):
            raise ValueError(
                f"prototype shape {tuple(prototypes.shape)} is incompatible with "
                f"embeddings of shape {tuple(z.shape)}")

        b, num_levels, _ = prototypes.shape
        # Dataset-level statistics: never part of this graph.
        prototypes = prototypes.detach().to(z.dtype)

        q = self.q_proj(z).view(b, self.num_heads, 1, self.head_dim)
        k = self.k_proj(prototypes).view(b, num_levels, self.num_heads, self.head_dim)
        v = self.v_proj(prototypes).view(b, num_levels, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (B, H, L, head_dim)
        v = v.transpose(1, 2)

        logits = (q * k).sum(dim=-1) * self.scale      # (B, H, L)
        attn = logits.softmax(dim=-1)

        context = (attn.unsqueeze(-1) * v).sum(dim=2)  # (B, H, head_dim)
        context = context.reshape(b, self.dim)
        if self.out_proj is not None:
            context = self.out_proj(context)
        if self.norm is not None:
            context = self.norm(context)

        refined = z + self.gamma * context
        if self.renormalize:
            refined = F.normalize(refined, dim=1)

        return refined, self._diagnostics(attn)

    @torch.no_grad()
    def _diagnostics(self, attn: Tensor) -> Dict[str, Tensor]:
        level_attn = attn.mean(dim=1)  # average over heads -> (B, L)
        entropy = -(level_attn.clamp_min(1e-12).log() * level_attn).sum(dim=1)
        return {
            "attn_per_level": level_attn.mean(dim=0).detach(),
            "attn_entropy": entropy.mean().detach(),
            "gamma": self.gamma.detach().reshape(()),
        }
