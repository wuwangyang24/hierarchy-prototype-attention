from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .hpa import HierarchicalPrototypeAttention
from Loss import (
    batch_knn_accuracy, gaussianity_metrics,
    SupConSoftPosLoss, TaxoConAugLoss,
    vanilla_supcon_loss, multi_similarity_loss, grafit_loss, maskcon_loss,
    bucsfr_loss,
)

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover - timm is a declared dependency
    _HAS_TIMM = False


# Supported fully fine-tuned backbones (timm model ids).
_SUPPORTED_BACKBONES = (
    "resnet18", "resnet50", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
    "convnext_tiny",
)


class Backbone(nn.Module):
    """Fully fine-tuned convolutional / ViT backbone with an optional projection
    head for supervised contrastive (SupCon) representation learning.

    The entire backbone is trainable (full fine-tuning). ``forward`` returns
    L2-normalized embeddings suitable for a cosine-similarity contrastive
    objective, so this model plugs directly into
    :class:`ContrastiveExperiment`.

    Despite the class name, the ``backbone`` argument selects which timm model to
    fine-tune (``resnet18``, ``resnet50``, ``vit_small_patch16_224``,
    ``swin_tiny_patch4_window7_224`` or ``convnext_tiny``).

    Args:
        backbone: timm model id to fully fine-tune (see ``_SUPPORTED_BACKBONES``).
        img_size: square input size fed to the backbone (any size >= 32).
        embedding_dim: dimension of the output (projected) embedding.
        proj_hidden_dim: hidden width of the 2-layer projection MLP.
        temperature: softmax temperature for the SupCon loss.
        use_proj_head: if True, add a 2-layer MLP projection head on top of the
            backbone features; otherwise output the L2-normalized backbone
            features directly.
        use_hpa: add a Hierarchical Prototype Attention block on the final
            embedding. When False the model is the untouched CE/SupCon baseline.
        hpa_heads: attention heads of the HPA block.
        hpa_gamma: HPA residual scale (initial value when it is learned).
        hpa_learn_gamma: learn the residual scale instead of fixing it.
        hpa_layernorm: LayerNorm the HPA prototype context before the residual.
        pretrained: load ImageNet-pretrained backbone weights.
    """

    # The pipeline uses this flag to skip image reconstruction/sampling logging.
    supports_image_generation = False

    def __init__(self,
                 backbone: str = "resnet18",
                 img_size: int = 224,
                 embedding_dim: int = 256,
                 proj_hidden_dim: int = 2048,
                 temperature: float = 0.1,
                 use_proj_head: bool = True,
                 supcon_soft_pos: bool = False,
                 supcon_soft_pos_tau: float = 0.1,
                 supcon_denom_pos_weight: bool = False,
                 taxocon_aug: bool = False,
                 sinkhorn: bool = False,
                 sinkhorn_iters: int = 5,
                 grad_checkpointing: bool = False,
                 num_classes: Optional[int] = None,
                 cross_entropy: bool = False,
                 grafit_predictor: bool = False,
                 use_hpa: bool = False,
                 hpa_heads: int = 1,
                 hpa_gamma: float = 0.0,
                 hpa_learn_gamma: bool = True,
                 hpa_layernorm: bool = True,
                 pretrained: bool = True) -> None:
        super().__init__()

        if not _HAS_TIMM:
            raise ImportError(
                "timm is required for Backbone. Install it with `pip install timm`."
            )

        if backbone not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose from {list(_SUPPORTED_BACKBONES)}."
            )

        self.backbone_name = backbone
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_proj_head = use_proj_head

        # Feature-extractor backbone (num_classes=0 -> pooled features, no head).
        # The whole backbone is trainable (full fine-tuning). Transformer
        # backbones need img_size to build position embeddings for non-default
        # input sizes.
        create_kwargs = dict(pretrained=pretrained, num_classes=0)
        if backbone.startswith(("vit", "swin")):
            create_kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **create_kwargs)
        feat_dim = self.backbone.num_features

        # Trade compute for memory: recompute backbone activations in the
        # backward pass instead of storing them (allows larger batches).
        if grad_checkpointing:
            from .grad_checkpoint import enable_grad_checkpointing
            enable_grad_checkpointing(self.backbone)

        # Trainable projection head mapping backbone features -> embedding space.
        if self.use_proj_head:
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_hidden_dim),
                nn.GELU(),
                nn.Linear(proj_hidden_dim, embedding_dim),
            )
        else:
            self.projection = None

        # Width of what forward() returns: the head is optional.
        self.output_dim = embedding_dim if self.use_proj_head else feat_dim

        # Optional linear classifier for the supervised cross-entropy baseline.
        # It sits on the same (normalized) embedding the contrastive losses use.
        if cross_entropy:
            if not num_classes or num_classes < 2:
                raise ValueError("cross_entropy requires num_classes >= 2.")
            clf_in = embedding_dim if self.use_proj_head else feat_dim
            self.classifier = nn.Linear(clf_in, num_classes)
        else:
            self.classifier = None

        # BYOL-style predictor for Grafit's instance-level term: it sits on the
        # online branch only, which is what breaks the collapse symmetry.
        if grafit_predictor:
            self.grafit_predictor = nn.Sequential(
                nn.Linear(self.output_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, self.output_dim),
            )
        else:
            self.grafit_predictor = None

        # Hierarchical Prototype Attention on the final image embedding. When
        # disabled the module is absent and encode() is the untouched baseline.
        self.hpa = HierarchicalPrototypeAttention(
            dim=self.output_dim,
            num_heads=hpa_heads,
            learn_gamma=hpa_learn_gamma,
            gamma=hpa_gamma,
            layernorm=hpa_layernorm,
        ) if use_hpa else None

        self.supcon_soft_pos_loss = SupConSoftPosLoss(
            pos_weight_tau=supcon_soft_pos_tau,
            sinkhorn=sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            denom_pos_weight=supcon_denom_pos_weight,
        ) if supcon_soft_pos else None

        self.taxocon_aug_loss = TaxoConAugLoss(
            pos_weight_tau=supcon_soft_pos_tau,
            sinkhorn=sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            denom_pos_weight=supcon_denom_pos_weight,
        ) if taxocon_aug else None

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable (backbone + projection head) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, x: Tensor, normalize: bool = True) -> Tensor:
        """Return embeddings for a batch of images [N, 3, H, W].

        By default the embeddings are L2-normalized (for the cosine-similarity
        contrastive objective). Pass ``normalize=False`` to obtain the raw
        projected features.
        """
        feats = self.backbone(x)          # (N, feat_dim)
        if self.projection is not None:
            feats = self.projection(feats)  # (N, embedding_dim)
        return F.normalize(feats, dim=1) if normalize else feats

    def forward(self, x: Tensor, normalize: bool = True, **kwargs) -> Tensor:
        return self.encode(x, normalize=normalize)

    def encode_with_hpa(
        self, x: Tensor, prototype_fn: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Encode ``x`` and refine the embeddings with ancestor prototypes.

        ``prototype_fn`` maps the base embeddings (N, D) to their detached
        ancestor prototypes (N, L, D); it is a callable so that inference-time
        nearest-prototype retrieval can reuse this single forward pass.

        Returns ``(refined_embeddings, hpa_diagnostics)``.
        """
        if self.hpa is None:
            raise RuntimeError("Model was built without use_hpa=True.")
        z = self.encode(x, normalize=True)
        with torch.no_grad():
            prototypes = prototype_fn(z)
        return self.hpa(z, prototypes)

    def loss_function(self, embeddings: Tensor, labels: Tensor,
                      **kwargs) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return vanilla_supcon_loss(embeddings, labels, **kwargs)

    def supcon_soft_pos_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return self.supcon_soft_pos_loss(embeddings, labels, **kwargs)

    def taxocon_aug_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return self.taxocon_aug_loss(embeddings, labels, **kwargs)

    def vanilla_supcon_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return vanilla_supcon_loss(embeddings, labels, **kwargs)

    def ms_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        return multi_similarity_loss(embeddings, labels, **kwargs)

    def grafit_predict(self, embeddings: Tensor) -> Tensor:
        """Normalized predictor output q(g(x)) for Grafit's instance term."""
        if self.grafit_predictor is None:
            raise RuntimeError(
                "Model was built without grafit_predictor=True.")
        return F.normalize(self.grafit_predictor(embeddings), dim=1)

    def grafit_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return grafit_loss(embeddings, labels, **kwargs)

    def maskcon_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        return maskcon_loss(embeddings, labels, **kwargs)

    def bucsfr_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("class_logits", self.classify(embeddings))
        return bucsfr_loss(embeddings, labels, **kwargs)

    def classify(self, embeddings: Tensor) -> Optional[Tensor]:
        """Coarse-class logits, or None when the model has no classifier head."""
        return None if self.classifier is None else self.classifier(embeddings)

    def ce_loss_function(
        self, embeddings: Tensor, labels: Tensor, **kwargs,
    ) -> Dict[str, Tensor]:
        if self.classifier is None:
            raise RuntimeError(
                "ce_loss_function requires the model to be built with "
                "cross_entropy=True."
            )
        labels = labels.view(-1).long()
        logits = self.classifier(embeddings)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == labels).float().mean()
            k = min(5, logits.size(1))
            top5 = (logits.topk(k, dim=1).indices
                    == labels.unsqueeze(1)).any(dim=1).float().mean()
        return {"loss": loss, "ce_top1": top1, "ce_top5": top5}

    @staticmethod
    @torch.no_grad()
    def _gaussianity_metrics(z: Tensor) -> Dict[str, Tensor]:
        return gaussianity_metrics(z)

    @staticmethod
    @torch.no_grad()
    def _batch_knn_accuracy(logits: Tensor, labels: Tensor,
                            self_mask: Tensor) -> dict:
        return batch_knn_accuracy(logits, labels, self_mask)
