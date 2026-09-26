from typing import Any, Callable, Dict, Optional, Tuple

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from Models import Backbone
from Hierarchy import HierarchyManager, PrototypeBank, progress_bar
from Loss import (
    BuCSFRDendrogram, GrafitMemoryBank, MaskConQueue, multiview_similarity,
)


class ContrastiveExperiment(pl.LightningModule):
    """LightningModule wrapping the backbone model for supervised contrastive
    (SupCon) training on synthesis-program labels.

    Args:
        model: the :class:`Backbone` model to train.
        lr: learning rate for the AdamW optimizer.
        weight_decay: L2 weight decay for the optimizer.
        temperature: softmax temperature for the contrastive loss.
        scheduler_gamma: multiplicative LR decay per epoch (None to disable).
    """

    def __init__(self,
                 model: Backbone,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 temperature: float = 0.1,
                 scheduler_gamma: float = 0.95,
                 scheduler: str = "exponential",
                 warmup_epochs: int = 0,
                 max_epochs: int = 100,
                 supcon_softpos: bool = False,
                 supcon_inst: bool = False,
                 supcon_inst_weight: float = 1.0,
                 taxocon_aug: bool = False,
                 vanilla_supcon: bool = False,
                 ms_loss: bool = False,
                 ms_thresh: float = 0.5,
                 ms_margin: float = 0.1,
                 ms_scale_pos: float = 2.0,
                 ms_scale_neg: float = 40.0,
                 grafit: bool = False,
                 grafit_lam: float = 1.0,
                 grafit_bank_size: int = 0,
                 maskcon: bool = False,
                 maskcon_w: float = 1.0,
                 maskcon_soft_temperature: float = 0.1,
                 maskcon_queue_size: int = 4096,
                 bucsfr: bool = False,
                 bucsfr_alpha: float = 0.5,
                 bucsfr_queue_size: int = 4096,
                 bucsfr_clusters_per_class: int = 20,
                 bucsfr_threshold: float = 1.1,
                 bucsfr_warmup_epochs: int = 10,
                 bucsfr_refresh_every: int = 1,
                 cross_entropy: bool = False,
                 supcon_soft_pos_tau: float = 0.1,
                 denom_pos_weight: bool = False,
                 tau_annealing: bool = False,
                 supcon_tau_start: float = 0.1,
                 supcon_tau_end: float = 0.1,
                 no_pos_weight_epoch: int = 0,
                 sinkhorn: bool = False,
                 sinkhorn_iters: int = 5,
                 EMA_pos_weight: bool = False,
                 EMA_momentum: float = 0.999,
                 use_hpa: bool = False,
                 hpa_levels: int = 3,
                 hierarchy_warmup_epochs: int = 5,
                 hierarchy_update_interval: int = 5,
                 hierarchy_metric: str = "cosine",
                 hierarchy_linkage: str = "average",
                 hierarchy_snapshot_dir: Optional[str] = None,
                 hierarchy_max_samples_per_class: Optional[int] = None,
                 hierarchy_seed: int = 0,
                 train_cat: str = "train",
                 test_cats: Optional[list] = None) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.supcon_softpos = supcon_softpos
        self.supcon_inst = supcon_inst
        self.supcon_inst_weight = supcon_inst_weight
        self.taxocon_aug = taxocon_aug
        self.vanilla_supcon = vanilla_supcon
        self.ms_loss = ms_loss
        self.ms_thresh = ms_thresh
        self.ms_margin = ms_margin
        self.ms_scale_pos = ms_scale_pos
        self.ms_scale_neg = ms_scale_neg
        self.grafit = grafit
        self.grafit_lam = grafit_lam
        # One memory-bank slot per training image, addressed by dataset index.
        self.grafit_bank = (
            GrafitMemoryBank(grafit_bank_size,
                             getattr(model, "output_dim", model.embedding_dim))
            if grafit and grafit_bank_size > 0 else None
        )
        self.maskcon = maskcon
        self.maskcon_w = maskcon_w
        self.maskcon_soft_temperature = maskcon_soft_temperature
        # MoCo-style FIFO queue of momentum keys feeding the masked soft labels.
        self.maskcon_queue = (
            MaskConQueue(maskcon_queue_size,
                         getattr(model, "output_dim", model.embedding_dim))
            if maskcon and maskcon_queue_size > 0 else None
        )
        self.bucsfr = bucsfr
        self.bucsfr_alpha = bucsfr_alpha
        self.bucsfr_warmup_epochs = bucsfr_warmup_epochs
        self.bucsfr_refresh_every = max(bucsfr_refresh_every, 1)
        # Same MoCo-style key queue as MaskCon; here it holds the candidates
        # instance selection samples its positives and negatives from.
        self.bucsfr_queue = (
            MaskConQueue(bucsfr_queue_size,
                         getattr(model, "output_dim", model.embedding_dim))
            if bucsfr and bucsfr_queue_size > 0 else None
        )
        self.bucsfr_dendrogram = BuCSFRDendrogram(
            clusters_per_class=bucsfr_clusters_per_class,
            threshold=bucsfr_threshold,
        ) if bucsfr else None
        # Latest dendrogram: {"im2cluster", "centroids", "density"}.
        self._bucsfr_clusters: Optional[Dict[str, torch.Tensor]] = None
        self.cross_entropy = cross_entropy
        self.supcon_soft_pos_tau = supcon_soft_pos_tau
        self.denom_pos_weight = denom_pos_weight
        self.tau_annealing = tau_annealing
        self.supcon_tau_start = supcon_tau_start
        self.supcon_tau_end = supcon_tau_end
        if no_pos_weight_epoch < 0:
            raise ValueError("no_pos_weight_epoch must be non-negative")
        self.no_pos_weight_epoch = no_pos_weight_epoch
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.train_cat = train_cat
        self.test_cats = list(test_cats) if test_cats else ["test"]

        # CE + Hierarchical Prototype Attention. The hierarchy is latent and
        # rebuilt from the evolving feature space every few epochs.
        if use_hpa and not cross_entropy:
            raise ValueError("use_hpa requires cross_entropy=True (CE is the only "
                             "objective in this version).")
        if hierarchy_warmup_epochs < 0:
            raise ValueError("hierarchy_warmup_epochs must be non-negative")
        self.use_hpa = use_hpa
        self.hierarchy_warmup_epochs = hierarchy_warmup_epochs
        self.hierarchy_update_interval = max(hierarchy_update_interval, 1)
        self.hierarchy_snapshot_dir = hierarchy_snapshot_dir
        self.hierarchy_manager = HierarchyManager(
            num_levels=hpa_levels,
            metric=hierarchy_metric,
            linkage_method=hierarchy_linkage,
            max_samples_per_class=hierarchy_max_samples_per_class,
            seed=hierarchy_seed,
        ) if use_hpa else None
        self.prototype_bank = PrototypeBank()
        self._hpa_diagnostics: Optional[Dict[str, torch.Tensor]] = None

        # Optional EMA "teacher": a momentum-updated copy of the model whose
        # embeddings drive the soft-positive weights (decoupling the positive
        # weighting from the noisy online embeddings).
        if not 0.0 <= EMA_momentum < 1.0:
            raise ValueError("EMA_momentum must be in [0, 1)")
        self.EMA_pos_weight = EMA_pos_weight
        self.EMA_momentum = EMA_momentum
        # Grafit's instance term needs the EMA target branch f_xi of Eq. 1;
        # MaskCon needs the same branch as its momentum key encoder.
        if EMA_pos_weight or grafit or supcon_inst or maskcon or bucsfr:
            self.ema_model = copy.deepcopy(model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

        self.save_hyperparameters(ignore=["model"])

    def train(self, mode: bool = True):
        # Keep the EMA teacher in eval mode regardless of the module's mode.
        super().train(mode)
        if self.ema_model is not None:
            self.ema_model.eval()
        return self

    @torch.no_grad()
    def _update_ema(self) -> None:
        m = self.EMA_momentum
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(m).add_(p.detach(), alpha=1.0 - m)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def _ema_pos_weight_sim(self, images: torch.Tensor,
                            multiview: bool = False) -> Optional[torch.Tensor]:
        """Cosine-similarity matrix from the EMA teacher's embeddings, used to
        drive the soft-positive weights. Returns None when EMA weighting is off
        or the current epoch still uses uniform positive weights. With
        ``multiview`` the similarity is averaged over all view pairs instead of
        being taken from view 0 alone."""
        if not self.EMA_pos_weight or not self._use_pos_weighting():
            return None
        if images.ndim == 5:
            if multiview:
                b, v = images.shape[:2]
                ema_views = self.ema_model(images.flatten(0, 1)).view(b, v, -1)
                return multiview_similarity(ema_views)
            images = images[:, 0]
        ema_emb = self.ema_model(images)  # normalized embeddings
        return ema_emb @ ema_emb.t()

    def _byol_views(self, images: torch.Tensor):
        """Encode a (B, V, C, H, W) batch into the online embedding of view 0
        plus the predictor / EMA-target view stacks driving the instance term.
        Single-view (B, C, H, W) batches get no instance term."""
        if images.ndim != 5:
            return self.model(images), None, None
        b, v = images.shape[:2]
        flat = images.flatten(0, 1)
        online = self.model(flat)
        predictions = self.model.grafit_predict(online).view(b, v, -1).transpose(0, 1)
        with torch.no_grad():
            targets = self.ema_model(flat).view(b, v, -1).transpose(0, 1)
        # The supervised term uses a single augmentation (Grafit appendix B.1).
        return online.view(b, v, -1)[:, 0], predictions, targets

    def _maskcon_views(self, images: torch.Tensor):
        """Encode a batch into MaskCon's query / momentum-key pair: the online
        embedding of view 0 and the EMA encoder's embedding of view 1. Single-view
        batches reuse view 0 for the key, so the explicit positive collapses onto
        the query itself (only meaningful at validation time)."""
        if images.ndim != 5:
            q = self.model(images)
            with torch.no_grad():
                k = self.ema_model(images)
            return q, k
        q = self.model(images[:, 0])
        with torch.no_grad():
            k = self.ema_model(images[:, 1 if images.size(1) > 1 else 0])
        return q, k

    def _bucsfr_cluster_labels(self, sample_idx: Optional[torch.Tensor]):
        """Dendrogram cluster id of each sample in the batch, or None before the
        first dendrogram (warmup) / at validation time (no dataset index)."""
        if self._bucsfr_clusters is None or sample_idx is None:
            return None
        im2cluster = self._bucsfr_clusters["im2cluster"]
        return im2cluster[sample_idx.view(-1).to(im2cluster.device)]

    @torch.no_grad()
    def _refresh_bucsfr_dendrogram(self) -> None:
        """Re-encode the training set with the momentum encoder and rebuild the
        per-coarse-class dendrogram (one merge per class per call)."""
        from torch.utils.data import DataLoader

        dataset = self.trainer.datamodule.train_dataset
        loader = DataLoader(
            dataset,
            batch_size=self.trainer.datamodule.batch_size,
            shuffle=False,
            num_workers=self.trainer.datamodule.num_workers,
            pin_memory=True,
        )

        features = torch.zeros(len(dataset), self.model.output_dim,
                               device=self.device)
        labels = torch.zeros(len(dataset), dtype=torch.long, device=self.device)
        was_training = self.model.training
        self.ema_model.eval()
        for batch in loader:
            # (image, label, [test_labels], index) depending on the dataset.
            images, batch_labels, idx = batch[0], batch[1], batch[-1]
            images = images.to(self.device)
            if images.ndim == 5:  # multi-view batch: cluster the first view
                images = images[:, 0]
            idx = idx.to(self.device)
            features[idx] = self.ema_model(images).float()
            labels[idx] = batch_labels.view(-1).to(self.device)
        self.model.train(was_training)

        self._bucsfr_clusters = self.bucsfr_dendrogram.build(features, labels)
        n_clusters = self._bucsfr_clusters["centroids"].size(0)
        print(f"[BuCSFR] epoch {self.current_epoch}: dendrogram has "
              f"{n_clusters} clusters over {len(self.bucsfr_dendrogram.clusters_per_class)} "
              f"coarse classes", flush=True)
        self.log("train_bucsfr_num_clusters", float(n_clusters),
                 on_step=False, on_epoch=True, rank_zero_only=True)

    def on_train_epoch_start(self) -> None:
        if self._should_refresh_hierarchy():
            self._refresh_hierarchy()
        if not self.bucsfr or self.current_epoch < self.bucsfr_warmup_epochs:
            return
        if (self.current_epoch - self.bucsfr_warmup_epochs) % self.bucsfr_refresh_every:
            return
        self._refresh_bucsfr_dendrogram()

    # ── CE + Hierarchical Prototype Attention ────────────────────────────────
    # Information-leakage contract: everything below may read the image, the
    # coarse training label and the dataset index. Fine-grained / evaluation
    # labels (batch[2] of the datasets) are never touched here.

    def _should_refresh_hierarchy(self) -> bool:
        if not self.use_hpa:
            return False
        elapsed = self.current_epoch - self.hierarchy_warmup_epochs
        return elapsed >= 0 and elapsed % self.hierarchy_update_interval == 0

    def _hpa_active(self) -> bool:
        return self.use_hpa and self.prototype_bank.is_ready

    @torch.no_grad()
    def _encode_train_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base (HPA-free) normalized embeddings and coarse labels of the train set."""
        from torch.utils.data import DataLoader

        datamodule = self.trainer.datamodule
        dataset = datamodule.train_dataset
        if not getattr(dataset, "return_index", False):
            raise RuntimeError(
                "HPA needs the training dataset to yield its sample index; "
                "build the datamodule with return_index=True.")

        loader = DataLoader(
            dataset,
            batch_size=datamodule.batch_size,
            shuffle=False,
            num_workers=datamodule.num_workers,
            pin_memory=True,
        )

        features = torch.zeros(len(dataset), self.model.output_dim,
                               dtype=torch.float32, device=self.device)
        labels = torch.zeros(len(dataset), dtype=torch.long, device=self.device)
        was_training = self.model.training
        self.model.eval()
        for batch in progress_bar(
                loader,
                f"[HPA] encoding train set (epoch {self.current_epoch})"):
            # Positional access only: batch[2] holds evaluation labels and must
            # not reach the hierarchy.
            images, coarse_labels, idx = batch[0], batch[1], batch[-1]
            images = images.to(self.device, non_blocking=True)
            if images.ndim == 5:
                images = images[:, 0]
            idx = idx.to(self.device)
            features[idx] = self.model.encode(images, normalize=True).float()
            labels[idx] = coarse_labels.view(-1).to(self.device)
        self.model.train(was_training)
        return features, labels

    @torch.no_grad()
    def _refresh_hierarchy(self) -> None:
        """Rebuild the per-coarse-class dendrogram and its node prototypes."""
        features, coarse_labels = self._encode_train_set()
        snapshot = self.hierarchy_manager.build(
            embeddings=features.cpu().numpy(),
            coarse_labels=coarse_labels.cpu().numpy(),
        )
        self.prototype_bank.load(snapshot, self.device)

        stats = snapshot.stats()
        print(f"[HPA] epoch {self.current_epoch}: "
              f"{int(stats['num_nodes'])} internal nodes over "
              f"{len(snapshot.nodes_per_coarse)} coarse classes "
              f"({stats['nodes_per_coarse_mean']:.1f}/class, "
              f"mean node size {stats['node_size_mean']:.1f})", flush=True)
        self.log_dict(
            {
                "train_hierarchy_num_nodes": stats["num_nodes"],
                "train_hierarchy_nodes_per_class": stats["nodes_per_coarse_mean"],
                "train_hierarchy_node_size_mean": stats["node_size_mean"],
                "train_hierarchy_node_size_median": stats["node_size_median"],
            },
            on_step=False, on_epoch=True, rank_zero_only=True,
        )

        if self.hierarchy_snapshot_dir and self.trainer.is_global_zero:
            os.makedirs(self.hierarchy_snapshot_dir, exist_ok=True)
            snapshot.save(os.path.join(
                self.hierarchy_snapshot_dir,
                f"hierarchy_epoch{self.current_epoch:04d}.npz"))

    def _prototype_fn(self, sample_idx: Optional[torch.Tensor]
                      ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Prototype retrieval closure for one batch.

        Training samples are members of the hierarchy, so their own ancestors
        are gathered by dataset index. Held-out images have no membership, so
        each level falls back to its nearest prototype (no labels involved).
        """
        if self.training and sample_idx is not None:
            idx = sample_idx.reshape(-1)
            return lambda _z: self.prototype_bank.lookup(idx)
        return self.prototype_bank.lookup_nearest

    def _encode_for_ce(self, images: torch.Tensor,
                       sample_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Embedding fed to the CE classifier: z (baseline) or z~ (CE + HPA)."""
        if images.ndim == 5:
            images = images[:, 0]
        if not self._hpa_active():
            self._hpa_diagnostics = None
            return self.model(images)
        embeddings, self._hpa_diagnostics = self.model.encode_with_hpa(
            images, self._prototype_fn(sample_idx))
        return embeddings

    def _log_hpa_diagnostics(self, stage: str) -> None:
        diag = self._hpa_diagnostics
        if diag is None:
            return
        metrics = {
            f"{stage}_hpa_gamma": diag["gamma"],
            f"{stage}_hpa_attn_entropy": diag["attn_entropy"],
        }
        for level, weight in enumerate(diag["attn_per_level"]):
            metrics[f"{stage}_hpa_attn_level{level}"] = weight
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      sync_dist=(stage == "val"))

    def _multi_views(self, images: torch.Tensor):
        """Encode a (B, V, C, H, W) batch into the view-0 embedding plus the
        full (B, V, D) view stack used for the augmentation-averaged positive
        similarities. Single-view batches yield a V=1 stack."""
        if images.ndim != 5:
            embeddings = self.model(images)
            return embeddings, embeddings.unsqueeze(1)
        b, v = images.shape[:2]
        view_embeddings = self.model(images.flatten(0, 1)).view(b, v, -1)
        return view_embeddings[:, 0], view_embeddings


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _current_supcon_tau(self) -> float:
        if not self.tau_annealing:
            return self.supcon_soft_pos_tau
        weighted_epoch = max(self.current_epoch - self.no_pos_weight_epoch, 0)
        weighted_epochs = max(self.max_epochs - self.no_pos_weight_epoch, 1)
        progress = min(weighted_epoch / max(weighted_epochs - 1, 1), 1.0)
        return self.supcon_tau_start + (
            self.supcon_tau_end - self.supcon_tau_start
        ) * progress

    def _use_pos_weighting(self) -> bool:
        return self.current_epoch >= self.no_pos_weight_epoch

    def _step(self, batch: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Support both (images, labels) and (images, train_labels, test_labels).
        # ``test_labels`` may carry several evaluation taxonomy levels as a
        # (B, num_test_cats) tensor; the loss only monitors the first level.
        # Grafit's memory bank additionally appends the dataset index, which
        # only the train dataset yields.
        sample_idx = None
        if (self.grafit_bank is not None or self.bucsfr or self.use_hpa) and self.training:
            *batch, sample_idx = batch
        if len(batch) == 3:
            images, labels, test_labels = batch
        else:
            images, labels = batch
            test_labels = None

        if test_labels is not None and test_labels.ndim > 1:
            loss_test_labels = test_labels[:, 0]
        else:
            loss_test_labels = test_labels

        if self.vanilla_supcon:
            embeddings = self.model(images)
            loss_dict = self.model.vanilla_supcon_loss_function(
                embeddings, labels, temperature=self.temperature)
        elif self.ms_loss:
            embeddings = self.model(images)
            loss_dict = self.model.ms_loss_function(
                embeddings, labels, thresh=self.ms_thresh,
                margin=self.ms_margin, scale_pos=self.ms_scale_pos,
                scale_neg=self.ms_scale_neg)
        elif self.grafit:
            # Multi-view train batches are (B, V, C, H, W); val stays
            # (B, C, H, W), where only the coarse kNN term is defined.
            embeddings, predictions, targets = self._byol_views(images)
            loss_dict = self.model.grafit_loss_function(
                embeddings, labels, lam=self.grafit_lam,
                temperature=self.temperature,
                predictions=predictions, targets=targets,
                bank=self.grafit_bank, sample_idx=sample_idx)
        elif self.maskcon:
            # Train batches are (B, V, C, H, W): view 0 is the query, view 1 the
            # momentum key generating the coarse-masked soft labels.
            embeddings, keys = self._maskcon_views(images)
            loss_dict = self.model.maskcon_loss_function(
                embeddings, labels, keys=keys,
                temperature=self.temperature,
                soft_temperature=self.maskcon_soft_temperature,
                w=self.maskcon_w, queue=self.maskcon_queue,
                update_queue=self.training)
        elif self.bucsfr:
            # Same query / momentum-key pair as MaskCon; the dendrogram built
            # at the start of the epoch selects the positives and negatives.
            embeddings, keys = self._maskcon_views(images)
            loss_dict = self.model.bucsfr_loss_function(
                embeddings, labels, keys=keys,
                temperature=self.temperature,
                alpha=self.bucsfr_alpha,
                cluster_labels=self._bucsfr_cluster_labels(sample_idx),
                centroids=(self._bucsfr_clusters["centroids"]
                           if self._bucsfr_clusters else None),
                density=(self._bucsfr_clusters["density"]
                         if self._bucsfr_clusters else None),
                queue=self.bucsfr_queue,
                update_queue=self.training)
        elif self.supcon_softpos:
            if self.supcon_inst:
                embeddings, predictions, targets = self._byol_views(images)
            else:
                embeddings, predictions, targets = self.model(images), None, None
            supcon_tau = self._current_supcon_tau()
            loss_dict = self.model.supcon_soft_pos_loss_function(
                embeddings, labels, temperature=self.temperature,
                pos_weight_tau=supcon_tau,
                use_pos_weighting=self._use_pos_weighting(),
                pos_weight_sim=self._ema_pos_weight_sim(images),
                predictions=predictions, targets=targets,
                inst_weight=self.supcon_inst_weight,
                test_labels=loss_test_labels)
        elif self.taxocon_aug:
            # Train batches are (B, V, C, H, W): the extra views only feed the
            # positive-weight similarities, the SupCon term stays on view 0.
            embeddings, view_embeddings = self._multi_views(images)
            loss_dict = self.model.taxocon_aug_loss_function(
                embeddings, labels, temperature=self.temperature,
                pos_weight_tau=self._current_supcon_tau(),
                use_pos_weighting=self._use_pos_weighting(),
                view_embeddings=view_embeddings,
                pos_weight_sim=self._ema_pos_weight_sim(images, multiview=True),
                test_labels=loss_test_labels)
        elif self.cross_entropy:
            # Supervised cross-entropy baseline: a linear classifier on the same
            # normalized embedding the contrastive losses operate on. With HPA
            # the classifier sees the prototype-refined embedding instead.
            embeddings = self._encode_for_ce(images, sample_idx)
            loss_dict = self.model.ce_loss_function(embeddings, labels)
        else:
            embeddings = self.model(images)
            loss_dict = self.model.loss_function(
                embeddings, labels, temperature=self.temperature)

        return loss_dict, embeddings, labels, test_labels

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict, _, _, _ = self._step(batch)
        self._log_hpa_diagnostics("train")
        if self.supcon_softpos or self.taxocon_aug:
            self.log("train_supcon_tau", self._current_supcon_tau(),
                     on_step=False, on_epoch=True)
        if self.supcon_softpos or self.taxocon_aug:
            self.log("train_pos_weight_active", float(self._use_pos_weighting()),
                     on_step=False, on_epoch=True)
        self.log(
            "train_loss", loss_dict["loss"],
            on_step=True, on_epoch=True, prog_bar=True,
        )
        for key in ("suspicion_mean", "suspicion_same_testcat", "suspicion_diff_testcat",
                    "pos_weight_same_testcat", "pos_weight_diff_testcat",
                    "pos_weight_testcat_ratio", "ce_top1", "ce_top5"):
            if key in loss_dict:
                self.log(f"train_{key}", loss_dict[key], on_step=True, on_epoch=True)
        return loss_dict["loss"]

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        # Update the EMA teacher after the optimizer step has updated the model.
        if self.ema_model is not None:
            self._update_ema()


    def on_validation_epoch_start(self) -> None:
        self._val_embeddings = []
        self._val_train_labels = []
        self._val_test_labels = []

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss_dict, embeddings, train_labels, test_labels = self._step(batch)
        self._log_hpa_diagnostics("val")
        self.log(
            "val_loss", loss_dict["loss"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        for key in ("pos_weight_same_testcat", "pos_weight_diff_testcat",
                    "pos_weight_testcat_ratio", "ce_top1", "ce_top5"):
            if key in loss_dict:
                self.log(f"val_{key}", loss_dict[key], on_step=False, on_epoch=True,
                         sync_dist=True)
        if test_labels is not None:
            self._val_embeddings.append(embeddings.detach().cpu())
            self._val_train_labels.append(train_labels.detach().view(-1).cpu())
            # Keep the (B, num_test_cats) layout so each taxonomy level stays
            # in its own column for per-level evaluation.
            tl = test_labels.detach().cpu()
            self._val_test_labels.append(tl if tl.ndim == 2 else tl.view(-1, 1))
        return loss_dict["loss"]

    def on_validation_epoch_end(self) -> None:
        if not getattr(self, "_val_embeddings", None):
            return

        val_embeddings = torch.cat(self._val_embeddings, dim=0)
        val_train_labels = torch.cat(self._val_train_labels, dim=0)
        val_test_labels = torch.cat(self._val_test_labels, dim=0)
        self._val_embeddings = []
        self._val_train_labels = []
        self._val_test_labels = []

        val_embeddings = self._gather_across_ranks(val_embeddings)
        val_train_labels = self._gather_across_ranks(val_train_labels)
        val_test_labels = self._gather_across_ranks(val_test_labels)

        device = self.device
        val_embeddings = val_embeddings.to(device)
        val_train_labels = val_train_labels.to(device)
        val_test_labels = val_test_labels.to(device)
        if val_test_labels.ndim == 1:
            val_test_labels = val_test_labels.unsqueeze(1)

        # Recall@k + linear probe on the train_cat labels.
        self._eval_and_log(val_embeddings, val_train_labels,
                           f"val_train_{self.train_cat}", prog_bar=False)

        # Recall@k + linear probe on each test_cat taxonomy level.
        for i, name in enumerate(self.test_cats):
            self._eval_and_log(
                val_embeddings, val_test_labels[:, i],
                f"val_test_{name}", prog_bar=(i == 0),
            )

        # Cophenetic correlation between embeddings and the taxonomy tree.
        # Needs at least two ranks to define a non-trivial hierarchy.
        if val_test_labels.size(1) >= 2:
            coph = self._cophenetic_correlation(val_embeddings, val_test_labels)
            if coph is not None:
                self.log_dict(
                    {
                        "val_cophenetic_spearman": coph["spearman"],
                        "val_cophenetic_pearson": coph["pearson"],
                        "val_cophenetic_cpcc": coph["cpcc"],
                        "val_cophenetic_dendro_tax": coph["dendro_tax"],
                    },
                    prog_bar=False, sync_dist=False,
                )

    def _eval_and_log(self, embeddings: torch.Tensor, labels: torch.Tensor,
                      prefix: str, prog_bar: bool = False) -> None:
        """Compute Recall@k, kNN-vote and linear-probe accuracy for one label set."""
        recall = self._full_set_recall_at_k(embeddings, labels)
        knn = self._full_set_knn_accuracy(embeddings, labels)
        probe = self._linear_probe(embeddings, labels)
        self.log_dict(
            {
                f"{prefix}_recall_at1": recall[1],
                f"{prefix}_recall_at3": recall[3],
                f"{prefix}_recall_at5": recall[5],
                f"{prefix}_recall_at10": recall[10],
                f"{prefix}_knn_top1": knn[1],
                f"{prefix}_knn_top3": knn[3],
                f"{prefix}_knn_top5": knn[5],
                f"{prefix}_knn_top10": knn[10],
                f"{prefix}_linprobe_top1": probe["top1_acc"],
                f"{prefix}_linprobe_top5": probe["top5_acc"],
            },
            prog_bar=prog_bar, sync_dist=False,
        )

    @staticmethod
    def _gather_across_ranks(tensor: torch.Tensor) -> torch.Tensor:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor
        gathered: list = [None] * world_size
        torch.distributed.all_gather_object(gathered, tensor.cpu())
        return torch.cat([t.to(tensor.device) for t in gathered], dim=0)

    @staticmethod
    @torch.no_grad()
    def _cophenetic_correlation(
        embeddings: torch.Tensor, levels: torch.Tensor,
        max_samples: int = 2048, seed: int = 42,
        metric: str = "cosine", linkage_method: str = "average",
    ) -> Optional[Dict[str, float]]:
        """Correlate pairwise embedding distances with the taxonomy tree.

        ``levels`` is an (N, R) integer tensor of taxonomy codes ordered coarse
        -> fine. The ground-truth cophenetic distance between two samples is
        ``R - depth(LCA)``, where the LCA depth counts leading ranks whose full
        ancestral prefix matches. Returns None if scipy is unavailable or there
        are too few samples/ranks to form a hierarchy.
        """
        try:
            from scipy.cluster.hierarchy import linkage, cophenet
            from scipy.spatial.distance import pdist, squareform
            from scipy.stats import spearmanr, pearsonr
        except ImportError:
            return None

        lvl = levels.detach().cpu().numpy()
        X = embeddings.detach().cpu().float().numpy()
        n, num_levels = lvl.shape
        if n < 3 or num_levels < 2:
            return None

        # Subsample for tractable O(n^2) pairwise distances.
        if n > max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=max_samples, replace=False)
            X = X[idx]
            lvl = lvl[idx]
            n = max_samples

        # Ground-truth ultrametric via cumulative-prefix LCA depth.
        lca_depth = np.zeros((n, n), dtype=np.int32)
        for r in range(num_levels):
            _, codes = np.unique(lvl[:, : r + 1], axis=0, return_inverse=True)
            lca_depth += (codes[:, None] == codes[None, :]).astype(np.int32)
        tax = (num_levels - lca_depth).astype(np.float64)
        np.fill_diagonal(tax, 0.0)
        tax_condensed = squareform(tax, checks=False)

        emb_condensed = pdist(X, metric=metric)
        if not np.isfinite(emb_condensed).all() or emb_condensed.std() == 0:
            return None

        spearman_r, _ = spearmanr(emb_condensed, tax_condensed)
        pearson_r, _ = pearsonr(emb_condensed, tax_condensed)
        Z = linkage(emb_condensed, method=linkage_method)
        cpcc, coph_dists = cophenet(Z, emb_condensed)
        dendro_tax_r, _ = spearmanr(coph_dists, tax_condensed)

        return {
            "spearman": float(spearman_r),
            "pearson": float(pearson_r),
            "cpcc": float(cpcc),
            "dendro_tax": float(dendro_tax_r),
        }

    @staticmethod
    @torch.no_grad()
    def _linear_probe(
        embeddings: torch.Tensor, labels: torch.Tensor,
        train_fraction: float = 0.8, seed: int = 42,
        lr: float = 0.1, epochs: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Train/test linear probe on the val embeddings."""
        n = embeddings.size(0)
        num_classes = int(labels.max().item()) + 1
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split = int(n * train_fraction)
        train_idx, test_idx = perm[:split], perm[split:]

        normed = F.normalize(embeddings, dim=1).float()
        train_e, train_l = normed[train_idx], labels[train_idx]
        test_e, test_l = normed[test_idx], labels[test_idx]

        dim = train_e.size(1)
        classifier = torch.nn.Linear(dim, num_classes, device=train_e.device)

        with torch.enable_grad():
            optimizer = torch.optim.LBFGS(classifier.parameters(), lr=lr, max_iter=20)
            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(classifier(train_e), train_l)
                loss.backward()
                return loss.detach()
            for _ in range(epochs):
                optimizer.step(closure)

        classifier.eval()
        logits = classifier(test_e)
        top1 = (logits.argmax(dim=1) == test_l).float().mean()
        k = min(5, num_classes)
        top5 = (logits.topk(k, dim=1).indices == test_l.unsqueeze(1)).any(dim=1).float().mean()
        return {"top1_acc": top1, "top5_acc": top5}

    @staticmethod
    @torch.no_grad()
    def _full_set_recall_at_k(
        embeddings: torch.Tensor, labels: torch.Tensor,
        ks: Tuple[int, ...] = (1, 3, 5, 10), chunk_size: int = 1024,
    ) -> Dict[int, torch.Tensor]:
        """Recall@k over the entire val set (cosine, leave-one-out).

        A sample counts as a hit if any of its k nearest neighbours across the
        full set shares its label. Computed in row chunks to bound memory.
        """
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        n = embeddings.size(0)
        max_k = min(max(ks), n - 1)
        if max_k < 1:
            return {k: torch.tensor(1.0, device=embeddings.device) for k in ks}

        hits = {k: torch.zeros(n, dtype=torch.bool, device=embeddings.device) for k in ks}
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sim = embeddings[start:end] @ embeddings.t()       # (chunk, n)
            rows = torch.arange(end - start, device=embeddings.device)
            sim[rows, torch.arange(start, end, device=embeddings.device)] = float("-inf")
            topk_idx = sim.topk(max_k, dim=1).indices           # (chunk, max_k)
            match = labels[topk_idx] == labels[start:end].unsqueeze(1)
            for k in ks:
                hits[k][start:end] = match[:, :min(k, max_k)].any(dim=1)
        return {k: hits[k].float().mean() for k in ks}

    @staticmethod
    @torch.no_grad()
    def _full_set_knn_accuracy(
        embeddings: torch.Tensor, labels: torch.Tensor,
        ks: Tuple[int, ...] = (1, 3, 5, 10), chunk_size: int = 1024,
    ) -> Dict[int, torch.Tensor]:
        """Top-k kNN classification accuracy (cosine, leave-one-out).

        Each sample is classified by a majority vote over its k nearest
        neighbours across the full set; ties are broken by summed cosine
        similarity. Computed in row chunks to bound memory.
        """
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        n = embeddings.size(0)
        max_k = min(max(ks), n - 1)
        if max_k < 1:
            return {k: torch.tensor(0.0, device=embeddings.device) for k in ks}

        num_classes = int(labels.max().item()) + 1
        correct = {k: torch.zeros(n, dtype=torch.bool, device=embeddings.device)
                   for k in ks}
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rows = end - start
            sim = embeddings[start:end] @ embeddings.t()       # (rows, n)
            sim[torch.arange(rows, device=embeddings.device),
                torch.arange(start, end, device=embeddings.device)] = float("-inf")
            topk = sim.topk(max_k, dim=1)
            nbr_labels = labels[topk.indices]                   # (rows, max_k)
            # Similarity is in [-1, 1], so the scaled tiebreak can never
            # outweigh a single vote.
            tiebreak = topk.values.clamp_min(0) * (1.0 / (2 * max_k))
            for k in ks:
                kk = min(k, max_k)
                votes = torch.zeros(rows, num_classes, device=embeddings.device)
                votes.scatter_add_(1, nbr_labels[:, :kk],
                                   torch.ones(rows, kk, device=embeddings.device))
                votes.scatter_add_(1, nbr_labels[:, :kk], tiebreak[:, :kk])
                correct[k][start:end] = votes.argmax(dim=1) == labels[start:end]
        return {k: correct[k].float().mean() for k in ks}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_gamma is None and self.scheduler_type == "exponential":
            return optimizer

        if self.scheduler_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-7
            )
        else:
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.scheduler_gamma
            )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, main_scheduler],
                milestones=[self.warmup_epochs]
            )
        else:
            scheduler = main_scheduler

        return [optimizer], [scheduler]
