import argparse
import hashlib
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class BestValLossReporter(Callback):
    """Track the epoch with the lowest ``val_loss`` and, at the end of
    training, print a table of the Recall@k / linear-probe metrics recorded at
    that epoch together with the positive-weight temperature and Sinkhorn
    settings."""

    def __init__(self, pos_weight_tau, sinkhorn: bool, sinkhorn_iters: int,
                 stats_dir: str, report_name: str) -> None:
        super().__init__()
        self.pos_weight_tau = pos_weight_tau
        self.sinkhorn = sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.stats_dir = stats_dir
        self.report_name = report_name
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_metrics: dict = {}
        self.best_cophenetic: dict = {}

    def on_validation_end(self, trainer, pl_module) -> None:
        # Use on_validation_end (not on_validation_epoch_end) so that the
        # LightningModule has already logged this epoch's Recall@k / linear-probe
        # metrics; callback on_validation_epoch_end hooks run *before* the
        # module's, so callback_metrics would otherwise hold stale values.
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = int(trainer.current_epoch)
            self.best_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if ("recall_at" in k or "knn_top" in k or "linprobe_top" in k)
            }
            self.best_cophenetic = {
                k: float(v)
                for k, v in metrics.items()
                if "cophenetic" in k
            }

    def on_fit_end(self, trainer, pl_module) -> None:
        if self.best_epoch is None:
            return

        # Group the recorded metrics by their label-set prefix so each row of
        # the table corresponds to one evaluation (e.g. val_test_phylum).
        prefixes = sorted({
            k.rsplit("_", 2)[0] for k in self.best_metrics
        })

        def fmt(value):
            return f"{value:.4f}" if value is not None else "-"

        header = ["Eval", "Recall@1", "Recall@5", "kNN@1", "kNN@5",
                  "LinProbe@1", "LinProbe@5"]
        rows = []
        for prefix in prefixes:
            rows.append([
                prefix,
                fmt(self.best_metrics.get(f"{prefix}_recall_at1")),
                fmt(self.best_metrics.get(f"{prefix}_recall_at5")),
                fmt(self.best_metrics.get(f"{prefix}_knn_top1")),
                fmt(self.best_metrics.get(f"{prefix}_knn_top5")),
                fmt(self.best_metrics.get(f"{prefix}_linprobe_top1")),
                fmt(self.best_metrics.get(f"{prefix}_linprobe_top5")),
            ])

        widths = [
            max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
            for i in range(len(header))
        ]

        def render(cells):
            return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        tau_str = (
            f"{self.pos_weight_tau:.4g}"
            if isinstance(self.pos_weight_tau, (int, float))
            else str(self.pos_weight_tau)
        )
        sinkhorn_str = f"yes ({self.sinkhorn_iters} iters)" if self.sinkhorn else "no"

        lines = [
            "",
            "=" * max(60, sum(widths) + 3 * (len(widths) - 1)),
            f"Best val_loss: {self.best_val_loss:.6f} @ epoch {self.best_epoch}",
            f"Positive-weight Tau: {tau_str}",
            f"Sinkhorn: {sinkhorn_str}",
            "-" * max(60, sum(widths) + 3 * (len(widths) - 1)),
        ]
        if rows:
            lines.append(render(header))
            lines.append("-+-".join("-" * w for w in widths))
            lines.extend(render(r) for r in rows)
        else:
            lines.append("(no Recall@k / kNN / linear-probe metrics were recorded)")
        if self.best_cophenetic:
            width = max(60, sum(widths) + 3 * (len(widths) - 1))
            lines.append("-" * width)
            lines.append("Cophenetic correlation (embeddings vs. taxonomy):")
            lines.append(f"  Spearman:   {fmt(self.best_cophenetic.get('val_cophenetic_spearman'))}")
            lines.append(f"  Pearson:    {fmt(self.best_cophenetic.get('val_cophenetic_pearson'))}")
            lines.append(f"  CPCC:       {fmt(self.best_cophenetic.get('val_cophenetic_cpcc'))}")
            lines.append(f"  Dendro-Tax: {fmt(self.best_cophenetic.get('val_cophenetic_dendro_tax'))}")
        lines.append("=" * max(60, sum(widths) + 3 * (len(widths) - 1)))
        lines.append("")
        report = "\n".join(lines)
        print(report)

        # Persist the same report next to the checkpoints under a 'stats' folder.
        os.makedirs(self.stats_dir, exist_ok=True)
        with open(os.path.join(self.stats_dir, self.report_name),
                  "w", encoding="utf-8") as f:
            f.write(report)

        # Regenerate the dataset's Excel workbook from every report in this
        # folder so it fills in as Vanilla / per-τ runs complete. Optional:
        # skipped cleanly if openpyxl (or the generator) is unavailable.
        if trainer.is_global_zero:
            try:
                from generate_reports import collect_dataset, build_workbook
                report_paths = [
                    os.path.join(self.stats_dir, f)
                    for f in os.listdir(self.stats_dir)
                    if f.endswith("_best_val_loss_report.txt")
                ]
                by_kind = collect_dataset(report_paths)
                if by_kind:
                    dataset_name = os.path.basename(self.stats_dir.rstrip(os.sep))
                    model_name = os.path.basename(
                        os.path.dirname(self.stats_dir.rstrip(os.sep)))
                    out_path = os.path.join(
                        self.stats_dir, f"{model_name}_{dataset_name}_metrics.xlsx")
                    build_workbook(by_kind, out_path)
                    print(f"[report] wrote Excel workbook: {out_path}")
            except Exception as e:  # noqa: BLE001 - reporting must not break training
                print(f"[report] skipped Excel workbook generation: {e}")

from Models import Backbone
from dataset import (ContrastiveDataModule, InatDataModule,
                     FGVCAircraftDataModule)
from experiment import ContrastiveExperiment

# Use file-system based tensor sharing to avoid /dev/shm exhaustion, which
# otherwise hangs DataLoader workers in containers with a small shared-memory
# mount (e.g. Docker/SageMaker default of 64MB).
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a contrastive backbone with PyTorch Lightning + W&B")

    # Data
    parser.add_argument("--dataset", type=str, default="myzus",
                        choices=["myzus", "inat", "aircraft"],
                        help="Dataset to use: 'myzus' (default synthesis-program dataset), "
                             "'inat' (iNaturalist 2021 mini) or 'aircraft' (FGVC-Aircraft)")
    parser.add_argument("--train_cat", type=str, default="class",
                        help="Taxonomy level for contrastive training labels (inat only). "
                             "Options: kingdom, phylum, class, order, family, genus. "
                             "For --dataset aircraft: manufacturer, family or variant.")
    parser.add_argument("--test_cat", type=str, nargs="+", default=["phylum"],
                        help="Taxonomy level(s) for kNN / linear-probe evaluation "
                             "labels (inat only). One or more of: kingdom, phylum, "
                             "class, order, family, genus. When several are given, "
                             "kNN and linear-probe metrics are logged for each. "
                             "For --dataset aircraft: manufacturer, family or variant.")
    parser.add_argument("--superclass", type=str, default=None,
                        help="Keep only iNat categories whose 'supercategory' matches "
                             "this value (e.g. Plants, Insects). inat only.")
    parser.add_argument("--inat_train_metadata", type=str, default="train_mini.json",
                        help="Path to iNat2021 train metadata JSON")
    parser.add_argument("--inat_val_metadata", type=str, default="val.json",
                        help="Path to iNat2021 val metadata JSON")
    parser.add_argument("--inat_train_dir", type=str, default="inat2021/train_mini",
                        help="Directory containing iNat2021 training images")
    parser.add_argument("--inat_val_dir", type=str, default="inat2021/val",
                        help="Directory containing iNat2021 validation images")
    parser.add_argument("--aircraft_root", type=str, default="data/fgvc_aircraft",
                        help="Root directory for torchvision's FGVCAircraft dataset "
                             "(aircraft only)")
    parser.add_argument("--aircraft_train_split", type=str, default="trainval",
                        choices=["train", "val", "trainval"],
                        help="FGVC-Aircraft split used for training")
    parser.add_argument("--aircraft_val_split", type=str, default="test",
                        choices=["train", "val", "trainval", "test"],
                        help="FGVC-Aircraft split used for evaluation")
    parser.add_argument("--aircraft_download", action="store_true",
                        help="Download the FGVC-Aircraft archive if it is missing")
    parser.add_argument("--img_size", type=int, default=96, help="Square image size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Model
    parser.add_argument("--model", type=str, default="backbone",
                        choices=["backbone"],
                        help="Which model to train: 'backbone' (a fully fine-tuned "
                             "backbone trained with supervised contrastive losses; "
                             "pick the architecture with --backbone)")
    parser.add_argument("--in_channels", type=int, default=3)

    # Contrastive model (only used when --model backbone)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_small_patch16_224",
                                 "swin_tiny_patch4_window7_224", "convnext_tiny"],
                        help="Backbone to fully fine-tune when --model backbone: "
                             "'resnet18', 'resnet50', 'vit_small_patch16_224', "
                             "'swin_tiny_patch4_window7_224' or 'convnext_tiny'")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Projected embedding dimension for the contrastive head")
    parser.add_argument("--proj_hidden_dim", type=int, default=2048,
                        help="Hidden width of the 2-layer projection MLP")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for the SupCon loss")
    parser.add_argument("--use_proj_head", action="store_true",
                        help="Use the projection head on top of the backbone features. "
                             "If not set, the backbone features are directly L2-normalized.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                        help="Enable gradient (activation) checkpointing on the backbone "
                             "to trade extra compute for lower memory (allows larger "
                             "batches). Only used with --model backbone.")

    # CE + Hierarchical Prototype Attention (requires --cross_entropy)
    parser.add_argument("--use_hpa", action="store_true",
                        help="Refine the final embedding with Hierarchical Prototype "
                             "Attention before the CE classifier. The hierarchy is "
                             "discovered from coarse labels only. Without this flag "
                             "the run is the plain CE baseline.")
    parser.add_argument("--hpa_levels", type=int, default=3,
                        help="Number L of ancestor prototypes each image attends to "
                             "(local -> intermediate -> broad). Default: 3")
    parser.add_argument("--hpa_heads", type=int, default=1,
                        help="Attention heads in the HPA block. Default: 1")
    parser.add_argument("--hpa_gamma", type=float, default=0.0,
                        help="Residual scale of the prototype context. Initial value "
                             "of the learned scalar, or the fixed value when "
                             "--hpa_fixed_gamma is set. Default: 0.0")
    parser.add_argument("--hpa_fixed_gamma", action="store_true",
                        help="Keep the HPA residual scale fixed at --hpa_gamma "
                             "instead of learning it.")
    parser.add_argument("--hpa_no_layernorm", action="store_true",
                        help="Disable the LayerNorm on the HPA prototype context.")
    parser.add_argument("--hierarchy_warmup_epochs", type=int, default=5,
                        help="CE-only epochs before the first hierarchy is built "
                             "(HPA is inactive until then). Default: 5")
    parser.add_argument("--hierarchy_update_interval", type=int, default=5,
                        help="Rebuild the hierarchy and prototypes every N epochs "
                             "after warm-up. Default: 5")
    parser.add_argument("--hierarchy_metric", type=str, default="cosine",
                        help="Pairwise distance for the agglomerative clustering")
    parser.add_argument("--hierarchy_linkage", type=str, default="average",
                        help="Linkage criterion for the agglomerative clustering")
    parser.add_argument("--hierarchy_snapshot_dir", type=str, default=None,
                        help="Directory to dump one .npz per hierarchy refresh for "
                             "offline analysis (see analyze_hierarchy.py). Disabled "
                             "by default.")

    # Contrastive dataset (synthesis-program labels; only used when --model backbone)
    parser.add_argument("--contrastive_metadata", type=str, nargs="+", default=None,
                        help="JSON metadata (compounds -> plates -> image paths) for "
                             "the contrastive dataset. Multiple files can be provided "
                             "and will be merged. Required for --model backbone.")
    parser.add_argument("--contrastive_labels", type=str, default=None,
                        help="CSV/Excel mapping compounds to synthesis-program labels. "
                             "Required for --model backbone.")
    parser.add_argument("--contrastive_root_dir", type=str, default=None,
                        help="Root directory prepended to image paths in the JSON. "
                             "Required for --model backbone.")
    parser.add_argument("--contrastive_compound_col", type=str, default="compound",
                        help="Compound-ID column in the label CSV. Default: compound")
    parser.add_argument("--contrastive_label_col", type=str, default="synthesis_program",
                        help="Synthesis-program column in the label CSV. "
                             "Default: synthesis_program")
    parser.add_argument("--contrastive_min_per_class", type=int, default=2,
                        help="Drop synthesis programs with fewer distinct compounds. "
                             "Default: 2")
    parser.add_argument("--contrastive_filter_efficacy", type=float, default=0,
                        help="Keep only compounds with Efficacy >= this value")
    parser.add_argument("--contrastive_use_control", action="store_true",
                        help="Also include per-plate control images as training samples")
    parser.add_argument("--contrastive_classes_per_batch", type=int, default=0,
                        help="P for class-balanced P x K sampling: distinct synthesis "
                             "programs per batch. When > 0 (with "
                             "--contrastive_samples_per_class), guarantees positives and "
                             "negatives in every batch; effective batch size = P * K "
                             "(overrides --batch_size for training).")
    parser.add_argument("--contrastive_samples_per_class", type=int, default=0,
                        help="K for class-balanced P x K sampling: images per synthesis "
                             "program per batch.")
    parser.add_argument("--compound_level", action="store_true",
                        help="Compute the contrastive loss at the compound level "
                             "instead of the synthesis-program level: each compound "
                             "becomes its own class, so positives are images of the "
                             "same compound (across plates/replicates).")
    parser.add_argument("--supcon_soft_pos_loss", action="store_true",
                        help="Use supervised contrastive (SupCon) loss with "
                             "similarity-weighted positives: positive pairs that are "
                             "more similar get larger weight, forming tighter "
                             "sub-clusters.")
    parser.add_argument("--supcon_soft_pos_tau", type=float, default=0.1,
                        help="Temperature for the positive-pair softmax weighting in "
                             "--supcon_soft_pos_loss (lower = more weight on closest "
                             "positives). Default: 0.1")
    parser.add_argument("--supcon_inst", action="store_true",
                        help="Add Grafit's BYOL-style instance-level term to "
                             "--supcon_soft_pos_loss. Needs multi-view batches "
                             "(--grafit_views) and adds an EMA target network.")
    parser.add_argument("--supcon_inst_weight", type=float, default=1.0,
                        help="Weight of the instance term in "
                             "L_supcon + w * L_inst. Default: 1.0")
    parser.add_argument("--taxocon_aug", action="store_true",
                        help="Use TaxoCon-Aug: the SupCon soft-positive loss whose "
                             "positive-pair weights come from the similarity averaged "
                             "over augmented views (--grafit_views) instead of a "
                             "single view. Shares --supcon_soft_pos_tau, "
                             "--tau_annealing and --denominator_pos_weight.")
    parser.add_argument("--denominator_pos_weight", action="store_true",
                        help="With --supcon_soft_pos_loss, also re-weight the positive "
                             "terms inside the SupCon denominator by the soft positive "
                             "weights (negatives stay at weight 1).")
    parser.add_argument("--tau_annealing", action="store_true",
                        help="Linearly anneal the SupCon soft-positive temperature from "
                             "--supcon_tau_start to --supcon_tau_end over training.")
    parser.add_argument("--supcon_tau_start", type=float, default=0.1,
                        help="Initial SupCon soft-positive temperature when "
                             "--tau_annealing is enabled. Default: 0.1")
    parser.add_argument("--supcon_tau_end", type=float, default=0.1,
                        help="Final SupCon soft-positive temperature when "
                             "--tau_annealing is enabled. Default: 0.1")
    parser.add_argument("--no_pos_weight_epoch", type=int, default=0,
                        help="Number of initial epochs that use uniform positive "
                            "weights before SupCon soft-positive weighting "
                            "starts. "
                             "Default: 0")
    parser.add_argument("--vanilla_supcon", action="store_true",
                        help="Use the plain Supervised Contrastive (SupCon) loss (no "
                             "soft positives): supervised single-view "
                             "SupCon with the coupled denominator.")
    parser.add_argument("--ms_loss", action="store_true",
                        help="Use the Multi-Similarity loss (Wang et al., CVPR 2019) "
                             "with pair mining, as in the official MS-Loss repo.")
    parser.add_argument("--ms_thresh", type=float, default=0.5,
                        help="Multi-Similarity similarity offset (lambda). Default: 0.5")
    parser.add_argument("--ms_margin", type=float, default=0.1,
                        help="Multi-Similarity pair-mining margin (epsilon). Default: 0.1")
    parser.add_argument("--ms_scale_pos", type=float, default=2.0,
                        help="Multi-Similarity positive scale (alpha). Default: 2.0")
    parser.add_argument("--ms_scale_neg", type=float, default=40.0,
                        help="Multi-Similarity negative scale (beta). Default: 40.0")
    parser.add_argument("--grafit", action="store_true",
                        help="Use the Grafit loss (Touvron et al., 2020): the coarse "
                             "kNN/NCA loss plus a BYOL-style instance-level term over "
                             "augmented views of the same image.")
    parser.add_argument("--grafit_lam", type=float, default=1.0,
                        help="Grafit instance weight in L_knn + lam*L_inst. "
                             "Paper default: 1.0")
    parser.add_argument("--grafit_views", type=int, default=2,
                        help="Augmented views per image in the training batches when "
                             "--grafit is set (the instance-level positives). "
                             "Default: 2")
    parser.add_argument("--grafit_bank", action="store_true",
                        help="Score Grafit's kNN loss against a memory bank holding "
                             "one embedding per training image (as in the paper) "
                             "instead of the in-batch embeddings. Costs "
                             "batch_size x train_size logits per step.")
    parser.add_argument("--maskcon", action="store_true",
                        help="Use the MaskCon loss (Feng & Patras, CVPR 2023): "
                             "MoCo-style contrastive learning whose soft targets "
                             "over a momentum-key queue are masked by the coarse "
                             "labels. Needs multi-view batches (--grafit_views) and "
                             "adds a momentum encoder (--EMA_momentum).")
    parser.add_argument("--maskcon_w", type=float, default=1.0,
                        help="MaskCon mixing weight between the masked soft target "
                             "and the purely self-supervised one (1.0 = pure "
                             "MaskCon, 0.0 = MoCo). Default: 1.0")
    parser.add_argument("--maskcon_soft_tau", type=float, default=0.1,
                        help="MaskCon soft-label temperature t0 used on the key "
                             "branch (--temperature is the main t). Default: 0.1")
    parser.add_argument("--maskcon_queue_size", type=int, default=4096,
                        help="Size of MaskCon's momentum-key queue. 0 falls back to "
                             "the in-batch keys. Default: 4096")
    parser.add_argument("--bucsfr", action="store_true",
                        help="Use the BuCSFR loss (Shi et al., ICCV 2025): MoCo-style "
                             "contrastive learning whose positives/negatives are "
                             "selected from a dendrogram built bottom-up inside each "
                             "coarse class, plus a coarse classification term. Needs "
                             "multi-view batches (--grafit_views) and adds a momentum "
                             "encoder (--EMA_momentum).")
    parser.add_argument("--bucsfr_alpha", type=float, default=0.5,
                        help="BuCSFR weight of the contrastive term in "
                             "alpha*L_con + (1-alpha)*L_ce. Default: 0.5")
    parser.add_argument("--bucsfr_queue_size", type=int, default=4096,
                        help="Size of BuCSFR's momentum-key queue (the instance "
                             "selection candidates). Default: 4096")
    parser.add_argument("--bucsfr_clusters_per_class", type=int, default=20,
                        help="Initial dendrogram leaves per coarse class; the paper "
                             "recommends 3-4x the expected number of fine-grained "
                             "classes. Default: 20")
    parser.add_argument("--bucsfr_threshold", type=float, default=1.1,
                        help="BuCSFR merge threshold T: a cluster pair is merged when "
                             "min(L_j,L_k)/I_jk < T (smaller = merge less eagerly, "
                             "recommended for fine-grained datasets). Default: 1.1")
    parser.add_argument("--bucsfr_warmup_epochs", type=int, default=10,
                        help="Epochs of plain MoCo training before the first "
                             "dendrogram is built. Default: 10")
    parser.add_argument("--bucsfr_refresh_every", type=int, default=1,
                        help="Rebuild the dendrogram (and merge one pair per coarse "
                             "class) every N epochs after warmup. Default: 1")
    parser.add_argument("--cross_entropy", action="store_true",
                        help="Supervised cross-entropy baseline: train a linear "
                             "classifier on the (normalized) embedding over the "
                             "train-label classes. Discarded at eval; kNN / "
                             "linear-probe still run on the embeddings.")
    parser.add_argument("--sinkhorn", action="store_true",
                        help="Use Sinkhorn-Knopp iterations to produce a doubly-stochastic "
                             "positive weight matrix instead of row-wise softmax. "
                             "Only used with the soft-positive losses.")
    parser.add_argument("--sinkhorn_iters", type=int, default=5,
                        help="Number of Sinkhorn-Knopp iterations. Default: 5")
    parser.add_argument("--EMA_pos_weight", action="store_true",
                        help="Compute the soft-positive weights from an EMA "
                             "(momentum-updated) teacher copy of the model instead "
                             "of the online embeddings. Used with "
                             "--supcon_soft_pos_loss or --taxocon_aug.")
    parser.add_argument("--EMA_momentum", type=float, default=0.999,
                        help="Momentum for the EMA teacher weight update when "
                             "--EMA_pos_weight is set (ema = m*ema + (1-m)*online). "
                             "Default: 0.999")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.95)
    parser.add_argument("--scheduler", type=str, default="exponential",
                        choices=["exponential", "cosine"],
                        help="LR scheduler: 'exponential' (decay by gamma each epoch) "
                             "or 'cosine' (anneal to ~0 over all epochs). Default: exponential")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs before the main schedule. Default: 0")
    parser.add_argument("--epochs", type=int, default=100)

    # Trainer / hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Lightning precision (e.g. 16-mixed, bf16-mixed, 32-true)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic algorithms (reproducible but slower)")
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N training epochs (default: 1)")
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                        help="Lightning logging interval in steps. Lower it when an "
                             "epoch has fewer batches than this (small datasets / "
                             "large batches), otherwise training metrics are not logged.")
    parser.add_argument("--seed", type=int, default=42)

    # Logging / checkpoints
    parser.add_argument("--project", type=str, default="hierarchy-prototype-attention",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--entity", type=str, default="fm_val",
                        help="W&B entity (team or username)")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="Optional W&B run tags, space separated")
    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


def ensure_wandb_login() -> None:
    """Ensure W&B is authenticated via the WANDB_API_KEY env var or a prior
    `wandb login`. Raises a clear error if no credentials are available."""
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        return

    # Fall back to cached credentials (e.g. from `wandb login`).
    if wandb.api.api_key:
        return

    raise RuntimeError(
        "Weights & Biases is not authenticated. Set the WANDB_API_KEY "
        "environment variable or run `wandb login` before training."
    )


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    ensure_wandb_login()

    # Inputs are fixed-size, so let cuDNN pick the fastest conv algorithms.
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True

    if args.taxocon_aug and args.supcon_soft_pos_loss:
        raise ValueError(
            "--taxocon_aug and --supcon_soft_pos_loss are mutually exclusive: "
            "TaxoCon-Aug is the multi-view variant of the SupCon soft-positive "
            "loss. Pass only --taxocon_aug (it reuses --supcon_soft_pos_tau)."
        )

    if args.use_hpa and not args.cross_entropy:
        raise ValueError(
            "--use_hpa requires --cross_entropy: CE is the only training "
            "objective in this version of Hierarchical Prototype Attention."
        )

    # Multi-view batches are only meaningful for the instance-level term.
    supcon_inst = args.supcon_soft_pos_loss and args.supcon_inst
    use_instance_term = args.grafit or supcon_inst
    grafit_views = args.grafit_views if (
        use_instance_term or args.taxocon_aug or args.maskcon or args.bucsfr) else 0
    # BuCSFR needs the dataset index to look up each sample's dendrogram cluster.
    grafit_bank = (args.grafit and args.grafit_bank) or args.bucsfr

    args.in_channels = 3

    if args.dataset == "inat":
        # iNaturalist 2021 dataset
        datamodule = InatDataModule(
            train_metadata=args.inat_train_metadata,
            val_metadata=args.inat_val_metadata,
            train_image_dir=args.inat_train_dir,
            val_image_dir=args.inat_val_dir,
            train_cat=args.train_cat,
            test_cat=args.test_cat,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            classes_per_batch=args.contrastive_classes_per_batch,
            samples_per_class=args.contrastive_samples_per_class,
            superclass=args.superclass,
            grafit_views=grafit_views,
            grafit_bank=grafit_bank,
            return_index=args.use_hpa,
            seed=args.seed,
        )
    elif args.dataset == "aircraft":
        # FGVC-Aircraft (manufacturer / family / variant hierarchy)
        datamodule = FGVCAircraftDataModule(
            root=args.aircraft_root,
            train_split=args.aircraft_train_split,
            val_split=args.aircraft_val_split,
            train_cat=args.train_cat,
            test_cat=args.test_cat,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            classes_per_batch=args.contrastive_classes_per_batch,
            samples_per_class=args.contrastive_samples_per_class,
            download=args.aircraft_download,
            grafit_views=grafit_views,
            grafit_bank=grafit_bank,
            return_index=args.use_hpa,
            seed=args.seed,
        )
    else:
        # Myzus (default) dataset
        missing = [name for name, val in (
            ("--contrastive_metadata", args.contrastive_metadata),
            ("--contrastive_labels", args.contrastive_labels),
            ("--contrastive_root_dir", args.contrastive_root_dir),
        ) if not val]
        if missing:
            raise ValueError(
                f"--model backbone requires {', '.join(missing)} to build the "
                "synthesis-program-labelled contrastive dataset."
            )

        datamodule = ContrastiveDataModule(
            image_metadata_json=args.contrastive_metadata,
            label_metadata_csv=args.contrastive_labels,
            root_dir=args.contrastive_root_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            compound_col=args.contrastive_compound_col,
            label_col=args.contrastive_label_col,
            min_compounds_per_class=args.contrastive_min_per_class,
            filter_by_efficacy=args.contrastive_filter_efficacy,
            use_control=args.contrastive_use_control,
            classes_per_batch=args.contrastive_classes_per_batch,
            samples_per_class=args.contrastive_samples_per_class,
            compound_level=args.compound_level,
            grafit_views=grafit_views,
            grafit_bank=grafit_bank,
            return_index=args.use_hpa,
            seed=args.seed,
        )

    # The cross-entropy baseline needs the number of training-label classes
    # up front to size its linear classifier head, and Grafit's memory bank
    # needs one slot per training image. setup() is idempotent; Lightning
    # calls it again internally during fit().
    num_classes = None
    grafit_bank_size = 0
    if args.cross_entropy or grafit_bank:
        datamodule.setup()
        grafit_bank_size = (len(datamodule.train_dataset)
                            if (args.grafit and args.grafit_bank) else 0)
    if args.cross_entropy or args.bucsfr:
        num_classes = (datamodule.num_train_classes
                       if args.dataset in ("inat", "aircraft")
                       else datamodule.num_classes)

    model = Backbone(
        backbone=args.backbone,
        img_size=args.img_size,
        embedding_dim=args.embedding_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        temperature=args.temperature,
        use_proj_head=args.use_proj_head,
        supcon_soft_pos=args.supcon_soft_pos_loss,
        supcon_soft_pos_tau=args.supcon_soft_pos_tau,
        supcon_denom_pos_weight=args.denominator_pos_weight,
        taxocon_aug=args.taxocon_aug,
        sinkhorn=args.sinkhorn,
        sinkhorn_iters=args.sinkhorn_iters,
        grad_checkpointing=args.grad_checkpointing,
        num_classes=num_classes,
        cross_entropy=args.cross_entropy or args.bucsfr,
        grafit_predictor=use_instance_term,
        use_hpa=args.use_hpa,
        hpa_heads=args.hpa_heads,
        hpa_gamma=args.hpa_gamma,
        hpa_learn_gamma=not args.hpa_fixed_gamma,
        hpa_layernorm=not args.hpa_no_layernorm,
    )

    experiment = ContrastiveExperiment(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        scheduler_gamma=args.scheduler_gamma,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        vanilla_supcon=args.vanilla_supcon,
        ms_loss=args.ms_loss,
        ms_thresh=args.ms_thresh,
        ms_margin=args.ms_margin,
        ms_scale_pos=args.ms_scale_pos,
        ms_scale_neg=args.ms_scale_neg,
        grafit=args.grafit,
        grafit_lam=args.grafit_lam,
        grafit_bank_size=grafit_bank_size,
        maskcon=args.maskcon,
        maskcon_w=args.maskcon_w,
        maskcon_soft_temperature=args.maskcon_soft_tau,
        maskcon_queue_size=args.maskcon_queue_size,
        bucsfr=args.bucsfr,
        bucsfr_alpha=args.bucsfr_alpha,
        bucsfr_queue_size=args.bucsfr_queue_size,
        bucsfr_clusters_per_class=args.bucsfr_clusters_per_class,
        bucsfr_threshold=args.bucsfr_threshold,
        bucsfr_warmup_epochs=args.bucsfr_warmup_epochs,
        bucsfr_refresh_every=args.bucsfr_refresh_every,
        supcon_softpos=args.supcon_soft_pos_loss,
        supcon_inst=supcon_inst,
        supcon_inst_weight=args.supcon_inst_weight,
        taxocon_aug=args.taxocon_aug,
        cross_entropy=args.cross_entropy,
        supcon_soft_pos_tau=args.supcon_soft_pos_tau,
        denom_pos_weight=args.denominator_pos_weight,
        tau_annealing=args.tau_annealing,
        supcon_tau_start=args.supcon_tau_start,
        supcon_tau_end=args.supcon_tau_end,
        no_pos_weight_epoch=args.no_pos_weight_epoch,
        sinkhorn=args.sinkhorn,
        sinkhorn_iters=args.sinkhorn_iters,
        EMA_pos_weight=args.EMA_pos_weight,
        EMA_momentum=args.EMA_momentum,
        use_hpa=args.use_hpa,
        hpa_levels=args.hpa_levels,
        hierarchy_warmup_epochs=args.hierarchy_warmup_epochs,
        hierarchy_update_interval=args.hierarchy_update_interval,
        hierarchy_metric=args.hierarchy_metric,
        hierarchy_linkage=args.hierarchy_linkage,
        hierarchy_snapshot_dir=args.hierarchy_snapshot_dir,
        train_cat=args.train_cat,
        test_cats=args.test_cat,
    )

    # Build checkpoint suffix (also used as default W&B run name).
    proj_tag = "Proj" if args.use_proj_head else "NoProj"
    p_val = args.contrastive_classes_per_batch
    k_val = args.contrastive_samples_per_class
    level_tag = "_Comp" if args.compound_level else ""
    test_cat_tag = "-".join(args.test_cat)
    dataset_tag = ""
    if args.dataset == "inat":
        dataset_tag = f"_inat_{args.train_cat}->{test_cat_tag}"
        if args.superclass:
            dataset_tag += f"_{args.superclass}"
    elif args.dataset == "aircraft":
        dataset_tag = f"_aircraft_{args.train_cat}->{test_cat_tag}"
    # "FFT" = full fine-tuning; short per-architecture tag.
    backbone_tag = {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "vit_small_patch16_224": "ViTs16",
        "swin_tiny_patch4_window7_224": "SwinT",
        "convnext_tiny": "ConvNeXtT",
    }.get(args.backbone, args.backbone)
    model_prefix = f"FFT_{backbone_tag}"
    ema_pw_tag = f"-EMA{args.EMA_momentum}" if args.EMA_pos_weight else ""
    vanilla_supcon_tag = "_VanillaSupCon" if args.vanilla_supcon else ""
    ms_loss_tag = (
        f"_MS-L{args.ms_thresh}-M{args.ms_margin}"
        f"-A{args.ms_scale_pos}-B{args.ms_scale_neg}"
    ) if args.ms_loss else ""
    cross_entropy_tag = "_CrossEntropy" if args.cross_entropy else ""
    hpa_tag = (
        f"_HPA-L{args.hpa_levels}-H{args.hpa_heads}"
        f"-G{args.hpa_gamma}{'fix' if args.hpa_fixed_gamma else ''}"
        f"-W{args.hierarchy_warmup_epochs}-E{args.hierarchy_update_interval}"
    ) if args.use_hpa else ""
    grafit_tag = (
        f"_Grafit-Lam{args.grafit_lam}-Views{args.grafit_views}"
        f"{'-Bank' if args.grafit_bank else ''}"
    ) if args.grafit else ""
    maskcon_tag = (
        f"_MaskCon-W{args.maskcon_w}-T0{args.maskcon_soft_tau}"
        f"-Q{args.maskcon_queue_size}-Views{args.grafit_views}"
        f"-M{args.EMA_momentum}"
    ) if args.maskcon else ""
    bucsfr_tag = (
        f"_BuCSFR-A{args.bucsfr_alpha}-C{args.bucsfr_clusters_per_class}"
        f"-T{args.bucsfr_threshold}-W{args.bucsfr_warmup_epochs}"
        f"-Q{args.bucsfr_queue_size}-Views{args.grafit_views}"
        f"-M{args.EMA_momentum}"
    ) if args.bucsfr else ""
    supcon_softpos_tag = (
        f"_SupConSoftPos-{'LinearTau' + str(args.supcon_tau_start) + 'to' + str(args.supcon_tau_end) if args.tau_annealing else 'Tau' + str(args.supcon_soft_pos_tau)}"
        f"{'-NoPosWeight' + str(args.no_pos_weight_epoch) if args.no_pos_weight_epoch else ''}"
        f"{'-DenomPosW' if args.denominator_pos_weight else ''}"
        f"{'-Sinkhorn' + str(args.sinkhorn_iters) if args.sinkhorn else ''}"
        f"{'-Inst' + str(args.supcon_inst_weight) + 'Views' + str(args.grafit_views) if args.supcon_inst else ''}"
        f"{ema_pw_tag}"
    ) if args.supcon_soft_pos_loss else ""
    taxocon_aug_tag = (
        f"_TaxoConAug-{'LinearTau' + str(args.supcon_tau_start) + 'to' + str(args.supcon_tau_end) if args.tau_annealing else 'Tau' + str(args.supcon_soft_pos_tau)}"
        f"-Views{args.grafit_views}"
        f"{'-NoPosWeight' + str(args.no_pos_weight_epoch) if args.no_pos_weight_epoch else ''}"
        f"{'-DenomPosW' if args.denominator_pos_weight else ''}"
        f"{'-Sinkhorn' + str(args.sinkhorn_iters) if args.sinkhorn else ''}"
        f"{ema_pw_tag}"
    ) if args.taxocon_aug else ""
    ckpt_suffix = (
        f"{model_prefix}"
        f"_P{p_val}_K{k_val}_BS{args.batch_size}"
        f"_{proj_tag}"
        f"_T{args.temperature}"
        f"{level_tag}"
        f"{vanilla_supcon_tag}"
        f"{ms_loss_tag}"
        f"{grafit_tag}"
        f"{maskcon_tag}"
        f"{bucsfr_tag}"
        f"{cross_entropy_tag}"
        f"{hpa_tag}"
        f"{supcon_softpos_tag}"
        f"{taxocon_aug_tag}"
        f"{dataset_tag}"
    )

    # Resume from a previous run if a `last.ckpt` already exists for this config.
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", ckpt_suffix)
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_from = resume_ckpt if os.path.exists(resume_ckpt) else None

    # Only pin a deterministic W&B run id when we are actually resuming from a
    # checkpoint, so the resumed training continues logging to the same run.
    # When starting fresh we let W&B generate a new id: reusing a fixed id fails
    # if that id was previously created and then deleted on the W&B server.
    if resume_from is not None:
        wandb_run_id = hashlib.sha1(
            f"{args.project}/{args.run_name or ckpt_suffix}".encode()
        ).hexdigest()[:16]
        print(f"[resume] Found {resume_ckpt}; resuming W&B run {wandb_run_id}")
    else:
        wandb_run_id = None

    # Logger (Weights & Biases)
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or ckpt_suffix,
        entity=args.entity,
        tags=args.tags,
        save_dir=args.output_dir,
        log_model=False,
        id=wandb_run_id,
        resume="allow",
    )
    wandb_logger.log_hyperparams(vars(args))

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor]

    # Keep the last epoch's checkpoint (last.ckpt) for all models.
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=0))

    # Also keep the checkpoint with the lowest validation loss.
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-val-loss",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    ))

    # Report the best-val-loss epoch's metrics as a table at the end of
    # training.
    # The active soft-positive loss determines which temperature governs the
    # positive-pair weighting.
    if args.supcon_soft_pos_loss:
        pos_weight_tau = (
            f"{args.supcon_tau_start}->{args.supcon_tau_end}"
            if args.tau_annealing else args.supcon_soft_pos_tau
        )
    elif args.taxocon_aug:
        pos_weight_tau = (
            f"{args.supcon_tau_start}->{args.supcon_tau_end}"
            if args.tau_annealing else args.supcon_soft_pos_tau
        )
    else:
        pos_weight_tau = "n/a"

    # Organise reports under <model>/<dataset>/ (e.g. resnet18/mammals/).
    model_folder = {
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "vit_small_patch16_224": "vits16",
        "swin_tiny_patch4_window7_224": "swint",
    }.get(args.backbone, args.backbone)
    if args.dataset == "inat":
        dataset_folder = args.superclass or f"{args.train_cat}_to_{'-'.join(args.test_cat)}"
    elif args.dataset == "aircraft":
        dataset_folder = f"aircraft_{args.train_cat}_to_{'-'.join(args.test_cat)}"
    else:
        dataset_folder = args.dataset

    callbacks.append(BestValLossReporter(
        pos_weight_tau=pos_weight_tau,
        sinkhorn=args.sinkhorn,
        sinkhorn_iters=args.sinkhorn_iters,
        stats_dir=os.path.join(args.output_dir, "reports", model_folder, dataset_folder),
        report_name=f"{ckpt_suffix}_best_val_loss_report.txt",
    ))

    # Trainer
    # Parse --devices: "auto" stays as-is; comma-separated digits become a
    # list of ints so Lightning selects the right GPU(s) (e.g. "0" -> [0]).
    devices = args.devices
    if devices != "auto":
        try:
            devices = [int(d) for d in devices.split(",")]
        except ValueError:
            pass  # let Lightning handle unexpected values
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.val_every_n_epochs,
        deterministic=args.deterministic,
    )

    trainer.fit(experiment, datamodule=datamodule, ckpt_path=resume_from)


if __name__ == "__main__":
    main()
