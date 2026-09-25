"""Offline diagnostics for the discovered latent hierarchy.

EVALUATION ONLY. This script reads the hierarchy snapshots dumped by
``train.py --hierarchy_snapshot_dir ...`` and compares them against the hidden
fine-grained labels. It is never imported by the training code and its metrics
must never influence training, hyper-parameter selection or model selection.

Example
-------
    python analyze_hierarchy.py \
        --snapshots results/hierarchy/hierarchy_epoch*.npz \
        --inat_metadata train_mini.json --inat_image_dir inat2021/train_mini \
        --train_cat order --fine_cat species
"""

import argparse
import glob
import os
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a discovered latent hierarchy against hidden fine labels")
    parser.add_argument("--snapshots", type=str, nargs="+", required=True,
                        help="Hierarchy .npz files (globs allowed)")
    parser.add_argument("--inat_metadata", type=str, required=True,
                        help="iNat2021 metadata JSON matching the training split")
    parser.add_argument("--inat_image_dir", type=str, required=True,
                        help="Image directory matching the training split")
    parser.add_argument("--train_cat", type=str, required=True,
                        help="Coarse taxonomy level the model was trained on")
    parser.add_argument("--fine_cat", type=str, default="species",
                        help="Hidden fine taxonomy level to evaluate against")
    parser.add_argument("--superclass", type=str, default=None,
                        help="Same --superclass filter used during training")
    parser.add_argument("--max_pairs", type=int, default=2_000_000,
                        help="Sampled pairs for the pairwise-agreement metric")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_fine_labels(args: argparse.Namespace) -> np.ndarray:
    """Fine labels in the exact dataset order the snapshots were indexed with."""
    from dataset import InatDataModule

    raw = InatDataModule._parse_inat_json(
        args.inat_metadata, args.inat_image_dir,
        args.train_cat, [args.fine_cat], args.superclass)
    classes = sorted({s[2][0] for s in raw})
    cat2idx = {c: i for i, c in enumerate(classes)}
    return np.asarray([cat2idx[s[2][0]] for s in raw], dtype=np.int64)


def cluster_purity(assignment: np.ndarray, fine: np.ndarray) -> float:
    """Sample-weighted fraction of each node covered by its majority fine label."""
    correct = 0
    for node in np.unique(assignment):
        members = fine[assignment == node]
        correct += np.bincount(members).max()
    return float(correct) / len(fine)


def pairwise_agreement(assignment: np.ndarray, fine: np.ndarray,
                       max_pairs: int, seed: int) -> Dict[str, float]:
    """Precision/recall of 'same node' as a predictor of 'same fine label'."""
    rng = np.random.default_rng(seed)
    n = len(fine)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    same_node = assignment[i] == assignment[j]
    same_fine = fine[i] == fine[j]
    tp = float(np.sum(same_node & same_fine))
    precision = tp / max(float(same_node.sum()), 1.0)
    recall = tp / max(float(same_fine.sum()), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"pair_precision": precision, "pair_recall": recall, "pair_f1": f1}


def latent_vs_taxonomy_correlation(level_nodes: np.ndarray, fine: np.ndarray,
                                   coarse: np.ndarray, max_samples: int,
                                   seed: int) -> Dict[str, float]:
    """Spearman correlation between latent tree depth-of-LCA and taxonomy agreement.

    The latent distance between two samples is the number of hierarchy levels at
    which they do **not** share the selected node (0 = same node everywhere).
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(fine)
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    nodes, f, c = level_nodes[idx], fine[idx], coarse[idx]

    num_levels = nodes.shape[1]
    latent = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for l in range(num_levels):
        latent += (nodes[:, l][:, None] != nodes[:, l][None, :])
    # Taxonomy distance: 0 same fine class, 1 same coarse only, 2 otherwise.
    tax = (f[:, None] != f[None, :]).astype(np.float32) + \
          (c[:, None] != c[None, :]).astype(np.float32)

    triu = np.triu_indices(len(idx), k=1)
    rho, _ = spearmanr(latent[triu], tax[triu])
    return {"latent_tax_spearman": float(rho)}


def analyze(path: str, fine: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    data = np.load(path)
    level_nodes = data["level_node_ids"]
    coarse = data["coarse_labels"]
    sample_ids = data["sample_ids"]

    if len(fine) != len(sample_ids):
        raise ValueError(
            f"{path}: snapshot has {len(sample_ids)} samples but the metadata "
            f"yields {len(fine)}; check --inat_metadata / --superclass.")
    fine = fine[sample_ids]

    metrics: Dict[str, float] = {}
    for level in range(level_nodes.shape[1]):
        assignment = level_nodes[:, level]
        metrics[f"level{level}_purity"] = cluster_purity(assignment, fine)
        metrics[f"level{level}_num_nodes"] = float(len(np.unique(assignment)))
        for key, value in pairwise_agreement(
                assignment, fine, args.max_pairs, args.seed).items():
            metrics[f"level{level}_{key}"] = value
    metrics.update(latent_vs_taxonomy_correlation(
        level_nodes, fine, coarse, max_samples=2000, seed=args.seed))
    return metrics


def main() -> None:
    args = parse_args()
    paths: List[str] = sorted(
        p for pattern in args.snapshots for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No snapshots matched {args.snapshots}")

    fine = load_fine_labels(args)
    for path in paths:
        metrics = analyze(path, fine, args)
        print(f"\n=== {os.path.basename(path)} ===")
        for key, value in metrics.items():
            print(f"  {key:32s} {value:.4f}")


if __name__ == "__main__":
    main()
