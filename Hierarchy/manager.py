"""Latent hierarchy construction from coarse labels only.

:class:`HierarchyManager` runs agglomerative clustering **independently inside
every coarse class** over the current embeddings of the training set, keeps the
full dendrogram, and selects a small number of ancestor nodes per sample along
its leaf-to-root path (local -> intermediate -> broad).

Information-leakage contract
----------------------------
:meth:`HierarchyManager.build` accepts *only* sample ids, coarse training
labels and current embeddings. Family / genus / species / any evaluation label
must never be passed in or inspected here.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import os
import sys

import numpy as np
from tqdm.auto import tqdm

try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAS_SCIPY = False


@dataclass
class HierarchySnapshot:
    """Everything mini-batch training and offline analysis need from one refresh.

    Attributes:
        prototypes: ``(K, D)`` L2-normalized prototypes of the selected internal
            nodes, in float32. Row ``k`` corresponds to ``node_ids[k]``.
        level_proto_index: ``(N, L)`` row index into ``prototypes`` for every
            training sample and hierarchy level, ordered local -> broad.
        slot_rows: one array of prototype rows per level, holding the distinct
            nodes that were selected at that level. Used for label-free
            nearest-prototype lookup at inference time.
        node_ids: ``(K,)`` globally unique dendrogram node id per prototype.
        node_sizes: ``(K,)`` number of descendant training samples per node.
        node_coarse: ``(K,)`` coarse class each node belongs to.
        level_node_ids: ``(N, L)`` global node ids (the serializable twin of
            ``level_proto_index``).
        sample_ids: ``(N,)`` dataset indices the rows above refer to.
        coarse_labels: ``(N,)`` coarse training label per sample.
        nodes_per_coarse: coarse label -> number of selected nodes.
    """

    prototypes: np.ndarray
    level_proto_index: np.ndarray
    slot_rows: List[np.ndarray]
    node_ids: np.ndarray
    node_sizes: np.ndarray
    node_coarse: np.ndarray
    level_node_ids: np.ndarray
    sample_ids: np.ndarray
    coarse_labels: np.ndarray
    nodes_per_coarse: Dict[int, int] = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return self.level_proto_index.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.prototypes.shape[1]

    def stats(self) -> Dict[str, float]:
        counts = np.asarray(list(self.nodes_per_coarse.values()), dtype=np.float64)
        return {
            "num_nodes": float(self.prototypes.shape[0]),
            "nodes_per_coarse_mean": float(counts.mean()) if counts.size else 0.0,
            "nodes_per_coarse_max": float(counts.max()) if counts.size else 0.0,
            "node_size_mean": float(self.node_sizes.mean()) if self.node_sizes.size else 0.0,
            "node_size_median": float(np.median(self.node_sizes)) if self.node_sizes.size else 0.0,
        }

    def save(self, path: str) -> None:
        """Serialize the snapshot for offline hierarchy analysis."""
        np.savez_compressed(
            path,
            prototypes=self.prototypes.astype(np.float16),
            level_proto_index=self.level_proto_index,
            level_node_ids=self.level_node_ids,
            node_ids=self.node_ids,
            node_sizes=self.node_sizes,
            node_coarse=self.node_coarse,
            sample_ids=self.sample_ids,
            coarse_labels=self.coarse_labels,
        )


class HierarchyManager:
    """Builds a latent dendrogram per coarse class and selects ancestor nodes.

    Args:
        num_levels: number of ancestor prototypes ``L`` each sample attends to.
        metric: pairwise distance for the agglomerative clustering.
        linkage_method: linkage criterion (``average`` in the first version).
        level_fractions: position of each selected ancestor along the
            leaf-to-root path, as a fraction of the path length. Defaults to
            evenly spaced values ``l / (L + 1)`` -> ``0.25, 0.5, 0.75`` for
            ``L = 3``. Must be sorted ascending so that the selected nodes come
            out ordered local -> broad.
        max_samples_per_class: cap on how many samples of a coarse class enter
            the dendrogram. ``pdist`` is quadratic in time and memory, so large
            classes are clustered on a random subset and the remaining samples
            are attached to their nearest selected prototype per level. ``None``
            uses every sample.
        seed: seed of the subsampling RNG.
        progress: show a tqdm bar over the coarse classes while clustering.
    """

    def __init__(self,
                 num_levels: int = 3,
                 metric: str = "cosine",
                 linkage_method: str = "average",
                 level_fractions: Optional[Sequence[float]] = None,
                 max_samples_per_class: Optional[int] = None,
                 seed: int = 0,
                 progress: bool = True) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if max_samples_per_class is not None and max_samples_per_class < 3:
            raise ValueError("max_samples_per_class must be >= 3")
        if level_fractions is None:
            level_fractions = [(i + 1) / (num_levels + 1) for i in range(num_levels)]
        level_fractions = [float(f) for f in level_fractions]
        if len(level_fractions) != num_levels:
            raise ValueError("level_fractions must have num_levels entries")
        if any(b < a for a, b in zip(level_fractions, level_fractions[1:])):
            raise ValueError("level_fractions must be sorted ascending (local -> broad)")
        if not all(0.0 <= f <= 1.0 for f in level_fractions):
            raise ValueError("level_fractions must lie in [0, 1]")

        self.num_levels = num_levels
        self.metric = metric
        self.linkage_method = linkage_method
        self.level_fractions = np.asarray(level_fractions, dtype=np.float64)
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed
        self.progress = progress

    def build(self, embeddings: np.ndarray, coarse_labels: np.ndarray,
              sample_ids: Optional[np.ndarray] = None) -> HierarchySnapshot:
        """Cluster ``embeddings`` within each coarse class and select ancestors.

        Args:
            embeddings: ``(N, D)`` L2-normalized embeddings of the training set.
            coarse_labels: ``(N,)`` coarse training labels. The **only** label
                information this component is ever allowed to see.
            sample_ids: ``(N,)`` dataset indices; defaults to ``arange(N)``.
        """
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for hierarchy construction. "
                "Install it with `pip install scipy`."
            )

        embeddings = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        coarse_labels = np.asarray(coarse_labels).reshape(-1).astype(np.int64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D (N, D) array")
        if embeddings.shape[0] != coarse_labels.shape[0]:
            raise ValueError("embeddings and coarse_labels must have the same length")
        n_total, dim = embeddings.shape
        if sample_ids is None:
            sample_ids = np.arange(n_total, dtype=np.int64)
        sample_ids = np.asarray(sample_ids).reshape(-1).astype(np.int64)

        level_proto_index = np.full((n_total, self.num_levels), -1, dtype=np.int64)
        level_node_ids = np.full((n_total, self.num_levels), -1, dtype=np.int64)
        proto_blocks: List[np.ndarray] = []
        node_id_blocks: List[np.ndarray] = []
        node_size_blocks: List[np.ndarray] = []
        node_coarse_blocks: List[np.ndarray] = []
        nodes_per_coarse: Dict[int, int] = {}

        row_offset = 0   # running number of prototypes emitted so far
        node_offset = 0  # keeps dendrogram node ids unique across coarse classes
        rng = np.random.default_rng(self.seed)

        classes = np.unique(coarse_labels)
        progress = progress_bar(classes, "[HPA] clustering coarse classes",
                                disable=not self.progress)
        for coarse in progress:
            member_idx = np.flatnonzero(coarse_labels == coarse)
            fit_idx, rest_idx = self._split_for_fit(member_idx, rng)
            progress.set_postfix(cls=int(coarse), n=int(fit_idx.size),
                                 refresh=False)
            # Every clustering problem sees exactly one coarse class.
            block = self._build_one_class(embeddings[fit_idx])

            n_nodes = block["num_tree_nodes"]
            protos, node_local_ids, sizes, sel_rows = (
                block["prototypes"], block["node_ids"], block["node_sizes"],
                block["level_rows"],
            )

            level_proto_index[fit_idx] = sel_rows + row_offset
            level_node_ids[fit_idx] = node_local_ids[sel_rows] + node_offset

            if rest_idx.size:
                rest_rows = _nearest_rows_per_level(
                    embeddings[rest_idx], protos, sel_rows)
                level_proto_index[rest_idx] = rest_rows + row_offset
                level_node_ids[rest_idx] = node_local_ids[rest_rows] + node_offset

            proto_blocks.append(protos)
            node_id_blocks.append(node_local_ids + node_offset)
            node_size_blocks.append(sizes)
            node_coarse_blocks.append(np.full(protos.shape[0], coarse, dtype=np.int64))
            nodes_per_coarse[int(coarse)] = int(protos.shape[0])

            row_offset += protos.shape[0]
            node_offset += n_nodes

        prototypes = np.concatenate(proto_blocks, axis=0) if proto_blocks else \
            np.zeros((0, dim), dtype=np.float32)
        slot_rows = [np.unique(level_proto_index[:, l])
                     for l in range(self.num_levels)]

        return HierarchySnapshot(
            prototypes=prototypes,
            level_proto_index=level_proto_index,
            slot_rows=slot_rows,
            node_ids=np.concatenate(node_id_blocks) if node_id_blocks
            else np.zeros(0, dtype=np.int64),
            node_sizes=np.concatenate(node_size_blocks) if node_size_blocks
            else np.zeros(0, dtype=np.int64),
            node_coarse=np.concatenate(node_coarse_blocks) if node_coarse_blocks
            else np.zeros(0, dtype=np.int64),
            level_node_ids=level_node_ids,
            sample_ids=sample_ids,
            coarse_labels=coarse_labels,
            nodes_per_coarse=nodes_per_coarse,
        )

    def _split_for_fit(self, member_idx: np.ndarray, rng: np.random.Generator
                       ) -> tuple:
        """Split one class into the samples that are clustered and the rest."""
        cap = self.max_samples_per_class
        if cap is None or member_idx.size <= cap:
            return member_idx, np.zeros(0, dtype=np.int64)
        perm = rng.permutation(member_idx.size)
        # Sorted so the dendrogram is invariant to the draw order.
        return np.sort(member_idx[perm[:cap]]), np.sort(member_idx[perm[cap:]])

    def _build_one_class(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Dendrogram + ancestor selection for the samples of a single coarse class.

        Returns the deduplicated prototypes of the selected nodes, their tree
        node ids and descendant counts, and an ``(n, L)`` array of prototype
        rows per sample (already ordered local -> broad).
        """
        n, dim = x.shape
        if n <= 2:
            # Degenerate class: the only meaningful node is the class itself.
            root_proto = _l2_normalize(x.mean(axis=0, keepdims=True))
            return {
                "prototypes": root_proto.astype(np.float32),
                "node_ids": np.zeros(1, dtype=np.int64),
                "node_sizes": np.asarray([n], dtype=np.int64),
                "level_rows": np.zeros((n, self.num_levels), dtype=np.int64),
                "num_tree_nodes": 1,
            }

        condensed = pdist(x.astype(np.float64), metric=self.metric)
        z = linkage(condensed, method=self.linkage_method)
        del condensed

        num_tree_nodes = 2 * n - 1
        children = z[:, :2].astype(np.int64)

        # Bottom-up sums give every internal node's prototype in one O(n * D)
        # pass, without materializing descendant lists.
        sums = np.zeros((num_tree_nodes, dim), dtype=np.float64)
        counts = np.zeros(num_tree_nodes, dtype=np.int64)
        sums[:n] = x
        counts[:n] = 1
        for i in range(n - 1):
            a, b = children[i]
            sums[n + i] = sums[a] + sums[b]
            counts[n + i] = counts[a] + counts[b]

        sel_nodes = self._select_ancestors(n, children)

        uniq_nodes, inverse = np.unique(sel_nodes.ravel(), return_inverse=True)
        protos = _l2_normalize(sums[uniq_nodes] / counts[uniq_nodes][:, None])

        return {
            "prototypes": protos.astype(np.float32),
            "node_ids": uniq_nodes.astype(np.int64),
            "node_sizes": counts[uniq_nodes].astype(np.int64),
            "level_rows": inverse.reshape(n, self.num_levels).astype(np.int64),
            "num_tree_nodes": num_tree_nodes,
        }

    def _select_ancestors(self, n: int, children: np.ndarray) -> np.ndarray:
        """Pick ``L`` evenly spaced ancestors on each leaf's path to the root.

        A single iterative DFS from the root keeps the current root-to-node
        ancestor stack, so every leaf's full path is available in O(1) without
        walking parent pointers per leaf (which would be O(n^2) on an unbalanced
        dendrogram).
        """
        num_tree_nodes = 2 * n - 1
        root = num_tree_nodes - 1
        selected = np.empty((n, self.num_levels), dtype=np.int64)
        path = np.empty(num_tree_nodes, dtype=np.int64)

        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if node < n:
                # path[0:depth] is root -> ... -> parent, so reversing it gives
                # the local -> broad ordering the attention block expects.
                positions = np.rint(self.level_fractions * (depth - 1)).astype(np.int64)
                np.clip(positions, 0, depth - 1, out=positions)
                selected[node] = path[depth - 1 - positions]
            else:
                path[depth] = node
                left, right = children[node - n]
                stack.append((int(left), depth + 1))
                stack.append((int(right), depth + 1))

        return selected


def progress_bar(iterable: Iterable, desc: str, disable: bool = False,
                 total: Optional[int] = None) -> tqdm:
    """tqdm bar that stays readable in log files.

    Without a TTY every refresh emits a new line, so redirected runs get a
    fixed-width bar printed at most once per ``HPA_PROGRESS_INTERVAL`` seconds
    (default 30) instead of once per item.
    """
    tty = sys.stdout.isatty()
    interval = float(os.environ.get("HPA_PROGRESS_INTERVAL", 30.0))
    # The monitor thread would force an extra line between our own refreshes.
    tqdm.monitor_interval = 0
    return tqdm(iterable, desc=desc, total=total, disable=disable,
                file=sys.stdout, leave=not tty, dynamic_ncols=tty,
                ncols=None if tty else 300,
                mininterval=0.1 if tty else interval,
                maxinterval=float("inf") if not tty else 10.0)


def _nearest_rows_per_level(x: np.ndarray, prototypes: np.ndarray,
                            sel_rows: np.ndarray) -> np.ndarray:
    """Attach held-out samples of a class to the nearest prototype per level.

    Both sides are L2-normalized, so the dot product ranks by cosine similarity.
    """
    z = _l2_normalize(x.astype(np.float32))
    rows = np.empty((x.shape[0], sel_rows.shape[1]), dtype=np.int64)
    for level in range(sel_rows.shape[1]):
        candidates = np.unique(sel_rows[:, level])
        rows[:, level] = candidates[
            (z @ prototypes[candidates].T).argmax(axis=1)]
    return rows


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)
