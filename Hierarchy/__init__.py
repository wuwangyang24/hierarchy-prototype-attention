"""Latent hierarchy discovery from coarse labels (see :mod:`Hierarchy.manager`)."""

from .manager import HierarchyManager, HierarchySnapshot, progress_bar
from .prototypes import PrototypeBank

__all__ = ["HierarchyManager", "HierarchySnapshot", "PrototypeBank",
           "progress_bar"]
