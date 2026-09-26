import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

# Backbones are pretrained with ImageNet normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_view_transform(img_size: int, rotation: float = 30.0,
                         translate: float = 0.1,
                         min_scale: float = 0.5,
                         gaussian_blur: float = 0.5) -> T.Compose:
    """Stochastic augmentation pipeline producing one random view of an image.

    Uses geometric augmentations and optional Gaussian blur: a random-resized
    crop (scale in ``[min_scale, 1.0]``) that makes the views differ in
    framing/zoom, plus random rotation and translation, and Gaussian blur
    applied with probability ``gaussian_blur``. Drawing the transform ``V``
    times from the same image gives ``V`` correlated views. Output is a float
    tensor normalized with ImageNet statistics.
    """
    transforms = [
        T.RandomResizedCrop(
            (img_size, img_size), scale=(min_scale, 1.0), antialias=True),
        T.RandomAffine(degrees=rotation, translate=(translate, translate)),
    ]
    if gaussian_blur > 0:
        kernel_size = img_size // 20 * 2 + 1  # odd kernel ~ 5% of image size
        transforms.append(
            T.RandomApply([T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))],
                          p=gaussian_blur))
    transforms += [
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


# ──────────────────────────────────────────────────────────────────────────────
# iNaturalist 2021 dataset support
# ──────────────────────────────────────────────────────────────────────────────


class InatContrastiveDataset(Dataset):
    """iNaturalist dataset returning (image, train_label, test_labels).

    ``train_label`` is used for contrastive loss and ``test_labels`` is a
    1-D tensor of evaluation labels, one per requested ``test_cat`` taxonomy
    level, used for kNN / linear-probe evaluation.

    With ``num_views > 1`` the (stochastic) transform is drawn ``num_views``
    times per image and the image becomes a ``[num_views, C, H, W]`` stack.
    ``return_index`` appends the dataset index, which addresses Grafit's
    memory-bank slots.
    """

    def __init__(self, samples: List[Tuple[str, int, Tuple[int, ...]]],
                 transform: T.Compose, num_views: int = 1,
                 return_index: bool = False) -> None:
        self.samples = samples
        self.transform = transform
        self.num_views = num_views
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        path, train_label, test_labels = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        if self.num_views > 1:
            image = torch.stack([self.transform(img) for _ in range(self.num_views)])
        else:
            image = self.transform(img)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        if self.return_index:
            return image, train_label, test_labels, index
        return image, train_label, test_labels


class InatDataModule(pl.LightningDataModule):
    """LightningDataModule for iNaturalist 2021 mini dataset.

    Loads train/val metadata JSONs (COCO-style with ``images``, ``annotations``,
    ``categories``), assigns contrastive labels from ``train_cat`` taxonomy level,
    and evaluation labels from one or more ``test_cat`` taxonomy levels.

    Args:
        train_metadata: path to train_mini.json
        val_metadata: path to val.json
        train_image_dir: path to training images (e.g. inat2021/train_mini)
        val_image_dir: path to val images (e.g. inat2021/val)
        train_cat: taxonomy column for contrastive training (e.g. 'class')
        test_cat: taxonomy column(s) for kNN evaluation (e.g. 'phylum'). A
            single string or a list of levels; each level is evaluated
            independently during validation.
        img_size: square image size
        batch_size: mini-batch size
        num_workers: DataLoader workers
        seed: RNG seed
    """

    def __init__(self,
                 train_metadata: str,
                 val_metadata: str,
                 train_image_dir: str,
                 val_image_dir: str,
                 train_cat: str = "class",
                 test_cat: Union[str, List[str]] = "phylum",
                 img_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 superclass: Optional[str] = None,
                 grafit_views: int = 0,
                 grafit_bank: bool = False,
                 return_index: bool = False,
                 seed: int = 42) -> None:
        super().__init__()
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.train_cat = train_cat
        self.test_cats = [test_cat] if isinstance(test_cat, str) else list(test_cat)
        self.superclass = superclass
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grafit_views = grafit_views
        self.grafit_bank = grafit_bank
        self.return_index = grafit_bank or return_index
        self.seed = seed

        self.train_classes: List[str] = []
        self.test_classes: List[List[str]] = []
        self._val_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_train_classes(self) -> int:
        return len(self.train_classes)

    @property
    def num_test_classes(self) -> List[int]:
        return [len(classes) for classes in self.test_classes]

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _train_transform(self, transform: T.Compose) -> T.Compose:
        # Grafit's instance term needs stochastic crops so that the views of an
        # image differ; otherwise the deterministic eval transform is used.
        if self.grafit_views > 1:
            return build_view_transform(self.img_size)
        return transform

    @staticmethod
    def _parse_inat_json(metadata_path: str, image_dir: str,
                         train_cat: str, test_cats: List[str],
                         superclass: Optional[str] = None
                         ) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """Parse iNat2021 COCO-style JSON, return (path, train_cat_value, test_cat_values).

        ``test_cat_values`` is a tuple with one taxonomy value per level in
        ``test_cats``. If ``superclass`` is given, only categories whose
        ``supercategory`` field matches it (case-insensitive) are kept.
        """
        with open(metadata_path) as f:
            data = json.load(f)

        # Build category_id -> taxonomy mapping
        cat_map: Dict[int, Dict[str, str]] = {}
        for cat in data["categories"]:
            cat_map[cat["id"]] = cat

        # Build image_id -> file_name mapping
        img_map: Dict[int, str] = {}
        for img in data["images"]:
            img_map[img["id"]] = img["file_name"]

        sc = superclass.lower() if superclass else None
        samples = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if img_id not in img_map or cat_id not in cat_map:
                continue
            cat_info = cat_map[cat_id]
            if sc is not None and str(cat_info.get("supercategory", "")).lower() != sc:
                continue
            train_val = cat_info.get(train_cat)
            test_vals = tuple(cat_info.get(tc) for tc in test_cats)
            if train_val is None or any(v is None for v in test_vals):
                continue
            file_name = img_map[img_id]
            full_path = os.path.join(image_dir, file_name)
            samples.append((full_path, str(train_val), tuple(str(v) for v in test_vals)))

        return samples

    def setup(self, stage: Optional[str] = None) -> None:
        train_raw = self._parse_inat_json(
            self.train_metadata, self.train_image_dir,
            self.train_cat, self.test_cats, self.superclass)
        val_raw = self._parse_inat_json(
            self.val_metadata, self.val_image_dir,
            self.train_cat, self.test_cats, self.superclass)

        # Build unified label encodings across both splits
        all_train_cats = sorted(set(s[1] for s in train_raw + val_raw))
        self.train_classes = all_train_cats
        train_cat2idx = {c: i for i, c in enumerate(all_train_cats)}

        # One label encoding per test taxonomy level.
        self.test_classes = [
            sorted(set(s[2][i] for s in train_raw + val_raw))
            for i in range(len(self.test_cats))
        ]
        test_cat2idx = [
            {c: j for j, c in enumerate(classes)} for classes in self.test_classes
        ]

        def encode(raw):
            return [
                (p, train_cat2idx[tc],
                 tuple(test_cat2idx[i][ec[i]] for i in range(len(self.test_cats))))
                for p, tc, ec in raw
            ]

        train_samples = encode(train_raw)
        val_samples = encode(val_raw)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(val_samples)

        self._val_labels = [s[1] for s in val_samples]

        transform = self._build_transform()
        self.train_dataset = InatContrastiveDataset(
            train_samples, self._train_transform(transform),
            num_views=max(self.grafit_views, 1),
            return_index=self.return_index)
        self.val_dataset = InatContrastiveDataset(val_samples, transform)

        test_summary = ", ".join(
            f"{name}({n})" for name, n in zip(self.test_cats, self.num_test_classes)
        )
        print(
            f"[InatDataModule] superclass={self.superclass}, "
            f"train_cat='{self.train_cat}' ({self.num_train_classes} classes), "
            f"test_cats=[{test_summary}], "
            f"images: train={len(train_samples)}, val={len(val_samples)}",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# FGVC-Aircraft dataset support
# ──────────────────────────────────────────────────────────────────────────────

# torchvision's FGVCAircraft exposes a 3-level hierarchy, coarse to fine.
AIRCRAFT_LEVELS = ("manufacturer", "family", "variant")


class FGVCAircraftDataModule(pl.LightningDataModule):
    """LightningDataModule for the FGVC-Aircraft dataset.

    Mirrors :class:`InatDataModule`: contrastive labels come from the
    ``train_cat`` annotation level and evaluation labels from one or more
    ``test_cat`` levels, each of which must be one of
    ``manufacturer`` / ``family`` / ``variant``.

    torchvision exposes a single annotation level per dataset instance, so one
    instance per level is built per split and the per-level labels are joined on
    the image file path.

    Args:
        root: dataset root passed to ``torchvision.datasets.FGVCAircraft``
        train_split: split used for training (``train``/``val``/``trainval``)
        val_split: split used for evaluation (usually ``test``)
        train_cat: annotation level for contrastive training labels
        test_cat: annotation level(s) for kNN / linear-probe evaluation
        download: download the archive if it is missing
    """

    def __init__(self,
                 root: str,
                 train_split: str = "trainval",
                 val_split: str = "test",
                 train_cat: str = "variant",
                 test_cat: Union[str, List[str]] = "family",
                 img_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 download: bool = False,
                 grafit_views: int = 0,
                 grafit_bank: bool = False,
                 return_index: bool = False,
                 seed: int = 42) -> None:
        super().__init__()
        self.root = root
        self.train_split = train_split
        self.val_split = val_split
        self.train_cat = train_cat
        self.test_cats = [test_cat] if isinstance(test_cat, str) else list(test_cat)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.grafit_views = grafit_views
        self.grafit_bank = grafit_bank
        self.return_index = grafit_bank or return_index
        self.seed = seed

        invalid = [c for c in [self.train_cat] + self.test_cats
                   if c not in AIRCRAFT_LEVELS]
        if invalid:
            raise ValueError(
                f"FGVC-Aircraft annotation levels must be one of {AIRCRAFT_LEVELS}; "
                f"got {invalid}."
            )

        self.train_classes: List[str] = []
        self.test_classes: List[List[str]] = []
        self._val_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_train_classes(self) -> int:
        return len(self.train_classes)

    @property
    def num_test_classes(self) -> List[int]:
        return [len(classes) for classes in self.test_classes]

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _train_transform(self, transform: T.Compose) -> T.Compose:
        # Grafit's instance term needs stochastic crops so that the views of an
        # image differ; otherwise the deterministic eval transform is used.
        if self.grafit_views > 1:
            return build_view_transform(self.img_size)
        return transform

    def prepare_data(self) -> None:
        if self.download:
            self._load_split(self.train_split, download=True)
            self._load_split(self.val_split, download=True)

    def _load_split(self, split: str, download: bool = False
                    ) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """Return (path, train_cat_value, test_cat_values) for one split."""
        from torchvision.datasets import FGVCAircraft

        per_level: Dict[str, Dict[str, str]] = {}
        files: Optional[List[str]] = None
        for level in dict.fromkeys([self.train_cat] + self.test_cats):
            ds = FGVCAircraft(root=self.root, split=split,
                              annotation_level=level, download=download)
            per_level[level] = {
                str(path): ds.classes[label]
                for path, label in zip(ds._image_files, ds._labels)
            }
            if files is None:
                files = [str(path) for path in ds._image_files]

        return [
            (path, per_level[self.train_cat][path],
             tuple(per_level[tc][path] for tc in self.test_cats))
            for path in (files or [])
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        train_raw = self._load_split(self.train_split)
        val_raw = self._load_split(self.val_split)

        # Unified label encodings across both splits.
        self.train_classes = sorted(set(s[1] for s in train_raw + val_raw))
        train_cat2idx = {c: i for i, c in enumerate(self.train_classes)}

        self.test_classes = [
            sorted(set(s[2][i] for s in train_raw + val_raw))
            for i in range(len(self.test_cats))
        ]
        test_cat2idx = [
            {c: j for j, c in enumerate(classes)} for classes in self.test_classes
        ]

        def encode(raw):
            return [
                (p, train_cat2idx[tc],
                 tuple(test_cat2idx[i][ec[i]] for i in range(len(self.test_cats))))
                for p, tc, ec in raw
            ]

        train_samples = encode(train_raw)
        val_samples = encode(val_raw)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(val_samples)

        self._val_labels = [s[1] for s in val_samples]

        transform = self._build_transform()
        self.train_dataset = InatContrastiveDataset(
            train_samples, self._train_transform(transform),
            num_views=max(self.grafit_views, 1),
            return_index=self.return_index)
        self.val_dataset = InatContrastiveDataset(val_samples, transform)

        test_summary = ", ".join(
            f"{name}({n})" for name, n in zip(self.test_cats, self.num_test_classes)
        )
        print(
            f"[FGVCAircraftDataModule] train_cat='{self.train_cat}' "
            f"({self.num_train_classes} classes), test_cats=[{test_summary}], "
            f"images: train={len(train_samples)}, val={len(val_samples)}",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
