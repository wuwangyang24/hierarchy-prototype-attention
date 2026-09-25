import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

# Backbones are pretrained with ImageNet normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class PKBatchSampler(Sampler):
    """Class-balanced (P x K) batch sampler for supervised contrastive training.

    Each yielded batch contains ``classes_per_batch`` (P) distinct synthesis
    programs with ``samples_per_class`` (K) images each, so every batch is
    guaranteed to hold multiple positives per program (same class) and multiple
    negatives (different classes). The effective batch size is ``P * K``.

    Classes with fewer than K samples are sampled with replacement. The number
    of batches per epoch defaults to ``len(labels) // (P * K)``.
    """

    def __init__(self, labels: List[int], classes_per_batch: int,
                 samples_per_class: int, num_batches: Optional[int] = None,
                 seed: int = 0) -> None:
        super().__init__(None)
        self.labels = np.asarray(labels)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self._epoch = 0

        self.label_to_indices: Dict[int, np.ndarray] = {}
        for idx, lab in enumerate(self.labels):
            self.label_to_indices.setdefault(int(lab), []).append(idx)
        self.label_to_indices = {
            lab: np.asarray(idxs) for lab, idxs in self.label_to_indices.items()
        }
        self.unique_labels = list(self.label_to_indices.keys())

        # Can't draw more distinct classes than exist.
        self.classes_per_batch = min(classes_per_batch, len(self.unique_labels))
        if self.classes_per_batch < 2:
            raise ValueError(
                "PKBatchSampler needs at least 2 synthesis programs to form "
                f"contrastive batches, found {len(self.unique_labels)}."
            )

        batch_size = self.classes_per_batch * self.samples_per_class
        if num_batches is None:
            num_batches = len(self.labels) // batch_size
        self.num_batches = max(1, num_batches)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        # Vary the shuffle each epoch while staying reproducible.
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.num_batches):
            chosen = rng.choice(
                self.unique_labels, size=self.classes_per_batch, replace=False)
            batch: List[int] = []
            for lab in chosen:
                idxs = self.label_to_indices[int(lab)]
                replace = len(idxs) < self.samples_per_class
                picked = rng.choice(idxs, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in picked)
            yield batch


class ContrastiveImageDataset(Dataset):
    """Loads RGB images labelled by synthesis program for contrastive training.

    Each item is ``(image_tensor, label_idx)`` where ``label_idx`` is the
    integer-encoded synthesis-program class. Images are resized, scaled to
    ``[0, 1]``, and normalized with ImageNet statistics.

    With ``num_views > 1`` the (stochastic) transform is drawn ``num_views``
    times per image and the item becomes ``([num_views, C, H, W], label_idx)``;
    this is what Grafit's instance-level term needs. ``return_index`` appends
    the dataset index, which addresses Grafit's memory-bank slots.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform: T.Compose,
                 num_views: int = 1, return_index: bool = False) -> None:
        self.samples = samples
        self.transform = transform
        self.num_views = num_views
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = read_image(path, mode=ImageReadMode.RGB)
        if self.num_views > 1:
            image = torch.stack([self.transform(img) for _ in range(self.num_views)])
        else:
            image = self.transform(img)
        if self.return_index:
            return image, label, index
        return image, label


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


class ContrastiveDataModule(pl.LightningDataModule):
    """LightningDataModule serving synthesis-program-labelled images for
    supervised contrastive (SupCon) training of the backbone model.

    Labels are derived by joining an image-metadata JSON (compound -> plates ->
    image paths, same format as the classifier callback) with a label CSV/Excel
    mapping each compound to a ``synthesis_program`` class.

    Args:
        image_metadata_json: JSON mapping compounds to plate/image paths.
        label_metadata_csv: CSV/Excel with compound -> synthesis-program labels.
        root_dir: base directory prepended to the relative image paths.
        img_size: square image size.
        batch_size: mini-batch size for both loaders.
        num_workers: DataLoader worker processes.
        val_split: fraction of images held out for validation.
        compound_col: compound-ID column in the label CSV.
        label_col: synthesis-program column in the label CSV.
        min_compounds_per_class: drop classes with fewer distinct compounds.
        filter_by_efficacy: keep only compounds with ``Efficacy`` >= this value
            (ignored if the column is absent or the value is 0/None).
        use_control: also include per-plate control images as training samples.
        classes_per_batch: P for P x K class-balanced sampling. When > 0 (with
            ``samples_per_class`` > 0), each train batch holds this many distinct
            synthesis programs, guaranteeing positives and negatives per batch.
        samples_per_class: K images per program for P x K sampling.
        compound_level: derive contrastive labels at the compound level instead
            of the synthesis-program level. Each compound becomes its own class,
            so positives are images of the same compound (across plates /
            replicates) rather than of the same synthesis program.
        seed: RNG seed for the train/val split.
    """

    def __init__(self,
                 image_metadata_json,
                 label_metadata_csv: str,
                 root_dir: str,
                 img_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 val_split: float = 0.1,
                 compound_col: str = "compound",
                 label_col: str = "synthesis_program",
                 min_compounds_per_class: int = 2,
                 filter_by_efficacy: Optional[float] = 0,
                 use_control: bool = False,
                 classes_per_batch: int = 0,
                 samples_per_class: int = 0,
                 compound_level: bool = False,
                 grafit_views: int = 0,
                 grafit_bank: bool = False,
                 seed: int = 42) -> None:
        super().__init__()
        self.image_metadata_json = image_metadata_json
        self.label_metadata_csv = label_metadata_csv
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.compound_col = compound_col
        self.label_col = label_col
        self.min_compounds_per_class = min_compounds_per_class
        self.filter_by_efficacy = filter_by_efficacy
        self.use_control = use_control
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.compound_level = compound_level
        self.grafit_views = grafit_views
        self.grafit_bank = grafit_bank
        self.seed = seed

        self.classes: List[str] = []
        self._train_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def use_pk_sampler(self) -> bool:
        return (self.classes_per_batch > 0 and self.samples_per_class > 0)

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_compound_labels(self) -> Dict[str, str]:
        """Return a {compound_id: synthesis_program} map, after optional
        efficacy filtering and dropping classes with too few compounds."""
        suffix = os.path.splitext(self.label_metadata_csv)[1].lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(self.label_metadata_csv)
        else:
            df = pd.read_csv(self.label_metadata_csv)

        if (self.filter_by_efficacy and self.filter_by_efficacy > 0
                and "Efficacy" in df.columns):
            df = df[df["Efficacy"] >= self.filter_by_efficacy]

        df = df[[self.compound_col, self.label_col]].dropna()
        df[self.compound_col] = df[self.compound_col].astype(str)
        df[self.label_col] = df[self.label_col].astype(str)

        # Drop classes with fewer than the required number of distinct compounds.
        min_cpc = max(self.min_compounds_per_class, 2)
        counts = df.groupby(self.label_col)[self.compound_col].nunique()
        valid_classes = set(counts[counts >= min_cpc].index)
        df = df[df[self.label_col].isin(valid_classes)]

        return dict(zip(df[self.compound_col], df[self.label_col]))

    def _build_samples(self) -> List[Tuple[str, int]]:
        # Support single path or list of paths for metadata JSONs.
        paths = self.image_metadata_json
        if isinstance(paths, str):
            paths = [paths]
        metadata = []
        for p in paths:
            print(f"[ContrastiveDataModule] Loading metadata: {p} ...", flush=True)
            with open(p) as f:
                metadata.extend(json.load(f))
        print(f"[ContrastiveDataModule] Loaded {len(metadata)} entries from {len(paths)} file(s)", flush=True)

        comp2label = self._load_compound_labels()
        print(f"[ContrastiveDataModule] Label map: {len(comp2label)} compounds", flush=True)
        if not comp2label:
            raise RuntimeError(
                "No compounds with valid synthesis-program labels remained after "
                "filtering. Check --contrastive_labels / --contrastive_min_per_class."
            )

        if self.compound_level:
            # Each compound is its own contrastive class; positives are images
            # of the same compound (across plates / replicates).
            self.classes = sorted(comp2label.keys())
            label2idx = {c: i for i, c in enumerate(self.classes)}

            def label_for(compound_id: str) -> int:
                return label2idx[compound_id]
        else:
            self.classes = sorted(set(comp2label.values()))
            label2idx = {c: i for i, c in enumerate(self.classes)}

            def label_for(compound_id: str) -> int:
                return label2idx[comp2label[compound_id]]

        subsets = ("treated", "control") if self.use_control else ("treated",)
        samples: List[Tuple[str, int]] = []
        for entry in metadata:
            cid = str(entry["Compound"])
            if cid not in comp2label:
                continue
            label_idx = label_for(cid)
            for plate_id, plate_data in entry.items():
                if plate_id == "Compound":
                    continue
                for subset in subsets:
                    for rel in plate_data.get(subset, []):
                        samples.append((os.path.join(self.root_dir, rel), label_idx))

        if not samples:
            raise RuntimeError(
                "No labelled images found. Check --contrastive_metadata / "
                "--contrastive_root_dir and the compound-ID join."
            )

        # Recompute classes to only include those with actual images.
        actual_labels = sorted(set(label for _, label in samples))
        if self.compound_level:
            self.classes = [self.classes[i] for i in actual_labels]
        else:
            self.classes = [self.classes[i] for i in actual_labels]
        # Remap labels to contiguous 0..K-1
        old2new = {old: new for new, old in enumerate(actual_labels)}
        samples = [(path, old2new[label]) for path, label in samples]

        return samples

    def setup(self, stage: Optional[str] = None) -> None:
        samples = self._build_samples()

        rng = np.random.default_rng(self.seed)

        if self.compound_level:
            # Split at the compound level: all images of a given compound go
            # entirely to train or val to avoid data leakage.
            label_to_indices: Dict[int, List[int]] = {}
            for idx, (_, label) in enumerate(samples):
                label_to_indices.setdefault(label, []).append(idx)
            all_labels = list(label_to_indices.keys())
            rng.shuffle(all_labels)
            n_val_classes = max(1, int(len(all_labels) * self.val_split))
            val_labels = set(all_labels[:n_val_classes])
            train_idx = [i for lab, idxs in label_to_indices.items()
                         if lab not in val_labels for i in idxs]
            val_idx = [i for lab in val_labels for i in label_to_indices[lab]]
        else:
            indices = rng.permutation(len(samples))
            n_val = int(len(samples) * self.val_split)
            val_idx = indices[:n_val].tolist()
            train_idx = indices[n_val:].tolist()

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        # Shuffle val samples so that the val dataloader (shuffle=False) does
        # not serve compound-grouped batches, which would inflate batch-level
        # metrics like kNN accuracy.
        rng.shuffle(val_samples)

        transform = self._build_transform()
        if self.grafit_views > 1:
            # Grafit's instance term needs several augmented crops of the same
            # image; validation stays single-view for honest kNN metrics.
            view_transform = build_view_transform(self.img_size)
            self.train_dataset = ContrastiveImageDataset(
                train_samples, view_transform, num_views=self.grafit_views,
                return_index=self.grafit_bank)
        else:
            self.train_dataset = ContrastiveImageDataset(
                train_samples, transform, return_index=self.grafit_bank)
        self.val_dataset = ContrastiveImageDataset(val_samples, transform)
        self._train_labels = [label for _, label in train_samples]
        self._val_labels = [label for _, label in val_samples]
        print(
            f"[ContrastiveDataModule] {len(samples)} images, "
            f"{self.num_classes} "
            f"{'compounds' if self.compound_level else 'synthesis programs'} "
            f"(train={len(train_samples)}, val={len(val_samples)})",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        # Class-balanced P x K sampling guarantees positives and negatives per
        # batch; falls back to plain random shuffling when disabled.
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._train_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
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
        # Always use a plain full-coverage loader for validation. The PK sampler
        # oversamples small classes with replacement (duplicate images) and
        # restricts each batch to a few classes, which trivially inflates the
        # leave-one-out val_knn_acc. Evaluating every unique image once against
        # all classes gives an honest metric.
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
        classes_per_batch: P for P x K sampling (0 disables)
        samples_per_class: K for P x K sampling
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
                 classes_per_batch: int = 0,
                 samples_per_class: int = 0,
                 superclass: Optional[str] = None,
                 grafit_views: int = 0,
                 grafit_bank: bool = False,
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
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.grafit_views = grafit_views
        self.grafit_bank = grafit_bank
        self.seed = seed

        self.train_classes: List[str] = []
        self.test_classes: List[List[str]] = []
        self._train_labels: List[int] = []
        self._val_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_train_classes(self) -> int:
        return len(self.train_classes)

    @property
    def num_test_classes(self) -> List[int]:
        return [len(classes) for classes in self.test_classes]

    @property
    def use_pk_sampler(self) -> bool:
        return self.classes_per_batch > 0 and self.samples_per_class > 0

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

        self._train_labels = [s[1] for s in train_samples]
        self._val_labels = [s[1] for s in val_samples]

        transform = self._build_transform()
        self.train_dataset = InatContrastiveDataset(
            train_samples, self._train_transform(transform),
            num_views=max(self.grafit_views, 1),
            return_index=self.grafit_bank)
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
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._train_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
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
        # Always use a plain full-coverage loader for validation. The PK sampler
        # oversamples small classes with replacement (duplicate images) and
        # restricts each batch to a few classes, which trivially inflates the
        # leave-one-out val_knn_acc. Evaluating every unique image once against
        # all classes gives an honest metric.
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
                 classes_per_batch: int = 0,
                 samples_per_class: int = 0,
                 download: bool = False,
                 grafit_views: int = 0,
                 grafit_bank: bool = False,
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
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.download = download
        self.grafit_views = grafit_views
        self.grafit_bank = grafit_bank
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
        self._train_labels: List[int] = []
        self._val_labels: List[int] = []
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def num_train_classes(self) -> int:
        return len(self.train_classes)

    @property
    def num_test_classes(self) -> List[int]:
        return [len(classes) for classes in self.test_classes]

    @property
    def use_pk_sampler(self) -> bool:
        return self.classes_per_batch > 0 and self.samples_per_class > 0

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

        self._train_labels = [s[1] for s in train_samples]
        self._val_labels = [s[1] for s in val_samples]

        transform = self._build_transform()
        self.train_dataset = InatContrastiveDataset(
            train_samples, self._train_transform(transform),
            num_views=max(self.grafit_views, 1),
            return_index=self.grafit_bank)
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
        if self.use_pk_sampler:
            batch_sampler = PKBatchSampler(
                labels=self._train_labels,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
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
