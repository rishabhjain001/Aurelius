"""
data.py — Flickr30K data pipeline for Aurelius CLIP-VLM

Handles dataset loading, deterministic train/validation split, image transforms
(weak and strong augmentation pipelines), caption sampling, and DataLoader
construction with a caption-aware collate function.

Public interface:
    - Flickr30KDataset
    - make_loaders(config, hf_dataset=None) -> (train_loader, val_loader)

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 25.3
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset rejection helper
# ---------------------------------------------------------------------------

def _check_dataset_name(name: str) -> None:
    """Reject any dataset other than flickr30k (case-insensitive) per Req 2.8."""
    if name.lower() != "flickr30k":
        msg = (
            f"Only 'flickr30k' is supported, got '{name}'. "
            "Active dataset unchanged."
        )
        logger.error(msg)
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def _split_indices(
    n_total: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic disjoint split. Same seed -> identical (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_total)
    rng.shuffle(all_idx)
    n_val = int(round(n_total * val_fraction))
    val_idx = sorted(all_idx[:n_val].tolist())
    train_idx = sorted(all_idx[n_val:].tolist())
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Flickr30KDataset(Dataset):
    """Flickr30K image-caption dataset backed by any list-like of dict rows.

    Each row must contain a PIL image at key "image" and either a list of five
    captions at "captions" or a single/list caption at "caption".

    is_train=True samples one of the five captions uniformly at random each call
    (respects the global Python `random` seed). is_train=False always uses the
    first caption for deterministic evaluation.
    """

    def __init__(self, hf_dataset, indices, transform, is_train: bool) -> None:
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        item = self.hf_dataset[self.indices[idx]]

        if "captions" in item:
            captions = item["captions"]
        elif "caption" in item:
            captions = item["caption"]
            if not isinstance(captions, list):
                captions = [captions]
        else:
            raise KeyError(
                "Dataset row missing 'captions' or 'caption' field."
            )

        if self.is_train and len(captions) > 1:
            caption = captions[random.randint(0, len(captions) - 1)]
        else:
            caption = captions[0]

        image_tensor = self.transform(item["image"])
        return image_tensor, caption


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_base_transform(image_size, mean, std) -> transforms.Compose:
    """Deterministic transform used for validation: resize + tensor + normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def _get(cfg, key, default=None):
    """Attribute-or-mapping getter so config sub-objects behave uniformly."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def _build_train_transform(config, mode: str) -> transforms.Compose:
    """Build train-time transform for either 'weak' or 'strong' augmentation."""
    if mode not in ("weak", "strong"):
        raise ValueError(
            f"augmentation mode must be 'weak' or 'strong', got '{mode}'."
        )

    image_size = config.image_size
    mean = list(config.image_mean)
    std = list(config.image_std)

    aug_cfg = _get(config.augmentation, mode)
    if aug_cfg is None:
        raise ValueError(f"config.augmentation.{mode} is missing.")

    jitter = _get(aug_cfg, "color_jitter", {}) or {}
    ops: list = []

    if mode == "weak":
        ops.append(transforms.Resize((image_size, image_size)))
        if _get(aug_cfg, "random_horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.ColorJitter(
            brightness=_get(jitter, "brightness", 0.0),
            contrast=_get(jitter, "contrast", 0.0),
            saturation=_get(jitter, "saturation", 0.0),
            hue=_get(jitter, "hue", 0.0),
        ))
    else:  # strong
        crop_cfg = _get(aug_cfg, "random_resized_crop", {}) or {}
        scale = tuple(_get(crop_cfg, "scale", [0.08, 1.0]))
        ops.append(transforms.RandomResizedCrop(size=image_size, scale=scale))
        if _get(aug_cfg, "random_horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.ColorJitter(
            brightness=_get(jitter, "brightness", 0.0),
            contrast=_get(jitter, "contrast", 0.0),
            saturation=_get(jitter, "saturation", 0.0),
            hue=_get(jitter, "hue", 0.0),
        ))
        gray_p = _get(aug_cfg, "random_grayscale", 0.0)
        if gray_p:
            ops.append(transforms.RandomGrayscale(p=gray_p))
        blur_cfg = _get(aug_cfg, "gaussian_blur", None)
        if blur_cfg is not None:
            sigma = tuple(_get(blur_cfg, "sigma", [0.1, 2.0]))
            ops.append(transforms.GaussianBlur(kernel_size=3, sigma=sigma))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch):
    """Stack image tensors and keep captions as a Python list of strings."""
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return images, captions


# ---------------------------------------------------------------------------
# make_loaders
# ---------------------------------------------------------------------------

def _load_flickr30k():
    """Load Flickr30K from HuggingFace.

    Tries nlphuji/flickr30k first. If unavailable, downloads the
    clip-benchmark/wds_flickr30k tar shards directly and builds an in-memory
    list of {"image": PIL.Image, "captions": list[str]} dicts.
    """
    from datasets import load_dataset  # local import: avoids startup cost
    try:
        ds = load_dataset("nlphuji/flickr30k", split="test")
        logger.info("Loaded nlphuji/flickr30k (%d rows)", len(ds))
        return ds
    except Exception as first_err:
        logger.warning(
            "nlphuji/flickr30k unavailable (%s); downloading wds tar shards.",
            first_err,
        )
        return _load_flickr30k_from_tars()


def _load_flickr30k_from_tars():
    """Download clip-benchmark/wds_flickr30k tar shards, extract JPEGs to disk,
    and return a lazy _DiskDataset that reads images on demand per __getitem__.

    Only file paths and caption strings are kept in memory (~5MB for 30K rows).
    """
    import tarfile
    import json as _json
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "aurelius_flickr30k"
    img_dir = cache_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.jsonl"

    # Fast path: manifest exists and images are on disk
    if manifest_path.exists():
        logger.info("Loading from disk manifest …")
        rows: list = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                rows.append((Path(rec["path"]), rec["captions"]))
        logger.info("Manifest loaded: %d images", len(rows))
        return _DiskDataset(rows)

    rows = []
    splits_shards = [("train", 5), ("test", 5)]

    with manifest_path.open("w", encoding="utf-8") as manifest_fh:
        for split, shard_count in splits_shards:
            for shard_idx in range(shard_count):
                tar_path = cache_dir / f"{split}_{shard_idx}.tar"
                hf_name = f"{split}/{shard_idx}.tar"

                if not tar_path.exists():
                    logger.info(
                        "Downloading %s shard %d/%d …", split, shard_idx + 1, shard_count
                    )
                    downloaded = hf_hub_download(
                        repo_id="clip-benchmark/wds_flickr30k",
                        filename=hf_name,
                        repo_type="dataset",
                        local_dir=str(cache_dir),
                    )
                    downloaded_path = Path(downloaded)
                    if downloaded_path != tar_path:
                        downloaded_path.rename(tar_path)

                logger.info(
                    "Extracting %s shard %d/%d to disk …", split, shard_idx + 1, shard_count
                )
                with tarfile.open(tar_path, "r") as tf:
                    members = tf.getmembers()
                    by_key: dict = {}
                    for m in members:
                        if "." not in m.name:
                            continue
                        stem, ext = m.name.rsplit(".", 1)
                        if ext in ("jpg", "txt"):
                            by_key.setdefault(stem, {})[ext] = m

                    for stem, parts in sorted(by_key.items()):
                        if "jpg" not in parts or "txt" not in parts:
                            continue
                        try:
                            jpg_f = tf.extractfile(parts["jpg"])
                            txt_f = tf.extractfile(parts["txt"])
                            if jpg_f is None or txt_f is None:
                                continue
                            safe_stem = stem.replace("/", "_").replace("\\", "_")
                            img_path = img_dir / f"{split}_{safe_stem}.jpg"
                            if not img_path.exists():
                                img_path.write_bytes(jpg_f.read())
                            else:
                                jpg_f.read()  # consume to avoid tarfile seek issues
                            text = txt_f.read().decode("utf-8", errors="replace")
                            captions = [c.strip() for c in text.split("\n") if c.strip()]
                            rows.append((img_path, captions))
                            manifest_fh.write(
                                _json.dumps({"path": str(img_path), "captions": captions}) + "\n"
                            )
                        except Exception as exc:
                            logger.debug("Skipping %s/%s: %s", split, stem, exc)

                logger.info(
                    "%s shard %d done — total images on disk: %d",
                    split, shard_idx + 1, len(rows),
                )

    logger.info("Flickr30K extracted to disk: %d images at %s", len(rows), img_dir)
    return _DiskDataset(rows)


class _DiskDataset:
    """Lazy-loading dataset: stores only paths+captions, decodes JPEG per __getitem__."""

    def __init__(self, rows: list) -> None:
        # rows: list of (Path, list[str])
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image as _PIL
        img_path, captions = self._rows[idx]
        image = _PIL.open(img_path).convert("RGB")
        return {"image": image, "captions": captions}


def make_loaders(
    config, hf_dataset: Optional[object] = None
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) from config and optional preloaded dataset."""
    _check_dataset_name(config.dataset)

    if hf_dataset is None:
        hf_dataset = _load_flickr30k()

    mode = getattr(config, "augmentation_mode", "weak")

    train_idx, val_idx = _split_indices(
        len(hf_dataset), config.val_split_fraction, config.val_split_seed
    )

    # Optionally cap the training set for CPU-feasible runs (Req 2.7: val stays fixed).
    max_train = getattr(config, "train_max_samples", None)
    if max_train is not None and max_train < len(train_idx):
        # Use first max_train indices from the already-shuffled train split so
        # the selection is deterministic given the same val_split_seed.
        train_idx = train_idx[:int(max_train)]

    train_transform = _build_train_transform(config, mode)
    val_transform = _build_base_transform(
        config.image_size, config.image_mean, config.image_std
    )

    train_ds = Flickr30KDataset(hf_dataset, train_idx, train_transform, is_train=True)
    val_ds = Flickr30KDataset(hf_dataset, val_idx, val_transform, is_train=False)

    num_workers = getattr(config, "num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    return train_loader, val_loader
