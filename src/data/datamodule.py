from __future__ import annotations

from typing import Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.adapters.base import DatasetAdapter
from src.data.adapters.cifar10_adapter import CIFAR10Adapter
from src.data.adapters.miniimagenet_adapter import MiniImageNetAdapter
from src.data.adapters.mnist_adapter import MNISTAdapter


def _build_adapter(cfg) -> DatasetAdapter:
    id_name = cfg.data.id
    root = cfg.data.root
    name = str(id_name).lower()
    if name == "cifar10":
        return CIFAR10Adapter(
            root=root,
            val_from_train=bool(cfg.data.val_from_train),
            val_split=float(cfg.data.val_split),
            seed=int(cfg.seed),
            normalize=bool(cfg.data.normalize),
            random_rotation_degrees=float(cfg.data.random_rotation_degrees),
            val_use_train_transform=bool(cfg.data.val_use_train_transform),
        )
    if name == "mnist":
        return MNISTAdapter(
            root=root,
            val_from_train=bool(cfg.data.val_from_train),
            val_split=float(cfg.data.val_split),
            seed=int(cfg.seed),
            normalize=bool(cfg.data.normalize),
            image_size=int(cfg.data.image_size),
            grayscale_to_rgb=bool(cfg.data.grayscale_to_rgb),
            random_crop_padding=int(cfg.data.random_crop_padding),
        )
    if name in {"miniimagenet", "mini-imagenet"}:
        return MiniImageNetAdapter(root=root)
    raise ValueError(f"Unsupported data.id: {id_name}. Supported: cifar10, mnist, miniimagenet")


class InfoEDLDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = _build_adapter(cfg)
        self._id: Dict[str, DataLoader] = {}
        self._ood: Dict[str, DataLoader] = {}

    def setup(self, stage: str | None = None) -> None:
        if not self._id:
            self._id = self.adapter.id_dataloaders(
                batch_size=self.cfg.data.batch_size,
                num_workers=self.cfg.data.num_workers,
            )

        should_load_ood = stage in {None, "predict"}
        if should_load_ood and not self._ood:
            self._ood = self.adapter.ood_dataloaders(
                names=self.cfg.data.ood_list,
                batch_size=self.cfg.data.batch_size,
                num_workers=self.cfg.data.num_workers,
            )

    def train_dataloader(self) -> DataLoader:
        return self._id["train"]

    def val_dataloader(self) -> DataLoader:
        return self._id["val"]

    def test_dataloader(self) -> DataLoader:
        return self._id["test"]

    def ood_dataloaders(self) -> Dict[str, DataLoader]:
        return self._ood
