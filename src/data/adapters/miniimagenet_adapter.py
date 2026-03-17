from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.adapters.base import DatasetAdapter, NormalizationSpec


class MiniImageNetAdapter(DatasetAdapter):
    """Folder-based adapter for mini-ImageNet/CUB experiments.

    Expected layout under root:
    - miniimagenet/train/<class>/*.jpg
    - miniimagenet/val/<class>/*.jpg
    - miniimagenet/test/<class>/*.jpg
    - cub/test/<class>/*.jpg
    """

    def __init__(self, root: str = "./data") -> None:
        self.root = Path(root)
        self._norm = NormalizationSpec(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def num_classes(self) -> int:
        return 100

    def class_names(self) -> list[str]:
        train_dir = self.root / "miniimagenet" / "train"
        if not train_dir.exists():
            return []
        ds = datasets.ImageFolder(train_dir)
        return [name for name, _ in sorted(ds.class_to_idx.items(), key=lambda x: x[1])]

    def normalization_spec(self) -> NormalizationSpec:
        return self._norm

    def _train_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(92),
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self._norm.mean, self._norm.std),
            ]
        )

    def _eval_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(self._norm.mean, self._norm.std),
            ]
        )

    def _imagefolder(self, rel: str, train: bool):
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        return datasets.ImageFolder(path, transform=self._train_tf() if train else self._eval_tf())

    def imagefolder_dataset(self, rel: str, train: bool = False):
        return self._imagefolder(rel=rel, train=train)

    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        train_ds = self._imagefolder("miniimagenet/train", train=True)
        val_ds = self._imagefolder("miniimagenet/val", train=False)
        test_ds = self._imagefolder("miniimagenet/test", train=False)
        return {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        }

    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        out: Dict[str, DataLoader] = {}
        for name in names:
            if name != "cub":
                raise ValueError(f"Unsupported OOD dataset for miniImageNet ID: {name}")
            ds = self._imagefolder("cub/test", train=False)
            out[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return out
