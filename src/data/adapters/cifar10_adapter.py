from __future__ import annotations

from typing import Dict, Iterable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.adapters.base import DatasetAdapter, NormalizationSpec


class CIFAR10Adapter(DatasetAdapter):
    def __init__(self, root: str = "./data") -> None:
        self.root = root
        self._norm = NormalizationSpec(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
        )

    def num_classes(self) -> int:
        return 10

    def class_names(self) -> list[str]:
        return list(datasets.CIFAR10(self.root, train=True, download=True).classes)

    def normalization_spec(self) -> NormalizationSpec:
        return self._norm

    def _train_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self._norm.mean, self._norm.std),
            ]
        )

    def _eval_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self._norm.mean, self._norm.std)]
        )

    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        train_ds = datasets.CIFAR10(self.root, train=True, download=True, transform=self._train_tf())
        test_ds = datasets.CIFAR10(self.root, train=False, download=True, transform=self._eval_tf())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return {"train": train_loader, "val": val_loader, "test": test_loader}

    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        eval_tf = self._eval_tf()
        out: Dict[str, DataLoader] = {}
        for name in names:
            if name == "svhn":
                ds = datasets.SVHN(self.root, split="test", download=True, transform=eval_tf)
            elif name == "cifar100":
                ds = datasets.CIFAR100(self.root, train=False, download=True, transform=eval_tf)
            else:
                raise ValueError(f"Unsupported OOD dataset: {name}")
            out[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return out
