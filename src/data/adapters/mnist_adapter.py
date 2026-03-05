from __future__ import annotations

from typing import Dict, Iterable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.adapters.base import DatasetAdapter, NormalizationSpec


class MNISTAdapter(DatasetAdapter):
    def __init__(self, root: str = "./data") -> None:
        self.root = root
        # Convert grayscale to 3-channel for ResNet-style backbones.
        self._norm = NormalizationSpec(
            mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081)
        )

    def num_classes(self) -> int:
        return 10

    def class_names(self) -> list[str]:
        return [str(i) for i in range(10)]

    def normalization_spec(self) -> NormalizationSpec:
        return self._norm

    def _train_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(self._norm.mean, self._norm.std),
            ]
        )

    def _eval_tf(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(self._norm.mean, self._norm.std),
            ]
        )

    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        train_ds = datasets.MNIST(self.root, train=True, download=True, transform=self._train_tf())
        test_ds = datasets.MNIST(self.root, train=False, download=True, transform=self._eval_tf())
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
            if name == "kmnist":
                ds = datasets.KMNIST(self.root, train=False, download=True, transform=eval_tf)
            elif name == "fashionmnist":
                ds = datasets.FashionMNIST(self.root, train=False, download=True, transform=eval_tf)
            else:
                raise ValueError(f"Unsupported OOD dataset for MNIST ID: {name}")
            out[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return out
