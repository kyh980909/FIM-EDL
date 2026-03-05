from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable

from torch.utils.data import DataLoader


@dataclass
class NormalizationSpec:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class DatasetAdapter(ABC):
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def class_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def normalization_spec(self) -> NormalizationSpec:
        raise NotImplementedError

    @abstractmethod
    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        raise NotImplementedError

    @abstractmethod
    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        raise NotImplementedError
