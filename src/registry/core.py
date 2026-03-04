from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, TypeVar


T = TypeVar("T")


class RegistryError(ValueError):
    pass


@dataclass
class Registry(Generic[T]):
    name: str

    def __post_init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            if key in self._items:
                raise RegistryError(f"Duplicate registration in {self.name}: '{key}'")
            self._items[key] = item
            return item

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            candidates = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise RegistryError(
                f"Unknown key '{key}' for registry '{self.name}'. Candidates: {candidates}"
            )
        return self._items[key]

    def keys(self) -> Iterable[str]:
        return self._items.keys()
