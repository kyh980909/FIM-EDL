from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager


@contextmanager
def _without_repo_root(repo_root: str):
    original = list(sys.path)
    try:
        abs_root = os.path.abspath(repo_root)
        filtered = []
        for p in original:
            abs_p = os.path.abspath(p or os.getcwd())
            if abs_p == abs_root:
                continue
            filtered.append(p)
        sys.path[:] = filtered
        yield
    finally:
        sys.path[:] = original


def import_wandb(repo_root: str = "."):
    if "wandb" in sys.modules:
        mod = sys.modules["wandb"]
        if hasattr(mod, "init"):
            return mod
        del sys.modules["wandb"]

    with _without_repo_root(repo_root):
        wandb = importlib.import_module("wandb")

    if not hasattr(wandb, "init"):
        raise RuntimeError("Failed to import official wandb package")
    return wandb
