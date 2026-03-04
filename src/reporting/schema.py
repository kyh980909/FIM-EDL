from __future__ import annotations

from src.contracts.schemas import RESULTS_SCHEMA_VERSION


def get_results_schema_version() -> str:
    return RESULTS_SCHEMA_VERSION
