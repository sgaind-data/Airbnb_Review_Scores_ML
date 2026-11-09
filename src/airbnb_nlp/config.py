"""Typed access to configs/data.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    processed_dir: str
    date_format: str
    low_max: float
    high_min: float
    min_words: int
    max_per_listing: int
    per_class: int
    test_size: float
    val_size: float
    seed: int

    def __post_init__(self) -> None:
        if not self.low_max < self.high_min:
            raise ValueError("low_max must be below high_min")
        if not 0 < self.test_size + self.val_size < 1:
            raise ValueError("test_size + val_size must be between 0 and 1")


def load_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    with open(path) as f:
        return DataConfig(**yaml.safe_load(f))
