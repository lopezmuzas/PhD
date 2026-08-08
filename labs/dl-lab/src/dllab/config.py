"""Configuración de experimentos: un YAML por experimento, tipado con dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class DataConfig:
    name: str = "moons"
    n_samples: int = 4000
    noise: float = 0.2
    val_split: float = 0.2
    batch_size: int = 128
    num_workers: int = 0


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    patience: int = 8          # early stopping; 0 = desactivado


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 42
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            name=raw.get("name", "baseline"),
            seed=raw.get("seed", 42),
            device=raw.get("device", "auto"),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat(self) -> dict[str, Any]:
        """Versión plana (model.hidden, train.lr...) para loggear como hiperparámetros."""
        out: dict[str, Any] = {}
        for section, value in self.to_dict().items():
            if isinstance(value, dict):
                for k, v in value.items():
                    out[f"{section}.{k}"] = v
            else:
                out[section] = value
        return out
