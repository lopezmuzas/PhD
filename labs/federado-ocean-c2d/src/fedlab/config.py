"""Configuracion de entrenamiento. Un dataclass inmutable, sin magia."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 0.05
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> TrainConfig:
        """Construye desde JSON externo ignorando claves desconocidas.

        Necesario porque `algoCustomData.json` lo escribe el orquestador y puede
        traer campos que este cliente no entiende (o venir de otra version).
        """
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
