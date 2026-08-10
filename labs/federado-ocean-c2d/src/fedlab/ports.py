"""Puertos: los contratos que el dominio necesita del mundo exterior.

Regla de oro (DIP): el dominio depende de ESTAS abstracciones, nunca de
implementaciones concretas. Ningun modulo de `domain/` importa `os`, `requests`
ni nada de Ocean.

Son `Protocol` (tipado estructural): un adaptador no necesita heredar de nada,
solo tener los metodos. Eso es ISP llevado al extremo -- interfaces de 2 metodos.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# Un modelo es una lista ordenada de arrays. Nada mas.
# Simple y estupido: sirve para NumPy, PyTorch o scikit-learn por igual.
Params = list[np.ndarray]


# --8<-- [start:puertos]
@runtime_checkable
class ParameterStore(Protocol):
    """De donde vienen los pesos globales y a donde van las actualizaciones.

    Implementaciones: InMemoryStore (tests), FileStore (lab local),
    C2DStore (dentro del contenedor de Ocean).
    """

    def load_global(self) -> Params | None:
        """Pesos de la ronda actual. `None` en la ronda 0 (arranque en frio)."""
        ...

    def save_update(self, params: Params, meta: dict) -> None:
        """Publica el resultado del entrenamiento local."""
        ...


@runtime_checkable
class DataSource(Protocol):
    """De donde salen los datos locales del nodo."""

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (X, y). X con forma (n, d), y con forma (n,)."""
        ...


@runtime_checkable
class Learner(Protocol):
    """Un modelo entrenable. La unica parte que sabe de gradientes."""

    def get_params(self) -> Params: ...

    def set_params(self, params: Params) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray, cfg) -> dict:
        """Entrena in-place. Devuelve metricas escalares serializables."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Metricas sobre datos no vistos. Sin efectos secundarios."""
        ...
# --8<-- [end:puertos]
