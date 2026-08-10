"""Modelos minimos en NumPy puro. Sin scikit-learn, sin PyTorch.

La restriccion es deliberada: en este laboratorio queremos ver el protocolo,
no el framework. Cada modelo expone la misma interfaz para que el protocolo no
sepa cual esta usando.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegresionLineal:
    """Minimos cuadrados en forma cerrada (ecuaciones normales con ridge minimo).

    Forma cerrada y no descenso de gradiente a proposito: el laboratorio trata
    del protocolo, no de la optimizacion. Sin tasa de aprendizaje que ajustar,
    el resultado es determinista y las cifras de la documentacion se reproducen.
    """

    n_features: int
    ridge: float = 1e-8
    pesos: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pesos:
            self.pesos = {
                "w": np.zeros(self.n_features, dtype=float),
                "b": np.zeros(1, dtype=float),
            }

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        A = np.hstack([X, np.ones((len(X), 1))])
        G = A.T @ A + self.ridge * np.eye(A.shape[1])
        coef = np.linalg.solve(G, A.T @ y)
        self.pesos = {"w": coef[:-1].copy(), "b": np.array([coef[-1]])}

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return X @ self.pesos["w"] + self.pesos["b"]

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predecir(X) - y) ** 2))


@dataclass
class RegresionLogistica:
    n_features: int
    pasos: int = 3000
    lr: float = 0.5
    pesos: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pesos:
            self.pesos = {
                "w": np.zeros(self.n_features, dtype=float),
                "b": np.zeros(1, dtype=float),
            }

    def _p(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.pesos["w"] + self.pesos["b"]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        for _ in range(self.pasos):
            err = self._p(X) - y
            self.pesos["w"] -= self.lr * (X.T @ err) / n
            self.pesos["b"] -= self.lr * err.mean()

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return (self._p(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predecir(X) == y))


MODELOS = {"lineal": RegresionLineal, "logistica": RegresionLogistica}


def construir(tipo: str, n_features: int, pesos: dict | None = None):
    cls = MODELOS[tipo]
    m = cls(n_features=n_features)
    if pesos:
        m.pesos = {k: np.asarray(v, dtype=float) for k, v in pesos.items()}
    return m
