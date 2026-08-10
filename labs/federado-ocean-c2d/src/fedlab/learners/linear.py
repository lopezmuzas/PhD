"""Regresion lineal por descenso de gradiente. Ejemplo 01.

DOS parametros: w y b. Puedes imprimirlos en cada ronda y ver a ojo como se
acercan a la verdad. Ese es todo el objetivo pedagogico.
"""

from __future__ import annotations

import numpy as np

from ..config import TrainConfig
from ..ports import Params


class LinearRegressor:
    """Cumple el puerto `Learner`. No hereda de nada (tipado estructural)."""

    def __init__(self, n_features: int = 1):
        self.w = np.zeros(n_features)
        self.b = np.zeros(1)

    # -- puerto Learner ----------------------------------------------------
    def get_params(self) -> Params:
        return [self.w.copy(), self.b.copy()]

    def set_params(self, params: Params) -> None:
        w, b = params
        if w.shape != self.w.shape:
            raise ValueError(f"Se esperaba w{self.w.shape}, llego w{w.shape}.")
        self.w = np.asarray(w, dtype=np.float64).copy()
        self.b = np.asarray(b, dtype=np.float64).copy()

    def fit(self, X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> dict:
        n = len(X)
        for _ in range(cfg.epochs):
            error = self._predict(X) - y
            self.w -= cfg.lr * (X.T @ error) / n
            self.b -= cfg.lr * error.mean()
        return self.evaluate(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        mse = float(np.mean((self._predict(X) - y) ** 2))
        return {"mse": mse}

    # -- interno -----------------------------------------------------------
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b[0]
