"""Regresion logistica binaria por descenso de gradiente. Ejemplos 02 y 03.

El MISMO learner sirve para los dos ejemplos: solo cambia `n_features`. Eso es
la senal de que la abstraccion esta en el sitio correcto.
"""

from __future__ import annotations

import numpy as np

from ..config import TrainConfig
from ..ports import Params


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Estable numericamente: exp(-|z|) nunca desborda."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressor:
    def __init__(self, n_features: int):
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
            error = self.predict_proba(X) - y
            self.w -= cfg.lr * (X.T @ error) / n
            self.b -= cfg.lr * error.mean()
        return self.evaluate(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        p = self.predict_proba(X)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        acc = float(np.mean((p > 0.5).astype(float) == y))
        return {"loss": loss, "accuracy": acc}

    # -- interno -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w + self.b[0])
