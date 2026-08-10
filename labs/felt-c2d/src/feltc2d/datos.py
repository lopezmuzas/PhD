"""Generadores de datos sinteticos. Sin descargas, sin dependencias externas."""
from __future__ import annotations

import numpy as np


def recta_por_tramos(n_por_nodo=50, tramos=((0, 3), (3, 6), (6, 10)), ruido=0.5, semilla=0):
    """y = 3x + 2. Cada nodo ve un tramo DISJUNTO del eje X.

    Ninguno puede estimar bien la recta en solitario: la pendiente local esta
    mal condicionada cuando el rango de X es estrecho.
    """
    rng = np.random.default_rng(semilla)
    fuera = []
    for lo, hi in tramos:
        X = rng.uniform(lo, hi, size=(n_por_nodo, 1))
        y = 3 * X[:, 0] + 2 + rng.normal(0, ruido, n_por_nodo)
        fuera.append((X, y))
    return fuera


def hospitales(tamanos=(200, 60, 20), n_features=10, escala=2.5, semilla=7):
    """Tres centros con MUY distinto numero de pacientes y la misma patologia.

    Devuelve (particiones, (X_test, y_test)).
    """
    rng = np.random.default_rng(semilla)
    w = rng.normal(0, escala, n_features)
    b = 0.3

    def muestrear(n):
        X = rng.normal(0, 1, size=(n, n_features))
        p = 1 / (1 + np.exp(-(X @ w + b)))
        return X, (rng.uniform(size=n) < p).astype(int)

    particiones = [muestrear(n) for n in tamanos]
    X_test, y_test = muestrear(2000)
    return particiones, (X_test, y_test)


def reparto_dirichlet(X, y, n_nodos=4, alpha=1.0, semilla=0):
    """Reparte por clase con proporciones Dirichlet. alpha bajo = mas sesgo."""
    rng = np.random.default_rng(semilla)
    idx_por_nodo = [[] for _ in range(n_nodos)]
    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * n_nodos)
        cortes = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, trozo in enumerate(np.split(idx, cortes)):
            idx_por_nodo[i].extend(trozo.tolist())
    return [(X[np.array(i)], y[np.array(i)]) for i in idx_por_nodo if len(i) > 1]
