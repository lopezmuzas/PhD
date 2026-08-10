"""Generadores de datos sinteticos y particionado entre nodos.

Todo con NumPy puro: sin scikit-learn, sin descargas. La imagen Docker se queda
en ~150 MB y los labs arrancan sin red.

Todo recibe `seed`: un experimento federado que no es reproducible no se puede
depurar.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Generadores
# --------------------------------------------------------------------------


def make_linear(n: int = 300, w: float = 3.0, b: float = 2.0,
                noise: float = 0.5, x_range: tuple[float, float] = (-5, 5),
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y = w*x + b + ruido. Ejemplo 01: dos parametros, verdad conocida."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_range[0], x_range[1], size=(n, 1))
    y = w * X[:, 0] + b + rng.normal(0, noise, size=n)
    return X, y


def make_blobs(n: int = 600, n_features: int = 2, separation: float = 2.5,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dos gaussianas separadas -> clasificacion binaria. Ejemplo 02."""
    rng = np.random.default_rng(seed)
    half = n // 2
    center = np.zeros(n_features)
    center[0] = separation
    X = np.vstack([
        rng.normal(-center / 2, 1.0, size=(half, n_features)),
        rng.normal(+center / 2, 1.0, size=(n - half, n_features)),
    ])
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    order = rng.permutation(n)
    return X[order], y[order]


def make_patients(n: int = 900, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tabla clinica ficticia de 6 variables. Ejemplo 03.

    Columnas: edad, imc, presion, glucosa, colesterol, fumador.
    Riesgo = combinacion lineal + ruido, umbralizado. Datos 100% inventados.
    """
    rng = np.random.default_rng(seed)
    edad = rng.normal(55, 15, n)
    imc = rng.normal(27, 5, n)
    presion = rng.normal(130, 18, n)
    glucosa = rng.normal(100, 25, n)
    colesterol = rng.normal(200, 40, n)
    fumador = rng.binomial(1, 0.3, n).astype(float)

    X = np.column_stack([edad, imc, presion, glucosa, colesterol, fumador])
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)  # estandarizar: GD lo agradece

    logit = (0.9 * Xs[:, 0] + 0.7 * Xs[:, 1] + 0.5 * Xs[:, 2]
             + 0.8 * Xs[:, 3] + 0.3 * Xs[:, 4] + 0.6 * Xs[:, 5])
    p = 1 / (1 + np.exp(-(logit + rng.normal(0, 0.4, n))))
    y = (p > 0.5).astype(float)
    return Xs, y


# --------------------------------------------------------------------------
# Particionado entre nodos
# --------------------------------------------------------------------------


def split_iid(X, y, n_nodes: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparto aleatorio uniforme. El caso facil: cada nodo ve lo mismo."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return [(X[part], y[part]) for part in np.array_split(idx, n_nodes)]


def split_dirichlet(X, y, n_nodes: int, alpha: float = 0.5,
                    seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparto NO-IID controlado por `alpha`. La perilla mas util del laboratorio.

        alpha = 100  -> practicamente IID
        alpha = 1.0  -> desbalanceo realista
        alpha = 0.1  -> patologico (un nodo casi solo ve una clase)

    Es el metodo estandar en la literatura de FL para simular heterogeneidad,
    y es lo que hace que el federado deje de ser trivial.
    """
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_nodes)]

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_nodes))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for node, chunk in enumerate(np.split(idx, cuts)):
            buckets[node].extend(chunk.tolist())

    out = []
    for b in buckets:
        if not b:  # un nodo vacio rompe FedAvg: mejor fallar aqui que en la ronda 3
            raise ValueError(
                f"alpha={alpha} dejo un nodo sin datos. Sube alpha o baja n_nodes."
            )
        b = np.array(b)
        rng.shuffle(b)
        out.append((X[b], y[b]))
    return out


def train_test_split(X, y, test_frac: float = 0.2, seed: int = 0):
    """Holdout para el orquestador. Nunca se reparte entre nodos."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]
