"""Utilidades de visualización para la serie de notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def mostrar_digitos(X, y=None, n: int = 10, pred=None, titulo: str | None = None):
    """Rejilla de imágenes de MNIST. X puede ser (n,784) o (n,28,28)."""
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)
    n = min(n, len(X))
    cols = min(n, 10)
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(1.2 * cols, 1.4 * filas))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < n:
            ax.imshow(X[i], cmap="gray_r", interpolation="nearest")
            if y is not None:
                etiqueta = f"{int(y[i])}"
                if pred is not None:
                    etiqueta = f"{int(pred[i])}≠{int(y[i])}" if pred[i] != y[i] else etiqueta
                ax.set_title(etiqueta, fontsize=9)
        ax.axis("off")
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def curva_aprendizaje(historiales: dict, clave: str = "acierto_eval", ylabel="acierto"):
    """Compara varias curvas. `historiales` es {etiqueta: objeto historial}."""
    plt.figure(figsize=(8, 4.5))
    for etiqueta, h in historiales.items():
        serie = getattr(h, clave)
        plt.plot(range(1, len(serie) + 1), serie, marker="o", ms=3, label=etiqueta)
    plt.xlabel("época")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def matriz_confusion(y_true, y_pred, n_clases: int = 10):
    """Matriz de confusión + figura. Devuelve (matriz, figura)."""
    M = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_xticks(range(n_clases))
    ax.set_yticks(range(n_clases))
    for i in range(n_clases):
        for j in range(n_clases):
            if M[i, j]:
                ax.text(
                    j, i, M[i, j], ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.max() / 2 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return M, fig


def mostrar_pesos(red, capa: int = 0, n: int = 30):
    """Dibuja como imágenes los pesos entrantes de las neuronas ocultas.

    Cada neurona de la primera capa oculta tiene 784 pesos, uno por píxel:
    reordenados en 28×28 se ven como el "patrón" que esa neurona detecta.
    """
    W = red.pesos[capa]
    n = min(n, W.shape[0])
    cols = 10
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(1.2 * cols, 1.3 * filas))
    lim = np.abs(W[:n]).max()
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < n:
            ax.imshow(W[i].reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.axis("off")
    fig.suptitle("Pesos entrantes de las neuronas ocultas")
    plt.tight_layout()
    return fig
