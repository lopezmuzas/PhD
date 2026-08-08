"""Carga de MNIST en el formato que usa la serie de notebooks de Nielsen.

El libro trabaja con listas de tuplas `(x, y)` donde cada `x` es un vector
columna de 784x1 y cada `y`, en el conjunto de entrenamiento, es un vector
columna de 10x1 codificado one-hot. Puede parecer un capricho, pero tiene
una razón didáctica: obliga a que la aritmética matricial del backpropagation
sea explícita, sin que numpy tape errores de dimensiones con broadcasting.

Descargamos los ficheros con torchvision (que ya está en la imagen) y a partir
de ahí trabajamos solo con numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Ejemplo = tuple[np.ndarray, np.ndarray]
Conjunto = list[Ejemplo]

DEFAULT_ROOT = Path("data/raw")


def _descargar(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST

    root.mkdir(parents=True, exist_ok=True)
    tr = MNIST(str(root), train=True, download=True)
    te = MNIST(str(root), train=False, download=True)

    X_tr = tr.data.numpy().astype(np.float64) / 255.0     # (60000, 28, 28)
    y_tr = tr.targets.numpy().astype(np.int64)
    X_te = te.data.numpy().astype(np.float64) / 255.0
    y_te = te.targets.numpy().astype(np.int64)
    return X_tr, y_tr, X_te, y_te


def one_hot(j: int, n: int = 10) -> np.ndarray:
    """Vector columna de n×1 con un 1 en la posición j.

    Es la respuesta "ideal" que queremos que produzca la capa de salida.
    """
    v = np.zeros((n, 1))
    v[j] = 1.0
    return v


def load_mnist_arrays(root: str | Path = DEFAULT_ROOT, n_val: int = 10_000):
    """Devuelve los datos como arrays planos: (X, y) por partición.

    X tiene forma (n, 784) con valores en [0, 1]; y es un vector de enteros.
    Útil para graficar y para comparar con scikit-learn o PyTorch.
    """
    X_tr, y_tr, X_te, y_te = _descargar(Path(root))
    X_tr = X_tr.reshape(len(X_tr), -1)
    X_te = X_te.reshape(len(X_te), -1)

    # Nielsen reserva 10.000 de los 60.000 de entrenamiento para validación.
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    X_tr, y_tr = X_tr[:-n_val], y_tr[:-n_val]
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def load_mnist(
    root: str | Path = DEFAULT_ROOT, n_val: int = 10_000
) -> tuple[Conjunto, Conjunto, Conjunto]:
    """Devuelve (entrenamiento, validación, test) como listas de tuplas.

    - entrenamiento: (x de 784×1, y one-hot de 10×1)
    - validación y test: (x de 784×1, dígito entero)

    La asimetría es deliberada: durante el entrenamiento necesitamos el vector
    objetivo completo para calcular el error; para evaluar solo necesitamos
    saber si la neurona más activa coincide con el dígito correcto.
    """
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_mnist_arrays(root, n_val)

    entrenamiento = [(x.reshape(784, 1), one_hot(int(j))) for x, j in zip(X_tr, y_tr)]
    validacion = [(x.reshape(784, 1), int(j)) for x, j in zip(X_val, y_val)]
    test = [(x.reshape(784, 1), int(j)) for x, j in zip(X_te, y_te)]
    return entrenamiento, validacion, test


def submuestra(datos: Conjunto, n: int, seed: int = 0) -> Conjunto:
    """Extrae n ejemplos al azar. Se usa para las demos de sobreajuste."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(datos), size=n, replace=False)
    return [datos[i] for i in idx]
