"""Adaptadores del puerto `DataSource`.

CsvSource lee lo que el nodo Ocean deja en /data/inputs.
SyntheticSource genera datos al vuelo: util para probar la imagen sin dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


class CsvSource:
    """CSV con cabecera; la ULTIMA columna es la etiqueta.

    Formato deliberadamente rigido. Un contrato aburrido y explicito ahorra
    horas de depuracion dentro de un contenedor al que no puedes entrar.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.path.exists():
            raise FileNotFoundError(f"No existe el dataset: {self.path}")
        data = np.genfromtxt(self.path, delimiter=",", skip_header=1, dtype=np.float64)
        if data.ndim == 1:  # una sola fila
            data = data.reshape(1, -1)
        if np.isnan(data).any():
            raise ValueError(f"{self.path} contiene NaN o celdas no numericas.")
        if data.shape[1] < 2:
            raise ValueError(f"{self.path} necesita >= 2 columnas (features + etiqueta).")
        return data[:, :-1], data[:, -1]


class SyntheticSource:
    """Genera datos con una funcion inyectada. Cero ficheros, cero red."""

    def __init__(self, generator: Callable[[], tuple[np.ndarray, np.ndarray]]):
        self._generator = generator

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return self._generator()


class ArraySource:
    """Envuelve arrays que ya tienes en memoria. Para simulacion y tests."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


def write_csv(path: str | Path, X: np.ndarray, y: np.ndarray) -> None:
    """Escribe en el formato que espera CsvSource. Usado por los generadores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["label"])
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=header,
               comments="", fmt="%.6f")
