"""Implementación desde cero siguiendo el libro de Michael Nielsen.

Ver notebooks/ (distribuidos por temática) para la serie explicada paso a paso.
"""

from dllab.nielsen.data import load_mnist, load_mnist_arrays, one_hot, submuestra
from dllab.nielsen.improved import (
    CosteCuadratico,
    CosteEntropiaCruzada,
    RedMejorada,
)
from dllab.nielsen.network import Red, comprobar_gradiente, sigmoide, sigmoide_prima

__all__ = [
    "load_mnist",
    "load_mnist_arrays",
    "one_hot",
    "submuestra",
    "Red",
    "RedMejorada",
    "CosteCuadratico",
    "CosteEntropiaCruzada",
    "sigmoide",
    "sigmoide_prima",
    "comprobar_gradiente",
]
