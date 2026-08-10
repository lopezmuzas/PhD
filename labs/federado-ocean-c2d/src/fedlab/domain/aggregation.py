"""Agregacion federada. Funciones puras: entran numeros, salen numeros.

Cero I/O, cero dependencias de framework. Esto es lo que hace que el corazon del
sistema sea testeable en milisegundos.

OCP en accion: anadir FedProx o un agregador robusto = anadir una funcion nueva
aqui. Nada de lo que ya funciona se toca.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ports import Params

# Una actualizacion local = (pesos, numero de muestras con que se entrenaron)
Update = tuple[Params, int]


def _validate(updates: Sequence[Update]) -> None:
    if not updates:
        raise ValueError("No hay actualizaciones que agregar.")
    ref = [np.asarray(p).shape for p in updates[0][0]]
    for i, (params, n) in enumerate(updates):
        if n <= 0:
            raise ValueError(f"Cliente {i}: num_samples debe ser > 0, es {n}.")
        shapes = [np.asarray(p).shape for p in params]
        if shapes != ref:
            raise ValueError(
                f"Cliente {i}: formas {shapes} != referencia {ref}. "
                "Todos los nodos deben usar la misma arquitectura."
            )


# --8<-- [start:fedavg]
def fedavg(updates: Sequence[Update]) -> Params:
    """FedAvg de McMahan et al. (2017): media PONDERADA por numero de muestras.

        w_global = sum_k (n_k / n_total) * w_k

    El error clasico es hacer la media aritmetica simple. Con nodos
    desbalanceados (1000 filas vs 50) el resultado es sencillamente incorrecto:
    da el mismo voto a quien aporta 50 muestras que a quien aporta 1000.
    """
    _validate(updates)
    total = sum(n for _, n in updates)
    n_layers = len(updates[0][0])
    return [
        sum(np.asarray(params[i], dtype=np.float64) * (n / total) for params, n in updates)
        for i in range(n_layers)
    ]

# --8<-- [end:fedavg]


def simple_mean(updates: Sequence[Update]) -> Params:
    """Media sin ponderar. Existe SOLO para demostrar en los tests que difiere
    de fedavg cuando los nodos estan desbalanceados. No la uses en serio."""
    _validate(updates)
    n_layers = len(updates[0][0])
    k = len(updates)
    return [
        sum(np.asarray(params[i], dtype=np.float64) for params, _ in updates) / k
        for i in range(n_layers)
    ]


def params_delta(a: Params, b: Params) -> float:
    """Norma L2 de la diferencia. Util como criterio de parada y para logs."""
    return float(np.sqrt(sum(np.sum((np.asarray(x) - np.asarray(y)) ** 2) for x, y in zip(a, b))))
