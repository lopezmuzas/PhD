"""Simulador federado en memoria. El bucle completo sin Docker ni Ocean.

Es el mismo `run_local_round` y el mismo `fedavg` que usan los labs 2 y 3.
Solo cambia el adaptador de almacenamiento (InMemoryStore). Esa es la prueba de
que la arquitectura hexagonal esta bien montada.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..adapters.sources import ArraySource
from ..adapters.stores import InMemoryStore
from ..config import TrainConfig
from ..domain.aggregation import fedavg
from ..domain.round import run_local_round
from ..ports import Learner, Params


def simulate(
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    learner_factory: Callable[[], Learner],
    cfg: TrainConfig,
    rounds: int = 10,
    on_round: Callable[[int, Params, list[dict]], None] | None = None,
) -> Params:
    """Ejecuta `rounds` rondas de FedAvg sobre particiones ya repartidas.

    `learner_factory` crea un modelo NUEVO por nodo y ronda: garantiza que no
    hay estado compartido por accidente entre clientes (el bug silencioso mas
    comun al simular FL en un solo proceso).
    """
    global_params: Params | None = None

    for r in range(1, rounds + 1):
        updates, metrics = [], []
        for X, y in partitions:
            store = InMemoryStore(initial=global_params)
            result = run_local_round(store, ArraySource(X, y), learner_factory(), cfg)
            updates.append((result.params, result.num_samples))
            metrics.append(result.metrics)

        global_params = fedavg(updates)
        if on_round:
            on_round(r, global_params, metrics)

    return global_params
