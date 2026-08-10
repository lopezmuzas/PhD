"""El caso de uso: UNA ronda local de entrenamiento.

Estas ~15 lineas son el corazon del sistema y no saben nada de Ocean, de Docker
ni de ficheros. Reciben tres puertos y los orquestan. Por eso el mismo codigo
corre en un test unitario, en tu portatil y dentro de un contenedor C2D sin red.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainConfig
from ..ports import DataSource, Learner, ParameterStore, Params


@dataclass(frozen=True)
class RoundResult:
    params: Params
    num_samples: int
    metrics: dict


# --8<-- [start:caso-de-uso]
def run_local_round(
    store: ParameterStore,
    source: DataSource,
    learner: Learner,
    cfg: TrainConfig,
) -> RoundResult:
    """Cargar pesos globales -> entrenar en local -> publicar actualizacion."""
    X, y = source.load()

    global_params = store.load_global()
    if global_params is not None:
        learner.set_params(global_params)  # continuamos la ronda anterior

    metrics = learner.fit(X, y, cfg)
    params = learner.get_params()

    store.save_update(params, {"num_samples": len(X), **metrics})
    return RoundResult(params=params, num_samples=len(X), metrics=metrics)
# --8<-- [end:caso-de-uso]
