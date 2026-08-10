"""ENTRYPOINT del contenedor C2D. El unico fichero que conoce Ocean.

Esto es el "composition root": el sitio donde se decide QUE implementacion
concreta se enchufa a cada puerto. Todo lo demas recibe sus dependencias ya
construidas y no sabe de donde vienen.

Se configura por variables de entorno (`algorithm.envs` en la peticion de
compute):

    FL_ENV=c2d|local     donde estamos (por defecto: c2d)
    FL_MODEL=linear|logistic
    FL_FEATURES=6        numero de columnas de entrada
    FL_DATA_FILE=...     opcional, ruta explicita al dataset
    FL_EPOCHS, FL_LR, FL_SEED   opcionales; algoCustomData tiene prioridad

Salida: /data/outputs/update.npz + /data/outputs/metrics.json
"""

from __future__ import annotations

import os
import sys
import traceback

from ..adapters.sources import CsvSource
from ..adapters.stores import C2DStore, FileStore, find_dataset_file
from ..config import TrainConfig
from ..domain.round import run_local_round
from ..learners.linear import LinearRegressor
from ..learners.logistic import LogisticRegressor

LEARNERS = {"linear": LinearRegressor, "logistic": LogisticRegressor}


def build_learner():
    """Factory minima. Anadir un modelo = una linea en LEARNERS."""
    name = os.getenv("FL_MODEL", "logistic")
    if name not in LEARNERS:
        raise ValueError(f"FL_MODEL={name!r} desconocido. Opciones: {sorted(LEARNERS)}")
    return LEARNERS[name](n_features=int(os.getenv("FL_FEATURES", "1")))


# --8<-- [start:composition-root]
def build_store():
    if os.getenv("FL_ENV", "c2d") == "local":
        return FileStore(os.getenv("FL_INPUT_DIR", "./work/inputs"),
                         os.getenv("FL_OUTPUT_DIR", "./work/outputs"))
    return C2DStore()
# --8<-- [end:composition-root]


def build_config(store) -> TrainConfig:
    """Prioridad: algoCustomData > variables de entorno > valores por defecto.

    Motivo: algoCustomData cambia por ronda sin reconstruir la imagen.
    """
    env_cfg = {
        k: t(os.environ[v])
        for k, v, t in [("epochs", "FL_EPOCHS", int), ("lr", "FL_LR", float),
                        ("seed", "FL_SEED", int)]
        if v in os.environ
    }
    custom = store.read_custom_data().get("config", {}) if hasattr(store, "read_custom_data") else {}
    return TrainConfig.from_dict({**env_cfg, **custom})


def main() -> int:
    store = build_store()

    data_file = find_dataset_file(str(store.input_dir))
    if data_file is None:
        print(f"[ERROR] No hay dataset en {store.input_dir}. "
              f"Contenido: {sorted(p.name for p in store.input_dir.glob('*'))}",
              file=sys.stderr)
        return 2

    source = CsvSource(data_file)
    learner = build_learner()
    cfg = build_config(store)

    print(f"[fedlab] dataset={data_file} modelo={os.getenv('FL_MODEL', 'logistic')} cfg={cfg}")
    result = run_local_round(store, source, learner, cfg)

    print(f"[fedlab] OK n={result.num_samples} metricas={result.metrics}")
    print(f"[fedlab] escrito en {store.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Sin red y sin shell: el traceback en stdout es tu unica ventana.
        # El nodo lo recoge en /data/logs/algorithm.log.
        traceback.print_exc()
        sys.exit(1)
