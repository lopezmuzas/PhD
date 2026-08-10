"""LAB 1 -- El bucle federado completo con CARPETAS, sin Docker y sin Ocean.

Ejecuta exactamente el mismo `run_local_round` y el mismo `fedavg` que usaran
los labs 2 y 3. Lo unico que cambia es el adaptador: FileStore en vez de
C2DStore. Si esto funciona, la logica federada es correcta y cualquier fallo
posterior sera de infraestructura.

Estructura que crea (imita /data de Ocean):

    work/
      ronda_01/
        hospital_a/  inputs/global.npz   outputs/update.npz  outputs/metrics.json
        hospital_b/  ...

    python scripts/run_lab1.py --rounds 15

Abre los ficheros. Toca el global.npz a mano. Rompelo. Es el objetivo del lab:
ver el flujo de datos con tus propios ojos antes de meterlo en un contenedor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fedlab.adapters.sources import CsvSource
from fedlab.adapters.stores import FileStore
from fedlab.config import TrainConfig
from fedlab.domain.aggregation import fedavg, params_delta
from fedlab.domain.round import run_local_round
from fedlab.learners.logistic import LogisticRegressor
from fedlab.serialization import load_params_npz, read_json, save_params_npz

NODES = ["hospital_a", "hospital_b", "hospital_c"]
N_FEATURES = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="carpeta con los CSV")
    ap.add_argument("--work", default="work", help="carpeta de trabajo")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    data_dir, work = Path(args.data), Path(args.work)
    for node in NODES:
        csv = data_dir / f"ex03_{node}.csv"
        if not csv.exists():
            raise SystemExit(f"Falta {csv}. Ejecuta antes: python scripts/make_datasets.py")

    cfg = TrainConfig(epochs=40, lr=0.4)
    global_params = None

    print(f"{'ronda':>6} {'muestras':>9} {'accuracy media':>15} {'delta global':>13}")
    print("-" * 48)

    for r in range(1, args.rounds + 1):
        round_dir = work / f"ronda_{r:02d}"
        updates, accs = [], []

        for node in NODES:
            node_dir = round_dir / node
            inputs, outputs = node_dir / "inputs", node_dir / "outputs"
            inputs.mkdir(parents=True, exist_ok=True)

            # 1. El "orquestador" deposita los pesos globales en /inputs
            if global_params is not None:
                save_params_npz(inputs / FileStore.GLOBAL_FILE, global_params)

            # 2. El "contenedor" entrena y escribe en /outputs
            store = FileStore(inputs, outputs)
            source = CsvSource(data_dir / f"ex03_{node}.csv")
            run_local_round(store, source, LogisticRegressor(N_FEATURES), cfg)

            # 3. El orquestador recoge lo que hay en /outputs
            params = load_params_npz(outputs / FileStore.UPDATE_FILE)
            meta = read_json(outputs / FileStore.META_FILE)
            updates.append((params, int(meta["num_samples"])))
            accs.append(meta["accuracy"])

        previous = global_params
        global_params = fedavg(updates)
        total = sum(n for _, n in updates)
        delta = params_delta(global_params, previous) if previous else float("nan")
        print(f"{r:>6} {total:>9} {np.mean(accs):>15.4f} {delta:>13.5f}")

    save_params_npz(work / "modelo_final.npz", global_params)
    print(f"\nModelo final en {work / 'modelo_final.npz'}")
    print(f"Inspecciona el flujo completo en {work}/ronda_01/hospital_a/")


if __name__ == "__main__":
    main()
