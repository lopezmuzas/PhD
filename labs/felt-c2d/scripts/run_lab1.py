"""Lab 1 - el pipeline de dos algoritmos con carpetas locales, sin Docker.

Genera datos, ejecuta el algoritmo local una vez por nodo, ejecuta el
algoritmo de agregacion, y desciega el resultado. Es exactamente lo que hace
la aplicacion de FELT, pero sin Ocean ni blockchain.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d import crypto
from feltc2d.datos import recta_por_tramos
from feltc2d.models import construir
from feltc2d.protocol import fase3_descegar

RAIZ = Path(__file__).resolve().parents[1]
TRABAJO = RAIZ / "trabajo"


def main() -> int:
    if TRABAJO.exists():
        import shutil
        shutil.rmtree(TRABAJO)

    particiones = recta_por_tramos()
    sk, pk = crypto.generar_par_de_claves()
    semillas = [1001, 1002, 1003]

    print("[1/4] escribiendo datasets en el patron de C2D")
    for i, (X, y) in enumerate(particiones):
        did = TRABAJO / f"nodo{i}" / "inputs" / f"did:op:{i:040x}"
        did.mkdir(parents=True)
        np.savetxt(did / "datos.csv", np.hstack([X, y[:, None]]), delimiter=",")
        print(f"      nodo{i}: {len(X)} filas -> {did.relative_to(RAIZ)}")

    print("[2/4] job C2D por nodo: entrenamiento local")
    rutas = []
    for i, s in enumerate(semillas):
        base = TRABAJO / f"nodo{i}"
        cmd = [sys.executable, str(RAIZ / "algorithms" / "local_algorithm.py"),
               "--input_folder", str(base / "inputs"),
               "--output_folder", str(base / "outputs"),
               "--aggregation_key", pk.hex(), "--seed", str(s), "--modelo", "lineal"]
        subprocess.run(cmd, check=True)
        rutas.append(str(base / "outputs" / "model"))

    print("[3/4] job C2D de agregacion")
    agg = TRABAJO / "agregacion"
    (agg / "inputs").mkdir(parents=True)
    (agg / "inputs" / "algoCustomData.json").write_text(json.dumps({"model_urls": rutas}))
    subprocess.run([sys.executable, str(RAIZ / "algorithms" / "aggregation_algorithm.py"),
                    "--input_folder", str(agg / "inputs"),
                    "--output_folder", str(agg / "outputs"),
                    "--private_key", sk.hex()], check=True)

    print("[4/4] descegado en tu maquina")
    global_ = (agg / "outputs" / "model").read_bytes()
    final = fase3_descegar(global_, semillas)
    w, b = final["pesos"]["w"][0], final["pesos"]["b"][0]

    X_all = np.vstack([X for X, _ in particiones])
    y_all = np.concatenate([y for _, y in particiones])
    mse = construir("lineal", 1, final["pesos"]).mse(X_all, y_all)

    print(f"\n   modelo final: w={w:.4f}  b={b:.4f}   (verdad: 3.0 / 2.0)")
    print(f"   MSE global  : {mse:.4f}")
    print(f"\n   artefactos en {TRABAJO.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
