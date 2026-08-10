"""Algoritmo de ENTRENAMIENTO LOCAL. Se ejecuta dentro del contenedor C2D.

Espeja la estructura de `simple_algorithm/local_algorithm.py` de FELT Labs,
pero sin dependencias de Ocean: se puede ejecutar en local apuntando a
carpetas cualesquiera.

    python algorithms/local_algorithm.py \
        --input_folder ./data/nodo0 --output_folder ./out/nodo0 \
        --aggregation_key <hex> --seed 1001 --modelo lineal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feltc2d.ocean import CARPETA_ENTRADA, CARPETA_SALIDA, ConfigOcean, cargar_csv
from feltc2d.protocol import fase1_entrenar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_folder", type=Path, default=CARPETA_ENTRADA)
    ap.add_argument("--output_folder", type=Path, default=CARPETA_SALIDA)
    ap.add_argument("--aggregation_key", required=True, help="clave publica en hex")
    ap.add_argument("--seed", type=int, required=True, help="semilla del ruido")
    ap.add_argument("--modelo", default="lineal", choices=["lineal", "logistica"])
    ap.add_argument("--target_column", type=int, default=-1)
    args = ap.parse_args()

    cfg = ConfigOcean(entrada=args.input_folder, salida=args.output_folder)
    # algoCustomData.json puede sobreescribir los parametros no fijos
    custom = cfg.leer_custom_data()
    modelo = custom.get("modelo", args.modelo)
    seed = int(custom.get("seed", args.seed))

    X, y = cargar_csv(cfg, args.target_column)
    resultado = fase1_entrenar(X, y, modelo, bytes.fromhex(args.aggregation_key), seed)

    destino = cfg.escribir_modelo(resultado.cifrado)
    print(f"entrenado n={len(X)} modelo={modelo} -> {destino} ({len(resultado.cifrado)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
