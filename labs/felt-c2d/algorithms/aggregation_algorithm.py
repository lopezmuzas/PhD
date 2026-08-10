"""Algoritmo de AGREGACION. Se ejecuta dentro de OTRO contenedor C2D.

Recibe las rutas (o URLs) de los modelos locales por `algoCustomData.json`:

    {"model_urls": ["./out/nodo0/model", "./out/nodo1/model"]}

Es el segundo job del pipeline. Su clave privada es lo que impide que el
cientifico de datos lea los modelos individuales.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feltc2d.ocean import CARPETA_ENTRADA, CARPETA_SALIDA, ConfigOcean
from feltc2d.protocol import fase2_agregar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_folder", type=Path, default=CARPETA_ENTRADA)
    ap.add_argument("--output_folder", type=Path, default=CARPETA_SALIDA)
    ap.add_argument("--private_key", required=True, help="clave privada en hex")
    ap.add_argument("--min_models", type=int, default=2)
    ap.add_argument("--sin_ponderar", action="store_true",
                    help="usa media simple (util para reproducir el bug clasico)")
    args = ap.parse_args()

    cfg = ConfigOcean(entrada=args.input_folder, salida=args.output_folder)
    custom = cfg.leer_custom_data()
    urls = custom.get("model_urls", [])

    if len(urls) < args.min_models:
        print(f"ERROR: {len(urls)} modelos, se exigen {args.min_models}", file=sys.stderr)
        return 1

    cifrados = [Path(u).read_bytes() for u in urls]
    global_ = fase2_agregar(cifrados, bytes.fromhex(args.private_key),
                            ponderado=not args.sin_ponderar)

    destino = cfg.escribir_modelo(global_)
    print(f"agregados {len(cifrados)} modelos -> {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
