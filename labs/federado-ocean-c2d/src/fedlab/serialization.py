"""Serializacion de parametros. SIN PICKLE. Nunca.

Motivo de seguridad, no de estilo: el orquestador deserializa datos que vienen de
N proveedores que no controla. `np.load(..., allow_pickle=True)` sobre un fichero
hostil es ejecucion remota de codigo. Aqui solo se mueven numeros.

Dos formatos, dos usos:
  - JSON  -> viaja por la API de Ocean dentro de `algoCustomData` (modelos pequenos).
  - NPZ   -> ficheros en /data/outputs (cualquier tamano, mas eficiente).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ports import Params

MAX_JSON_BYTES = 256 * 1024  # margen prudente para algoCustomData


def params_to_json(params: Params) -> list:
    """Arrays -> listas anidadas de floats de Python."""
    return [np.asarray(p, dtype=np.float64).tolist() for p in params]


def params_from_json(data: list) -> Params:
    """Listas anidadas -> arrays float64."""
    return [np.asarray(p, dtype=np.float64) for p in data]


def check_json_size(params: Params) -> int:
    """Falla pronto y con un mensaje util si el modelo no cabe en algoCustomData."""
    size = len(json.dumps(params_to_json(params)).encode("utf-8"))
    if size > MAX_JSON_BYTES:
        raise ValueError(
            f"El modelo ocupa {size} bytes en JSON (limite {MAX_JSON_BYTES}). "
            "Publica los pesos como asset y pasalos en `datasets`, "
            "en lugar de usar algoCustomData. Ver docs/05-contrato-c2d.md."
        )
    return size


def save_params_npz(path: str | Path, params: Params) -> None:
    """Guarda como arrays nombrados p0, p1, ... El orden importa y se preserva."""
    np.savez(Path(path), **{f"p{i}": np.asarray(p) for i, p in enumerate(params)})


def load_params_npz(path: str | Path) -> Params:
    """Carga con allow_pickle=False explicito."""
    with np.load(Path(path), allow_pickle=False) as npz:
        return [npz[f"p{i}"] for i in range(len(npz.files))]


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
