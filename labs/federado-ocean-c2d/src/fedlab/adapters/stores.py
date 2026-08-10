"""Adaptadores del puerto `ParameterStore`. Aqui vive TODO el I/O.

Tres implementaciones, misma interfaz, complejidad creciente:

    InMemoryStore  -> tests unitarios (Lab 0)
    FileStore      -> carpetas de tu portatil (Lab 1)
    C2DStore       -> /data/inputs y /data/outputs de Ocean (Labs 2 y 3)

LSP: el dominio no distingue una de otra. Ese es exactamente el punto -- si el
Lab 1 funciona, el Lab 3 tambien, salvo problemas de infraestructura.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from ..ports import Params
from ..serialization import (
    load_params_npz,
    params_from_json,
    read_json,
    save_params_npz,
    write_json,
)


class InMemoryStore:
    """Sin disco. Para tests: instantaneo y sin efectos colaterales."""

    def __init__(self, initial: Params | None = None):
        self._global = initial
        self.updates: list[tuple[Params, dict]] = []

    def load_global(self) -> Params | None:
        return self._global

    def save_update(self, params: Params, meta: dict) -> None:
        self.updates.append((params, meta))

    def set_global(self, params: Params) -> None:
        self._global = params


class FileStore:
    """Dos carpetas locales que imitan el contrato de C2D.

    El Lab 1 usa esto para que puedas depurar el flujo completo sin Docker.
    """

    GLOBAL_FILE = "global.npz"
    UPDATE_FILE = "update.npz"
    META_FILE = "metrics.json"

    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def load_global(self) -> Params | None:
        path = self.input_dir / self.GLOBAL_FILE
        return load_params_npz(path) if path.exists() else None

    def save_update(self, params: Params, meta: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_params_npz(self.output_dir / self.UPDATE_FILE, params)
        write_json(self.output_dir / self.META_FILE, meta)


class C2DStore:
    """Adaptador real de Ocean Compute-to-Data.

    Rutas verificadas contra el codigo de ocean-node (compute_engine_docker.ts):
      /data/inputs   -- assets descargados + algoCustomData.json (siempre presente)
      /data/ddos     -- DDOs de los assets
      /data/outputs  -- lo que escribas aqui es lo unico que sale del contenedor
      /data/logs     -- stdout/stderr recogidos por el nodo

    Los pesos globales llegan dentro de algoCustomData.json, bajo la clave
    "params". Es el unico canal de entrada dinamico que no exige republicar un
    asset en cada ronda. Ver docs/05-contrato-c2d.md.
    """

    UPDATE_FILE = "update.npz"
    META_FILE = "metrics.json"
    CUSTOM_DATA = "algoCustomData.json"

    DEFAULT_INPUT = "/data/inputs"
    DEFAULT_OUTPUT = "/data/outputs"

    def __init__(self, input_dir: str | None = None, output_dir: str | None = None):
        # Las variables de entorno permiten montar un /data falso en los tests
        # sin parchear nada. El contenedor real no las define y usa /data.
        self.input_dir = Path(input_dir or os.getenv("FL_INPUT_DIR", self.DEFAULT_INPUT))
        self.output_dir = Path(output_dir or os.getenv("FL_OUTPUT_DIR", self.DEFAULT_OUTPUT))

    def read_custom_data(self) -> dict:
        """algoCustomData completo. El nodo lo escribe SIEMPRE, aunque sea `{}`."""
        path = self.input_dir / self.CUSTOM_DATA
        if not path.exists():
            return {}
        try:
            return read_json(path)
        except Exception:  # JSON corrupto no debe tumbar el job entero
            return {}

    def load_global(self) -> Params | None:
        data = self.read_custom_data().get("params")
        return params_from_json(data) if data else None

    def save_update(self, params: Params, meta: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_params_npz(self.output_dir / self.UPDATE_FILE, params)
        write_json(self.output_dir / self.META_FILE, meta)


def find_dataset_file(input_dir: str = "/data/inputs") -> str | None:
    """Localiza el CSV que el nodo descargo en /data/inputs.

    El nodo lo guarda con el NOMBRE ORIGINAL del fichero remoto, que el
    algoritmo no conoce de antemano. Por eso buscamos por extension y
    descartamos explicitamente algoCustomData.json.

    Si defines la variable de entorno FL_DATA_FILE, gana esa.
    """
    explicit = os.getenv("FL_DATA_FILE")
    if explicit:
        return explicit
    candidates = sorted(
        p for p in glob.glob(str(Path(input_dir) / "*"))
        if Path(p).name != C2DStore.CUSTOM_DATA and Path(p).suffix in {".csv", ".npz", ".txt"}
    )
    return candidates[0] if candidates else None
