"""Las rutas del contenedor C2D. Identicas a las que usa feltlabs.

    /data/inputs/<DID>/<fichero>      los datasets, una carpeta por DID
    /data/inputs/algoCustomData.json  parametros del job (JSON arbitrario)
    /data/outputs/model               UNICO fichero de salida, nombre fijo

El nombre `model` no es opcional: es lo que el resto del pipeline espera.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CARPETA_ENTRADA = Path("/data/inputs")
CARPETA_SALIDA = Path("/data/outputs")
CUSTOM_DATA = "algoCustomData.json"


@dataclass
class ConfigOcean:
    entrada: Path = CARPETA_ENTRADA
    salida: Path = CARPETA_SALIDA

    @property
    def custom_data_path(self) -> Path:
        return self.entrada / CUSTOM_DATA

    def leer_custom_data(self) -> dict:
        if not self.custom_data_path.exists():
            return {}
        return json.loads(self.custom_data_path.read_text())

    def escribir_modelo(self, datos: bytes) -> Path:
        self.salida.mkdir(parents=True, exist_ok=True)
        destino = self.salida / "model"
        destino.write_bytes(datos)
        return destino


# --8<-- [start:datasets]
def datasets(config: ConfigOcean) -> dict[str, list[Path]]:
    """Mapea DID -> ficheros. Cada subcarpeta de /data/inputs es un dataset."""
    fuera = {}
    for p in sorted(config.entrada.iterdir()):
        if p.is_dir():
            fuera[p.name] = sorted(f for f in p.glob("**/*") if f.is_file())
    return fuera


def cargar_csv(config: ConfigOcean, columna_objetivo: int = -1):
    """Concatena todos los CSV de todos los datasets. Ultima columna = y."""
    import numpy as np

    trozos = []
    for _did, ficheros in datasets(config).items():
        for f in ficheros:
            trozos.append(np.genfromtxt(f, delimiter=","))
    if not trozos:
        raise FileNotFoundError(f"sin datos en {config.entrada}")
    datos = np.concatenate(trozos, axis=0)
    y = datos[:, columna_objetivo]
    X = np.delete(datos, columna_objetivo, axis=1)
    return X, y
# --8<-- [end:datasets]
