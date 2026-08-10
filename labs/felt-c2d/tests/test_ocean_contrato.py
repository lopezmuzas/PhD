"""Contrato de C2D: rutas, nombres de fichero y formato de salida."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d.ocean import CUSTOM_DATA, ConfigOcean, cargar_csv, datasets


def test_rutas_por_defecto():
    from feltc2d.ocean import CARPETA_ENTRADA, CARPETA_SALIDA
    assert str(CARPETA_ENTRADA) == "/data/inputs"
    assert str(CARPETA_SALIDA) == "/data/outputs"
    assert CUSTOM_DATA == "algoCustomData.json"


def test_una_carpeta_por_did(tmp_path):
    cfg = ConfigOcean(entrada=tmp_path / "in", salida=tmp_path / "out")
    for did in ["did:op:aaa", "did:op:bbb"]:
        d = cfg.entrada / did
        d.mkdir(parents=True)
        (d / "datos.csv").write_text("1,2,3\n")
    cfg.custom_data_path.write_text("{}")
    assert set(datasets(cfg)) == {"did:op:aaa", "did:op:bbb"}


def test_custom_data_se_lee(tmp_path):
    cfg = ConfigOcean(entrada=tmp_path / "in", salida=tmp_path / "out")
    cfg.entrada.mkdir(parents=True)
    cfg.custom_data_path.write_text(json.dumps({"modelo": "logistica", "seed": 7}))
    assert cfg.leer_custom_data()["seed"] == 7


def test_custom_data_ausente_no_revienta(tmp_path):
    cfg = ConfigOcean(entrada=tmp_path / "in", salida=tmp_path / "out")
    cfg.entrada.mkdir(parents=True)
    assert cfg.leer_custom_data() == {}


def test_la_salida_se_llama_model(tmp_path):
    cfg = ConfigOcean(entrada=tmp_path / "in", salida=tmp_path / "out")
    destino = cfg.escribir_modelo(b"x")
    assert destino.name == "model"


def test_ultima_columna_es_y(tmp_path):
    cfg = ConfigOcean(entrada=tmp_path / "in", salida=tmp_path / "out")
    d = cfg.entrada / "did:op:1"
    d.mkdir(parents=True)
    np.savetxt(d / "a.csv", np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 8.0]]), delimiter=",")
    X, y = cargar_csv(cfg)
    assert X.shape == (2, 2)
    assert list(y) == [9.0, 8.0]
