import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from feltc2d import noise


def _pesos():
    return {"w": np.array([1.0, -2.0, 0.5]), "b": np.array([0.25])}


def test_es_determinista():
    a = noise.anadir_ruido(_pesos(), 42)
    b = noise.anadir_ruido(_pesos(), 42)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_semillas_distintas_dan_ruidos_distintos():
    a = noise.anadir_ruido(_pesos(), 1)
    b = noise.anadir_ruido(_pesos(), 2)
    assert not np.allclose(a["w"], b["w"])


def test_parametros_distintos_reciben_ruido_distinto():
    r1 = noise.ruido_como(np.zeros(3), 7, "w")
    r2 = noise.ruido_como(np.zeros(3), 7, "b")
    assert not np.allclose(r1, r2)


def test_el_ruido_tapa_la_senal():
    p = _pesos()
    cegado = noise.anadir_ruido(p, 5)
    assert np.abs(cegado["w"] - p["w"]).min() > 10  # senal/ruido pesima


def test_media_simple_se_cancela():
    semillas = [1, 2, 3]
    cegados = [noise.anadir_ruido(_pesos(), s) for s in semillas]
    medio = {k: np.mean([c[k] for c in cegados], axis=0) for k in cegados[0]}
    limpio = noise.quitar_ruido(medio, semillas)
    assert np.allclose(limpio["w"], _pesos()["w"])


def test_media_PONDERADA_necesita_alfa():
    """El bug clasico: con nodos de distinto tamano, la media simple no vale."""
    semillas = [1, 2, 3]
    alfa = [0.6, 0.3, 0.1]
    cegados = [noise.anadir_ruido(_pesos(), s) for s in semillas]
    ponderado = {
        k: sum(a * c[k] for a, c in zip(alfa, cegados)) for k in cegados[0]
    }
    assert np.allclose(noise.quitar_ruido(ponderado, semillas, alfa)["w"], _pesos()["w"])
    assert not np.allclose(noise.quitar_ruido(ponderado, semillas)["w"], _pesos()["w"])


def test_alfa_de_longitud_incorrecta_falla():
    with pytest.raises(ValueError):
        noise.quitar_ruido(_pesos(), [1, 2], [1.0])
