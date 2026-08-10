"""El roundtrip: parametros -> disco/JSON -> parametros, sin perder nada.

Los bugs de serializacion son los mas caros de depurar en C2D, porque se
manifiestan como "el job termino pero los pesos son basura" tres rondas despues.
"""

import numpy as np
import pytest

from fedlab.serialization import (
    check_json_size,
    load_params_npz,
    params_from_json,
    params_to_json,
    save_params_npz,
)

PARAMS = [np.array([1.5, -2.25, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5])]


def test_roundtrip_npz_preserva_valores_y_formas(tmp_path):
    path = tmp_path / "p.npz"
    save_params_npz(path, PARAMS)
    out = load_params_npz(path)
    assert len(out) == len(PARAMS)
    for a, b in zip(PARAMS, out):
        assert a.shape == b.shape
        assert np.allclose(a, b)


def test_roundtrip_npz_preserva_el_ORDEN(tmp_path):
    """p0, p1, p2... El orden ES el contrato. np.savez no garantiza orden de claves."""
    params = [np.array([float(i)]) for i in range(12)]  # >9 fuerza el orden lexicografico a fallar
    path = tmp_path / "p.npz"
    save_params_npz(path, params)
    out = load_params_npz(path)
    assert [float(p[0]) for p in out] == list(range(12))


def test_roundtrip_json():
    out = params_from_json(params_to_json(PARAMS))
    for a, b in zip(PARAMS, out):
        assert np.allclose(a, b)
        assert a.shape == b.shape


def test_npz_rechaza_pickle(tmp_path):
    """Garantiza que nunca cargamos objetos arbitrarios de un tercero."""
    path = tmp_path / "malicioso.npz"
    np.savez(path, p0=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_params_npz(path)


def test_check_json_size_falla_con_modelo_grande():
    with pytest.raises(ValueError, match="algoCustomData"):
        check_json_size([np.zeros(100_000)])


def test_check_json_size_acepta_modelo_pequeno():
    assert check_json_size([np.zeros(10)]) > 0
