"""Tests del dominio. Sin I/O, sin red, sin Docker. Milisegundos."""

import numpy as np
import pytest

from fedlab.domain.aggregation import fedavg, params_delta, simple_mean


def test_fedavg_pondera_por_muestras():
    """EL test. Si este falla, todo el resto del laboratorio miente.

    Un nodo con 3x mas datos debe pesar 3x mas. Un nodo con 1 muestra no puede
    tener el mismo voto que uno con 1000.
    """
    a = [np.array([0.0, 0.0])]
    b = [np.array([2.0, 2.0])]
    assert np.allclose(fedavg([(a, 1), (b, 3)])[0], [1.5, 1.5])
    assert np.allclose(simple_mean([(a, 1), (b, 3)])[0], [1.0, 1.0])  # el bug


def test_fedavg_coincide_con_media_si_esta_balanceado():
    a, b = [np.array([1.0])], [np.array([3.0])]
    assert np.allclose(fedavg([(a, 10), (b, 10)])[0], simple_mean([(a, 10), (b, 10)])[0])


def test_fedavg_preserva_formas_multicapa():
    p = [np.ones((3, 2)), np.zeros(2)]
    out = fedavg([(p, 5), (p, 5)])
    assert [x.shape for x in out] == [(3, 2), (2,)]
    assert np.allclose(out[0], 1.0)


def test_rechaza_arquitecturas_incompatibles():
    """Un nodo con otro modelo debe fallar RUIDOSAMENTE, no propagar basura."""
    with pytest.raises(ValueError, match="formas"):
        fedavg([([np.zeros(2)], 1), ([np.zeros(3)], 1)])


def test_rechaza_lista_vacia():
    with pytest.raises(ValueError):
        fedavg([])


def test_rechaza_num_samples_invalido():
    with pytest.raises(ValueError, match="num_samples"):
        fedavg([([np.zeros(2)], 0)])


def test_params_delta_es_cero_para_iguales():
    p = [np.array([1.0, 2.0])]
    assert params_delta(p, p) == pytest.approx(0.0)
    assert params_delta(p, [np.array([1.0, 5.0])]) == pytest.approx(3.0)
