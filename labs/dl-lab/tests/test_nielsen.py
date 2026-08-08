"""Tests de la implementación desde cero. No descargan MNIST: usan datos sintéticos."""

import numpy as np

from dllab.nielsen.data import one_hot
from dllab.nielsen.improved import CosteCuadratico, CosteEntropiaCruzada, RedMejorada
from dllab.nielsen.network import Red, comprobar_gradiente, sigmoide, sigmoide_prima


def datos_juguete(n, rng):
    """Clasificación binaria: ¿la suma de las entradas supera 3?"""
    return [(v, one_hot(int(v.sum() > 3.0), 2)) for v in (rng.random((6, 1)) for _ in range(n))]


def test_sigmoide():
    assert sigmoide(np.array([0.0])) == 0.5
    # el máximo de la derivada es 0.25, en z = 0
    assert np.isclose(sigmoide_prima(np.array([0.0])), 0.25)
    assert sigmoide_prima(np.array([20.0])) < 1e-6


def test_formas_de_la_red():
    red = Red([4, 3, 2], seed=0)
    assert [w.shape for w in red.pesos] == [(3, 4), (2, 3)]
    assert [b.shape for b in red.sesgos] == [(3, 1), (2, 1)]
    assert red.propagar(np.zeros((4, 1))).shape == (2, 1)


def test_backprop_coincide_con_gradiente_numerico():
    """La prueba que de verdad importa: backprop está bien implementado."""
    red = Red([6, 5, 4, 3], seed=1)
    x = np.random.default_rng(0).random((6, 1))
    y = one_hot(2, 3)
    for diferencia in comprobar_gradiente(red, x, y):
        assert diferencia < 1e-7


def test_la_red_aprende():
    rng = np.random.default_rng(0)
    tr = datos_juguete(600, rng)
    ev = [(x, int(np.argmax(y))) for x, y in datos_juguete(200, rng)]
    red = Red([6, 8, 2], seed=3)
    hist = red.sgd(tr, epocas=12, tam_minilote=10, eta=2.0, datos_eval=ev, verbose=False, seed=3)
    assert hist.acierto_eval[-1] > 0.85
    assert hist.coste_entrenamiento[-1] < hist.coste_entrenamiento[0]


def test_deltas_de_los_costes():
    z = np.array([[2.0]])
    a = sigmoide(z)
    y = np.array([[0.0]])
    # el cuadrático arrastra el factor sigmoide_prima; la entropía cruzada no
    assert np.isclose(CosteCuadratico.delta(z, a, y), (a - y) * sigmoide_prima(z))
    assert np.isclose(CosteEntropiaCruzada.delta(z, a, y), a - y)
    assert abs(CosteEntropiaCruzada.delta(z, a, y)) > abs(CosteCuadratico.delta(z, a, y))


def test_red_mejorada_aprende():
    rng = np.random.default_rng(1)
    tr = datos_juguete(600, rng)
    ev = [(x, int(np.argmax(y))) for x, y in datos_juguete(200, rng)]
    red = RedMejorada([6, 8, 2], coste=CosteEntropiaCruzada, seed=3)
    hist = red.sgd(tr, epocas=15, tam_minilote=10, eta=0.5, lmbda=0.1,
                   datos_eval=ev, monitorizar_entrenamiento=True, verbose=False, seed=3)
    assert hist.acierto_eval[-1] > 0.85
    assert len(hist.acierto_entrenamiento) == len(hist.acierto_eval)


def test_inicializacion_escalada_produce_pesos_menores():
    grande = RedMejorada([784, 30, 10], init="grande", seed=0)
    escalada = RedMejorada([784, 30, 10], init="escalada", seed=0)
    assert escalada.pesos[0].std() < grande.pesos[0].std() / 10
