"""El caso de uso, con dobles de prueba. Aqui se demuestra el LSP."""

import numpy as np
import pytest

from fedlab.adapters.sources import ArraySource
from fedlab.adapters.stores import InMemoryStore
from fedlab.config import TrainConfig
from fedlab.domain.round import run_local_round
from fedlab.learners.linear import LinearRegressor


class FakeLearner:
    """Doble de prueba: no entrena, solo registra que le llamaron y con que."""

    def __init__(self):
        self.params = [np.array([0.0])]
        self.received = None
        self.fit_calls = 0

    def get_params(self):
        return [p.copy() for p in self.params]

    def set_params(self, params):
        self.received = params
        self.params = [p.copy() for p in params]

    def fit(self, X, y, cfg):
        self.fit_calls += 1
        self.params = [self.params[0] + 1.0]
        return {"loss": 0.42}

    def evaluate(self, X, y):
        return {"loss": 0.42}


def test_ronda_sin_pesos_globales_no_llama_a_set_params():
    """Ronda 0 (arranque en frio): el learner conserva su inicializacion."""
    learner = FakeLearner()
    run_local_round(InMemoryStore(), ArraySource(np.zeros((5, 1)), np.zeros(5)),
                    learner, TrainConfig())
    assert learner.received is None
    assert learner.fit_calls == 1


def test_ronda_con_pesos_globales_los_carga_antes_de_entrenar():
    learner = FakeLearner()
    store = InMemoryStore(initial=[np.array([7.0])])
    run_local_round(store, ArraySource(np.zeros((5, 1)), np.zeros(5)), learner, TrainConfig())
    assert np.allclose(learner.received[0], [7.0])
    assert np.allclose(store.updates[0][0][0], [8.0])  # 7 + 1 del fit


def test_ronda_publica_num_samples_y_metricas():
    store = InMemoryStore()
    run_local_round(store, ArraySource(np.zeros((17, 1)), np.zeros(17)),
                    FakeLearner(), TrainConfig())
    _, meta = store.updates[0]
    assert meta["num_samples"] == 17
    assert meta["loss"] == 0.42


def test_el_dominio_funciona_con_cualquier_store(tmp_path):
    """LSP: FileStore e InMemoryStore son intercambiables sin tocar el dominio."""
    from fedlab.adapters.stores import FileStore

    for store in [InMemoryStore(), FileStore(tmp_path / "in", tmp_path / "out")]:
        result = run_local_round(store, ArraySource(np.ones((10, 1)), np.ones(10)),
                                 LinearRegressor(n_features=1), TrainConfig(epochs=3))
        assert result.num_samples == 10
        assert "mse" in result.metrics


def test_learner_rechaza_pesos_de_otra_arquitectura():
    learner = LinearRegressor(n_features=1)
    with pytest.raises(ValueError, match="Se esperaba"):
        learner.set_params([np.zeros(5), np.zeros(1)])
