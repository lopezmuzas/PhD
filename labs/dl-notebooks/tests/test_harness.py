"""Harness tests. Run in seconds; stop N00 from breaking silently.

ES: Tests del arnés. Corren en segundos y evitan que N00 se rompa sin avisar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab import harness as H


@H.datasets.register("dummy")
def build_dummy_dataset(n_samples=64, batch_size=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, 1)).astype("float32")
    y = (2 * x[:, 0] + 1).astype("float32")

    def make_loader(start, end):
        subset = TensorDataset(torch.tensor(x[start:end]), torch.tensor(y[start:end]))
        return DataLoader(subset, batch_size=batch_size)

    return make_loader(0, 48), make_loader(48, n_samples)


@H.models.register("dummy")
def build_dummy_model():
    return nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Flatten(0))


@H.optimizers.register("adam")
def build_adam(params, **kwargs):
    return torch.optim.Adam(params, **kwargs)


CONFIG = {
    "name": "test",
    "dataset": "dummy",
    "model": "dummy",
    "optimizer_args": {"lr": 1e-2},
    "epochs": 3,
}


def final_loss(seed: int) -> float:
    return H.run_experiment(CONFIG, seed=seed, save=False, verbose=False).metric()


def test_same_seed_gives_same_result():
    assert final_loss(0) == final_loss(0)


def test_different_seed_gives_different_result():
    assert final_loss(0) != final_loss(1)


def test_loss_decreases():
    result = H.run_experiment(CONFIG, seed=0, save=False, verbose=False)
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_run_saves_four_files(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "RUNS_DIR", tmp_path)
    result = H.run_experiment(CONFIG, seed=0, save=True, verbose=False)
    run_path = tmp_path / result.run_id
    for filename in ["config.json", "metrics.csv", "weights.pt", "meta.json"]:
        assert (run_path / filename).exists(), filename


def test_callbacks_are_called():
    class CountingCallback(H.Callback):
        def __init__(self):
            self.calls = {"start": 0, "batch": 0, "epoch": 0, "end": 0}

        def on_train_start(self, state):
            self.calls["start"] += 1

        def on_batch_end(self, state, loss):
            self.calls["batch"] += 1

        def on_epoch_end(self, state):
            self.calls["epoch"] += 1

        def on_train_end(self, state):
            self.calls["end"] += 1

    callback = CountingCallback()
    H.run_experiment(CONFIG, seed=0, callbacks=[callback], save=False, verbose=False)

    assert callback.calls["start"] == 1
    assert callback.calls["end"] == 1
    assert callback.calls["epoch"] == CONFIG["epochs"]
    assert callback.calls["batch"] > 0


def test_unknown_component_fails_loudly():
    with pytest.raises(KeyError, match="unknown model"):
        H.run_experiment(dict(CONFIG, model="does-not-exist"), save=False, verbose=False)
