"""Test rápido: si esto pasa, el entorno está bien montado."""

import torch

from dllab.data.synthetic import make_moons_loaders
from dllab.models.mlp import MLP
from dllab.training import train
from dllab.utils.seed import set_seed


def test_forward_shape():
    model = MLP(in_features=2, n_classes=2, hidden=[16, 16])
    out = model(torch.randn(8, 2))
    assert out.shape == (8, 2)
    assert model.n_params > 0


def test_train_reduces_loss():
    set_seed(0)
    train_dl, val_dl, in_f, n_c = make_moons_loaders(n_samples=600, batch_size=64)
    model = MLP(in_features=in_f, n_classes=n_c, hidden=[32, 32])
    history = train(model, train_dl, val_dl, epochs=5, progress=False)
    assert history.train_loss[-1] < history.train_loss[0]
