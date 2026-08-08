"""MNIST en formato PyTorch, con la misma partición que la serie de Nielsen."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DEFAULT_ROOT = Path("data/raw")

# Media y desviación típica del conjunto de entrenamiento de MNIST.
MEDIA, DESV = 0.1307, 0.3081


def mnist_tensors(root: str | Path = DEFAULT_ROOT, normalizar: bool = True):
    """Descarga MNIST y lo devuelve como tensores (X, y) por partición."""
    from torchvision.datasets import MNIST

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tr = MNIST(str(root), train=True, download=True)
    te = MNIST(str(root), train=False, download=True)

    X_tr = tr.data.float().div_(255.0)
    X_te = te.data.float().div_(255.0)
    if normalizar:
        X_tr = (X_tr - MEDIA) / DESV
        X_te = (X_te - MEDIA) / DESV
    return (X_tr, tr.targets.long()), (X_te, te.targets.long())


def mnist_loaders(
    root: str | Path = DEFAULT_ROOT,
    batch_size: int = 128,
    flatten: bool = True,
    n_val: int = 10_000,
    num_workers: int = 0,
    normalizar: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Devuelve (train_dl, val_dl, test_dl).

    flatten=True entrega vectores de 784 (para redes densas);
    flatten=False entrega imágenes de 1×28×28 (para convolucionales).
    """
    (X_tr, y_tr), (X_te, y_te) = mnist_tensors(root, normalizar)

    if flatten:
        X_tr = X_tr.reshape(len(X_tr), -1)
        X_te = X_te.reshape(len(X_te), -1)
    else:
        X_tr = X_tr.unsqueeze(1)
        X_te = X_te.unsqueeze(1)

    full = TensorDataset(X_tr, y_tr)
    n_train = len(full) - n_val
    train_ds = Subset(full, range(n_train))
    val_ds = Subset(full, range(n_train, len(full)))
    test_ds = TensorDataset(X_te, y_te)

    comun = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **comun),
        DataLoader(val_ds, batch_size=max(batch_size * 2, 256), shuffle=False, **comun),
        DataLoader(test_ds, batch_size=max(batch_size * 2, 256), shuffle=False, **comun),
    )
