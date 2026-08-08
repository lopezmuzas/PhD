"""Datasets sintéticos para probar arquitecturas rápido, sin descargar nada."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def make_moons_loaders(
    n_samples: int = 4000,
    noise: float = 0.2,
    val_split: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Devuelve (train_dl, val_dl, in_features, n_classes)."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long())

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, X_tr.shape[1], int(len(np.unique(y)))
