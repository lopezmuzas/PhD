"""Perceptrón multicapa parametrizable: el 'hola mundo' del laboratorio."""

from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden = hidden or [64, 64]
        act = ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name: str, in_features: int, n_classes: int, **kwargs) -> nn.Module:
    """Registro de modelos: añade aquí tus arquitecturas nuevas."""
    registry = {"mlp": MLP}
    if name not in registry:
        raise ValueError(f"Modelo desconocido: {name}. Disponibles: {list(registry)}")
    return registry[name](in_features=in_features, n_classes=n_classes, **kwargs)
