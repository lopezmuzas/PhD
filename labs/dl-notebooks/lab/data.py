"""Synthetic tabular and 2D datasets, each carrying its own performance ceiling.

ES: Generadores sintéticos. Cada uno sabe cuál es el mejor resultado posible,
que es lo que permite distinguir un 0.87 excelente de un 0.87 mediocre.

Comment convention / Convenio: docstrings in English say WHAT, Spanish notes say WHY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from lab import harness

Task = Literal["regression", "classification"]
SplitStrategy = Literal["random", "by_group", "temporal"]


# ─────────────────────────────────────────────────────────────────────────────
# The container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SyntheticDataset:
    """Inputs, targets, and the best score any model could possibly reach.

    ES: `ceiling` es la razón de ser de esta clase. Sin ese número, un
    resultado no se puede interpretar: no sabes si te falta un 1% o un 12%.
    """

    inputs: np.ndarray
    targets: np.ndarray
    task: Task
    ceiling: float
    ceiling_metric: str
    description: str
    groups: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.inputs = np.asarray(self.inputs, dtype=np.float32)
        dtype = np.float32 if self.task == "regression" else np.int64
        self.targets = np.asarray(self.targets, dtype=dtype)

    def __len__(self) -> int:
        return len(self.inputs)

    @property
    def n_features(self) -> int:
        return self.inputs.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.targets.max()) + 1 if self.task == "classification" else 0

    def summary(self) -> str:
        return (f"{self.description}\n"
                f"  samples={len(self)}  features={self.n_features}  task={self.task}\n"
                f"  ceiling: {self.ceiling:.4f} {self.ceiling_metric}")

    # ── splitting ────────────────────────────────────────────────────────────
    def split(self, val_fraction: float = 0.2, strategy: SplitStrategy = "random",
              seed: int = 0) -> tuple["SyntheticDataset", "SyntheticDataset"]:
        """Split into train and validation.

        ES: `by_group` y `temporal` existen porque el split aleatorio miente
        cuando los datos tienen estructura. Se demuestra en N15.
        """
        indices = self._split_indices(val_fraction, strategy, seed)
        return self._subset(indices["train"]), self._subset(indices["val"])

    def _split_indices(self, val_fraction: float, strategy: SplitStrategy,
                       seed: int) -> dict[str, np.ndarray]:
        n = len(self)
        rng = np.random.default_rng(seed)

        if strategy == "temporal":
            cut = int(n * (1 - val_fraction))
            return {"train": np.arange(cut), "val": np.arange(cut, n)}

        if strategy == "by_group":
            if self.groups is None:
                raise ValueError("this dataset has no groups; generate it with n_groups>1")
            unique = np.unique(self.groups)
            rng.shuffle(unique)
            n_val_groups = max(1, int(len(unique) * val_fraction))
            val_groups = set(unique[:n_val_groups])
            is_val = np.array([g in val_groups for g in self.groups])
            return {"train": np.where(~is_val)[0], "val": np.where(is_val)[0]}

        shuffled = rng.permutation(n)
        cut = int(n * (1 - val_fraction))
        return {"train": shuffled[:cut], "val": shuffled[cut:]}

    def _subset(self, indices: np.ndarray) -> "SyntheticDataset":
        return SyntheticDataset(
            inputs=self.inputs[indices], targets=self.targets[indices],
            task=self.task, ceiling=self.ceiling, ceiling_metric=self.ceiling_metric,
            description=self.description,
            groups=None if self.groups is None else self.groups[indices],
            metadata=self.metadata,
        )

    # ── torch ────────────────────────────────────────────────────────────────
    def to_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        tensors = TensorDataset(torch.from_numpy(self.inputs), torch.from_numpy(self.targets))
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    def to_loaders(self, batch_size: int = 32, val_fraction: float = 0.2,
                   strategy: SplitStrategy = "random",
                   seed: int = 0) -> tuple[DataLoader, DataLoader]:
        train, val = self.split(val_fraction, strategy, seed)
        return train.to_loader(batch_size, shuffle=True), val.to_loader(batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────
def make_line(n_samples: int = 512, slope: float = 3.0, intercept: float = 2.0,
              noise_std: float = 0.5, seed: int = 0) -> SyntheticDataset:
    """y = slope*x + intercept + noise. Ceiling: MSE = noise_std**2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n_samples, 1))
    y = slope * x[:, 0] + intercept + rng.normal(0, noise_std, n_samples)
    return SyntheticDataset(
        inputs=x, targets=y, task="regression",
        ceiling=noise_std ** 2, ceiling_metric="mse",
        description=f"line y={slope}x+{intercept}, noise_std={noise_std}",
        metadata={"slope": slope, "intercept": intercept, "noise_std": noise_std},
    )


def make_two_gaussians(n_samples: int = 512, separation: float = 2.0,
                       spread: float = 1.0, n_features: int = 2,
                       seed: int = 0) -> SyntheticDataset:
    """Two isotropic gaussian blobs. Ceiling: the Bayes accuracy.

    ES: Con dos gaussianas de igual covarianza, el error de Bayes es exacto:
    Phi(-d / 2*sigma). Es el único caso del módulo donde el techo sale de una
    fórmula cerrada, y por eso es el mejor ejemplo para explicar el concepto.
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    offset = np.zeros(n_features)
    offset[0] = separation

    class_0 = rng.normal(0, spread, size=(half, n_features))
    class_1 = rng.normal(0, spread, size=(n_samples - half, n_features)) + offset
    inputs = np.vstack([class_0, class_1])
    targets = np.concatenate([np.zeros(half), np.ones(n_samples - half)])

    order = rng.permutation(n_samples)
    bayes_accuracy = 1 - norm.cdf(-separation / (2 * spread))

    return SyntheticDataset(
        inputs=inputs[order], targets=targets[order], task="classification",
        ceiling=bayes_accuracy, ceiling_metric="accuracy",
        description=f"two gaussians, separation={separation}, spread={spread}",
        metadata={"separation": separation, "spread": spread},
    )


def make_xor(n_samples: int = 512, noise_std: float = 0.1,
             seed: int = 0) -> SyntheticDataset:
    """The four XOR quadrants. Ceiling: perfect, and no line can reach it."""
    rng = np.random.default_rng(seed)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    picked = rng.integers(0, 4, size=n_samples)
    inputs = corners[picked] + rng.normal(0, noise_std, size=(n_samples, 2))
    return SyntheticDataset(
        inputs=inputs, targets=labels[picked], task="classification",
        ceiling=1.0, ceiling_metric="accuracy",
        description="XOR — not linearly separable",
    )


def make_spirals(n_samples: int = 512, n_turns: float = 2.0, noise_std: float = 0.1,
                 seed: int = 0) -> SyntheticDataset:
    """Two interleaved spirals. Ceiling: ~1.0 while the noise keeps them apart."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    t = np.sqrt(rng.uniform(0, 1, half)) * n_turns * 2 * np.pi
    radius = t / (n_turns * 2 * np.pi) * 3

    def arm(phase):
        return np.stack([radius * np.cos(t + phase), radius * np.sin(t + phase)], axis=1)

    inputs = np.vstack([arm(0), arm(np.pi)]) + rng.normal(0, noise_std, size=(2 * half, 2))
    targets = np.concatenate([np.zeros(half), np.ones(half)])
    order = rng.permutation(len(inputs))
    return SyntheticDataset(
        inputs=inputs[order], targets=targets[order], task="classification",
        ceiling=1.0, ceiling_metric="accuracy",
        description=f"two spirals, {n_turns} turns, noise_std={noise_std}",
    )


def make_moons(n_samples: int = 512, separation: float = 0.5, noise_std: float = 0.1,
               seed: int = 0) -> SyntheticDataset:
    """Two interleaving half circles. Where k-means breaks (N02, N17)."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    angle = rng.uniform(0, np.pi, half)
    upper = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    lower = np.stack([1 - np.cos(angle), 1 - np.sin(angle) - separation], axis=1)
    inputs = np.vstack([upper, lower]) + rng.normal(0, noise_std, size=(2 * half, 2))
    targets = np.concatenate([np.zeros(half), np.ones(half)])
    order = rng.permutation(len(inputs))
    return SyntheticDataset(
        inputs=inputs[order], targets=targets[order], task="classification",
        ceiling=1.0, ceiling_metric="accuracy",
        description=f"two moons, separation={separation}",
    )


def make_pure_noise(n_samples: int = 512, n_features: int = 10,
                    seed: int = 0) -> SyntheticDataset:
    """Inputs and targets are INDEPENDENT. Ceiling: the majority class.

    ES: El dataset más importante del módulo. Cualquier modelo que supere el
    techo en validación está haciendo trampa o midiendo mal (N10, N14).
    """
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n_samples, n_features))
    targets = rng.integers(0, 2, size=n_samples)
    majority = max(np.mean(targets), 1 - np.mean(targets))
    return SyntheticDataset(
        inputs=inputs, targets=targets, task="classification",
        ceiling=float(majority), ceiling_metric="accuracy",
        description="pure noise — no relationship between inputs and targets",
    )


def make_shapes(n_samples: int = 512, image_size: int = 16, noise_std: float = 0.1,
                seed: int = 0) -> SyntheticDataset:
    """Tiny images of squares and circles, for CNNs without downloading anything."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, 1, image_size, image_size))
    targets = rng.integers(0, 2, size=n_samples)
    grid_y, grid_x = np.mgrid[0:image_size, 0:image_size]

    for i, shape in enumerate(targets):
        size = rng.integers(4, image_size // 2)
        top = rng.integers(0, image_size - size)
        left = rng.integers(0, image_size - size)
        if shape == 0:
            images[i, 0, top:top + size, left:left + size] = 1.0
        else:
            center_y, center_x, radius = top + size / 2, left + size / 2, size / 2
            mask = (grid_y - center_y) ** 2 + (grid_x - center_x) ** 2 <= radius ** 2
            images[i, 0][mask] = 1.0

    images += rng.normal(0, noise_std, images.shape)
    dataset = SyntheticDataset(
        inputs=images.reshape(n_samples, -1), targets=targets, task="classification",
        ceiling=1.0, ceiling_metric="accuracy",
        description=f"squares vs circles, {image_size}x{image_size}",
        metadata={"image_size": image_size, "shape": (1, image_size, image_size)},
    )
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Modifiers — everything we will want to break later is a parameter here
# ─────────────────────────────────────────────────────────────────────────────
def add_label_noise(dataset: SyntheticDataset, fraction: float,
                    seed: int = 0) -> SyntheticDataset:
    """Flip a fraction of labels. Lowers the ceiling to (1 - fraction).

    ES: El techo baja porque ni el modelo perfecto puede acertar una etiqueta
    que está mal. Ese descenso es lo que hace medible el ruido de etiqueta.
    """
    rng = np.random.default_rng(seed)
    corrupted = dataset._subset(np.arange(len(dataset)))
    n_flips = int(len(dataset) * fraction)
    flip_at = rng.choice(len(dataset), n_flips, replace=False)
    n_classes = dataset.n_classes
    corrupted.targets = corrupted.targets.copy()
    corrupted.targets[flip_at] = (corrupted.targets[flip_at] +
                                  rng.integers(1, n_classes, n_flips)) % n_classes
    corrupted.ceiling = dataset.ceiling * (1 - fraction)
    corrupted.description += f" + {fraction:.0%} label noise"
    return corrupted


def make_imbalanced(dataset: SyntheticDataset, minority_fraction: float,
                    seed: int = 0) -> SyntheticDataset:
    """Drop majority-class samples until the minority is `minority_fraction`."""
    rng = np.random.default_rng(seed)
    is_minority = dataset.targets == 1
    minority_idx = np.where(is_minority)[0]
    majority_idx = np.where(~is_minority)[0]
    n_minority = int(len(majority_idx) * minority_fraction / (1 - minority_fraction))
    keep = np.concatenate([majority_idx, rng.choice(minority_idx,
                                                    min(n_minority, len(minority_idx)),
                                                    replace=False)])
    rng.shuffle(keep)
    out = dataset._subset(keep)
    out.description += f" + imbalance ({minority_fraction:.0%} minority)"
    return out


def add_duplicates(dataset: SyntheticDataset, fraction: float,
                   seed: int = 0) -> SyntheticDataset:
    """Repeat a fraction of samples. Leaks across any random split (N15)."""
    rng = np.random.default_rng(seed)
    n_copies = int(len(dataset) * fraction)
    repeated = rng.choice(len(dataset), n_copies, replace=False)
    indices = np.concatenate([np.arange(len(dataset)), repeated])
    rng.shuffle(indices)
    out = dataset._subset(indices)
    out.description += f" + {fraction:.0%} duplicates"
    return out


def add_groups(dataset: SyntheticDataset, n_groups: int = 4, shift: float = 1.0,
               seed: int = 0) -> SyntheticDataset:
    """Assign samples to groups and shift each one. Random splits now leak (N15).

    ES: Simula sitios, sujetos o sensores distintos. El modelo puede aprender
    el grupo en vez del fenómeno, y el split aleatorio no lo detecta.
    """
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=len(dataset))
    offsets = rng.normal(0, shift, size=(n_groups, dataset.n_features))
    out = dataset._subset(np.arange(len(dataset)))
    out.inputs = (out.inputs + offsets[groups]).astype(np.float32)
    out.groups = groups
    out.description += f" + {n_groups} groups"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Harness registration — one thin wrapper per generator
# ─────────────────────────────────────────────────────────────────────────────
GENERATORS = {
    "line": make_line,
    "two_gaussians": make_two_gaussians,
    "xor": make_xor,
    "spirals": make_spirals,
    "moons": make_moons,
    "pure_noise": make_pure_noise,
    "shapes": make_shapes,
}


def _register_all() -> None:
    """Expose every generator to the harness as '<name>'.

    ES: Así el config dice {"dataset": "spirals"} y N01 no toca el arnés.
    """
    for name, generator in GENERATORS.items():
        def builder(_generator=generator, batch_size=32, val_fraction=0.2,
                    split_strategy="random", split_seed=0, **generator_kwargs):
            dataset = _generator(**generator_kwargs)
            return dataset.to_loaders(batch_size, val_fraction, split_strategy, split_seed)

        harness.datasets.register(name)(builder)


_register_all()
