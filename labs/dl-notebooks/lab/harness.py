"""Experiment harness: turns a config dict into results on disk.

ES: Arnés de experimentos. Un experimento es un diccionario; esto lo convierte
en resultados en disco.

Design rule / Regla de diseño:
    Everything that CHANGES between experiments lives in the config.
    Everything that DOESN'T lives here.

Comment convention / Convenio de comentarios:
    Docstrings in English describe WHAT. Spanish notes explain WHY.
"""
from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

RUNS_DIR = Path("runs")
DEFAULT_LOSS = "mse_loss"

ModelBuilder = Callable[..., nn.Module]
DatasetBuilder = Callable[..., tuple[DataLoader, DataLoader]]
OptimizerBuilder = Callable[..., Optimizer]


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
class Registry(dict):
    """Maps a name to a builder, so configs stay JSON-serializable.

    ES: El config dice "mlp" en vez de importar una clase. Eso es lo que
    permite guardarlo en disco y compararlo después.
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def register(self, name: str) -> Callable:
        def decorator(builder):
            self[name] = builder
            return builder
        return decorator

    def build(self, name: str, **kwargs):
        if name not in self:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self)}")
        return self[name](**kwargs)


models = Registry("model")
datasets = Registry("dataset")
optimizers = Registry("optimizer")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random source we know about.

    ES: "Todas las que conocemos" no es "todas". En GPU quedan operaciones no
    deterministas aunque la semilla sea idéntica. Eso se mide en N13, no se
    asume aquí.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def current_git_commit() -> str:
    """Short commit hash, or 'no-git' when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Components:
    """Everything a training loop needs, built from a config."""

    train_loader: DataLoader
    val_loader: DataLoader
    model: nn.Module
    optimizer: Optimizer
    loss_fn: Callable


@dataclass
class TrainingState:
    """What callbacks can inspect while training runs.

    ES: Existe para que N06 pueda enchufar diagnósticos SIN tocar el bucle.
    """

    config: dict
    model: nn.Module
    optimizer: Optimizer
    device: torch.device
    epoch: int = 0
    step: int = 0
    history: list[dict] = field(default_factory=list)
    scratch: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Outcome of a single run."""

    run_id: str
    config: dict
    seed: int
    history: list[dict]
    model: nn.Module
    elapsed_seconds: float
    scratch: dict = field(default_factory=dict)

    @property
    def final_metrics(self) -> dict:
        return self.history[-1]

    def metric(self, name: str = "val_loss") -> float:
        return self.final_metrics[name]


class Callback:
    """Empty hooks. N06 will subclass this for diagnostics.

    ES: Es un contrato, no una implementación. Está vacío a propósito.
    """

    def on_train_start(self, state: TrainingState) -> None: ...
    def on_batch_end(self, state: TrainingState, loss: float) -> None: ...
    def on_epoch_end(self, state: TrainingState) -> None: ...
    def on_train_end(self, state: TrainingState) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def build_components(config: dict, seed: int) -> Components:
    """Instantiate data, model and optimizer from the config."""
    set_seed(seed)
    train_loader, val_loader = datasets.build(
        config["dataset"], **config.get("dataset_args", {}))
    model = models.build(config["model"], **config.get("model_args", {}))
    optimizer = optimizers.build(
        config.get("optimizer", "adam"),
        params=model.parameters(),
        **config.get("optimizer_args", {"lr": 1e-3}),
    )
    loss_fn = getattr(torch.nn.functional, config.get("loss", DEFAULT_LOSS))
    return Components(train_loader, val_loader, model, optimizer, loss_fn)


def resolve_device(config: dict) -> torch.device:
    requested = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(components: Components, state: TrainingState,
                    callbacks: Iterable[Callback]) -> float:
    """Run one full pass over the training set. Returns mean loss."""
    components.model.train()
    total_loss, seen = 0.0, 0

    for inputs, targets in components.train_loader:
        inputs = inputs.to(state.device)
        targets = targets.to(state.device)

        components.optimizer.zero_grad()
        loss = components.loss_fn(components.model(inputs), targets)
        loss.backward()
        components.optimizer.step()

        state.step += 1
        total_loss += loss.item() * len(inputs)
        seen += len(inputs)
        for callback in callbacks:
            callback.on_batch_end(state, loss.item())

    return total_loss / seen


@torch.no_grad()
def evaluate(components: Components, device: torch.device) -> dict:
    """Mean loss over the validation set."""
    components.model.eval()
    total_loss, seen = 0.0, 0
    for inputs, targets in components.val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        total_loss += components.loss_fn(components.model(inputs), targets).item() * len(inputs)
        seen += len(inputs)
    return {"loss": total_loss / max(seen, 1)}


def run_experiment(config: dict, seed: int | None = None,
                   callbacks: list[Callback] | None = None,
                   save: bool = True, verbose: bool = True) -> ExperimentResult:
    """Train one model end to end and return its result.

    ES: Lanzar dos veces el mismo config con la misma semilla debe dar
    exactamente lo mismo. Si no, la semilla no llega a algún sitio.
    """
    started_at = time.time()
    seed = config.get("seed", 0) if seed is None else seed
    callbacks = callbacks or []
    device = resolve_device(config)

    components = build_components(config, seed)
    components.model.to(device)

    state = TrainingState(config=config, model=components.model,
                          optimizer=components.optimizer, device=device)
    for callback in callbacks:
        callback.on_train_start(state)

    total_epochs = config["epochs"]
    log_every = max(1, total_epochs // 5)

    for epoch in range(total_epochs):
        state.epoch = epoch
        train_loss = train_one_epoch(components, state, callbacks)
        val_metrics = evaluate(components, device)

        state.history.append({"epoch": epoch, "train_loss": train_loss,
                              **{f"val_{k}": v for k, v in val_metrics.items()}})
        for callback in callbacks:
            callback.on_epoch_end(state)

        if verbose and (epoch % log_every == 0 or epoch == total_epochs - 1):
            print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_metrics['loss']:.5f}")

    for callback in callbacks:
        callback.on_train_end(state)

    result = ExperimentResult(
        run_id=make_run_id(config, seed),
        config=config, seed=seed, history=state.history,
        model=components.model, scratch=state.scratch,
        elapsed_seconds=round(time.time() - started_at, 2),
    )
    if save:
        save_run(result)
    return result


def make_run_id(config: dict, seed: int) -> str:
    name = config.get("name", "run")
    return f"{name}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_run(result: ExperimentResult) -> Path:
    """Write config, metrics, weights and metadata.

    ES: meta.json es el que casi nadie guarda y el que más falta hace a los
    seis meses.
    """
    run_path = RUNS_DIR / result.run_id
    run_path.mkdir(parents=True, exist_ok=True)

    (run_path / "config.json").write_text(json.dumps(result.config, indent=2))
    pd.DataFrame(result.history).to_csv(run_path / "metrics.csv", index=False)
    torch.save(result.model.state_dict(), run_path / "weights.pt")
    (run_path / "meta.json").write_text(json.dumps({
        "run_id": result.run_id,
        "seed": result.seed,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": current_git_commit(),
        "elapsed_seconds": result.elapsed_seconds,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    }, indent=2))
    return run_path


def load_run(run_id: str) -> dict:
    """Read a saved run back from disk."""
    run_path = RUNS_DIR / run_id
    return {
        "run_id": run_id,
        "config": json.loads((run_path / "config.json").read_text()),
        "meta": json.loads((run_path / "meta.json").read_text()),
        "history": pd.read_csv(run_path / "metrics.csv"),
    }


def list_runs(pattern: str = "") -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir() and pattern in p.name)


# ─────────────────────────────────────────────────────────────────────────────
# Comparing
# ─────────────────────────────────────────────────────────────────────────────
def compare_runs(run_ids: list[str], metric: str = "val_loss") -> pd.DataFrame:
    """One row per run, with its config and final metric."""
    rows = []
    for run_id in run_ids:
        run = load_run(run_id)
        flat_config = {k: v for k, v in run["config"].items() if not isinstance(v, dict)}
        rows.append({
            "run_id": run_id,
            **flat_config,
            # ES: DESPUÉS de flat_config a propósito. El config trae una clave
            # "seed" (la de por defecto) y meta trae la que se usó de verdad.
            # Si esta línea va antes, el config la pisa y comparar N semillas
            # devuelve N filas que dicen todas la misma. Manda meta.
            "seed": run["meta"]["seed"],
            "final": run["history"][metric].iloc[-1],
            "best": run["history"][metric].min(),
            "seconds": run["meta"]["elapsed_seconds"],
        })
    return pd.DataFrame(rows)


def plot_runs(run_ids: list[str], metrics=("train_loss", "val_loss"),
              log_scale: bool = True, ax=None):
    """Overlay learning curves from several runs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for run_id in run_ids:
        history = load_run(run_id)["history"]
        label = run_id.split("_")[0]
        for index, metric in enumerate(metrics):
            if metric in history:
                ax.plot(history["epoch"], history[metric],
                        label=f"{label} · {metric}",
                        linestyle="-" if index == 0 else "--", alpha=0.9)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────────────────────
def _with_override(config: dict, key: str, value) -> dict:
    """Copy of config with one key replaced. Supports 'optimizer_args.lr'."""
    updated = json.loads(json.dumps(config))
    if "." in key:
        section, inner_key = key.split(".", 1)
        updated.setdefault(section, {})[inner_key] = value
    else:
        updated[key] = value
    return updated


def sweep(base_config: dict, key: str, values: list, seeds=(0,)) -> list[str]:
    """Run the same config varying ONE key. Returns the run ids."""
    run_ids = []
    for value in values:
        for seed in seeds:
            config = _with_override(base_config, key, value)
            config["name"] = f"{base_config.get('name', 'run')}-{key.split('.')[-1]}{value}"
            print(f"▶ {key}={value}  seed={seed}")
            run_ids.append(run_experiment(config, seed=seed, verbose=False).run_id)
    return run_ids


def repeat_with_seeds(config: dict, n_seeds: int = 5,
                      metric: str = "val_loss") -> pd.DataFrame:
    """Same experiment, several seeds. Prints the spread.

    ES: Esa dispersión es tu umbral de credibilidad: por debajo de ella,
    ninguna diferencia es un resultado. Se mide en serio en N13.
    """
    run_ids = []
    for seed in range(n_seeds):
        config = dict(config, name=f"{config.get('name', 'run')}-rep")
        print(f"▶ seed {seed}")
        run_ids.append(run_experiment(config, seed=seed, verbose=False).run_id)

    table = compare_runs(run_ids, metric)
    mean, std = table["final"].mean(), table["final"].std()
    print(f"\n{metric}: mean {mean:.5f} · std {std:.5f} · "
          f"range [{table['final'].min():.5f}, {table['final'].max():.5f}]")
    print(f"→ A difference smaller than ~{2 * std:.5f} is NOT a result.")
    return table
