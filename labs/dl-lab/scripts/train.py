#!/usr/bin/env python
"""Punto de entrada para runs reproducibles.

    python scripts/train.py --config experiments/mlp_moons.yaml
    python scripts/train.py --config experiments/mlp_moons.yaml --override train.lr=3e-4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dllab.config import ExperimentConfig
from dllab.data.synthetic import make_moons_loaders
from dllab.models.mlp import build_model
from dllab.training import train
from dllab.utils.device import describe_device, get_device
from dllab.utils.seed import set_seed


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, field_name = key.partition(".")
        target = getattr(cfg, section) if field_name else cfg
        attr = field_name or section
        current = getattr(target, attr)
        value = json.loads(raw) if raw.strip().startswith(("[", "{")) else type(current)(raw)
        setattr(target, attr, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--override", nargs="*", default=[], help="p.ej. train.lr=3e-4")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args.override)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(describe_device(device))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.outdir / "runs" / f"{cfg.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    train_dl, val_dl, in_features, n_classes = make_moons_loaders(
        n_samples=cfg.data.n_samples,
        noise=cfg.data.noise,
        val_split=cfg.data.val_split,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )

    model = build_model(
        cfg.model.name,
        in_features=in_features,
        n_classes=n_classes,
        hidden=cfg.model.hidden,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
    )
    print(model)

    history = train(
        model,
        train_dl,
        val_dl,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        patience=cfg.train.patience,
        device=device,
        log_dir=run_dir / "tb",
        ckpt_path=run_dir / "best.pt",
    )

    best = history.best_epoch
    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best + 1,
        "val_loss": history.val_loss[best],
        "val_acc": history.val_acc[best],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
