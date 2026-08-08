"""Bucle de entrenamiento mínimo pero completo: métricas, early stopping y TensorBoard.

Lo importante es que sea el MISMO código el que se ejecute desde un notebook y
desde `scripts/train.py`, para que un experimento exploratorio se pueda promover
a run reproducible sin reescribir nada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    @property
    def best_epoch(self) -> int:
        return int(min(range(len(self.val_loss)), key=lambda i: self.val_loss[i])) if self.val_loss else -1


def _build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizador desconocido: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: str = "adamw",
    patience: int = 0,
    device: torch.device | None = None,
    log_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    progress: bool = True,
) -> History:
    """Entrena y devuelve el histórico. Guarda el mejor checkpoint si `ckpt_path`."""
    device = device or torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = _build_optimizer(optimizer, model.parameters(), lr, weight_decay)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))

    history = History()
    best_val, bad_epochs = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, correct, n = 0.0, 0, 0
        it = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not progress)
        for xb, yb in it:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.size(0)
            it.set_postfix(loss=f"{running / n:.4f}")

        tr_loss, tr_acc = running / max(n, 1), correct / max(n, 1)
        va_loss, va_acc = evaluate(model, val_dl, criterion, device)

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        if writer is not None:
            writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
            writer.add_scalars("accuracy", {"train": tr_acc, "val": va_acc}, epoch)

        if va_loss < best_val - 1e-5:
            best_val, bad_epochs = va_loss, 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stopping en la época {epoch} (mejor val_loss={best_val:.4f})")
                break

    if writer is not None:
        writer.close()
    return history
