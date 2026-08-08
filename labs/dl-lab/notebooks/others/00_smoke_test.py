# %% [markdown]
# # 00 · Smoke test del laboratorio
#
# Comprueba que el entorno funciona (torch, dispositivo, paquete `dllab`) y entrena
# una red diminuta de principio a fin.
#
# Este notebook funciona **igual en local (Docker) y en Google Colab**: la primera
# celda detecta dónde se está ejecutando y prepara el entorno.

# %%
# --- Bootstrap: local o Colab ---------------------------------------------
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/TU_USUARIO/dl-lab.git"  # <-- cambia esto

if IN_COLAB:
    subprocess.run(["git", "clone", "-q", REPO_URL, "/content/dl-lab"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/dl-lab"], check=True)
    sys.path.insert(0, "/content/dl-lab/src")

print("Colab" if IN_COLAB else "Local (Docker)")

# %%
import matplotlib.pyplot as plt
import torch

from dllab import describe_device, get_device, set_seed

set_seed(42)
device = get_device()
print(describe_device(device))

# %% [markdown]
# ## Datos y modelo

# %%
from dllab.data.synthetic import make_moons_loaders
from dllab.models.mlp import MLP

train_dl, val_dl, in_features, n_classes = make_moons_loaders(n_samples=4000, noise=0.2)
model = MLP(in_features=in_features, n_classes=n_classes, hidden=[64, 64], dropout=0.1)
print(model)
print(f"Parámetros entrenables: {model.n_params:,}")

# %% [markdown]
# ## Entrenamiento
#
# `log_dir` escribe eventos de TensorBoard en `outputs/runs/`; en local puedes verlos
# con `make tb` y abrir http://localhost:6006.

# %%
from dllab.training import train

history = train(
    model,
    train_dl,
    val_dl,
    epochs=25,
    lr=1e-3,
    device=device,
    log_dir=None if IN_COLAB else "outputs/runs/smoke_test",
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.train_loss, label="train")
axes[0].plot(history.val_loss, label="val")
axes[0].set_title("Loss")
axes[1].plot(history.train_acc, label="train")
axes[1].plot(history.val_acc, label="val")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.set_xlabel("época")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Frontera de decisión

# %%
import numpy as np

X = torch.cat([xb for xb, _ in val_dl]).numpy()
y = torch.cat([yb for _, yb in val_dl]).numpy()

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
)
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

model.eval()
with torch.no_grad():
    zz = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, alpha=0.25, levels=1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="coolwarm", edgecolors="none")
plt.title("Frontera de decisión aprendida")
plt.show()

# %% [markdown]
# Si has llegado hasta aquí, el laboratorio está listo.
#
# Siguiente paso: copia este notebook a `01_...`, o lanza el mismo experimento
# de forma reproducible con
# `python scripts/train.py --config experiments/mlp_moons.yaml`.
