# %% [markdown]
# # 6 · Puente a PyTorch
#
# Hemos escrito una red a mano y entendemos cada línea. Ahora traducimos lo mismo
# a PyTorch, comprobamos que da resultados equivalentes, y a partir de ahí hacemos
# lo que en numpy puro sería inviable.
#
# Este orden importa. Empezar por el framework te deja con una caja negra que
# funciona; llegar a él desde la implementación manual te deja sabiendo qué hace
# cada llamada por dentro.

# %%
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(["git", "clone", "-q", "https://github.com/TU_USUARIO/dl-lab.git", "/content/dl-lab"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/dl-lab"], check=True)
    sys.path.insert(0, "/content/dl-lab/src")

# %%
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from dllab import describe_device, get_device, set_seed
from dllab.data.mnist import mnist_loaders
from dllab.training import train

set_seed(42)
device = get_device()
print(describe_device(device))

train_dl, val_dl, test_dl = mnist_loaders(batch_size=10, flatten=True)


# %% [markdown]
# ## La misma red, en cinco líneas
#
# Compara mentalmente con `src/dllab/nielsen/network.py`. Todo lo que allí
# escribimos —el bucle de backprop, la acumulación de gradientes por minilote,
# la actualización de pesos— aquí lo cubre `autograd` y el optimizador.

# %%
class RedSigmoide(nn.Module):
    """Equivalente a Red([784, 30, 10]) del notebook 4."""

    def __init__(self, n_ocultas: int = 30):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(784, n_ocultas),
            nn.Sigmoid(),
            nn.Linear(n_ocultas, 10),
        )

    def forward(self, x):
        return self.capas(x)


modelo = RedSigmoide()
print(modelo)
print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")

# %% [markdown]
# Una diferencia deliberada: no ponemos sigmoide en la salida. `CrossEntropyLoss`
# de PyTorch aplica internamente log-softmax, que es la versión multiclase de la
# entropía cruzada del notebook 5, y lo hace de forma numéricamente más estable
# que si lo encadenáramos a mano.
#
# ## Entrenar

# %%
t0 = time.perf_counter()
hist = train(modelo, train_dl, val_dl, epochs=15, lr=0.5, optimizer="sgd", device=device)
t_torch = time.perf_counter() - t0

print(f"\nAcierto en validación: {hist.val_acc[-1]:.2%}")
print(f"Tiempo: {t_torch:.0f} s")


# %% [markdown]
# Resultado equivalente al de nuestra implementación manual, en una fracción del
# tiempo. La ganancia no viene de un algoritmo distinto —es el mismo backprop—
# sino de que PyTorch procesa el minilote entero como una operación matricial en
# vez de recorrerlo ejemplo a ejemplo con un bucle de Python.
#
# ## Y ahora, lo que antes no podíamos hacer
#
# Tres cambios que en numpy habrían costado un buen rato implementar:
# activaciones ReLU, optimizador Adam, y una capa más.

# %%
class RedModerna(nn.Module):
    def __init__(self):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.capas(x)


train_dl_128, val_dl_128, test_dl_128 = mnist_loaders(batch_size=128, flatten=True)

moderna = RedModerna()
hist_moderna = train(moderna, train_dl_128, val_dl_128, epochs=15, lr=1e-3,
                     optimizer="adamw", device=device)
print(f"\nAcierto en validación: {hist_moderna.val_acc[-1]:.2%}")


# %% [markdown]
# ## Recuperar la estructura espacial: una convolucional
#
# En el notebook 1 señalamos que aplanar la imagen destruye la información de
# vecindad entre píxeles. Una red convolucional la aprovecha: cada filtro es un
# detector pequeño que se desliza por toda la imagen, y los mismos pesos sirven
# en cualquier posición.

# %%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rasgos = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.clasificador(self.rasgos(x))


# esta vez SIN aplanar: la red quiere las imágenes en 2D
train_img, val_img, test_img = mnist_loaders(batch_size=128, flatten=False)

convnet = ConvNet()
hist_conv = train(convnet, train_img, val_img, epochs=8, lr=1e-3,
                  optimizer="adamw", device=device)
print(f"\nAcierto en validación: {hist_conv.val_acc[-1]:.2%}")

# %%
plt.figure(figsize=(9, 4.5))
for etiqueta, h in [
    ("sigmoide 30 (nb 4)", hist),
    ("MLP moderno", hist_moderna),
    ("convolucional", hist_conv),
]:
    plt.plot(range(1, len(h.val_acc) + 1), [a * 100 for a in h.val_acc], "o-", ms=4, label=etiqueta)
plt.xlabel("época")
plt.ylabel("acierto en validación (%)")
plt.ylim(90, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## El recorrido completo
#
# | Método | Acierto | Dónde |
# |---|---|---|
# | Azar | 10% | — |
# | Plantillas | ~82% | nb 1 |
# | Red sigmoide, coste cuadrático | ~95% | nb 4 |
# | + entropía cruzada, L2, init | ~97-98% | nb 5 |
# | MLP moderno (ReLU + Adam) | ~98% | aquí |
# | Convolucional | ~99% | aquí |
#
# Del 82% al 99% es reducir el error de 1 de cada 5 a 1 de cada 100.
#
# ## Convertirlo en un experimento reproducible
#
# Los notebooks son para explorar. Cuando una configuración merece la pena, se
# fija en un YAML y se lanza desde la línea de comandos, con su carpeta de
# resultados, su checkpoint y su registro:
#
# ```bash
# python scripts/train.py --config experiments/mnist_convnet.yaml
# ```
#
# Ése es el flujo de trabajo del laboratorio: **explorar en notebook, consolidar
# en script**.
#
# ---
#
# ## Ejercicios
#
# 1. Reproduce con PyTorch el experimento del coste cuadrático frente a la
#    entropía cruzada (`nn.MSELoss` vs `nn.CrossEntropyLoss`). ¿Se ve el mismo
#    frenazo inicial que en el notebook 5?
# 2. Sustituye las sigmoides por ReLU en `RedSigmoide`, dejando todo lo demás
#    igual. ¿Cuánto gana? ¿Por qué ReLU no sufre el problema de saturación?
# 3. Mira los filtros aprendidos por la primera capa convolucional
#    (`convnet.rasgos[0].weight`) como imágenes de 3×3. Compáralos con los pesos
#    de las neuronas ocultas del notebook 4.
# 4. Añade aumento de datos (pequeñas rotaciones y traslaciones con
#    `torchvision.transforms`). Es el camino habitual para pasar del 99% al
#    99,5%.
