# %% [markdown]
# # 3 · Backpropagation
#
# **Capítulo 2 del libro.** El más denso de la serie, y el que más compensa.
#
# Backpropagation es el algoritmo que calcula $\partial C/\partial w$ para los
# 23.860 parámetros a la vez. Es lo que hace viable entrenar redes, y llevaba
# décadas siendo el contenido de la caja negra de cualquier framework.
#
# Antes de entrar: **hay una alternativa obvia y hay que entender por qué no
# sirve**. Para estimar $\partial C/\partial w_j$ podríamos mover ese peso un
# poquito y ver cuánto cambia el coste:
#
# $$\frac{\partial C}{\partial w_j} \approx \frac{C(w + \epsilon e_j) - C(w)}{\epsilon}$$
#
# Funciona. Pero exige un pase completo por la red **por cada parámetro**: 23.860
# pases para un solo gradiente. Backpropagation obtiene los 23.860 con un pase
# hacia adelante y uno hacia atrás. Es unas diez mil veces más rápido, y esa
# diferencia es la que separa lo posible de lo imposible.

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
import numpy as np

from dllab.nielsen.network import Red, comprobar_gradiente, sigmoide, sigmoide_prima
from dllab.nielsen.data import one_hot

# %% [markdown]
# ## Notación
#
# Hay que fijarla o el resto no se entiende. Es la del libro:
#
# - $w^l_{jk}$: peso de la conexión que va de la neurona $k$ de la capa $l-1$
#   a la neurona $j$ de la capa $l$.
# - $b^l_j$: sesgo de la neurona $j$ de la capa $l$.
# - $z^l_j = \sum_k w^l_{jk} a^{l-1}_k + b^l_j$: la **entrada ponderada**.
# - $a^l_j = \sigma(z^l_j)$: la activación.
#
# El orden de los subíndices en $w^l_{jk}$ (destino primero, origen después)
# parece del revés. Tiene su motivo: permite escribir el paso hacia adelante como
# una multiplicación de matrices limpia, sin transponer nada:
#
# $$a^l = \sigma(w^l a^{l-1} + b^l)$$
#
# Necesitamos también el **producto de Hadamard** $s \odot t$: multiplicación
# elemento a elemento, que en numpy es simplemente `*`.

# %%
s = np.array([[1], [2]])
t = np.array([[3], [4]])
print("Hadamard:\n", s * t)

# %% [markdown]
# ## La idea central: el error de una neurona
#
# Backprop no calcula $\partial C/\partial w$ directamente. Introduce una
# cantidad intermedia, el **error** de la neurona $j$ de la capa $l$:
#
# $$\delta^l_j \equiv \frac{\partial C}{\partial z^l_j}$$
#
# Es cuánto cambia el coste si perturbas la entrada ponderada de esa neurona.
# Si $\delta^l_j$ es casi cero, esa neurona ya hace prácticamente lo mejor que
# puede; si es grande, ahí hay margen de mejora.
#
# La estrategia es: calcular $\delta$ en la capa de salida, propagarlo hacia
# atrás capa por capa, y de ahí sacar todos los gradientes.
#
# ## Las cuatro ecuaciones
#
# **BP1 — el error en la capa de salida:**
#
# $$\delta^L = \nabla_a C \odot \sigma'(z^L)$$
#
# Dos factores: cuánto cambia el coste al cambiar la salida ($\nabla_a C$), y
# cuánto cambia la salida al cambiar $z$ ($\sigma'$). Regla de la cadena pura.
# Con el coste cuadrático, $\nabla_a C = (a^L - y)$.
#
# **BP2 — propagar el error hacia atrás:**
#
# $$\delta^l = \left((w^{l+1})^T \delta^{l+1}\right) \odot \sigma'(z^l)$$
#
# Ésta es la ecuación que da nombre al algoritmo. La transpuesta $(w^{l+1})^T$
# mueve el error en sentido contrario al de la red: reparte la culpa de la capa
# $l+1$ entre las neuronas de la capa $l$, en proporción a lo mucho que cada una
# contribuyó.
#
# **BP3 — gradiente respecto a los sesgos:**
#
# $$\frac{\partial C}{\partial b^l_j} = \delta^l_j$$
#
# El error *es* el gradiente del sesgo. Sin más.
#
# **BP4 — gradiente respecto a los pesos:**
#
# $$\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \, \delta^l_j$$
#
# Producto de la activación de entrada por el error de salida. De aquí sale una
# conclusión práctica muy citada: **si $a^{l-1}_k$ es pequeña, ese peso apenas
# aprende**. Las neuronas poco activas dejan sus conexiones congeladas.

# %% [markdown]
# ## De las ecuaciones al código
#
# Vamos a ejecutarlo paso a paso sobre una red diminuta para poder mirar todas
# las dimensiones.

# %%
red = Red([4, 3, 2], seed=42)
x = np.array([[0.5], [0.1], [0.9], [0.3]])
y = one_hot(1, 2)

print("Formas de los pesos: ", [w.shape for w in red.pesos])
print("Formas de los sesgos:", [b.shape for b in red.sesgos])

# %%
# --- PASO HACIA ADELANTE, guardando todo lo que backprop necesitará
activacion = x
activaciones = [x]
zs = []

for i, (b, w) in enumerate(zip(red.sesgos, red.pesos), start=1):
    z = w @ activacion + b
    zs.append(z)
    activacion = sigmoide(z)
    activaciones.append(activacion)
    print(f"Capa {i}:  z{z.shape} -> a{activacion.shape}")

print("\nSalida de la red:\n", activaciones[-1].ravel().round(4))
print("Objetivo:\n", y.ravel())

# %% [markdown]
# Guardar las $z$ y las $a$ del pase hacia adelante no es opcional: el pase hacia
# atrás las necesita todas. Ese es el coste en memoria del algoritmo, y la razón
# de que entrenar consuma mucha más RAM que hacer inferencia.

# %%
# --- BP1: error en la capa de salida
delta = (activaciones[-1] - y) * sigmoide_prima(zs[-1])
print("δ^L =", delta.ravel().round(5))

# --- BP3 y BP4 en la última capa
grad_b_L = delta
grad_w_L = delta @ activaciones[-2].T
print("\n∂C/∂b de la última capa:", grad_b_L.ravel().round(5))
print("∂C/∂w de la última capa (forma", grad_w_L.shape, "):\n", grad_w_L.round(5))

# %%
# --- BP2: propagamos a la capa anterior
delta_anterior = (red.pesos[-1].T @ delta) * sigmoide_prima(zs[-2])
print("δ^{L-1} =", delta_anterior.ravel().round(5))
print("\nMagnitud del error por capa:")
print(f"  capa de salida: {np.abs(delta).mean():.6f}")
print(f"  capa oculta:    {np.abs(delta_anterior).mean():.6f}")

# %% [markdown]
# Fíjate en que el error de la capa oculta es notablemente **menor** que el de la
# de salida. No es casualidad: en cada paso hacia atrás multiplicamos por
# $\sigma'$, cuyo máximo es 0,25. El error se encoge a medida que retrocede.
#
# Con dos capas es inofensivo. Con veinte, el gradiente que llega a las primeras
# capas es tan minúsculo que dejan de aprender: es el **problema del gradiente
# evanescente**, capítulo 5 del libro, y la razón de que durante años no se
# supiera entrenar redes profundas.

# %% [markdown]
# ## La prueba de fuego: comprobación numérica
#
# El código de arriba es correcto o no lo es, y creerlo no basta. Comparamos el
# gradiente analítico de backprop con la aproximación por diferencias centradas:
#
# $$\frac{\partial C}{\partial w} \approx \frac{C(w+\epsilon) - C(w-\epsilon)}{2\epsilon}$$
#
# Si la diferencia relativa está por debajo de $10^{-7}$, la implementación es
# correcta. **Escribe siempre esta comprobación cuando implementes un gradiente a
# mano.** Un backprop con un signo cambiado no falla: entrena mal, y eso es mucho
# más difícil de detectar.

# %%
red_test = Red([6, 5, 4, 3], seed=1)
x_test = np.random.default_rng(0).random((6, 1))
y_test = one_hot(2, 3)

diferencias = comprobar_gradiente(red_test, x_test, y_test)
for i, d in enumerate(diferencias, start=1):
    veredicto = "correcto" if d < 1e-7 else "REVISAR"
    print(f"Capa {i}: diferencia relativa {d:.2e}   {veredicto}")

# %% [markdown]
# ## Y ahora, el argumento de la velocidad
#
# Midamos lo que decíamos al principio.

# %%
red_grande = Red([784, 30, 10], seed=0)
x_g = np.random.default_rng(1).random((784, 1))
y_g = one_hot(3)

t0 = time.perf_counter()
for _ in range(100):
    red_grande.backprop(x_g, y_g)
t_backprop = (time.perf_counter() - t0) / 100

n_params = sum(w.size for w in red_grande.pesos) + sum(b.size for b in red_grande.sesgos)

t0 = time.perf_counter()
for _ in range(200):
    red_grande.propagar(x_g)
t_forward = (time.perf_counter() - t0) / 200

print(f"Parámetros: {n_params:,}")
print(f"Un backprop completo:      {t_backprop*1000:.3f} ms")
print(f"Un pase hacia adelante:    {t_forward*1000:.3f} ms")
print(f"\nEstimación numérica del mismo gradiente:")
print(f"  {2*n_params:,} pases hacia adelante = {2*n_params*t_forward:.1f} s")
print(f"  es decir, ~{2*n_params*t_forward/t_backprop:,.0f} veces más lento")

# %% [markdown]
# Cuatro órdenes de magnitud. Y eso para **un solo ejemplo**: multiplícalo por
# 50.000 ejemplos y 30 épocas.
#
# Backprop no es una optimización menor. Es la diferencia entre entrenar una red
# en un minuto y no poder entrenarla nunca.
#
# ---
#
# ## Ejercicios
#
# 1. Cambia deliberadamente un signo en `backprop` (en `src/dllab/nielsen/network.py`)
#    y vuelve a lanzar `comprobar_gradiente`. Comprueba que lo detecta.
# 2. Añade una capa oculta más ([784, 30, 30, 10]) y mide la magnitud media de
#    $\delta$ en cada capa. ¿Cuánto se encoge por capa?
# 3. La comprobación numérica usa diferencias centradas en lugar de
#    $(C(w+\epsilon)-C(w))/\epsilon$. Prueba las dos y compara la precisión.
#    ¿Por qué la centrada es mejor? (Pista: desarrollo de Taylor.)
