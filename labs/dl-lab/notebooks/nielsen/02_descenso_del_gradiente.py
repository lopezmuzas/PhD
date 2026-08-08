# %% [markdown]
# # 2 · Aprender es bajar una colina
#
# **Capítulo 1 del libro, segunda mitad.**
#
# Tenemos 23.860 parámetros que ajustar y ninguna forma de elegirlos a mano.
# Este notebook construye el procedimiento que los encuentra solos.

# %%
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(["git", "clone", "-q", "https://github.com/TU_USUARIO/dl-lab.git", "/content/dl-lab"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/dl-lab"], check=True)
    sys.path.insert(0, "/content/dl-lab/src")

# %%
import matplotlib.pyplot as plt
import numpy as np


# %% [markdown]
# ## Necesitamos una medida del error
#
# Queremos maximizar el número de dígitos acertados. Pero no podemos optimizar
# eso directamente, y la razón es sutil e importante:
#
# **El número de aciertos es una función escalonada de los pesos.** Si mueves un
# peso un poquito, casi siempre aciertas exactamente los mismos dígitos que antes.
# La medida no se inmuta hasta que un ejemplo cruza la frontera de decisión, y
# entonces salta de golpe. Es el mismo problema del perceptrón, otra vez.
#
# Necesitamos algo que responda **suavemente** a cambios pequeños. El coste
# cuadrático:
#
# $$C(w, b) = \frac{1}{2n} \sum_x \left\| y(x) - a(x) \right\|^2$$
#
# donde $y(x)$ es la respuesta correcta (el vector one-hot) y $a(x)$ lo que
# produce la red. Es una media de distancias al cuadrado: vale 0 si la red acierta
# perfectamente y crece cuanto peor lo hace.
#
# Fíjate en la maniobra: **el coste es un sustituto**. No es lo que nos importa
# —nos importan los aciertos—, pero sí es algo que podemos optimizar, y bajarlo
# suele subir los aciertos. Esta distinción entre la métrica que te importa y la
# función que minimizas aparece en todo el aprendizaje automático.

# %% [markdown]
# ## La intuición del valle
#
# Imagina el coste como la altura de un terreno, y los pesos como tus
# coordenadas. Buscamos el punto más bajo. Estamos a ciegas —el terreno tiene
# 23.860 dimensiones—, pero podemos palpar la pendiente bajo nuestros pies y dar
# un paso cuesta abajo. Repetir.
#
# Formalmente: si nos movemos una cantidad pequeña $\Delta v$, el coste cambia
# aproximadamente
#
# $$\Delta C \approx \nabla C \cdot \Delta v$$
#
# Queremos que $\Delta C$ sea negativo. La elección
#
# $$\Delta v = -\eta \nabla C$$
#
# lo garantiza, porque entonces $\Delta C \approx -\eta \|\nabla C\|^2$, que es
# negativo siempre. El parámetro $\eta > 0$ es la **tasa de aprendizaje**: el
# tamaño del paso.
#
# Vamos a verlo sobre una superficie de dos variables, donde sí podemos dibujar.

# %%
def coste_juguete(v):
    """Un valle alargado. El mínimo está en (0, 0)."""
    x, y = v
    return 0.5 * (x**2 + 8 * y**2)


def gradiente_juguete(v):
    x, y = v
    return np.array([x, 8 * y])


def descender(v0, eta, pasos=40):
    v = np.array(v0, dtype=float)
    camino = [v.copy()]
    for _ in range(pasos):
        v = v - eta * gradiente_juguete(v)
        camino.append(v.copy())
    return np.array(camino)


xs = np.linspace(-5, 5, 300)
ys = np.linspace(-2.5, 2.5, 300)
XX, YY = np.meshgrid(xs, ys)
ZZ = 0.5 * (XX**2 + 8 * YY**2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, eta in zip(axes, [0.02, 0.12, 0.245]):
    ax.contour(XX, YY, ZZ, levels=25, cmap="Blues", alpha=0.7)
    camino = descender([-4.5, 2.0], eta)
    ax.plot(camino[:, 0], camino[:, 1], "o-", ms=3, lw=1, color="crimson")
    ax.plot(0, 0, "k*", ms=14)
    ax.set_title(f"η = {eta}\ncoste final: {coste_juguete(camino[-1]):.4f}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.5, 2.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# Los tres paneles cuentan la historia completa de la tasa de aprendizaje:
#
# - **η pequeña**: cada paso es diminuto, el descenso es seguro pero lentísimo.
#   No llega al fondo en 40 pasos.
# - **η adecuada**: baja rápido y converge.
# - **η grande**: los pasos sobrepasan el mínimo y rebota de lado a lado. Todavía
#   converge, pero mal. Un poco más grande y **divergiría**, alejándose cada vez
#   más.
#
# Observa también el zigzag en el tercer panel. Ocurre porque el valle es más
# empinado en un eje que en otro. Con 23.860 dimensiones esto es la norma, no la
# excepción, y es la motivación de optimizadores más listos como Adam.

# %%
# ¿Qué pasa si nos pasamos de verdad?
for eta in [0.24, 0.25, 0.26]:
    camino = descender([-4.5, 2.0], eta, pasos=30)
    final = coste_juguete(camino[-1])
    estado = "diverge" if final > 1e3 else "converge"
    print(f"η = {eta}: coste final {final:>12.4g}   {estado}")

# %% [markdown]
# El salto entre 0,25 y 0,26 no es gradual: es un umbral. Por encima de él, la
# optimización se rompe. Cuando veas pérdidas que se vuelven `NaN`, este suele
# ser el motivo.

# %% [markdown]
# ## De gradiente a gradiente estocástico
#
# La definición de $C$ promedia sobre **todos** los ejemplos de entrenamiento.
# Calcular $\nabla C$ exactamente exige recorrer los 50.000 dígitos para dar
# **un solo paso**. Inviable.
#
# La solución es una apuesta estadística: coger una muestra aleatoria pequeña
# —un **minilote** de, digamos, 10 ejemplos— y usar su gradiente medio como
# estimación del verdadero.
#
# $$\nabla C \approx \frac{1}{m}\sum_{j=1}^{m} \nabla C_{x_j}$$
#
# La estimación es ruidosa, pero es **5.000 veces más barata**. Damos muchos
# pasos aproximados en vez de uno exacto, y salimos ganando con mucho margen.
# Cuando se han usado todos los ejemplos una vez, se ha completado una **época**.
#
# Veámoslo: el mismo problema, con gradiente exacto y con gradiente ruidoso.

# %%
rng = np.random.default_rng(0)


def descender_ruidoso(v0, eta, pasos=40, ruido=1.5):
    v = np.array(v0, dtype=float)
    camino = [v.copy()]
    for _ in range(pasos):
        g = gradiente_juguete(v) + rng.normal(0, ruido, size=2)
        v = v - eta * g
        camino.append(v.copy())
    return np.array(camino)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (titulo, camino) in zip(
    axes,
    [
        ("Gradiente exacto", descender([-4.5, 2.0], 0.12)),
        ("Gradiente estimado (ruidoso)", descender_ruidoso([-4.5, 2.0], 0.12)),
    ],
):
    ax.contour(XX, YY, ZZ, levels=25, cmap="Blues", alpha=0.7)
    ax.plot(camino[:, 0], camino[:, 1], "o-", ms=3, lw=1, color="crimson")
    ax.plot(0, 0, "k*", ms=14)
    ax.set_title(f"{titulo}\ncoste final: {coste_juguete(camino[-1]):.4f}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.5, 2.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# El camino ruidoso es feo, tiembla y no se posa exactamente en el mínimo: se
# queda dando vueltas en una zona baja. Para nuestros fines, eso basta. Y el
# ruido tiene una ventaja inesperada que se aprecia en problemas reales: ayuda a
# escapar de mínimos locales malos en los que un descenso exacto se quedaría
# atrapado.
#
# ## El pseudocódigo completo
#
# ```
# para cada época:
#     barajar los datos de entrenamiento
#     partirlos en minilotes de tamaño m
#     para cada minilote:
#         calcular el gradiente medio del minilote     <-- backpropagation
#         w ← w − (η/m) · Σ ∂C/∂w
#         b ← b − (η/m) · Σ ∂C/∂b
# ```
#
# Todo está resuelto salvo una línea: **cómo se calcula ese gradiente**. Con
# 23.860 parámetros, y necesitando la derivada respecto a cada uno, hace falta
# un algoritmo eficiente.
#
# Ese algoritmo es backpropagation, y es el notebook siguiente.
#
# ---
#
# ## Ejercicios
#
# 1. Cambia `coste_juguete` por una función con dos mínimos (por ejemplo
#    añadiendo un término coseno). Lanza el descenso desde varios puntos de
#    partida. ¿Siempre encuentra el mismo?
# 2. En el ejemplo ruidoso, prueba a reducir η progresivamente
#    (`eta_t = eta_0 / (1 + t/10)`). ¿Se posa mejor en el mínimo? Esto es un
#    *learning rate schedule*.
# 3. Estima cuántas operaciones cuesta una época con minilotes de 10 frente a
#    usar los 50.000 ejemplos de golpe. ¿Cuántos pasos de gradiente da cada uno?
