# %%
# 1 · El problema: enseñar a una máquina a leer dígitos

**Serie basada en [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/)
de Michael Nielsen** — capítulo 1, primera mitad.

El código y las explicaciones de estos notebooks son originales; siguen el
recorrido pedagógico del libro, que es lo verdaderamente valioso de él.
Te recomiendo leer el capítulo correspondiente antes o después de cada notebook.

---

## Por qué los dígitos manuscritos

Reconocer un `5` escrito a mano es algo que haces sin pensar y que resulta
extraordinariamente difícil de programar. Ese contraste es el argumento central
del libro, y merece la pena sentirlo antes de escribir una sola red.

Intenta definir con reglas qué es un 5: "un trazo horizontal arriba, una línea
que baja por la izquierda, una curva que se abre a la derecha". En cuanto lo
escribes, aparecen los contraejemplos: los 5 sin el trazo superior, los que se
cierran como un 6, los inclinados. Cada regla necesita excepciones, y las
excepciones necesitan excepciones.

La idea de las redes neuronales es no escribir esas reglas. Se le dan al sistema
miles de ejemplos etiquetados y se deja que **infiera** las reglas él mismo.

# %%
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/TU_USUARIO/dl-lab.git"

if IN_COLAB:
    subprocess.run(["git", "clone", "-q", REPO_URL, "/content/dl-lab"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/dl-lab"], check=True)
    sys.path.insert(0, "/content/dl-lab/src")

# %%
import matplotlib.pyplot as plt
import numpy as np

from dllab.nielsen.data import load_mnist_arrays
from dllab.nielsen.viz import mostrar_digitos

# %% [markdown]
# ## MNIST
#
# 70.000 imágenes de dígitos manuscritos, en escala de grises, de 28×28 píxeles.
# Procede de formularios del censo estadounidense y de estudiantes de instituto,
# y lleva desde los años noventa siendo el banco de pruebas de referencia.
#
# La partición que usamos es la del libro:
#
# | Conjunto | Tamaño | Para qué |
# |---|---|---|
# | Entrenamiento | 50.000 | Ajustar los pesos |
# | Validación | 10.000 | Elegir hiperparámetros |
# | Test | 10.000 | Medir una sola vez, al final |
#
# Separar validación de test importa: si eliges la tasa de aprendizaje mirando el
# test, el test deja de ser una estimación honesta de cómo funcionará con datos
# nuevos. Habrás ajustado tus decisiones a él.

# %%
(X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_mnist_arrays()

print(f"Entrenamiento: {X_tr.shape}   etiquetas {y_tr.shape}")
print(f"Validación:    {X_val.shape}")
print(f"Test:          {X_te.shape}")
print(f"Rango de valores: [{X_tr.min():.1f}, {X_tr.max():.1f}]")

# %%
mostrar_digitos(X_tr[:20], y_tr[:20], n=20, titulo="Veinte ejemplos de entrenamiento")
plt.show()

# %% [markdown]
# ## Una imagen es un vector de 784 números
#
# La red no ve una imagen: ve 784 números entre 0 y 1, uno por píxel, ordenados
# en fila. Toda la estructura espacial —que dos píxeles vecinos están juntos— se
# pierde en ese aplanado.
#
# Merece la pena detenerse aquí: **esa pérdida es una limitación real** de esta
# arquitectura. Si barajaras los 784 píxeles siempre en el mismo orden, la red
# aprendería exactamente igual de bien. Las redes convolucionales, que veremos al
# final de la serie, existen precisamente para recuperar esa información.

# %%
digito = X_tr[0].reshape(28, 28)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1, 2.5])
ax1.imshow(digito, cmap="gray_r")
ax1.set_title(f"Etiqueta: {y_tr[0]}")
ax1.axis("off")
ax2.plot(X_tr[0], lw=0.8)
ax2.set_title("Los mismos datos: 784 valores en fila")
ax2.set_xlabel("índice del píxel")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## El intento ingenuo: comparar con una plantilla
#
# Antes de las redes, probemos lo más simple que se nos ocurra. Calculamos la
# imagen media de cada dígito y clasificamos cada imagen nueva asignándole la
# plantilla más parecida. Es un clasificador razonable y nos da un **suelo**:
# cualquier método que no lo supere no está aportando nada.

# %%
plantillas = np.stack([X_tr[y_tr == d].mean(axis=0) for d in range(10)])
mostrar_digitos(plantillas, np.arange(10), n=10, titulo="La imagen media de cada dígito")
plt.show()

# distancia euclídea de cada imagen de test a cada plantilla
distancias = ((X_te[:, None, :] - plantillas[None, :, :]) ** 2).sum(axis=2)
pred_plantilla = distancias.argmin(axis=1)
acierto_plantilla = (pred_plantilla == y_te).mean()

print(f"Acierto con plantillas: {acierto_plantilla:.2%}")
print(f"Acierto de un clasificador al azar: {1/10:.2%}")


# %% [markdown]
# Alrededor del 82%. Suena bien comparado con el 10% del azar, pero significa que
# **falla casi 1 de cada 5 veces**. Ese es el listón que hay que batir.
#
# El problema de fondo es que la media borra la variabilidad: un 1 inclinado y un
# 1 vertical se promedian en una mancha que no se parece a ninguno de los dos.
#
# ## Perceptrones: la primera idea
#
# Un perceptrón toma varias entradas binarias y produce una salida binaria.
# Cada entrada tiene un peso, y la neurona se activa si la suma ponderada supera
# un umbral:
#
# $$
# \text{salida} =
# \begin{cases}
# 0 & \text{si } \sum_j w_j x_j + b \le 0 \\
# 1 & \text{si } \sum_j w_j x_j + b > 0
# \end{cases}
# $$
#
# donde $b$ es el **sesgo** (el umbral, cambiado de signo). Interpretación útil:
# el sesgo mide lo predispuesta que está la neurona a activarse.
#
# Los perceptrones pueden implementar cualquier función lógica —con puertas NAND
# se construye cualquier circuito—, así que en principio pueden calcular
# cualquier cosa. El problema es otro: **no se pueden entrenar bien**.

# %%
def perceptron(x, w, b):
    return int(np.dot(w, x) + b > 0)


# Una puerta NAND: se activa salvo cuando las dos entradas valen 1
w, b = np.array([-2, -2]), 3
print("Puerta NAND con un perceptrón:")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"  {x} -> {perceptron(np.array(x), w, b)}")

# %% [markdown]
# ### Por qué no se pueden entrenar
#
# Aprender consiste en ajustar pesos poco a poco y comprobar si el resultado
# mejora. Con perceptrones eso no funciona: la salida es un escalón. Un cambio
# minúsculo en un peso o no cambia nada, o hace que la neurona salte de 0 a 1 de
# golpe, y ese salto se propaga en cascada por toda la red.
#
# No hay forma de "afinar". Es como intentar sintonizar una radio con un
# interruptor.
#
# ## Neuronas sigmoides: la solución
#
# Se sustituye el escalón por una curva suave, la función logística:
#
# $$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \sum_j w_j x_j + b$$
#
# La salida ya no es 0 o 1, sino cualquier valor intermedio. La forma general se
# parece a la del escalón —satura cerca de 0 y de 1—, pero la transición es
# gradual, y eso lo cambia todo: **ahora un cambio pequeño en un peso produce un
# cambio pequeño en la salida**. Se puede afinar.

# %%
from dllab.nielsen.network import sigmoide, sigmoide_prima

z = np.linspace(-8, 8, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(z, sigmoide(z), lw=2, label=r"$\sigma(z)$")
ax1.step(z, (z > 0).astype(float), where="mid", ls="--", color="gray", label="escalón")
ax1.set_title("Sigmoide frente a escalón")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(z, sigmoide_prima(z), lw=2, color="crimson")
ax2.set_title(r"Su derivada $\sigma'(z)=\sigma(z)(1-\sigma(z))$")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ⚠️ El panel de la derecha guarda una advertencia
#
# La derivada vale como mucho 0,25, y se desploma hacia cero cuando $|z|$ crece.
# Cuando una neurona está **saturada** (muy segura, con salida cerca de 0 o de 1)
# su derivada es casi nula. Y como veremos, el gradiente es proporcional a esa
# derivada: una neurona saturada **aprende muy despacio, aunque esté equivocada**.
#
# Este detalle reaparecerá dos veces en la serie: en el notebook 5, donde lo
# resolvemos cambiando la función de coste, y como causa del problema del
# gradiente evanescente en redes profundas.

# %% [markdown]
# ## La arquitectura: 784 → 30 → 10
#
# - **Capa de entrada**: 784 neuronas, una por píxel.
# - **Capa oculta**: 30 neuronas. El número es arbitrario; lo variaremos después.
# - **Capa de salida**: 10 neuronas, una por dígito. La respuesta de la red es el
#   índice de la neurona más activa.
#
# Una pregunta que merece pensarse: podríamos usar solo 4 neuronas de salida y
# leer el resultado en binario ($2^4 = 16 > 10$). Sería más compacto. ¿Por qué
# 10 funciona mejor en la práctica?
#
# La intuición del libro es que cada neurona de salida puede especializarse en
# reconocer evidencia de *una* forma concreta —bucles, trazos, esquinas—, mientras
# que una codificación binaria obligaría a cada neurona a responder a un conjunto
# de dígitos sin nada visual en común. La segunda neurona del código binario
# tendría que activarse para 2, 3, 6 y 7. ¿Qué rasgo comparten?

# %%
tamaños = [784, 30, 10]
n_pesos = sum(a * b for a, b in zip(tamaños[:-1], tamaños[1:]))
n_sesgos = sum(tamaños[1:])

print(f"Arquitectura: {' → '.join(map(str, tamaños))}")
print(f"Pesos:  {n_pesos:,}")
print(f"Sesgos: {n_sesgos:,}")
print(f"Total de parámetros a ajustar: {n_pesos + n_sesgos:,}")

# %% [markdown]
# Casi 24.000 números que hay que encontrar. No hay ninguna posibilidad de
# elegirlos a mano: necesitamos un procedimiento automático.
#
# Eso es el notebook siguiente.
#
# ---
#
# ## Ejercicios
#
# 1. Con las plantillas, ¿qué dígitos se confunden más? Calcula el acierto por
#    clase y mira los peores. ¿Coincide con lo que te resultaría difícil a ti?
# 2. Multiplica todos los pesos y sesgos de un perceptrón por una constante
#    positiva grande. ¿Cambia su comportamiento? ¿Y si haces lo mismo con una
#    neurona sigmoide? (Pista: dibuja $\sigma(cz)$ para $c$ grande.)
# 3. Normaliza las plantillas antes de comparar (distancia coseno en vez de
#    euclídea). ¿Mejora el 82%?
