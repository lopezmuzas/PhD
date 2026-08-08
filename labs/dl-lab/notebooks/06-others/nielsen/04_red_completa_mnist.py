# %% [markdown]
# # 4 · La red completa sobre MNIST
#
# **Cierre del capítulo 1.** Juntamos las piezas y entrenamos.
#
# Si vienes de los tres notebooks anteriores, ya lo tienes todo: una arquitectura
# (784→30→10), una medida del error (coste cuadrático), un método para bajarlo
# (SGD) y un algoritmo para calcular el gradiente (backprop). Aquí solo hay que
# ensamblarlo y darle a ejecutar.
#
# ⏱️ **Aviso de tiempo**: el entrenamiento completo tarda unos 3-8 minutos en CPU.
# Es numpy puro, sin GPU y sin vectorizar entre ejemplos, exactamente igual que en
# el libro. Esa lentitud es informativa: te hace sentir en los dedos por qué
# existen los frameworks.

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

from dllab.nielsen.data import load_mnist, load_mnist_arrays
from dllab.nielsen.network import Red
from dllab.nielsen.viz import curva_aprendizaje, matriz_confusion, mostrar_digitos, mostrar_pesos

entrenamiento, validacion, test = load_mnist()
print(f"{len(entrenamiento):,} ejemplos de entrenamiento")
print(f"Forma de una x: {entrenamiento[0][0].shape}   de una y: {entrenamiento[0][1].shape}")

# %% [markdown]
# ## Primero, un ensayo corto
#
# Antes de lanzar 30 épocas conviene comprobar que la cosa aprende. Tres épocas
# bastan para ver la tendencia.

# %%
red_prueba = Red([784, 30, 10], seed=1)
print(f"Acierto antes de entrenar: {red_prueba.evaluar(validacion) / len(validacion):.2%}")

hist_prueba = red_prueba.sgd(
    entrenamiento, epocas=3, tam_minilote=10, eta=3.0, datos_eval=validacion, seed=1
)

# %% [markdown]
# Del 10% (azar puro) a más del 90% en tres pasadas. Y ya en la primera época
# supera con holgura el 82% de las plantillas del notebook 1.
#
# ## El entrenamiento completo
#
# Los hiperparámetros son los del libro: 30 épocas, minilotes de 10, η = 3,0.
# Esa η parecería enorme en un framework moderno; funciona aquí porque el coste
# cuadratico con sigmoides produce gradientes muy pequeños.

# %%
red = Red([784, 30, 10], seed=42)
hist = red.sgd(
    entrenamiento, epocas=30, tam_minilote=10, eta=3.0, datos_eval=validacion, seed=42
)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(range(1, len(hist.acierto_eval) + 1), np.array(hist.acierto_eval) * 100, "o-", ms=4)
ax1.axhline(82, ls="--", color="gray", label="plantillas (nb 1)")
ax1.set_xlabel("época")
ax1.set_ylabel("acierto en validación (%)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(range(1, len(hist.coste_entrenamiento) + 1), hist.coste_entrenamiento, "o-", ms=4, color="crimson")
ax2.set_xlabel("época")
ax2.set_ylabel("coste de entrenamiento")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mejor acierto en validación: {max(hist.acierto_eval):.2%}")
print(f"Tiempo total: {sum(hist.segundos):.0f} s")

# %% [markdown]
# Alrededor del 95%. Una red de 30 neuronas ocultas, escrita en 150 líneas de
# numpy, clasifica correctamente 19 de cada 20 dígitos manuscritos que no ha
# visto nunca.
#
# Merece la pena detenerse un segundo en lo que significa. En ningún momento le
# hemos dicho qué es un bucle, ni un trazo, ni una esquina. Solo le hemos dado
# ejemplos y un procedimiento para reducir un error.
#
# ## Ahora la medición honesta: el test
#
# Hemos mirado validación 30 veces. El test lo miramos **una sola vez**.

# %%
aciertos_test = red.evaluar(test)
print(f"Acierto en test: {aciertos_test} / {len(test)} = {aciertos_test/len(test):.2%}")

# %% [markdown]
# ## ¿En qué se equivoca?
#
# El número global esconde lo interesante. Vamos a mirar los errores.

# %%
y_true = np.array([y for _, y in test])
y_pred = np.array([red.predecir(x) for x, _ in test])

M, fig = matriz_confusion(y_true, y_pred)
plt.show()

acierto_por_clase = [(d, (y_pred[y_true == d] == d).mean()) for d in range(10)]
print("Acierto por dígito (de peor a mejor):")
for d, a in sorted(acierto_por_clase, key=lambda t: t[1]):
    print(f"  {d}: {a:.2%}")

# %%
# Las confusiones más frecuentes
confusiones = [(M[i, j], i, j) for i in range(10) for j in range(10) if i != j]
print("Confusiones más habituales:")
for n, real, pred in sorted(confusiones, reverse=True)[:6]:
    print(f"  {n:3d} veces confundió un {real} con un {pred}")

# %%
(X_te, y_te) = load_mnist_arrays()[2]
fallos = np.where(y_pred != y_true)[0][:20]
mostrar_digitos(
    X_te[fallos], y_true[fallos], n=20, pred=y_pred[fallos],
    titulo="Errores de la red (predicho ≠ real)",
)
plt.show()

# %% [markdown]
# Muchos de esos fallos son dígitos que a un humano también le costarían. Otros
# son claramente legibles y la red los yerra igualmente. Esa distinción marca el
# techo de lo que se puede mejorar: los primeros son ruido irreducible del
# conjunto de datos; los segundos, margen real de mejora.
#
# ## ¿Qué ha aprendido cada neurona oculta?
#
# Cada una de las 30 neuronas ocultas tiene 784 pesos entrantes, uno por píxel.
# Si los reordenamos en 28×28 podemos verlos como una imagen: el patrón al que
# esa neurona responde.

# %%
mostrar_pesos(red, capa=0, n=30)
plt.show()

# %% [markdown]
# No se ven dígitos reconocibles. Se ven manchas, trazos difusos, regiones
# positivas y negativas. Cada neurona detecta algo así como "hay tinta en esta
# zona y no en esta otra", y la capa de salida combina 30 de esas pistas.
#
# Esto es honesto sobre lo que hemos construido: **no hay comprensión, hay
# correlaciones útiles**. Y también anticipa por qué las convolucionales
# funcionan mejor: sus filtros sí aprenden detectores de bordes localizados y
# reutilizables en toda la imagen.

# %% [markdown]
# ## Experimentos
#
# Aquí es donde el notebook se convierte en laboratorio. Cada celda tarda unos
# minutos; lánzalas y compara.
#
# ### ¿Cuántas neuronas ocultas hacen falta?

# %%
resultados = {}
for n_ocultas in [10, 30, 100]:
    r = Red([784, n_ocultas, 10], seed=7)
    h = r.sgd(entrenamiento, epocas=10, tam_minilote=10, eta=3.0,
              datos_eval=validacion, verbose=False, seed=7)
    resultados[f"{n_ocultas} ocultas"] = h
    print(f"{n_ocultas:3d} ocultas -> {h.acierto_eval[-1]:.2%}  ({sum(h.segundos):.0f}s)")

curva_aprendizaje(resultados)
plt.ylabel("acierto en validación")
plt.show()

# %% [markdown]
# Más neuronas ayudan, pero con rendimientos decrecientes y coste lineal en
# tiempo. Pasar de 10 a 30 gana mucho; de 30 a 100, bastante menos.
#
# ### La tasa de aprendizaje

# %%
resultados_eta = {}
for eta in [0.1, 1.0, 3.0, 10.0]:
    r = Red([784, 30, 10], seed=7)
    h = r.sgd(entrenamiento, epocas=8, tam_minilote=10, eta=eta,
              datos_eval=validacion, verbose=False, seed=7)
    resultados_eta[f"η = {eta}"] = h
    print(f"η = {eta:5.1f} -> {h.acierto_eval[-1]:.2%}")

curva_aprendizaje(resultados_eta)
plt.ylabel("acierto en validación")
plt.show()

# %% [markdown]
# Se reconoce el patrón del notebook 2: demasiado pequeña avanza a paso de
# tortuga, demasiado grande se vuelve inestable.
#
# ---
#
# ## Dónde estamos
#
# | Método | Acierto |
# |---|---|
# | Azar | 10% |
# | Plantillas (media por clase) | ~82% |
# | Red 784→30→10, coste cuadrático | ~95% |
#
# Ese 95% deja un 5% de fallos. El notebook siguiente lo baja a la mitad **sin
# cambiar la arquitectura**: solo cambiando la función de coste, añadiendo
# regularización y mejorando la inicialización.
#
# ---
#
# ## Ejercicios
#
# 1. Entrena una red **sin capa oculta** ([784, 10]). ¿Cuánto acierta? El
#    resultado es más alto de lo que uno esperaría: ¿qué está haciendo esa red?
# 2. Prueba tamaños de minilote de 1, 10 y 100 con el mismo η. Compara acierto
#    **y tiempo por época**. ¿Cuál es el mejor compromiso?
# 3. Guarda los pesos de la red entrenada con `np.savez` y escribe una función
#    que cargue y clasifique una imagen suelta.
# 4. Coge los 20 errores de arriba y clasifícalos tú a mano. ¿Con cuántos aciertas
#    tú? Eso estima el techo humano en esas imágenes concretas.
