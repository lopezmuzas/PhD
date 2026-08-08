# %% [markdown]
# # 5 · Tres cambios que llevan del 95% al 98%
#
# **Capítulo 3 del libro.**
#
# La arquitectura no se toca: siguen siendo 784→30→10 con neuronas sigmoides.
# Cambiamos tres cosas que no son la red, y el error se reduce a la mitad:
#
# 1. La **función de coste** (entropía cruzada en vez de cuadrática)
# 2. La **regularización** (penalizar pesos grandes)
# 3. La **inicialización** de los pesos
#
# Es una lección que va más allá de MNIST: en aprendizaje profundo, buena parte
# de las mejoras no vienen de arquitecturas más grandes sino de detalles del
# procedimiento de entrenamiento.

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

from dllab.nielsen.data import load_mnist, one_hot, submuestra
from dllab.nielsen.improved import CosteCuadratico, CosteEntropiaCruzada, RedMejorada
from dllab.nielsen.network import sigmoide, sigmoide_prima
from dllab.nielsen.viz import curva_aprendizaje

entrenamiento, validacion, test = load_mnist()


# %% [markdown]
# ## Problema 1: la red aprende despacio justo cuando más se equivoca
#
# Este es el hallazgo más contraintuitivo del capítulo, y conviene verlo en
# miniatura antes de razonarlo.
#
# Cogemos **una sola neurona**, con una sola entrada fija a 1, y le pedimos que
# aprenda a producir 0. La entrenamos dos veces: partiendo de un peso pequeño
# (empieza casi acertando) y de un peso grande (empieza muy equivocada).

# %%
def entrenar_neurona(w0, b0, coste, eta=0.15, pasos=300, objetivo=0.0):
    w, b, historial = w0, b0, []
    for _ in range(pasos):
        z = w * 1.0 + b
        a = sigmoide(z)
        historial.append(a)
        if coste == "cuadratico":
            # dC/dz incluye el factor σ'(z)
            dz = (a - objetivo) * sigmoide_prima(z)
        else:
            # con entropía cruzada el σ'(z) se cancela al derivar
            dz = a - objetivo
        w -= eta * dz * 1.0
        b -= eta * dz
    return historial


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (w0, b0, titulo) in zip(
    [ax1, ax2],
    [(0.6, 0.9, "Arranque cercano al objetivo"), (2.0, 2.0, "Arranque muy equivocado")],
):
    ax.plot(entrenar_neurona(w0, b0, "cuadratico"), label="coste cuadrático", lw=2)
    ax.plot(entrenar_neurona(w0, b0, "entropia"), label="entropía cruzada", lw=2)
    ax.set_title(titulo)
    ax.set_xlabel("paso")
    ax.grid(alpha=0.3)
    ax.legend()
ax1.set_ylabel("salida de la neurona (objetivo: 0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# En el panel izquierdo los dos costes se comportan parecido. En el derecho, el
# cuadrático se queda **plano durante decenas de pasos** antes de arrancar.
#
# La causa está en BP1, del notebook 3:
#
# $$\delta^L = (a - y) \odot \sigma'(z)$$
#
# Ese $\sigma'(z)$ es el problema. Cuando la neurona está muy equivocada, $z$ es
# grande en valor absoluto, la sigmoide está saturada y $\sigma'(z) \approx 0$.
# El gradiente se anula justo en el caso en que más urgía corregir.
#
# ### La entropía cruzada
#
# $$C = -\frac{1}{n}\sum_x \left[ y \ln a + (1-y)\ln(1-a) \right]$$
#
# Al derivarla respecto a $z$, el $\sigma'(z)$ del numerador se cancela con el
# denominador que aporta la derivada del logaritmo, y queda:
#
# $$\frac{\partial C}{\partial z} = a - y$$
#
# El gradiente ahora es **proporcional al error**. Cuanto más equivocada está la
# neurona, más rápido aprende — que es lo que uno querría desde el principio.
#
# La lección general: cuando algo aprende sospechosamente despacio, mira si hay
# un término derivado que se está anulando.

# %% [markdown]
# ## Comparación en MNIST
#
# ⏱️ Cada entrenamiento son ~3 minutos.

# %%
historiales = {}

r_cuad = RedMejorada([784, 30, 10], coste=CosteCuadratico, init="grande", seed=42)
historiales["cuadrático"] = r_cuad.sgd(
    entrenamiento, epocas=15, tam_minilote=10, eta=3.0, datos_eval=validacion, verbose=False, seed=42
)
print(f"cuadrático:       {historiales['cuadrático'].acierto_eval[-1]:.2%}")

r_ent = RedMejorada([784, 30, 10], coste=CosteEntropiaCruzada, init="grande", seed=42)
historiales["entropía cruzada"] = r_ent.sgd(
    entrenamiento, epocas=15, tam_minilote=10, eta=0.5, datos_eval=validacion, verbose=False, seed=42
)
print(f"entropía cruzada: {historiales['entropía cruzada'].acierto_eval[-1]:.2%}")

curva_aprendizaje(historiales)
plt.ylabel("acierto en validación")
plt.show()

# %% [markdown]
# Nota sobre las η distintas: la entropía cruzada produce gradientes bastante más
# grandes, así que necesita pasos más pequeños. Comparar dos funciones de coste
# con la misma η compararía otra cosa.
#
# ## Problema 2: sobreajuste
#
# Para verlo con claridad, entrenamos con solo 1.000 ejemplos. Con pocos datos,
# la red tiene parámetros de sobra para **memorizarlos**.

# %%
pocos = submuestra(entrenamiento, 1000, seed=0)

r_sobre = RedMejorada([784, 30, 10], coste=CosteEntropiaCruzada, init="grande", seed=3)
h_sobre = r_sobre.sgd(
    pocos, epocas=100, tam_minilote=10, eta=0.5, lmbda=0.0,
    datos_eval=validacion, monitorizar_entrenamiento=True, verbose=False, seed=3,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(h_sobre.coste_entrenamiento, label="entrenamiento")
ax1.set_title("Coste sobre los datos de entrenamiento")
ax1.set_xlabel("época")
ax1.grid(alpha=0.3)

ax2.plot(np.array(h_sobre.acierto_entrenamiento) * 100, label="entrenamiento")
ax2.plot(np.array(h_sobre.acierto_eval) * 100, label="validación")
ax2.set_title("Acierto")
ax2.set_xlabel("época")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Acierto final en entrenamiento: {h_sobre.acierto_entrenamiento[-1]:.2%}")
print(f"Acierto final en validación:    {h_sobre.acierto_eval[-1]:.2%}")

# %% [markdown]
# El coste de entrenamiento baja sin parar y el acierto sobre esos mismos datos
# llega al 100%. Pero el acierto en validación se estanca pronto y deja de
# mejorar.
#
# La red ha dejado de aprender a leer dígitos y ha empezado a memorizar esos mil
# en concreto. **La brecha entre las dos curvas es la definición operativa del
# sobreajuste**, y vigilarla es el hábito más importante que puedes coger.
#
# ### Regularización L2
#
# Se añade al coste un término que penaliza los pesos grandes:
#
# $$C = C_0 + \frac{\lambda}{2n}\sum_w w^2$$
#
# En la actualización, esto se traduce en encoger cada peso un poquito en cada
# paso:
#
# $$w \rightarrow \left(1 - \frac{\eta\lambda}{n}\right) w - \frac{\eta}{m}\sum \frac{\partial C_0}{\partial w}$$
#
# De ahí el nombre *weight decay*. La red conserva un peso grande solo si los
# datos lo justifican de forma sostenida.
#
# La intuición: entre dos explicaciones que encajan con los datos, la de pesos
# pequeños es más simple y suele generalizar mejor. Es una navaja de Ockham
# convertida en término matemático — y conviene decir que es una heurística que
# funciona, no un teorema.

# %%
comparacion = {"sin regularizar": h_sobre}
for lmbda in [1.0, 5.0]:
    r = RedMejorada([784, 30, 10], coste=CosteEntropiaCruzada, init="grande", seed=3)
    h = r.sgd(pocos, epocas=100, tam_minilote=10, eta=0.5, lmbda=lmbda,
              datos_eval=validacion, verbose=False, seed=3)
    comparacion[f"λ = {lmbda}"] = h
    print(f"λ = {lmbda}: validación {h.acierto_eval[-1]:.2%}")

curva_aprendizaje(comparacion)
plt.ylabel("acierto en validación")
plt.title("Efecto de la regularización con solo 1.000 ejemplos")
plt.show()

# %% [markdown]
# ## Problema 3: la inicialización
#
# Con pesos $N(0,1)$ y 784 entradas, la $z$ de una neurona oculta tiene
# desviación típica $\approx\sqrt{784} = 28$. Casi todas las neuronas arrancan
# saturadas, y ya sabemos lo que eso significa.
#
# La corrección es dividir por $\sqrt{n_{\text{entradas}}}$, con lo que la
# desviación típica de $z$ pasa a ser del orden de 1.

# %%
rng = np.random.default_rng(0)
x_ejemplo = np.abs(rng.standard_normal((784, 1))) * 0.3

z_grande = (rng.standard_normal((30, 784)) @ x_ejemplo).ravel()
z_escalada = (rng.standard_normal((30, 784)) / np.sqrt(784) @ x_ejemplo).ravel()

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(z_grande, bins=25, alpha=0.6, label=f"N(0,1): σ(z) ≈ {z_grande.std():.1f}")
ax.hist(z_escalada, bins=25, alpha=0.6, label=f"N(0,1/√n): σ(z) ≈ {z_escalada.std():.1f}")
ax.axvspan(-2, 2, color="green", alpha=0.1, label="zona no saturada")
ax.set_xlabel("z de las neuronas ocultas al inicializar")
ax.legend()
plt.tight_layout()
plt.show()

# %%
comparacion_init = {}
for init in ["grande", "escalada"]:
    r = RedMejorada([784, 30, 10], coste=CosteEntropiaCruzada, init=init, seed=11)
    h = r.sgd(entrenamiento, epocas=15, tam_minilote=10, eta=0.5, lmbda=5.0,
              datos_eval=validacion, verbose=False, seed=11)
    comparacion_init[f"init {init}"] = h
    print(f"init {init:9s}: {h.acierto_eval[-1]:.2%}")

curva_aprendizaje(comparacion_init)
plt.ylabel("acierto en validación")
plt.show()

# %% [markdown]
# La diferencia se nota sobre todo en las **primeras épocas**: con buena
# inicialización la red arranca ya aprendiendo, en vez de gastar varias épocas
# saliendo de la saturación. Muchas veces el acierto final es parecido, pero se
# llega antes y con menos riesgo.
#
# ## Los tres cambios juntos

# %%
red_final = RedMejorada([784, 30, 10], coste=CosteEntropiaCruzada, init="escalada", seed=99)
h_final = red_final.sgd(
    entrenamiento, epocas=30, tam_minilote=10, eta=0.5, lmbda=5.0,
    datos_eval=validacion, verbose=True, seed=99,
)

print(f"\nMejor en validación: {max(h_final.acierto_eval):.2%}")
print(f"Acierto en test:     {red_final.acierto(test):.2%}")

# %% [markdown]
# ## Resumen de la serie hasta aquí
#
# | Método | Acierto | Notebook |
# |---|---|---|
# | Azar | 10% | — |
# | Plantillas | ~82% | 1 |
# | Red + coste cuadrático | ~95% | 4 |
# | Red + entropía + L2 + init | ~97-98% | 5 |
#
# Misma arquitectura, mismo número de parámetros. Todo lo ganado viene de
# entender **por qué** el entrenamiento iba mal.
#
# Para bajar del 2% de error hace falta cambiar de arquitectura, y para eso ya
# conviene un framework. Ése es el último notebook.
#
# ---
#
# ## Ejercicios
#
# 1. Busca el mejor λ con una rejilla sobre validación (prueba 0.1, 1, 5, 10, 20)
#    usando el conjunto de entrenamiento completo. ¿Coincide con el de 1.000
#    ejemplos? ¿Por qué el λ óptimo depende del tamaño del conjunto?
# 2. Activa la parada temprana (`paciencia=5`). ¿Cuántas épocas ahorra?
# 3. La regularización L2 no penaliza los sesgos. Modifica el código para que sí
#    lo haga y comprueba si cambia algo. ¿Por qué crees que se excluyen?
# 4. Implementa la regularización L1 ($\lambda\sum|w|$) y compara. ¿Qué le pasa a
#    la distribución de los pesos? (Pista: dibuja un histograma de los pesos con
#    cada una.)
