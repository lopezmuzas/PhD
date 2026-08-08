---
title: "Backpropagation explicado de menos a más — Cálculo y Álgebra Lineal"
tags: [deep-learning, arquitecturas]
status: borrador
updated: 2026-08-08
---

# Backpropagation explicado de menos a más — Cálculo y Álgebra Lineal

> Guía de estudio paso a paso: partimos de un modelo recién inicializado con pesos aleatorios, entra el primer batch, se comete un error… y hay que ir hacia atrás a corregir los pesos. Aquí se explica **qué significa "ir hacia atrás"**, tanto en teoría como con números concretos, empezando por la derivada como pendiente y terminando en el backpropagation matricial completo.

---

## 🌄 Nivel 0 — La foto general: ¿qué problema estamos resolviendo?

Un modelo recién inicializado es un conjunto de matrices (`W_E`, `W_Q`, `W_K`, `W_V`, capas densas…) llenas de **números aleatorios microscópicos** (ej. 0.008, -0.011). Cuando el primer batch atraviesa el modelo, la predicción es basura pura: ruido.

El entrenamiento consiste en repetir este ciclo millones de veces:

```
1. Forward pass:  el batch atraviesa el modelo → predicción
2. Loss:          se compara la predicción con la respuesta correcta → un número (el error)
3. Backward pass: se calcula CUÁNTO contribuyó cada peso individual a ese error
4. Update:        cada peso se mueve un poquito en la dirección que reduce el error
```

El paso 3 es el **backpropagation**. La pregunta que responde es asombrosamente concreta:

> "Si yo moviera este peso concreto `w` una pizca hacia arriba, ¿el error subiría o bajaría, y cuánto?"

Esa pregunta, formulada matemáticamente, es una **derivada**: `∂Loss/∂w`. Y el modelo tiene que responderla para **cada uno de sus millones (o billones) de pesos**, en cada step. Backpropagation es simplemente el algoritmo que calcula todas esas derivadas de forma eficiente.

---

## 📉 Nivel 1 — La derivada como pendiente (la intuición del esquiador)

Olvida las redes neuronales un momento. Imagina una función de una sola variable:

```
L(w) = (w - 3)²
```

Esta función es una parábola con su mínimo en `w = 3`. Interpreta `L` como "el error" y `w` como "el único peso del modelo". Queremos encontrar el valor de `w` que hace el error mínimo.

### La derivada te dice la pendiente del terreno
La derivada de esa función es:

```
dL/dw = 2(w - 3)
```

Evalúala en distintos puntos:

| Valor actual de w | dL/dw = 2(w−3) | Interpretación |
|---|---|---|
| w = 5 | +4 | Pendiente positiva: si subo w, el error SUBE. Debo bajar w. |
| w = 3 | 0 | Pendiente cero: estoy en el fondo del valle. No me muevo. |
| w = 1 | −4 | Pendiente negativa: si subo w, el error BAJA. Debo subir w. |

La regla de oro del entrenamiento sale directamente de esta tabla: **muévete siempre en dirección contraria a la pendiente** (si la pendiente es positiva, baja; si es negativa, sube). Eso es *descenso por gradiente*:

```
w_nuevo = w_viejo − η · (dL/dw)
```

Donde `η` (eta) es el **learning rate**: el tamaño del pasito. Si `η = 0.1` y estamos en `w = 5`:

```
w_nuevo = 5 − 0.1 · 4 = 4.6
```

Nos acercamos a 3. Repite esto muchas veces y `w` converge al mínimo. **Todo el entrenamiento de un LLM de billones de parámetros es exactamente esta fórmula**, solo que aplicada a billones de `w` a la vez.

### ¿Por qué "gradiente" y no "derivada"?
Cuando la función depende de muchas variables (`L(w_1, w_2, ..., w_n)`), el conjunto de todas las derivadas parciales se llama **gradiente**:

```
∇L = [∂L/∂w_1, ∂L/∂w_2, ..., ∂L/∂w_n]
```

Geométricamente, el gradiente es un vector que apunta en la dirección de **máxima subida** del error. Por eso el update resta el gradiente: caminamos en la dirección de máxima bajada. El esquiador mira la montaña de error y desciende por la línea de máxima pendiente.

---

## ⛓️ Nivel 2 — La regla de la cadena: el corazón del "hacia atrás"

El problema real: en una red neuronal, el error `L` **no depende directamente de un peso `w`**, sino a través de una cadena de operaciones intermedias. Ejemplo mínimo con una sola "neurona":

```
z = w · x        (multiplicación por el peso)
a = tanh(z)      (activación no lineal)
L = (a − y)²     (error cuadrático contra la respuesta correcta y)
```

Queremos `∂L/∂w`, pero `L` no contiene a `w` de forma directa: `w` afecta a `z`, `z` afecta a `a`, y `a` afecta a `L`. La **regla de la cadena** del cálculo dice que las influencias se multiplican eslabón a eslabón:

```
∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)
```

Léelo como una cadena de "cuánto afecta cada cosa a la siguiente":
- `∂z/∂w = x` → si muevo el peso, z se mueve proporcionalmente a la entrada.
- `∂a/∂z = 1 − tanh²(z)` → derivada de la tanh.
- `∂L/∂a = 2(a − y)` → cuánto duele el error al mover la salida.

**Backpropagation = aplicar la regla de la cadena empezando por el final (el Loss) y retrocediendo eslabón a eslabón hasta cada peso.** Por eso se llama "propagación hacia atrás": el error se calcula al final, y su "culpa" se va repartiendo hacia atrás, capa por capa, multiplicando derivadas locales.

### Ejemplo numérico completo (una neurona)
Datos: entrada `x = 2`, peso inicial aleatorio `w = 0.5`, respuesta correcta `y = 0.9`, learning rate `η = 0.1`.

**Forward pass (guardando cada valor intermedio):**
```
z = w·x = 0.5 · 2 = 1.0
a = tanh(1.0) ≈ 0.7616
L = (0.7616 − 0.9)² ≈ (−0.1384)² ≈ 0.01916
```

**Backward pass (regla de la cadena, del final al principio):**
```
∂L/∂a = 2(a − y) = 2(−0.1384) = −0.2768
∂a/∂z = 1 − tanh²(1.0) = 1 − 0.5800 = 0.4200
∂z/∂w = x = 2
∂L/∂w = (−0.2768) · (0.4200) · (2) ≈ −0.2325
```

**Update:**
```
w_nuevo = 0.5 − 0.1·(−0.2325) = 0.5233
```

El gradiente era negativo → el peso sube. Comprobación: con `w = 0.5233`, ahora `a = tanh(1.0465) ≈ 0.7803`, más cerca de 0.9. El error bajó de 0.01916 a 0.01433. **Un step de entrenamiento completo, hecho a mano.**

### Detalle crucial: el forward pass guarda los ingredientes
Fíjate en que para calcular las derivadas necesitamos los valores intermedios del forward pass (`z`, `a`, `x`). Por eso, durante el entrenamiento, la GPU **memoriza todas las activaciones intermedias** de la pasada hacia adelante: son los ingredientes del backward. (Esta es la razón principal de que entrenar consuma muchísima más memoria que hacer inferencia.)

---

## 🕸️ Nivel 3 — Varias capas: la culpa fluye hacia atrás en cascada

Añadamos una segunda capa. Ahora hay dos pesos, `w_1` y `w_2`:

```
z_1 = w_1 · x          a_1 = tanh(z_1)
z_2 = w_2 · a_1        a_2 = tanh(z_2)
L   = (a_2 − y)²
```

Para el peso de la **última capa**, la cadena es corta:

```
∂L/∂w_2 = (∂L/∂a_2) · (∂a_2/∂z_2) · (∂z_2/∂w_2)
        =  2(a_2−y)  · (1−tanh²(z_2)) · a_1
```

Para el peso de la **primera capa**, la cadena atraviesa toda la segunda capa:

```
∂L/∂w_1 = (∂L/∂a_2)·(∂a_2/∂z_2) · (∂z_2/∂a_1) · (∂a_1/∂z_1) · (∂z_1/∂w_1)
        =  2(a_2−y) ·(1−tanh²(z_2)) ·   w_2    ·(1−tanh²(z_1)) ·    x
```

### La observación que hace eficiente al algoritmo
Compara ambas fórmulas: los dos primeros factores, `2(a_2−y)·(1−tanh²(z_2))`, **son idénticos**. A esa cantidad se le llama el **delta** de la capa 2:

```
δ_2 = (∂L/∂a_2) · (∂a_2/∂z_2)
```

Y el delta de la capa 1 se construye reutilizando el de la capa 2:

```
δ_1 = δ_2 · w_2 · (1−tanh²(z_1))
```

Con lo que los gradientes quedan limpísimos:

```
∂L/∂w_2 = δ_2 · a_1
∂L/∂w_1 = δ_1 · x
```

**Esto ES backpropagation como algoritmo**: en vez de recalcular la cadena completa para cada peso (carísimo), se calcula el delta una sola vez por capa, empezando por la última, y cada capa anterior recicla el delta de la siguiente. El error se "propaga hacia atrás" como una señal que se transforma en cada capa que atraviesa. Una sola pasada hacia atrás calcula los gradientes de TODOS los pesos.

Aquí también se ve, algebraicamente, el **desvanecimiento del gradiente** de las RNN del documento anterior: cada capa hacia atrás multiplica el delta por un peso y por una derivada de tanh (que siempre vale entre 0 y 1). Muchas capas → muchos factores menores que 1 multiplicados → el delta que llega a las primeras capas es microscópico.

---

## 🔢 Nivel 4 — Versión matricial: cuando los pesos son matrices

En un modelo real no hay pesos sueltos sino matrices. Una capa densa hace:

```
Z = X · W        X: [batch × d_in]   W: [d_in × d_out]   Z: [batch × d_out]
A = f(Z)
```

La pregunta "¿cuánto afecta cada peso al error?" ahora tiene una respuesta por cada celda de `W`, así que el gradiente `∂L/∂W` es **una matriz del mismo tamaño exacto que W**, donde cada celda dice cuánto y hacia dónde mover ese peso concreto.

Las fórmulas del nivel 3 se convierten en tres reglas matriciales que son las que ejecuta la GPU:

```
δ = (∂L/∂A) ⊙ f'(Z)              # delta de la capa: [batch × d_out]
∂L/∂W = X^T · δ                  # gradiente de los pesos: [d_in × d_out]  ✓ mismo tamaño que W
∂L/∂X = δ · W^T                  # señal que se propaga a la capa anterior: [batch × d_in]
```

Tres observaciones importantes:

1. **`∂L/∂W = X^T · δ`** — el gradiente de una matriz de pesos es "la entrada que recibió, transpuesta, multiplicada por el delta que le llega". Fíjate en la simetría con el forward (`Z = X·W`): el backward reutiliza las mismas piezas, transpuestas.
2. **`∂L/∂X = δ · W^T`** — esta es la señal que continúa el viaje hacia atrás: se convierte en el `∂L/∂A` de la capa anterior. La transposición `W^T` es la manifestación algebraica de "deshacer el camino": si en el forward multiplicaste por `W`, en el backward la señal vuelve multiplicada por `W^T`.
3. **El batch entra gratis** — como `X` tiene una fila por cada ejemplo del batch, la multiplicación `X^T·δ` **suma automáticamente las contribuciones de todos los ejemplos del batch** en un solo gradiente. Por eso se entrena por batches: un solo par de multiplicaciones matriciales calcula el gradiente promedio de mil frases a la vez, en paralelo.

Y el update es la misma fórmula del esquiador, ahora entre matrices:

```
W_nuevo = W_viejo − η · (∂L/∂W)
```

---

## 🎯 Nivel 5 — El punto de partida real: Softmax + Cross-Entropy

En un modelo de lenguaje, el error no es un error cuadrático sino esto: la última capa proyecta al vocabulario y produce **logits** (una puntuación por palabra), el softmax los convierte en probabilidades, y el loss es la **entropía cruzada** (cross-entropy): castigar al modelo según cuán poca probabilidad le dio a la palabra correcta.

```
logits:  s = [s_1, s_2, ..., s_V]           (V = tamaño del vocabulario)
softmax: p_i = e^(s_i) / Σ e^(s_j)
loss:    L = −log(p_correcta)
```

Ejemplo con vocabulario de 4 palabras, donde la correcta es la 3ª:

```
s = [1.0, 2.0, 0.5, 0.1]
p ≈ [0.23, 0.63, 0.14, 0.09]  →  normalizado: [0.229, 0.622, 0.139, 0.093]
L = −log(0.139) ≈ 1.97        (error alto: solo le dio 13.9% a la correcta)
```

### La derivada más elegante del deep learning
Aquí ocurre algo precioso: cuando derivas la combinación softmax + cross-entropy respecto de los logits, todo el aparato de exponenciales y logaritmos colapsa en una resta:

```
∂L/∂s = p − y_onehot
```

Es decir: **el delta inicial del backpropagation es literalmente "probabilidades predichas menos la respuesta correcta"**. Con el ejemplo:

```
p        = [0.229, 0.622, 0.139, 0.093]
y_onehot = [0,     0,     1,     0    ]
∂L/∂s    = [0.229, 0.622, −0.861, 0.093]
```

Lee el vector resultante: cada palabra incorrecta recibe un gradiente positivo proporcional a la probabilidad que se le dio de más ("bájame las apuestas por estas"), y la correcta recibe un gradiente negativo grande ("súbeme la apuesta por esta"). Ese vector es la **chispa inicial** que arranca toda la cascada del backward: es el primer delta, el que entra por la última capa y se va transformando hacia atrás mediante las reglas del Nivel 4.

---

## 🏗️ Nivel 6 — El recorrido completo en un Transformer (uniendo todo)

Ahora sí, el proceso completo de tu escenario: pesos aleatorios, entra el primer batch.

### Forward pass (guardando todo)
```
One-Hot → ·W_E → X → +P (posición) → ·W_Q,W_K,W_V → Q,K,V
→ softmax(Q·K^T/√d)·V = Z → capas densas (FFN) → ... → logits → softmax → Loss
```
Con pesos aleatorios, el softmax final reparte probabilidad casi uniforme (~1/V a cada palabra del vocabulario). El Loss es enorme: `−log(1/V)`. Con V=50,000, `L ≈ 10.8`. Ese número es el punto de partida típico de todo entrenamiento.

### Backward pass (la cascada de deltas)
1. **Chispa inicial:** `δ_logits = p − y_onehot` (Nivel 5), calculada para cada posición de cada frase del batch.
2. **Capa de proyección al vocabulario:** con las reglas del Nivel 4 se obtiene su gradiente (`entrada^T · δ`) y se pasa la señal hacia atrás (`δ · W^T`).
3. **Capas densas (FFN):** misma receta, capa por capa hacia atrás.
4. **Bloque de atención:** aquí la cadena se ramifica, porque `Z = softmax(Q·K^T/√d)·V` depende de Q, K y V a la vez. La regla de la cadena reparte el delta entrante en tres flujos: uno hacia `W_V` (a través de V), y dos hacia `W_Q` y `W_K` (atravesando la derivada del softmax y el producto `Q·K^T`). Conceptualmente no hay nada nuevo: son las mismas reglas matriciales, solo que el grafo de dependencias tiene bifurcaciones, y cuando un valor influye en el error por varios caminos, sus gradientes **se suman**.
5. **Embeddings:** la señal llega al principio del todo y produce `∂L/∂W_E`. Solo se actualizan las columnas de las palabras que aparecieron en el batch (las demás recibieron delta cero, por la estructura one-hot).

### Update (optimizador)
Con todos los gradientes calculados:
```
W ← W − η · ∂L/∂W        (para CADA matriz del modelo, a la vez)
```
En la práctica no se usa el descenso por gradiente "crudo" sino **Adam**: la misma idea, pero cada peso mantiene una media móvil de sus gradientes recientes (momento, para no zigzaguear) y una media de sus magnitudes (para adaptar el tamaño del paso individualmente por peso). La esencia no cambia: pendiente → paso en contra.

### Y vuelta a empezar
Entra el siguiente batch, con los pesos ya ligeramente menos aleatorios. El Loss baja de 10.8 a 10.7, luego a 10.5… Millones de steps después, las matrices han dejado de ser ruido: la geometría del lenguaje ha emergido, exclusivamente, de esta repetición de resta-de-pendientes.

---

## 🧭 Resumen mental de toda la escalera

| Nivel | Idea | Fórmula clave |
|---|---|---|
| 0 | El entrenamiento es un ciclo: predecir → medir error → repartir culpas → ajustar | forward / loss / backward / update |
| 1 | La derivada es la pendiente; muévete en contra de ella | `w ← w − η·(dL/dw)` |
| 2 | Si el error depende del peso a través de pasos intermedios, las influencias se multiplican | `∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w` |
| 3 | Cada capa calcula su delta reciclando el de la capa siguiente: una sola pasada para todos los pesos | `δ_1 = δ_2 · w_2 · f'(z_1)` |
| 4 | Con matrices, el gradiente de W es entrada^T por delta; la señal retrocede vía W^T; el batch se suma solo | `∂L/∂W = X^T·δ`, `∂L/∂X = δ·W^T` |
| 5 | En modelos de lenguaje, la chispa inicial del backward es una simple resta | `δ_inicial = p − y_onehot` |
| 6 | En el Transformer, la cascada atraviesa FFN y atención (repartiéndose entre Q, K, V) hasta llegar a W_E | los gradientes de caminos ramificados se suman |

**La frase para llevarse a casa:** backpropagation no es magia ni un mecanismo misterioso; es la regla de la cadena del cálculo de bachillerato, organizada de forma inteligente (reciclando deltas de capa en capa) y expresada en multiplicaciones matriciales para que la GPU pueda repartir la culpa del error entre billones de pesos en milisegundos.
