---
title: "Cómo funcionaba el Deep Learning antes de 'Attention Is All You Need' — Álgebra Lineal"
tags: [deep-learning, arquitecturas]
status: borrador
updated: 2026-08-08
---

# Cómo funcionaba el Deep Learning antes de "Attention Is All You Need" — Álgebra Lineal

> Explicación paso a paso de las arquitecturas de Deep Learning (MLP, RNN, LSTM, Seq2Seq, Attention temprana) previas al Transformer (2017), vistas desde el álgebra lineal, con una comparación final frente a los Transformers actuales.

---

## 🧩 Sección 1 — El punto de partida: el Perceptrón Multicapa (MLP)

Antes de que existiera cualquier arquitectura pensada para secuencias, el bloque fundamental era el **Perceptrón Multicapa (MLP)**, una simple cadena de multiplicaciones matriciales y no linealidades.

Dado un vector de entrada `x` (por ejemplo, un embedding de una palabra, de tamaño `d×1`), una capa densa hace:

```
h = f(W·x + b)
```

Donde:
- `W` es una matriz de pesos `[d_out × d_in]`.
- `b` es un vector de sesgo `[d_out × 1]`.
- `f` es una función no lineal (sigmoide, tanh, ReLU).

### El problema estructural del MLP
Un MLP espera una entrada de **tamaño fijo**. Si tu frase tiene 3 palabras, puedes concatenar sus 3 vectores en uno solo y multiplicarlo por `W`. Pero si la siguiente frase tiene 7 palabras, la matriz `W` ya no calza dimensionalmente. Algebraicamente, el MLP no tiene ningún mecanismo para procesar secuencias de longitud variable ni para relacionar una palabra con otra según su posición. Cada entrada es un vector aislado; no hay noción de "orden" ni de "contexto que fluye".

Esto obligó a la comunidad a diseñar arquitecturas que reutilizaran la **misma matriz de pesos** en cada paso de una secuencia: nacen las Redes Neuronales Recurrentes.

---

## 🔁 Sección 2 — Redes Neuronales Recurrentes (RNN)

La idea algebraica de una RNN es simple pero poderosa: en vez de una sola multiplicación `W·x`, se mantiene un **vector de estado oculto** `h_t` que se actualiza en cada paso de tiempo `t`, combinando la entrada actual con la memoria del paso anterior:

```
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b)
```

Donde, para una dimensión oculta `d`:
- `W_xh` → matriz `[d × d_emb]` que proyecta la palabra actual.
- `W_hh` → matriz `[d × d]` que proyecta el estado anterior (la "memoria").
- `h_(t-1)` → vector `[d × 1]`, el resumen de todo lo procesado hasta ahora.

### El flujo secuencial obligatorio
Para procesar la frase "El gato come pescado", la RNN hace:

```
h_1 = tanh(W_xh·x_El   + W_hh·h_0 + b)
h_2 = tanh(W_xh·x_gato + W_hh·h_1 + b)
h_3 = tanh(W_xh·x_come + W_hh·h_2 + b)
h_4 = tanh(W_xh·x_pescado + W_hh·h_3 + b)
```

Nótese el problema algebraico clave: **para calcular `h_3` necesitas obligatoriamente `h_2`, y para `h_2` necesitas `h_1`**. No existe forma de paralelizar estas multiplicaciones; la GPU debe esperar, paso a paso, a que termine el cálculo anterior. Es una cadena de dependencias, no una operación matricial masiva en paralelo.

### El vector `h_final` como "resumen" de toda la frase
Al terminar de procesar toda la secuencia, `h_4` (el último estado oculto) se considera el resumen vectorial de toda la frase completa. Ese único vector de tamaño `d×1` debe cargar, comprimida, toda la información de "El gato come pescado".

### El problema matemático: el desvanecimiento del gradiente
Durante el backpropagation (llamado aquí *Backpropagation Through Time*), el gradiente de la primera palabra debe viajar multiplicándose repetidamente por `W_hh` en cada paso hacia atrás:

```
∂h_4/∂h_1 ≈ W_hh · W_hh · W_hh
```

Si los valores propios de `W_hh` son menores que 1, el producto se acerca exponencialmente a cero (*vanishing gradient*); si son mayores que 1, explota (*exploding gradient*). Algebraicamente, esto significa que **la información de las primeras palabras de una frase larga se diluye o se descontrola** antes de llegar a las últimas capas. Una RNN "olvida" rápidamente el comienzo de una oración larga.

---

## 🚪 Sección 3 — LSTM y GRU: puertas hechas de matrices

Para mitigar el desvanecimiento del gradiente, se diseñaron las **LSTM (Long Short-Term Memory)**. La idea algebraica es añadir un segundo vector, la "celda de memoria" `C_t`, y controlar cuánta información entra, sale o se olvida mediante **compuertas**, que no son más que multiplicaciones matriciales pasadas por una sigmoide (`σ`), cuyo resultado (entre 0 y 1) actúa como un filtro elemento a elemento.

```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)     # Puerta de olvido
i_t = σ(W_i · [h_(t-1), x_t] + b_i)     # Puerta de entrada
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t        # Actualización de la memoria
o_t = σ(W_o · [h_(t-1), x_t] + b_o)     # Puerta de salida
h_t = o_t ⊙ tanh(C_t)
```

El símbolo `⊙` es el **producto de Hadamard** (multiplicación elemento a elemento, no producto matricial), y `[h_(t-1), x_t]` es la concatenación de ambos vectores en uno solo antes de multiplicar.

Geométricamente, cada compuerta aprende su propia matriz de pesos para decidir, dimensión por dimensión, "cuánto de esta información dejo pasar". Esto alarga considerablemente la memoria efectiva de la red, pero **no elimina el problema de fondo**: sigue siendo una cadena estrictamente secuencial, imposible de paralelizar, y la información de un token muy lejano sigue teniendo que atravesar cientos de multiplicaciones antes de llegar al final.

---

## 📦 Sección 4 — Seq2Seq: el cuello de botella del vector único

La arquitectura dominante en traducción automática antes del Transformer (2014-2016) era **Encoder-Decoder (Seq2Seq)**, construida sobre RNN o LSTM.

### Estructura algebraica
1. **Encoder:** una RNN/LSTM procesa toda la frase de entrada palabra por palabra, y se queda solo con el **último estado oculto**, `h_n`, un único vector `[d × 1]`.
2. **Vector de contexto:** ese `h_n` se convierte en el "resumen" de la frase completa, sin importar si tenía 5 o 50 palabras. Es literalmente un cuello de botella: toda la información debe comprimirse en ese único vector de dimensión fija.
3. **Decoder:** otra RNN/LSTM recibe ese único vector de contexto y, a partir de él, genera la frase traducida palabra por palabra:

```
s_0 = h_n                                    # el decoder "arranca" con el contexto
s_t = tanh(W_s · s_(t-1) + W_y · y_(t-1) + b)
y_t = softmax(W_out · s_t)
```

### El problema geométrico del cuello de botella
Si `h_n` tiene, por ejemplo, 512 dimensiones, **una frase de 3 palabras y una de 40 palabras deben comprimirse igualmente en esas mismas 512 dimensiones**. Cuanto más larga la frase de entrada, más información se pierde o se mezcla en ese único vector. Es el equivalente a pedirle a alguien que lea un párrafo entero y luego lo resuma en una sola palabra antes de empezar a traducirlo: inevitablemente se pierden detalles, sobre todo del principio de la frase.

Este cuello de botella fue, precisamente, el problema que impulsó la invención de la atención.

---

## 🎯 Sección 5 — La atención "clásica" (Bahdanau/Luong, 2014-2015): el puente hacia el Transformer

Antes del *self-attention* del Transformer, ya existía un mecanismo de atención, pero limitado al **Decoder mirando hacia el Encoder** (attention "cruzada"), no las palabras de una frase mirándose entre sí.

### La idea algebraica
En vez de que el decoder reciba solo el último `h_n`, se le permite **consultar todos los estados ocultos del encoder** (`h_1, h_2, ..., h_n`) en cada paso de generación, calculando un "peso de afinidad" con cada uno:

```
e_ti = v^T · tanh(W_1·s_(t-1) + W_2·h_i)     # puntuación de afinidad (Bahdanau)
α_ti = softmax(e_ti)                          # normalización
c_t  = Σ α_ti · h_i                           # vector de contexto ponderado
```

Aquí `α_ti` funciona como un peso de importancia entre el estado actual del decoder `s_(t-1)` y cada palabra `h_i` del encoder. El vector de contexto `c_t` ya no es fijo: **cambia dinámicamente en cada paso**, mirando con más "peso" las palabras del encoder que son relevantes para generar la siguiente palabra.

### Por qué esto no es todavía un Transformer
Esta atención resolvía el cuello de botella del vector único, pero mantenía dos limitaciones estructurales:
1. **Seguía dependiendo de RNN/LSTM por debajo**: los `h_i` se calculaban de forma secuencial, así que el problema de paralelización no desaparecía.
2. **Solo conectaba Decoder→Encoder**, no había atención entre las palabras de una misma frase (self-attention). Las palabras del encoder nunca "hablaban" directamente entre sí mediante `Q·K^T`; el encoder seguía siendo una simple RNN.

Fue justamente la idea de "¿y si eliminamos la RNN y dejamos que todo funcione solo a base de atención (Q, K, V) entre todos los tokens?" la que dio título al paper de 2017: *Attention Is All You Need*.

---

## 🧵 Sección 6 — Los embeddings antes del Transformer: vectores estáticos

Otra diferencia algebraica importante estaba en la propia matriz de embedding `W_E`. Antes del Transformer, los vectores de palabras más usados (Word2Vec, GloVe) eran **estáticos**: la palabra "banco" tenía siempre el mismo vector `x_banco`, sin importar el contexto de la frase.

```
x_banco = W_E · o_banco   (siempre el mismo vector, en cualquier frase)
```

Estos vectores se entrenaban previamente (pre-entrenamiento no contextual) usando el principio de que palabras que aparecen en contextos similares deben tener vectores similares (hipótesis distribucional), típicamente maximizando `v_palabra · v_contexto` para pares de palabras cercanas en un corpus (Skip-gram / CBOW).

La consecuencia geométrica: "Me senté en el banco" y "Retiré dinero del banco" **usan exactamente el mismo punto en el espacio vectorial para "banco"**, aunque el significado sea distinto. La única forma de desambiguar el significado era dejar que la RNN, al procesar la frase palabra por palabra, fuera modificando el estado oculto `h_t`, pero el vector de entrada `x_banco` en sí nunca cambiaba.

---

## 🖼️ Sección 7 — Nota aparte: redes convolucionales (CNN) para texto

Como alternativa a la recurrencia, algunos modelos (Facebook's ConvS2S, 2017) usaron **convoluciones 1D** sobre secuencias de texto. Algebraicamente, una convolución es una matriz de pesos pequeña (el "kernel", por ejemplo `[3 × d]`) que se desliza sobre la secuencia, multiplicando solo una ventana local de palabras vecinas a la vez:

```
h_i = f(W_kernel · [x_(i-1), x_i, x_(i+1)] + b)
```

Esto sí se puede paralelizar (todas las ventanas se calculan a la vez, a diferencia de la RNN), pero cada capa solo "ve" una ventana local de palabras cercanas. Para que una palabra al principio de la frase influya en una palabra al final, se necesitan **muchas capas apiladas** (para ampliar el "campo receptivo"), lo cual sigue siendo indirecto y costoso comparado con que cualquier palabra pueda mirar directamente a cualquier otra.

---

## 📊 Sección 8 — Comparación: antes del Transformer vs. Transformers actuales

| Aspecto | Antes (RNN / LSTM / Seq2Seq) | Transformers actuales |
|---|---|---|
| **Procesamiento de la secuencia** | Estrictamente secuencial: `h_t` depende de `h_(t-1)`. Imposible paralelizar en el tiempo. | Todas las palabras se procesan simultáneamente vía `Q·K^T` sobre toda la matriz `X` a la vez. Altamente paralelizable en GPU/TPU. |
| **Memoria de contexto** | Comprimida en un único vector oculto de tamaño fijo (`h_n`), que actúa como cuello de botella. | Cada palabra tiene acceso directo (mediante atención) a todas las demás palabras de la secuencia, sin comprimir nada en un solo vector intermedio. |
| **Distancia entre palabras relacionadas** | El gradiente y la información deben atravesar tantos pasos como palabras haya entre ellas → desvanecimiento del gradiente en frases largas. | La distancia algebraica entre dos palabras cualesquiera es siempre "un paso" (un único producto escalar `q_i · k_j`), sin importar cuán lejos estén en la frase. |
| **Embeddings** | Estáticos: la misma palabra tiene siempre el mismo vector, sin importar el contexto (Word2Vec, GloVe). | Contextuales: el vector de cada palabra se recalcula en cada capa según las demás palabras de la frase (self-attention), por lo que "banco" cambia de vector según el contexto. |
| **Cómo se relacionan las palabras entre sí** | Indirectamente, a través de la memoria secuencial acumulada (RNN) o de una atención Decoder→Encoder limitada (Bahdanau/Luong). | Directamente, mediante *self-attention*: cualquier palabra puede calcular su afinidad (`Q·K^T`) con cualquier otra palabra de la misma frase, en ambas direcciones. |
| **Escalabilidad** | Entrenar en secuencias muy largas o en datasets masivos era costoso, porque no se aprovechaba el paralelismo de la GPU. | Se pudo escalar a miles de millones/billones de parámetros (GPT, etc.) precisamente porque el cálculo es paralelo y matricial de principio a fin. |
| **Multimodalidad** | Muy limitada: arquitecturas distintas y poco compatibles para texto, imagen o audio. | Cualquier modalidad se proyecta al mismo espacio vectorial (embeddings compartidos) y entra al mismo mecanismo de atención, como se explicaba en el documento sobre Attention Is All You Need. |

### La idea de fondo
Todo el "antes" del Transformer giraba en torno a **comprimir información secuencialmente en un vector cada vez más cargado**, arrastrando ese vector paso a paso y sufriendo pérdidas de información en el camino. El Transformer eliminó por completo la recurrencia y reemplazó ese arrastre secuencial por **una única operación matricial masiva (`Q·K^T·V`) que conecta todas las palabras entre sí al mismo tiempo**. Ese cambio de paradigma —de "memoria que fluye paso a paso" a "todos hablan con todos en paralelo"— es, en esencia, lo que el título del paper anunciaba: *la atención es todo lo que necesitas*, ya no hacía falta la recurrencia ni la convolución.
