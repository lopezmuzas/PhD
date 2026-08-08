---
title: "La Biblia de las Redes Neuronales"
tags: [deep-learning, arquitecturas]
status: borrador
updated: 2026-08-08
---

# La Biblia de las Redes Neuronales
### Del Perceptrón al Transformer: MLP → CNN → RNN → Atención → *Attention Is All You Need*

> **Esta guía no está pensada para leerse como un texto plano, sino para ejecutarse como un árbol de dependencias.** Cada bloque contiene un solo problema y una sola solución, con sus recursos ya filtrados en dos niveles. Lee primero la Sección 0.B (metodología) antes de empezar.

---

## 0.A Mapa mental del recorrido

```
Neurona biológica (inspiración: sinapsis, umbral, plasticidad)
   │
   ▼
Perceptrón (1958)
   │
   ▼
MLP + Backpropagation (1986)  ──► primer "deep learning" que funciona
   │
   ├──► Redes Convolucionales (CNN)  →  visión por computador
   │
   └──► Redes Recurrentes (RNN)  →  secuencias / lenguaje
              │
              ▼
        Problema del gradiente que desaparece/explota (vanishing/exploding gradient)
              │
              ▼
        LSTM (1997) / GRU (2014)  →  "parchean" el problema con puertas de memoria
              │
              ▼
        Seq2Seq (2014): Encoder-Decoder para traducción automática
              │
              ▼
        Cuello de botella: TODO el significado en UN solo vector
              │
              ▼
        Mecanismo de Atención en RNN (Bahdanau 2014, Luong 2015)
              │
              ▼
        "¿Y si quitamos la recurrencia y nos quedamos solo con la atención?"
              │
              ▼
        Attention Is All You Need (2017) → el Transformer
              │
              ▼
        BERT, GPT, ChatGPT, Claude, LLaMA... (2018-hoy)
```

---

## 0.B Cómo usar esta guía: metodología "Del Grafo al Código"

### Los 4 pasos

**Paso 1 — Fragmentación atómica (chunking).**
Cada sesión de estudio cubre **un solo bloque** = una transición del mapa mental. No estudies "RNN, LSTM y Atención" de golpe; estudia exclusivamente "por qué el perceptrón falló con XOR y cómo el MLP lo resuelve con backprop".

**Paso 2 — Bucle de 3 niveles de abstracción.**
Para cada bloque, consume los recursos en este orden estricto. Nunca saltes de la teoría al paper sin pasar por la intuición.

```
1. INTUICIÓN (visual)     ──►  2. FORMALIZACIÓN (lectura)  ──►  3. EJECUCIÓN
   3Blue1Brown/StatQuest        Blog ilustrado / esta guía        Feynman o código
   15-30 min                    20-30 min                          15 min
```

- **Intuición**: ve el vídeo del bloque a velocidad 1x. El objetivo no es memorizar la fórmula, sino entender **qué problema de la arquitectura anterior se está intentando resolver**.
- **Formalización**: lee el post ilustrado (Alammar / Olah) o el apartado concreto de esta guía.
- **Validación**: responde por escrito las 3 preguntas de cierre del bloque (están al final de cada uno).

**Paso 3 — Filtrado agresivo de recursos.**
La trampa es querer verlo y leerlo todo. En cada bloque los recursos ya vienen separados:

- **🟢 Nivel Base (obligatorio)**: vídeo de intuición + post ilustrado + la sección de esta guía. Con esto basta.
- **🔵 Nivel Profundo (solo si el nivel base no te ha bastado)**: paper original (leyendo solo introducción, arquitectura y conclusiones) o implementación en código.

Si tras el nivel base lo has entendido: marca el checklist y pasa al siguiente bloque. **No acumules lecturas repetitivas.**

**Paso 4 — Hitos de codificación (hands-on).**
Solo 3 hitos prácticos en toda la ruta, para no perder el contacto con el grafo de cómputo:

| Hito | Cuándo | Qué hacer |
|---|---|---|
| **Hito 1** | Tras el Bloque 1 | Programa a mano (numpy o Python puro) el forward pass y el backprop de un MLP simple que resuelva XOR. Base: sección **12.B.6**. |
| **Hito 2** | Tras el Bloque 5/6 | Entrena una RNN o LSTM básica en PyTorch para predecir el siguiente carácter de un texto corto. |
| **Hito 3** | Tras el Bloque 7 | Sigue el vídeo *Let's build GPT* de Karpathy escribiendo el código a la vez. |

### Plantilla de ejecución semanal (45-60 min por sesión, 1 bloque por sesión)

| Sesión | Bloque | Foco |
|---|---|---|
| 0 | Bloque 0 | La neurona biológica → de dónde salen pesos, sesgos y activaciones (sesión corta, de contexto) |
| 1 | Bloque 1 | Perceptrón, MLP, Backprop → visualización + secciones 12.A / 12.B |
| 2 | Bloque 2 | CNN y ResNet → invariancia, filtros, skip connections |
| 3 | Bloques 3 y 4 | RNN + vanishing gradient → por qué multiplicar matrices en el tiempo destruye el gradiente |
| 4 | Bloque 5 | LSTM / GRU → las puertas de memoria como solución arquitectónica |
| 5 | Bloque 6 | Seq2Seq + atención → el cuello de botella del vector fijo y Bahdanau |
| 6 | Bloque 7 (7.1-7.2) | Self-Attention, Q/K/V → Jay Alammar + 3Blue1Brown |
| 7 | Bloque 7 (7.3-7.4) | Transformer completo → Yannic Kilcher o paper original |
| 8+ | Práctica | Karpathy / implementación en código (Hito 3) |

### Criterio de parada (Definition of Done)

> Sabes que has aprendido la materia cuando eres capaz de reproducir de memoria la tabla de la **Sección 8 (línea temporal resumen)**, razonando la columna *"problema que resuelve"* de cada hito **sin mirar el documento**.

---

# BLOQUE 0 — La neurona biológica: qué copiamos y qué no
### 🗓️ Sesión 0 · 🎯 Problema: ¿de dónde sale la idea de "neurona artificial"?

> Este bloque es de contexto, no de arquitectura. No necesitas ser experto en neurociencia — solo entender **de qué se inspiraron los pesos, los sesgos y las funciones de activación**, y dónde la analogía deja de funcionar (que es antes de lo que suele contarse).

## 0. Cómo aprende una neurona biológica

### 0.1 Anatomía mínima

Una neurona es una célula especializada en **recibir, integrar y transmitir señales eléctricas**. Tiene cuatro partes que importan:

```
   Dendritas          Soma (cuerpo)        Axón              Terminales
   (entradas)         (integración)        (transmisión)     (salidas → sinapsis)
      \|/                 ___                                     /|\
    ───┼───────────────► (   ) ──────────────────────────────►  ──┼───
      /|\                 ‾‾‾                                     \|/
   miles de              suma todo lo      cable de salida    conecta con las
   conexiones            que llega         (único)            dendritas de otras
   entrantes                                                  neuronas
```

- **Dendritas**: reciben señales de otras neuronas. Una sola neurona puede tener miles de conexiones entrantes.
- **Soma**: integra (suma) todas esas señales entrantes.
- **Axón**: si la suma supera cierto umbral, dispara una señal eléctrica que viaja por el axón.
- **Sinapsis**: el punto de contacto donde el axón de una neurona se comunica con la dendrita de la siguiente. **Aquí es donde reside el aprendizaje.**

### 0.2 Cómo se transmite la señal

Es un proceso **mixto: eléctrico dentro de la neurona, químico entre neuronas.**

1. Llegan señales a las dendritas y el soma las va acumulando.
2. Si la acumulación supera un **umbral de disparo** (unos −55 mV), la neurona dispara un **potencial de acción**: un pulso eléctrico que recorre el axón.
3. Si no llega al umbral, **no pasa nada**. No hay disparo "a medias".
4. Al llegar al final del axón, el pulso libera **neurotransmisores** (moléculas químicas) en la sinapsis, que cruzan el hueco y afectan a la siguiente neurona.

Dos propiedades clave de este proceso:

- **Todo o nada**: la neurona dispara o no dispara. La intensidad del pulso individual es siempre la misma. Lo que codifica "cuánta señal" es la **frecuencia** de disparos (más estímulo → dispara más veces por segundo), no la amplitud del pulso.
- **Suma de entradas**: la neurona no reacciona a una sola entrada, sino a la suma de todas ellas, tanto en el espacio (muchas dendritas a la vez) como en el tiempo (señales seguidas se acumulan).

### 0.3 Excitación e inhibición: el origen del "signo" del peso

No todas las conexiones empujan en la misma dirección:

- **Sinapsis excitatorias**: acercan a la neurona al umbral → favorecen el disparo.
- **Sinapsis inhibitorias**: alejan del umbral → dificultan el disparo.

Esto es exactamente lo que en una red artificial representa el **signo del peso**: positivo = excitatorio, negativo = inhibitorio. Que los pesos puedan ser negativos no es un truco matemático arbitrario, tiene su correlato biológico directo.

### 0.4 Cómo aprende: plasticidad sináptica

Aquí está la parte importante. **El aprendizaje no cambia las neuronas, cambia la fuerza de las conexiones entre ellas.**

Una sinapsis puede volverse más fuerte o más débil, físicamente: liberando más o menos neurotransmisor, teniendo más o menos receptores, creando nuevas conexiones o eliminando las que no se usan. Esto se llama **plasticidad sináptica**.

La regla clásica que lo describe es la **regla de Hebb** (Donald Hebb, 1949), resumida popularmente como:

> **"Neurons that fire together, wire together."**
> Si la neurona A participa repetidamente en hacer disparar a la neurona B, la conexión A→B se refuerza.

Los mecanismos concretos que se han medido en el laboratorio son:

- **LTP (potenciación a largo plazo)**: uso repetido y coordinado → la sinapsis se fortalece de forma duradera.
- **LTD (depresión a largo plazo)**: activación descoordinada o falta de uso → la sinapsis se debilita.

**Esa "fuerza de la sinapsis" es, conceptualmente, el peso de una red neuronal artificial.** Aprender = ajustar esas fuerzas.

---

## 0.5 La traducción al perceptrón: qué copiamos exactamente

El perceptrón de Rosenblatt (1958) es una **caricatura funcional** de todo lo anterior:

$$\text{salida} = f\left(\sum_i w_i x_i + b\right)$$

| Biología | Modelo artificial | Qué representa |
|---|---|---|
| Dendritas (entradas de otras neuronas) | $x_1, x_2, ..., x_n$ | Las señales que llegan |
| Fuerza de cada sinapsis (nº de receptores, neurotransmisor liberado) | **Peso** $w_i$ | Cuánto importa esa entrada concreta |
| Sinapsis excitatoria / inhibitoria | Peso **positivo / negativo** | Si esa entrada empuja a favor o en contra |
| Integración de señales en el soma | **Suma ponderada** $\sum w_i x_i$ | Acumular todo lo que llega |
| Umbral de disparo (−55 mV) | **Sesgo (bias)** $b$ | Qué tan fácil o difícil es que se active |
| Disparo "todo o nada" / frecuencia de disparo | **Función de activación** $f$ | Convertir la suma en una salida |
| Plasticidad sináptica (LTP / LTD) | **Entrenamiento** (ajuste de $w$ y $b$) | El aprendizaje propiamente dicho |
| Axón hacia otras neuronas | Salida hacia la siguiente capa | Propagación de la señal |

### El sesgo (bias), explicado desde aquí

El sesgo suele ser lo que peor se entiende, y desde la biología es evidente: **es el umbral de disparo**. Una neurona con umbral bajo se activa con poco estímulo; una con umbral alto necesita mucha señal acumulada.

En el modelo, un bias grande y positivo hace que la neurona se active fácilmente; uno muy negativo hace que necesite mucha evidencia acumulada para activarse. Por eso el bias **también se aprende**: la red no solo aprende *qué* entradas importan (pesos), sino *cuánta* evidencia total necesita para reaccionar (sesgo).

### Las funciones de activación, explicadas desde aquí

La función de activación existe para imitar el hecho de que **una neurona no responde de forma proporcional y lineal a lo que recibe**:

- **Escalón (perceptrón original)**: la imitación más literal del "todo o nada". Dispara (1) o no dispara (0). Problema: no es derivable, así que no se puede entrenar con backpropagation.
- **Sigmoide / tanh**: versión suavizada. Refleja que la frecuencia de disparo crece con el estímulo pero **se satura** (una neurona tiene un máximo físico de disparos por segundo). Al ser derivable, permite entrenar con backprop — pero su saturación es justo lo que causa el problema del vanishing gradient (Bloque 4).
- **ReLU** (`max(0, x)`): sorprendentemente, es la más parecida al comportamiento real medido en muchas neuronas — por debajo del umbral no hay respuesta (0), y por encima la frecuencia de disparo crece de forma aproximadamente lineal. Además su derivada es 1 en la zona activa, lo que evita el desvanecimiento del gradiente. Por eso desbancó a la sigmoide desde AlexNet (2012).

**El punto crucial:** sin función de activación, apilar capas es inútil — la composición de funciones lineales sigue siendo lineal, y tendrías un simple clasificador lineal por muchas capas que pongas. **La no linealidad es lo que hace posible el deep learning**, y su justificación es a la vez matemática y biológica.

---

## 0.6 Dónde la analogía se rompe (importante)

Esta parte se omite a menudo y conviene tenerla clara desde el principio: **las redes neuronales artificiales no son un modelo del cerebro.** Son una abstracción inspirada en él, y muy simplificada.

| Cerebro | Red neuronal artificial |
|---|---|
| Comunica con **pulsos discretos en el tiempo** (spikes); el *cuándo* dispara codifica información | Trabaja con números continuos; el tiempo no existe salvo que se modele aparte |
| Aprende **localmente**: cada sinapsis se ajusta con información disponible en su entorno inmediato | Aprende con **backpropagation**: requiere propagar una señal global de error hacia atrás por toda la red |
| No se ha encontrado un mecanismo biológico equivalente a backprop (el "problema de la asignación de crédito" sigue abierto) | Backprop es el núcleo del entrenamiento |
| ~86.000 millones de neuronas con ~10.000 conexiones cada una, consumiendo ~20 W | Modelos grandes consumen megavatios para entrenarse |
| Aprende **continuamente**, de pocos ejemplos, sin olvidar lo anterior | Se entrena en una fase separada; aprender algo nuevo puede provocar *olvido catastrófico* |
| Neuromoduladores (dopamina, serotonina) cambian globalmente el modo de funcionamiento | No hay equivalente estándar |
| La estructura física cambia: crecen y mueren conexiones | La arquitectura es fija durante el entrenamiento |

**Conclusión honesta:** la neurona biológica dio la **metáfora inicial** (entradas ponderadas, umbral, activación, aprendizaje por ajuste de conexiones), y esa metáfora fue suficiente para arrancar el campo. Pero todo lo que viene a partir del Bloque 1 —backpropagation, convoluciones, puertas LSTM, atención— **se justifica por razones matemáticas y de ingeniería, no biológicas**. El Transformer no imita nada del cerebro: es una solución a un problema de flujo de gradientes y paralelización.

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 📖 Esta sección. Con entender la tabla de 0.5 (biología → modelo) es suficiente para continuar.

**🔵 Nivel Profundo (solo si te interesa el tema)**
- 🌐 Khan Academy — módulo de neurona, potencial de acción y sinapsis (biología de bachillerato, gratuito).
- 📄 Donald Hebb (1949), *The Organization of Behavior* — el origen de "fire together, wire together".
- 📄 McCulloch & Pitts (1943), *A Logical Calculus of the Ideas Immanent in Nervous Activity* — el primer modelo matemático de una neurona.
- 🌐 Busca "Spiking Neural Networks" si te interesa la línea de investigación que **sí** intenta modelar los pulsos temporales de forma realista.

### ✅ Validación (3 frases)
1. ¿Dónde reside físicamente el aprendizaje en el cerebro, y cómo se llama ese fenómeno?
2. ¿Qué representan, en términos biológicos, el peso, el sesgo y la función de activación?
3. Nombra dos diferencias fundamentales entre cómo aprende el cerebro y cómo aprende una red artificial.

---

# BLOQUE 1 — El Perceptrón y el MLP
### 🗓️ Sesión 1 · 🎯 Problema: un clasificador lineal no puede resolver XOR

## 1. El Perceptrón y el Multi-Layer Perceptron (MLP)

### 1.1 El Perceptrón (Frank Rosenblatt, 1958)
Es la neurona artificial más simple: una suma ponderada de entradas + un sesgo (*bias*), pasada por una función escalón. Se entrena ajustando los pesos cuando se equivoca. Es, en esencia, un clasificador lineal.

**Limitación fatal:** en 1969, Marvin Minsky y Seymour Papert demostraron en su libro *Perceptrons* que un perceptrón simple **no puede resolver el problema XOR** (no es linealmente separable). Esto frenó la investigación en redes neuronales durante casi 15 años (el primer "invierno de la IA").

### 1.2 El MLP y el redescubrimiento del Backpropagation (1986)
La solución al problema XOR era simple sobre el papel: apilar varias capas de neuronas (capas ocultas) con funciones de activación **no lineales** (sigmoide, tanh). Eso es un **Multi-Layer Perceptron (MLP)**.

El problema real era **cómo entrenar** esas capas ocultas: ¿cómo sabe una neurona de la capa 1 si se equivocó, si el error solo se mide en la salida final? La respuesta fue el algoritmo de **backpropagation** (retropropagación del error), popularizado en 1986 por Rumelhart, Hinton y Williams: usando la regla de la cadena del cálculo diferencial, el error se propaga "hacia atrás" capa por capa, permitiendo calcular el gradiente de cada peso respecto a la función de pérdida y actualizarlo con **descenso de gradiente**.

Con esto nace la receta que sigue vigente hoy: **arquitectura + función de pérdida + backpropagation + descenso de gradiente**.

**Ideas clave de esta sección:**
- Una sola capa lineal = clasificador lineal, limitado.
- Apilar capas + no linealidades = aproximador universal de funciones (teorema de aproximación universal).
- Backprop no es "otro algoritmo", es la aplicación mecánica de la regla de la cadena a todo el grafo de cómputo.

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 🎥 **INTUICIÓN** — 3Blue1Brown, *Neural Networks* (caps. 1-4): [playlist en YouTube](https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s)
  - **Cap. 1 — ¿Qué es una red neuronal?**: usa el reconocimiento de dígitos escritos a mano (MNIST) para explicar qué son neuronas, capas, pesos y sesgos, y por qué una red se puede pensar como una función que va ajustando esos parámetros.
  - **Cap. 2 — Descenso de gradiente**: explica cómo aprende la red — define una función de coste, muestra por qué minimizarla equivale a "bajar una colina" en un espacio de muchísimas dimensiones, e introduce la idea de backpropagation como el método eficiente para calcular esa dirección de bajada.
  - **Cap. 3 y 4 — Backpropagation (intuición + cálculo)**: primero da la intuición de cómo el error de la salida se reparte hacia atrás entre las neuronas anteriores según su responsabilidad; después formaliza esa intuición con la regla de la cadena, capa por capa, hasta llegar a las fórmulas reales que se usan para entrenar.
- 📖 **FORMALIZACIÓN** — Secciones **12.A** (sin matemáticas) y **12.B** (con matemáticas) de esta guía.

**🔵 Nivel Profundo (solo si hace falta)**
- 🎥 StatQuest — *Neural Networks Pt. 1-3* (main ideas, backprop, ReLU): [índice de vídeos](https://statquest.org/video_index.html)
  - **Resumen:** Josh Starmer construye una red pequeñísima a mano para mostrar, paso a paso y sin código, cómo se combinan las entradas, cómo actúan funciones de activación como la sigmoide o la ReLU, y cómo el algoritmo de backpropagation ajusta los pesos comparando la predicción con el valor real. Menos animación, más "hazlo con lápiz y papel".
- 🌐 Michael Nielsen — [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) (libro online gratuito, con MNIST paso a paso).
- 📄 Rumelhart, Hinton, Williams (1986), *Learning representations by back-propagating errors* — [PDF](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
- 📄 Minsky & Papert, *Perceptrons* (1969) — contexto histórico del primer invierno de la IA.

### ✅ Validación (escribe 3 frases antes de pasar al Bloque 2)
1. ¿Qué fallaba en el perceptrón simple?
2. ¿Qué elemento de diseño se añadió para resolverlo, y qué algoritmo hizo falta para entrenarlo?
3. ¿Qué nuevo problema introduce esta solución? *(pista: ¿qué pasa con el gradiente cuando apilas muchas capas?)*

### 🛠️ Hito 1 de codificación
Programa a mano, solo con numpy o Python puro, el forward pass y el backprop de un MLP que resuelva XOR. Usa la sección **12.B.6** como plantilla.

---

# BLOQUE 2 — CNN y ResNet
### 🗓️ Sesión 2 · 🎯 Problema: un MLP destruye la estructura espacial de una imagen

## 2. Redes Neuronales Convolucionales (CNN)

### 2.1 Motivación
Un MLP tratando una imagen la "aplana" en un vector larguísimo, perdiendo toda la estructura espacial (qué píxel está al lado de cuál) y necesitando un número de parámetros enorme. Las CNN resuelven esto con dos ideas:

1. **Pesos compartidos (filtros/kernels):** el mismo detector de bordes se aplica en toda la imagen, no uno distinto por posición.
2. **Conectividad local:** cada neurona solo "mira" una región pequeña (campo receptivo), no toda la imagen.

Esto reduce drásticamente el número de parámetros y da a la red una propiedad natural: **invarianza (aproximada) a la traslación** — un gato sigue siendo un gato se mueva donde se mueva en la imagen.

### 2.2 Hitos
- **LeNet-5** (Yann LeCun, 1998): reconocimiento de dígitos escritos a mano (cheques bancarios). Primera CNN "moderna" entrenada con backprop a escala real.
- **AlexNet** (Krizhevsky, Sutskever, Hinton, 2012): gana ImageNet por un margen enorme usando GPUs y ReLU. Es el momento que reinicia toda la ola de "deep learning" moderna.
- **VGG** (2014): demuestra que apilar muchas capas de filtros pequeños (3x3) funciona muy bien.
- **ResNet** (He et al., 2015): introduce las **conexiones residuales** (*skip connections*), que permiten entrenar redes de cientos de capas sin que el gradiente se pierda por el camino — una solución **arquitectónica** al mismo problema de fondo que veremos en el Bloque 4.

**Por qué importa para el resto de la historia:** ResNet es el primer gran ejemplo de "si el gradiente no fluye bien a través de muchas capas, cambia la arquitectura para darle una autopista directa". Esa misma filosofía reaparecerá en LSTM y, más adelante, en las conexiones residuales que usa el propio Transformer.

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 🎥 **INTUICIÓN** — StatQuest, *Convolutional Neural Networks, Clearly Explained* (con notebook): [índice de vídeos](https://statquest.org/video_index.html)
  - **Resumen:** parte de un MLP normal aplicado a una imagen para mostrar por qué explota en número de parámetros, y a partir de ahí introduce el filtro convolucional como un "escáner" que se desliza por la imagen buscando el mismo patrón (un borde, una textura) en cualquier posición. Cubre también el *max pooling* y cómo se apilan varias capas convolucionales antes de aplanar y conectar con capas densas.
- 📖 **FORMALIZACIÓN** — esta sección (2.1 y 2.2), prestando especial atención a las skip connections de ResNet.

**🔵 Nivel Profundo (solo si hace falta)**
- 🌐 CS231n (Stanford) — [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) — notas + vídeos.
- 📄 He, Zhang, Ren, Sun (2015), *Deep Residual Learning* (ResNet) — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) — **el más relevante para la narrativa de esta guía**.
- 📄 Krizhevsky et al. (2012), *AlexNet* — [PDF (NeurIPS)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- 📄 LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* — [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- 📄 Simonyan & Zisserman (2014), *VGG* — [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

### ✅ Validación (3 frases)
1. ¿Qué fallaba en el MLP al procesar imágenes?
2. ¿Qué dos ideas de diseño lo resuelven, y qué propiedad emergente conceden?
3. ¿Qué problema aparece al apilar muchas capas convolucionales, y cómo lo resuelve ResNet?

---

# BLOQUE 3 — Redes Recurrentes (RNN)
### 🗓️ Sesión 3 (primera mitad) · 🎯 Problema: las secuencias tienen longitud variable y memoria

## 3. Redes Neuronales Recurrentes (RNN)

### 3.1 Motivación
Las CNN y los MLP asumen entradas de tamaño fijo. Pero el lenguaje, el audio, las series temporales... son **secuencias de longitud variable**, con dependencias entre elementos alejados en el tiempo. Necesitamos una arquitectura con "memoria".

### 3.2 Idea central
Una RNN procesa la secuencia elemento a elemento, manteniendo un **estado oculto** \(h_t\) que se actualiza en cada paso combinando la entrada actual \(x_t\) con el estado anterior \(h_{t-1}\):

\[
h_t = f(W_x x_t + W_h h_{t-1} + b)
\]

Ese mismo bloque de pesos (\(W_x, W_h\)) se **reutiliza en cada paso temporal** — es la misma idea de "compartir pesos" que en las CNN, pero a lo largo del tiempo en vez de a lo largo del espacio. Si "desenrollamos" (*unroll*) la RNN en el tiempo, se parece a una red muy profunda donde cada "capa" es un paso temporal.

Esto permite manejar secuencias de longitud arbitraria y, en teoría, capturar dependencias de largo alcance (lo dicho al principio de una frase afecta a lo dicho al final).

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 🎥 **INTUICIÓN** — StatQuest, [*Recurrent Neural Networks (RNNs), Clearly Explained!!!*](https://www.youtube.com/watch?v=AsNTP8Kwu80)
  - **Resumen:** parte de un problema muy concreto (predecir un valor cuando cada ejemplo tiene una cantidad distinta de datos de entrada, como días de cotización de una acción) para mostrar por qué un MLP normal no sirve, al necesitar un tamaño de entrada fijo. A partir de ahí construye la RNN "desenrollando" el mismo bloque de pesos en cada paso temporal, y muestra cómo la salida de un paso se retroalimenta al siguiente para dar a la red una forma de "memoria".
- 📖 **FORMALIZACIÓN** — esta sección (3.1 y 3.2). Céntrate en entender el *unrolling*: es la clave para el Bloque 4.

**🔵 Nivel Profundo (solo si hace falta)**
- 🌐 Andrej Karpathy — [*The Unreasonable Effectiveness of Recurrent Neural Networks*](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - **Resumen:** demuestra, con una RNN entrenada carácter a carácter, que un modelo tan simple puede generar desde código en C que casi compila hasta texto con la estructura de una obra de Shakespeare. El argumento central es que darle a una red un estado interno a lo largo de una secuencia es sorprendentemente potente — la motivación perfecta antes de ver por qué hicieron falta LSTM y atención.
- 📄 Elman (1990), *Finding Structure in Time* — uno de los papers fundacionales de las RNN modernas.

### ✅ Validación (3 frases)
1. ¿Qué fallaba en CNN/MLP con datos secuenciales?
2. ¿Qué elemento de diseño se añadió (estado oculto + pesos compartidos en el tiempo)?
3. ¿Qué pasa con el gradiente al desenrollar la red 100 pasos? → **eso es exactamente el Bloque 4.**

---

# BLOQUE 4 — El problema del gradiente (vanishing / exploding)
### 🗓️ Sesión 3 (segunda mitad) · 🎯 Problema: multiplicar muchas derivadas destruye la señal de aprendizaje

> **Este es el bloque bisagra de toda la guía.** Es el problema técnico concreto que motiva LSTM, luego la atención, y en última instancia el propio Transformer. Si solo entiendes un bloque a fondo, que sea este.

## 4. El problema del desvanecimiento (y explosión) del gradiente

### 4.1 ¿Qué pasa exactamente?
Para entrenar una RNN se usa **Backpropagation Through Time (BPTT)**: se desenrolla la red en el tiempo y se aplica backprop normal sobre esa red "profunda" (una capa por paso temporal). El problema es que, por la regla de la cadena, el gradiente que llega al paso \(t=1\) desde un error en \(t=100\) es un **producto de ~100 matrices de derivadas** (una por cada paso intermedio):

\[
\frac{\partial h_{100}}{\partial h_1} = \prod_{t=2}^{100} \frac{\partial h_t}{\partial h_{t-1}}
\]

- Si esas derivadas tienden a ser **< 1** en promedio, el producto tiende a **0** exponencialmente rápido → **vanishing gradient** (gradiente que desaparece). La red "olvida" lo que pasó hace muchos pasos: no puede aprender dependencias de largo alcance.
- Si tienden a ser **> 1**, el producto **explota** → **exploding gradient**. Los pesos se actualizan con saltos gigantes y el entrenamiento diverge (a veces se ve como pérdida = `NaN`).

La explosión se puede mitigar de forma sencilla con **gradient clipping** (recortar la norma del gradiente). El desvanecimiento es el problema realmente difícil, y es estructural: no es un bug, es una consecuencia matemática de multiplicar muchos números pequeños.

### 4.2 Consecuencia práctica
Una RNN "vainilla" (simple) en la práctica solo puede aprender dependencias de **unos pocos pasos** (5-10), no las decenas o cientos de palabras que necesitaríamos para entender un párrafo o traducir una frase larga.

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 📖 **FORMALIZACIÓN** — esta sección + sección **12.B.9** de esta guía (por qué importa en la práctica).
- 🎥 **INTUICIÓN** — el problema se explica como motivación al inicio del vídeo de LSTM del Bloque 5. Puedes verlo aquí y encadenar directamente con el Bloque 5.

**🔵 Nivel Profundo (solo si hace falta)**
- 📄 Pascanu, Mikolov, Bengio (2013), *On the difficulty of training RNNs* — [arXiv:1211.5063](https://arxiv.org/abs/1211.5063) — el más accesible de los tres; formaliza el *gradient clipping*.
- 📄 Bengio, Simard, Frasconi (1994), *Learning Long-Term Dependencies with Gradient Descent is Difficult*.
- 📄 Hochreiter (1991), tesis de diploma (en alemán) — el primer análisis riguroso del problema.

### ✅ Validación (3 frases)
1. ¿Por qué el BPTT convierte una RNN en una red "muy profunda"?
2. ¿Por qué un producto de muchas derivadas <1 tiende a cero, y qué implica eso para el aprendizaje?
3. ¿Cuál de los dos problemas (vanishing / exploding) se arregla fácil y cuál no, y por qué?

---

# BLOQUE 5 — LSTM y GRU
### 🗓️ Sesión 4 · 🎯 Problema: darle al gradiente una autopista para atravesar el tiempo

## 5. LSTM y GRU: la primera solución de peso

### 5.1 LSTM (Long Short-Term Memory) — Hochreiter & Schmidhuber, 1997
La idea genial del LSTM es separar **dos tipos de memoria**:
- Una **memoria de largo plazo** (*cell state*, \(C_t\)) que fluye casi sin obstáculos a través del tiempo, como una "cinta transportadora".
- Una **memoria de corto plazo** (*hidden state*, \(h_t\)).

Y controlar el flujo de información con tres **puertas** (*gates*), cada una una capa sigmoide que decide "cuánto dejar pasar" (entre 0 y 1):
- **Forget gate**: qué porcentaje de la memoria de largo plazo se olvida.
- **Input gate**: qué porcentaje de la nueva información entra a la memoria de largo plazo.
- **Output gate**: qué parte de la memoria de largo plazo se expone como salida en este paso.

Al permitir que el *cell state* se actualice mediante sumas (en vez de solo multiplicaciones repetidas), el gradiente puede fluir hacia atrás sin desvanecerse tan fácilmente. Es, conceptualmente, un antepasado directo de las **conexiones residuales** que veríamos años después en ResNet y en el Transformer.

### 5.2 GRU (Gated Recurrent Unit) — Cho et al., 2014
Una simplificación del LSTM: fusiona el *cell state* y el *hidden state* en uno solo, y usa solo dos puertas (*reset* y *update*) en vez de tres. Funciona casi igual de bien en muchas tareas con menos parámetros, por lo que se popularizó mucho en 2014-2017.

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 🎥 **INTUICIÓN** — StatQuest, [*Long Short-Term Memory (LSTM), Clearly Explained*](https://www.youtube.com/watch?v=YCzL96nL7j0) (21 min, con doblaje en español)
  - **Resumen:** explica el LSTM como tres etapas consecutivas dentro de cada celda: (1) decidir qué porcentaje de la memoria de largo plazo se olvida (*forget gate*), (2) actualizar esa memoria combinando lo recordado con la nueva entrada (*input gate*), y (3) generar, a partir de ella, la memoria de corto plazo que pasa al siguiente paso (*output gate*). Deja claro por qué esta separación en dos memorias evita el vanishing gradient.
- 🌐 **FORMALIZACIÓN** — Christopher Olah, [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — el artículo ilustrado de referencia, con los diagramas que todo el mundo acaba copiando.

**🔵 Nivel Profundo (solo si hace falta)**
- 📄 Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 📄 Cho et al. (2014), *RNN Encoder-Decoder* (introduce GRU) — [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)

### ✅ Validación (3 frases)
1. ¿Qué fallaba en la RNN simple?
2. ¿Qué elemento de diseño lo resuelve, y por qué una **suma** protege mejor el gradiente que una multiplicación?
3. ¿Qué problema **sigue existiendo** aunque uses LSTM? *(pista: sigue siendo secuencial, y comprimir una frase entera en un vector sigue siendo un cuello de botella → Bloque 6)*

### 🛠️ Hito 2 de codificación
Entrena una RNN o LSTM básica en PyTorch para predecir el siguiente carácter de un texto corto.

---

# BLOQUE 6 — Seq2Seq y el nacimiento de la Atención
### 🗓️ Sesión 5 · 🎯 Problema: comprimir una frase entera en un solo vector de tamaño fijo

## 6. Seq2Seq y el nacimiento de la Atención

### 6.1 El modelo Encoder-Decoder (Sutskever, Vinyals, Le — 2014)
Para tareas como traducción automática (secuencia de entrada → secuencia de salida, de longitudes distintas) se propone **Sequence to Sequence (Seq2Seq)**:
- Un **encoder** (una RNN/LSTM) lee toda la frase de entrada y la "comprime" en **un único vector de contexto** de tamaño fijo (el último estado oculto).
- Un **decoder** (otra RNN/LSTM) toma ese vector y genera la frase de salida, palabra a palabra.

### 6.2 El cuello de botella
El problema es evidente: **toda una frase, sin importar si tiene 5 o 50 palabras, se aprieta en un único vector de tamaño fijo**. Es como pedirle a alguien que lea un párrafo entero y luego resuma todo su significado en una sola palabra antes de empezar a traducir. Cuanto más larga la frase, peor funciona el modelo — el mismo síntoma del vanishing gradient, visto ahora desde el ángulo de la arquitectura.

### 6.3 La solución: Atención (Bahdanau et al., 2014)
La idea, presentada por Bahdanau, Cho y Bengio, es simple pero transformó el campo: **en vez de forzar al decoder a usar un solo vector de contexto fijo, dejemos que en cada paso de la generación "mire hacia atrás" y decida a qué palabras de la entrada prestar más atención.**

Mecánicamente:
1. El encoder ya no produce un solo vector final, sino un estado oculto **por cada palabra** de entrada.
2. En cada paso del decoder, se calcula una puntuación de "compatibilidad" entre el estado actual del decoder y cada uno de esos estados del encoder.
3. Esas puntuaciones se normalizan con un `softmax` → son los **pesos de atención**.
4. El vector de contexto para ese paso es la **suma ponderada** de los estados del encoder según esos pesos.

Esto es, literalmente, un mecanismo que aprende **a qué prestar atención**, dando nombre al concepto. Poco después, Luong et al. (2015) simplificaron y generalizaron la idea (atención "global" vs "local", y variantes multiplicativas más eficientes de calcular).

**Esta es la pieza que, tres años más tarde, los autores de "Attention Is All You Need" se preguntarán: ¿y si la atención no es solo un complemento de la RNN, sino todo lo que necesitamos?**

### 📚 Recursos filtrados

**🟢 Nivel Base (obligatorio)**
- 🌐 **INTUICIÓN + FORMALIZACIÓN** — Jay Alammar, [*Visualizing A Neural Machine Translation Model (seq2seq with attention)*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  - **Resumen:** anima paso a paso un modelo seq2seq clásico y luego el mismo modelo con atención, mostrando visualmente cómo cambia el "foco" del decoder sobre las palabras de la entrada en cada paso de la traducción. La mejor forma de ver por qué la atención soluciona el cuello de botella del vector único.
- 📖 Esta sección (6.1 a 6.3).

**🔵 Nivel Profundo (solo si hace falta)**
- 📄 Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) — **el paper que introduce la atención; corto y muy legible.**
- 📄 Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning* — [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- 📄 Luong, Pham, Manning (2015), *Effective Approaches to Attention-based NMT* — [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)

### ✅ Validación (3 frases)
1. ¿Qué fallaba en el Seq2Seq clásico (cuello de botella)?
2. ¿Qué elemento se añadió, y de dónde sale el vector de contexto ahora?
3. ¿Qué problema **sigue existiendo** aunque uses atención sobre una RNN? *(pista: la recurrencia sigue impidiendo paralelizar)*

---

# BLOQUE 7 — El Transformer (*Attention Is All You Need*)
### 🗓️ Sesiones 6 y 7 · 🎯 Problema: la recurrencia impide paralelizar y alarga el camino entre palabras

> **Divide este bloque en dos sesiones:** Sesión 6 → apartados 7.1 y 7.2 (self-attention, Q/K/V). Sesión 7 → apartados 7.3 y 7.4 (arquitectura completa e impacto).

## 7. Attention Is All You Need (Vaswani et al., 2017): el Transformer

### 7.1 La pregunta que lo cambió todo
Para 2017, la atención ya se usaba como un *complemento* de las RNN. El equipo de Google Brain/Research (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin) se hizo una pregunta radical: **si la atención es lo que realmente hace el trabajo pesado, ¿para qué seguimos arrastrando la recurrencia?**

Las RNN tienen dos problemas estructurales, incluso con LSTM/atención:
1. **Secuencialidad forzada:** el paso \(t\) depende de haber calculado el paso \(t-1\). Esto impide paralelizar el cómputo dentro de una secuencia — muy mal para aprovechar GPUs/TPUs.
2. **Camino largo entre palabras lejanas:** aunque la atención ayuda, la información entre la palabra 1 y la palabra 100 en una RNN todavía tiene que "viajar" por muchos pasos intermedios.

**Attention Is All You Need** propone eliminar por completo la recurrencia (y la convolución) y construir un modelo **basado únicamente en mecanismos de atención**: el **Transformer**.

### 7.2 Los ingredientes clave

**a) Self-Attention (auto-atención)**
En vez de que el decoder atienda al encoder (como en Bahdanau), cada palabra de una secuencia atiende a **todas las demás palabras de la misma secuencia** (incluida ella misma) para construir su representación. Cada palabra se proyecta en tres vectores: **Query (Q)**, **Key (K)** y **Value (V)**. La atención se calcula como:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Intuición: la Query de una palabra "pregunta" a las Keys de todas las demás palabras cuánto se parecen a lo que busca; esos parecidos (normalizados con softmax) ponderan cuánto de cada Value se incorpora a la nueva representación de la palabra.

**b) Multi-Head Attention**
En vez de calcular una sola atención, se calculan **varias en paralelo** ("cabezas"), cada una pudiendo aprender a fijarse en un tipo distinto de relación (sintáctica, semántica, correferencia, etc.), y luego se concatenan los resultados.

**c) Positional Encoding**
Como ya no hay recurrencia ni convolución, el modelo no tiene ninguna noción inherente de **orden** de las palabras (la self-attention trata la secuencia como un conjunto). Para solucionarlo, se suman a los *embeddings* de entrada unas señales sinusoidales que codifican la posición de cada palabra.

**d) Arquitectura Encoder-Decoder**
El Transformer mantiene la estructura general encoder-decoder de Seq2Seq, pero cada bloque de encoder/decoder es una pila de capas de self-attention + una red feed-forward, todas conectadas con **conexiones residuales** y normalización (*layer norm*) — el mismo espíritu de "autopista para el gradiente" que ya habíamos visto en ResNet y LSTM, ahora aplicado a un modelo sin recurrencia.

### 7.3 Por qué fue una revolución
- **Paralelización masiva:** al no depender secuencialmente de sí mismo, todo el cálculo de self-attention se puede hacer en paralelo para toda la secuencia → entrenamiento muchísimo más rápido en GPU/TPU.
- **Dependencias de largo alcance "gratis":** la distancia entre dos palabras cualesquiera en la self-attention es **O(1)** (una sola operación), no O(n) como en una RNN — se acabó (en gran medida) el problema del gradiente que desaparece a través de la secuencia.
- **Escalabilidad:** esta arquitectura resultó escalar excepcionalmente bien con más datos y más parámetros, lo que sentó las bases de la era de los grandes modelos de lenguaje.

### 7.4 Lo que vino después (para cerrar el círculo)
- **BERT** (2018, Google): usa solo el *encoder* del Transformer, entrenado para entender contexto bidireccional.
- **GPT** (2018 en adelante, OpenAI): usa solo el *decoder*, entrenado de forma autoregresiva para generar texto — la línea directa hacia ChatGPT y los LLM actuales.
- Modelos como Claude, LLaMA, Gemini, etc. son, en su núcleo arquitectónico, variantes (mayormente decoder-only) de este mismo Transformer de 2017.

### 📚 Recursos filtrados

**🟢 Nivel Base — Sesión 6 (self-attention, Q/K/V)**
- 🎥 **INTUICIÓN** — 3Blue1Brown, [*But what is a GPT? Visual intro to Transformers*](https://www.youtube.com/watch?v=wjZofJX0v4M) (cap. 5)
  - **Resumen:** explica qué significa cada palabra de "Generative Pre-trained Transformer": cómo el texto se convierte en tokens y embeddings, cómo esos vectores se refinan capa a capa para incorporar contexto, y cómo al final la red produce una distribución de probabilidad sobre el siguiente token (incluyendo *temperature*).
- 🎥 **INTUICIÓN** — 3Blue1Brown, [*Attention in transformers, visually explained*](https://www.youtube.com/watch?v=eMlx5fFNoYc) (cap. 6)
  - **Resumen:** el capítulo dedicado por completo a la atención. Muestra cómo cada palabra parte de un embedding "genérico" y cómo, vía las matrices Q, K y V, incorpora el contexto de las palabras cercanas (ejemplo clásico: "banco" cambia según si aparece "río" o "dinero"). Cubre el enmascarado (*masking*), las múltiples cabezas y cómo se combinan sus salidas.
- 🌐 **FORMALIZACIÓN** — Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — el recurso más citado del mundo para entender el Transformer visualmente.

**🟢 Nivel Base — Sesión 7 (arquitectura completa)**
- 🎥 Yannic Kilcher, [*Attention Is All You Need* explicado](https://www.youtube.com/watch?v=iDulhoQ2pro) (~27 min)
  - **Resumen:** lectura guiada del paper original: cómo se procesaba el lenguaje antes (RNN/CNN), el problema de las dependencias de largo alcance, y luego el mecanismo de atención, el positional encoding y el cálculo de los pesos sobre la matriz de la frase. Cierra con el impacto del paper (mejores resultados en traducción con menos tiempo de entrenamiento).
- 📖 Apartados 7.3 y 7.4 de esta guía.

**🔵 Nivel Profundo (solo si hace falta)**
- 📄 **El paper original**: Vaswani et al. (2017), *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — léelo *después* de la explicación ilustrada, y solo introducción, arquitectura y conclusiones.
- 🎥 StatQuest — serie sobre *Transformer Neural Networks* y *Decoder-Only Transformers* (con notebooks) — [índice](https://statquest.org/video_index.html)
- 📄 Devlin et al. (2018), *BERT* — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- 📄 Radford et al. (2018), *GPT-1* — [PDF (OpenAI)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### ✅ Validación (3 frases)
1. ¿Qué dos cosas fallaban en las RNN con atención?
2. ¿Qué se eliminó y qué hubo que añadir a cambio? *(pista: sin recurrencia se pierde la noción de orden)*
3. ¿Qué problema nuevo introduce la self-attention? *(pista: coste cuadrático O(n²) respecto a la longitud de la secuencia — el problema abierto que motiva casi toda la investigación posterior en contextos largos)*

### 🛠️ Hito 3 de codificación
Sigue el vídeo de Andrej Karpathy [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m) escribiendo el código a la vez.
- **Resumen:** en la primera mitad construye un modelo de lenguaje simple (bigramas) e introduce la atención como "paso de mensajes" entre nodos de un grafo dirigido; en la segunda completa el Transformer añadiendo atención multi-cabeza, red feed-forward, conexiones residuales y *layer norm*. Termina entrenando nanoGPT sobre texto de Shakespeare.

---

## 8. Línea temporal resumen

| Año | Hito | Problema que resuelve |
|---|---|---|
| 1958 | Perceptrón (Rosenblatt) | Primera neurona artificial entrenable |
| 1969 | *Perceptrons* (Minsky & Papert) | Demuestra el límite del perceptrón simple (XOR) |
| 1986 | Backpropagation (Rumelhart, Hinton, Williams) | Permite entrenar MLP con capas ocultas |
| 1997 | LSTM (Hochreiter & Schmidhuber) | Vanishing gradient en RNN |
| 1998 | LeNet-5 (LeCun) | Reconocimiento de imágenes con CNN |
| 2012 | AlexNet (Krizhevsky et al.) | CNN a gran escala + GPU → explosión del deep learning |
| 2014 | GRU (Cho et al.) | Alternativa más simple al LSTM |
| 2014 | Seq2Seq (Sutskever et al.) | Traducción automática con RNN encoder-decoder |
| 2014 | Atención (Bahdanau et al.) | Cuello de botella del vector de contexto fijo |
| 2015 | ResNet (He et al.) | Entrenar redes muy profundas (conexiones residuales) |
| 2015 | Atención Luong | Simplifica y generaliza la atención |
| 2017 | **Attention Is All You Need** (Vaswani et al.) | Elimina la recurrencia; paralelización + dependencias largas |
| 2018 | BERT / GPT-1 | Pre-entrenamiento a gran escala con Transformers |
| 2020+ | GPT-3, LLaMA, Claude, etc. | Escalado masivo de la misma arquitectura |

---

## 9. Checklist de progreso por bloques

Marca un bloque **solo cuando hayas escrito las 3 frases de validación**, no cuando hayas visto el vídeo. Ver ≠ entender.

| ✅ | Sesión | Bloque | Nivel Base completado | Validación escrita | Hito de código |
|---|---|---|---|---|---|
| ☐ | 0 | **B0** — Neurona biológica | ☐ Sec. 0 (tabla 0.5) | ☐ | — |
| ☐ | 1 | **B1** — Perceptrón / MLP / Backprop | ☐ 3B1B caps. 1-4 + sec. 12.A/12.B | ☐ | ☐ Hito 1 (XOR a mano) |
| ☐ | 2 | **B2** — CNN y ResNet | ☐ StatQuest CNN + sec. 2 | ☐ | — |
| ☐ | 3a | **B3** — RNN | ☐ StatQuest RNN + sec. 3 | ☐ | — |
| ☐ | 3b | **B4** — Vanishing gradient | ☐ Sec. 4 + sec. 12.B.9 | ☐ | — |
| ☐ | 4 | **B5** — LSTM / GRU | ☐ StatQuest LSTM + Colah | ☐ | ☐ Hito 2 (char-RNN PyTorch) |
| ☐ | 5 | **B6** — Seq2Seq + Atención | ☐ Alammar seq2seq + sec. 6 | ☐ | — |
| ☐ | 6 | **B7.1-7.2** — Self-Attention, Q/K/V | ☐ 3B1B caps. 5-6 + Illustrated Transformer | ☐ | — |
| ☐ | 7 | **B7.3-7.4** — Transformer completo | ☐ Yannic Kilcher + sec. 7.3/7.4 | ☐ | — |
| ☐ | 8+ | **Práctica** | — | — | ☐ Hito 3 (Karpathy, Let's build GPT) |

**Criterio de parada final (Definition of Done):** reproduce de memoria la tabla de la Sección 8, razonando la columna *"problema que resuelve"* de cada hito sin mirar el documento.

---

## 10. Glosario rápido

- **Backpropagation**: algoritmo que calcula el gradiente de la función de pérdida respecto a cada peso, aplicando la regla de la cadena hacia atrás en el grafo de cómputo.
- **Descenso de gradiente**: método de optimización que actualiza los pesos en la dirección opuesta al gradiente para minimizar la pérdida.
- **Vanishing/Exploding gradient**: fenómeno por el cual el gradiente se hace exponencialmente pequeño o grande al propagarse por muchas capas/pasos temporales.
- **Cell state / hidden state (LSTM)**: memoria de largo plazo y de corto plazo, respectivamente, dentro de una celda LSTM.
- **Vector de contexto**: representación comprimida de una secuencia de entrada usada por el decoder en Seq2Seq.
- **Self-attention**: mecanismo por el cual cada elemento de una secuencia calcula su nueva representación atendiendo a todos los demás elementos de la misma secuencia.
- **Query, Key, Value (Q, K, V)**: las tres proyecciones lineales de cada token usadas para calcular la atención.
- **Multi-Head Attention**: varias operaciones de atención en paralelo, cada una aprendiendo un tipo distinto de relación entre tokens.
- **Positional Encoding**: señal añadida a los embeddings de entrada para darle al Transformer información sobre el orden de la secuencia, ya que la self-attention por sí sola no la tiene.
- **Conexión residual (skip connection)**: atajo que suma la entrada de una capa a su salida, facilitando el flujo del gradiente en redes profundas.

---

## 11. Bibliografía y recursos completos (lista maestra)

### Papers fundamentales (orden cronológico)
1. Rosenblatt (1958) — *The Perceptron*
2. Minsky & Papert (1969) — *Perceptrons*
3. Rumelhart, Hinton, Williams (1986) — [*Learning representations by back-propagating errors*](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
4. LeCun et al. (1998) — [*Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
5. Hochreiter & Schmidhuber (1997) — [*Long Short-Term Memory*](https://www.bioinf.jku.at/publications/older/2604.pdf)
6. Bengio, Simard, Frasconi (1994) — *Learning Long-Term Dependencies with Gradient Descent is Difficult*
7. Krizhevsky, Sutskever, Hinton (2012) — [*ImageNet Classification with Deep CNNs* (AlexNet)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
8. Simonyan & Zisserman (2014) — [*VGG*](https://arxiv.org/abs/1409.1556)
9. Sutskever, Vinyals, Le (2014) — [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215)
10. Cho et al. (2014) — [*GRU*](https://arxiv.org/abs/1406.1078)
11. Bahdanau, Cho, Bengio (2014) — [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473)
12. Luong, Pham, Manning (2015) — [*Effective Approaches to Attention-based NMT*](https://arxiv.org/abs/1508.04025)
13. He, Zhang, Ren, Sun (2015) — [*Deep Residual Learning* (ResNet)](https://arxiv.org/abs/1512.03385)
14. Pascanu, Mikolov, Bengio (2013) — [*On the difficulty of training RNNs*](https://arxiv.org/abs/1211.5063)
15. **Vaswani et al. (2017) — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)**
16. Devlin et al. (2018) — [*BERT*](https://arxiv.org/abs/1810.04805)
17. Radford et al. (2018) — [*GPT-1*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
18. Brown et al. (2020) — [*GPT-3: Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)

### Vídeos de YouTube (con resumen de cada uno)

| Vídeo | Resumen en una línea |
|---|---|
| [3Blue1Brown — Serie completa *Neural Networks / Deep Learning*](https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s) | MLP, descenso de gradiente y backpropagation explicados con animaciones desde cero (caps. 1-4). |
| [3Blue1Brown — *But what is a GPT?* (cap. 5)](https://www.youtube.com/watch?v=wjZofJX0v4M) | Tokens, embeddings y cómo un Transformer produce la siguiente palabra. |
| [3Blue1Brown — *Attention in transformers, visually explained* (cap. 6)](https://www.youtube.com/watch?v=eMlx5fFNoYc) | Query/Key/Value, masking y multi-cabeza, animado sobre un ejemplo de frase real. |
| [StatQuest — Índice completo de vídeos y notebooks](https://statquest.org/video_index.html) | Catálogo con RNN, LSTM, CNN y Transformers, cada uno con notebook en PyTorch. |
| [StatQuest — *Recurrent Neural Networks, Clearly Explained!!!*](https://www.youtube.com/watch?v=AsNTP8Kwu80) | Por qué un MLP no sirve para secuencias de longitud variable y cómo la RNN lo resuelve reutilizando pesos en el tiempo. |
| [StatQuest — *Long Short-Term Memory, Clearly Explained*](https://www.youtube.com/watch?v=YCzL96nL7j0) | Las tres puertas del LSTM (forget, input, output) explicadas paso a paso. |
| [Yannic Kilcher — *Attention Is All You Need* explicado](https://www.youtube.com/watch?v=iDulhoQ2pro) | Lectura técnica guiada del paper original del Transformer. |
| [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Construye un GPT completo en código, de un modelo de bigramas a un Transformer funcional (nanoGPT). |
| [Andrej Karpathy — blog *The Unreasonable Effectiveness of RNNs*](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Demuestra el poder (y los límites) de una RNN entrenada carácter a carácter. |

### Blogs / artículos ilustrados
- Jay Alammar — [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- Jay Alammar — [*Visualizing A Neural Machine Translation Model (seq2seq + attention)*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- Christopher Olah — [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Michael Nielsen — [*Neural Networks and Deep Learning* (libro online gratuito)](http://neuralnetworksanddeeplearning.com/)

### Cursos
- Stanford CS231n — [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- Stanford CS224n — Natural Language Processing with Deep Learning (cubre RNN, atención y Transformers en profundidad)

---

## 12. Apéndice: Backpropagation y la regla de la cadena, de cero a experto

Esta sección tiene dos versiones de la misma idea: **12.A** la explica sin matemáticas, con analogías; **12.B** la explica con las matemáticas reales (cálculo, código). Se recomienda leer primero 12.A y luego 12.B — la intuición hace que las fórmulas de después tengan sentido en vez de ser solo símbolos.

## 12.A Versión conceptual (sin matemáticas)

### La idea central

Imagina que estás jugando a un juego: tienes que ajustar varios diales (los "pesos" de la red) para que, al final, salga el resultado correcto. Después de cada intento, alguien te dice "te equivocaste por esto" (el error). **Backpropagation es el método para averiguar, de todos tus diales, cuál tuvo más culpa del error y cuánto hay que girarlo — y en qué dirección.**

### Paso 1: la red hace una predicción (forward pass)

La información entra por un lado y atraviesa capas, una tras otra, como una cadena de montaje:

```
Entrada → [Capa 1] → [Capa 2] → [Capa 3] → Predicción
```

Cada capa transforma un poco la información y se la pasa a la siguiente. Al final sale una predicción.

### Paso 2: comparas con la respuesta correcta

```
Predicción: "perro"     Respuesta real: "gato"     → ERROR
```

Ese error es un número: qué tan lejos estuvo la predicción de la realidad.

### Paso 3: repartir la culpa hacia atrás (esto es backpropagation)

> **El error no se generó todo en la última capa. Se fue "acumulando" a lo largo de toda la cadena de capas. Backpropagation reparte ese error hacia atrás, capa por capa, preguntando en cada una: "¿cuánto contribuiste tú a este error?"**

Piénsalo como una cadena de responsables en una fábrica:

```
Materia prima → Obrero 1 → Obrero 2 → Obrero 3 → Producto defectuoso
```

Si el producto sale mal, no le echas la culpa solo al último obrero. Vas hacia atrás preguntando: "Obrero 3, ¿qué hiciste con lo que te dio el Obrero 2? ¿Tu paso empeoró las cosas?" y luego "Obrero 2, ¿qué hiciste con lo que te dio el Obrero 1?" — así hasta llegar al principio de la cadena.

Backpropagation hace exactamente eso, pero matemáticamente: empieza en la salida (donde está el error) y va "preguntando" hacia atrás, capa por capa, cuánto contribuyó cada peso al error final.

### Paso 4: cada peso se ajusta según su "culpa"

Una vez que sabes cuánto contribuyó cada peso al error, lo mueves un poquito **en la dirección que reduce ese error**. No mueves todos los pesos igual — los que tuvieron mucha culpa se ajustan más, los que tuvieron poca culpa casi no se tocan.

```
Peso que causó mucho error   →  se ajusta MUCHO
Peso que casi no influyó     →  se ajusta POCO o nada
```

### Paso 5: repetir muchísimas veces

Con un solo ajuste no basta. Le enseñas miles/millones de ejemplos a la red, y cada vez repite: predecir → medir error → repartir culpa hacia atrás → ajustar pesos un poquito. Con el tiempo, los pesos se van acomodando hasta que la red predice bien.

### ¿Por qué "hacia atrás" y no "hacia adelante"?

Porque para saber "cuánto contribuyó el peso de la capa 1 al error", **necesitas saber primero cómo esa contribución se transformó al pasar por las capas 2 y 3** antes de llegar a la salida. No puedes saber la culpa de la capa 1 sin antes saber la culpa que "heredó" de las capas que vienen después. Por eso el cálculo tiene que empezar en la salida y viajar hacia atrás, capa por capa — nunca al revés.

### Conectando con las arquitecturas ya vistas

- En un **MLP** normal: la cadena de "obreros" son las capas, y el error viaja hacia atrás una vez por cada capa.
- En una **RNN**: la cadena de "obreros" son los **pasos en el tiempo** (palabra 1, palabra 2, palabra 3...). Si la frase es larga, la cadena es larguísima, y el error se va "diluyendo" al viajar por tantos pasos — ese es justo el problema del vanishing gradient (sección 4): la culpa que le llega al primer paso es casi cero, aunque en realidad sí tuvo responsabilidad.
- En **LSTM/ResNet/Transformer**: se añaden "atajos" en la cadena (conexiones que saltan directamente hacia atrás sin pasar por todos los intermedios), para que la culpa pueda viajar más lejos sin perderse por el camino.
- En **atención (Q, K, V)**: cuando la cadena se bifurca en tres caminos distintos (uno para Q, otro para K, otro para V), backpropagation reparte la culpa por cada una de las tres ramas por separado, según cuánto contribuyó cada una — como si el Obrero 2 en realidad hubiera dado su trabajo a tres personas distintas, y cada una recibe su parte de culpa según lo que realmente hizo.

### En una sola frase, sin matemáticas

> **Backpropagation es el proceso de, tras cada predicción, repartir el error de vuelta hacia atrás por toda la cadena de capas para averiguar cuánto contribuyó cada peso, y así saber cuánto y en qué dirección ajustar cada uno para que la próxima predicción sea mejor.**

---

## 12.B Versión técnica (con matemáticas)

### 12.B.1 La derivada (una variable)

$$f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

Es la tasa de cambio: cuánto cambia $f(x)$ si mueves $x$ un poquito. El signo de $f'(x)$ basta para optimizar: mueve $x$ en la dirección contraria al gradiente y $f$ baja.

### 12.B.2 Derivada parcial y gradiente

Con varias variables, $\frac{\partial f}{\partial x}$ es "cuánto cambia $f$ moviendo solo $x$". El **gradiente** $\nabla f$ es el vector de todas esas derivadas parciales. En una red, $f$ = loss y las variables son todos los pesos: el gradiente indica, para cada peso, en qué dirección moverlo para bajar el loss.

### 12.B.3 La regla de la cadena

Para una función compuesta $y = f(g(x))$:

$$\frac{dy}{dx} = \frac{dy}{dg}\cdot\frac{dg}{dx}$$

Con más eslabones, $y = f_3(f_2(f_1(x)))$:

$$\frac{dy}{dx} = \frac{dy}{df_3}\cdot\frac{df_3}{df_2}\cdot\frac{df_2}{df_1}\cdot\frac{df_1}{dx}$$

**Una red neuronal es una composición de funciones** (capa 1, capa 2, capa 3...), así que calcular el efecto de un peso de la capa 1 sobre el loss final es aplicar esta regla a través de todas las capas intermedias.

### 12.B.4 El grafo de cómputo

Cualquier red se dibuja como un grafo dirigido: nodos = operaciones, aristas = flujo de datos.

```python
z = w * x + b
a = sigmoid(z)
loss = (a - y_true)**2
```
```
x ──┐
    ×── z ── sigmoid ── a ── (a-y)² ── loss
w ──┘
b ────────────┘
```

Cada nodo sabe derivarse a sí mismo localmente. Backprop compone esas derivadas locales con la regla de la cadena, sin necesitar cálculo global.

### 12.B.5 Backpropagation = chain rule aplicada al grafo, de atrás hacia adelante

**Forward pass**: se ejecuta el grafo de izquierda a derecha, guardando resultados intermedios.
**Backward pass**: se recorre de derecha a izquierda, calculando "cuánto cambia el loss" respecto a cada nodo, reutilizando lo ya calculado (de ahí su eficiencia).

```python
d_loss_d_a = 2*(a - y_true)      # derivada local de (a-y)^2
d_a_d_z    = a * (1 - a)         # derivada local de sigmoid
d_z_d_w    = x                   # derivada local de w*x+b respecto a w
d_z_d_b    = 1

d_loss_d_z = d_loss_d_a * d_a_d_z              # regla de la cadena
d_loss_d_w = d_loss_d_z * d_z_d_w
d_loss_d_b = d_loss_d_z * d_z_d_b
```

Cada línea es "gradiente acumulado hasta aquí × derivada local de este paso". Ese es, sin más, todo el algoritmo.

### 12.B.6 Ejemplo numérico completo

```python
x, w, b, y_true = 2.0, 0.5, 0.1, 1.0

# FORWARD
z = w*x + b                 # 1.1
a = 1/(1+math.exp(-z))      # sigmoid(1.1) ≈ 0.7503
loss = (a - y_true)**2      # ≈ 0.0624

# BACKWARD
d_loss_d_a = 2*(a - y_true)          # ≈ -0.4995
d_a_d_z    = a*(1-a)                 # ≈ 0.1873
d_loss_d_z = d_loss_d_a * d_a_d_z    # ≈ -0.0936
d_loss_d_w = d_loss_d_z * x          # ≈ -0.1871
d_loss_d_b = d_loss_d_z * 1          # ≈ -0.0936

# UPDATE (descenso de gradiente)
lr = 0.1
w = w - lr * d_loss_d_w
b = b - lr * d_loss_d_b
```

Esto es exactamente lo que hace `loss.backward()` en PyTorch, automatizado y escalado a millones de parámetros.

### 12.B.7 Generalización a matrices

Con matrices ($X\cdot W_Q = Q$), la derivada es una matriz Jacobiana, pero la regla se mantiene — derivada acumulada × derivada local, con las transposiciones correctas:

$$Q = X\cdot W_Q \;\Rightarrow\; \frac{\partial L}{\partial W_Q} = X^T\cdot \frac{\partial L}{\partial Q} \qquad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Q}\cdot W_Q^T$$

Los frameworks de autodiferenciación (autograd de PyTorch/TF) construyen el grafo automáticamente en el forward pass y aplican esta regla nodo por nodo en el backward — nadie deriva esto a mano en la práctica.

### 12.B.8 Aplicado a Q, K, V de la atención

```
loss
 │ (chain rule)
salida = softmax(QKᵀ/√d)·V
 │                    │
 ├─ gradiente → V → W_V     (una rama)
 │
 └─ gradiente → QKᵀ ─┬─ gradiente → Q → W_Q   (otra rama)
                      └─ gradiente → K → W_K   (otra rama)
```

Un único backward pass; la regla de la cadena se **ramifica** en el punto donde Q, K y V se separan (todas parten de la misma $X$), y cada rama acumula su propio gradiente hasta llegar a su matriz de pesos.

### 12.B.9 Por qué importa en la práctica

- **Vanishing/exploding gradient** (sección 4): producto de muchas derivadas locales en cadenas largas (RNN en el tiempo, redes muy profundas) — si esas derivadas son consistentemente <1 o >1, el producto tiende a 0 o a infinito.
- **ResNet, LSTM, Transformer** mitigan esto dando "atajos" (sumas en vez de solo multiplicaciones) para que el gradiente tenga un camino corto y directo hacia atrás.
- **Autograd** construye el grafo dinámicamente en cada forward pass y `.backward()` lo recorre aplicando este mismo algoritmo para millones de parámetros en paralelo, en GPU.

---

## 13. Anexo: Eje cronológico completo del Deep Learning (1943 – 2026/27)

Esta es una cronología más amplia que la tabla de la sección 8 (que se centraba solo en el camino MLP→Transformer). Aquí se incluyen también visión por computador, aprendizaje por refuerzo, modelos generativos y los hitos más recientes, hasta la actualidad (agosto de 2026). Los años 2026-2027 reflejan el estado del campo en el momento de escribir este documento; dado el ritmo actual, es esperable que sigan apareciendo modelos nuevos con frecuencia mensual.

| Año | Hito | Qué aportó |
|---|---|---|
| 1943 | Neurona de McCulloch-Pitts | Primer modelo matemático de una neurona artificial (sin aprendizaje aún). |
| 1958 | Perceptrón (Rosenblatt) | Primera neurona *entrenable*; clasificador lineal. |
| 1969 | *Perceptrons* (Minsky & Papert) | Muestra el límite del perceptrón simple (XOR) → primer "invierno de la IA". |
| 1986 | Backpropagation (Rumelhart, Hinton, Williams) | Permite entrenar redes con capas ocultas (MLP). Revive el campo. |
| 1989-1998 | LeNet (LeCun) | Primeras CNN aplicadas a reconocimiento de dígitos escritos a mano. |
| 1997 | LSTM (Hochreiter & Schmidhuber) | Resuelve (en gran medida) el vanishing gradient en RNN. |
| 2006 | *Deep Belief Networks* (Hinton et al.) | Pre-entrenamiento no supervisado capa a capa; acuña el término "deep learning" moderno. |
| 2009 | ImageNet (Fei-Fei Li et al.) | Dataset masivo y etiquetado de imágenes; base de la competición que impulsaría las CNN. |
| 2012 | AlexNet (Krizhevsky, Sutskever, Hinton) | Gana ImageNet por un margen enorme usando GPUs + ReLU → explosión del deep learning moderno. |
| 2013 | Word2Vec (Mikolov et al., Google) | Embeddings de palabras aprendidos eficientemente; base del NLP moderno. |
| 2014 | GAN — Generative Adversarial Networks (Goodfellow et al.) | Dos redes compitiendo (generador vs discriminador); nace la generación de imágenes moderna. |
| 2014 | Seq2Seq (Sutskever et al.) + Atención (Bahdanau et al.) | Traducción automática con encoder-decoder; nace el mecanismo de atención. |
| 2014 | VGG / GoogLeNet | CNN más profundas y eficientes para visión por computador. |
| 2015 | ResNet (He et al.) | Conexiones residuales; permite entrenar redes de cientos de capas. |
| 2015 | Batch Normalization (Ioffe & Szegedy) | Estabiliza y acelera el entrenamiento de redes profundas. |
| 2016 | AlphaGo (DeepMind) | Vence al campeón mundial de Go; combina deep learning + búsqueda en árbol + refuerzo. |
| 2017 | AlphaGo Zero / AlphaZero | Aprende Go, ajedrez y shogi solo jugando contra sí mismo, sin datos humanos. |
| **2017** | **Attention Is All You Need (Vaswani et al.)** | **Nace el Transformer: elimina la recurrencia, solo atención.** |
| 2018 | BERT (Google) | Transformer *encoder-only*; pre-entrenamiento bidireccional, cambia el NLP aplicado. |
| 2018 | GPT-1 (OpenAI) | Transformer *decoder-only* autoregresivo; primer paso hacia la familia GPT. |
| 2018 | AlphaFold 1 (DeepMind) | Primeros pasos serios de deep learning en predicción de estructura de proteínas. |
| 2019 | GPT-2 (OpenAI) | Escalado del decoder-only; genera texto coherente a mayor escala (1.5B parámetros). |
| 2020 | GPT-3 (OpenAI) | 175B parámetros; demuestra *few-shot learning* casi sin ajuste fino. |
| 2020-2021 | AlphaFold 2 (DeepMind) | Resuelve el problema de 50 años del plegamiento de proteínas con precisión casi experimental. |
| 2020 | Vision Transformer / ViT (Google) | Lleva la arquitectura Transformer (antes solo NLP) a la visión por computador. |
| 2020 | Diffusion Models (DDPM, Ho et al.) | Formalización moderna de los modelos de difusión, base de la generación de imágenes actual. |
| 2021 | CLIP / DALL-E (OpenAI) | Conecta texto e imagen en un mismo espacio; generación de imágenes a partir de texto. |
| 2022 | Chinchilla (DeepMind) | Redefine las "leyes de escalado": más datos, no solo más parámetros, es clave. |
| 2022 | Stable Diffusion (Stability AI / CompVis) | Modelo de difusión texto-a-imagen open-source; democratiza la generación de imágenes. |
| **2022** | **ChatGPT (OpenAI, nov. 2022)** | Basado en GPT-3.5 + RLHF; 100M de usuarios en 2 meses, el momento de inflexión pública de la IA generativa. |
| 2023 | GPT-4 (OpenAI) | Multimodal (texto + imagen), salto grande en razonamiento. |
| 2023 | Llama / Llama 2 (Meta) | Modelos abiertos de gran escala; impulsan todo un ecosistema open-source. |
| 2023 | Claude 1/2 (Anthropic), Gemini/Bard (Google) | Se consolida la carrera de varios laboratorios compitiendo con modelos "frontera". |
| 2024 | GPT-4o, Claude 3/3.5, Gemini 1.5, Llama 3 | Modelos multimodales en tiempo real; ventanas de contexto muy largas. |
| 2024 | OpenAI o1 | Primer modelo "de razonamiento" con *chain-of-thought* aprendido explícitamente antes de responder. |
| 2024 | AlphaFold 3 (DeepMind + Isomorphic Labs) | Extiende la predicción de estructuras a complejos proteína-ADN-ARN-ligandos. |
| Ene. 2025 | DeepSeek R1 (DeepSeek, China) | Modelo de razonamiento open-source que iguala a o1 con un coste de entrenamiento drásticamente menor; sacude la narrativa de "solo los gigantes pueden competir". |
| 2025 | OpenAI o3, Gemini Deep Think, Claude 3.7/4 | Modelos de razonamiento más potentes; Gemini Deep Think logra medalla de oro en el IMO 2025. |
| Abr. 2025 | Llama 4 (Meta) | Variantes "Scout" y "Maverick", multimodales, contexto extendido. |
| 2025 | Qwen 3 (Alibaba), Mistral 3, Grok 3 (xAI) | Consolidación de laboratorios chinos y europeos como competidores de primer nivel. |
| 2025 | Agentes de IA autónomos | Modelos capaces de ejecutar tareas de varios pasos usando herramientas (navegar, programar, operar software) sin supervisión constante. |
| Jun. 2026 | Claude Fable 5 / Claude Mythos 5 (Anthropic) | Primeros modelos de la nueva categoría "Mythos", por encima de Opus; acceso restringido inicialmente por controles de exportación de EE. UU., restaurado el 1 de julio de 2026. |
| 2026 | Claude Sonnet 5, Claude Opus 4.8, Claude Haiku 4.5 (Anthropic); GPT-5.x (OpenAI) | Generación actual de modelos "frontera" en el momento de escribir este documento. |
| 2026-2027 (en curso) | Modelos de razonamiento + agentes + multimodalidad convergiendo | La tendencia dominante: modelos que razonan paso a paso, usan herramientas de forma autónoma, y procesan texto/imagen/audio/vídeo de forma unificada. Nuevos lanzamientos aparecen con frecuencia mensual entre los laboratorios principales (OpenAI, Anthropic, Google DeepMind, Meta, xAI, DeepSeek, Alibaba, Mistral). |

**Nota sobre los años 2026-2027:** a diferencia del resto de la cronología (hitos ya asentados y bien documentados), esta franja describe el estado del campo en el momento de escribir el documento (agosto de 2026) y la tendencia inmediata hacia adelante. Conviene verificar con fuentes actualizadas (páginas oficiales de los laboratorios, o una búsqueda web) para conocer los lanzamientos más recientes, dado el ritmo de publicación actual.

---

### Nota final
Este documento es un mapa de navegación, no un sustituto de leer los papers originales ni de programar las arquitecturas. La secuencia recomendada es: **intuición visual (3Blue1Brown/StatQuest) → artículo ilustrado (Alammar/Olah) → paper original → implementación propia (Karpathy)**. Ese orden minimiza la frustración y maximiza la retención.
