---
title: "La Biblia de las Redes Neuronales"
tags: [archivo]
status: borrador
updated: 2026-08-08
---

# La Biblia de las Redes Neuronales
### Del Perceptrón al Transformer: MLP → CNN → RNN → Atención → *Attention Is All You Need*

> Documento de referencia pensado para leerse de principio a fin como una narrativa histórica y técnica. Cada bloque incluye: **la idea clave**, **por qué apareció**, **qué problema resolvía / creaba**, y **recursos para profundizar** (papers, vídeos, artículos).

---

## 0. Mapa mental del recorrido

```
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

**Recursos:**
- 📄 Paper: Rumelhart, Hinton, Williams (1986), *Learning representations by back-propagating errors* — [Nature, PDF (Univ. Toronto)](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
- 📄 Minsky & Papert, *Perceptrons* (1969) — contexto histórico del primer invierno de la IA.
- 🎥 3Blue1Brown — *Neural Networks* (serie completa, la mejor introducción visual que existe): [playlist en YouTube](https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s)
  - **Cap. 1 — ¿Qué es una red neuronal?**: usa el reconocimiento de dígitos escritos a mano (MNIST) para explicar qué son neuronas, capas, pesos y sesgos, y por qué una red se puede pensar como una función que va ajustando esos parámetros.
  - **Cap. 2 — Descenso de gradiente**: explica cómo aprende la red — define una función de coste, muestra por qué minimizarla equivale a "bajar una colina" en un espacio de muchísimas dimensiones, e introduce la idea de backpropagation como el método eficiente para calcular esa dirección de bajada.
  - **Cap. 3 y 4 — Backpropagation (intuición + cálculo)**: primero da la intuición de cómo el error de la salida se reparte hacia atrás entre las neuronas anteriores según su responsabilidad; después formaliza esa intuición con la regla de la cadena, capa por capa, hasta llegar a las fórmulas reales que se usan para entrenar.
- 🎥 StatQuest — *Neural Networks Pt. 1-3* (main ideas, backprop, ReLU) — canal: [StatQuest with Josh Starmer](https://statquest.org/video_index.html)
  - **Resumen:** Josh Starmer construye una red pequeñísima a mano para mostrar, paso a paso y sin código, cómo se combinan las entradas, cómo actúan funciones de activación como la sigmoide o la ReLU, y cómo el algoritmo de backpropagation ajusta los pesos comparando la predicción con el valor real. Es el complemento perfecto a 3Blue1Brown: menos animación, más "hazlo con lápiz y papel".
- 🌐 Michael Nielsen — [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) (libro online gratuito, muy pedagógico, con el ejemplo de reconocimiento de dígitos MNIST paso a paso)

---

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
- **ResNet** (He et al., 2015): introduce las **conexiones residuales** (*skip connections*), que permiten entrenar redes de cientos de capas sin que el gradiente se pierda por el camino — una solución **arquitectónica** al mismo problema de fondo que veremos en la sección 4 (el gradiente que se degrada al atravesar muchas capas).

**Por qué importa para el resto de la historia:** ResNet es el primer gran ejemplo de "si el gradiente no fluye bien a través de muchas capas, cambia la arquitectura para darle una autopista directa". Esa misma filosofía reaparecerá en LSTM y, más adelante, en las conexiones residuales que usa el propio Transformer.

**Recursos:**
- 📄 LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* — [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- 📄 Krizhevsky, Sutskever, Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet) — [PDF (NeurIPS)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- 📄 Simonyan & Zisserman (2014), *Very Deep Convolutional Networks* (VGG) — [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- 📄 He, Zhang, Ren, Sun (2015), *Deep Residual Learning for Image Recognition* (ResNet) — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- 🎥 StatQuest — vídeo y notebook de *Convolutional Neural Networks, Clearly Explained*.
  - **Resumen:** parte de un MLP normal aplicado a una imagen para mostrar por qué explota en número de parámetros, y a partir de ahí introduce el filtro convolucional como un "escáner" que se desliza por la imagen buscando el mismo patrón (un borde, una textura) en cualquier posición. Cubre también el *max pooling* (reducir el tamaño manteniendo lo más relevante) y cómo se apilan varias capas convolucionales antes de aplanar y conectar con capas densas para la clasificación final.
- 🌐 CS231n (Stanford) — [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) — el curso de referencia mundial, notas + vídeos.

---

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

**Recursos:**
- 🎥 StatQuest — [*Recurrent Neural Networks (RNNs), Clearly Explained!!!*](https://www.youtube.com/watch?v=AsNTP8Kwu80)
  - **Resumen:** parte de un problema muy concreto (predecir un valor cuando cada ejemplo tiene una cantidad distinta de datos de entrada, como días de cotización de una acción) para mostrar por qué un MLP normal no sirve, al necesitar un tamaño de entrada fijo. A partir de ahí construye la RNN "desenrollando" el mismo bloque de pesos en cada paso temporal, y muestra visualmente cómo la salida de un paso se retroalimenta al siguiente para dar a la red una forma de "memoria".
- 🎥 Andrej Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks* (entrada de blog clásica, con ejemplos de generación de texto carácter a carácter) — [char-rnn blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - **Resumen:** demuestra, con una RNN entrenada carácter a carácter, que un modelo tan simple puede generar desde código en C que casi compila hasta texto con la estructura de una obra de Shakespeare, solo aprendiendo estadísticas del texto. El argumento central del artículo es que darle a una red la capacidad de mantener un estado interno a lo largo de una secuencia es sorprendentemente potente, incluso con arquitecturas sencillas — la motivación perfecta antes de entender por qué luego hicieron falta LSTM y atención para ir más lejos.
- 📄 Elman (1990), *Finding Structure in Time* — uno de los papers fundacionales de las RNN modernas.

---

## 4. El problema del desvanecimiento (y explosión) del gradiente

Esta es la sección **bisagra** de todo el documento: es el problema técnico concreto que va a motivar LSTM, luego la atención, y en última instancia el propio Transformer.

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

**Recursos:**
- 📄 Hochreiter (1991), tesis de diploma (en alemán) — el primer análisis riguroso del problema.
- 📄 Bengio, Simard, Frasconi (1994), *Learning Long-Term Dependencies with Gradient Descent is Difficult* — IEEE Transactions on Neural Networks. El paper en inglés más citado sobre el tema.
- 📄 Pascanu, Mikolov, Bengio (2013), *On the difficulty of training Recurrent Neural Networks* — [arXiv:1211.5063](https://arxiv.org/abs/1211.5063) — formaliza el *gradient clipping*.
- 🎥 3Blue1Brown / StatQuest mencionan el problema como motivación antes de explicar LSTM (ver recursos de la sección 5).

---

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

**Recursos:**
- 📄 Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — [PDF (bioinf.jku.at)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 📄 Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (introduce GRU) — [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- 🎥 StatQuest — [*Long Short-Term Memory (LSTM), Clearly Explained*](https://www.youtube.com/watch?v=YCzL96nL7j0) (21 min, muy recomendable, con doblaje disponible en español)
  - **Resumen:** explica el LSTM como tres etapas consecutivas dentro de cada celda: (1) decidir qué porcentaje de la memoria de largo plazo se olvida (*forget gate*), (2) actualizar esa memoria de largo plazo combinando lo que se recuerda con la nueva información de entrada (*input gate*), y (3) generar, a partir de esa memoria actualizada, la memoria de corto plazo que se pasa al siguiente paso (*output gate*). Termina mostrando el LSTM "en acción" prediciendo valores reales, y deja claro por qué esta separación en dos memorias evita el problema del vanishing/exploding gradient de una RNN simple.
- 🌐 Christopher Olah — [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — el artículo ilustrado más famoso sobre LSTM, con los diagramas que todo el mundo termina copiando en sus apuntes.

---

## 6. Seq2Seq y el nacimiento de la Atención

### 6.1 El modelo Encoder-Decoder (Sutskever, Vinyals, Le — 2014)
Para tareas como traducción automática (secuencia de entrada → secuencia de salida, de longitudes distintas) se propone **Sequence to Sequence (Seq2Seq)**:
- Un **encoder** (una RNN/LSTM) lee toda la frase de entrada y la "comprime" en **un único vector de contexto** de tamaño fijo (el último estado oculto).
- Un **decoder** (otra RNN/LSTM) toma ese vector y genera la frase de salida, palabra a palabra.

### 6.2 El cuello de botella
El problema es evidente: **toda una frase, sin importar si tiene 5 o 50 palabras, se aprieta en un único vector de tamaño fijo**. Es como pedirle a alguien que lea un párrafo entero y luego resuma todo su significado en una sola palabra antes de empezar a traducir. Cuanto más larga la frase, peor funciona el modelo — justo el mismo síntoma del vanishing gradient, visto ahora desde el ángulo de la arquitectura, no solo del entrenamiento.

### 6.3 La solución: Atención (Bahdanau et al., 2014)
La idea, presentada por Bahdanau, Cho y Bengio, es simple pero transformó el campo: **en vez de forzar al decoder a usar un solo vector de contexto fijo, dejemos que en cada paso de la generación "mire hacia atrás" y decida a qué palabras de la entrada prestar más atención.**

Mecánicamente:
1. El encoder ya no produce un solo vector final, sino un estado oculto **por cada palabra** de entrada.
2. En cada paso del decoder, se calcula una puntuación de "compatibilidad" entre el estado actual del decoder y cada uno de esos estados del encoder.
3. Esas puntuaciones se normalizan con un `softmax` → son los **pesos de atención**.
4. El vector de contexto para ese paso es la **suma ponderada** de los estados del encoder según esos pesos.

Esto es, literalmente, un mecanismo que aprende **a qué prestar atención**, dando nombre al concepto. Poco después, Luong et al. (2015) simplificaron y generalizaron la idea (atención "global" vs "local", y variantes multiplicativas más eficientes de calcular).

**Esta es la pieza que, tres años más tarde, los autores de "Attention Is All You Need" se preguntarán: ¿y si la atención no es solo un complemento de la RNN, sino todo lo que necesitamos?**

**Recursos:**
- 📄 Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning with Neural Networks* — [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- 📄 Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) — **el paper que introduce la atención**.
- 📄 Luong, Pham, Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation* — [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- 🌐 Jay Alammar — [*Visualizing A Neural Machine Translation Model (seq2seq with attention)*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — el "precuela" ilustrado de su famoso post sobre el Transformer.
  - **Resumen:** anima paso a paso un modelo seq2seq clásico y luego el mismo modelo con atención, mostrando visualmente cómo cambia el "foco" del decoder sobre las distintas palabras de la entrada en cada paso de la traducción. Es la mejor forma de ver de un vistazo por qué la atención soluciona el cuello de botella del vector de contexto único.

---

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

**Recursos:**
- 📄 **El paper original**: Vaswani et al. (2017), *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — lectura obligatoria, aunque conviene leerlo *después* de ver alguna explicación ilustrada.
- 🌐 **Jay Alammar** — [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — probablemente el recurso más citado del mundo para entender el Transformer visualmente (referenciado en cursos de MIT, Stanford, Harvard, etc.).
- 🎥 Yannic Kilcher — [*Attention Is All You Need* (paper explicado)](https://www.youtube.com/watch?v=iDulhoQ2pro) — repaso técnico del paper, ~27 min.
  - **Resumen:** hace una lectura guiada del paper original: parte de cómo se procesaba el lenguaje antes (RNN/CNN), explica el problema de las dependencias de largo alcance, y luego diseca el mecanismo de atención del paper, el positional encoding y cómo se calculan los pesos de atención sobre la matriz de la frase. Cierra comentando el impacto del paper (mejores resultados en traducción con menos tiempo de entrenamiento) y su generalización a otras tareas como el *parsing* de constituyentes en inglés.
- 🎥 3Blue1Brown — [*But what is a GPT? Visual intro to Transformers*](https://www.youtube.com/watch?v=wjZofJX0v4M) (Capítulo 5 de la serie *Deep Learning*)
  - **Resumen:** explica qué significa cada palabra de "Generative Pre-trained Transformer": cómo el texto se convierte en tokens y embeddings, cómo esos vectores se van refinando capa a capa para incorporar contexto, y cómo al final la red produce una distribución de probabilidad sobre el siguiente token (con conceptos como *temperature* para controlar la aleatoriedad de la generación). Es la introducción visual a la arquitectura completa antes de entrar en el detalle de la atención en el capítulo 6.
- 🎥 3Blue1Brown — [*Attention in transformers, visually explained*](https://www.youtube.com/watch?v=eMlx5fFNoYc) (Capítulo 6 de la serie *Deep Learning*)
  - **Resumen:** el capítulo dedicado por completo al mecanismo de atención. Muestra cómo cada palabra empieza con un embedding "genérico" y cómo, a través de las matrices de Query, Key y Value, ese embedding se va actualizando para incorporar el contexto de las palabras cercanas (el ejemplo clásico: la palabra "banco" cambia de significado según si hay cerca "río" o "dinero"). Cubre también el enmascarado (*masking*) para que un token no pueda "mirar" al futuro, el uso de múltiples cabezas de atención en paralelo, y cómo se combinan sus resultados en la matriz de salida.
- 🎥 Andrej Karpathy — [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m)
  - **Resumen:** en la primera mitad construye un modelo de lenguaje muy simple (bigramas) y a partir de ahí introduce la atención como una forma de "paso de mensajes" entre nodos de un grafo dirigido; en la segunda mitad completa la arquitectura del Transformer añadiendo atención multi-cabeza, una red feed-forward, conexiones residuales y *layer norm*, siguiendo de cerca el paper original. Termina entrenando el modelo resultante (nanoGPT) sobre texto de Shakespeare y comparándolo, en tamaño, con GPT-3 y ChatGPT.
- 🎥 StatQuest — serie sobre *Transformer Neural Networks*, *Decoder-Only Transformers* y *Coding a ChatGPT-like Transformer from scratch* (con notebooks gratuitos) — [statquest.org/video_index.html](https://statquest.org/video_index.html)
- 🎥 Andrej Karpathy — [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — construye un GPT (decoder-only Transformer) línea a línea; el mejor recurso para pasar de la teoría a código real.
- 📄 Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers* — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- 📄 Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1) — [PDF (OpenAI)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

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

## 9. Ruta de estudio sugerida (checklist)

- [ ] Ver la serie de **3Blue1Brown** completa (caps. 1-4 de Neural Networks) para intuición de MLP + backprop.
- [ ] Leer el libro de **Michael Nielsen** en paralelo, deteniéndose en el capítulo de backpropagation.
- [ ] Ver **StatQuest: CNN** → leer el paper de AlexNet en diagonal (secciones de arquitectura e ImageNet).
- [ ] Ver **StatQuest: RNN** → entender por qué se "desenrolla" en el tiempo.
- [ ] Leer/entender el problema del **vanishing gradient** (Pascanu et al. 2013 es el más accesible).
- [ ] Ver **StatQuest: LSTM** + leer *Understanding LSTM Networks* de Colah.
- [ ] Leer el paper de **Bahdanau (atención)** — es corto y muy legible.
- [ ] Leer **The Illustrated Transformer** de Jay Alammar de principio a fin.
- [ ] Ver el vídeo de **Yannic Kilcher** sobre *Attention Is All You Need*.
- [ ] Leer el **paper original** de Vaswani et al.
- [ ] (Opcional, para pasar a código) Ver **"Let's build GPT" de Andrej Karpathy** y programarlo a la vez.

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

### Nota final
Este documento es un mapa de navegación, no un sustituto de leer los papers originales ni de programar las arquitecturas. La secuencia recomendada es: **intuición visual (3Blue1Brown/StatQuest) → artículo ilustrado (Alammar/Olah) → paper original → implementación propia (Karpathy)**. Ese orden minimiza la frustración y maximiza la retención.
