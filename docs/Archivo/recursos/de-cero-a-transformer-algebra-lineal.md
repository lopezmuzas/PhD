---
title: "De cero a Transformer: aprender LLMs desde el Álgebra Lineal"
tags: [recursos]
status: borrador
updated: 2026-08-08
---

# De cero a Transformer: aprender LLMs desde el Álgebra Lineal

Guía de recursos (mayoritariamente visuales) ordenada como una ruta de aprendizaje progresiva: **álgebra lineal → cálculo/gradientes → redes neuronales y backpropagation → transformers y LLMs**. Todos los recursos son gratuitos.

---

## 0. Cómo usar esta guía

El orden importa más que la cantidad. Sugerencia de ritmo:

1. Semana 1-2: Álgebra lineal esencial (bloque 1)
2. Semana 3: Cálculo y gradientes (bloque 2)
3. Semana 4-5: Redes neuronales y backpropagation, con código (bloque 3)
4. Semana 6-8: Atención y Transformers (bloque 4)
5. En paralelo: herramientas interactivas (bloque 5) para "tocar" los números

No hace falta dominar cada bloque al 100% antes de avanzar — la serie de Karpathy en particular obliga a repasar cálculo y álgebra sobre la marcha, lo cual refuerza el aprendizaje.

---

## 1. Álgebra lineal — las bases geométricas

### 🎥 3Blue1Brown — *Essence of Linear Algebra* (playlist completa)
La mejor introducción visual que existe a vectores, transformaciones lineales, matrices, determinantes, espacios de columnas/nulos, autovectores/autovalores y cambio de base. Todo se explica con animaciones geométricas, exactamente el "lenguaje visual" que luego usarás para entender embeddings y atención.
- Playlist: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- Web con todos los capítulos y notas: https://www.3blue1brown.com/topics/linear-algebra

Capítulos clave para LLMs (si vas con prisa, prioriza estos):
- Vectores, qué son realmente
- Transformaciones lineales y matrices
- Multiplicación de matrices como composición de transformaciones
- Producto punto (dot product) y su intuición geométrica — **esto es literalmente el corazón de la atención**
- Cambio de base
- Autovectores y autovalores (útil para intuir PCA/representaciones, menos crítico para backprop en sí)

### 🎥 MIT 18.06 — Linear Algebra, Gilbert Strang (OpenCourseWare)
Más formal que 3Blue1Brown pero el "clásico" de referencia mundial. Ideal como complemento cuando quieras profundizar en la teoría (espacios vectoriales, descomposición SVD, etc.).
- Curso completo (vídeos + notas + ejercicios): https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- Vídeos en YouTube: buscar "MIT 18.06 Linear Algebra Gilbert Strang"

### 🖱️ Immersive Linear Algebra (libro interactivo online)
Un "libro de texto" interactivo en el navegador: cada gráfico 3D se puede rotar y manipular en tiempo real mientras lees la teoría. Perfecto para complementar 3Blue1Brown con manipulación directa.
- http://immersivemath.com/ila/index.html

### 📖 Matrix Calculus for Deep Learning (paper/guía visual)
Cuando llegues a backpropagation necesitarás derivar respecto a matrices y vectores, no solo escalares. Esta guía (Parr & Howard) explica notación y reglas del cálculo matricial usado en deep learning con muchos ejemplos paso a paso.
- https://explained.ai/matrix-calculus/

---

## 2. Cálculo y gradientes — la base de backpropagation

### 🎥 3Blue1Brown — *Essence of Calculus*
Repaso visual de derivadas, regla de la cadena y el concepto de gradiente como vector de máxima pendiente — justo lo que necesitas antes de ver backprop.
- Playlist: https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr

### 🎥 3Blue1Brown — *Gradient descent, how neural networks learn* (Deep Learning, Cap. 2)
Explica visualmente qué es el descenso de gradiente y por qué "aprender" es minimizar una función de coste moviéndose en la dirección opuesta al gradiente.
- https://www.youtube.com/watch?v=IHZwWFHWa-w

---

## 3. Redes neuronales y Backpropagation

### 🎥 3Blue1Brown — Serie *Neural Networks* (capítulos 1 a 4)
La introducción visual definitiva antes de meterte en código:
1. **But what is a neural network?** — neuronas, capas, pesos y sesgos como multiplicación de matrices: https://www.youtube.com/watch?v=aircAruvnKk
2. **Gradient descent, how neural networks learn**: https://www.youtube.com/watch?v=IHZwWFHWa-w
3. **What is backpropagation really doing?** — la intuición visual de cómo se propaga el error hacia atrás por la red: https://www.youtube.com/watch?v=Ilg3gGewQ5U
4. **Backpropagation calculus** — la mecánica formal con la regla de la cadena aplicada capa a capa: https://www.youtube.com/watch?v=tIeHLnjs5U8

Web con la serie completa y ejercicios interactivos: https://www.3blue1brown.com/topics/neural-networks

### 🎥 Andrej Karpathy — *Neural Networks: Zero to Hero* (la serie más recomendada para pasar de teoría a código real)
Curso gratuito en YouTube donde Karpathy (ex-Director de IA en Tesla, cofundador de OpenAI) construye backpropagation y un Transformer **desde cero, línea a línea de código**, sin usar `autograd` de PyTorch hasta que entiendes cómo funciona por dentro.
- Página del curso con todos los vídeos y notebooks: https://karpathy.ai/zero-to-hero.html
- Repositorio de código: https://github.com/karpathy/nn-zero-to-hero

Orden recomendado de los vídeos:
1. **The spelled-out intro to neural networks and backpropagation: building micrograd** — construye un motor de autograd de 100 líneas; verás backpropagation al nivel de operaciones escalares individuales (sumas y multiplicaciones), con el grafo computacional dibujado a mano. Es, con diferencia, la explicación más "spelled out" (deletreada) que existe de backprop.
   - https://www.youtube.com/watch?v=VMj-3S1tku0
   - Repo: https://github.com/karpathy/micrograd
2. **The spelled-out intro to language modeling: building makemore** — de escalares a tensores con PyTorch, primer modelo de lenguaje (bigramas).
3. **Building makemore Part 2: MLP** — perceptrón multicapa, embeddings, entrenamiento.
4. **Building makemore Part 3: Activations & Gradients, BatchNorm** — por qué las redes profundas "explotan" o "desaparecen" en gradiente, y cómo se soluciona.
5. **Building makemore Part 4: Becoming a Backprop Ninja** — backprop manual a través de una red completa (cross-entropy, capas lineales, tanh, batchnorm), sin `.backward()`. Es el ejercicio definitivo para entender el álgebra matricial detrás del backprop real (no solo escalar).
6. **Building makemore Part 5: Building a WaveNet** — arquitecturas más profundas, antesala de los Transformers.
7. **Let's build GPT: from scratch, in code, spelled out** — el vídeo que conecta todo lo anterior con la arquitectura Transformer completa (ver bloque 4).

### 📖 CS231n (Stanford) — Notas de Backpropagation
Las notas escritas del curso de Stanford explican backprop con el formalismo de "grafos computacionales" y ejemplos con derivadas locales, muy complementario a los vídeos.
- https://cs231n.github.io/optimization-2/

### 🎥 StatQuest — *Neural Networks / Backpropagation* (Josh Starmer)
Si prefieres explicaciones más lentas y "para principiantes totales" con ejemplos numéricos muy sencillos antes de ver el enfoque matricial de Karpathy, StatQuest es un buen puente.
- Buscar en YouTube: "StatQuest Backpropagation"

---

## 4. Atención y Transformers — donde todo converge

### 🎥 3Blue1Brown — Serie *Deep Learning*, capítulos 5 y 6 (los que mencionas)
El núcleo visual de los LLM modernos:
- **Cap. 5 — "But what is a GPT? Visual intro to transformers"**: tokens, embeddings, el espacio de alta dimensión donde "viven" los vectores de palabras, y cómo fluye la información por los bloques del Transformer.
  - https://www.youtube.com/watch?v=wjZofJX0v4M
- **Cap. 6 — "Attention in transformers, visually explained"**: cómo la matriz de entrada se desdobla en Query/Key/Value, el producto escalar Q·K, la cuadrícula de Softmax y cómo esta "succiona" información de V para actualizar cada vector. Es la mejor animación disponible de la fórmula `softmax(QKᵀ/√d)V`.
  - https://www.youtube.com/watch?v=eMlx5fFNoYc
- Existe también un capítulo adicional sobre MLPs dentro del Transformer y cómo almacenan "hechos":
  - **"How might LLMs store facts"**: buscar en la playlist de Neural Networks de 3Blue1Brown.

### 🎥 Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*
Construye un GPT completo (self-attention, multi-head attention, positional encoding, feed-forward, residual connections) en ~2 horas de código en vivo, partiendo de los conceptos ya sentados en la serie zero-to-hero.
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- Repo relacionado (nanoGPT): https://github.com/karpathy/nanoGPT

### 📖 Jay Alammar — *The Illustrated Transformer* (el clásico escrito)
El artículo visual más citado del mundo sobre Transformers — usado en cursos de Stanford, MIT, Harvard. Diagramas paso a paso de cómo se calculan Q, K, V, cómo se combinan las cabezas de atención (multi-head) y cómo se ensamblan encoder/decoder.
- https://jalammar.github.io/illustrated-transformer/

### 📖 Jay Alammar — *The Illustrated GPT-2*
Continuación centrada en arquitecturas decoder-only (la familia GPT, que es la que usan la mayoría de LLMs actuales), con las mismas animaciones detalladas del mecanismo de atención en modo autoregresivo (token a token).
- https://jalammar.github.io/illustrated-gpt2/

### 📄 Paper original — *Attention Is All You Need* (Vaswani et al., 2017)
El paper que introdujo la arquitectura Transformer. Denso mates, pero después de los recursos anteriores se lee con mucha más soltura y merece la pena tenerlo como referencia final.
- https://arxiv.org/abs/1706.03762

---

## 5. Herramientas interactivas — para "tocar" los números

### 🖱️ LLM Visualization (bbycroft.net) — la joya interactiva
Renderizado 3D interactivo en tiempo real de un Transformer completo (modelo de juguete o arquitecturas tipo GPT-2/nano-GPT). Escribes una frase, ves cómo se "iluminan" las celdas de memoria, y puedes pasar el cursor sobre las matrices Q/K/V para ver numéricamente cada multiplicación de matrices y cada softmax celda por celda.
- https://bbycroft.net/llm

### 🖱️ Transformer Explainer (Georgia Tech / poloclub)
Visualización interactiva que corre un GPT-2 real en el navegador: escribes texto y ves en vivo cómo los componentes internos (embeddings, atención, MLP) generan cada siguiente token.
- https://poloclub.github.io/transformer-explainer/

### 🖱️ TensorFlow Playground
No es un Transformer, pero es la mejor forma de "jugar" con una red neuronal simple (capas, pesos, funciones de activación) y ver en tiempo real cómo cambia la frontera de decisión — útil para consolidar backprop antes de pasar a atención.
- https://playground.tensorflow.org/

### 🖱️ matrixcalculus.org
Calculadora simbólica de derivadas matriciales/vectoriales — muy útil para verificar a mano tus cálculos de backprop en las capas lineales.
- http://www.matrixcalculus.org/

---

## 6. Extra: para consolidar y profundizar

- **Michael Nielsen — "Neural Networks and Deep Learning"** (libro online gratuito, recomendado por el propio 3Blue1Brown en su primer vídeo): explicación matemática rigurosa pero accesible, con el mismo espíritu paso a paso.
  http://neuralnetworksanddeeplearning.com/
- **Distill.pub** (archivo histórico): artículos visuales e interactivos de investigación en deep learning, incluyendo interpretabilidad de redes. Buen sitio para explorar una vez tengas las bases.
  https://distill.pub/
- **Discord de Karpathy / comunidad de "Zero to Hero"**: mencionado en la página del curso, útil para resolver dudas mientras sigues los vídeos.

---

## Resumen del "camino crítico" (si solo tienes tiempo para lo esencial)

1. 3Blue1Brown — *Essence of Linear Algebra* (vectores, matrices, producto punto)
2. 3Blue1Brown — *Neural Networks* caps. 1-4 (intuición de red y backprop)
3. Karpathy — *building micrograd* (backprop en código, paso a paso)
4. Karpathy — *becoming a backprop ninja* (backprop matricial real)
5. 3Blue1Brown — *But what is a GPT* + *Attention, visually explained* (caps. 5-6)
6. Jay Alammar — *The Illustrated Transformer*
7. Karpathy — *Let's build GPT*
8. Jugar con **bbycroft.net/llm** en paralelo a todo lo anterior

Con esta ruta, el álgebra lineal deja de ser abstracta: cada matriz que veas en 3Blue1Brown o Karpathy la podrás "tocar" numéricamente en bbycroft.net, y cada derivada que calcules a mano la podrás verificar en matrixcalculus.org.
