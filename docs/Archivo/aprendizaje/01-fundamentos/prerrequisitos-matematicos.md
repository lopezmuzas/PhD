---
title: "Fase 0 — Prerrequisitos Matemáticos y de Código para Deep RL"
tags: [fundamentos, matematicas, neurociencia]
status: borrador
updated: 2026-08-08
---

# Fase 0 — Prerrequisitos Matemáticos y de Código para Deep RL

> **Objetivo de la fase**: dominar el entorno base sobre el que se construye el Aprendizaje por Refuerzo profundo. Sin esto, los papers de RL se vuelven inaccesibles y el código de los agentes es magia negra.
>
> **Tu situación de partida**: ya has tocado PyTorch/TensorFlow alguna vez → ligero en la parte de frameworks, **denso en matemáticas**.
>
> **Tiempo estimado realista**: 6–10 semanas a ritmo de 1–2 h/día. No es una carrera; el objetivo es *entender*, no *terminar*.

---

## Cómo usar esta guía

Esta guía es un **índice maestro**. Está organizada por bloques, y dentro de cada bloque tienes:

- **Por qué importa en RL** → la conexión directa con el aprendizaje por refuerzo
- **Recursos principales** → curso central por el que te guías
- **Refuerzo en vídeo/audio** → tu formato preferido
- **Lectura en inglés** → para integrar el idioma técnico
- **Ejercicios mínimos** → no te puedes saltar esto
- **Checklist** → criterios concretos para saber que ese bloque está dominado

Cuando termines un bloque, vuelves aquí, lo marcas, y me pides que profundicemos en el siguiente con ejemplos, ejercicios y código.

---

## Mapa de la Fase 0

| Bloque | Tema | Peso |
|---|---|---|
| 0.1 | Álgebra Lineal | 🔥🔥🔥🔥 Crítico |
| 0.2 | Cálculo Multivariable | 🔥🔥🔥🔥 Crítico |
| 0.3 | Probabilidad y Estadística | 🔥🔥🔥🔥🔥 El más crítico para RL |
| 0.4 | Cadenas de Markov y MDPs | 🔥🔥🔥🔥🔥 Puente directo a RL |
| 0.5 | Python avanzado | 🔥🔥 Repaso enfocado |
| 0.6 | PyTorch para RL | 🔥🔥🔥 Específico, no generalista |

---

## 0.1 — Álgebra Lineal

### Por qué importa en RL

Una política neuronal es una composición de transformaciones lineales (matrices de pesos) intercaladas con no linealidades. Las representaciones de estado son vectores. Las funciones de valor se aproximan con productos escalares. Cuando leas un paper de PPO o SAC y veas $\theta^\top \phi(s)$, $\nabla_\theta \log \pi_\theta(a|s)$, o $\mathbb{E}[\nabla \log \pi \cdot A]$, necesitas que esos símbolos te resulten *naturales*, no obstáculos.

### Recurso principal (vídeo, EN)

- **3Blue1Brown — Essence of Linear Algebra** → [Playlist YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab). Es el punto de partida innegociable. 15 vídeos cortos, visualmente impecables. Te da la **intuición geométrica** que no encontrarás en ningún libro.

### Curso estructurado

- **MIT 18.06 — Gilbert Strang** → [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). El clásico absoluto. Strang explica con un cariño que te enamora del tema. Ve las clases 1–10, 14, 21, 22 como mínimo (eigenvalores incluidos).
- Alternativa más moderna: **Mathematics for Machine Learning Specialization (Imperial College, Coursera)** — [Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning). Más orientado a ML, menos profundo que Strang.

### Lectura en inglés

- **"Mathematics for Machine Learning"** — Deisenroth, Faisal, Ong. PDF gratis legal en [mml-book.com](https://mml-book.com/). Capítulos 2–4. Es el libro de referencia moderno; léelo en paralelo a los vídeos.

### Temas mínimos que tienes que dominar

Vectores, espacios vectoriales, combinaciones lineales y base. Producto escalar y norma (L1, L2, y por qué la regularización se llama así). Matrices como transformaciones lineales (no como tablas de números). Producto matricial, inversa, transpuesta. Determinante e interpretación geométrica. Rango, espacio columna y espacio nulo. Eigenvalores y eigenvectores (clave para PCA y para entender la estabilidad de operadores como el de Bellman). Descomposiciones: SVD al menos a nivel intuitivo.

### Checklist 0.1

- [ ] Sé explicar qué le hace una matriz a un vector sin usar números
- [ ] Sé calcular un producto matricial a mano para matrices 3×3
- [ ] Entiendo qué es un eigenvector y por qué importa
- [ ] Sé qué hace `torch.matmul`, `@`, y por qué a veces falla por dimensiones

---

## 0.2 — Cálculo Multivariable y Matricial

### Por qué importa en RL

**Todo en deep RL es gradient descent sobre una función de pérdida estocástica.** El algoritmo REINFORCE es literalmente $\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$. Si no entiendes qué es ese gradiente, ni qué cadena de reglas se aplica para calcularlo en backprop, no puedes entender *por qué* funciona REINFORCE, ni por qué falla, ni cómo corregirlo con baselines o ventajas.

### Recurso principal (vídeo, EN)

- **3Blue1Brown — Essence of Calculus** → [Playlist YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr). 12 vídeos. Te da la intuición de derivada como tasa de cambio y de integral como acumulación.
- **3Blue1Brown — Neural Networks (vídeos 3 y 4)** → backpropagation explicada visualmente. Vital después.

### Curso estructurado

- **Mathematics for Machine Learning: Multivariate Calculus (Imperial College)** → [Coursera](https://www.coursera.org/learn/multivariate-calculus-machine-learning). Está pensado para ML: gradientes, Jacobiano, Hessiano, regla de la cadena en redes neuronales, Taylor multivariable. Es el match perfecto.

### Para nivel pro (cuando ya estés sólido)

- **MIT 18.S096 — Matrix Calculus for Machine Learning and Beyond** → [OCW](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/). Curso corto de invierno en MIT, dirigido por Edelman y Johnson. **Cubre exactamente lo que necesitas para entender backprop de verdad**: forward vs. reverse mode AD, derivadas como operadores lineales, gradientes de funciones matriciales. Es oro puro y casi nadie lo conoce.

### Lectura en inglés

- **"The Matrix Calculus You Need For Deep Learning"** — Parr & Howard. [Artículo gratis en explained.ai](https://explained.ai/matrix-calculus/). Es el mejor recurso corto que existe. Léelo dos veces.
- **MML Book**, capítulo 5 (Vector Calculus).

### Temas mínimos

Derivadas parciales y el gradiente como vector que apunta hacia el ascenso más pronunciado. Regla de la cadena multivariable — la base literal del backprop. Jacobiano (cuando la función va de $\mathbb{R}^n$ a $\mathbb{R}^m$). Hessiano y su uso en métodos de segundo orden (Newton, y por qué TRPO los aproxima). Gradiente descendente y por qué el `learning rate` mata o salva tu entrenamiento. Diferenciación automática (forward vs. reverse mode) — entiende por qué backprop es reverse mode, no por qué "es magia de PyTorch".

### Checklist 0.2

- [ ] Sé derivar a mano $\nabla_\theta \|y - X\theta\|^2$ y obtener la solución de mínimos cuadrados
- [ ] Sé explicar qué hace `.backward()` en PyTorch en términos de la regla de la cadena
- [ ] Entiendo por qué los gradientes "explotan" o "se desvanecen" en redes profundas
- [ ] Puedo escribir el gradiente de una pérdida de cross-entropy a mano

---

## 0.3 — Probabilidad y Estadística

### Por qué importa en RL — y esta es la sección que más vas a usar

Una política $\pi(a|s)$ **es** una distribución de probabilidad condicional. El entorno $P(s'|s,a)$ **es** otra. La función de valor $V^\pi(s) = \mathbb{E}_\pi[\sum \gamma^t r_t]$ es una **esperanza**. KL-divergence aparece en TRPO/PPO. Entropía aparece como bonus en SAC. Sin probabilidad sólida, RL es leer hieroglíficos. Esta es **la** sección donde no puedes pasar de puntillas.

### Recurso principal (vídeo, EN) — el bueno bueno

- **Harvard Statistics 110 — Joe Blitzstein** → [Playlist YouTube](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo). 34 clases. Es **probablemente el mejor curso de probabilidad en internet**. Blitzstein es excepcional explicando. Cubre desde axiomas hasta cadenas de Markov en la lección 31–34, perfecto puente con el bloque 0.4. El libro asociado, *Introduction to Probability*, también es gratis en su web.

### Refuerzo y enfoque ML

- **StatQuest with Josh Starmer** → [Canal YouTube](https://www.youtube.com/@statquest). Vídeos de 10–20 minutos sobre conceptos puntuales (Bayes, MLE, distribuciones, EM, etc.). Ideal para reforzar después de ver Blitzstein. Su estilo es divulgativo pero correcto.
- **DeepLearning.AI — Probability & Statistics for ML** → [Coursera](https://www.coursera.org/learn/machine-learning-probability-and-statistics). Más enfocado a aplicación práctica.

### Lectura en inglés

- **MML Book**, capítulo 6 (Probability and Distributions).
- **"Pattern Recognition and Machine Learning"** — Bishop. Capítulos 1, 2, 8. Es denso pero seminal. Léelo en consultas puntuales, no de cabo a rabo todavía.

### Temas mínimos

Espacio muestral, eventos, axiomas. Probabilidad condicional y la fórmula que más vas a ver en tu vida: la regla de la cadena $P(A,B) = P(A|B)P(B)$. Teorema de Bayes — entiéndelo no solo formalmente sino como mecanismo de **actualización de creencias**. Variables aleatorias discretas y continuas. Esperanza y varianza, propiedades de linealidad. Distribuciones clave: Bernoulli, Binomial, Categórica (softmax es esto), Gaussiana univariante y multivariante. Funciones de densidad vs. funciones de masa. Independencia y independencia condicional. Maximum Likelihood Estimation (MLE) — es lo que estás haciendo cuando entrenas casi cualquier red. Cross-entropy y KL-divergence — entiende que minimizar cross-entropy *es* MLE. Ley de los grandes números y teorema central del límite (justifican que Monte Carlo funcione, y RL es Monte Carlo aplicado).

### Checklist 0.3

- [ ] Resuelvo el problema de Monty Hall y sé explicar por qué con Bayes
- [ ] Sé derivar la MLE de una Gaussiana
- [ ] Entiendo qué es la KL-divergence y por qué no es simétrica
- [ ] Sé qué es una política estocástica y puedo escribirla como distribución softmax o gaussiana
- [ ] Puedo escribir y entender $\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum f(x_i)$ y por qué es la base de Monte Carlo

---

## 0.4 — Cadenas de Markov y MDPs (puente a RL)

### Por qué importa en RL

**Un MDP (Markov Decision Process) es literalmente el modelo matemático del entorno de RL.** Si dominas cadenas de Markov, entender MDPs es trivial: un MDP es una cadena de Markov + acciones + recompensas. La ecuación de Bellman emerge naturalmente del cálculo de esperanzas sobre transiciones de Markov. Este bloque es el último escalón antes de saltar a RL.

### Recursos en vídeo

- **Harvard Stat 110, Lectures 31–34** → ya las tienes si hiciste 0.3 completo. Cubren cadenas de Markov, matriz de transición, distribución estacionaria, reversibilidad.
- **MIT 6.041 — Probabilistic Systems Analysis (Tsitsiklis)** → [OCW](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/). Lecciones sobre Markov chains. Tsitsiklis es uno de los autores del libro de RL canónico (Neuro-Dynamic Programming).
- **Mutual Information (YouTube)** → [Canal](https://www.youtube.com/@Mutual_Information). Tiene una serie excelente y visual sobre RL que empieza desde MDPs. Es lo más cercano a "3Blue1Brown pero de RL".

### Lectura en inglés (anticipo de Fase 1)

- **Sutton & Barto — "Reinforcement Learning: An Introduction"** (2nd ed.) → [PDF gratis legal](http://incompleteideas.net/book/RLbook2020.pdf). Capítulos 1, 2 y 3 ya. El capítulo 3 (Finite MDPs) lo deberías leer al final de la Fase 0 como puente. **Este libro será tu biblia durante toda la Fase 1 y 2.**

### Temas mínimos

Propiedad de Markov: el futuro depende solo del presente, no del pasado. Matriz de transición y por qué sus filas suman 1. Distribución estacionaria y cuándo existe (irreducibilidad, aperiodicidad). Procesos de decisión de Markov (MDP) — la tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Política $\pi$ determinista vs. estocástica. Función de valor $V^\pi$ y función Q $Q^\pi$. Ecuaciones de Bellman (expectativa y optimalidad). Por qué el factor de descuento $\gamma < 1$ es matemáticamente necesario.

### Checklist 0.4

- [ ] Sé calcular la distribución estacionaria de una cadena de Markov de 3 estados a mano
- [ ] Puedo escribir la ecuación de Bellman para $V^\pi$ sin mirar
- [ ] Entiendo por qué $Q^\pi(s,a) = R(s,a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$
- [ ] He leído Sutton & Barto capítulos 1–3

---

## 0.5 — Python avanzado (repaso enfocado)

### Por qué importa en RL

El código de RL real (Stable-Baselines3, CleanRL, RLlib) usa **a saco**: decoradores, dataclasses, generadores (para replay buffers iterables), context managers (para `torch.no_grad()`), type hints, herencia y composición, y a veces multiprocessing/async para paralelizar entornos. No necesitas ser un Pythonista superlativo, pero sí leer este código sin tropezarte.

### Recursos en vídeo

- **Corey Schafer — Python Tutorials** → [Canal YouTube](https://www.youtube.com/@coreyms). Sus vídeos de decoradores, generadores, OOP y context managers son la referencia.
- **mCoding (James Murphy)** → [Canal YouTube](https://www.youtube.com/@mCoding). Vídeos cortos sobre patrones idiomáticos modernos: typing, dataclasses, pattern matching, async. Nivel medio-alto.
- **ArjanCodes** → [Canal YouTube](https://www.youtube.com/@ArjanCodes). Diseño y patrones en Python, útil cuando quieras estructurar tu propio repo de agentes.

### Lectura en inglés

- **"Fluent Python"** — Luciano Ramalho (2ª ed.). El libro de Python avanzado por excelencia. No lo leas entero; ve los capítulos sobre data classes, iteradores/generadores, decoradores, context managers, concurrencia.

### Temas mínimos

Decoradores (incluyendo decoradores con argumentos). Generadores y la diferencia con listas en memoria — crítico para replay buffers grandes. Context managers (`with`, `__enter__`/`__exit__`, `contextlib`). Dataclasses y `NamedTuple` (Stable-Baselines3 los usa para representar transiciones). Type hints y `typing` (`Optional`, `Union`, `Callable`, `Protocol`). NumPy avanzado: broadcasting, vectorización, ejes (axis), `np.einsum`. Manejo básico de paralelismo: `multiprocessing.Pool` y `concurrent.futures` (lo usarás para paralelizar entornos gym).

### Checklist 0.5

- [ ] Escribo un decorador `@timer` que mida cuánto tarda una función
- [ ] Implemento un generador para un replay buffer infinito
- [ ] Entiendo cuándo usar `dataclass` vs. `NamedTuple` vs. clase normal
- [ ] Sé qué hace `np.einsum('ij,jk->ik', A, B)` y por qué a veces es más rápido

---

## 0.6 — PyTorch para RL (no para CV genérico)

### Por qué importa en RL

Ya has usado PyTorch, así que aquí el objetivo no es repasar `nn.Linear`, sino **dominar lo que RL usa específicamente**: distribuciones (`torch.distributions`), `log_prob`, `entropy`, manipulación cuidadosa de gradientes (`.detach()`, `with torch.no_grad()`), y el ciclo de pérdida no estándar (no hay un dataset fijo, las trayectorias se generan online).

### Recursos en vídeo

- **PyTorch Official Tutorials — Reinforcement Learning (DQN)** → [pytorch.org](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). Tutorial oficial corto, te enseña el ciclo "samplear → almacenar → actualizar" típico de RL.
- **Aleksa Gordić — The AI Epiphany** → [Canal YouTube](https://www.youtube.com/@TheAIEpiphany). Tiene vídeos *line-by-line* de implementaciones de DQN, PPO, etc. en PyTorch. Cuando avances a Fase 1+, vuelve aquí.
- **Yannic Kilcher** → [Canal](https://www.youtube.com/@YannicKilcher). Para paper reviews. Lo guardas para más adelante.

### Recursos clave para RL específicamente

- **CleanRL** → [GitHub](https://github.com/vwxyzjn/cleanrl). **Esto es oro**: implementaciones de un único fichero de cada algoritmo de RL en PyTorch. Cuando llegues a Fase 1, lo leerás constantemente. Empieza ojeando `dqn.py` para ver qué piezas de PyTorch usan.
- **Spinning Up in Deep RL — OpenAI** → [spinningup.openai.com](https://spinningup.openai.com/en/latest/). Aún no entres en los algoritmos; lee la sección "Intro to RL" y "Key Concepts" para ver vocabulario y notación que coincide con Sutton & Barto.

### Temas mínimos

Tensores y operaciones, broadcasting (igual que NumPy pero con `.to(device)`). `autograd`: cuándo se acumulan gradientes y cuándo se debe llamar a `.zero_grad()`. `.detach()` y `with torch.no_grad()` — sutilísimos en RL: la red target en DQN se detacha, los advantages se detachan en PPO, etc. Equivocarse aquí mete bugs *silenciosos*. `torch.distributions.Categorical` y `Normal` — la base de las políticas estocásticas; `log_prob(action)` es lo que multiplicas por el advantage. Optimizadores (Adam, RMSProp) y schedulers de learning rate. GPU: mover tensores y modelos, debuggear errores de device mismatch.

### Checklist 0.6

- [ ] Escribo una política gaussiana con `torch.distributions.Normal` y obtengo $\log \pi(a|s)$ correctamente
- [ ] Sé cuándo aplicar `.detach()` y puedo explicar qué pasaría sin él en DQN
- [ ] Entiendo qué hace `loss.backward()` línea por línea en términos de grafo computacional
- [ ] He clonado CleanRL y entiendo la estructura de `dqn.py` aunque no todos los detalles

---

## Pre-final: lo que se ve cuando se cierra la Fase 0

Cuando termines esto, deberías poder abrir el [paper original de DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) o el [paper de PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) y, aunque no entiendas todos los detalles algorítmicos (eso es Fase 1+), las **ecuaciones no te parezcan extrañas**. Reconocerás esperanzas, gradientes, KL-divergence, advantages, ratios de probabilidad. Eso es exactamente lo que esta fase compra.

---

## Plan sugerido de 8 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1–2 | 0.1 Álgebra Lineal | 0.5 Python (decoradores, generadores) |
| 3–4 | 0.2 Cálculo Multivariable | 0.5 Python (NumPy avanzado) |
| 5–6 | 0.3 Probabilidad (Stat 110 intensivo) | 0.6 PyTorch (refresco) |
| 7 | 0.4 Markov chains + MDPs | Sutton & Barto cap. 1–3 |
| 8 | 0.6 PyTorch específico de RL | Leer CleanRL `dqn.py` |

Si descubres que un bloque te cuesta más, **estíralo sin culpa**. La prisa en la Fase 0 se paga con intereses en la Fase 1.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:
- Explicaciones más extensas con ejemplos numéricos
- Ejercicios concretos resueltos paso a paso
- Implementaciones en código (Python/PyTorch) de los conceptos
- Conexiones explícitas con un algoritmo concreto de RL

Mi recomendación: empieza por **0.3 (Probabilidad)** si tienes prisa por llegar a RL, o por **0.1 (Álgebra Lineal)** si quieres construir la base más sólida posible. Las dos son válidas.
