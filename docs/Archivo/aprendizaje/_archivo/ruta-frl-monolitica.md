---
title: "Ruta de aprendizaje (con recursos) — Del ciclo de entrenamiento a los paradigmas"
tags: [archivo]
status: borrador
updated: 2026-08-08
---

# Ruta de aprendizaje (con recursos) — Del ciclo de entrenamiento a los paradigmas

> **Documento companion** del *Modelo Mental: del Deep Learning al Aprendizaje Federado con Refuerzo*. Mientras aquel ordena los **conceptos**, este ordena los **recursos** (PDFs, vídeos, cursos, práctica) y el **orden de estudio**. Se construye por módulos; de momento cubre los **cimientos matemáticos** + el **punto 2** (ciclo de entrenamiento) + el **punto 3** (paradigmas de aprendizaje). Los módulos de FL, Data Spaces y FRL se añadirán después.

**Cómo leer cada recurso:** `[ tipo · nivel · tiempo ]`
- Tipo: 📘 libro/PDF · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica (código)
- Nivel: ⭐ introductorio · ⭐⭐ intermedio · ⭐⭐⭐ avanzado

**Regla de oro:** no intentes leerlo *todo*. Cada módulo trae una **ruta mínima** (lo imprescindible) y luego *profundización opcional*. Empieza por la ruta mínima y vuelve a lo opcional solo si lo necesitas.

---

## Módulo 0 — Cimientos matemáticos: álgebra lineal (y el puente de cálculo)

> **Por qué primero:** el ciclo de entrenamiento del Módulo 1 *es* álgebra lineal en movimiento. Una capa de una red es una multiplicación matriz-vector `Wx + b`; el *backpropagation* es la **regla de la cadena** aplicada sobre esas operaciones. Sin estos cimientos, el Módulo 1 se memoriza pero no se entiende.

### Lo que de verdad necesitas (no hace falta un curso entero)

Para entender el entrenamiento de una red, prioriza exactamente esto:

| Concepto de álgebra lineal | Dónde aparece en el entrenamiento |
|---|---|
| Vectores y matrices | Datos `x`, pesos `W`, activaciones |
| **Multiplicación matriz-vector / matriz-matriz** | **El forward pass de una capa: `Wx + b`** |
| Producto escalar (dot product) | Lo que calcula cada neurona |
| Transpuesta `Wᵀ` | Aparece al propagar gradientes hacia atrás |
| Broadcasting (suma del sesgo) | Sumar `b` a todo el lote |
| Normas (L1, L2) | Funciones de pérdida y regularización |

Y el **puente de cálculo** (mínimo imprescindible, no es álgebra lineal pero lo necesitas para el backprop):
- Derivada y derivada parcial · **gradiente** (vector de derivadas parciales) · **regla de la cadena** · noción de jacobiano (la regla de la cadena en versión vectorial).

### Ruta mínima

- 🎥⭐ **3Blue1Brown — *Essence of Linear Algebra*** · https://www.3blue1brown.com/topics/linear-algebra
  La mejor intuición *geométrica* de vectores, matrices, multiplicación como composición de transformaciones, y eigenvectores. ~3 h en total. Imprescindible para "ver" lo que hace una capa.
- 🎥⭐ **3Blue1Brown — *Essence of Calculus*** (solo capítulos de derivada y **regla de la cadena**) · https://www.3blue1brown.com/topics/calculus
  Con esto cubres el puente de cálculo necesario para el backprop.

### Profundización opcional (referencia, no lectura lineal)

- 📘⭐⭐ **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (gratuito, oficial) · web: https://mml-book.com/ · PDF: https://mml-book.github.io/book/mml-book.pdf
  El libro de referencia. Para esta ruta basta el **Cap. 2 (Linear Algebra)**, **Cap. 3 (Analytic Geometry: normas, productos internos)**, **Cap. 5 (Vector Calculus: gradientes, regla de la cadena)**. Los Cap. 4 (descomposiciones, SVD) y 6 (probabilidad) son útiles más adelante.
- 🎥🔗⭐⭐ **Gilbert Strang — MIT 18.06 Linear Algebra** (MIT OpenCourseWare, gratuito) · https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
  El curso clásico si quieres una base sólida y formal de álgebra lineal. Es más de lo que necesitas para arrancar, pero es la referencia.

**Checkpoint del módulo:** sabes explicar, sin mirar, qué hace `Wx + b` geométricamente y por qué la regla de la cadena permite calcular cómo afecta cada peso al error.

---

## Módulo 1 — El ciclo de entrenamiento de una red profunda (punto 2)

> **Mapa mental:** Sección 2.3. El objetivo es interiorizar el bucle: **forward → pérdida → backpropagation → optimización**, y la frase clave del documento: *"entrenar = calcular gradientes y actualizar pesos"* (que es justo lo que el FL acaba compartiendo en lugar de los datos).

### Conceptos que debes dominar al salir

1. **Forward pass:** cómo la entrada atraviesa capas de transformaciones lineales + no linealidades (activaciones) hasta la predicción.
2. **Función de pérdida:** cómo se mide el error (entropía cruzada para clasificación, MSE para regresión).
3. **Backpropagation:** la regla de la cadena que reparte el error hacia atrás y calcula el gradiente respecto a cada peso.
4. **Optimización:** descenso por gradiente y sus variantes (SGD, Adam), tasa de aprendizaje, lotes (batches), épocas.
5. **Generalización:** overfitting/underfitting, train/val/test, regularización (enlaza con el Módulo 2.1).

### Ruta mínima (en este orden)

- 🎥⭐ **3Blue1Brown — *Neural Networks*** · https://www.3blue1brown.com/topics/neural-networks
  4 vídeos que explican, con intuición visual, qué es una red, qué es el descenso por gradiente y **qué es el backpropagation**. El mejor primer contacto. ~1 h.
- 🧑‍💻⭐⭐ **Andrej Karpathy — *Neural Networks: Zero to Hero*** (Lección 1: *building micrograd*) · https://karpathy.ai/zero-to-hero.html · código: https://github.com/karpathy/nn-zero-to-hero
  Construye un motor de *backpropagation* desde cero en Python, escalar a escalar. Tras esta lección, el backprop deja de ser magia. **Es el recurso decisivo de este módulo**: codifícalo tú mismo, no solo lo veas.
- 🔗⭐⭐ **Dive into Deep Learning (d2l.ai)** — capítulos *Linear Neural Networks* y *Multilayer Perceptrons* · https://d2l.ai/
  Libro interactivo y gratuito, con código ejecutable. Aquí pasas del escalar (micrograd) a las operaciones con tensores/lotes reales.

### Profundización opcional

- 🔗⭐⭐ **CS231n (Stanford) — notas de Backpropagation y Optimization** · https://cs231n.github.io/
  Las notas escritas más claras sobre el flujo de gradientes y los optimizadores. Ideal como referencia para consolidar.
- 📘⭐⭐⭐ **Deep Learning** — Goodfellow, Bengio, Courville (gratuito, HTML) · https://www.deeplearningbook.org/
  El libro de referencia teórico. Para este módulo: Cap. 6 (redes feedforward) y Cap. 8 (optimización). Denso; úsalo como consulta, no como lectura inicial.
- 🔗🧑‍💻⭐⭐ **fast.ai — Practical Deep Learning for Coders** · https://course.fast.ai/
  Enfoque *top-down* y muy práctico si prefieres entrenar modelos reales pronto y entender la teoría después.

**Checkpoint del módulo:** puedes describir el bucle de entrenamiento de memoria y, mejor aún, has reimplementado un micro-backprop que entrena. Entiendes por qué "lo que viaja en el FL son los gradientes/pesos".

---

## Módulo 2 — Los paradigmas de aprendizaje (punto 3)

> **Mapa mental:** Sección 3 (eje ① "CÓMO"). La idea organizadora: lo que distingue a los paradigmas es **de dónde viene la señal de aprendizaje**. Estudia primero el espectro de la supervisión y deja el RL para el final, porque es el más extenso y el que conecta con tu objetivo (FRL).

### 2.1 Supervisado, no supervisado y el espectro de la supervisión

> Mapa mental: 3.1, 3.2, 3.4.

- 🎥⭐ **StatQuest with Josh Starmer — playlist de Machine Learning** · https://www.youtube.com/@statquest
  Explicaciones cortas y clarísimas de clasificación, regresión, clustering, sesgo-varianza, regularización, train/test. El mejor sitio para fijar los fundamentos del supervisado y el no supervisado.
- 🔗🧑‍💻⭐⭐ **Andrew Ng — Machine Learning Specialization (Coursera, audita gratis)** · https://www.coursera.org/specializations/machine-learning-introduction
  El curso introductorio de referencia. Cubre supervisado y no supervisado con práctica. Puedes auditarlo sin pagar el certificado.

**Checkpoint:** distingues con soltura qué problema es supervisado, no supervisado o intermedio, y por qué etiquetar es el cuello de botella que motiva el resto.

### 2.2 Auto-supervisado (self-supervised) — el motor de los foundation models

> Mapa mental: 3.3. Clave para entender los LLMs y muy relevante para preentrenar sobre datos privados en FL.

- 🔗⭐⭐ **Lilian Weng — *Self-Supervised Representation Learning*** · https://lilianweng.github.io/posts/2019-11-10-self-supervised/
  El artículo de referencia que recorre las familias de tareas pretexto (contrastivas, predictivas, enmascaradas). Denso pero excelente.
- 🧑‍💻⭐⭐ **Karpathy — *Zero to Hero*, lección "Let's build GPT"** · https://karpathy.ai/zero-to-hero.html
  Ver el auto-supervisado en acción: predecir el siguiente token construyendo un mini-GPT desde cero.

**Checkpoint:** sabes explicar cómo un modelo "fabrica sus propias etiquetas" y por qué la receta *preentrenar (SSL) → fine-tuning* domina hoy.

### 2.3 Aprendizaje por refuerzo (RL) — el paradigma clave para el FRL

> Mapa mental: 3.5 y Sección 4.3. Es el módulo más largo. Ruta: primero intuición y MDPs, luego algoritmos (DQN → PPO), y solo después RL offline.

**Ruta mínima (en este orden):**

- 🔗⭐ **OpenAI Spinning Up — *Intro to RL*** · https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
  La introducción conceptual más limpia: agente, entorno, estado, acción, recompensa, política, función de valor. Empieza aquí.
- 🔗🧑‍💻⭐⭐ **Hugging Face — Deep RL Course** (gratuito, con práctica en Colab) · https://huggingface.co/learn/deep-rl-course/unit0/introduction
  Curso por unidades de Q-Learning → Deep Q-Learning → métodos de política → actor-critic, entrenando agentes reales. La mejor vía *práctica*. (Nota: la parte de retos/leaderboard ya no está operativa, pero toda la teoría y los hands-on siguen siendo válidos.)
- 📘⭐⭐⭐ **Sutton & Barto — *Reinforcement Learning: An Introduction* (2.ª ed., gratuito)** · https://incompleteideas.net/book/the-book-2nd.html
  La biblia del RL. No lo leas entero de golpe: úsalo para profundizar en MDPs (Cap. 3), métodos de valor (Cap. 6) y políticas/actor-critic (Cap. 13) a medida que los ves en el curso de Hugging Face.

**Profundización opcional:**

- 🎥⭐⭐⭐ **David Silver — RL Course (UCL/DeepMind)** · http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html
  Las clases magistrales clásicas de RL. Más teóricas; complemento ideal de Sutton & Barto.
- 🧑‍💻⭐⭐ **Spinning Up — implementaciones de algoritmos (VPG, PPO, DDPG, SAC...)** · https://spinningup.openai.com/en/latest/
  Implementaciones limpias y documentadas para leer y modificar. Aquí ves DQN/PPO/SAC del catálogo de la Sección 4.3.

**Sobre el RL offline** (muy relevante para el FRL en dominios regulados): tras dominar el RL online, busca en Spinning Up y Sutton & Barto los conceptos de *off-policy* y *distribution shift*; el algoritmo de referencia es **CQL (Conservative Q-Learning)**. Lo abordaremos en detalle en el módulo de FRL.

**Checkpoint del módulo:** sabes plantear un problema como un MDP, distingues online/offline y on/off-policy, y has entrenado al menos un agente (p. ej., el lander de la Unit 1 de Hugging Face).

### 2.4 Paradigmas mixtos (puente hacia el FRL)

> Mapa mental: 3.6. Solo necesitas una primera pasada conceptual aquí; profundizaremos en el módulo de FRL.

- 🔗⭐⭐ **Hugging Face — *Illustrating Reinforcement Learning from Human Feedback (RLHF)*** · https://huggingface.co/blog/rlhf
  Explica el encadenamiento SFT → modelo de recompensa → RL (PPO) que hay detrás de ChatGPT. El mejor ejemplo de "paradigma mixto" supervisado + RL.

**Checkpoint:** entiendes por qué imitation learning, RLHF y RL offline son híbridos supervisado↔RL, justo el territorio del FRL.

---

## Resumen de la ruta (orden recomendado)

1. **Módulo 0 (ruta mínima):** 3Blue1Brown *Linear Algebra* + regla de la cadena de *Calculus*.
2. **Módulo 1:** 3Blue1Brown *Neural Networks* → Karpathy *micrograd* (codifícalo) → d2l.ai.
3. **Módulo 2.1–2.2:** StatQuest / Andrew Ng → Lilian Weng (self-supervised).
4. **Módulo 2.3:** Spinning Up *Intro* → Hugging Face Deep RL Course → Sutton & Barto (consulta).
5. **Módulo 2.4:** RLHF de Hugging Face (primera pasada).

**Tiempo orientativo de la ruta mínima:** ~25–40 h de estudio activo (con la práctica de micrograd y al menos un agente de RL entrenado). Las profundizaciones opcionales pueden multiplicar eso según hasta dónde quieras llegar.

> **Próximos módulos a añadir:** Módulo 3 (Aprendizaje Federado: FedAvg, no-IID, privacidad), Módulo 4 (Data Spaces: IDSA, Gaia-X, conectores) y Módulo 5 (FRL). Avísame y los preparamos con el mismo formato.
