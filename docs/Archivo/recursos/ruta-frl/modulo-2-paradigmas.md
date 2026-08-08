---
title: "Módulo 2 — Los paradigmas de aprendizaje"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 2 — Los paradigmas de aprendizaje

> 🧭 **Ruta FRL:** [Índice](index.md) · [0](modulo-0-cimientos-matematicos.md) · [1](modulo-1-ciclo-entrenamiento.md) · **2** · [3](modulo-3-aprendizaje-federado.md)
> 🗺️ **Mapa mental:** Sección 3 (eje ① "CÓMO").
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Idea organizadora:** lo que distingue a los paradigmas es **de dónde viene la señal de aprendizaje**. Estudia primero el espectro de la supervisión y deja el RL para el final, porque es el más extenso y el que conecta con tu objetivo (FRL).

## 2.1 Supervisado, no supervisado y el espectro de la supervisión

> Mapa mental: 3.1, 3.2, 3.4.

- 🎥⭐ **StatQuest with Josh Starmer — playlist de Machine Learning** · https://www.youtube.com/@statquest
  Explicaciones cortas y clarísimas de clasificación, regresión, clustering, sesgo-varianza, regularización, train/test. El mejor sitio para fijar los fundamentos del supervisado y el no supervisado.
- 🔗🧑‍💻⭐⭐ **Andrew Ng — Machine Learning Specialization (Coursera, audita gratis)** · https://www.coursera.org/specializations/machine-learning-introduction
  El curso introductorio de referencia. Cubre supervisado y no supervisado con práctica. Puedes auditarlo sin pagar el certificado.

**Checkpoint:** distingues con soltura qué problema es supervisado, no supervisado o intermedio, y por qué etiquetar es el cuello de botella que motiva el resto.

## 2.2 Auto-supervisado (self-supervised) — el motor de los foundation models

> Mapa mental: 3.3. Clave para entender los LLMs y muy relevante para preentrenar sobre datos privados en FL (Módulo 3).

- 🔗⭐⭐ **Lilian Weng — *Self-Supervised Representation Learning*** · https://lilianweng.github.io/posts/2019-11-10-self-supervised/
  El artículo de referencia que recorre las familias de tareas pretexto (contrastivas, predictivas, enmascaradas). Denso pero excelente.
- 🧑‍💻⭐⭐ **Karpathy — *Zero to Hero*, lección "Let's build GPT"** · https://karpathy.ai/zero-to-hero.html
  Ver el auto-supervisado en acción: predecir el siguiente token construyendo un mini-GPT desde cero.

**Checkpoint:** sabes explicar cómo un modelo "fabrica sus propias etiquetas" y por qué la receta *preentrenar (SSL) → fine-tuning* domina hoy.

## 2.3 Aprendizaje por refuerzo (RL) — el paradigma clave para el FRL

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

## 2.4 Paradigmas mixtos (puente hacia el FRL)

> Mapa mental: 3.6. Solo necesitas una primera pasada conceptual aquí; profundizaremos en el módulo de FRL.

- 🔗⭐⭐ **Hugging Face — *Illustrating Reinforcement Learning from Human Feedback (RLHF)*** · https://huggingface.co/blog/rlhf
  Explica el encadenamiento SFT → modelo de recompensa → RL (PPO) que hay detrás de ChatGPT. El mejor ejemplo de "paradigma mixto" supervisado + RL.

**Checkpoint:** entiendes por qué imitation learning, RLHF y RL offline son híbridos supervisado↔RL, justo el territorio del FRL.

➡️ Siguiente: [Módulo 3 — Aprendizaje Federado](modulo-3-aprendizaje-federado.md)
