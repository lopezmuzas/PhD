---
title: "Módulo 3 — Aprendizaje Federado (Federated Learning, FL)"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 3 — Aprendizaje Federado (Federated Learning, FL)

> 🧭 **Ruta FRL:** [Índice](index.md) · [0](modulo-0-cimientos-matematicos.md) · [1](modulo-1-ciclo-entrenamiento.md) · [2](modulo-2-paradigmas.md) · **3**
> 🗺️ **Mapa mental:** Sección 5 (el eje ③ "DÓNDE").
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Objetivo:** entender el FL como *"el modelo viaja, los datos no"*: rondas de entrenamiento local + agregación (FedAvg). Saber distinguir sus tipos, identificar sus retos y conocer las contramedidas, y haber entrenado un modelo federado simulado.
**Prerrequisito clave:** [Módulo 1](modulo-1-ciclo-entrenamiento.md). El FL **es** el ciclo de entrenamiento del Módulo 1, pero **compartiendo gradientes/pesos en lugar de los datos**. La intuición de "entrenar submodelos y agregar" ya la viste en el bagging (mapa mental §4.4).

## Conceptos que debes dominar al salir

1. **El problema:** por qué no se pueden centralizar los datos — privacidad (RGPD), regulación sectorial, silos organizativos, competencia comercial. Solución: *mover el cómputo al dato, no el dato al cómputo*.
2. **FedAvg (la ronda):** el servidor envía el modelo global `wₜ` → cada cliente entrena localmente con sus datos → suben **solo** los pesos/gradientes → el servidor **agrega** con una media ponderada `wₜ₊₁ = Σ (nₖ/n)·wᵏ` → repetir hasta converger.
3. **Tipos de FL:**
   - **Horizontal (HFL):** mismas *features*, distintos *individuos* (dos hospitales con los mismos análisis y pacientes distintos).
   - **Vertical (VFL):** mismos *individuos*, distintas *features* (un banco y una telco con los mismos clientes y datos diferentes).
   - **Federated Transfer Learning:** poco solapamiento en ambos ejes.
   - **Cross-device** (millones de móviles poco fiables) vs **cross-silo** (pocas organizaciones grandes y estables → **el escenario natural de los data spaces**).
4. **Retos y contramedidas:**
   - **No-IID** (heterogeneidad estadística) → FedProx, SCAFFOLD, personalización.
   - **Privacidad residual** (los gradientes pueden filtrar datos) → privacidad diferencial (DP), agregación segura (SMPC), cifrado homomórfico.
   - **Comunicación** (subir/bajar modelos es caro) → compresión, cuantización, menos rondas.
   - **Clientes maliciosos** (envenenamiento) → agregación robusta (Krum, mediana).
   - **Heterogeneidad de sistemas** → selección de clientes, modelos adaptativos.
5. **FL ≠ privacidad perfecta por sí solo:** minimiza la exposición (los datos no viajan), pero necesita DP/agregación segura para garantías fuertes.

## Ruta mínima (en este orden)

- 🔗⭐ **Flower — *What is Federated Learning?*** (tutorial conceptual) · https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
  La mejor intuición de partida. Su resumen en una línea: *centralizado = lleva los datos al cómputo; federado = lleva el cómputo a los datos*. Sin prerrequisitos de FL.
- 🔗🧑‍💻⭐ **DeepLearning.AI × Flower Labs — *Intro to Federated Learning*** (curso corto, ~1–2 h) · https://www.deeplearning.ai/short-courses/intro-to-federated-learning
  Construyes un sistema FL con Flower + PyTorch y cubre privacidad diferencial y ancho de banda. La mejor vía práctica *guiada*. (Auditable / gratuito durante la beta de la plataforma.)
- 🧑‍💻⭐⭐ **Flower — *Quickstart PyTorch*** (hands-on real) · https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
  Entrena una CNN sobre CIFAR-10 con FedAvg y 10 nodos simulados. Aquí pasas de "entenderlo" a "ejecutarlo".

## Fundamento (papers de referencia)

- 📘⭐⭐ **FedAvg — McMahan et al. (2017): *Communication-Efficient Learning of Deep Networks from Decentralized Data*** · https://arxiv.org/abs/1602.05629
  El paper fundacional del FL. Léelo *después* del curso para fijar la matemática de la agregación y el porqué del ahorro de comunicación (10–100× menos rondas que el SGD sincronizado).
- 📘⭐⭐⭐ **Survey — Kairouz et al. (2021): *Advances and Open Problems in Federated Learning*** · https://arxiv.org/abs/1912.04977
  La referencia enciclopédica (no-IID, privacidad, VFL, retos abiertos). Consúltalo **por temas**, no de forma lineal.
  - 🔗⭐⭐ Si el survey se hace cuesta arriba, empieza por este resumen accesible de **OpenMined** · https://openmined.org/blog/advances-and-open-problems-in-federated-learning/

## Profundización opcional

- 🧑‍💻⭐⭐ **Flower — repositorio y ejemplos** (FedProx, FedBN para no-IID, DP...) · https://github.com/adap/flower
  Código real para experimentar con estrategias más allá de FedAvg.
- 🔗⭐⭐ **Frameworks alternativos:** TensorFlow Federated (https://www.tensorflow.org/federated) y FedML — útiles para comparar ecosistemas. Flower es el más recomendable para empezar por ser *framework-agnostic*.
- 🔗🧑‍💻⭐⭐ **DeepLearning.AI × Flower — *Federated Fine-tuning of LLMs with Private Data*** (parte 2 del curso) · https://www.deeplearning.ai/short-courses/intro-to-federated-learning
  Si te interesa el ángulo LLM: federar el *fine-tuning* con PEFT + DP. Conecta con el auto-supervisado del Módulo 2.2.
- Para profundizar en privacidad: **privacidad diferencial (DP)** y **agregación segura** (Bonawitz et al.) — ambas tratadas en el curso de DeepLearning.AI y en el survey.

## Checkpoint del módulo

Sabes describir una ronda de FedAvg de memoria, distingues HFL/VFL y cross-device/cross-silo, y **has entrenado un modelo federado simulado con Flower**. Entiendes por qué el FL minimiza pero no garantiza por sí solo la privacidad.

## Puente a los siguientes módulos

- El **cross-silo entre organizaciones** necesita una capa de gobernanza (contratos, soberanía del dato, conectores) → **Módulo 4: Data Spaces** (mapa mental §6).
- Federar **RL** en vez de aprendizaje supervisado (agregar políticas/Q-networks en lugar de clasificadores) → **Módulo 5: FRL** (mapa mental §7). Es decir: **FRL = Módulo 2.3 (RL) + Módulo 3 (federado)**.
