---
title: "Módulo 5 — Federated Reinforcement Learning (FRL): el objetivo final"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 5 — Federated Reinforcement Learning (FRL): el objetivo final

> 🧭 **Ruta FRL:** [Índice](index.md) · [0](modulo-0-cimientos-matematicos.md) · [1](modulo-1-ciclo-entrenamiento.md) · [2](modulo-2-paradigmas.md) · [3](modulo-3-aprendizaje-federado.md) · [4](modulo-4-data-spaces.md) · **5**
> 🗺️ **Mapa mental:** Sección 7.
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Objetivo:** unir los dos ejes. El **FRL = aprendizaje por refuerzo (eje ① CÓMO)** entrenado de forma **federada (eje ③ DÓNDE)**, idealmente gobernado por un **data space** (Módulo 4). Aquí no agregas clasificadores: agregas **agentes** (políticas y funciones de valor).
**Prerrequisitos imprescindibles:** [Módulo 2.3 (RL)](modulo-2-paradigmas.md) **y** [Módulo 3 (FL)](modulo-3-aprendizaje-federado.md). El Módulo 4 aporta el contexto de gobernanza.

> ⚠️ **Cambio de madurez:** el FRL es un campo **de investigación reciente**, no un producto con curso "llave en mano". Por eso este módulo se apoya en *surveys* y papers, y la práctica consiste en **combinar tú mismo** el RL del Módulo 2.3 con Flower del Módulo 3.

## Conceptos que debes dominar al salir

1. **Qué cambia respecto al FL supervisado:** en FL clásico se agregan modelos que predicen **etiquetas**; en FRL se agregan **políticas π(a|s)** o **funciones de valor (Q-networks)** aprendidas por **interacción**. La ronda de FedAvg sigue igual, pero el objeto que viaja y se promedia es un **agente**.
2. **El mapeo conceptual (survey de Qi et al.):** las tres dimensiones del FL —muestra, *feature*, *label*— se corresponden en FRL con **entorno, estado y acción**. Esto te deja reutilizar toda la intuición del Módulo 3.
3. **HFRL vs VFRL** (espejo del HFL/VFL del Módulo 3):
   - **HFRL (horizontal):** muchos agentes en entornos **parecidos** con experiencias propias, que colaboran para aprender una política mejor o común (flotas de robots, coches, dispositivos *edge* ante tareas similares). Es la mayoría de los trabajos.
   - **VFRL (vertical):** agentes que observan **partes distintas del mismo entorno** (observaciones parciales complementarias) y colaboran sin compartirlas. El *FedRL* de Zhuo et al. es el ejemplo seminal.
4. **Retos propios (FL + RL se suman):** heterogeneidad de entornos/recompensas (el no-IID llevado al RL); **inestabilidad** (el RL ya es inestable; federarlo agrava el "qué y cuándo agregar"); exploración distribuida y eficiencia de muestras; privacidad de trayectorias/políticas (DP, agregación segura); comunicación en *edge* con poca banda.
5. **RL offline dentro del FRL:** en dominios regulados (sanidad, finanzas, industria) **no se puede explorar en vivo**; cada participante aporta su **dataset histórico** y se federa el aprendizaje **offline** (CQL del Módulo 2.3). Es el escenario más realista para el **cross-silo + data space**.
6. **Aplicaciones:** *edge*/IoT, control y optimización, comunicaciones, robótica (navegación), conducción autónoma, sanidad.

## Ruta mínima (en este orden)

> **Antes de empezar:** confirma que tienes hechos el **Módulo 2.3 (RL)** y el **Módulo 3 (FL)**. El FRL no se entiende sin ambos.

- 📘⭐⭐ **Survey — Qi, Zhou, Lei, Zheng (2021): *Federated Reinforcement Learning: Techniques, Applications, and Open Challenges*** · https://arxiv.org/abs/2108.11887
  **La puerta de entrada.** Empieza por su tutorial conjunto de FL+RL y por la taxonomía **HFRL/VFRL**, que encaja directamente con el HFL/VFL que ya viste. Resume además los trabajos por área de aplicación.
- 📘⭐⭐ **Paper fundacional — Zhuo, Feng, Lin, Xu, Yang (2019): *Federated Deep Reinforcement Learning (FedRL)*** · https://arxiv.org/abs/1901.08277
  El origen del término. Caso de **VFRL** con observaciones parciales y privacidad mediante diferenciales gaussianas. Léelo después del survey para ver la formulación concreta.
- 🧑‍💻⭐⭐⭐ **Práctica: constrúyelo combinando lo que ya sabes** (no hay framework dominante de FRL).
  Toma un agente del Módulo 2.3 (p. ej. DQN o PPO con Stable-Baselines3 o el curso de Hugging Face) y **federa su entrenamiento con Flower** (Módulo 3): en lugar de promediar los pesos de un clasificador, promedia los de tu **red de política / Q-network**. Punto de partida concreto: el *Quickstart PyTorch* de Flower (https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html), sustituyendo el modelo CNN por tu red de RL.

## Profundización opcional

- 📘⭐⭐ **Aplicación en robótica — Liu et al. (2019): *Lifelong Federated Reinforcement Learning*** (navegación en sistemas de robots *cloud*) · https://arxiv.org/abs/1901.06455
  Buen ejemplo de **HFRL** aplicado y de cómo el aprendizaje continuo (Módulo 3.7 del mapa mental) entra en juego.
- 🔗⭐⭐ **RL offline (CQL) antes de federarlo:** repasa *Conservative Q-Learning* desde la literatura del Módulo 2.3. Es el ingrediente que hace realista el FRL **cross-silo** en dominios donde explorar en vivo es inaceptable.
- 🔭 **Dirección emergente — *federated RLHF* / *federated agent RL* para LLMs:** combina el RLHF del Módulo 2.4 con el federado del Módulo 3 (afinar LLMs con preferencias privadas distribuidas). Área muy nueva (2024–2025); busca los términos "FedRLHF" y "federated RLHF" para los últimos trabajos.

## Checkpoint final

Sabes explicar **qué se agrega** en FRL (políticas/Q-networks, no clasificadores), distingues **HFRL/VFRL** y los mapeas con HFL/VFL, entiendes por qué el **RL offline** encaja con el cross-silo regulado, y podrías esbozar un prototipo combinando un agente de RL con Flower.

## Cierre de la ruta — cómo encaja todo

```
Módulo 0 (álgebra lineal)
   └─► Módulo 1 (ciclo de entrenamiento)
          ├─► Módulo 2 (paradigmas: ... RL en 2.3) ─┐
          └─► Módulo 3 (federado: FedAvg) ──────────┤
                 └─► Módulo 4 (data spaces) ─────────┤
                                                     ▼
                                          Módulo 5 — FRL
              = RL (2.3) entrenado de forma FEDERADA (3),
                gobernado por un DATA SPACE (4),
                sobre los cimientos de 0–1.
```

Has recorrido los **tres ejes** del mapa mental hasta su intersección. El siguiente paso natural ya no es teoría, sino **elegir un caso de uso** (sanidad, energía, movilidad...) y prototiparlo: un agente de RL + Flower + el marco de gobernanza de un data space.

🎯 **Ruta completa.** El siguiente entregable previsto es el **PDF consolidado** con el índice y los seis módulos (0–5).
