---
title: "13. Herramientas y variantes"
tags: [federado, herramientas, indice, referencia]
status: esbozo
---

# 13. Herramientas y variantes

El mapa del terreno para construir un sistema con **redes profundas o
transformers** sobre datos federados en C2D. No es un laboratorio: es material
de consulta.

> **Requisito previo.** → [11](../11-federado-sobre-data-spaces/11-indice.md) y
> → [12](../12-federado-con-felt-labs/12-indice.md). Esta sección asume que ya
> tienes un pipeline funcionando y que el problema ahora es escalarlo a modelos
> de verdad.

---

## La idea que ordena la sección

```none
┌──────────────────────────────────────────────────────────────┐
│  ALGORITMO      FedAvg, FedProx, LoRA federado, one-shot…     │
│                 ✅ TE SIRVE TODO, TAL CUAL                     │
├──────────────────────────────────────────────────────────────┤
│  TRANSPORTE     gRPC, sockets, conexiones persistentes        │
│                 ❌ NADA FUNCIONA DENTRO DE C2D                 │
└──────────────────────────────────────────────────────────────┘
```

Los frameworks se usan para **la capa de algoritmo y para simular**. El
transporte lo pone tu adaptador de ficheros. Toda la sección se deriva de ahí.

---

## Índice

### Orientarse

| Sección | Qué responde |
|---|---|
| [13.0 Por qué existe esta sección](13.0-por-que-existe-esta-seccion.md) | ¿Qué falta para pasar de modelos convexos a redes profundas? **Empieza aquí** |
| [13.1 El mapa de decisiones](13.1-el-mapa-de-decisiones.md) | Las seis decisiones, y cuáles cierra C2D |

### Elegir

| Sección | Qué responde |
|---|---|
| [13.2 Frameworks](13.2-frameworks.md) | Flower, FLARE, OpenFL, Substra: cuál y para qué exactamente |
| [13.3 Variantes de agregación](13.3-variantes-de-agregacion.md) | Mis silos son muy distintos y converge mal. ¿Qué agregador? |
| [13.4 Transformers y LoRA federado](13.4-transformers-y-lora-federado.md) | Ajustar un transformer sin morir de comunicación |
| [13.5 One-shot federated learning](13.5-one-shot-federated-learning.md) | ¿Puedo hacerlo en una sola ronda? **La página con más potencial para la tesis** |

### Capas transversales

| Sección | Qué responde |
|---|---|
| [13.6 Privacidad y seguridad](13.6-privacidad-y-seguridad.md) | La escalera de garantías, y qué cuesta cada peldaño |
| [13.7 La capa de data space](13.7-la-capa-de-data-space.md) | Gaia-X, EDC, Apheris: qué copiar de quien ya lo hizo |

### Decidir

| Sección | Qué responde |
|---|---|
| [13.8 La pila recomendada](13.8-la-pila-recomendada.md) | **Si solo lees una página, esta.** Qué montar y en qué orden |
| [13.9 Recursos por formato](13.9-recursos-por-formato.md) | Lo mismo, ordenado por interactivo / vídeo / código / paper |

---

## Los tres huecos de investigación

De todo lo recorrido en las secciones 11 a 13, desarrollados en → [13.8](13.8-la-pila-recomendada.md):

| # | Hueco | Por qué es tuyo |
|---|---|---|
| ① | **¿Cuándo basta una ronda?** | Ya tienes medio resultado empírico y toda la infraestructura |
| ② | Agregación robusta **con** cegado | Son incompatibles de forma ingenua. Difícil, y toca criptografía |
| ③ | Promediar funciones $Q$ sobre soportes distintos | La pregunta de → 1.6, aún abierta. El núcleo de la tesis |

---

## Si tienes poco tiempo

| Tiempo | Ruta |
|---|---|
| **1 hora** | [Cómic de Google](https://federated.withgoogle.com/) → [explorable de PAIR](https://pair.withgoogle.com/explorables/federated-learning/) |
| **1 tarde** | 13.0 → 13.8 |
| **1 semana** | 13.0 → 13.1 → 13.3 → 13.4 → 13.5 → 13.8 |
| **Solo quiero código** | [13.9, sección 🛠️](13.9-recursos-por-formato.md) |

---

## Nota sobre el formato de los recursos

Cada página termina con recursos etiquetados 🎮 interactivo · 🎥 vídeo ·
🛠️ ejecutable · 📖 divulgativo · 📄 paper, en ese orden de prioridad.

**Cuando un tema no tiene material interactivo ni divulgativo, se dice.** Pasa en
one-shot FL, en LoRA federado y en agregación robusta: son demasiado recientes o
demasiado específicos. Fingir lo contrario haría perder el tiempo.

---

## Enlaces con el resto del libro

- **→ 1.5** — Las 3 decisiones. Las seis de → 13.1 son su refinamiento para el eje ③.
- **→ 1.6** — RL Offline. El hueco ③ sale de ahí.
- **→ 3.3** — Optimizadores. FedAdam y FedYogi son sus primos federados.
- **→ 3.6** — Escalar el entrenamiento. FedAvg es paralelismo de datos con
  sincronización infrecuente.
- **→ 3.7** — Transferencia y fine-tuning. → 13.4 es su capa federada.
- **→ 3.9** — Cómo medir. Nada de → 13.3 se decide sin intervalos y semillas.
- **→ 11**, **→ 12** — Los dos pipelines que esta sección enseña a escalar.
