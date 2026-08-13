---
title: "2. Aprendizaje Federado (FL)"
tags: [federated-learning, FedAvg, non-IID, privacidad, flower]
status: esbozo
---

# Aprendizaje Federado (Federated Learning)

> **Principio fundamental**: *centralizado = llevas los datos al cómputo; federado = llevas el cómputo a los datos.*

---

## El problema que resuelve

ML centralizado asume que tienes un dataset $\mathcal{D}$ en un servidor. FL asume que $\mathcal{D}$ está partido entre $K$ participantes, $\mathcal{D}_k$ vive en el participante $k$ y **no se mueve nunca**.

Matemáticamente, el objetivo es minimizar:

$$F(\theta) = \sum_{k=1}^K \frac{|\mathcal{D}_k|}{|\mathcal{D}|} F_k(\theta)$$

donde $F_k$ es el riesgo empírico local de cada participante.

---

## Tipos de FL

| Tipo | Qué se comparte | Ejemplo |
|---|---|---|
| **Horizontal (HFL)** | Mismas features, distintos individuos | 2 hospitales con los mismos campos de historial clínico y pacientes distintos |
| **Vertical (VFL)** | Mismos individuos, distintas features | Un banco y una aseguradora con los mismos clientes pero datos distintos |
| **Cross-device** | Millones de dispositivos, intermitentes, baja confianza | Teclado de Android (Gboard) |
| **Cross-silo** | Pocas organizaciones grandes, estables, alta confianza | Hospitales, energéticas → **el escenario natural de los Data Spaces** |

---

## FedAvg — el algoritmo fundacional

**McMahan et al. (2017)** — el paper que dio nombre al campo.

```
Para cada ronda t:
  1. Servidor envía θ_t a subconjunto S_t de clientes
  2. Cada cliente k ∈ S_t hace E epochs de SGD local → θ_t^k
  3. Servidor agrega: θ_{t+1} = Σ (n_k / n_S) · θ_t^k
  4. Repetir hasta convergencia
```

**Hiperparámetros clave**: `E` (epochs locales), `B` (batch size local), `C` (fracción de clientes por ronda), `η` (lr local).

### La trampa del non-IID

FedAvg asume datos IID entre clientes. En la práctica, **no lo son**. Hay 5 fuentes de heterogeneidad estadística:

1. **Label distribution skew** — cada cliente tiene distintos $P(y)$ (hospital pediátrico vs. geriátrico)
2. **Feature distribution skew** — $P(x)$ varía (rayos X de distintos fabricantes)
3. **Concept drift** — $P(y|x)$ varía (diagnósticos que difieren entre escuelas médicas)
4. **Covariate shift** — $P(x|y)$ varía
5. **Quantity skew** — algunos tienen miles de muestras, otros decenas

**Consecuencia**: en non-IID, las actualizaciones locales divergen de la dirección óptima global → *client drift*.

---

## La familia de fixes para non-IID

| Algoritmo | Qué corrige | Cómo | Coste extra |
|---|---|---|---|
| **FedProx** | Client drift moderado | Término proximal $\frac{\mu}{2}\|\theta_k - \theta_{global}\|^2$ | ~5 líneas de código |
| **SCAFFOLD** | Client drift severo | Control variates por cliente | 2× comunicación |
| **FedNova** | Heterogeneidad de compute | Normaliza steps locales entre clientes | Bajo |
| **FedOpt/FedAdam** | Convergencia lenta | Adam en el servidor sobre pseudo-gradiente | Solo lado servidor |

**Regla heurística**: non-IID moderado → FedProx. non-IID severo → SCAFFOLD. Clientes con compute muy distinto → FedNova.

---

## Privacidad: por qué FL solo no garantiza nada

FL minimiza la exposición (los datos raw no viajan), pero los gradientes **filtran información**:

- **Gradient inversion (Zhu et al. 2019 — DLG)**: recuperan imágenes pixel a pixel desde gradientes compartidos.
- **Membership inference**: infieren si un dato concreto estaba en el dataset de entrenamiento.
- **Backdoor poisoning**: cliente malicioso introduce un trigger oculto en el modelo global.

### Defensas

| Defensa | Garantía | Dónde aplica |
|---|---|---|
| **DP-SGD** (Abadi 2016) | Formal matemática $(ε, δ)$-DP | Contra gradient inversion y membership inference |
| **Secure Aggregation** (Bonawitz 2017) | El servidor solo ve la suma, nunca updates individuales | Contra servidor curioso |
| **Krum / Median / Trimmed Mean** | Byzantine-robust | Contra clientes maliciosos (poisoning) |

**Receta estándar de producción**: DP + Secure Aggregation combinados.

---

## Personalización: cuando un modelo global no funciona

En entornos muy heterogéneos (el caso de Data Spaces), puede ser mejor tener **un modelo por participante** que comparte conocimiento sin converger a uno único.

| Familia | Cómo | Ejemplo |
|---|---|---|
| **Fine-tuning local** | FedAvg global + fine-tune local | La baseline trivial, sorprendentemente competitiva |
| **Meta-learning (Per-FedAvg)** | MAML federado — modelo global óptimo como punto de partida | Conexión directa con FRL personalizado |
| **Multi-task (Ditto, pFedMe)** | Loss conjunta global + local | Equilibrio entre generalización y especificidad |
| **Backbone compartido** | Compartir capas base, personalizar head | FedPer — ideal para modelos grandes |

---

## Herramientas

- **[Flower](https://flower.ai/docs/)** — el framework de referencia en investigación. Soporta PyTorch, TF, JAX. Simulación + deployment real con la misma API.
- **[Opacus](https://opacus.ai/)** — DP-SGD en PyTorch. Integración nativa con Flower.
- **[Flower Baselines](https://github.com/flwrlabs/flower)** — implementaciones reproducibles de FedAvg, FedProx, SCAFFOLD, etc. El equivalente a CORL para FL.

---

## Referencias fundamentales

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Kairouz et al. (2021). *Advances and Open Problems in Federated Learning*. [arXiv:1912.04977](https://arxiv.org/abs/1912.04977)
- Abadi et al. (2016). *Deep Learning with Differential Privacy*. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- Bonawitz et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning*. [eprint](https://eprint.iacr.org/2017/281)

<!-- nav-start -->

---

← Anterior: [1. Introducción](../01-introduccion/index.md)  
Siguiente: [3. Data Spaces y Gobernanza](../03-data-spaces-gobernanza/index.md) →

<!-- nav-end -->
