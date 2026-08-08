---
title: Batch Normalization
tags: [teoria, regularizacion, arquitectura]
status: revisado
updated: 2026-08-06
fuentes:
  - https://arxiv.org/abs/1502.03167
  - https://arxiv.org/abs/1805.11604
---

# Batch Normalization

!!! abstract "En una frase"
    Normaliza las activaciones de cada capa usando la media y varianza del
    minibatch, lo que permite tasas de aprendizaje más altas y estabiliza
    el entrenamiento de redes profundas.

## Formulación

Para un minibatch $\mathcal{B} = \{x_1, \dots, x_m\}$:

$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i
\qquad
\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}
\qquad
y_i = \gamma \hat{x}_i + \beta
$$

Los parámetros $\gamma$ y $\beta$ se aprenden: la red puede deshacer la
normalización si le conviene.

## El matiz que importa

La explicación original —reducir el *internal covariate shift*— fue cuestionada
por Santurkar et al. (2018), que argumentan que el efecto real es suavizar el
paisaje de optimización. La técnica funciona; la explicación de por qué
funciona sigue abierta.

## En la práctica

!!! warning "Train vs eval"
    En inferencia se usan medias móviles, no las del batch. Olvidar
    `model.eval()` es la causa número uno de "mi accuracy de validación es
    absurda". Ver [Depurar un entrenamiento](../../guias/depurar-entrenamiento.md).

- Con batches pequeños (< 16) degrada: considera `GroupNorm` o `LayerNorm`.
- En Transformers domina `LayerNorm`, que no depende del batch.

## Preguntas abiertas

- [ ] ¿Cómo interactúa con dropout cuando van en la misma capa?
- [ ] Medir el efecto real en `labs/dl-lab` con y sin BN a igualdad de épocas.
