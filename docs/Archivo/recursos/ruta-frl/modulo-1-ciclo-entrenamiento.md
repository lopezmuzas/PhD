---
title: "Módulo 1 — El ciclo de entrenamiento de una red profunda"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 1 — El ciclo de entrenamiento de una red profunda

> 🧭 **Ruta FRL:** [Índice](index.md) · [0](modulo-0-cimientos-matematicos.md) · **1** · [2](modulo-2-paradigmas.md) · [3](modulo-3-aprendizaje-federado.md)
> 🗺️ **Mapa mental:** Sección 2.3.
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Objetivo:** interiorizar el bucle **forward → pérdida → backpropagation → optimización**, y la frase clave del documento: *"entrenar = calcular gradientes y actualizar pesos"* (que es justo lo que el FL acaba compartiendo en lugar de los datos).
**Prerrequisito:** [Módulo 0](modulo-0-cimientos-matematicos.md).

## Conceptos que debes dominar al salir

1. **Forward pass:** cómo la entrada atraviesa capas de transformaciones lineales + no linealidades (activaciones) hasta la predicción.
2. **Función de pérdida:** cómo se mide el error (entropía cruzada para clasificación, MSE para regresión).
3. **Backpropagation:** la regla de la cadena que reparte el error hacia atrás y calcula el gradiente respecto a cada peso.
4. **Optimización:** descenso por gradiente y sus variantes (SGD, Adam), tasa de aprendizaje, lotes (batches), épocas.
5. **Generalización:** overfitting/underfitting, train/val/test, regularización (enlaza con el Módulo 2).

## Ruta mínima (en este orden)

- 🎥⭐ **3Blue1Brown — *Neural Networks*** · https://www.3blue1brown.com/topics/neural-networks
  4 vídeos que explican, con intuición visual, qué es una red, qué es el descenso por gradiente y **qué es el backpropagation**. El mejor primer contacto. ~1 h.
- 🧑‍💻⭐⭐ **Andrej Karpathy — *Neural Networks: Zero to Hero*** (Lección 1: *building micrograd*) · https://karpathy.ai/zero-to-hero.html · código: https://github.com/karpathy/nn-zero-to-hero
  Construye un motor de *backpropagation* desde cero en Python, escalar a escalar. Tras esta lección, el backprop deja de ser magia. **Es el recurso decisivo de este módulo**: codifícalo tú mismo, no solo lo veas.
- 🔗⭐⭐ **Dive into Deep Learning (d2l.ai)** — capítulos *Linear Neural Networks* y *Multilayer Perceptrons* · https://d2l.ai/
  Libro interactivo y gratuito, con código ejecutable. Aquí pasas del escalar (micrograd) a las operaciones con tensores/lotes reales.

## Profundización opcional

- 🔗⭐⭐ **CS231n (Stanford) — notas de Backpropagation y Optimization** · https://cs231n.github.io/
  Las notas escritas más claras sobre el flujo de gradientes y los optimizadores. Ideal como referencia para consolidar.
- 📘⭐⭐⭐ **Deep Learning** — Goodfellow, Bengio, Courville (gratuito, HTML) · https://www.deeplearningbook.org/
  El libro de referencia teórico. Para este módulo: Cap. 6 (redes feedforward) y Cap. 8 (optimización). Denso; úsalo como consulta, no como lectura inicial.
- 🔗🧑‍💻⭐⭐ **fast.ai — Practical Deep Learning for Coders** · https://course.fast.ai/
  Enfoque *top-down* y muy práctico si prefieres entrenar modelos reales pronto y entender la teoría después.

## Checkpoint del módulo

Puedes describir el bucle de entrenamiento de memoria y, mejor aún, has reimplementado un micro-backprop que entrena. Entiendes por qué "lo que viaja en el FL son los gradientes/pesos".

➡️ Siguiente: [Módulo 2 — Los paradigmas de aprendizaje](modulo-2-paradigmas.md)
