---
title: "Módulo 0 — Cimientos matemáticos: álgebra lineal (y el puente de cálculo)"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 0 — Cimientos matemáticos: álgebra lineal (y el puente de cálculo)

> 🧭 **Ruta FRL:** [Índice](index.md) · **0** · [1](modulo-1-ciclo-entrenamiento.md) · [2](modulo-2-paradigmas.md) · [3](modulo-3-aprendizaje-federado.md)
> 🗺️ **Mapa mental:** prerrequisito de la Sección 2.3.
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Por qué primero:** el ciclo de entrenamiento del Módulo 1 *es* álgebra lineal en movimiento. Una capa de una red es una multiplicación matriz-vector `Wx + b`; el *backpropagation* es la **regla de la cadena** aplicada sobre esas operaciones. Sin estos cimientos, el Módulo 1 se memoriza pero no se entiende.

## Lo que de verdad necesitas (no hace falta un curso entero)

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

## Ruta mínima

- 🎥⭐ **3Blue1Brown — *Essence of Linear Algebra*** · https://www.3blue1brown.com/topics/linear-algebra
  La mejor intuición *geométrica* de vectores, matrices, multiplicación como composición de transformaciones, y eigenvectores. ~3 h en total. Imprescindible para "ver" lo que hace una capa.
- 🎥⭐ **3Blue1Brown — *Essence of Calculus*** (solo capítulos de derivada y **regla de la cadena**) · https://www.3blue1brown.com/topics/calculus
  Con esto cubres el puente de cálculo necesario para el backprop.

## Profundización opcional (referencia, no lectura lineal)

- 📘⭐⭐ **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (gratuito, oficial) · web: https://mml-book.com/ · PDF: https://mml-book.github.io/book/mml-book.pdf
  El libro de referencia. Para esta ruta basta el **Cap. 2 (Linear Algebra)**, **Cap. 3 (Analytic Geometry: normas, productos internos)** y **Cap. 5 (Vector Calculus: gradientes, regla de la cadena)**. Los Cap. 4 (descomposiciones, SVD) y 6 (probabilidad) son útiles más adelante.
- 🎥🔗⭐⭐ **Gilbert Strang — MIT 18.06 Linear Algebra** (MIT OpenCourseWare, gratuito) · https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
  El curso clásico si quieres una base sólida y formal. Es más de lo que necesitas para arrancar, pero es la referencia.

## Checkpoint del módulo

Sabes explicar, sin mirar, qué hace `Wx + b` geométricamente y por qué la regla de la cadena permite calcular cómo afecta cada peso al error.

➡️ Siguiente: [Módulo 1 — El ciclo de entrenamiento](modulo-1-ciclo-entrenamiento.md)
