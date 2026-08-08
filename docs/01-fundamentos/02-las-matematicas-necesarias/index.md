---
title: "2. Las matemáticas necesarias"
tags: [matematicas]
status: esbozo
---

# 2. Las matemáticas necesarias

No se estudian enteras antes de empezar: **se vuelve a ellas cada vez que hagan
falta**. Esta parte lista lo que de verdad se usa, no el temario completo de
cada asignatura.

| Sección | Para qué sirve en la práctica |
|---|---|
| [2.1 Álgebra lineal](2.1-algebra-lineal.md) | Una capa de red *es* una transformación lineal |
| [2.2 Derivadas](2.2-derivadas-y-regla-de-la-cadena.md) | Backpropagation *es* la regla de la cadena |
| [2.3 Probabilidad](2.3-probabilidad.md) | Toda función de pérdida es una verosimilitud |
| [2.4 Optimización](2.4-optimizacion.md) | Cómo se convierte el gradiente en aprendizaje |

**Atajo honesto:** con `autodiff` + `máxima verosimilitud` + `SVD` se explica el
80 % de lo que ocurre dentro de una red. El resto es ingeniería y sesgo
inductivo.
