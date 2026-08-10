---
title: "2. Las matemáticas necesarias"
tags: [matematicas, indice]
status: esbozo
---

# 2. Las matemáticas necesarias

No se estudian enteras antes de empezar: **se vuelve a ellas cada vez que hagan
falta**. Esta parte lista lo que de verdad se usa, no el temario completo de
cada asignatura.

Por eso este módulo se lee **al revés que los demás**: no de principio a fin,
sino entrando por el problema concreto que te ha bloqueado. La tabla de
"¿atascado en…?" del final es la puerta principal.

---

## Índice

### Preliminar

| Sección | Para qué sirve en la práctica |
|---|---|
| [2.0 Notación y formas](2.0-notacion-y-formas.md) | El 80 % de los errores de un principiante son de dimensiones, no de matemáticas |

### El núcleo

| Sección | Para qué sirve en la práctica |
|---|---|
| [2.1 Álgebra lineal](2.1-algebra-lineal.md) | Una capa de red *es* una transformación lineal |
| [2.2 Derivadas](2.2-derivadas-y-regla-de-la-cadena.md) | Backpropagation *es* la regla de la cadena |
| [2.3 Probabilidad](2.3-probabilidad.md) | Toda función de pérdida es una verosimilitud |
| [2.4 Optimización](2.4-optimizacion.md) | Cómo se convierte el gradiente en aprendizaje |

### El resto de lo que se usa a diario

| Sección | Para qué sirve en la práctica |
|---|---|
| [2.5 Teoría de la información](2.5-teoria-de-la-informacion.md) | La entropía cruzada, la KL y las pérdidas contrastivas salen todas de aquí |
| [2.6 Estadística e inferencia](2.6-estadistica-e-inferencia.md) | Distinguir un resultado de una anécdota |
| [2.7 Geometría en alta dimensión](2.7-geometria-alta-dimension.md) | Por qué los embeddings funcionan y por qué la intuición 3D te engaña |
| [2.8 Procesos de decisión de Markov](2.8-mdp-y-bellman.md) | El formalismo del refuerzo, y por qué el operador de Bellman converge |

---

## Atajo honesto

Con **autodiff + máxima verosimilitud + SVD** se explica el 80 % de lo que
ocurre dentro de una red. El resto es ingeniería y sesgo inductivo.

Ampliado a la lista completa, el mínimo real que hay que dominar es:

```none
① Multiplicación matriz-vector y qué le hace al espacio     → 2.1
② Regla de la cadena sobre un grafo de cómputo              → 2.2
③ Máxima verosimilitud: por qué la pérdida es la que es     → 2.3
④ Descenso de gradiente y por qué el ruido ayuda            → 2.4
⑤ Entropía cruzada y KL, entendidas y no memorizadas        → 2.5
⑥ Intervalos de confianza y bootstrap                       → 2.6
```

Seis ideas. Todo lo demás es consulta.

---

## Lo que NO hace falta

Decirlo explícitamente ahorra meses, porque el instinto del autodidacta es
empezar por los cimientos formales y no llegar nunca:

| No hace falta | Por qué |
|---|---|
| Teoría de la medida | La probabilidad que usas es discreta o con densidades bien portadas |
| Demostraciones de convergencia | Casi ninguna aplica al caso no convexo real |
| Análisis real riguroso | Las funciones con las que trabajas son diferenciables casi en todas partes, y `autograd` se encarga |
| Cálculo integral avanzado | Salvo en modelos generativos, casi nunca integras a mano |
| Álgebra abstracta | Aparece en redes equivariantes; ni antes ni fuera de ahí |
| Calcular gradientes a mano | **Una vez**, para entenderlo (→ 3.2). Después, nunca más |

La regla: si llevas dos semanas estudiando matemáticas sin entrenar nada, te has
desviado.

---

## Cuándo volver a cada sección

La forma real de usar este módulo. Entra por la fila que describa tu bloqueo:

| ¿Atascado en…? | Vuelve a |
|---|---|
| Un error de dimensiones que no entiendes | **2.0** |
| Por qué una red sin no-linealidad colapsa a una capa | **2.1** |
| Qué es un embedding y por qué se pueden sumar | 2.1, 2.7 |
| Qué hace realmente `.backward()` (→ 3.2) | **2.2** |
| Por qué esta pérdida y no otra (→ 3.3) | 2.3, **2.5** |
| Por qué weight decay ≡ prior gaussiano (→ 3.4) | **2.3** |
| Momento, Adam, tasa de aprendizaje (→ 3.3) | **2.4** |
| Por qué el ruido del SGD regulariza (→ 3.4) | **2.4** |
| La ELBO de un VAE, InfoNCE, destilación (→ 3.3) | **2.5** |
| Cuántas semillas hacen falta; intervalos (→ 3.9) | **2.6** |
| Calibración y qué significa una probabilidad (→ 3.9) | 2.3, **2.6** |
| Por qué la similitud coseno y no la euclídea | **2.7** |
| Por qué en alta dimensión "todo está lejos de todo" | **2.7** |
| Ecuación de Bellman, factor de descuento, política | **2.8** |
| Por qué el error de extrapolación se propaga (→ 1.6) | 2.8, 2.4 |

---

## Recursos generales del módulo

Los específicos van en cada sección. Estos cubren varias a la vez:

| Recurso | Formato | Nota |
|---|---|---|
| [3Blue1Brown — Álgebra lineal](https://www.3blue1brown.com/topics/linear-algebra) y [Cálculo](https://www.3blue1brown.com/topics/calculus) | 📺🎮 | **El punto de partida para 2.1 y 2.2.** La intuición geométrica que ningún libro transmite. El capítulo 3 de álgebra lineal (matriz como transformación) es el que más rinde |
| [Mathematics for Machine Learning](https://mml-book.github.io) (Deisenroth, Faisal & Ong) | 📖 | **Gratuito.** El libro que cubre exactamente el recorte de este módulo: álgebra, cálculo matricial, probabilidad y optimización, con el ML como destino explícito |
| [Immersive Linear Algebra](http://immersivemath.com/ila/) | 🎮 | Libro de álgebra lineal con **todas** las figuras interactivas |
| [Seeing Theory](https://seeing-theory.brown.edu) (Brown) | 🎮 | Probabilidad, estimación e inferencia, visuales. Cubre 2.3 y 2.6 |
| [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) | 📄 | Formulario de derivadas matriciales. No se lee: se consulta cuando un paper te deja tirado |
| [Dive into Deep Learning — Apéndice matemático](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/) | 📖🛠️ | El mismo recorte, pero con código ejecutable al lado |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Simon Prince) | 📖 | Gratuito. Sus apéndices matemáticos son cortos y están escritos para llegar al deep learning, no para un curso |