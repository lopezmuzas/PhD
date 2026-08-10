---
title: "3. Cómo se entrena una red"
tags: [entrenamiento, indice]
status: esbozo
---

# 3. Cómo se entrena una red

La maquinaria que comparten **todas** las redes, sea cual sea la arquitectura y
la forma de aprender. Es lo que se repite en cada proyecto y donde se pierde la
mayor parte del tiempo cuando algo no funciona.

> **Qué NO cubre este módulo.** Las arquitecturas concretas (MLP, CNN,
> Transformer) van en el módulo 4. De dónde sale la señal de aprendizaje
> (supervisado, auto-supervisado, refuerzo) está en → 1.5, eje ①. Aquí se
> documenta solo la maquinaria común.

---

## El módulo en cuatro fases

```none
┌─ PREPARAR ───────────────────────────────────────────┐
│  3.0  Los datos, antes del modelo                    │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ EJECUTAR ───────────────────────────────────────────┐
│  3.1  El ciclo de entrenamiento                      │
│  3.2  Backpropagation                                │
│  3.3  Pérdidas y optimizadores                       │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ CONTROLAR ──────────────────────────────────────────┐
│  3.4  Regularización                                 │
│  3.5  Precisión numérica y memoria                   │
│  3.6  Escalar: de una GPU a muchas                   │
│  3.7  Transferencia y fine-tuning                    │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ VERIFICAR ──────────────────────────────────────────┐
│  3.8  Problemas típicos                              │
│  3.9  Cómo medir si funciona                         │
└──────────────────────────────────────────────────────┘
```

---

## Índice

### Preparar

| Sección | Qué responde |
|---|---|
| [3.0](3.0-los-datos.md) | ¿Qué hay que hacer con los datos **antes** de escribir una sola línea de modelo? |

### Ejecutar

| Sección | Qué responde |
|---|---|
| [3.1](3.1-el-ciclo-de-entrenamiento.md) | ¿Cuál es el bucle básico? |
| [3.2](3.2-backpropagation.md) | ¿Cómo se reparte la culpa del error? |
| [3.3](3.3-perdidas-y-optimizadores.md) | ¿Cómo se mide el error y cómo se corrige? |

### Controlar

| Sección | Qué responde |
|---|---|
| [3.4](3.4-regularizacion.md) | ¿Qué solución encuentra, entre todas las que ajustan? |
| [3.5](3.5-precision-y-memoria.md) | ¿Por qué no cabe, por qué va lento y por qué da NaN? |
| [3.6](3.6-escalar-el-entrenamiento.md) | ¿Qué cambia al pasar de una GPU a muchas? |
| [3.7](3.7-transferencia-y-finetuning.md) | ¿Por qué casi nadie entrena desde cero? |

### Verificar

| Sección | Qué responde |
|---|---|
| [3.8](3.8-problemas-tipicos.md) | ¿Qué falla y cómo se detecta? |
| [3.9](3.9-como-medir-si-funciona.md) | ¿Cómo sé que de verdad funciona? |

---

## Orden de lectura

**Primera pasada (entender):** 3.1 → 3.2 → 3.3 → 3.4. Es la columna vertebral
conceptual, y se puede leer seguida.

**Segunda pasada (hacer):** 3.0 → 3.8 → 3.9. Son las tres que evitan perder
semanas, y solo tienen sentido cuando ya has intentado entrenar algo.

**Cuando lo necesites:** 3.5, 3.6 y 3.7 son de consulta. No se leen enteras, se
abren cuando el modelo no cabe, cuando hay más de una GPU, o cuando parte de un
preentrenado.

---

## Las dependencias que importan

Tres hilos atraviesan varias secciones. Conviene tenerlos identificados porque
son la fuente habitual de conclusiones falsas:

| Hilo | Dónde aparece | Por qué importa |
|---|---|---|
| **Lote ↔ tasa de aprendizaje** | 3.1, 3.3, 3.4 | Son dos mandos del mismo dial. Comparar tamaños de lote sin re-ajustar la tasa no mide lo que crees |
| **Normalización ↔ estabilidad ↔ warmup** | 3.4, 3.5, 3.8 | La necesidad de warmup en Transformers viene de dónde pusiste la normalización, no del optimizador |
| **Presupuesto de ajuste ↔ validez de la comparación** | 3.3, 3.8, 3.9 | Casi todas las "mejoras" que no se reproducen vienen de presupuestos desiguales |

---

## Enlaces con el resto de la documentación

- **→ 1.5** — Las 3 decisiones al diseñar un sistema de IA. Este módulo es
  transversal a los tres ejes: la maquinaria es la misma se aprenda como se
  aprenda y estén los datos donde estén.
- **→ 1.6** — RL Offline. Rompe el supuesto de que existe un conjunto de
  validación, y por tanto rompe buena parte de 3.9.
- **→ 2.3** — Verosimilitud y priors. Da el fundamento de las pérdidas (3.3) y del
  weight decay (3.4).
- **→ 2.4** — El ruido del SGD como regularizador implícito (3.4).
- **→ Módulo 4** — Arquitecturas. La normalización y las conexiones residuales
  aparecen aquí como maquinaria y allí como diseño.