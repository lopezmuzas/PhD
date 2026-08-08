---
title: Cómo está organizado esto
tags: [meta]
status: revisado
---

# Cómo está organizado esto

## El criterio de las cuatro carpetas

Está basado en [Diátaxis](https://diataxis.fr/). La clasificación no va por
tema sino por **la situación del lector**:

| Carpeta | Pregunta que responde | Modo del lector |
|---|---|---|
| `aprendizaje/` | ¿Por qué funciona esto? | Estudiando |
| `tesis/` | ¿Qué voy a investigar y por dónde? | Planificando |
| `guias/` | ¿Cómo hago X? | Trabajando |
| `referencia/` | ¿Cuál era el parámetro? | Consultando |
| `recursos/` | ¿Dónde leo más? | Explorando |
| `bitacora/` | ¿Qué hice el martes? | Recordando |

`aprendizaje/` y `tesis/` se confunden con facilidad. El criterio: si el
documento **explica** algo que ya es conocimiento establecido, va en
`aprendizaje/`; si **decide** algo sobre tu investigación (qué leer, qué
hueco atacar, qué construir), va en `tesis/`.

La documentación se vuelve ilegible cuando un tutorial y una referencia pelean
dentro del mismo fichero. Si una nota tuya hace dos cosas, pártela.

## Frontmatter obligatorio

Cada nota empieza con el bloque YAML de `meta/plantilla-nota.md`. Los campos
que hacen trabajo real:

- **`tags`** — alimenta el [índice de etiquetas](tags.md), que se genera solo.
  Es la navegación transversal: un tema puede vivir en una carpeta y aparecer
  bajo tres etiquetas.
- **`status`** — `borrador` señala lo que escribiste rápido y nunca revisaste.
  A los seis meses es la diferencia entre confiar en una nota o no.

## Nada de metadatos en el nombre del fichero

`2026-08-06-batchnorm-v2-REVISADO.md` es un frontmatter mal puesto. El nombre
del fichero es la URL: en minúsculas, con guiones, estable. Si renombras,
añade una redirección en `mkdocs.yml`.

## Bitácora

Un fichero por sesión de trabajo, `bitacora/2026-08-06.md`. No es documentación,
es un registro: qué probaste, qué falló, qué decidiste. Se escribe mal y rápido
a propósito. Cuando una entrada de bitácora se vuelve útil para tu yo futuro,
promuévela a `aprendizaje/` reescribiéndola en condiciones.

## Subcarpetas de `aprendizaje/`

Dentro de `aprendizaje/` la organización sí es temática, y sigue el
[índice maestro](../aprendizaje/00-mapa/indice-maestro.md). El prefijo numérico
no es un orden de lectura obligatorio: es el orden de dependencia conceptual.

| Carpeta | Partes del índice | Qué contiene |
|---|---|---|
| `00-mapa/` | 0 | El índice maestro y los modelos mentales generales |
| `01-fundamentos/` | 1–2 | Neurona biológica, matemáticas |
| `02-redes-neuronales/` | 3–9 | Anatomía, entrenamiento, arquitecturas, escala |
| `03-refuerzo/` | 10 | RL tabular, deep RL, online, offline |
| `04-federado-y-distribuido/` | 11 | FL, FRL, data spaces |

Cuando una carpeta pase de ~10 ficheros, pártela siguiendo la siguiente
subdivisión del índice. No la partas antes: una carpeta con dos ficheros es
peor que una lista.

## `_archivo/`

Versiones antiguas que un documento vigente ha sustituido. Se conservan por si
contienen algo que se perdió en la reescritura, pero están fuera de la
navegación (`hide: true` en su `.pages`). Si a los seis meses no has vuelto a
mirar una, bórrala.
