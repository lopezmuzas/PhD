---
title: "5. Tipos de modelo"
tags: [modelos, indice]
status: completo
eje: "¿con qué se implementa el sistema?"
---

# 5. Tipos de modelo

> **Pregunta 2 de 3:** ¿con qué se implementa el sistema?

El capítulo 4 respondía de dónde sale la señal. Este responde **qué objeto la recibe**.

Y la tesis del capítulo es incómoda a propósito: **el deep learning es una de las
opciones, no la única ni casi nunca la primera**. En proyectos reales, la mayoría de los
problemas se resuelven —o al menos se acotan— con algo mucho más barato. Elegir bien
aquí ahorra más tiempo que cualquier optimización posterior.

---

## 🪜 La escalera de escalado

Antes que la tabla, la regla práctica. En un proyecto real no *eliges* una familia:
**subes peldaños**.

```none
  baseline trivial ─▶ modelo clásico ─▶ árboles / boosting ─▶ preentrenado
                          ─▶ fine-tuning ─▶ entrenar desde cero
```

Dos normas que conviene tomarse en serio:

1. **Solo subes un peldaño cuando el anterior demuestra ser insuficiente con métricas**,
   no por intuición ni por presión de expectativas.
2. **Cada peldaño multiplica el coste**: de desarrollo, de cómputo, de latencia y —el que
   siempre se olvida— de mantenimiento a dos años vista.

El error más común y más caro del sector es empezar por el peldaño cinco.

---

## Las familias

| Familia | Cuándo es la respuesta correcta |
|---|---|
| [5.0 Sin modelo](5.0-baselines-y-heuristicas.md) | Reglas, heurísticas, baseline trivial. **Siempre lo primero.** |
| [5.1 Clásicos](5.1-modelos-clasicos.md) | Pocos datos, interpretabilidad obligatoria, coste bajo |
| [5.2 Árboles y boosting](5.2-arboles-y-boosting.md) | Datos tabulares: siguen ganando muy a menudo |
| [5.3 Probabilísticos](5.3-modelos-probabilisticos.md) | Necesitas incertidumbre calibrada o estructura causal |
| [5.4 Redes neuronales](5.4-redes-neuronales-la-historia.md) | Datos con estructura (imagen, texto, grafo) y volumen |
| [5.5 Preentrenados](5.5-modelos-preentrenados.md) | Ya existe un modelo que sabe lo que necesitas |

**Transversal:** [5.6 Ensembles](5.6-ensembles.md) — combinar modelos de *cualquier*
familia. No es una familia en sí, es una técnica que se aplica encima. Un random forest
es un ensemble de árboles; también se hacen ensembles de redes. Por eso vive fuera de la
tabla.

---

## 🎛️ Criterios de decisión

Una sola columna se queda corta. Estos son los seis ejes que deciden de verdad, y la
tabla existe para consultarse el día que haya que justificar una elección ante alguien.

| Familia | Datos necesarios | Tipo de dato | Interpretabilidad | Coste inferencia | Incertidumbre | Soberanía |
|---|---|---|---|---|---|---|
| **5.0 Sin modelo** | Ninguno | Cualquiera | Total | Nulo | — | Total |
| **5.1 Clásicos** | Decenas–miles | Tabular | Alta | Muy bajo | Parcial | Total |
| **5.2 Árboles/boosting** | Cientos–millones | Tabular | Media (importancias, SHAP) | Bajo | Parcial | Total |
| **5.3 Probabilísticos** | Pocos | Series, tabular | Alta | Bajo–medio | **Nativa y calibrada** | Total |
| **5.4 Redes** | Miles–millones | Imagen, texto, grafo, audio | Baja | Alto | Mala por defecto | Total (si entrenas tú) |
| **5.5 Preentrenados** | Casi ninguno | Imagen, texto, multimodal | Muy baja | Alto o externo | Mala | **⚠️ Depende del despliegue** |

> ⚠️ **La columna de soberanía no es decorativa.** En un Data Space regulado, "llamar a
> una API externa" puede estar directamente prohibido, y eso convierte la elección de
> familia en una decisión de cumplimiento antes que técnica. Un modelo peor que se
> ejecuta on-premise puede ser la única opción viable. Conviene decidir esta restricción
> **antes** de comparar métricas, no después.

---

## 🧬 El cruce con el capítulo 4

Los dos ejes son ortogonales: puedes combinar cualquier paradigma de aprendizaje con
cualquier familia de modelo. **Pero no todas las casillas están ocupadas, y el patrón de
huecos es informativo.**

| | Supervisado | Auto-supervisado | Refuerzo |
|---|---|---|---|
| **Clásicos** | ✅ Habitual | ❌ Sin sentido útil | ⚠️ Solo tabular/tabla Q |
| **Árboles/boosting** | ✅ Dominante en tabular | ❌ No aplica | ⚠️ Marginal |
| **Probabilísticos** | ✅ Habitual | ⚠️ Parcial | ✅ Model-based RL |
| **Redes** | ✅ | ✅ **Exclusivo** | ✅ **Exclusivo en la práctica** |

Los dos paradigmas más potentes del capítulo 4 —el preentrenamiento auto-supervisado
([§4.3](../04-formas-de-aprender/4.3-con-el-propio-dato-auto-supervisado.md)) y el deep RL
([§4.6](../04-formas-de-aprender/4.6-por-prueba-y-error-refuerzo.md))— **solo funcionan sobre redes neuronales**,
porque ambos exigen dos cosas que las demás familias no ofrecen: **representaciones
aprendidas y reutilizables**, y **diferenciabilidad de extremo a extremo**.

Esa restricción explica la paradoja del capítulo: las redes **pierden en datos tabulares**
frente al boosting y aun así acabaron dominando el campo. No ganaron por ser mejores
modelos, sino por ser el único sustrato sobre el que los paradigmas de aprendizaje más
escalables podían operar.

---

## ⚠️ Errores frecuentes que conviene desactivar

| Confusión | Aclaración |
|---|---|
| "Deep learning es lo mejor" | En datos tabulares el boosting sigue ganando muy a menudo, con una fracción del coste. |
| "Ensembles son una familia" | Son una técnica transversal. Se aplican sobre cualquier familia. |
| "Más parámetros = mejor" | Solo si tienes datos y estructura que lo justifiquen. Si no, es sobreajuste caro. |
| "El softmax me da una probabilidad" | Da un número entre 0 y 1 mal calibrado. Para incertidumbre real → §5.3. |
| "Interpretabilidad = importancia de variables" | Las importancias explican al modelo, no al fenómeno. No implican causalidad. |
| "Usar un preentrenado no es una decisión de arquitectura" | Lo es, y además arrastra dependencias externas, coste y restricciones legales. |

---

## 🗺️ Orden de lectura

**5.0 → 5.1 → 5.2 → 5.3 → 5.4 → 5.5 → 5.6.**

Sigue la escalera. La sección **5.4 está contada como historia**, porque cada arquitectura
existe por el muro contra el que chocó la anterior: **saber cuál es el muro es saber por
qué existe la arquitectura**. Es la sección más larga del capítulo y tiene subdivisión
interna propia.

---

## 📚 Recursos transversales

> ⚠️ Verifica los enlaces antes de fijarlos: van de memoria y sin acceso a búsqueda.

| Recurso | Formato | Nota |
|---|---|---|
| **scikit-learn — *"Choosing the right estimator"*** · [scikit-learn.org](https://scikit-learn.org/stable/machine_learning_map.html) | 🖼️ | El diagrama de flujo clásico para elegir modelo. Simplista, pero funciona como póster de referencia. |
| **StatQuest** · [canal](https://www.youtube.com/@statquest) | 🎥 | Cubre casi todas las familias de este capítulo en vídeos cortos e independientes. La referencia por defecto de todo el capítulo. |
| [**An Introduction to Statistical Learning**](https://www.statlearning.com) (ISL) | 📖 PDF libre | El mapa completo de 5.1–5.2 con el nivel de rigor justo. |
| **Grinsztajn et al.** — *"Why do tree-based models still outperform deep learning on tabular data?"* | 📄 | El respaldo empírico de la fila 5.2. Verifica el estado del arte: los transformers tabulares con aprendizaje en contexto (TabPFN y derivados) están empezando a competir en datasets pequeños. |

### Por completar
- [ ] Un ejemplo trabajado del mismo problema resuelto en los seis peldaños, con métricas y coste comparados. Sería la mejor pieza del capítulo.
- [ ] Tabla de coste real (latencia, memoria, € por millón de inferencias) medida en el laboratorio propio.

---

## 🔗 Continúa

- **Eje anterior:** [§4 Formas de aprender](../04-formas-de-aprender/index.md) — de dónde sale la señal.
- **Eje siguiente:** §6 — cuándo y cuántas veces se aprende.