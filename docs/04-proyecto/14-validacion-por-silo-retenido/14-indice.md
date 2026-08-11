---
title: "14. Validación por silo retenido"
tags: [rl-offline, evaluacion, tesis, indice]
status: esbozo
---

# 14. Validación por silo retenido

Si no tienes conjunto de validación pero tienes vecinos con datos, **compra la
evaluación**. Esta sección desarrolla la idea que puede ser el núcleo de la
tesis.

> **Requisito previo.** → [1.6](../../01-fundamentos/01-que-es-y-de-donde-viene/1.6-rl-offline-el-problema.md)
> (por qué el RL offline no tiene validación) y → 11 (el mecanismo de C2D).
> Conviene también → [3.9](../../01-fundamentos/03-como-se-entrena-una-red/3.9-como-medir-si-funciona.md).

---

## La idea, en cuatro pasos

```none
1. No tengo conjunto de validación.
2. Pero tengo vecinos con datos que NUNCA vieron mi política.
3. Los datos del silo B son un conjunto de retención genuino
   y fuera de distribución para la política entrenada en A, C, D.
4. El data space me da el mecanismo para usarlos sin verlos:
   un job C2D que devuelve UN ESCALAR, no un modelo.
```

---

## Índice

### Orientarse

| Página | Qué responde |
|---|---|
| [14.0 Por qué existe esta sección](14.0-por-que-existe-esta-seccion.md) | Los tres verbos, y por qué evaluar ≠ entrenar. **Empieza aquí** |
| [14.1 Leave-one-site-out](14.1-leave-one-site-out.md) | El término, su origen, y qué cambia al traducirlo a RL |

### El problema

| Página | Qué responde |
|---|---|
| [14.2 El problema de la cobertura](14.2-el-problema-de-la-cobertura.md) | ⚠️ **La página crítica.** Cuándo la estimación no significa nada |
| [14.3 Qué obtienes](14.3-que-obtienes.md) | Ordenar, valorar, transferir: tres ambiciones distintas |
| [14.4 Rotar los pliegues](14.4-rotar-los-pliegues.md) | Los 4 pliegues, la varianza como señal, y el confounder |

### El trabajo

| Página | Qué responde |
|---|---|
| [14.5 El experimento](14.5-el-experimento.md) | El diseño concreto. **Sin ejecutar todavía** |
| [14.6 El protocolo](14.6-el-protocolo.md) | La versión formalizada, y qué hay que declarar |
| [14.7 Recursos](14.7-recursos.md) | OPE, selección de política, validación externa |

---

## Las tres advertencias

Lo que hay que entender antes de invertir tiempo:

| # | Advertencia |
|---|---|
| ① | El silo retenido **no te da una métrica**: te da datos sobre los que ejecutar un estimador de OPE |
| ② | Con un silo de retención **no has resuelto el problema del soporte: lo has movido**. Antes tenías extrapolación al aprender; ahora la tienes al evaluar |
| ③ | Al rotar **no evalúas la misma política** cuatro veces: seleccionas un procedimiento, no una política |

---

## Estado

| Componente | Estado |
|---|---|
| Marco conceptual | **Escrito** |
| Verificación bibliográfica de la novedad | **Pendiente** — tarea de la semana 1 de septiembre |
| Diseño del experimento | **Escrito** (→ 14.5) |
| Experimento ejecutado | **No.** Sin cifras propias todavía |
| Protocolo formalizado | **Borrador** (→ 14.6) |

---

## Qué es nuevo y qué no

| Componente | Estado |
|---|---|
| LOSO en federado médico supervisado | **Estándar** |
| RL offline federado | **Existe** (Woo ICML 2024, FEDORA NeurIPS 2024, FDTR JASA 2024) |
| Selección de política offline / OPE | **Área activa**, con problemas abiertos reconocidos |
| **La combinación**, con el silo evaluador como servicio comprable | **No encontrado** — verificar |

---

## Enlaces con el resto del libro

- **→ 1.6** — RL Offline. La tensión de la cobertura es la de esa página, un paso
  después.
- **→ 3.9** — Cómo medir. Todo el protocolo se apoya en su parte B.
- **→ 11** — El mecanismo de C2D que hace posible el job de evaluación.
- **→ 12.2** — El cegado. Extensión 5 del experimento: ¿sobrevive el orden si el
  escalar viene cegado?
- **→ 13.5** — One-shot. El coste en jobs de rotar pliegues empuja en la misma
  dirección.
- **→ Estrategia, idea ①** — Esta sección es su desarrollo completo.
