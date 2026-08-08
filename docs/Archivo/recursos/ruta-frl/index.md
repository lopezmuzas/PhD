---
title: "Ruta de aprendizaje FRL — Índice"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Ruta de aprendizaje FRL — Índice

> **Companion** del *Modelo Mental: del Deep Learning al Aprendizaje Federado con Refuerzo*. Mientras el mapa mental ordena los **conceptos**, esta ruta ordena los **recursos** (PDFs, vídeos, cursos, práctica) y el **orden de estudio**. Un archivo por módulo.

## Módulos

| # | Módulo | Mapa mental | Estado |
|---|---|---|---|
| 0 | [Cimientos matemáticos (álgebra lineal)](modulo-0-cimientos-matematicos.md) | Prerrequisito de §2.3 | ✅ |
| 1 | [El ciclo de entrenamiento de una red profunda](modulo-1-ciclo-entrenamiento.md) | §2.3 | ✅ |
| 2 | [Los paradigmas de aprendizaje](modulo-2-paradigmas.md) | §3 (eje ① CÓMO) | ✅ |
| 3 | [Aprendizaje Federado (FL)](modulo-3-aprendizaje-federado.md) | §5 (eje ③ DÓNDE) | ✅ |
| 4 | [Data Spaces (gobernanza)](modulo-4-data-spaces.md) | §6 | ✅ |
| 5 | [Federated Reinforcement Learning (FRL)](modulo-5-frl.md) | §7 | ✅ |

## Orden recomendado y dependencias

```
Módulo 0 (álgebra lineal + regla de la cadena)
   └─► Módulo 1 (entrenamiento: forward, pérdida, backprop, optimización)
          ├─► Módulo 2 (paradigmas: supervisado ... RL)
          └─► Módulo 3 (federado: FedAvg, no-IID, privacidad)
                 ├─► Módulo 4 (data spaces)  ◄── necesita el "porqué" del cross-silo
                 └─► Módulo 5 (FRL)           ◄── = Módulo 2 (RL) + Módulo 3 (federado)
```

**Leyenda de recursos** (común a todos los módulos): `[ tipo · nivel · tiempo ]`
- Tipo: 📘 libro/PDF · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica (código)
- Nivel: ⭐ introductorio · ⭐⭐ intermedio · ⭐⭐⭐ avanzado

**Regla de oro:** cada módulo trae una **ruta mínima** (lo imprescindible) y **profundización opcional**. Haz primero las rutas mínimas de 0 → 1 → 2 → 3 y vuelve a lo opcional solo si lo necesitas.
