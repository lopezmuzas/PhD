---
title: "IV. Data Spaces"
tags: [data-spaces, federado, gobernanza, ocean, gaia-x]
status: esbozo
---

# Data Spaces — Por qué y hacia dónde

> **Conexión con el Deep Learning:** el DL te enseña *cómo* funciona un modelo. Los Data Spaces responden a *dónde están los datos* cuando no pueden centralizarse — que es la condición habitual en entornos industriales, sanitarios o energéticos europeos.

Este bloque cubre el tercer eje del mapa mental del doctorado: la **arquitectura de los datos**. Cuando distintas organizaciones quieren entrenar modelos conjuntos sobre datos que no pueden compartir, necesitan dos capas de solución:

1. **Técnica** → Aprendizaje Federado (FL): el modelo viaja, los datos no.
2. **Gobernanza** → Data Spaces: las reglas, identidades y contratos que hacen ese intercambio legalmente viable entre organizaciones distintas.

---

## El mapa del bloque

```
DATOS DISTRIBUIDOS
        │
        ├── 1. Centralizados ........... un servidor, todos los datos
        ├── 2. Distribuidos ............ repartidos, un mismo dueño
        └── 3. FEDERADOS ............... dueños distintos, datos no se comparten
                 │
                 ├── Aprendizaje Federado ... la solución técnica (FL, FRL)
                 └── Data Spaces ............ la gobernanza que lo hace posible
                           └── Ocean Protocol / Pontus-X ── implementación práctica
```

---

## Secciones de este bloque

| Sección | Qué cubre |
|---|---|
| [1. Introducción](01-introduccion/index.md) | Por qué los datos no pueden centralizarse siempre; el triángulo tecnología-regulación-economía |
| [2. Aprendizaje Federado](02-aprendizaje-federado/index.md) | FL clásico: FedAvg, heterogeneidad non-IID, privacidad diferencial, Flower |
| [3. Data Spaces y Gobernanza](03-data-spaces-gobernanza/index.md) | Qué es un Data Space, IDSA, Gaia-X, DSSC, EDC, regulación europea |
| [4. Paradigmas FL en Data Spaces](04-paradigmas-fl/index.md) | Los 6 paradigmas de FL aplicados a datos soberanos distribuidos |
| [5. Retos de Investigación](05-retos-investigacion/index.md) | La intersección RL offline + federado + Ocean = el nicho de la tesis |

---

## Recursos de referencia

- [DSSC Starter Kit](https://dssc.eu/) — el punto de entrada más claro de la UE sobre qué es un Data Space
- [IDS-RAM 4](https://docs.internationaldataspaces.org/) — el modelo de referencia de IDSA
- [Kairouz et al. 2021 — *Advances and Open Problems in FL*](https://arxiv.org/abs/1912.04977) — el survey de referencia del campo
- [McMahan et al. 2017 — *FedAvg*](https://arxiv.org/abs/1602.05629) — el paper fundacional de FL
