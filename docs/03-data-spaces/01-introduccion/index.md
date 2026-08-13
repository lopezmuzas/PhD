---
title: "1. Introducción — Por qué los datos no se pueden centralizar"
tags: [data-spaces, federado, privacidad, regulacion]
status: esbozo
---

# Por qué los datos no se pueden centralizar

> **El punto de partida:** en el mundo real, los datos más valiosos (sanitarios, industriales, financieros, energéticos) están repartidos entre organizaciones que no pueden ni quieren compartirlos directamente. Este bloque explica por qué y qué arquitecturas existen para trabajar con ellos igualmente.

---

## El triángulo tecnología–regulación–economía

Tres fuerzas actúan simultáneamente sobre los datos distribuidos:

```
        REGULACIÓN
        (GDPR, Data Act,
         AI Act, DGA)
             ▲
             │
             │
 TECNOLOGÍA ─┼─ ECONOMÍA
(privacidad, │  (datos como
  C2D, FL)   │  activo comercial)
```

- **Regulación**: el GDPR, el Data Governance Act (2022) y el Data Act (2023) establecen que los datos sensibles **no pueden moverse libremente**. Los sectores salud, energía y finanzas tienen restricciones adicionales.
- **Economía**: los datos tienen valor como activo. Las organizaciones no los comparten gratuitamente con competidores, aunque hacerlo les beneficiaría colectivamente (dilema del prisionero).
- **Tecnología**: la solución técnica (Federated Learning, Compute-to-Data) existe, pero sin la capa de gobernanza no es aplicable a entornos multi-organización reales.

---

## Las tres arquitecturas de datos

Hay una escala continua de descentralización:

| Arquitectura | Dónde están los datos | Quién es el dueño | Ejemplo |
|---|---|---|---|
| **Centralizada** | Un servidor | Un operador | Google, AWS |
| **Distribuida** | Varios nodos | Un solo dueño | CDN de Netflix |
| **Federada** | Cada organización | Cada organización | Hospitales, ITVs, energéticas |

La arquitectura federada es la que importa para la tesis: datos en **silos soberanos** de **dueños distintos** que **no se comparten**.

---

## Por qué FL solo no es suficiente

El Aprendizaje Federado (FL) resuelve el problema técnico: entrenar un modelo sin que los datos salgan de cada organización. Pero no resuelve:

- **¿Bajo qué contrato?** — ¿Quién es responsable si el modelo toma una decisión mala?
- **¿Con qué identidad?** — ¿Cómo verifica el servidor que el cliente es quien dice ser?
- **¿Con qué compensación?** — ¿Cómo se paga al que contribuye datos de más calidad?
- **¿Con qué auditoría?** — ¿Cómo demuestra el sistema ante un regulador que los datos de entrada cumplieron la normativa?

Esas preguntas las responde la capa de **gobernanza**, que en el ecosistema europeo se llama **Data Space**.

---

## La ecuación completa

```
Datos Federados = FL (técnica) + Data Space (gobernanza)
```

| Capa | Qué resuelve | Cómo |
|---|---|---|
| **Federated Learning** | El dato no sale del silo | El modelo viaja, los datos no |
| **Data Space** | Las reglas del intercambio | Contratos, identidades, conectores |
| **Ocean / Pontus-X** | La implementación técnica | Compute-to-Data, tokens, blockchain |

---

## Caso de uso motivador: el sector ITV

*Fuente: análisis de arquitectura de IA soberana, Abril 2026.*

10 empresas de Inspección Técnica de Vehículos (ITV) quieren entrenar un modelo de riesgo conjunto que identifique qué defectos leves en tractores de >15 años preceden a fallos críticos en dirección o frenos.

**El problema**: cada ITV tiene datos históricos de inspecciones en formato JSON. Los datos son sensibles (datos de clientes, datos operativos propietarios). Ninguna ITV acepta enviarlos a un servidor central.

**La solución**: un sistema C2D donde el algoritmo de ML viaja a cada ITV, entrena localmente sobre sus logs históricos, y el orquestador recibe solo el modelo resultante. Ninguna ITV ve los datos de las demás. El resultado es un modelo de predicción de riesgo entrenado sobre el dataset colectivo de las 10 ITVs sin que ninguna haya expuesto sus datos.

```
1. El algoritmo (contenedor Docker) se envía a cada ITV
2. Cada ITV ejecuta el entrenamiento localmente
3. El resultado (itv_policy_learned.json) regresa al orquestador
4. El orquestador agrega las 10 políticas → modelo global
5. El modelo global se distribuye a las 10 ITVs proporcionalmente a su contribución
```

---

## Regulación europea de referencia

| Norma | Qué establece |
|---|---|
| **GDPR** | Protección de datos personales; base legal para el tratamiento |
| **Data Governance Act (2022)** | Reutilización de datos del sector público; intermediarios de datos |
| **Data Act (2023)** | Derecho de acceso y portabilidad; datos IoT |
| **AI Act (2024)** | Trazabilidad de datos de entrenamiento para sistemas de IA de alto riesgo |
| **EHDS** | European Health Data Space — espacio sectorial para datos sanitarios |

<!-- nav-start -->

---

← Anterior: [6.6 Data spaces](../../02-deep-learning/06-donde-estan-los-datos/6.6-data-spaces.md)  
Siguiente: [2. Aprendizaje Federado](../02-aprendizaje-federado/index.md) →

<!-- nav-end -->
