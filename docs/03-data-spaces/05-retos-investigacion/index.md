---
title: "5. Retos de Investigación en FRL + Data Spaces"
tags: [investigacion, phd, frl, offline-rl, ocean, data-spaces]
status: revisado
---

# Retos de Investigación: FRL Offline sobre Data Spaces

> **Nicho de Investigación:** Entrenar políticas de **Deep Reinforcement Learning** en modo **offline** (off-policy), de forma **federada** entre múltiples proveedores de datos soberanos, dentro de un **Data Space** (Gaia-X / IDS) utilizando **Ocean Protocol / Pontus-X** (Compute-to-Data, datatokens, smart contracts) como infraestructura tecnológica.
>
> **Tesis del nicho:** El reto no está en una sola capa, sino en que cada nivel de la pila añade un problema abierto — y **la intersección vertical completa está prácticamente vacía** en la literatura científica.

---

## 🏗️ La Pila de Retos (Pila Tecnológica)

```
┌─────────────────────────────────────────────────────────┐
│ CAPA 4 — Gobernanza, Incentivos y Seguridad Web3 / UE   │ ──▶ Retos R12–R15
├─────────────────────────────────────────────────────────┤
│ CAPA 3 — Orquestación sobre Ocean Compute-to-Data (C2D) │ ──▶ Retos R8–R11
├─────────────────────────────────────────────────────────┤
│ CAPA 2 — Federated Offline RL (FRL)                     │ ──▶ Retos R3–R7  ★ Núcleo Científico
├─────────────────────────────────────────────────────────┤
│ CAPA 1 — Offline RL Monoagente (CQL, IQL, TD3+BC)       │ ──▶ Retos R1–R2  ★ Base Instrumental
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Matriz de Retos × Capa × Madurez × Pregunta de Investigación

| # | Reto | Capa | Madurez | Pregunta de Investigación Asociada |
|---|---|---|---|---|
| **R1** | *Distribution shift* y pesimismo | Capa 1 | Alta (CQL, IQL) | ¿Qué algoritmo offline local es más robusto como base federable? |
| **R2** | Evaluación offline (OPE) | Capa 1→2 | Media | ¿Cómo estimar la calidad de políticas cliente sin entorno en vivo ni centralizar datos? |
| **R3** | Políticas de comportamiento heterogéneas | Capa 2 | **Baja** | ¿Cómo ponderar/filtrar clientes según la calidad de su política generadora? |
| **R4** | MDPs heterogéneos y personalización | Capa 2 | **Baja** | ¿Global única o representación compartida + cabezas locales (pFRL)? |
| **R5** | Agregación del pesimismo | Capa 2 | **Muy baja** | ¿Cómo combinar penalizaciones de incertidumbre locales con garantías de cobertura colectiva? |
| **R6** | Teoría de convergencia (deep) | Capa 2 | **Muy baja** | ¿Cotas de sub-optimalidad para FRL offline con aproximación profunda de funciones? |
| **R7** | Benchmark federado offline | Capa 2 | **Inexistente** | ¿Cómo particionar D4RL / Minari de forma realista y no-IID para benchmarks FRL? |
| **R8** | Rondas iterativas sobre C2D | Capa 3 | **Muy baja** | ¿Qué patrón arquitectónico encadena jobs C2D efímeros manteniendo estado de modelo versionado? |
| **R9** | Orquestación descentralizada | Capa 3 | **Baja** | ¿Puede un smart contract coordinar rondas federadas sin punto central de fallo/fuga? |
| **R10** | Coste/latencia on-chain | Capa 3 | **Baja** | ¿Qué división *on-chain/off-chain* minimiza el gas/coste por ronda federada? |
| **R11** | Verificación del cómputo | Capa 3 | **Baja** | ¿TEEs, allow-lists o pruebas de cómputo verificable para jobs RL? |
| **R12** | Valoración de trayectorias e incentivos | Capa 4 | **Muy baja** | ¿Mecanismo tipo Shapley computable para valorar datasets de trayectorias en mercados de datos? |
| **R13** | *Poisoning* de políticas | Capa 4 | **Muy baja** | ¿Funcionan Krum/mediana con pérdidas RL no estacionarias? |
| **R14** | Privacidad residual (DP + RL) | Capa 4 | **Baja** | ¿Trade-off privacidad/rendimiento de DP-SGD en CQL federado? |
| **R15** | Gobernanza Web3 ↔ Gaia-X/IDS | Capa 4 | **Muy baja** | ¿Cómo mapear políticas de uso IDS a smart contracts de Ocean para FL? |

---

## 🎯 Tres Posicionamientos Estratégicos de Tesis Doctoral

Recomendación metodológica: elegir **una** capa como contribución principal y utilizar las demás como infraestructura de soporte.

```
OPCIÓN 1: Tesis Algorítmica (Capa 2)
  └─ Algoritmo de FRL offline robusto a heterogeneidad de políticas (R3+R5+R7).
  └─ Evaluación sobre benchmark federado propio (D4RL federado).

OPCIÓN 2: Tesis Arquitectónica (Capa 3)
  └─ Arquitectura de referencia para FL/FRL iterativo sobre C2D en Gaia-X/Pontus-X (R8+R9+R10).
  └─ Análisis de latencia, coste en gas y preservación de privacidad.

OPCIÓN 3: Tesis de Mecanismos / Tokenomics (Capa 4)
  └─ Mecanismo de valoración de trayectorias de datos en mercados descentralizados (R12+R2).
  └─ Protocolos Tit-for-Tat para prevenir free-riding en pools de datos competitivos.
```

---

## 📚 Estado del Arte y Referencias Canónicas

### Federated Learning (FL)
- **McMahan et al. (2017)** — *FedAvg*: [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- **Kairouz et al. (2021)** — *Advances and Open Problems in FL* (La "Biblia" del campo): [arXiv:1912.04977](https://arxiv.org/abs/1912.04977)
- **Li et al. (2020)** — *FedProx*: [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)
- **Karimireddy et al. (2020)** — *SCAFFOLD*: [arXiv:1910.06378](https://arxiv.org/abs/1910.06378)

### Offline RL & FRL
- **Levine et al. (2020)** — *Offline Reinforcement Learning: Tutorial and Review*: [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)
- **Kumar et al. (2020)** — *Conservative Q-Learning (CQL)*: [NeurIPS 2020](https://arxiv.org/abs/2006.04779)
- **Rengarajan et al. (2024)** — *FEDORA: Federated Ensemble-Directed Offline RL*: [arXiv:2305.03097](https://arxiv.org/abs/2305.03097)
- **Single-Policy Coverage FRL (2024)**: [arXiv:2402.05876](https://arxiv.org/abs/2402.05876)

### Data Spaces & Ocean Protocol
- **Ocean Protocol Documentation**: [docs.oceanprotocol.com](https://docs.oceanprotocol.com)
- **DSSC Blueprint**: [blueprint.dssc.eu](https://blueprint.dssc.eu/)
- **deltaDAO Pontus-X Use Cases**: [github.com/deltaDAO](https://github.com/deltaDAO/Ocean-Protocol-Use-Cases)

<!-- nav-start -->

---

← Anterior: [4. Paradigmas FL en Data Spaces](../04-paradigmas-fl/index.md)  
Siguiente: [7.1 Cómo se combinan las 3 decisiones](../../04-proyecto/07-combinaciones/7.1-como-se-mezclan.md) →

<!-- nav-end -->
