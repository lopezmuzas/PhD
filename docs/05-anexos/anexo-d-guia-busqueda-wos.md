---
title: "Anexo D — Guía de Búsqueda Bibliográfica en Web of Science"
tags: [anexo, wos, scopus, investigacion, busqueda, phd]
status: revisado
---

# Anexo D — Guía de Búsqueda Bibliográfica (WoS / Scopus)

> **Objetivo:** Proporcionar cadenas de búsqueda preparadas para **Web of Science (WoS)** y **Scopus**, metodologías para detectar **nichos emergentes con baja competencia**, y mapeos de temas adyacentes al Aprendizaje Federado y los Data Spaces.

---

## 🔍 1. Sintaxis de Búsqueda en Web of Science (WoS)

- **Campo recomendado**: `TS=` (Topic: busca en título, abstract, palabras clave del autor y *Keywords Plus*).
- **Operadores lógicos**: `AND`, `OR`, `NOT`. Usar comillas `"..."` para frases exactas y asterisco `*` para truncamiento (`federat*` captura *federated*, *federation*, *federating*).
- **Operador de proximidad `NEAR/n`**: Exige que dos términos aparezcan a menos de $n$ palabras de distancia. Ej: `"federated" NEAR/3 "learning"`.
- **Filtros por Categoría WoS**: *Computer Science, Artificial Intelligence*; *Computer Science, Information Systems*; *Telecommunications*.

---

## 🧱 2. Bloques de Palabras Clave (Keywords)

### Bloque A — Federated Learning y Variantes
```text
("federated learning" OR "federated machine learning" OR "federated reinforcement learning"
 OR "federated analytics" OR "federated AI" OR "collaborative learning" OR "decentralized learning"
 OR "distributed machine learning" OR "swarm learning" OR "split learning"
 OR "federated transfer learning" OR "personalized federated learning" OR "federated unlearning")
```

### Bloque B — Data Spaces y Soberanía del Dato
```text
("data space" OR "data spaces" OR "dataspace" OR "dataspaces"
 OR "data sovereignty" OR "sovereign data" OR "digital sovereignty"
 OR "data ecosystem*" OR "data sharing infrastructure" OR "data marketplace*"
 OR "International Data Spaces" OR "IDS-RAM" OR "Gaia-X" OR "GAIA-X"
 OR "Eclipse Dataspace" OR "EDC connector" OR "Catena-X" OR "Manufacturing-X"
 OR "European Health Data Space" OR "EHDS" OR "HealthData@EU"
 OR "data governance act" OR "data act" OR "usage control" OR "usage polic*")
```

### Bloque C — Aprendizaje por Refuerzo (RL)
```text
("reinforcement learning" OR "deep reinforcement learning" OR "multi-agent reinforcement learning"
 OR "offline reinforcement learning" OR "batch reinforcement learning" OR "policy gradient"
 OR "actor-critic" OR "Q-learning" OR "Markov decision process*" OR "conservative Q-learning")
```

### Bloque D — Mecanismos de Seguridad, Web3 y Gobernanza
```text
("incentive mechanism*" OR "Shapley" OR "contribution evaluation" OR "data valuation"
 OR "differential privacy" OR "secure aggregation" OR "homomorphic encryption"
 OR "trusted execution environment*" OR "verifiable credential*" OR "smart contract*"
 OR "blockchain" OR "machine unlearning" OR "byzantine" OR "poisoning attack*")
```

---

## 🎯 3. Cadenas de Búsqueda Listas para Usar

### Query 1: La Intersección Exacta del Doctorado (Nicho Específico)
```text
TS=(("federated reinforcement learning" OR ("federated learning" AND "reinforcement learning")) 
AND ("data space*" OR "dataspace*" OR "data sovereignty" OR "Gaia-X" OR "IDS-RAM" OR "Ocean Protocol"))
```

### Query 2: Federated Learning sobre Data Spaces (Visión Amplia)
```text
TS=(Bloque A AND Bloque B)
```

### Query 3: Mecanismos de Incentivos y Privacidad en Data Spaces
```text
TS=(Bloque B AND Bloque D)
```

---

## 📊 4. Metodología para Identificar Nichos de Investigación Emergentes

1. **Lanzar Query amplia** en WoS (ej. *Query 2*).
2. **Analizar Resultados por Año (2020-2026)**: Si el número de papers por año es $< 20$, el nicho está en fase de gestación (alta oportunidad).
3. **Mapear Citation Network**: Exportar las referencias a VosViewer o CitNetExplorer para identificar clusters de autores y encontrar vacíos entre clusters.
4. **Identificar "Celdas Vacías"**: Cruzar un paradigma de la matriz de la sección 7.1 contra una regulación o infraestructura europea real.
