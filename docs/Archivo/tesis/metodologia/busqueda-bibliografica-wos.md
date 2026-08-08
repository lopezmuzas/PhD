---
title: "Guía de exploración bibliográfica en Web of Science"
tags: [tesis, metodologia]
status: borrador
updated: 2026-08-08
---

# Guía de exploración bibliográfica en Web of Science
## Federated Learning ∩ Data Spaces / Data Sovereignty — cómo encontrar nichos con poca competencia

> Objetivo: darte (1) cadenas de búsqueda listas para pegar en WoS, (2) una metodología para detectar **nichos emergentes con baja competencia**, y (3) un mapa de temas adyacentes al aprendizaje (por refuerzo y otros) donde el cruce con espacios de datos está poco trabajado.

---

## 1. Cómo funciona la sintaxis en Web of Science (mini-recordatorio)

- Campo recomendado: **TS=** (Topic: título + abstract + keywords del autor + Keywords Plus).
- Operadores: `AND`, `OR`, `NOT`, comillas para frases exactas, `*` como comodín (`federat*` captura federated, federation, federating).
- `NEAR/n` exige que dos términos aparezcan a menos de *n* palabras de distancia — muy útil para forzar que "federated" y "learning" estén realmente relacionados con "data space" en el mismo contexto.
- Filtra después por año (2020-2026), tipo de documento (Article, Review, Proceedings) y categorías WoS (Computer Science, Artificial Intelligence; Computer Science, Information Systems; Telecommunications).

---

## 2. Bloques de keywords (construye tus queries combinándolos)

### Bloque A — Federated Learning y variantes
```
("federated learning" OR "federated machine learning" OR "federated reinforcement learning"
 OR "federated analytics" OR "federated AI" OR "collaborative learning" OR "decentralized learning"
 OR "distributed machine learning" OR "swarm learning" OR "split learning" OR "gossip learning"
 OR "federated transfer learning" OR "personalized federated learning" OR "federated unlearning")
```

### Bloque B — Data Spaces y soberanía
```
("data space" OR "data spaces" OR "dataspace" OR "dataspaces"
 OR "data sovereignty" OR "sovereign data" OR "digital sovereignty"
 OR "data ecosystem*" OR "data sharing infrastructure" OR "data marketplace*"
 OR "International Data Spaces" OR "IDS-RAM" OR "Gaia-X" OR "GAIA-X"
 OR "Eclipse Dataspace" OR "EDC connector" OR "Catena-X" OR "Manufacturing-X"
 OR "European Health Data Space" OR "EHDS" OR "HealthData@EU"
 OR "data governance act" OR "data act" OR "usage control" OR "usage polic*"
 OR "ODRL" OR "trusted data sharing" OR "data trustee*" OR "data intermediar*")
```

### Bloque C — Aprendizaje por refuerzo y paradigmas de aprendizaje
```
("reinforcement learning" OR "deep reinforcement learning" OR "multi-agent reinforcement learning"
 OR "offline reinforcement learning" OR "policy gradient" OR "actor-critic"
 OR "Q-learning" OR "bandit*" OR "Markov decision process*"
 OR "imitation learning" OR "transfer learning" OR "continual learning"
 OR "self-supervised learning" OR "meta-learning" OR "curriculum learning")
```

### Bloque D — Mecanismos transversales (donde suelen estar los nichos)
```
("incentive mechanism*" OR "Shapley" OR "contribution evaluation" OR "data valuation"
 OR "differential privacy" OR "secure aggregation" OR "homomorphic encryption"
 OR "trusted execution environment*" OR "verifiable credential*" OR "self-sovereign identity"
 OR "smart contract*" OR "blockchain" OR "machine unlearning" OR "right to be forgotten"
 OR "byzantine" OR "poisoning attack*" OR "model governance" OR "AI Act" OR "GDPR")
```

---

## 3. Queries listas para usar (de más amplia a más nicho)

### Q1 — Panorama general FL ∩ data spaces (para mapear el campo)
```
TS=(("federated learning" OR "federated analytics" OR "swarm learning")
AND ("data space*" OR "dataspace*" OR "data sovereignty" OR "Gaia-X"
     OR "International Data Spaces" OR "data ecosystem*"))
```
Esperable: pocos cientos de resultados como máximo. Es tu corpus base para la revisión sistemática.

### Q2 — Versión estricta con NEAR (elimina ruido de "space" como espacio físico)
```
TS=("federated learning" NEAR/15 ("data sovereignty" OR "sovereign data" OR "dataspace*" OR "data space*"))
```

### Q3 — FRL ∩ soberanía/espacios de datos (tu intersección de tesis; aquí casi no hay nada → nicho)
```
TS=(("federated reinforcement learning" OR ("reinforcement learning" AND "federated learning")
     OR "federated multi-agent")
AND ("data space*" OR "data sovereignty" OR "Gaia-X" OR "International Data Spaces"
     OR "usage control" OR "data governance"))
```
Si devuelve <20 resultados (probable), **enhorabuena: has confirmado empíricamente el gap**. Documenta el número y la fecha de búsqueda para tu propuesta.

### Q4 — Sectoriales (donde la financiación europea concentra proyectos)
```
TS=("federated learning" AND ("European Health Data Space" OR "EHDS" OR "HealthData@EU"))

TS=(("federated learning" OR "federated reinforcement learning")
AND ("Catena-X" OR "Manufacturing-X" OR "smart manufacturing" OR "Industry 5.0")
AND ("data sovereignty" OR "data space*"))

TS=("federated learning" AND ("energy data space" OR "mobility data space" OR "agricultural data space"))
```

### Q5 — Mecanismos poco explotados (candidatos a nicho, combinar Bloque D)
```
TS=("federated learning" AND "data space*" AND ("incentive*" OR "data valuation" OR "Shapley"))

TS=("federated unlearning" AND ("data sovereignty" OR "right to be forgotten" OR "GDPR"))

TS=(("federated learning" OR "federated reinforcement learning")
AND ("usage control" OR "ODRL" OR "usage polic*") AND ("connector*" OR "dataspace*"))

TS=("federated learning" AND ("self-sovereign identity" OR "verifiable credential*"))

TS=("federated reinforcement learning" AND ("incentive*" OR "mechanism design" OR "game theor*"))
```

### Q6 — Vigilancia de frentes muy recientes (2025-2026)
```
TS=(("federated fine-tuning" OR "federated LLM*" OR "federated large language model*")
AND ("data sovereignty" OR "data residency" OR "AI Act" OR "data space*"))

TS=("federated learning" AND "machine unlearning" AND "reinforcement learning")

TS=(("federated reinforcement learning") AND ("RLHF" OR "GRPO" OR "preference*"))
```

---

## 4. Metodología para detectar nichos con poca competencia

La clave no es solo "pocos papers", sino **pocos papers + señal de crecimiento + demanda externa (regulación/industria)**. Sigue este protocolo:

### Paso 1 — Mide el tamaño y la pendiente
Para cada query candidata, anota: nº total de resultados, distribución por año (en WoS: "Analyze Results" → Publication Years). Un nicho prometedor muestra: **<50 papers totales, pero crecimiento año a año desde ~2022-2023**. Si lleva plano desde 2019, probablemente es un callejón sin salida, no un nicho.

### Paso 2 — Mide la concentración de autores
"Analyze Results" → Authors / Affiliations. Si 2-3 grupos firman el 80% de los papers, hay poca competencia **pero** valora si podrás diferenciarte de ellos (o colaborar). Si los autores son muy dispersos y nadie ha consolidado el tema, es terreno abierto.

### Paso 3 — Cruza con citas
Ordena por "Times Cited". Un nicho sano tiene 1-3 papers seminales muy citados (la gente reconoce el problema) y muchos huecos sin resolver en sus secciones de "future work". Lee sistemáticamente las secciones de *open challenges* de los surveys: son listas de temas de tesis gratis.

### Paso 4 — Triangula fuera de WoS
WoS va con retraso (6-18 meses) respecto a arXiv. Para frentes 2025-2026 repite las queries en:
- **arXiv**: https://arxiv.org/list/cs.LG/recent (busca "federated" + "data space"/"sovereignty")
- **Google Scholar** con alertas guardadas para cada query
- **Scopus** (cobertura algo mayor de conferencias europeas que WoS)
- **OpenAlex** (gratuito, API): https://openalex.org/ — permite análisis bibliométrico programático
- **CORDIS** (proyectos europeos financiados — la financiación de hoy son los papers de dentro de 2 años): https://cordis.europa.eu/ — busca "federated learning" + "data space"

### Paso 5 — Mapea visualmente
Exporta los resultados de Q1 (formato "Full Record and Cited References", .txt o .bib) y analízalos con:
- **VOSviewer** (mapas de co-ocurrencia de keywords; los clústeres pequeños y recientes en la periferia del mapa son los nichos): https://www.vosviewer.com/
- **Bibliometrix / Biblioshiny** (R, análisis de temas emergentes, *thematic evolution*): https://www.bibliometrix.org/
- **CiteSpace** (detección de "research fronts" y citation bursts): https://citespace.podia.com/

En VOSviewer, usa el *overlay visualization* coloreado por año medio de publicación: los nodos amarillos (recientes) poco conectados con el núcleo son exactamente los nichos que buscas.

---

## 5. Mapa de nichos candidatos (mi lectura del campo, junio 2026)

Ordenados de mayor a menor "hueco" estimado (poca literatura + demanda creciente):

| # | Nicho | Por qué hay hueco | Query base |
|---|---|---|---|
| 1 | **FRL como servicio nativo de un espacio de datos** (políticas RL como activos gobernados por contratos ODRL sobre conectores EDC) | La literatura FRL ignora la capa de gobernanza; la de data spaces ignora las particularidades del RL | Q3 |
| 2 | **Mecanismos de incentivos y valoración de contribuciones (Shapley) en FL inter-organizacional dentro de data spaces** | Los incentivos se estudian en FL "abstracto", casi nunca anclados a contratos de datos reales | Q5.1 |
| 3 | **Federated unlearning con garantías regulatorias** (derecho al olvido en espacios de datos, EHDS/GDPR) | Frente abierto 2025-2026; special issues activos; casi nada aplicado a data spaces | Q5.2 |
| 4 | **Usage control en tiempo de entrenamiento**: hacer cumplir políticas de uso (ODRL) *durante* el entrenamiento federado, no solo en la transferencia | Los conectores controlan la transferencia, no qué hace el modelo con los datos | Q5.3 |
| 5 | **FL/FRL transfronterizo bajo EHDS** (HealthData@EU): heterogeneidad de nodos nacionales, gobernanza fragmentada | Regulación recién en vigor (2025); ventana 2026-2029 antes de que se sature | Q4.1 |
| 6 | **Identidad auto-soberana (SSI) + credenciales verificables para autenticar participantes FL** | Cruce entre dos comunidades que apenas se citan entre sí | Q5.4 |
| 7 | **Benchmarks reproducibles de FRL sobre infraestructura de data space real** | Gap reconocido explícitamente en los surveys de FRL/FMARL | Q3 + "benchmark*" |
| 8 | **Fine-tuning federado de LLMs bajo residencia de datos / AI Act** | Mucha tracción industrial, literatura académica aún inmadura; competencia creciendo rápido (entra ya o descártalo) | Q6.1 |
| 9 | **RL para orquestar el propio data space** (selección de clientes, scheduling, pricing dinámico de datos mediante RL) | Invierte la relación: RL al servicio de la infraestructura, no al revés | C ∩ B + "client selection" |

> Consejo estratégico: los nichos 1, 4 y 7 son los más defendibles para una tesis de 3-4 años porque combinan un gap claro, demanda regulatoria europea (financiación) y barrera de entrada técnica (hay que dominar dos comunidades a la vez, lo que filtra competencia).

---

## 6. Rutina de vigilancia (30 min/semana)

1. **Alertas**: guarda Q1, Q3 y tus 2-3 queries de nicho como alertas en WoS y Google Scholar.
2. **arXiv**: suscríbete al RSS de cs.LG y cs.DC filtrando "federated"; revisa los viernes.
3. **CORDIS**: revisa trimestralmente nuevos proyectos Horizon Europe con "data space" + "federated" (te dice dónde habrá papers, partners y plazas de doctorado).
4. **Cuaderno de gaps**: cada vez que un paper diga "we leave X for future work", anótalo con la cita. En 3 meses tendrás una lista de 30-50 problemas abiertos clasificables.
5. **Cada 6 meses**: regenera el mapa de VOSviewer y compara — verás qué nichos se están calentando (sal de ellos o publica rápido) y cuáles siguen vacíos.

---

## 7. Checklist antes de comprometerte con un nicho

- [ ] ¿Menos de ~50 papers en WoS y tendencia creciente desde 2022+?
- [ ] ¿Existe al menos un survey o paper seminal que reconozca el problema? (si nadie lo reconoce, quizá no es un problema)
- [ ] ¿Hay demanda externa? (regulación EHDS/AI Act/Data Act, proyectos CORDIS, casos industriales tipo Catena-X)
- [ ] ¿Puedes construir un prototipo evaluable con herramientas abiertas (Flower, EDC, Gymnasium) sin depender de datos privados inaccesibles?
- [ ] ¿Tu director/grupo tiene contactos o proyectos en ese ecosistema?
- [ ] ¿Puedes formular 2-3 preguntas de investigación con métricas de éxito claras?

---

*Generado el 10 de junio de 2026. Las queries usan sintaxis de Web of Science Core Collection; para Scopus sustituye TS= por TITLE-ABS-KEY() y NEAR/n por W/n.*
