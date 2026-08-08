---
title: "Federated Learning + Reinforcement Learning + Data Spaces"
tags: [tesis, estado-del-arte]
status: borrador
updated: 2026-08-08
---

# Federated Learning + Reinforcement Learning + Data Spaces
## Estado del arte (junio 2026) y ruta de aprendizaje para una tesis doctoral

> Documento de trabajo orientado a plantear una propuesta de PhD en la intersección de **Aprendizaje Federado (FL)**, **Aprendizaje por Refuerzo (RL)** y **Espacios de Datos (Data Spaces)**, con énfasis en el ecosistema europeo (IDSA, Gaia-X, EHDS, Data Spaces Support Centre).

---

## 1. Estado del arte

### 1.1. Aprendizaje Federado (Federated Learning, FL)

El FL, propuesto por Google (McMahan et al., 2017, algoritmo **FedAvg**), permite entrenar un modelo global de forma colaborativa entre múltiples clientes (dispositivos u organizaciones) sin centralizar los datos crudos: cada participante entrena localmente y solo comparte actualizaciones del modelo, que un agregador combina. Esto responde a preocupaciones de privacidad, soberanía del dato y cumplimiento regulatorio (GDPR, AI Act), y por eso es especialmente atractivo en sanidad, finanzas e IoT industrial.

**Retos abiertos consolidados en la literatura (surveys 2025-2026):**

Un survey reciente (MDPI Computers, 2026) sistematiza seis retos nucleares del FL: heterogeneidad (estadística y de sistemas), sobrecarga computacional, cuellos de botella de comunicación, selección de clientes, agregación/optimización y preservación de la privacidad, analizando cómo afectan a la convergencia, la equidad y la fiabilidad del sistema. Otros frentes activos:

- **Datos no-IID**: la heterogeneidad estadística entre clientes degrada la convergencia de FedAvg. Algoritmos clave: FedProx, SCAFFOLD, FedNova, MOON, y personalización (pFL, meta-learning, clustering de clientes).
- **Privacidad más allá de "no compartir datos"**: los gradientes filtran información (ataques de *gradient inversion*, p. ej. "Deep Leakage from Gradients"). Defensas: privacidad diferencial (DP), agregación segura (SecAgg/SMPC), cifrado homomórfico (HE), entornos de ejecución confiables (TEE).
- **Robustez**: ataques bizantinos, envenenamiento de modelo/datos, backdoors; agregadores robustos (Krum, Trimmed Mean, median-based).
- **FL descentralizado y blockchain**: eliminación del servidor central (gossip, blockchain-based FL) para evitar puntos únicos de fallo y de confianza.
- **Machine Unlearning federado** (frontera 2025-2026): borrar selectivamente la contribución de datos de un cliente del modelo global, motivado por el "derecho al olvido" del GDPR. IEEE ComSoc tiene un special issue abierto en 2026 sobre FL + unlearning.
- **FL para LLMs**: fine-tuning federado con LoRA/PEFT, que reduce el ancho de banda por ronda de gigabytes a cientos de megabytes; en 2026 es ya una técnica de producción para cumplir requisitos de residencia de datos (EHDS, AI Act art. 10, HIPAA).

**Referencias canónicas:**

| Referencia | Enlace |
|---|---|
| McMahan et al. (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg) | https://arxiv.org/abs/1602.05629 |
| Kairouz et al. (2021), "Advances and Open Problems in Federated Learning" (la "biblia" del campo) | https://arxiv.org/abs/1912.04977 |
| Li et al. (2020), "Federated Optimization in Heterogeneous Networks" (FedProx) | https://arxiv.org/abs/1812.06127 |
| Karimireddy et al. (2020), SCAFFOLD | https://arxiv.org/abs/1910.06378 |
| Survey FL 2026 (retos nucleares, MDPI Computers) | https://www.mdpi.com/2073-431X/15/3/155 |
| Survey "Privacy-Preserving Collaborative Intelligence" (2025) | https://arxiv.org/abs/2504.17703 |
| Survey FL descentralizado | https://arxiv.org/abs/2308.04604 |

---

### 1.2. Aprendizaje por Refuerzo (Reinforcement Learning, RL)

El RL formaliza la toma de decisiones secuenciales mediante Procesos de Decisión de Markov (MDP): un agente aprende una política que maximiza la recompensa acumulada interactuando con un entorno. Los avances en *deep RL* (DQN, PPO, SAC, TD3) han permitido abordar espacios de estados de alta dimensión. Líneas activas en 2025-2026:

- **RL multiagente (MARL)**: cooperación/competición entre agentes; problemas de no-estacionariedad, asignación de crédito y observabilidad parcial.
- **RL offline / batch RL**: aprender de datasets estáticos sin interacción (clave cuando interactuar es caro o peligroso: salud, industria).
- **RLHF / RL para LLMs**: PPO, DPO, GRPO aplicados al alineamiento y razonamiento de modelos de lenguaje (la conexión RL↔LLM es uno de los temas más calientes del momento).
- **Safe RL y RL con restricciones**: garantías de seguridad durante el aprendizaje, esenciales para despliegues industriales.

**Referencias y recursos canónicos:**

| Referencia | Enlace |
|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction* (2ª ed., libro completo gratuito) | http://incompleteideas.net/book/the-book-2nd.html |
| OpenAI Spinning Up in Deep RL (guía + código) | https://spinningup.openai.com/ |
| Survey MARL (Gronauer & Diepold, 2022) | https://link.springer.com/article/10.1007/s10462-021-09996-w |

---

### 1.3. Federated Reinforcement Learning (FRL): la intersección FL + RL

El **FRL** combina ambos paradigmas: múltiples agentes RL entrenan políticas localmente en sus propios entornos y comparten parámetros de política/valor (no trayectorias crudas) mediante un esquema federado. Es el núcleo metodológico natural de una tesis en esta intersección.

**Estado del campo (2025-2026):**

- El survey fundacional sigue siendo Qi et al. (2021), que distingue **HFRL** (Horizontal FRL: agentes en entornos similares, mismo espacio de estados/acciones) y **VFRL** (Vertical FRL: agentes con observaciones parciales complementarias del mismo entorno).
- La frontera actual es el **Federated Multi-Agent RL (FMARL)**: un survey exhaustivo de 2025 (Jing et al., *Expert Systems with Applications*) señala que el FRL clásico asume agentes aislados que solo comparten parámetros, mientras que muchas aplicaciones reales exigen agentes que **interactúan en un entorno compartido** bajo restricciones de privacidad y comunicación, lo que introduce retos nuevos: no-estacionariedad del entorno, asignación de crédito y observabilidad parcial.
- **Teoría de convergencia**: resultados de *linear speedup* bajo muestreo markoviano (Khodadadian et al., ICML 2022), FRL asíncrono con policy gradients (arXiv 2404.08003), enfoques basados en consenso para agentes homogéneos y heterogéneos (Giuseppi et al., 2025), y garantías frente a adversarios.
- **Survey 2025 que unifica regímenes**: "A Survey of Multi-Agent RL: Federated Learning and Cooperative and Noncooperative Decentralized Regimes" (julio 2025) organiza el campo en tres topologías —cooperación coordinada centralmente (federada), cooperación ad-hoc descentralizada y regímenes no cooperativos— y revisa formulaciones, garantías teóricas y rendimiento numérico.
- **FRL + LLMs** (frontera 2026): trabajos como FedMOA (federated GRPO para LLMs de razonamiento personalizados bajo recompensas heterogéneas) muestran la convergencia de FRL con el post-entrenamiento de LLMs.
- **Aplicaciones dominantes**: asignación de recursos en redes 5G/6G, computación en el borde (edge/IoT), conducción autónoma, robótica colaborativa en fabricación inteligente, sistemas de recomendación con preservación de privacidad, y tratamiento clínico con historiales electrónicos.

**Retos abiertos identificados en los surveys (candidatos a contribución de tesis):**

1. Convergencia teórica con entornos heterogéneos (distintos MDPs por cliente) — el análogo RL del problema no-IID.
2. Eficiencia de comunicación (las políticas RL requieren muchas rondas; compresión, actualización asíncrona, selección de agentes).
3. Privacidad de trayectorias: los modelos agregados pueden sufrir ataques de inferencia sobre las trayectorias locales; DP aplicada a policy gradients degrada mucho el rendimiento.
4. Robustez bizantina en FRL (agentes maliciosos que envenenan políticas).
5. Mecanismos de incentivos y *credit assignment* entre organizaciones participantes.
6. Benchmarks y reproducibilidad: no existe un benchmark estándar de FRL comparable a LEAF/FLamby en FL supervisado — **gap claro**.

**Referencias clave:**

| Referencia | Enlace |
|---|---|
| Qi et al. (2021), "Federated RL: Techniques, Applications, and Open Challenges" | https://arxiv.org/abs/2108.11887 |
| Jing et al. (2025), "Federated Multi-Agent RL: A Comprehensive Survey" (ESWA) | https://www.sciencedirect.com/science/article/abs/pii/S0957417425023474 |
| Survey MARL federado/descentralizado/no-cooperativo (2025) | https://arxiv.org/abs/2507.06278 |
| FRL asíncrono con policy gradient (convergencia) | https://arxiv.org/abs/2404.08003 |
| Khodadadian et al. (ICML 2022), linear speedup en FRL | https://proceedings.mlr.press/v162/khodadadian22a.html |
| Pinto Neto et al. (2023), FRL en IoT | https://www.mdpi.com/2076-3417/13/11/6497 |

---

### 1.4. Espacios de Datos (Data Spaces)

Un **espacio de datos** es una infraestructura federada para el intercambio soberano de datos entre organizaciones: los datos permanecen en origen y se comparten bajo políticas de uso negociadas y verificables, mediante conectores estandarizados, catálogos federados, identidades verificables y marcos de confianza. Es la apuesta estratégica europea (European Strategy for Data) para la economía del dato.

**Pilares del ecosistema (2025-2026):**

- **IDSA (International Data Spaces Association)**: define el **IDS-RAM** (Reference Architecture Model), que estructura el intercambio de datos en cinco capas (negocio, funcional, información, proceso, sistema) y define roles (proveedor, consumidor, intermediario, órgano de gobernanza). https://internationaldataspaces.org/
- **Gaia-X**: iniciativa europea de infraestructura de datos federada y soberana; aporta el Trust Framework, credenciales verificables y catálogos federados. Participantes con roles de proveedor, consumidor o federador, con negociación contractual automatizada. https://gaia-x.eu/
- **Eclipse Dataspace Components (EDC)**: la implementación open source de referencia (Connector, Federated Catalog, Identity Hub) usada en proyectos como Catena-X y Gaia-X4KI. https://projects.eclipse.org/projects/technology.edc
- **Data Spaces Support Centre (DSSC)**: financiado por la Comisión Europea, publica el **Data Spaces Blueprint** con los bloques de construcción esenciales. https://dssc.eu/
- **Espacios sectoriales**: Catena-X / Manufacturing-X (automoción/industria), Mobility Data Space, agricultura, energía y, de forma destacada, el sanitario:
- **EHDS (European Health Data Space)**: el **Reglamento (UE) 2025/327**, publicado el 5 de marzo de 2025 y en vigor desde el 26 de marzo de 2025, es el primer espacio de datos sectorial europeo con rango normativo. Establece uso primario (acceso del ciudadano a su historial en toda la UE) y secundario (investigación e innovación) de datos sanitarios electrónicos, con implantación gradual hasta 2029-2034: actos de ejecución hasta 2027, aplicación de las disposiciones principales de uso secundario en marzo de 2029. El acceso secundario se canaliza a través de Health Data Access Bodies nacionales y la infraestructura transfronteriza **HealthData@EU**, cuyo piloto viene probando desde 2022 el **análisis federado transfronterizo**, donde la computación se distribuye entre nodos nacionales en lugar de centralizarse.

**La conexión con FL/FRL (el corazón de tu tesis):**

- Artículos recientes (JMIR 2025, "Reality Check: The Aspirations of the EHDS...") señalan explícitamente el **aprendizaje automático federado como el enfoque analítico descentralizado** que el EHDS necesita para materializar sus aspiraciones, y documentan los retos técnicos y de gobernanza reales del análisis federado transfronterizo.
- Casos industriales como **EGOKIA** (Mondragon Assembly + Ikerlan, sobre conectores IDSA) demuestran FL sobre espacios de datos en fabricación, reportando mejoras de calidad de modelo de hasta el 30% gracias a técnicas federadas y reducción de cargas de comunicación.
- Un survey de 2025 sobre el panorama de espacios de datos (arXiv 2509.06983) sitúa Gaia-X y el DSSC Blueprint como referencias arquitectónicas para infraestructuras de datos seguras e interoperables.

**Gap de investigación (tu oportunidad):** la literatura de FRL casi nunca considera la capa de gobernanza, políticas de uso, contratos de datos e identidad de los espacios de datos; y la literatura de data spaces trata el FL como "un servicio más", sin abordar las particularidades del RL (interacción secuencial, trayectorias sensibles, recompensas heterogéneas, no-estacionariedad). **La integración de FRL como servicio nativo de un espacio de datos (p. ej., sobre conectores EDC, con políticas ODRL gobernando qué actualizaciones de política pueden compartirse, incentivos entre organizaciones, y cumplimiento EHDS/AI Act) está prácticamente inexplorada y es una línea de tesis viable y diferencial.**

---

### 1.5. Posibles preguntas de investigación para la tesis

1. **Arquitectura**: ¿Cómo orquestar FRL sobre la arquitectura de un espacio de datos (conectores EDC, catálogo federado, políticas de uso ODRL), de modo que el entrenamiento federado de políticas RL sea un *servicio soberano* del data space?
2. **Privacidad**: ¿Qué garantías formales (DP sobre trayectorias/gradientes de política, agregación segura) son compatibles con la convergencia del FRL en entornos organizacionales heterogéneos?
3. **Heterogeneidad**: ¿Cómo converge el FRL cuando cada organización tiene un MDP distinto (dinámicas y recompensas heterogéneas)? ¿Personalización de políticas vs. política global?
4. **Incentivos y gobernanza**: mecanismos de incentivo y contribución justa (Shapley, contratos de datos) para que organizaciones compitan/cooperen en un mismo espacio de datos.
5. **Dominio de aplicación**: validación en un espacio sectorial concreto (salud/EHDS: políticas de tratamiento desde historiales; industria/Manufacturing-X: control colaborativo entre plantas; energía/movilidad: gestión de recursos).
6. **Benchmark**: diseño de un benchmark reproducible de FRL sobre infraestructura de espacio de datos real (gap reconocido en los surveys).

---

## 2. Ruta de aprendizaje (Roadmap) hacia la tesis

> Estimación orientativa: 12-18 meses de preparación intensiva compatible con el primer año de doctorado. Cada fase incluye recursos públicos y gratuitos siempre que es posible.

### Fase 0 — Fundamentos matemáticos y de programación (1-2 meses, en paralelo)

**Qué dominar:** álgebra lineal, cálculo multivariable, probabilidad y estadística, optimización (descenso de gradiente, convexidad), Python científico (NumPy, PyTorch).

| Recurso | Tipo | Enlace |
|---|---|---|
| 3Blue1Brown — Essence of Linear Algebra | YouTube | https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab |
| 3Blue1Brown — Essence of Calculus | YouTube | https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr |
| Mathematics for Machine Learning (libro gratuito) | Libro | https://mml-book.github.io/ |
| PyTorch — tutoriales oficiales | Tutorial | https://pytorch.org/tutorials/ |

### Fase 1 — Machine Learning y Deep Learning (2-3 meses)

**Qué dominar:** aprendizaje supervisado, redes neuronales, entrenamiento distribuido básico, regularización, evaluación.

| Recurso | Tipo | Enlace |
|---|---|---|
| Andrew Ng — Machine Learning Specialization (auditable gratis) | Curso | https://www.coursera.org/specializations/machine-learning-introduction |
| 3Blue1Brown — Neural Networks | YouTube | https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi |
| Andrej Karpathy — Neural Networks: Zero to Hero | YouTube | https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ |
| Dive into Deep Learning (libro interactivo gratuito) | Libro | https://d2l.ai/ |

### Fase 2 — Reinforcement Learning (3-4 meses) ⭐ pilar 1

**Qué dominar:** MDPs, ecuaciones de Bellman, programación dinámica, Monte Carlo, TD-learning, Q-learning, DQN, policy gradients (REINFORCE, A2C, PPO), actor-critic (SAC), RL offline, introducción a MARL.

| Recurso | Tipo | Enlace |
|---|---|---|
| Sutton & Barto (2ª ed.) — libro de referencia, gratuito | Libro | http://incompleteideas.net/book/the-book-2nd.html |
| David Silver (DeepMind/UCL) — RL Course | YouTube | https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ |
| Stanford CS234 — Reinforcement Learning | YouTube | https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u |
| UC Berkeley CS285 — Deep RL (Sergey Levine) | YouTube | https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps |
| OpenAI Spinning Up in Deep RL | Tutorial + código | https://spinningup.openai.com/ |
| Hugging Face — Deep RL Course (gratuito, con práctica) | Curso | https://huggingface.co/learn/deep-rl-course/unit0/introduction |
| Gymnasium (entornos estándar de RL) | Herramienta | https://gymnasium.farama.org/ |
| Stable-Baselines3 (implementaciones fiables) | Herramienta | https://stable-baselines3.readthedocs.io/ |
| PettingZoo (entornos MARL) | Herramienta | https://pettingzoo.farama.org/ |

**Hito práctico:** implementar DQN y PPO desde cero; resolver entornos de Gymnasium; reproducir un experimento MARL sencillo en PettingZoo.

### Fase 3 — Federated Learning (2-3 meses) ⭐ pilar 2

**Qué dominar:** FedAvg y variantes (FedProx, SCAFFOLD), no-IID, personalización, eficiencia de comunicación, privacidad (DP, SecAgg, HE), robustez bizantina, frameworks.

| Recurso | Tipo | Enlace |
|---|---|---|
| Kairouz et al. — "Advances and Open Problems in FL" | Paper | https://arxiv.org/abs/1912.04977 |
| DeepLearning.AI — Intro to Federated Learning (con Flower) | Curso corto | https://www.deeplearning.ai/short-courses/intro-to-federated-learning/ |
| DeepLearning.AI — Federated Fine-tuning of LLMs | Curso corto | https://www.deeplearning.ai/short-courses/intro-to-federated-fine-tuning-of-llms-with-flower/ |
| Flower — tutoriales oficiales "What is FL" | Tutorial | https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html |
| Flower Labs — canal de YouTube (tutoriales + Flower AI Summit) | YouTube | https://www.youtube.com/@flwrlabs |
| Flower Baselines (reproducción de papers FL) | Código | https://flower.ai/docs/baselines/ |
| NVIDIA FLARE (FL industrial, agregación segura, auditoría) | Herramienta | https://github.com/NVIDIA/NVFlare |
| OpenFL (Intel/Linux Foundation) | Herramienta | https://github.com/securefederatedai/openfl |
| Opacus (DP para PyTorch) | Herramienta | https://opacus.ai/ |
| Google — Federated Learning comic (intuición inicial) | Divulgación | https://federated.withgoogle.com/ |

**Hito práctico:** montar una federación simulada con Flower (≥10 clientes, datos no-IID con particiones Dirichlet), comparar FedAvg vs. FedProx, y añadir privacidad diferencial con Opacus.

### Fase 4 — Federated Reinforcement Learning (2-3 meses) ⭐ intersección

**Qué dominar:** taxonomía HFRL/VFRL, convergencia de policy gradients federados, FMARL, FRL asíncrono, ataques/defensas específicos de FRL.

| Recurso | Tipo | Enlace |
|---|---|---|
| Qi et al. (2021) — survey fundacional de FRL | Paper | https://arxiv.org/abs/2108.11887 |
| Survey FMARL (Jing et al., 2025) | Paper | https://www.sciencedirect.com/science/article/abs/pii/S0957417425023474 |
| Survey MARL federado/descentralizado (2025) | Paper | https://arxiv.org/abs/2507.06278 |
| FRL asíncrono con policy gradients | Paper | https://arxiv.org/abs/2404.08003 |
| Khodadadian et al. — linear speedup (ICML 2022) | Paper | https://proceedings.mlr.press/v162/khodadadian22a.html |
| FRL en IoT (Pinto Neto et al., 2023) | Paper | https://www.mdpi.com/2076-3417/13/11/6497 |

**Hito práctico:** combinar Flower + Gymnasium/Stable-Baselines3 para federar el entrenamiento de una política (p. ej., varios clientes entrenando PPO en variantes del mismo entorno con dinámicas distintas y agregando parámetros). Este prototipo es ya un embrión de la tesis.

### Fase 5 — Data Spaces y marco regulatorio europeo (2 meses) ⭐ pilar 3

**Qué dominar:** IDS-RAM, Gaia-X Trust Framework, DSSC Blueprint, conectores EDC, políticas ODRL, EHDS, Data Act / Data Governance Act / AI Act.

| Recurso | Tipo | Enlace |
|---|---|---|
| IDSA — Reference Architecture Model (IDS-RAM 4) | Especificación | https://docs.internationaldataspaces.org/ids-knowledgebase/ |
| IDSA — canal de YouTube | YouTube | https://www.youtube.com/@InternationalDataSpaces |
| Gaia-X — documentación y Trust Framework | Especificación | https://docs.gaia-x.eu/ |
| Gaia-X — canal de YouTube (summits, tech deep dives) | YouTube | https://www.youtube.com/@Gaia-X_AISBL |
| Data Spaces Support Centre — Blueprint | Guía | https://dssc.eu/space/BVE/357073006/Data+Spaces+Blueprint |
| Eclipse Dataspace Components (EDC) — repositorio y docs | Código | https://github.com/eclipse-edc/Connector |
| EDC — Minimum Viable Dataspace (MVD, para montar tu propio data space de pruebas) | Código | https://github.com/eclipse-edc/MinimumViableDataspace |
| Reglamento (UE) 2025/327 — EHDS (texto oficial) | Norma | https://eur-lex.europa.eu/eli/reg/2025/327/oj |
| Comisión Europea — página oficial EHDS | Web | https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en |
| JMIR (2025) — "Reality Check: EHDS y análisis descentralizado" | Paper | https://www.jmir.org/2025/1/e76491 |
| Survey "Navigating the Data Space Landscape" (2025) | Paper | https://arxiv.org/abs/2509.06983 |
| Catena-X (espacio de datos de automoción, caso real) | Web | https://catena-x.net/ |

**Hito práctico:** desplegar el Minimum Viable Dataspace de EDC en local; entender el flujo completo: catálogo → negociación de contrato → política de uso → transferencia.

### Fase 6 — Integración, prototipo y propuesta de tesis (3-4 meses)

1. **Revisión sistemática propia**: mapear la literatura FRL ∩ data spaces (Scopus/Web of Science/arXiv) y documentar formalmente el gap. Una *systematic literature review* publicable es un excelente primer artículo de doctorado.
2. **Prototipo integrador**: FRL (Flower + SB3) ejecutándose *sobre* una infraestructura de espacio de datos (EDC MVD), donde las actualizaciones de política se intercambian como activos gobernados por contratos/políticas de uso.
3. **Redactar la propuesta de PhD**: pregunta de investigación, hipótesis, metodología, plan de publicaciones, caso de uso sectorial (recomendado: salud/EHDS o fabricación/Manufacturing-X, según el grupo de investigación de destino).
4. **Vigilancia científica continua**: alertas en Google Scholar ("federated reinforcement learning", "data space" + "federated learning", "EHDS federated"); seguimiento de NeurIPS, ICML, ICLR, AAMAS, AAAI y los workshops FL@conferencias (p. ej. International Workshop on Federated Learning); arXiv cs.LG y cs.DC.

---

## 3. Herramientas que debes dominar (resumen)

| Capa | Herramientas |
|---|---|
| Deep Learning | PyTorch |
| RL | Gymnasium, Stable-Baselines3, RLlib (Ray), PettingZoo (MARL) |
| FL | **Flower** (investigación, el más flexible), NVIDIA FLARE (industrial), OpenFL, TensorFlow Federated |
| Privacidad | Opacus (DP), OpenMined/PySyft, SEAL/TenSEAL (HE) |
| Data Spaces | Eclipse EDC + MVD, FIWARE, Pontus-X (Gaia-X) |
| Reproducibilidad | Docker, Weights & Biases / MLflow, Hydra |

---

## 4. Comunidades y eventos donde estar presente

- **Flower AI Summit** (anual, charlas en su canal de YouTube) — https://flower.ai/events/
- **IDSA events / Data Spaces Symposium** — https://internationaldataspaces.org/
- **Gaia-X Summit** (anual) — https://gaia-x.eu/
- **BDVA / Data Week** (Big Data Value Association, muy activa en España) — https://bdva.eu/
- **Workshops académicos**: FL-NeurIPS/ICML, FLIRT, DISCOLI; conferencias AAMAS (multiagente), MLSys (sistemas).
- En España: red de espacios de datos impulsada por Gaia-X Hub Spain (https://gaiax.es/) y el Kit Espacios de Datos — relevante para colaboraciones y financiación de la tesis.

---

## 5. Criterio de "estar listo para empezar la tesis"

Puedes considerarte preparado cuando seas capaz de:

1. Explicar y demostrar la convergencia de FedAvg y de un policy gradient básico.
2. Implementar desde cero PPO y federarlo con Flower sobre entornos heterogéneos.
3. Desplegar un mini espacio de datos con EDC y describir su modelo de gobernanza (contratos, políticas ODRL, identidad).
4. Citar y posicionar los 10-15 papers clave de FRL (sección 1.3) y articular en una página el gap FRL ∩ data spaces.
5. Redactar una propuesta de investigación de 6-10 páginas con pregunta, hipótesis, metodología y plan de validación.

---

*Documento generado el 10 de junio de 2026. Verifica los enlaces y la literatura más reciente antes de presentar la propuesta: el campo FRL y la regulación de espacios de datos (actos de ejecución del EHDS hasta 2027) evolucionan trimestre a trimestre.*
