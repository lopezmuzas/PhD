---
title: "Roadmap para aprender Federated Reinforcement Learning (FRL) — preparación para PhD"
tags: [tesis, roadmap]
status: borrador
updated: 2026-08-08
---

# Roadmap para aprender Federated Reinforcement Learning (FRL) — preparación para PhD

> Documento de trabajo, complementario al roadmap de *Federated Learning*. Enlaces verificados en junio de 2026.
> Objetivo: llegar a poder **leer y reproducir papers de frontera de FRL y proponer investigación original** en la intersección de federated learning (FL) y reinforcement learning (RL).

---

## Lee esto primero (cómo es realista aprender FRL)

FRL es un campo **joven y de nicho**. A diferencia de FL "clásico", **no existe un curso abierto dedicado ni una serie de charlas de referencia**. Eso cambia la estrategia de aprendizaje:

1. **No puedes saltarte ninguna de las dos bases.** FRL = RL ∩ FL. Necesitas RL sólido *y* FL sólido antes de tocar los papers de la intersección. Si vienes de mi otro roadmap de FL, ya tienes la mitad.
2. **El aprendizaje es vía papers, no vía vídeos.** Una vez tengas las bases, se aprende leyendo (y reimplementando) la literatura de FRL directamente. Esto es exactamente lo que harás como doctorando, así que es buen entrenamiento.
3. **Hay dos "sabores" de FRL** que debes distinguir desde el principio:
   - **HFRL (Horizontal)**: agentes en *el mismo* espacio estado-acción pero con entornos/datos distintos, que colaboran para aprender una política mejor (el caso más común).
   - **VFRL (Vertical)**: agentes que observan *partes distintas* del mismo entorno (espacios de features distintos).
4. **Hay dos comunidades trabajando en FRL** y conviene leer ambas:
   - **Teóricos** (CMU / Georgia Tech / Penn): garantías de convergencia, *linear speedup*, complejidad muestral. Aquí está la investigación más rigurosa y publicable hoy.
   - **Aplicados** (redes, edge/IoT, robótica, conducción autónoma): sistemas y casos de uso.

---

## Mapa del recorrido

```
Fase A: RL fundamentos  ─┐
                          ├─►  Fase C: FRL conceptual  ─►  Fase D: FRL teórico  ─►  Fase E: FRL aplicado  ─►  Fase F: fronteras
Fase B: FL fundamentos  ─┘        (surveys + base)         (convergencia)          (sistemas/casos)
```

---

## Fase A — Fundamentos de Reinforcement Learning

Si tu RL no es sólido, esto es lo primero. No avances a FRL sin esto.

### Recursos canónicos (estables, gratuitos)
- [ ] **Sutton & Barto — *Reinforcement Learning: An Introduction* (2ª ed.)** — la biblia de RL, libre:
  http://incompleteideas.net/book/the-book-2nd.html
- [ ] **David Silver — RL Course (DeepMind/UCL)** — el curso en vídeo de referencia (playlist YouTube oficial de DeepMind):
  https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
- [ ] **OpenAI — *Spinning Up in Deep RL*** — el mejor recurso práctico para deep RL (con código):
  https://spinningup.openai.com/
- [ ] *(Opcional, muy didáctico)* **Hugging Face — Deep RL Course** (práctico, con notebooks):
  https://huggingface.co/learn/deep-rl-course/

### Conceptos que debes dominar al salir de esta fase
- [ ] MDPs, función de valor, Q-función, ecuación de Bellman.
- [ ] **Q-learning** y **TD-learning** (claves: casi toda la teoría de FRL es sobre estos).
- [ ] **Policy gradient / actor-critic**, **PPO** (Schulman et al. 2017: https://arxiv.org/abs/1707.06347).
- [ ] Diferencia **on-policy vs off-policy**, **online vs offline RL**.
- [ ] Concepto de **muestreo markoviano** (las muestras de RL no son IID — esto es central en la teoría de FRL).

---

## Fase B — Fundamentos de Federated Learning

Cubierto en el roadmap de FL. Mínimo imprescindible antes de FRL:

- [ ] **FedAvg** — McMahan et al. 2017: https://arxiv.org/abs/1602.05629
- [ ] **Heterogeneidad / client drift** (FedProx, SCAFFOLD) y por qué el promediado falla con datos non-IID.
- [ ] **Privacidad**: differential privacy y secure aggregation (al menos la intuición).
- [ ] El concepto de **linear speedup**: si N clientes colaboran, ¿se reduce el error/complejidad por un factor N? Es *la* pregunta que la teoría de FRL intenta responder para RL.

---

## Fase C — FRL conceptual: surveys y papers fundacionales

**Meta**: tener el mapa completo del subcampo y leer los primeros papers que lo definieron.

### Surveys (lectura de orientación)
- [ ] **Qi, Zhou, Lei, Zheng (2021) — *Federated Reinforcement Learning: Techniques, Applications, and Open Challenges*** (el survey de referencia; define HFRL vs VFRL):
  https://arxiv.org/abs/2108.11887
- [ ] **Cheruiyot et al. (2025) — *A Survey of Multi-Agent RL: Federated Learning and Cooperative/Noncooperative Decentralized Regimes*** (survey reciente que sitúa FRL dentro de MARL):
  https://arxiv.org/abs/2507.06278

### Papers fundacionales (los primeros del campo)
- [ ] **Zhuo, Feng, Xu, Yang, Lin (2019) — *Federated Deep Reinforcement Learning (FedRL)*** (introduce el término; caso VFRL, usa diferenciales gaussianas para privacidad):
  https://arxiv.org/abs/1901.08277
- [ ] **Liu, Wang, Liu (2019) — *Lifelong Federated Reinforcement Learning: A Learning Architecture for Navigation in Cloud Robotic Systems*** (IEEE RA-L; el caso aplicado canónico, robótica):
  https://arxiv.org/abs/1901.06455

### Práctica (a partir de tu base de FL)
- [ ] Extiende tu implementación de FedAvg para que cada "cliente" sea un **agente Q-learning tabular** en un entorno tipo Grid-world (con ligeras variaciones entre agentes). Promedia las Q-tablas en el servidor. Observa si la colaboración acelera el aprendizaje. Esto es, en miniatura, federated Q-learning.

---

## Fase D — FRL teórico (el corazón de la investigación publicable hoy)

**Meta**: entender las garantías de convergencia y el fenómeno de *linear speedup* bajo muestreo markoviano y heterogeneidad. Aquí es donde, muy probablemente, nazca un tema de tesis riguroso.

> Estos papers son técnicos. Léelos *después* de tener RL + FL sólidos. Léelos con papel y lápiz.

- [ ] **Khodadadian, Sharma, Joshi, Maguluri (2022, ICML) — *Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling*** (paper teórico fundacional: prueba speedup lineal para TD y Q-learning federados):
  arXiv: https://arxiv.org/abs/2206.10185 · PMLR: https://proceedings.mlr.press/v162/khodadadian22a.html
- [ ] **Woo, Joshi, Chi (2023, ICML) — *The Blessing of Heterogeneity in Federated Q-Learning: Linear Speedup and Beyond*** (cómo la heterogeneidad puede *ayudar*; algoritmo de *importance averaging*):
  arXiv: https://arxiv.org/abs/2305.10697 · PMLR: https://proceedings.mlr.press/v202/woo23a.html
- [ ] **Jin et al. (2022) — *Federated Reinforcement Learning with Environment Heterogeneity*** (qué pasa cuando los entornos de los agentes difieren):
  https://arxiv.org/abs/2204.02634
- [ ] **Woo, Shi, Joshi, Chi (2024, ICML) — *Federated Offline Reinforcement Learning: Collaborative Single-Policy Coverage Suffices*** (FRL en el setting offline, muy relevante hoy):
  https://arxiv.org/abs/2406.05515
- [ ] **Wang, Mitra, Hassani, Pappas, Anderson (2023) — *Federated TD Learning with Linear Function Approximation under Environmental Heterogeneity***:
  https://arxiv.org/abs/2302.02212

### Conceptos clave que debes poder explicar al terminar
- [ ] Por qué el muestreo markoviano rompe los análisis IID estándar de FL.
- [ ] Qué es *linear speedup* y bajo qué condiciones se consigue (o se pierde) en RL federado.
- [ ] El trade-off **sample complexity vs communication complexity** (ver Salgia & Chi 2024: https://arxiv.org/abs/2408.16981).

---

## Fase E — FRL aplicado (heterogeneidad, robustez, casos de uso)

**Meta**: ver cómo se lleva FRL a la práctica y dónde están los problemas de ingeniería.

- [ ] **FedKL — *Tackling Data Heterogeneity in FRL by Penalizing KL Divergence*** (transfiere la idea de FedProx/PPO a FRL):
  https://arxiv.org/abs/2204.08125
- [ ] **Fault-Tolerant FRL with Theoretical Guarantee** (robustez ante agentes que fallan):
  https://arxiv.org/abs/2110.14074
- [ ] **Federated Deep RL for IoT with Decentralized Cooperative Edge Caching** (Wang et al. 2020, IEEE IoT-J) — caso de uso clásico en edge/IoT.
- [ ] **Federated Transfer Reinforcement Learning for Autonomous Driving** (Liang et al. 2019):
  https://arxiv.org/abs/1910.06001

### Práctica recomendada (proyecto de portfolio para PhD)
- [ ] Implementa **federated PPO o federated DQN** con varios agentes sobre entornos de **Gymnasium** ligeramente heterogéneos (p. ej. CartPole con masas/longitudes distintas por agente). Compara: (a) cada agente solo, (b) FedAvg de políticas, (c) un método que penalice la divergencia (estilo FedKL). Mide reward vs rondas de comunicación. Sube el código con un README claro.
  - Gymnasium: https://gymnasium.farama.org/
  - Para la parte federada puedes apoyarte en **Flower** (https://flower.ai/), que es agnóstico al tipo de modelo y sirve para RL.

---

## Fase F — Fronteras de investigación (2024–2026)

**Meta**: identificar huecos abiertos para formular preguntas de tesis.

- **FRL offline y trade-off muestra/comunicación**: línea muy activa del grupo Chi/Joshi.
- **FRL bajo restricciones de comunicación reales** (canales con borrado, compresión, *over-the-air*): ver trabajos de Dal Fabbro, Mitra, Pappas.
- **FRL + privacidad formal (DP)**: sorprendentemente poco explorado de forma rigurosa; oportunidad clara.
- **Federated MARL (multi-agente)**: ver el survey de 2025 (arXiv:2507.06278) y la encuesta de FMARL en Expert Systems with Applications (2025).
- **Personalización en FRL** (políticas personalizadas con representaciones compartidas): https://arxiv.org/abs/2411.15014
- **Heterogeneidad de restricciones / safe FRL**: https://arxiv.org/abs/2405.03236

> Estrategia de doctorando: elige **una** de estas líneas, lee sus 5–10 papers más recientes en NeurIPS/ICML/ICLR, y busca el supuesto que todos hacen y que podrías relajar. Ese suele ser tu primer paper.

---

## Base de conocimiento: tabla resumen de papers de FRL

| # | Paper | Autores (año) | Tipo | Enlace |
|---|-------|---------------|------|--------|
| 1 | Federated RL: Techniques, Applications, Open Challenges | Qi et al. (2021) | Survey | https://arxiv.org/abs/2108.11887 |
| 2 | Federated Deep RL (FedRL) | Zhuo et al. (2019) | Fundacional (VFRL) | https://arxiv.org/abs/1901.08277 |
| 3 | Lifelong FRL (navegación robótica) | Liu et al. (2019) | Fundacional (aplicado) | https://arxiv.org/abs/1901.06455 |
| 4 | FRL: Linear Speedup Under Markovian Sampling | Khodadadian et al. (2022) | Teoría | https://arxiv.org/abs/2206.10185 |
| 5 | Blessing of Heterogeneity in Federated Q-Learning | Woo, Joshi, Chi (2023) | Teoría | https://arxiv.org/abs/2305.10697 |
| 6 | FRL with Environment Heterogeneity | Jin et al. (2022) | Teoría | https://arxiv.org/abs/2204.02634 |
| 7 | Federated Offline RL | Woo, Shi, Joshi, Chi (2024) | Teoría/offline | https://arxiv.org/abs/2406.05515 |
| 8 | Federated TD Learning under Env. Heterogeneity | Wang et al. (2023) | Teoría (TD) | https://arxiv.org/abs/2302.02212 |
| 9 | FedKL (penalizar KL) | (2022) | Aplicado/heterogeneidad | https://arxiv.org/abs/2204.08125 |
| 10 | Fault-Tolerant FRL | (2021) | Robustez | https://arxiv.org/abs/2110.14074 |
| 11 | Survey MARL + FL | Cheruiyot et al. (2025) | Survey reciente | https://arxiv.org/abs/2507.06278 |
| 12 | Sample-Communication Trade-off in Fed. Q-Learning | Salgia, Chi (2024) | Teoría/frontera | https://arxiv.org/abs/2408.16981 |

---

## Prerrequisitos de RL (lista consolidada de vídeos/recursos)

| Recurso | Para qué | Enlace |
|---------|----------|--------|
| Sutton & Barto (libro) | Teoría RL completa | http://incompleteideas.net/book/the-book-2nd.html |
| David Silver — RL Course | Curso en vídeo de referencia | https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ |
| OpenAI Spinning Up | Deep RL práctico + código | https://spinningup.openai.com/ |
| HF Deep RL Course | Práctico con notebooks | https://huggingface.co/learn/deep-rl-course/ |
| Gymnasium | Entornos para experimentar | https://gymnasium.farama.org/ |

---

## Investigadores / grupos a seguir (FRL)

- **Gauri Joshi** (CMU) — teoría de FRL, linear speedup. Es probablemente la referencia central del subcampo.
- **Yuejie Chi** (CMU) — teoría de RL y Q-learning federado.
- **Siva Theja Maguluri** (Georgia Tech) — stochastic approximation, convergencia.
- **Aritra Mitra / George J. Pappas** (Penn) — FRL bajo restricciones de comunicación.
- **Jiin Woo** (CMU) — primera autora de varios de los papers clave recientes.

Pon sus Google Scholar en alertas. En FRL, seguir a 4–5 personas cubre casi toda la frontera teórica.

---

## Dónde se publica FRL

- **Teoría**: NeurIPS, ICML, ICLR, AISTATS; revistas IEEE TSP / IEEE Control Systems Letters (L-CSS) para la parte de stochastic approximation/control.
- **Aplicado**: IEEE IoT Journal, IEEE T-WC (wireless), conferencias de robótica (ICRA/IROS) para el lado de navegación/robots.
- **Revista generalista ML**: TMLR.

---

## Checklist de "estoy listo para investigar en FRL"

- [ ] Puedo derivar Q-learning y TD-learning y explicar por qué sus muestras no son IID.
- [ ] Entiendo FedAvg y el problema de heterogeneidad en FL.
- [ ] Puedo explicar qué es *linear speedup* y por qué es no trivial conseguirlo en RL.
- [ ] Distingo HFRL de VFRL y sé dar un ejemplo de cada uno.
- [ ] He leído al menos 2 papers teóricos (Khodadadian 2022, Woo 2023) y entiendo sus supuestos.
- [ ] He implementado un federated Q-learning o federated PPO básico con agentes heterogéneos.
- [ ] He identificado una línea (offline FRL / comunicación / privacidad / MARL) y leído sus papers recientes.

---

*Roadmap creado en junio de 2026. FRL evoluciona rápido y muchos resultados son recientes: prioriza papers de los últimos 18 meses en NeurIPS/ICML/ICLR y revisa las versiones más actuales en arXiv. Documento complementario al roadmap de Federated Learning.*
