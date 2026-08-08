---
title: "Roadmap Maestro — De cero a doctorado en Federated Reinforcement Learning sobre Dataspaces (Pontus-X / Gaia-X)"
tags: [tesis, roadmap]
status: borrador
updated: 2026-08-08
---

# Roadmap Maestro — De cero a doctorado en Federated Reinforcement Learning sobre Dataspaces (Pontus-X / Gaia-X)

> **Versión 2 — revisada con foco en doctorado aplicado a Federated RL sobre dataspaces**
>
> Este documento sustituye el roadmap original de 4 fases. Mantiene su columna vertebral pero corrige tres problemas críticos: (1) introduce Offline RL como bloque propio antes de Federated RL, porque tu setup real (Compute-to-Data en Pontus-X) es offline por construcción; (2) separa Federated Learning clásico de Federated RL; (3) sustituye la Fase 4 genérica por una Fase de especialización doctoral concreta.

---

## Tabla de fases (visión general)

| Fase | Foco | Duración | Salida tangible |
|---|---|---|---|
| **0** | Prerrequisitos mat. y código | 6–10 sem | Sutton & Barto cap. 1–3 entendidos |
| **1** | RL Tabular | 6–8 sem | Q-Learning/SARSA from-scratch en GridWorld y FrozenLake |
| **2** | Deep RL | 8–10 sem | DQN propio que supere Atari Pong en píxeles |
| **3** | SOTA online | 8–10 sem | PPO y SAC propios, agente en MuJoCo |
| **4** | **Offline RL** *(nuevo bloque)* | 6–8 sem | CQL/IQL replicados en D4RL; primera "tesis-relevant" pipeline |
| **5** | **Federated Learning clásico** *(nuevo bloque)* | 4–6 sem | FedAvg, non-IID, DP en Flower; deploy en Pontus-X testnet |
| **6** | **Federated RL** *(la fase de tu tesis)* | continuo | Tu contribución original |

Tiempo total estimado hasta empezar contribución doctoral original: **12–18 meses** a ritmo sostenido (no maratón). Después: 3+ años de doctorado real.

---

## Fase 1 — RL Tabular y Fundamentos Teóricos

**Objetivo**: que las ecuaciones de Bellman y el algoritmo de Q-Learning sean parte de tu intuición, no fórmulas que recuerdas. Que sepas demostrar (no solo enunciar) la convergencia de TD(0) en el caso tabular.

### Cambios sobre tu propuesta

Tu plan dice "lee los primeros 6 capítulos de Sutton & Barto". Súbelo a **capítulos 1–8**. El capítulo 7 (n-step bootstrapping) y el 8 (planning y Dyna-Q) son fundamentales: Dyna-Q es el ancestro directo de cualquier método model-based moderno, y la idea de "usar un modelo aprendido para generar experiencia sintética" es exactamente lo que motiva ciertos enfoques en Offline RL y Federated RL.

### Temario ampliado

MDPs finitos. Ecuaciones de Bellman de expectativa y de optimalidad. Programación dinámica: policy iteration y value iteration, y sus propiedades de convergencia (contracción de Banach). Monte Carlo prediction y control. TD(0), SARSA, Q-Learning, Expected SARSA. n-step methods y TD(λ). Planning con modelos (Dyna-Q). Exploration vs. exploitation: ε-greedy, UCB, optimismo en la inicialización.

**Bloque teórico extra (importante para doctorado)**: lee la sección de Sutton & Barto sobre convergencia de Q-Learning, y al menos hojea un capítulo de Agarwal, Jiang, Kakade & Sun ([RL Theory book, PDF gratis](https://rltheorybook.github.io/)) sobre PAC-MDP y sample complexity. No tienes que dominarlo aún, pero te quiero ver familiarizado con que "Q-Learning converge" es un teorema, no una observación experimental.

### Recursos clave

- **Sutton & Barto 2nd ed.**: tu biblia. [PDF gratis legal](http://incompleteideas.net/book/RLbook2020.pdf). Capítulos 1–8.
- **David Silver — UCL RL Course** ([YouTube playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)). Diez clases. Silver es co-autor de AlphaGo, y explica con una claridad que no he visto en nadie más. **Imprescindible** verlo en paralelo a Sutton & Barto.
- **Stanford CS234 — Reinforcement Learning (Emma Brunskill)** ([YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX)). Más moderno que Silver, con énfasis fuerte en teoría. La opción si quieres orientación más doctoral.
- **Mutual Information (YouTube)** — serie sobre RL visual al estilo 3Blue1Brown. Refuerzo intuitivo.

### Ejecución

Implementa **desde cero, solo con NumPy**: value iteration, policy iteration, every-visit MC, SARSA, Q-Learning, Dyna-Q. Entornos: GridWorld pequeño hecho a mano y FrozenLake de Gymnasium. Para cada algoritmo: gráfica de convergencia, sensibilidad a `α` y `γ`, comparación con baseline aleatorio. Guarda todo en un repositorio público — tu primer pieza de portfolio doctoral.

### Checkpoint Fase 1

- Demuestras (no solo enuncias) por qué value iteration converge.
- Implementas Q-Learning sin mirar pseudocódigo.
- Sabes explicar por qué Q-Learning es *off-policy* y SARSA es *on-policy*, y cuándo importa esa diferencia.
- Leído Sutton & Barto capítulos 1–8, ojeado capítulo 1 de Agarwal-Jiang-Kakade-Sun.

---

## Fase 2 — Deep Reinforcement Learning

**Objetivo**: el salto de tabular a aproximación de funciones. Entender por qué la combinación "bootstrapping + off-policy + function approximation" es la **Deadly Triad** que rompe la convergencia, y cómo los trucos prácticos (replay buffer, target network, gradient clipping) la mitigan.

### Cambios sobre tu propuesta

Añade **explícitamente** estos puntos que tu versión no menciona:
- **Deadly Triad** y por qué DQN tiene tantos trucos (target nets, replay): no son arbitrarios, atacan la triada.
- **Double DQN, Dueling DQN, Prioritized Replay, Rainbow**: la evolución del DQN original. No es opcional, son SOTA en value-based para discreto.
- **Variance reduction en policy gradients**: baselines, advantages, GAE (Generalized Advantage Estimation). Sin GAE, no entiendes PPO de verdad.
- **Reproducibilidad**: la crisis de RL. Lee [Henderson et al. 2017 — "Deep RL that Matters"](https://arxiv.org/abs/1709.06560). Cambia tu forma de evaluar agentes.

### Temario

Function approximation: lineal y neuronal. DQN: experience replay, target network, ε-greedy con decay. Doble DQN, Dueling DQN, Prioritized Experience Replay, Rainbow. REINFORCE con baseline. Actor-Critic (A2C/A3C). GAE. Entropía como regularizador. La Deadly Triad y por qué importa.

### Recursos

- **Hugging Face Deep RL Course** ([huggingface.co/deep-rl-course](https://huggingface.co/deep-rl-course/unit0/introduction)). Lo que tenías. Muy bueno como primera pasada.
- **Berkeley CS285 — Deep RL (Sergey Levine)** ([YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)). **Es la referencia académica para deep RL.** Levine es uno de los nombres centrales en RL Offline y robot learning. Pesado y largo, pero pasar por aquí es lo que separa un practicante de un doctorando. Para ti: ineludible.
- **OpenAI Spinning Up — Key Papers** ([spinningup.openai.com/.../keypapers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)). La lista corta de papers que tienes que haber leído.
- **CleanRL** ([github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)). Implementaciones single-file con tracking en W&B. Tu referencia de código.

### Ejecución

Implementa tu propio DQN desde cero en PyTorch y haz que supere Pong en Atari con observaciones de píxeles (es duro: paciencia, ~12–24h de GPU). Después, añade Double DQN y Dueling DQN como ablaciones y compara curvas con seeds múltiples y bandas de confianza (no medias sin error bars: eso es **el** error en RL). Implementa REINFORCE y A2C en CartPole y LunarLander.

### Checkpoint Fase 2

- DQN propio supera Atari Pong desde píxeles.
- Sabes explicar la Deadly Triad y qué hace cada truco para mitigarla.
- Implementaste GAE y entiendes cada término.
- Tus gráficas siempre llevan ≥5 seeds, intervalos de confianza y test estadístico mínimo.
- Has leído los 10 papers clave de Spinning Up.

---

## Fase 3 — Algoritmos de Estado del Arte (online)

**Objetivo**: dominar PPO y SAC al nivel de poder modificarlos. Estos serán tus baselines para el resto de la carrera.

### Cambios sobre tu propuesta

- Quita "transiciona a Stable-Baselines3" como objetivo. Úsalo como **baseline de comparación**, no como tu herramienta principal. Para investigación, lo que vas a modificar es tu propia implementación o CleanRL.
- Añade **TD3** explícitamente entre DDPG y SAC: TD3 es el puente entre los dos y es el algoritmo más usado en muchos benchmarks de control continuo.
- Añade un bloque sobre **trust regions y por qué importan matemáticamente**: TRPO antes de PPO, aunque PPO sea la versión práctica. La motivación matemática de TRPO (KL constraint, conservative policy iteration) es la base teórica que justifica PPO y que aparece en Federated RL para garantías de convergencia.

### Temario

TRPO (entiéndelo aunque no lo uses): policy improvement con KL constraint, natural gradient. PPO: clipping, GAE, KL adaptativo. DDPG: deterministic policy gradient, problema de exploración. TD3: twin critics, delayed updates, target policy smoothing. SAC: maximum entropy RL, soft Q-learning, automatic temperature tuning. Distributed training: vectorización de entornos (`SyncVectorEnv`, `AsyncVectorEnv`).

### Recursos

- **Spinning Up — algorithms pages** para PPO, DDPG, TD3, SAC. Pseudocódigo claro y discusión de truquillos prácticos.
- **CleanRL — `ppo.py`, `sac.py`, `td3.py`**: leer línea a línea. Hay versiones para Atari, MuJoCo y continuous control.
- **Berkeley CS285 lectures 5–9 y 13–14**.
- **"The 37 Implementation Details of PPO"** ([ICLR blog post](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)). Pieza de oro. PPO está plagado de trucos no publicados en el paper; este post los recoge todos.

### Ejecución

Implementa PPO y SAC desde cero. Compara tu PPO contra el de Stable-Baselines3 y el de CleanRL en MuJoCo (HalfCheetah, Hopper, Walker2d). Si la diferencia es >15%, tu implementación tiene un bug — busca cuál de los 37 detalles te falta. Documenta cada uno.

**Mini-proyecto pre-doctoral**: replica una mejora pequeña de un paper reciente sobre PPO o SAC y mide si reproduce. Esto te da la dinámica "leer paper + replicar + medir" que será tu día a día doctoral.

### Checkpoint Fase 3

- PPO propio funciona a ±15% del de SB3/CleanRL en MuJoCo.
- Sabes derivar el objective de PPO desde la formulación de trust region.
- Sabes explicar la diferencia matemática entre TD3 y SAC.
- Has reproducido la curva principal de al menos un paper.

---

## Fase 4 — Offline Reinforcement Learning *(nuevo bloque, crítico para ti)*

**Por qué este bloque existe**: en Pontus-X / Compute-to-Data, los datos *no se mueven*. El proveedor de datos los mantiene en su entorno y tú envías un script que se ejecuta allí. En la mayoría de escenarios reales del dataspace, no podrás hacer rollouts online con un entorno simulado: solo tendrás logs históricos del sistema (por ejemplo, telemetría industrial, datos de pacientes, transacciones financieras). Eso **es exactamente Offline RL**. Sin este bloque, tu tesis no tiene base.

### Objetivo

Entender por qué Offline RL es difícil (distributional shift, error de extrapolación en el Q-value), y dominar las dos familias principales de soluciones: regularización conservadora (CQL) y constraint implícito (IQL). Conocer también métodos sin RL clásico que están comiéndole terreno: Decision Transformer y la conexión con secuencia/lenguaje.

### Temario

El setup de Offline RL: política de comportamiento $\pi_\beta$, dataset $\mathcal{D}$, sin acceso al entorno. Por qué TD vanilla falla: extrapolation error sobre acciones fuera del soporte. Familias de soluciones: **policy constraint** (BCQ, BEAR), **value penalization** (CQL — Conservative Q-Learning), **implicit constraint** (IQL — Implicit Q-Learning), **model-based offline** (MOPO, MOReL, COMBO), **sequence modeling** (Decision Transformer, Trajectory Transformer). Off-policy evaluation (OPE): cómo evaluar un agente sin desplegarlo, fundamental para entornos donde no puedes "probar y ver".

### Recursos

- **Sergey Levine et al. — "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"** ([arXiv:2005.01643](https://arxiv.org/abs/2005.01643)). El survey de referencia. Léelo entero, despacio. Es tu nuevo libro de cabecera.
- **Berkeley CS285 — Lectures sobre Offline RL** (Levine, mismo autor del survey).
- **NeurIPS Offline RL Workshop** ([offline-rl-neurips.github.io](https://offline-rl-neurips.github.io/)). Todas las ediciones, los papers y los talks.
- **d3rlpy** ([github.com/takuseno/d3rlpy](https://github.com/takuseno/d3rlpy)): biblioteca PyTorch con implementaciones limpias de CQL, IQL, AWAC, BCQ, TD3+BC. Equivalente a Stable-Baselines3 pero para offline.
- **D4RL benchmark** ([github.com/Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL)). Datasets estándar (MuJoCo locomotion, AntMaze, Kitchen, Adroit). Tu campo de juego.
- **CORL** ([github.com/tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)): "CleanRL para offline". Implementaciones single-file de CQL, IQL, DT, etc.

### Ejecución

Replica CQL y IQL en D4RL (al menos halfcheetah-medium-replay y antmaze-medium-play). Reproduce las tablas del paper de IQL — esa es la prueba de que controlas el bloque. Implementa Decision Transformer y entiende por qué "tratar RL como predicción de secuencias" funciona; esto es muy relevante en federado porque encaja con paradigmas de pre-entrenamiento de fundación.

**Mini-proyecto puente a Fase 5**: monta un setup donde el dataset esté **particionado entre 3 nodos distintos** (todavía sin federación real, solo simulada) y entrena CQL local en cada uno. Compara con CQL entrenado sobre la unión. Esto te da la motivación experimental directa para Federated RL.

### Checkpoint Fase 4

- Replicas resultados de CQL e IQL en al menos 3 entornos D4RL.
- Sabes explicar el `expectile loss` de IQL y por qué evita queries OOD.
- Has leído entero el survey de Levine 2020.
- Empiezas a saber qué papers de offline son SOTA en 2025–2026 (TD7, ReBRAC, MOPO actualizados, etc.).

---

## Fase 5 — Federated Learning clásico (preparación)

**Por qué este bloque existe**: Federated RL = Federated Learning aplicado al setting secuencial. No puedes construir lo segundo sin dominar lo primero. Además, todos los conceptos de privacy-preserving (DP, secure aggregation, MPC) viven aquí, no en RL.

### Objetivo

Implementar FedAvg desde cero, entender heterogeneidad estadística (non-IID) y de sistemas, conocer las técnicas de privacy (Differential Privacy con DP-SGD, Secure Aggregation), y desplegar algo simple en infraestructura realista (Flower + Pontus-X testnet).

### Temario

FedAvg (McMahan et al., 2017): formulación, intuición, garantías básicas. Heterogeneidad: non-IID data, FedProx, SCAFFOLD. Comunicación eficiente: compresión, sparsification, local steps. Privacy: Differential Privacy (definición, mecanismo Gaussiano, DP-SGD de Abadi et al.), Secure Aggregation (Bonawitz et al.), introducción a Homomorphic Encryption y MPC (a nivel conceptual). Ataques: gradient inversion, membership inference, poisoning. Defensas básicas. **Personalized FL**: cuando el objetivo no es un modelo global sino varios modelos adaptados.

### Recursos

- **Survey 2025: "Federated Learning: A Survey on Privacy-Preserving Collaborative Intelligence"** ([arXiv:2504.17703](https://arxiv.org/abs/2504.17703)). Survey reciente y de calidad, con cobertura del estado actual incluyendo la conexión con RL.
- **Curso CMU 11-868 — Federated Learning** (búscalo en YouTube, hay ediciones de 2023 y 2024).
- **Flower** ([flower.ai](https://flower.ai/)): el framework de FL más usado en investigación. Soporta PyTorch, integra con simuladores y permite deploy real. Tutoriales oficiales son excelentes.
- **PySyft / OpenMined** ([openmined.org](https://openmined.org/)): orientado a privacy (DP, MPC, HE). Más experimental que Flower pero útil para entender los building blocks.
- **Opacus** ([opacus.ai](https://opacus.ai/)): librería de PyTorch para DP-SGD. Imprescindible si quieres tocar DP en serio.
- **Pontus-X Documentation** ([docs.pontus-x.eu](https://docs.pontus-x.eu/)) y **deltaDAO tutorials**: para entender Compute-to-Data, Ocean Protocol, y cómo se publican algoritmos en el dataspace.

### Ejecución

Implementa FedAvg desde cero (sin Flower) para que entiendas el ciclo `local_train → aggregate → broadcast`. Después migra a Flower y replica MNIST federado, CIFAR-10 federado con heterogeneidad non-IID controlada. Añade DP con Opacus y mide la degradación de accuracy en función de ε.

**Proyecto puente al doctorado**: despliega tu pipeline federado en una testnet de Pontus-X siguiendo su documentación. Aunque sea Hello World, te da el conocimiento operacional del ecosistema que muchos investigadores en FL nunca tocan.

### Checkpoint Fase 5

- FedAvg propio funciona en MNIST distribuido entre 10 clientes IID y non-IID.
- Sabes explicar por qué FedAvg falla en non-IID extremo y qué hacen FedProx y SCAFFOLD al respecto.
- Has implementado DP-SGD con Opacus y dibujado la curva privacy-utility (ε vs. accuracy).
- Has publicado un algoritmo simple (puede ser inferencia, no hace falta RL aún) en una testnet de Pontus-X.

---

## Fase 6 — Federated Reinforcement Learning *(el terreno de tu tesis)*

**Aquí ya no eres alumno; eres investigador.** Esta fase no termina: es donde vives durante el doctorado. Lo que sigue es solo el on-ramp.

### Objetivo

Conocer el estado del arte de FRL, identificar gaps abiertos relevantes para dataspaces, y empezar a producir contribuciones originales.

### Subdominios y dónde encaja Pontus-X

El campo de FRL se organiza en tres ejes ortogonales que necesitas tener clarísimos:

**Eje 1 — Tipo de heterogeneidad**: ¿los agentes comparten entorno (mismo MDP) o entornos distintos (MDPs heterogéneos)? El segundo caso es mucho más difícil y es donde está la frontera.

**Eje 2 — Qué se federa**: ¿gradientes de la red (FedAvg-style: QAvg, PAvg), modelos de entorno, representaciones latentes, o trajectories/datasets (más cercano a Offline FRL)?

**Eje 3 — Online vs. Offline**: ¿los agentes interactúan con su entorno (Online FRL) o solo tienen datasets fijos (Offline FRL)? **Para Pontus-X y Compute-to-Data, tu setup canónico es Offline FRL**: cada participante tiene un dataset histórico local y nunca lo expone. Esto es **una intersección poco explorada todavía** y es probablemente donde está la oportunidad de tesis original.

### Lecturas iniciales obligatorias

- **Qi et al. — "Federated Reinforcement Learning: Techniques, Applications, and Open Challenges"** ([arXiv:2108.11887](https://arxiv.org/abs/2108.11887)). El survey clásico de FRL, agrupa los trabajos en *Horizontal* y *Vertical* FRL por analogía con FL.
- **Jin et al. — "Federated Reinforcement Learning with Environment Heterogeneity"** ([arXiv:2204.02634](https://arxiv.org/abs/2204.02634)). Introduce QAvg y PAvg, las extensiones naturales de FedAvg a Q-Learning y policy gradients. Lectura básica.
- **Mitra et al. — "Towards Fast Rates for Federated and Multi-Task Reinforcement Learning"** ([arXiv:2409.05291](https://arxiv.org/abs/2409.05291)). Análisis teórico moderno con garantías de convergencia. Si tu tesis va por la rama teórica, este es tu paper.
- **Wang & Anwar (Columbia)** — tesis doctoral sobre Federated Learning for Reinforcement Learning and Control. Lectura completa cuando hayas leído los anteriores.
- **Surveys recientes 2024–2025** sobre FRL que aparezcan en arXiv en CS.LG / CS.DC. Búsqueda mensual ineludible.

### Vías de contribución doctoral (orientativas)

Dado tu setting (Pontus-X / Compute-to-Data), estas son las direcciones con más recorrido:

1. **Offline Federated RL en non-IID extremo**: combinar CQL/IQL con FedAvg cuando cada cliente tiene un dataset generado por una política distinta. Distributional shift agravado por heterogeneidad. **Es una intersección con relativamente poco trabajo publicado todavía.**

2. **FRL con privacy garantizada**: integrar DP-SGD o secure aggregation en el ciclo de actualización de Q-values. Las garantías estándar de DP se vuelven sutiles cuando el dataset es una trayectoria correlacionada (no IID por construcción).

3. **Personalized FRL**: cada cliente del dataspace tiene un MDP con dinámicas ligeramente distintas (porque cada fábrica/hospital es ligeramente distinto). El objetivo no es un agente único, sino un meta-agente que se adapta rápido. Conexión con meta-RL y MAML.

4. **Comunicación-eficiente FRL**: en Pontus-X, mover modelos cuesta dinero (literal: hay tokens). Reducir bytes comunicados es una métrica con valor de negocio directo.

5. **Sim2Real federado**: si parte del dataspace publica datos reales y otra parte simuladores, ¿cómo combinas ambos para entrenar un agente que se despliegue real?

Cualquiera de las cinco da para una tesis. Te recomiendo elegir 1+2 como combinación porque están alineados (Offline + DP en dataspaces) y porque la literatura es lo bastante delgada como para que tengas hueco.

### Ejecución continua durante el doctorado

Reading group semanal o quincenal con compañeros del laboratorio. Reproducción del paper más relevante de cada mes. Replicación inicial del paper de Jin et al. (QAvg/PAvg) como tu primer experimento real de FRL. Después, montaje de un benchmark propio que combine D4RL con particionado federado non-IID — si no existe uno bueno hoy, **publicarlo como dataset/benchmark es ya una primera contribución doctoral perfectamente válida** (workshops de NeurIPS y de ICLR sobre benchmarks aceptan esto).

### Métricas de progreso doctoral realistas

- **Año 1**: dominio del estado del arte, primer workshop paper (rechazo posible, no pasa nada).
- **Año 2**: primera contribución original aceptada (workshop fuerte o conferencia secundaria); benchmark propio publicado.
- **Año 3**: contribución principal aceptada en venue top (NeurIPS, ICML, ICLR) o equivalente; tesis articulada en tres papers.
- **Año 4**: defensa.

No es lineal. Ningún doctorado lo es.

---

## Cosas que tu roadmap original mencionaba y que **he movido o eliminado a propósito**

**RLHF**: si tu tesis es Federated RL sobre dataspaces, RLHF probablemente no es tu tema. RLHF vive en el ecosistema de LLMs (alignment, DPO, GRPO). Toca tangencialmente RL pero los problemas son muy distintos. Apártalo de tu camino salvo que tu director te lo pida explícitamente — meterlo como bloque obligatorio te dispersa.

**Sim2Real**: relevante si tu tesis es robótica. Para dataspaces industriales puede serlo, para dataspaces médicos/financieros no. Decisión a tomar con tu director.

**MARL (Multi-Agent RL)**: hay una superposición conceptual con FRL pero son **campos distintos**. MARL estudia agentes que **comparten entorno y se ven entre sí** (juegos, mercados, equipos de robots); FRL estudia agentes que **están en entornos separados y solo comparten parámetros**. Si tu tesis es FRL, MARL es lectura lateral, no central.

**"Leer ArXiv semanal"**: lo cambio por *reading group* + lectura dirigida por tu director. Leer ArXiv solo es agotador, ruidoso y poco eficiente sin alguien con quien discutir.

**"Competiciones NeurIPS"**: optativo, no obligatorio. Una buena competición puede dar visibilidad; mala gestión del tiempo te puede comer 6 meses sin producir paper.

---

## Cómo seguimos

Cuando termines la Fase 0 (o el bloque concreto en el que estés ahora), me pides que profundicemos en cualquiera de estas fases con su propio documento dedicado. La Fase 4 (Offline RL) y la Fase 6 (Federated RL) son las que vas a necesitar con más detalle porque son las menos cubiertas en cursos generalistas — los recursos están dispersos en papers.

Mi sugerencia: cuando llegues a Fase 4, hablamos largo de Offline RL. Cuando estés en Fase 5, te preparo una guía específica de Pontus-X / Compute-to-Data con código de ejemplo. Para Fase 6, lo natural es iterar sobre la propuesta de tesis a medida que vayas leyendo.
