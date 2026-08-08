---
title: "Fase 3 — Algoritmos de Estado del Arte (online): PPO, SAC y compañía"
tags: [rl, refuerzo]
status: borrador
updated: 2026-08-08
---

# Fase 3 — Algoritmos de Estado del Arte (online): PPO, SAC y compañía

> **Objetivo de la fase**: dominar PPO y SAC al nivel de poder **modificarlos**, no solo usarlos. Estos dos algoritmos serán tus **baselines durante el resto del doctorado** — todo paper de Federated RL los compara contra ellos, y tú vas a hacer lo mismo. Si los conoces a fondo, los demás (TD3, DDPG, TRPO) los entiendes como puntos de un mismo árbol.
>
> **Tu situación de partida**: vienes de Fase 2 con un DQN propio que supera Pong desde píxeles, GAE implementado a mano, y los 10 papers clave de Spinning Up leídos. Eso significa que **el salto a Fase 3 es conceptualmente pequeño pero técnicamente exigente**: la diferencia entre un PPO que funciona y uno que falla está en detalles de implementación, no en la teoría.
>
> **Tiempo estimado realista**: 8–10 semanas a ritmo de 1–2 h/día más una sesión larga de fin de semana para experimentos. No es opcional dedicar tiempo a debugging: en RL online, el ciclo "implementar → entrenar 8h → ver si funciona" es lento, y por eso hay que ser meticuloso desde el principio.

---

## Cómo usar esta guía

Esta guía es un **índice maestro** de la Fase 3. Está organizada en bloques que siguen el orden lógico de aprendizaje (TRPO → PPO → DDPG → TD3 → SAC), no por importancia. Dentro de cada bloque tienes:

- **Por qué importa para tu doctorado** → la conexión directa con Federated RL y Offline RL
- **Recursos principales** → curso o paper central por el que te guías
- **Refuerzo en vídeo/audio** → tu formato preferido
- **Lectura en inglés** → para integrar el idioma técnico y los papers seminales
- **Ejecución mínima** → no te puedes saltar esto: la fase se cierra implementando
- **Checklist** → criterios concretos para saber que ese bloque está dominado

Cuando termines un bloque, vuelves aquí, lo marcas, y me pides que profundicemos en el siguiente con derivaciones detalladas, ejemplos de código y debugging de los bugs más comunes.

---

## Mapa de la Fase 3

| Bloque | Tema | Peso |
|---|---|---|
| 3.1 | Policy Gradients refresco + Trust Regions (TRPO) | 🔥🔥🔥 Base teórica |
| 3.2 | PPO — el caballo de batalla | 🔥🔥🔥🔥🔥 Crítico, el que más vas a tocar |
| 3.3 | DDPG — el primer paso al control continuo | 🔥🔥 Histórico, contexto |
| 3.4 | TD3 — los tres trucos que arreglaron DDPG | 🔥🔥🔥 Imprescindible |
| 3.5 | SAC — Maximum Entropy RL | 🔥🔥🔥🔥🔥 Crítico, tu otro baseline |
| 3.6 | Distributed training y vectorización de entornos | 🔥🔥🔥 Práctico |
| 3.7 | Reproducibilidad seria y reporting estadístico | 🔥🔥🔥🔥 Doctoral |

---

## 3.1 — Policy Gradients y Trust Regions (TRPO)

### Por qué importa para tu doctorado

PPO es la versión práctica de TRPO. **No puedes entender PPO de verdad sin TRPO**, porque PPO es una aproximación heurística de un objetivo riguroso. La motivación matemática de TRPO — *conservative policy iteration*, KL constraint, monotonic improvement — es exactamente el tipo de argumento que vas a ver en los papers teóricos de Federated RL cuando quieran demostrar convergencia bajo agregación de políticas. Mitra et al. 2024, por ejemplo, usa este lenguaje para sus garantías. Si tu tesis va por la rama teórica, este bloque es donde se planta esa semilla.

### Recurso principal (vídeo, EN)

- **Berkeley CS285 — Lecture 9: Advanced Policy Gradients** → [YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps). Levine deriva TRPO desde *policy improvement bound* paso a paso. Es la mejor explicación pública que existe.

### Lectura obligatoria

- **Schulman et al. 2015 — "Trust Region Policy Optimization"** → [arXiv:1502.05477](https://arxiv.org/abs/1502.05477). El paper original. Lee al menos las secciones 1–4. La sección 3 (la prueba del *policy improvement bound*) es opcional la primera vez, pero vuelve a ella cuando hagas Fase 6.
- **Kakade & Langford 2002 — "Approximately Optimal Approximate Reinforcement Learning"** → es el paper de *Conservative Policy Iteration* que inspira TRPO. Lectura corta y muy profunda. Lo cito porque su construcción aparece en surveys de Federated RL.

### Refuerzo en vídeo

- **Spinning Up — TRPO** → [spinningup.openai.com/.../trpo.html](https://spinningup.openai.com/en/latest/algorithms/trpo.html). Pseudocódigo y discusión. Léelo después del paper.
- **Mutual Information — Policy Gradient series** → para refrescar el theorema del policy gradient con visualización.

### Temas mínimos

Theorema del policy gradient revisado: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) A^{\pi_\theta}(s,a)]$. Por qué los policy gradients vanilla son inestables (varianza alta + cambios de política bruscos). *Surrogate loss* y la importancia de muestrear con $\pi_{old}$ pero optimizar para $\pi_\theta$. Importance sampling ratio $r(\theta) = \pi_\theta(a|s) / \pi_{old}(a|s)$. KL-divergence como medida de "cuánto te has alejado" de la política anterior. *Natural gradient* y por qué la métrica de Fisher es la elección natural. TRPO: optimización con KL constraint, resolución con CG + line search. Por qué TRPO es teóricamente bonito pero prácticamente pesado (no se implementa con SGD vainilla).

### Ejecución mínima

**No tienes que implementar TRPO desde cero.** Es un dolor (conjugate gradient, Fisher-vector products) y nadie lo usa en producción. Pero sí tienes que:

- Leer una implementación de referencia entera (la de Spinning Up o la de [CleanRL TRPO fork](https://github.com/vwxyzjn/cleanrl)).
- Saber escribir el surrogate loss y derivarlo desde el policy gradient theorem.

### Checklist 3.1

- [ ] Sé derivar el surrogate loss de TRPO desde el policy gradient theorem
- [ ] Entiendo por qué la restricción es KL y no, por ejemplo, distancia euclídea en el espacio de parámetros
- [ ] Sé explicar qué es el natural gradient y por qué la métrica de Fisher importa
- [ ] He leído el paper de TRPO al menos hasta la sección 4

---

## 3.2 — PPO: el caballo de batalla

### Por qué importa para tu doctorado

PPO es el algoritmo de policy gradient **más usado en investigación** ahora mismo. Cuando leas papers de Federated RL con políticas, casi siempre el baseline es PPO o una variante federada de PPO (PPO + FedAvg, PPO + DP-SGD, etc.). Más importante: el famoso paper *"The 37 Implementation Details of PPO"* es **el ejemplo canónico** de que en RL "el algoritmo del paper" y "el algoritmo del código" son cosas distintas. Aprender PPO a fondo es aprender una **lección epistemológica**: la reproducibilidad en RL no es trivial, y tu tesis tendrá que enfrentarse a eso.

### Recurso principal (lectura, EN) — **el bueno bueno**

- **Huang et al. — "The 37 Implementation Details of Proximal Policy Optimization"** → [ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). **Pieza de oro**. Te explica los 37 truquillos no publicados en el paper original que separan un PPO que funciona de uno que no. Después de leerlo, ningún paper de RL te volverá a sorprender.

### Lectura obligatoria

- **Schulman et al. 2017 — "Proximal Policy Optimization Algorithms"** → [arXiv:1707.06347](https://arxiv.org/abs/1707.06347). El paper original. Corto, claro, esencial.
- **Spinning Up — PPO** → [spinningup.openai.com/.../ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html). Pseudocódigo y discusión.

### Refuerzo en vídeo

- **Berkeley CS285 — Lectures 5–9** → cubren policy gradients, actor-critic, GAE y trust regions. Cuando llegues a PPO, vuelve a la lecture 9.
- **Aleksa Gordić (The AI Epiphany) — PPO walkthrough** → [Canal](https://www.youtube.com/@TheAIEpiphany). Vídeos *line-by-line* sobre PPO. Útil cuando estés debuggeando.
- **CleanRL — `ppo.py`** → [GitHub](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py). **Léelo línea a línea con el post de los 37 detalles abierto en paralelo**. Es la mejor sesión de estudio que puedes hacer esta fase.

### Temas mínimos

Clipped surrogate objective: $L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$. Por qué el clipping aproxima la trust region sin necesidad de KL explícito. Entropy bonus para exploración. Value function loss (clipped o no — controversia). GAE (Generalized Advantage Estimation): el parámetro $\lambda$ y el sesgo-varianza que controla. Normalización de advantages (per-batch). Múltiples épocas y minibatches por rollout (PPO es *on-policy* pero con varias actualizaciones por batch, lo cual es sutil). Learning rate annealing. Orthogonal initialization de redes. Observation y reward normalization (running mean/std) — uno de los detalles más infravalorados. Variante "PPO-KL" con coeficiente adaptativo (poco usada, pero importante para Federated RL: hay papers que la usan para limitar drift entre clientes).

### Ejecución mínima

- **Implementa PPO desde cero en PyTorch**. Single file estilo CleanRL. CartPole primero (debe converger en 5 minutos), después LunarLander, después MuJoCo continuo (HalfCheetah, Hopper, Walker2d).
- **Compara con CleanRL y Stable-Baselines3** en las mismas semillas y misma configuración. Si tu PPO está a más de ±15% del de CleanRL en HalfCheetah, **busca cuál de los 37 detalles te falta**. No avances al siguiente bloque hasta resolverlo.
- Documenta en tu repo qué detalles activaste y cuáles no. Esa documentación es ya material publicable para un blog técnico.

### Checklist 3.2

- [ ] Mi PPO supera a ±15% de CleanRL en al menos 2 entornos MuJoCo
- [ ] Sé derivar el clipped objective desde el surrogate de TRPO
- [ ] He implementado GAE a mano y entiendo cada término de la fórmula recursiva $\delta_t + \gamma \lambda \hat{A}_{t+1}$
- [ ] Sé qué hace cada uno de los 37 detalles (al menos los 15 más críticos)
- [ ] Sé qué pasa si me olvido de normalizar los advantages (spoiler: el agente colapsa silenciosamente)

---

## 3.3 — DDPG: el primer paso al control continuo

### Por qué importa para tu doctorado

DDPG **ya no es SOTA**, pero es el ancestro directo de TD3 y SAC. Conocer DDPG y entender **por qué falla** es la mejor manera de apreciar qué resuelven TD3 y SAC. Es un bloque corto pero ineludible: si no, llegas a TD3 sin contexto.

### Lectura

- **Lillicrap et al. 2015 — "Continuous control with deep reinforcement learning"** → [arXiv:1509.02971](https://arxiv.org/abs/1509.02971). El paper de DDPG. Léelo entero, es corto.
- **Spinning Up — DDPG** → [spinningup.openai.com/.../ddpg.html](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
- **Silver et al. 2014 — "Deterministic Policy Gradient Algorithms"** → el paper teórico que justifica usar políticas determinísticas. Si tu tesis va por la rama teórica, léelo; si no, hojéalo.

### Temas mínimos

Deterministic Policy Gradient: $\nabla_\theta J(\theta) = \mathbb{E}_s[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s,a)|_{a=\mu_\theta(s)}]$. Actor determinístico + critic Q. Exploración: cómo añadir ruido (Ornstein-Uhlenbeck o Gaussiano) cuando la política es determinística. Replay buffer (igual que en DQN — off-policy). Target networks con soft update ($\tau$-Polyak). Por qué DDPG es **frágil**: sobrestimación del Q (igual que en DQN, pero aquí no hay max sobre acciones discretas que mitigue), sensibilidad brutal a hiperparámetros.

### Ejecución mínima

- Implementa DDPG en Pendulum-v1 (entorno barato de control continuo). Es suficiente: no necesitas pelearte con MuJoCo aquí porque la lección es **conceptual**.
- Observa empíricamente la inestabilidad: lanza 5 seeds y mira cómo divergen las curvas. Esa es la motivación visual para TD3.

### Checklist 3.3

- [ ] Mi DDPG entrena Pendulum-v1 con éxito moderado (a veces converge, a veces no — eso es DDPG)
- [ ] Sé explicar la diferencia entre policy gradient estocástico y determinístico
- [ ] Sé por qué DDPG necesita exploración añadida (la política no la incluye por sí misma)
- [ ] Veo en mis gráficas la sobrestimación del Q (Q-values predichos > returns reales)

---

## 3.4 — TD3: los tres trucos que arreglaron DDPG

### Por qué importa para tu doctorado

TD3 es el **algoritmo más usado como baseline en Offline RL** (junto con SAC). El paper de TD3+BC, que verás en Fase 4, es literalmente "TD3 con un regularizador de behavior cloning encima" — y es competitivo con CQL e IQL. Conocer TD3 a fondo es conocer el 50% del estado del arte en Offline RL.

### Lectura

- **Fujimoto, van Hoof, Meger 2018 — "Addressing Function Approximation Error in Actor-Critic Methods"** → [arXiv:1802.09477](https://arxiv.org/abs/1802.09477). El paper de TD3. Es **claro, corto y bien escrito**. Modelo de cómo se hace un buen paper de RL.
- **Spinning Up — TD3** → [spinningup.openai.com/.../td3.html](https://spinningup.openai.com/en/latest/algorithms/td3.html).

### Refuerzo en vídeo

- **CleanRL — `td3.py`** → [GitHub](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py). Implementación de referencia.

### Temas mínimos — los tres trucos

1. **Twin critics (clipped double Q-learning)**: dos redes Q independientes, y para el target se usa $\min(Q_1, Q_2)$. Reduce sobrestimación.
2. **Delayed policy updates**: el actor se actualiza cada $d$ pasos (típicamente $d=2$), mientras los critics se actualizan en cada paso. Razón: critics demasiado ruidosos producen actualizaciones de política basura.
3. **Target policy smoothing**: ruido gaussiano clippeado añadido a la acción target. Suaviza el target Q sobre acciones similares, evitando explotación de picos espurios del Q.

Cada uno se puede activar/desactivar y comparar. Si no haces esa ablación, no entiendes TD3 — entiendes solo el cocktail final.

### Ejecución mínima

- **Implementa TD3 desde cero** en PyTorch. Single file. Pruébalo en HalfCheetah, Hopper, Walker2d.
- **Ablation study**: corre TD3 con (a) todos los trucos, (b) sin twin critics, (c) sin delayed updates, (d) sin smoothing. 5 seeds cada uno. Esta gráfica es ya material de blog post.

### Checklist 3.4

- [ ] Mi TD3 supera HalfCheetah-v4 a ~9000 de return medio
- [ ] He hecho la ablación de los tres trucos y veo cuál pesa más
- [ ] Sé por qué los twin critics atacan exactamente la sobrestimación (no es tan obvio como en DQN)
- [ ] Sé el porqué del nombre "delayed": no es solo lentitud, es estabilidad numérica

---

## 3.5 — SAC: Maximum Entropy RL

### Por qué importa para tu doctorado

SAC es **el otro caballo de batalla** del control continuo, junto con PPO. En Offline RL es el algoritmo base sobre el que se construye CQL (Conservative Q-Learning), y en Federated RL aparece constantemente porque su naturaleza off-policy lo hace más data-efficient — y la eficiencia de datos es **el** problema central de Federated RL (los datos están repartidos y son escasos por cliente).

Más profundo: el principio de Maximum Entropy RL (MaxEnt RL) **no es una heurística**. Es un framework teórico completo con una conexión sólida con inferencia probabilística (RL como inferencia, Levine 2018). Esa conexión va a aparecer en tu Fase 4 y en tu doctorado cuando leas sobre Decision Transformer, control as inference, y métodos bayesianos en RL.

### Recurso principal

- **Haarnoja et al. 2018 — "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"** → [arXiv:1801.01290](https://arxiv.org/abs/1801.01290). El paper original.
- **Haarnoja et al. 2018 — "Soft Actor-Critic Algorithms and Applications"** → [arXiv:1812.05905](https://arxiv.org/abs/1812.05905). El paper del **automatic temperature tuning**, mucho más práctico que el primero. Es la versión de SAC que se usa en la práctica. Léelos en este orden.

### Lectura complementaria — **el bonus teórico**

- **Levine 2018 — "Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review"** → [arXiv:1805.00909](https://arxiv.org/abs/1805.00909). Conecta MaxEnt RL con inferencia bayesiana. **Lectura imprescindible si tu tesis tiene componente teórico**. Te lo aviso: es denso. Léelo dos veces, con un mes de margen entre ellas.

### Refuerzo en vídeo

- **Berkeley CS285 — Lecture 14: Variational Inference & RL** → Levine explica MaxEnt RL desde la perspectiva de inferencia. Es donde el lenguaje matemático de SAC encaja con el resto del ML.
- **CleanRL — `sac.py`** → [GitHub](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py).

### Temas mínimos

Objective MaxEnt: $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$. La entropía como bonus que premia políticas estocásticas. Soft Q-function y soft value function. Soft Bellman equation. Actor estocástico (gaussiano + tanh squashing — **¡el truco de la squash function!**, importante para entornos con acciones acotadas). Twin critics (heredado de TD3). Automatic temperature tuning: $\alpha$ se aprende para mantener una entropía objetivo. Reparameterization trick: por qué SAC samplea con $a = \mu + \sigma \cdot \epsilon$ en lugar de samplear de la distribución directamente — clave para que el gradiente fluya por la política.

### Ejecución mínima

- **Implementa SAC desde cero** con automatic temperature. Single file. HalfCheetah, Hopper, Walker2d y Humanoid si te atreves.
- Compara contra TD3 en los mismos entornos. SAC suele ganar en data efficiency, TD3 en pico final. Esa diferencia tiene explicación — sabérsela es saber SAC.
- **Mini-proyecto pre-doctoral**: replica una mejora pequeña de un paper reciente sobre SAC (REDQ, DroQ, TQC son candidatos buenos) y mide si reproduce el resultado del paper. Esto te da la dinámica **"leer paper → replicar → medir → reportar"** que será tu día a día doctoral.

### Checklist 3.5

- [ ] Mi SAC supera HalfCheetah-v4 a >10000 de return medio
- [ ] Sé derivar la soft Bellman equation desde el objetivo MaxEnt
- [ ] Sé qué hace el squash con tanh y por qué hay que corregir el log_prob ($\log(1 - \tanh^2(\cdot))$ correction)
- [ ] Sé explicar el reparameterization trick y por qué no se podría hacer con `Categorical`
- [ ] Sé en qué se diferencia matemáticamente SAC de TD3 (no solo "uno es estocástico y el otro determinístico")

---

## 3.6 — Distributed training y vectorización de entornos

### Por qué importa para tu doctorado

En Federated RL **el cuello de botella casi nunca es la GPU, es el rollout**: generar trayectorias es caro. Aprender a vectorizar entornos correctamente reduce el tiempo de experimento de "una semana por seed" a "una tarde por seed". Esto **decide cuántos experimentos puedes correr en tu doctorado**, y por tanto cuántas hipótesis puedes testear. No es un detalle técnico, es una palanca de productividad investigadora.

### Recursos

- **Gymnasium documentation — Vector Environments** → [gymnasium.farama.org/api/vector/](https://gymnasium.farama.org/api/vector/). `SyncVectorEnv` y `AsyncVectorEnv`. Lee la doc entera, son 20 minutos.
- **EnvPool** → [GitHub](https://github.com/sail-sg/envpool). Vectorización en C++ ultra-rápida para Atari y MuJoCo. CleanRL tiene versiones que la usan.
- **CleanRL — versiones `_envpool` y `_jax`** → para ver implementaciones modernas que aprovechan paralelismo serio.

### Temas mínimos

Sync vs Async vectorization: cuándo cada uno. Cómo cambian las dimensiones de tus tensores con $N$ entornos paralelos (de `(obs_dim,)` a `(N, obs_dim)`). Por qué los rollouts vectorizados son la base de PPO (`num_envs × num_steps = batch_size`). Reset por entorno (cuando uno termina y otros no). Manejo correcto de `done` masks en GAE — un bug aquí es invisible pero corrompe el aprendizaje. Seeding por entorno y reproducibilidad bajo paralelismo.

### Ejecución mínima

- Reescribe tu PPO de 3.2 con `SyncVectorEnv` (8–16 entornos en paralelo). Mide la mejora de wall-clock time.
- Pásalo a `AsyncVectorEnv` y mide de nuevo. Decide cuál vale la pena en tu hardware.

### Checklist 3.6

- [ ] Mi PPO entrena 4–8x más rápido con `SyncVectorEnv` correctamente configurado
- [ ] Sé manejar `done` flags y resets parciales sin corromper los advantages
- [ ] Entiendo por qué `AsyncVectorEnv` puede ser más lento si tu entorno es trivial
- [ ] Sé seedear de forma reproducible bajo paralelismo

---

## 3.7 — Reproducibilidad seria y reporting estadístico

### Por qué importa para tu doctorado

**Este bloque no es un extra; es el sello doctoral.** El paper de Henderson et al. 2017 ya te lo dijo en Fase 2: la mitad de los papers de RL no son reproducibles porque reportan mal. En tu doctorado, esto no se permite. Cualquier paper que mandes a NeurIPS / ICML / ICLR pasa por revisores que **te van a buscar errores de reporting**. Aprenderlo ahora ahorra trauma después.

### Recursos imprescindibles

- **Agarwal et al. 2021 — "Deep Reinforcement Learning at the Edge of the Statistical Precipice"** (NeurIPS Outstanding Paper) → [arXiv:2108.13264](https://arxiv.org/abs/2108.13264). **Léelo entero**. Es el paper que cambió cómo se reportan resultados en RL. Introduce IQM (Interquartile Mean), Performance Profiles, Probability of Improvement. La librería **rliable** ([github.com/google-research/rliable](https://github.com/google-research/rliable)) implementa todo esto. Úsala desde ya.
- **Henderson et al. 2017 — "Deep RL that Matters"** → [arXiv:1709.06560](https://arxiv.org/abs/1709.06560). Releélo si no lo hiciste en Fase 2.
- **Patterson et al. 2023 — "Empirical Design in Reinforcement Learning"** → [arXiv:2304.01315](https://arxiv.org/abs/2304.01315). Guía moderna y completa.

### Herramientas

- **Weights & Biases** ([wandb.ai](https://wandb.ai)) o **MLflow** para tracking. **No negociable**: si entrenas sin tracking, has tirado tiempo de GPU.
- **rliable** para reporting estadístico.
- **Hydra** ([hydra.cc](https://hydra.cc)) para configs limpias y reproducibles. CleanRL no lo usa pero tus proyectos doctorales lo agradecerán.

### Temas mínimos

Seeds: mínimo 5, idealmente 10. Bandas de confianza, no medias con sd. **IQM** y **Performance Profiles** en vez de "max over seeds" o medias simples. Por qué la varianza en RL es enorme y los outliers contaminan medias (de ahí IQM). Pruebas de hipótesis para "¿mi algoritmo es mejor que el baseline?" — bootstrap confidence intervals, probabilities of improvement. Documentación de hiperparámetros: tabla completa en el apéndice de cualquier paper o blog post. Logging de **todo**: returns, losses, gradientes (norma), KL real entre updates de PPO, fracción de clipping, entropía. Si no lo logueas, no lo puedes debuggear.

### Ejecución mínima

- Reescribe los resultados de tu Fase 2 (DQN/Pong) y de los bloques 3.2–3.5 con **rliable**. Producirás gráficos de Performance Profiles y de Probability of Improvement.
- Crea una **plantilla de experimento** en tu repo: estructura `configs/`, `runs/`, `logs/`, `scripts/`. Reutilízala en toda la Fase 4 y siguientes.

### Checklist 3.7

- [ ] Todas mis gráficas tienen ≥5 seeds, bandas de confianza por bootstrap e IQM como métrica principal
- [ ] Uso W&B (o equivalente) en cada experimento, sin excepción
- [ ] Sé hacer un Performance Profile de rliable y leerlo correctamente
- [ ] Mi código tiene un seed completo (Python, NumPy, PyTorch, env) reproducible
- [ ] Tengo una plantilla de experimento limpia que puedo clonar y reutilizar

---

## Pre-final: lo que se ve cuando se cierra la Fase 3

Cuando termines esto, deberías poder:

- Abrir cualquier paper de **Federated PPO** o **Federated SAC** y entender qué cambia respecto a la versión centralizada — qué se federa, qué se mantiene local, qué garantías sobreviven y cuáles se rompen. Ese es exactamente el modo de lectura que vas a necesitar a partir de Fase 5.
- Modificar PPO o SAC para experimentar con tus propias ideas: cambiar el surrogate loss, añadir un regularizador, sustituir el critic por un ensemble, etc. **Esa es la capacidad técnica mínima de un doctorando en RL.**
- Defender, ante tu director o ante un revisor, por qué TD3 sobreestima menos que DDPG y por qué SAC explora mejor que TD3, sin recurrir a vaguedades.
- Producir una gráfica de comparación entre algoritmos que pasaría peer review en un workshop de NeurIPS.

Si llegas aquí, **ya tienes el músculo técnico para empezar Fase 4 (Offline RL) sin miedo**. Offline RL es conceptualmente más sutil que online, pero técnicamente más sencillo (sin rollouts, los experimentos son rápidos). El cuello de botella se mueve de "implementar bien el algoritmo" a "leer y reproducir papers". Y eso es ya **trabajo doctoral**.

---

## Plan sugerido de 10 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 3.1 TRPO (teoría, sin implementar) | Releer GAE de Fase 2 |
| 2–3 | 3.2 PPO — implementación y debugging | 37 detalles + CleanRL `ppo.py` |
| 4 | 3.2 PPO — experimentos en MuJoCo | Comparativa con SB3/CleanRL |
| 5 | 3.3 DDPG en Pendulum | Lectura del paper original |
| 6 | 3.4 TD3 — implementación y ablation | MuJoCo locomotion |
| 7–8 | 3.5 SAC — implementación con auto-temp | Levine 2018 (RL as Inference) |
| 9 | 3.6 Vectorización + 3.7 Reproducibilidad | Refactor de todo el repo |
| 10 | Mini-proyecto: replicar un paper reciente | Escribir blog post o repo README serio |

Si descubres que un bloque te cuesta más (especialmente 3.2 PPO y 3.5 SAC), **estíralo sin culpa**. Es **mucho mejor** llegar a Fase 4 con PPO y SAC completamente dominados que llegar antes con la sensación de que "más o menos funcionan".

---

## Conexiones explícitas con tus fases futuras

Para que veas que esta fase no es un *check the box* sino una pieza estructural:

- **Fase 4 (Offline RL)**: CQL se construye sobre SAC. TD3+BC se construye sobre TD3. IQL es independiente pero usa la misma factorización actor-critic. **Sin Fase 3 sólida, Fase 4 es opaca.**
- **Fase 5 (FL clásico)**: las variantes federadas más estudiadas en políticas son FedAvg-PPO y FedAvg-SAC. El concepto de **client drift** en FedAvg se conecta directamente con el concepto de **policy drift** en TRPO/PPO (de ahí surgen propuestas como PPO-KL federado).
- **Fase 6 (Federated RL)**: tus baselines durante todo el doctorado serán PPO, SAC, TD3 (en sus variantes federadas y offline). Si no los conoces a fondo, no sabes contra qué estás compitiendo.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:

- Derivaciones completas paso a paso (especialmente para 3.1 TRPO y 3.5 MaxEnt RL)
- Implementaciones comentadas en PyTorch (single-file, estilo CleanRL)
- Lista de **bugs comunes** en cada algoritmo y cómo detectarlos en las curvas de aprendizaje
- Ablation studies guiados con código listo para correr

Mi recomendación: si vienes recién de Fase 2 y tu PPO funciona regular, empieza por **3.2 PPO a fondo** porque es el algoritmo de mayor retorno por hora invertida. Si ya tienes PPO sólido, salta a **3.5 SAC** porque es donde más vas a aprender conceptualmente (MaxEnt RL es una puerta a media literatura moderna). TRPO (3.1) y DDPG (3.3) son bloques **cortos** — los puedes hacer en paralelo a los otros sin problema.
