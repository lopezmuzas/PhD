---
title: "Fase 4 — Offline Reinforcement Learning"
tags: [rl, refuerzo]
status: borrador
updated: 2026-08-08
---

# Fase 4 — Offline Reinforcement Learning

> **Objetivo de la fase**: dominar el aprendizaje por refuerzo **a partir de datasets fijos**, sin interacción con el entorno. Entender por qué los algoritmos online (DQN, SAC) **fallan catastróficamente** en este setting — el problema del *distributional shift* y la **sobreestimación bootstrappeada** — y dominar las tres familias de soluciones modernas: **policy constraint** (BCQ, TD3+BC), **value regularization** (CQL) e **implicit Q-learning** (IQL). Salir con un pipeline replicado de CQL e IQL sobre D4RL/Minari, y con la primera lectura crítica de la literatura que es **directamente relevante para tu tesis en Federated RL sobre Compute-to-Data**.
>
> **Tu situación de partida**: vienes de Fase 3 (que pospones de momento) con DQN, A2C, PPO y SAC dominados. Aquí cambias de eje: ya no se trata de explorar mejor, se trata de **no engañarte con un dataset que no contiene lo que necesitas**.
>
> **Tiempo estimado realista**: 6–8 semanas a ritmo de 1–2 h/día. Más denso conceptualmente que la Fase 2, menos hambriento de GPU (los datasets son fijos, no hay rollouts caros). El cuello de botella aquí es **leer papers densos y reproducir resultados con disciplina experimental**, no entrenar 24h por seed.

---

## Cómo usar esta guía

Misma lógica que las anteriores, con dos avisos específicos:

**Este es el primer bloque "tesis-relevante" de tu roadmap.** Lo que aprendas aquí ya no es preparación: es **directamente** el setting de tu doctorado. En Pontus-X / Compute-to-Data, cada participante del dataspace publica un dataset histórico y **nunca permite interacción online con su entorno**. Eso es Offline RL por construcción. Lee esta fase con la lupa de "qué de esto va a aparecer en mi tesis", no con la de "qué bonito, otra rama de RL".

**Y un cambio cultural importante**: hasta ahora los benchmarks y los recursos eran cursos consolidados (Sutton & Barto, Silver, CS285). A partir de aquí, **la verdad vive en papers de los últimos 5 años** — muchos sin libro de texto que los recoja todavía. Tu lectura crítica de arXiv empieza aquí.

---

## Mapa de la Fase 4

| Bloque | Tema | Peso |
|---|---|---|
| 4.1 | El problema fundamental: distributional shift | 🔥🔥🔥🔥🔥 La pregunta central |
| 4.2 | Behavioral Cloning y el límite del aprendizaje por imitación | 🔥🔥🔥 La baseline obligatoria |
| 4.3 | Policy constraint: BCQ, BEAR, TD3+BC | 🔥🔥🔥🔥 Primera familia de soluciones |
| 4.4 | Value regularization: CQL | 🔥🔥🔥🔥🔥 El algoritmo más citado |
| 4.5 | Implicit Q-Learning: IQL | 🔥🔥🔥🔥🔥 El SOTA práctico |
| 4.6 | Model-based Offline RL: MOPO, COMBO, MOReL | 🔥🔥🔥 Conecta con Dyna-Q |
| 4.7 | Decision Transformer y la rama sequence-modeling | 🔥🔥🔥 Cambio de paradigma |
| 4.8 | Offline Policy Evaluation (OPE) | 🔥🔥🔥🔥 El problema "hermano" |
| 4.9 | Disciplina específica de Offline RL | 🔥🔥🔥🔥 No es disciplina genérica |
| 4.10 | Proyecto integrador: replicación + tesis-relevant pipeline | 🔥🔥🔥🔥🔥 La primera pieza doctoral real |

---

## 4.1 — El problema fundamental: distributional shift

### Por qué importa — y esta es la pregunta que define toda la fase

En online RL, si tu política se equivoca, **tomas más datos y aprendes a corregir**. En Offline RL, **no puedes**. Si tu política aprende a valorar mucho una acción que **no aparece en el dataset**, no hay forma de descubrir que ese valor era una alucinación de la red. Y peor: la actualización de Bellman **bootstrappea** sobre esos valores alucinados, creando un bucle de retroalimentación donde la sobreestimación se amplifica. Esto es el **distributional shift** entre la política de comportamiento $\pi_\beta$ (la que generó el dataset) y la política aprendida $\pi$ — y es **el** problema central de Offline RL.

Para tu tesis: en Pontus-X, cada cliente del dataspace tiene un dataset generado por **su propia política histórica** (cada hospital, cada fábrica). Cuando intentas federar, **cada cliente trae una $\pi_\beta$ distinta**. El distributional shift se compone con la heterogeneidad federada — y ahí es exactamente donde la literatura está fina y donde puede estar tu contribución original.

### Recurso principal — la pieza canónica

- **Levine, Kumar, Tucker & Fu (2020) — "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"** → [arXiv:2005.01643](https://arxiv.org/abs/2005.01643). **El tutorial fundacional del campo**. 80 páginas. Cubre desde la definición formal hasta una taxonomía completa de algoritmos. **Lectura obligatoria, integral, con lápiz**. Si solo lees una cosa en toda la Fase 4, que sea esta. Levine es uno de los nombres centrales del campo y este tutorial es la referencia canónica.

### Vídeo

- **Berkeley CS285 — Lectures de Offline RL (lecciones 15–16 aproximadamente, depende del año)** → [Curso](http://rail.eecs.berkeley.edu/deeprlcourse/). Levine explica su propio tutorial en formato vídeo. **Imprescindible**.
- **NeurIPS 2020 Tutorial — "Offline RL" by Levine, Kumar & Levine** → busca la grabación en YouTube. Versión más corta y digerible del paper anterior.

### Temas mínimos

Definición formal del setting Offline RL: dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ generado por una política de comportamiento $\pi_\beta$, sin acceso al entorno durante el entrenamiento. **Distributional shift**: por qué la política aprendida $\pi$ puede acabar muy lejos de $\pi_\beta$ en términos de la distribución de estados-acciones que visita. **Out-of-distribution (OOD) actions**: acciones que la red Q "ve" durante el max de la actualización de Bellman pero que **nunca aparecen en el dataset**. **Maximization bias amplificado**: en online RL hay max bias también, pero se autocorrige; en offline, se amplifica iteración a iteración. **Erroneous extrapolation**: las redes neuronales extrapolan mal fuera de su distribución de entrenamiento, y la actualización de Bellman las obliga a evaluar OOD actions. **Concentrability coefficient**: la medida teórica de cuánto difiere la política aprendida de la de comportamiento, y por qué aparece en las cotas de error.

### Ejercicio "wow"

Toma un agente SAC pre-entrenado en HalfCheetah (puedes entrenarlo tú o descargarlo). Genera un dataset de 1M transiciones con él. Ahora entrena **SAC vanilla** sobre ese dataset **sin** interacción con el entorno (solo replay del dataset). Observa cómo la curva de evaluación inicialmente sube y después **colapsa**. Eso es distributional shift en vivo. Es uno de los experimentos más reveladores que puedes hacer. Reproduce este experimento porque vas a referirte a él durante todo el doctorado.

### Checklist 4.1

- [ ] Leí el tutorial de Levine et al. 2020 entero, con notas propias
- [ ] Sé explicar el distributional shift con un dibujo en 2 minutos
- [ ] Reproduje el "colapso de SAC" en un setting offline
- [ ] Entiendo por qué la actualización de Bellman amplifica el problema en lugar de corregirlo
- [ ] Conecto el problema con tu setting de tesis (Pontus-X / Compute-to-Data)

---

## 4.2 — Behavioral Cloning y el límite del aprendizaje por imitación

### Por qué importa

BC es la baseline más simple imaginable: **ignora la recompensa, simplemente clona la política de comportamiento** por supervised learning $\min_\pi -\mathbb{E}_{(s,a) \sim \mathcal{D}}[\log \pi(a|s)]$. Es trivial de implementar, pero **es la baseline obligatoria contra la que cualquier Offline RL serio debe compararse**. Si tu CQL no supera BC en un dataset, **algo está mal**. Y en datasets *expert-only* (donde solo hay trayectorias casi óptimas), BC frecuentemente **gana a algoritmos más sofisticados** — y eso es una lección humilde sobre cuándo el problema es realmente difícil y cuándo no.

Conexión con la rama clásica: Imitation Learning (DAgger, GAIL) es una hermana mayor de BC. No te entretengas mucho ahí — para tu tesis es lateral — pero conoce los nombres.

### Recurso principal

- **Pomerleau (1988) — "ALVINN: An Autonomous Land Vehicle in a Neural Network"**. El primer paper de BC, históricamente. 4 páginas. Léelo por contexto.
- **Sección "Imitation Learning" del tutorial de Levine et al. 2020**.
- **Berkeley CS285 — Lecture 2 (Supervised Learning of Behaviors)** → la lección donde Levine introduce BC y por qué falla por *compounding errors*.

### Temas mínimos

BC como supervised learning con loss MLE sobre $(s, a)$ pairs. **Compounding errors**: por qué un BC entrenado con error $\epsilon$ por paso acumula error $O(T^2 \epsilon)$ en horizonte $T$ — el resultado clásico de Ross & Bagnell. DAgger como solución (requiere experto interactivo, **no aplica en offline puro**). Cuándo BC es competitivo: datasets expert-only, horizonte corto, alta cobertura del espacio de estados. Cuándo BC falla: datasets *mixed* (mezclas de buenas y malas políticas), datasets sub-óptimos, datasets que requieren **stitching** de subsegmentos de trayectorias distintas.

### Ejercicio

Implementa BC en PyTorch sobre datos D4RL/Minari de Hopper y Walker2d. Evalúa sobre tres tipos de dataset:
- `expert` (solo trayectorias casi óptimas)
- `medium-expert` (mezcla 50/50)
- `medium-replay` (replay buffer de un agente parcialmente entrenado)

Verás que BC se acerca al óptimo en `expert`, decae bastante en `medium-expert` y se hunde en `medium-replay`. **Esa tabla es tu mapa mental** de cuándo el problema offline es trivial y cuándo no.

### Checklist 4.2

- [ ] Implementé BC en PyTorch y lo evalué sobre los tres tipos de dataset D4RL/Minari
- [ ] Sé enunciar el resultado de compounding errors $O(T^2 \epsilon)$ y dar la intuición
- [ ] Entiendo por qué DAgger no es aplicable en offline puro
- [ ] Tengo una tabla numérica BC vs. random vs. expert sobre Hopper-v2/medium-expert

---

## 4.3 — Policy constraint: BCQ, BEAR, TD3+BC

### Por qué importa

Primera familia de soluciones al distributional shift: **forzar a que la política aprendida no se aleje mucho de $\pi_\beta$**. Esto se puede hacer (1) restringiendo el soporte de acciones que la política puede tomar a las que aparecen en el dataset (BCQ), (2) restringiendo en distribución MMD/KL entre $\pi$ y $\pi_\beta$ (BEAR), o (3) **simplemente añadiendo un término de behavioral cloning como regularizador** sobre TD3 (TD3+BC).

La lección clave: **TD3+BC, propuesto en 2021, demostró que casi todos los trucos sofisticados de BCQ/BEAR pueden sustituirse por un único término $\lambda \cdot \|\pi(s) - a\|^2$ en la loss del actor**. Es **un caso paradigmático de "minimalismo gana"** en deep learning, y deberías interiorizarlo: las soluciones más simples a veces se imponen.

### Lecturas obligatorias

- **Fujimoto, Meger & Precup (2019) — "Off-Policy Deep Reinforcement Learning without Exploration"** → [arXiv:1812.02900](https://arxiv.org/abs/1812.02900). El paper de **BCQ**, el primer método que ataca el problema offline directamente. Introduce el **VAE generador** para muestrear acciones del soporte de $\pi_\beta$.
- **Kumar, Fu, Tucker & Levine (2019) — "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction"** → [arXiv:1906.00949](https://arxiv.org/abs/1906.00949). El paper de **BEAR**. Concepto clave: **bootstrap error** — los errores en estados OOD se acumulan al hacer bootstrap. Restricción MMD entre $\pi$ y $\pi_\beta$.
- **Fujimoto & Gu (2021) — "A Minimalist Approach to Offline Reinforcement Learning"** → [arXiv:2106.06860](https://arxiv.org/abs/2106.06860). El paper de **TD3+BC**. Una página de algoritmo, resultados competitivos con CQL/BEAR. **Lee este con especial atención**: ejemplifica la cultura de "complica solo si es necesario" que es valiosa en investigación.

### Temas mínimos

**BCQ**: un VAE aprende a generar acciones de la behavior policy; durante el max de Bellman, solo se evalúan acciones generadas por el VAE (más una pequeña perturbación). **BEAR**: penalizar la distancia MMD entre $\pi(\cdot|s)$ y $\pi_\beta(\cdot|s)$ — mantiene el **soporte** sin requerir matching exacto. **TD3+BC**: la loss del actor pasa a $L_\pi = -\mathbb{E}[\lambda Q(s, \pi(s)) - \|\pi(s) - a\|^2]$, donde el segundo término es behavioral cloning y $\lambda$ se normaliza con la magnitud de Q para que el peso relativo sea estable. **State normalization**: pequeño truco crucial de TD3+BC — normalizar los estados a media 0 y std 1. Sin esto, los resultados decaen significativamente. **Por qué TD3 y no SAC**: TD3 es determinista, BC sobre una política determinista es trivial ($\|\pi(s) - a\|^2$). En SAC habría que matar la entropía o usar KL.

### Ejercicio

Implementa **TD3+BC** desde cero en PyTorch sobre D4RL/Minari Hopper-medium-v2 y Walker2d-medium-v2. Compara contra:
- BC (tu baseline del bloque 4.2)
- TD3 vanilla (sin el término BC) — **te dará la curva de colapso** del bloque 4.1, ahora replicada con disciplina
- TD3+BC

Para cada uno, ≥5 seeds y bandas de confianza. Tu TD3+BC debería superar a BC en `medium-expert` y caer cerca de BC en `expert`. Si no, hay un bug — compara contra **CORL** ([tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)), el repo de referencia.

### Checklist 4.3

- [ ] Implementé TD3+BC desde cero, con state normalization
- [ ] Mi TD3+BC supera BC en Hopper-medium-v2
- [ ] Sé explicar la diferencia conceptual entre BCQ (soporte por VAE), BEAR (MMD) y TD3+BC (BC en loss)
- [ ] Leí los tres papers y entiendo por qué Fujimoto & Gu 2021 fue un golpe a la complejidad

---

## 4.4 — Value regularization: Conservative Q-Learning (CQL)

### Por qué importa

Si la familia anterior ataca el problema **por el lado de la política**, CQL lo ataca **por el lado del Q-value**: aprende una Q-function **pesimista** para acciones OOD. La intuición: si la red Q nunca sobrestima fuera del dataset, el max de Bellman no puede explotar. Esto se consigue con un término extra en la loss del crítico que penaliza Q-values altos en acciones que **no están en el dataset** y los empuja hacia abajo, mientras mantiene los Q-values en las acciones del dataset.

**CQL es el algoritmo más citado de Offline RL.** Salió en 2020 y rápidamente se convirtió en el baseline obligatorio. Aprendértelo bien — sus loss, su derivación, sus failure modes — **es no negociable** para tu tesis.

### Recurso principal

- **Kumar, Zhou, Tucker & Levine (2020) — "Conservative Q-Learning for Offline Reinforcement Learning"** → [arXiv:2006.04779](https://arxiv.org/abs/2006.04779). **El paper de CQL**. Largo y técnico. Léelo dos veces. La sección de derivación matemática (apéndice) es donde está la enjundia: hay un **lower bound demostrable** sobre el valor real bajo la política aprendida. Esa garantía teórica es parte de por qué CQL se ganó la confianza del campo tan rápido.

### Vídeo

- **Aviral Kumar (autor) explica CQL** en seminarios de Berkeley y otros venues. Búscalo en YouTube. Es Kumar mismo desglosando su paper.
- **Berkeley CS285** dedica lección a CQL (la lección de Offline RL más reciente del curso).

### Temas mínimos

La loss de CQL tiene dos partes: (1) la loss estándar de Bellman (como en SAC), y (2) un **término regularizador conservativo**: $\alpha \cdot (\mathbb{E}_{a \sim \mu}[Q(s,a)] - \mathbb{E}_{a \sim \pi_\beta}[Q(s,a)])$, donde $\mu$ es una distribución que cubre acciones OOD. Intuitivamente: minimiza Q en acciones que no están en el dataset, maximiza Q en acciones que sí. **Variantes**: CQL($\mathcal{H}$) usa $\mu$ = distribución de máxima entropía, CQL($\rho$) usa $\mu$ derivada de la política actual. **Garantía teórica**: la Q aprendida es un **lower bound** de la Q real bajo la política. Esto es importante: significa que si la política aprendida tiene un Q estimado de $X$, **el Q real es al menos $X$**. **Hiperparámetro $\alpha$**: pesa cuán conservativo eres. Demasiado bajo → vuelves a tener distributional shift; demasiado alto → demasiado pesimista, la política aprendida es casi BC. **Importance sampling con logsumexp**: la implementación práctica usa el truco de logsumexp para aproximar la $\mu$. **Implementación en discreto (DiscreteCQL)** vs. continuo (CQL continuo, basado en SAC).

### Ejercicio

Implementa **CQL** desde cero o **estudia minuciosamente la implementación de CORL**. Es más sutil de implementar que TD3+BC — el término logsumexp con sampling es donde más bugs aparecen. Reproduce los resultados sobre los datasets de locomoción de D4RL/Minari:
- HalfCheetah-medium-v2, medium-replay-v2, medium-expert-v2
- Hopper, Walker2d con los tres dataset types

Tabla comparativa: BC, TD3+BC, CQL. Tu CQL debería igualar a CORL/d3rlpy seed-por-seed; si no, **es un bug**, no una variación.

### Checklist 4.4

- [ ] Leí el paper de CQL entero, incluyendo el apéndice con la demostración del lower bound
- [ ] Implementé (o estudié línea-a-línea) CQL con el término logsumexp
- [ ] Sé explicar por qué CQL es un *lower bound* y qué garantía exacta provee
- [ ] Mi CQL reproduce los números de CORL en al menos 3 datasets de locomoción
- [ ] Entiendo cuándo CQL falla (alpha mal tuneado, datasets demasiado narrow)

---

## 4.5 — Implicit Q-Learning (IQL)

### Por qué importa — y este es el SOTA práctico

CQL es teóricamente elegante pero **en la práctica IQL frecuentemente lo supera, con menos hiperparámetros y entrenamiento más estable**. La idea es **brillante por simple**: en lugar de evaluar acciones OOD durante el max de Bellman (lo que causa el problema), IQL **evita evaluar acciones OOD en absoluto**. Hace expectile regression para estimar $V(s) \approx \max_a Q(s,a)$ **sin tomar nunca un max sobre acciones**. Después extrae la política por *advantage-weighted regression*. **El max nunca aparece** — y por tanto el problema OOD desaparece de raíz.

Para tu tesis: IQL es probablemente el algoritmo offline que **vas a usar como baseline en tu primer paper**. Es estable, replicable, bien establecido. Aprender a implementarlo y a defenderlo es uno de los retornos más altos de toda la fase.

### Recurso principal

- **Kostrikov, Nair & Levine (2022) — "Offline Reinforcement Learning with Implicit Q-Learning"** → [arXiv:2110.06169](https://arxiv.org/abs/2110.06169). **El paper de IQL**. Corto (10 páginas el cuerpo), elegante. Esto es **lectura activa con bolígrafo en mano**. La intuición clave es la sección 3.1: ¿cómo se aproxima $\max_a Q(s,a)$ sin tomar el max? Respuesta: **expectile regression** con $\tau$ cercano a 1.

### Temas mínimos

**Expectile regression**: generalización asimétrica de la regresión cuadrática. Para $\tau \in (0,1)$, la regresión expectile-$\tau$ recupera el expectile $\tau$ de la distribución. Para $\tau \to 1$, recupera el supremo (el max). Concretamente, la loss es $L_2^\tau(u) = |\tau - \mathbb{1}(u<0)| \cdot u^2$. **Tres redes**: $V(s)$, $Q(s,a)$ y $\pi(a|s)$. **Loop**: actualizar $V$ por expectile regression sobre $Q$, actualizar $Q$ por Bellman estándar usando $V(s')$ (no $\max_{a'} Q(s',a')$), actualizar $\pi$ por **advantage-weighted regression**: $\pi(a|s) \propto \exp(\beta \cdot A(s,a)) \cdot \pi_\beta(a|s)$, parametrizado como una regresión ponderada sobre el dataset. **Por qué importa la AWR**: no requiere muestrear acciones de $\pi$ para evaluar — todo se hace sobre $(s,a)$ pairs del dataset. **Hiperparámetros**: $\tau \in [0.7, 0.9]$ típicamente, $\beta \in [3, 10]$. Notablemente pocos, comparado con CQL.

### Ejercicio obligatorio — el central de la fase

Implementa **IQL desde cero** en PyTorch sobre D4RL/Minari. Si has implementado TD3+BC y CQL antes, esto te llevará 1–2 días. Compara contra los anteriores:

| Algoritmo | hopper-medium | hopper-medium-replay | hopper-medium-expert | walker2d-... | halfcheetah-... |
|---|---|---|---|---|---|
| BC | | | | | |
| TD3+BC | | | | | |
| CQL | | | | | |
| **IQL** | | | | | |

Con ≥5 seeds, IQM con bootstrap CI 95%. **Esta tabla es lo que vas a poner en tu primer paper, así que hazla bien**. Compara con la tabla de CORL como ground truth — si tus números difieren más de un 10% de los suyos en cualquier celda, **investiga el bug** antes de seguir.

### Checklist 4.5

- [ ] Leí el paper de IQL dos veces, con foco en la sección de expectile regression
- [ ] Implementé IQL desde cero (V, Q y AWR policy)
- [ ] Mis números reproducen CORL ±10% en al menos 6 datasets D4RL
- [ ] Sé explicar en pizarra por qué IQL evita el problema OOD sin necesidad de regularizador conservativo
- [ ] Entiendo qué hace exactamente expectile regression con $\tau=0.7$ vs. $\tau=0.9$

---

## 4.6 — Model-based Offline RL: MOPO, COMBO, MOReL

### Por qué importa

Los algoritmos anteriores son **model-free**: aprenden directamente la política/Q sobre el dataset. Una alternativa: **aprender un modelo del entorno $\hat{P}(s'|s,a)$** desde el dataset, y usarlo para generar trayectorias sintéticas — esencialmente, **Dyna-Q de la Fase 1 llevado a deep**. La trampa es que el modelo aprendido es **inexacto fuera del soporte del dataset**, así que hay que pesimar las predicciones del modelo en zonas de alta incertidumbre (eso es MOPO) o combinar model-based con conservatism (COMBO).

**Para tu tesis hay una conexión doctoral concreta**: en Federated RL sobre dataspaces, **comunicar modelos del entorno** (en lugar de gradientes de Q o de política) es una de las cinco vías que tu roadmap maestro lista como posibles. Y la sub-rama **Offline Model-based + Federated** está particularmente poco explorada. Conoce este bloque aunque no lo implementes a fondo: te puede salir como tema de capítulo.

### Lecturas

- **Yu, Thomas, Yu, Ermon, Zou, Levine, Finn & Ma (2020) — "MOPO: Model-based Offline Policy Optimization"** → [arXiv:2005.13239](https://arxiv.org/abs/2005.13239). El paper canónico de model-based offline RL con pesimismo.
- **Kidambi, Rajeswaran, Netrapalli & Joachims (2020) — "MOReL: Model-Based Offline Reinforcement Learning"** → [arXiv:2005.05951](https://arxiv.org/abs/2005.05951). Paper contemporáneo a MOPO, enfoque alternativo a la pesimización (penaliza por divergencia del soporte).
- **Yu, Kumar, Rafailov, Rajeswaran, Levine & Finn (2021) — "COMBO: Conservative Offline Model-Based Policy Optimization"** → [arXiv:2102.08363](https://arxiv.org/abs/2102.08363). Combina lo mejor de MOPO y CQL.

### Temas mínimos

**Ensemble de modelos del entorno**: típicamente 5–7 redes neuronales que predicen $(s', r)$ a partir de $(s, a)$. La **varianza entre miembros del ensemble** sirve como estimador de incertidumbre — alta varianza = OOD = pesimar fuerte. **MOPO penalty**: $r_{pessimistic}(s,a) = r(s,a) - \lambda \cdot u(s,a)$, donde $u$ es la incertidumbre estimada. **Rollouts sintéticos cortos**: típicamente 1–5 pasos desde estados del dataset (rollouts largos amplifican el error del modelo). **Mezcla de datos reales y sintéticos** en el replay buffer del agente offline (típicamente SAC con conservatism). **Trade-off bias-variance del modelo**: rollouts más largos = más datos pero más error acumulado.

### Ejercicio (opcional pero recomendado para perfil doctoral)

Estudia la implementación de **MOPO en d3rlpy** o en CORL. Si te queda margen, replica los resultados sobre Hopper. Pero **el énfasis aquí es leer los papers, no implementar**: para tu tesis, model-based ofrece una vía de contribución pero no es el camino más corto. Conoce bien el lenguaje, ten una intuición del trade-off, y pasa al siguiente bloque.

### Checklist 4.6

- [ ] Leí MOPO y entiendo cómo se usa la varianza del ensemble como incertidumbre
- [ ] Sé explicar por qué los rollouts sintéticos en offline tienen que ser cortos
- [ ] Conecto el bloque con Dyna-Q (Fase 1) y con la posibilidad de federar modelos del entorno en mi tesis
- [ ] (Opcional) Estudié la implementación de MOPO en d3rlpy o CORL

---

## 4.7 — Decision Transformer y la rama sequence-modeling

### Por qué importa

Cambio de paradigma: **¿y si tratamos RL como un problema de predicción de secuencias, igual que un LLM?** Decision Transformer (Chen et al. 2021) hace exactamente eso: condiciona un transformer sobre $(R_{target}, s_1, a_1, R_{target}', s_2, a_2, \ldots)$, donde $R_{target}$ es el retorno deseado, y entrena por supervised learning a predecir la siguiente acción. **Cero TD learning, cero Bellman, cero distributional shift**. En su lugar, *prompt engineering* del retorno deseado.

Esto importa por dos razones:
1. **Conceptualmente**: es la evidencia más fuerte de que muchas técnicas "específicas de RL" pueden sustituirse por modelos de secuencias bien entrenados.
2. **Para tu tesis**: la rama está activa, ha generado Trajectory Transformer y muchas variantes, y conecta directamente con LLMs como policy backbones. Si tu director va por esa ruta, esto es central.

### Lecturas

- **Chen et al. (2021) — "Decision Transformer: Reinforcement Learning via Sequence Modeling"** → [arXiv:2106.01345](https://arxiv.org/abs/2106.01345). El paper original. Léelo: es elegante y conciso.
- **Janner, Li & Levine (2021) — "Offline Reinforcement Learning as One Big Sequence Modeling Problem"** → [arXiv:2106.02039](https://arxiv.org/abs/2106.02039). Trajectory Transformer. Hermano contemporáneo de DT.
- **Lectura crítica**: **Bhargava et al. (2023) — "When should we prefer Decision Transformers for Offline Reinforcement Learning?"** → [arXiv:2305.14550](https://arxiv.org/abs/2305.14550). Honesto sobre los límites de DT. **Spoiler**: en muchos benchmarks DT pierde contra CQL/IQL; el hype excede al rendimiento empírico. **Esta es la lectura que te previene del fanboyismo**.

### Temas mínimos

Arquitectura: GPT-style transformer con embeddings de $(R, s, a)$ secuenciales. **Return-to-go ($R_{target}$)**: en lugar de optimizar política, **especificas** qué retorno quieres conseguir y el modelo predice acciones. **Inference**: das el $R_{target}$ que quieres, observas $s_1$, el modelo te da $a_1$; ejecutas, observas $r_1, s_2$, actualizas $R_{target}' = R_{target} - r_1$, y repites. **Por qué no necesita Bellman**: porque no aprende una Q-function; aprende la **distribución condicional** $\pi(a | R_{target}, s, \text{historia})$. **Cuándo funciona bien**: datasets densos con buena variedad de retornos. **Cuándo falla**: tareas con stitching (combinar subtrayectorias) y datasets sparse-reward.

### Ejercicio (ligero)

No replicaes DT desde cero a menos que estés inspirado — es un transformer y tienes mejores cosas que hacer. **Estudia la implementación de CORL** (`dt.py`) y reprodúcela sobre uno o dos datasets. Compara contra tu IQL del bloque 4.5. Probablemente IQL gane en la mayoría de datasets D4RL de locomoción — esa es la lección honesta del bloque.

### Checklist 4.7

- [ ] Leí Decision Transformer y el paper crítico de Bhargava 2023
- [ ] Sé explicar cómo DT evita el distributional shift sin ser un truco de RL "tradicional"
- [ ] Entiendo qué es *return-to-go* y cómo se usa en inference
- [ ] Reproduje DT sobre al menos un dataset y comparé con IQL
- [ ] Tengo una opinión informada sobre cuándo DT es preferible y cuándo no

---

## 4.8 — Offline Policy Evaluation (OPE)

### Por qué importa — y este es probablemente el bloque más subestimado

Hasta ahora hemos hablado de **aprender** una política offline. Pero hay un problema gemelo, **igual de importante**: dado un dataset y una política candidata, **¿cuán buena es esa política?** Sin acceso al entorno, no puedes simplemente ejecutarla. Esto es **Offline Policy Evaluation (OPE)**, y es **crítico** para tu tesis por una razón concreta: en Pontus-X, no solo **entrenas offline**, sino que **evalúas offline**. Y si tu evaluación es ruidosa, todas las comparaciones entre algoritmos son ruido. Sin OPE robusto, no puedes decir "mi algoritmo federado es mejor" — solo puedes decir "mi número es más alto en mi métrica de evaluación, que puede estar correlacionada con el rendimiento real".

Es además un **subcampo donde la literatura está particularmente activa** (NeoRL incluye benchmarks de OPE específicos, Fu et al. tienen una line of work sobre OPE benchmarks).

### Lecturas

- **Voloshin, Le, Jiang & Yue (2019) — "Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning"** → [arXiv:1911.06854](https://arxiv.org/abs/1911.06854). Survey empírico de métodos de OPE. Buena entrada al campo.
- **Fu et al. (2021) — "Benchmarks for Deep Off-Policy Evaluation"** → [arXiv:2103.16596](https://arxiv.org/abs/2103.16596). El paper de **DOPE benchmark**. Estructura el campo.
- **Sección OPE del tutorial de Levine et al. 2020**.

### Temas mínimos

**Importance Sampling (IS) para OPE**: estimar $V^\pi$ a partir de datos generados por $\pi_\beta$ usando el ratio $\rho_t = \pi(a_t|s_t)/\pi_\beta(a_t|s_t)$. Variantes: per-decision IS, weighted IS, doubly robust IS. **El problema de la varianza explosiva**: en horizonte largo, los ratios IS se multiplican y la varianza puede ser astronómica. **Fitted Q Evaluation (FQE)**: entrena una Q-function por TD para evaluar una política dada — más estable que IS pero introduce bias. **Model-based OPE**: aprende un modelo del entorno, ejecuta la política en él. **Métodos de regression**: aprende $V^\pi$ por regresión directa. **Métricas de fiabilidad**: en lugar de un número, da intervalos de confianza estadísticamente correctos. **Conexión con tu tesis**: en federated OPE, los clientes evalúan localmente y agregan estimaciones; la composición de errores es no trivial.

### Ejercicio

Implementa **FQE** desde cero sobre un dataset D4RL/Minari. Entrena IQL sobre el mismo dataset, evalúa la política aprendida con FQE, y **compara contra la evaluación verdadera ejecutando la política en el entorno real**. Mide el sesgo y la varianza del estimador FQE. Esto es **exactamente** el experimento que tu tesis va a necesitar replicar en federated, así que prepáralo bien.

### Checklist 4.8

- [ ] Leí el survey de Voloshin 2019 y el paper de DOPE
- [ ] Implementé FQE y lo comparé con ground-truth en un dataset D4RL
- [ ] Sé explicar por qué importance sampling explota en varianza con horizonte largo
- [ ] Entiendo por qué OPE es crítico para tu tesis federada y por qué los estándares actuales son insuficientes en setting Pontus-X

---

## 4.9 — Disciplina experimental específica de Offline RL

### Por qué importa

La cultura experimental de la Fase 2 (≥5 seeds, IQM, `rliable`) **sigue aplicando** pero en Offline RL hay **patologías propias** que tienes que conocer:

- **Hiperparámetros sensibles**: CQL es notoriamente difícil de tunear; lo que funciona en Hopper-medium puede fallar en Walker-medium-expert. Tarasov et al. 2024 ("Revisiting the Minimalist Approach") demostraron que muchos resultados publicados de offline RL no se sostienen con hiperparámetros honestos.
- **No hay validation set claro**: en supervised learning tienes train/val/test; en offline RL **no tienes nada para hacer model selection sin ejecutar en el entorno**. Esto es **una contradicción fundamental** del paradigma offline puro.
- **Online eval para reportar es una trampa que todo el mundo usa**: la mayoría de papers de Offline RL **sí ejecutan en el entorno** para reportar el número final. Esto es honestamente cuestionable. OPE puro estaría bien, pero nadie lo hace porque no es competitivo aún.
- **Algoritmos sobre-tuneados a D4RL específicamente**: cuando emergieron benchmarks alternativos (NeoRL, V-D4RL), muchos algoritmos cayeron significativamente. El campo tiene un problema de overfitting a un benchmark.

### Lecturas

- **Tarasov, Kurenkov, Nikulin & Kolesnikov (2024) — "Revisiting the Minimalist Approach to Offline Reinforcement Learning"** → [arXiv:2305.09836](https://arxiv.org/abs/2305.09836). **Léelo entero**. Hace una auditoría honesta del campo y propone ReBRAC. Esta es la lectura que **debería preocupar a cualquier doctorando** que vaya a hacer offline RL.
- **Agarwal, Schwarzer, Castro, Courville & Bellemare (2021) — "Statistical Precipice"** → ya conocido de Fase 2. Sus métricas (IQM, performance profiles, optimality gap) **son obligatorias** en Offline RL también.
- **NeoRL: Qin, Gao, Zhang, Liu & Liu (2022)** → [arXiv:2102.00714](https://arxiv.org/abs/2102.00714). Benchmark alternativo a D4RL con datasets más realistas (industriales, médicos). **Mucho más cercano al setting de Pontus-X** que los datasets de locomoción de D4RL.

### Disciplina mínima específica de Offline RL

- **Reportar siempre `rliable` IQM** y bootstrap CI. La media simple oculta los failure modes.
- **Reportar al menos 3 dataset types** (`expert`, `medium`, `medium-replay` o equivalentes). Reportar solo uno es cherry-picking.
- **Reportar hiperparámetros completos en apéndice**. Si tu método requiere tuning específico por dataset, **dilo explícitamente** — eso es información, no debilidad.
- **Comparar con CORL/d3rlpy como ground truth de implementación** de baselines. Si tu BC pierde a CORL BC por 30%, **tu BC está mal implementado, no es que CORL sea misteriosamente mejor**.
- **Diferenciar entre evaluación online (rollouts en el simulador) y OPE pura**. Si reportas online, dilo. No vendas "es puramente offline" cuando tu evaluación no lo es.
- **Performance profiles** en lugar de tablas de medias cuando comparas múltiples algoritmos en múltiples datasets.

### Checklist 4.9

- [ ] Leí Tarasov et al. 2024 y sé enumerar 3 problemas reproducibilidad concretos del campo
- [ ] Uso `rliable` en todos mis experimentos offline
- [ ] Comparo contra CORL como ground truth implementacional
- [ ] Conozco NeoRL como benchmark alternativo a D4RL
- [ ] Reporto hiperparámetros completos por dataset, sin esconder el tuning

---

## 4.10 — Proyecto integrador: replicación + tesis-relevant pipeline

### Por qué importa

**Este es el primer artefacto de tu carrera que va a aparecer en tu tesis.** Lo de las Fases 1–2 eran portfolios bonitos. Esto **ya es producción doctoral preliminar**.

### Estructura mínima del repo

```
offline-rl-replication/
├── README.md                    # narrativo, gráficas, tabla replicación vs CORL
├── algorithms/
│   ├── bc.py                    # behavioral cloning
│   ├── td3_bc.py
│   ├── cql.py
│   ├── iql.py                   # el que vas a usar como baseline en tu tesis
│   └── fqe.py                   # para OPE
├── datasets/                    # interfaces a Minari + D4RL legacy
├── experiments/
│   ├── replication.py           # replica resultados de CORL para validación
│   ├── ablations.py
│   └── ope_study.py             # FQE vs online ground truth
├── configs/                     # Hydra YAMLs por algoritmo×dataset
├── figures/                     # generadas por rliable
├── reports/
│   └── replication_report.pdf   # 6 páginas: tu mini-paper de replicación
├── thesis_relevant/             # ← esta carpeta es nueva y es la importante
│   ├── README.md                # qué de este repo entra en tu tesis
│   └── notes_for_supervisor.md  # tu análisis del campo en relación a Pontus-X
├── tests/
├── requirements.txt
└── notebooks/
```

### Tabla obligatoria — la que vas a defender ante tu director

| Algoritmo | hopper-med | hopper-med-rep | hopper-med-exp | walker-med | walker-med-rep | walker-med-exp | halfch-med | halfch-med-rep | halfch-med-exp |
|---|---|---|---|---|---|---|---|---|---|
| BC | | | | | | | | | |
| TD3+BC | | | | | | | | | |
| CQL | | | | | | | | | |
| IQL | | | | | | | | | |

≥5 seeds, IQM, bootstrap CI 95%. Diferencia con CORL ≤10% en cada celda. **Este es el primer pedazo de trabajo que vas a mandar a un supervisor potencial.**

### Mini-paper de replicación (6 páginas)

Formato NeurIPS/ICML. Estructura:
1. **Intro**: motivación de la replicación, conexión con el tema de tesis (Federated RL sobre Compute-to-Data).
2. **Background**: 1 página sobre Offline RL y por qué importa para dataspaces.
3. **Algoritmos**: descripción rápida de los 4 baselines.
4. **Experimentos**: tabla de replicación, performance profiles, análisis de varianza entre seeds.
5. **Análisis específico para tesis**: ¿qué de estos algoritmos sobrevive cuando hay heterogeneidad entre clientes? Hipótesis (no tienes que demostrarlas todavía, solo formularlas).
6. **Trabajo futuro**: federación, OPE federado, privacy.

### `thesis_relevant/notes_for_supervisor.md`

Documento privado, 2–3 páginas, donde escribes:
- **Tu análisis** de qué algoritmos offline parecen más federables y por qué (IQL vs CQL vs TD3+BC desde la perspectiva de qué se envía entre clientes)
- **Una matriz de gaps** en la literatura: lo que hay y lo que no hay (Offline FRL en non-IID extremo, Offline FRL con DP, etc.)
- **Tres preguntas concretas** que querrías discutir con tu supervisor potencial

Este documento **no es para publicar**: es para **abrir la conversación doctoral**. Cuando contactes con un supervisor potencial, mandas el mini-paper + este documento + el repo público. **Esa terna te diferencia del 90% de los candidatos**.

### Checklist 4.10

- [ ] Repo público con BC, TD3+BC, CQL, IQL implementados
- [ ] Tabla de replicación contra CORL con ≥5 seeds e IQM, diferencias ≤10%
- [ ] Mini-paper de 6 páginas en `reports/`
- [ ] Documento `notes_for_supervisor.md` con análisis tesis-relevante
- [ ] FQE implementado y comparado contra evaluación online (bloque OPE)
- [ ] CI verde, tests, configs en Hydra, logging W&B

---

## Stack técnico canónico (resumen)

Para que no busques estas decisiones cada vez:

- **Librería de algoritmos**: **CORL** ([tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)). Single-file, calidad de investigación, con W&B integrado. **Tu referencia**.
- **Librería industrial alternativa**: **d3rlpy** ([takuseno/d3rlpy](https://github.com/takuseno/d3rlpy)). Más abstracto, mejor para producción. Útil para comparar.
- **Datasets**: **Minari** ([Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari)) es el sucesor oficial de D4RL bajo la Farama Foundation. **D4RL legacy sigue siendo válido** porque la mayoría de papers reportan sobre él, pero Minari es donde el ecosistema converge. Aprende a usar ambos.
- **Benchmark alternativo**: **NeoRL** para datasets más realistas (cercanos a dataspaces).
- **Evaluación estadística**: **`rliable`** (Agarwal et al.).
- **Tracking**: **W&B**.

---

## Pre-final: lo que se ve cuando se cierra la Fase 4

Cuando termines esto, deberías poder:
- Leer cualquier paper de Offline RL del 2020–2026 sin tropezar con la notación, los nombres de baselines, o los hiperparámetros estándar.
- Implementar IQL desde cero en una tarde, CQL en dos tardes.
- Defender en una pizarra la diferencia entre policy constraint (TD3+BC), value regularization (CQL) e implicit Q-learning (IQL), y predecir cuál es más adecuado para cada tipo de dataset.
- Tener una **opinión formada y defensible** sobre qué algoritmos offline son más prometedores para tu setting federado de dataspaces.
- Leer un paper de Federated Offline RL (Fase 6) y reconocer **exactamente** qué componente offline usa, qué problema de federación introduce, y qué gaps quedan. **Ese reconocimiento es la prueba de que Fase 4 está cerrada y Fase 6 puede empezar a tomar forma.**

---

## Plan sugerido de 8 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 4.1 Distributional shift + tutorial Levine 2020 | Setup d3rlpy/CORL/Minari |
| 2 | 4.2 BC baseline + 4.3 TD3+BC | Replicación TD3+BC vs CORL |
| 3 | 4.4 CQL (lectura del paper + apéndice) | Implementación CQL básica |
| 4 | 4.4 CQL completo + reproducción D4RL | Tabla CQL replicada |
| 5 | 4.5 IQL — la semana central | Implementación + replicación IQL |
| 6 | 4.6 MOPO/COMBO + 4.7 Decision Transformer | Lecturas, no implementación profunda |
| 7 | 4.8 OPE + FQE | Experimento OPE vs ground-truth |
| 8 | 4.9 Disciplina + 4.10 Mini-paper + repo | Documento para supervisor |

Si IQL te cuesta más de una semana, **estíralo sin culpa**. Es el algoritmo que más te va a rentar durante la tesis.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:
- Derivaciones desarrolladas (especialmente expectile regression para IQL y el lower bound de CQL)
- Código PyTorch comentado línea a línea
- Análisis críticos de los papers principales — qué prometen, qué cumplen, qué esconden
- Conexiones concretas con Federated RL para preparar el terreno de Fase 6

Mi recomendación de orden:
1. **4.1 + 4.2 + 4.3 como un bloque dedicado** (problema fundamental + baselines): es la columna conceptual de toda la fase. Sin esto sólido, los algoritmos sofisticados son magia.
2. **4.4 + 4.5 como otro bloque dedicado** (CQL e IQL): los dos algoritmos que vas a usar más en tu tesis. Inviértele tiempo asimétrico.
3. **4.8 + 4.9 + 4.10 como bloque final** (OPE + disciplina + mini-paper): aquí es donde construyes el artefacto que vas a usar para iniciar conversaciones doctorales.

Y un aviso final, que repito porque vale repetir: **la Fase 4 ya no es preparación**. Es **el primer terreno tesis-relevante** de tu camino. Léela con la lupa de "qué de esto entra en mi tesis", y el documento `notes_for_supervisor.md` del bloque 4.10 es **literalmente** el primer artefacto que vas a presentar como pre-candidato a un programa doctoral. Hazlo con la seriedad que merece.
