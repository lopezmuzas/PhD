---
title: "Fase 2 — Deep Reinforcement Learning"
tags: [rl, refuerzo]
status: borrador
updated: 2026-08-08
---

# Fase 2 — Deep Reinforcement Learning

> **Objetivo de la fase**: dar el salto de tabular a **aproximación de funciones** con redes neuronales. Entender por qué la combinación *bootstrapping + off-policy + function approximation* es la **Deadly Triad** que rompe la convergencia teórica, y por qué los trucos prácticos de DQN (replay buffer, target network, gradient clipping) no son arbitrarios sino respuestas concretas a esa triada. Salir con un **DQN propio** que supere Atari Pong en píxeles, y con la cultura experimental de un investigador serio: ≥5 seeds, bandas de confianza, tests estadísticos, y nada de "media sin error bars".
>
> **Tu situación de partida**: vienes de la Fase 1 con Q-Learning y SARSA implementados desde cero en NumPy, sabiendo demostrar la convergencia de value iteration. PyTorch lo tienes a punto desde la Fase 0. Aquí entra **por fin** en juego de verdad.
>
> **Tiempo estimado realista**: 8–10 semanas a ritmo de 1–2 h/día. Y a diferencia de la Fase 1, **esta fase consume GPU**: entrenar DQN en Pong tarda ~12–24h por seed. Si no tienes GPU propia, presupuesta créditos de Colab Pro, Lambda Labs, Modal o vast.ai (~50–100 € en total razonablemente).

---

## Cómo usar esta guía

Misma lógica que las fases anteriores. Pero un aviso específico para esta:

**Deep RL es la fase donde aparecen los bugs silenciosos.** En tabular, si Q-Learning está mal implementado, no aprende y te enteras enseguida. En Deep RL, un `.detach()` olvidado puede hacer que tu DQN aprenda *algo* — pero peor de lo que debería — y nunca te enteres porque la curva sube. **La disciplina experimental no es opcional aquí**: si no comparas contra baselines establecidos (CleanRL, SB3) seed-por-seed, no sabes si tu implementación funciona o si tienes un bug que estás interpretando como "es que mi versión es más lenta".

---

## Mapa de la Fase 2

| Bloque | Tema | Peso |
|---|---|---|
| 2.1 | Function approximation y la Deadly Triad | 🔥🔥🔥🔥🔥 La pregunta central de la fase |
| 2.2 | DQN: el algoritmo y sus trucos | 🔥🔥🔥🔥🔥 El paper fundacional |
| 2.3 | Mejoras sobre DQN: Double, Dueling, PER, Rainbow | 🔥🔥🔥🔥 SOTA en value-based discreto |
| 2.4 | Policy Gradients: REINFORCE | 🔥🔥🔥🔥 La otra rama de Deep RL |
| 2.5 | Variance reduction: baselines, advantages, GAE | 🔥🔥🔥🔥🔥 Sin esto no entiendes PPO |
| 2.6 | Actor-Critic: A2C/A3C | 🔥🔥🔥 Puente a la Fase 3 |
| 2.7 | Reproducibilidad y disciplina experimental | 🔥🔥🔥🔥🔥 El sello de investigador |
| 2.8 | Ingeniería: vectorización, wrappers, logging | 🔥🔥🔥 Lo que nadie te enseña |
| 2.9 | Proyecto integrador: DQN propio en Atari Pong | 🔥🔥🔥🔥🔥 Tu segunda pieza de portfolio |

---

## 2.1 — Function approximation y la Deadly Triad

### Por qué importa — y esta es la pregunta que define toda la fase

En la Fase 1 demostraste que Q-Learning converge cuando la tabla Q es exacta. **En cuanto sustituyes la tabla por una red neuronal, esa demostración se rompe**. Y no se rompe en un sentido teórico abstracto: se rompe en la práctica con divergencias visibles. La razón es la **Deadly Triad** identificada por Sutton & Barto: cuando combinas (1) function approximation, (2) bootstrapping, y (3) off-policy learning, la convergencia ya no está garantizada. DQN funciona **a pesar** de tener las tres, no gracias a ellas — y todos los "trucos" del paper de Mnih (target network, replay buffer, clipping, frame stacking) son **respuestas concretas a la triada**. Si entiendes esto, todo el paper de DQN deja de ser una receta y se convierte en una secuencia de decisiones razonadas.

### Recurso principal

- **Sutton & Barto, capítulos 9, 10 y 11** → [PDF gratis legal](http://incompleteideas.net/book/RLbook2020.pdf). El 11 es **el capítulo de la Deadly Triad** — Off-policy methods with approximation. Léelo entero. Es denso, pero es el único sitio donde está explicado a fondo.
- **David Silver — Lecture 6 (Value Function Approximation)** → [YouTube](https://www.youtube.com/watch?v=UoPei5o4fps). Silver explica linear function approximation antes de pasar a neuronal. **No la saltes** porque "ya sabes redes": la intuición lineal te da el lenguaje para entender por qué la triada es un problema.

### Refuerzo teórico

- **Berkeley CS285 — Lecture 7 (Value Function Methods)** y **Lecture 8 (Deep RL with Q-Functions)** → [Curso completo](http://rail.eecs.berkeley.edu/deeprlcourse/). Las lecciones siguen activas en Spring 2026, con grabaciones de Fall 2023 disponibles. Levine explica la triada con la severidad que merece. Para ti, **CS285 deja de ser opcional a partir de esta fase**: es la referencia académica de Deep RL.
- **Tsitsiklis & Van Roy (1997) — "An Analysis of Temporal-Difference Learning with Function Approximation"**. El paper clásico que demuestra que TD lineal off-policy puede diverger. Léelo: 4 páginas relevantes, ejemplo de Baird (counterexample of divergence) incluido.

### Temas mínimos

Function approximation lineal: $\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$ y por qué con features adecuadas TD on-policy converge pero off-policy no. Gradient TD methods (GTD, TDC) y semi-gradient methods — entiende **la diferencia**: TD estándar no es realmente gradient descent sobre ninguna función de pérdida bien definida (es **semi-gradient**). **Counterexample de Baird** — el ejemplo de 7 estados que rompe TD lineal off-policy. Reprodúcelo a mano en una hoja. Definición precisa de la Deadly Triad y por qué cada uno de los tres ingredientes contribuye. Bootstrapping vs. Monte Carlo en función aproximada. Convergencia de TD(0) en el caso on-policy lineal (proyección sobre el espacio de features). Por qué el "true gradient" del TD error (residual gradient) sí converge pero es lento — y por qué nadie lo usa en la práctica.

### Ejercicio obligatorio

Implementa el **counterexample de Baird** en NumPy puro. Es un MDP de 7 estados con features lineales específicos. Verás cómo los pesos divergen visiblemente con TD(0) off-policy. Es uno de los experimentos más esclarecedores que puedes hacer en tu vida en RL.

### Checklist 2.1

- [ ] Sé enunciar la Deadly Triad y explicar por qué cada uno de sus tres componentes es necesario para la divergencia
- [ ] Implementé el counterexample de Baird y vi los pesos divergir
- [ ] Entiendo la diferencia entre semi-gradient y true gradient TD methods
- [ ] He leído el capítulo 11 de Sutton & Barto completo

---

## 2.2 — DQN: el algoritmo y sus trucos

### Por qué importa

Es **el paper que reinició el campo** de Deep RL. Mnih et al. 2015 demostraron que con suficientes trucos, podías aprender directamente de píxeles a una política superhumana en docenas de juegos Atari. Implementarlo entero, desde cero, y hacer que **funcione de verdad** (no que "compila"), es **el rito de paso** de cualquier doctorando en RL.

Pero la lección importante no es el algoritmo en sí; es la **estructura de problema-respuesta**: cada truco ataca un fallo concreto de "Q-Learning + red neuronal". Replay buffer ataca correlación temporal y eficiencia de datos. Target network ataca el "moving target" que rompe la convergencia. Frame stacking ataca la no-markovianidad de un solo frame de Atari. Clipping de recompensa y de gradiente atacan la inestabilidad numérica. Saber por qué cada uno está ahí es la diferencia entre copiar el paper y entenderlo.

### Recurso principal

- **Mnih et al. (2015) — "Human-level control through deep reinforcement learning"** → [Nature](https://www.nature.com/articles/nature14236) (también versión preprint del 2013, "Playing Atari with Deep RL" en arXiv). **Léelo entero**, incluyendo el supplementary material. Marca cada decisión de implementación: por qué Huber loss en vez de MSE, por qué RMSProp con esos hiperparámetros, por qué clipear el TD error a [-1, 1], por qué actualizar el target cada 10k steps. Este paper es **lectura activa con bolígrafo**, no de cama.
- **Berkeley CS285 — Lecture 8 (Deep RL with Q-Functions)** → [Curso completo](http://rail.eecs.berkeley.edu/deeprlcourse/). Levine explica los trucos con su severidad habitual.

### Refuerzo

- **CleanRL — `dqn.py` y `dqn_atari.py`** → [GitHub](https://github.com/vwxyzjn/cleanrl). **Esta es tu referencia de código de aquí en adelante**. Lectura obligatoria línea por línea. CleanRL son implementaciones single-file: todo el algoritmo en un fichero, sin abstracciones. Es la forma más pedagógica de código de RL que existe.
- **Aleksa Gordić — The AI Epiphany, DQN walkthrough** → [YouTube](https://www.youtube.com/@TheAIEpiphany). Vídeos *line-by-line* de implementaciones. Buen complemento a CleanRL.
- **Lilian Weng — "A (Long) Peek into Reinforcement Learning"** → [Blog](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) y los posts específicos de DQN. Las notas de Lilian son referencia universal en el campo; léelas.

### Temas mínimos

Arquitectura de la Q-network: CNN sobre 4 frames apilados de 84×84, dos cabezas (online y target). **Experience replay**: por qué rompe la correlación temporal, tamaño típico (1M transiciones), implementación con buffer circular. **Target network**: por qué un objetivo que se mueve cada paso impide convergencia, frecuencia de actualización (hard update cada 10k steps vs. soft update con $\tau$ — Polyak averaging). **ε-greedy con decay lineal** de 1.0 → 0.1 en el primer millón de frames. **Huber loss** (pseudo-Huber) en lugar de MSE — más robusta a outliers en el TD error. **Frame stacking**: 4 frames consecutivos para inferir velocidad y dirección (markovianización de un POMDP). **Preprocessing** estándar de Atari: grayscale, resize a 84×84, max pooling entre frames consecutivos, episodic life, no-op start. **Clipping**: del TD error y/o del gradiente. **Skipping de frames** (action repeat = 4).

### Stack técnico actualizado (importante)

El ecosistema ha cambiado desde el paper original; estos son los nombres canónicos hoy:

- **Gymnasium** (sucesor mantenido de OpenAI Gym) + **ale-py** para el emulador. La instalación canónica es `pip install gymnasium[atari] gymnasium[accept-rom-license]`.
- Entorno: **`ALE/Pong-v5`** con `frameskip=4` y sticky actions. Olvídate de `PongNoFrameskip-v4` (deprecated en gymnasium moderno).
- **Stable-Baselines3 wrappers**: `stable_baselines3.common.atari_wrappers.AtariWrapper` te da los wrappers canónicos (max-pool, skip, grayscale, resize, frame-stack). Úsalos: reescribirlos a mano es una pérdida de tiempo y una fuente de bugs.
- **`AutoROM --accept-license`** para descargar los ROMs (necesario una vez).

### Ejercicio obligatorio — el grande de la fase

Implementa **DQN desde cero en PyTorch**, con tu propio replay buffer, target network y arquitectura CNN. Apóyate en `cleanrl/dqn_atari.py` como referencia pero **no copies y pegues**: escribe línea a línea entendiendo cada una. Entrénalo en **ALE/Pong-v5 desde píxeles** hasta que supere la media de +18 (de 21 posibles) sostenida sobre los últimos 100 episodios. Esto te llevará ~12–24h por seed en GPU media.

Antes de lanzar el experimento largo, **valida en CartPole-v1**: tu DQN con una MLP pequeña (sin CNN) debe resolver CartPole en <500 episodios consistentemente. Si CartPole no funciona, **no lances Pong**: tienes un bug. Esa regla — "verifica en juguete antes de tirar GPU" — te ahorrará semanas de tu vida investigadora.

### Checklist 2.2

- [ ] Implementé DQN desde cero, con replay buffer y target network propios
- [ ] Mi DQN resuelve CartPole-v1 consistentemente
- [ ] Mi DQN supera Pong en píxeles con media ≥+18 sobre últimos 100 episodios
- [ ] Leí Mnih et al. 2015 entero y entiendo qué problema concreto resuelve cada truco
- [ ] Sé explicar por qué hay `.detach()` en el target network y qué pasaría sin él

---

## 2.3 — Mejoras sobre DQN: Double, Dueling, PER, Rainbow

### Por qué importa

DQN vanilla tiene problemas conocidos: **sobreestimación sistemática** del valor Q (Hasselt 2010), separación insuficiente entre valor de estado y ventaja de acción, muestreo uniforme del buffer ignorando que algunas transiciones son más informativas que otras. Cada paper subsiguiente — Double, Dueling, PER — ataca uno de esos problemas. Rainbow (Hessel et al. 2018) demostró que **las mejoras son complementarias**: combinarlas todas da un agente que destruye al DQN original.

Para ti hay además una conexión doctoral concreta: **Prioritized Experience Replay** es **prima cercana de prioritized sweeping** de Dyna-Q (Fase 1) y **ancestro directo** de los métodos modernos de sample-efficient RL — exactamente lo que vas a necesitar en Offline RL (Fase 4) y en Federated RL con comunicación cara (Fase 6) donde cada muestra cuesta dinero.

### Lecturas obligatorias (papers)

- **Van Hasselt et al. (2016) — "Deep Reinforcement Learning with Double Q-learning"** → [arXiv:1509.06461](https://arxiv.org/abs/1509.06461). El bug: el $\max$ en el target de DQN sobrestima sistemáticamente. La fix: usar la red online para *seleccionar* la acción y la red target para *evaluarla*. Cambio de **una línea de código**, mejora masiva.
- **Wang et al. (2016) — "Dueling Network Architectures for Deep Reinforcement Learning"** → [arXiv:1511.06581](https://arxiv.org/abs/1511.06581). Descomposición $Q(s,a) = V(s) + A(s,a)$ en dos cabezas de la red. Útil cuando muchas acciones tienen valores parecidos.
- **Schaul et al. (2016) — "Prioritized Experience Replay"** → [arXiv:1511.05952](https://arxiv.org/abs/1511.05952). Muestrear transiciones con probabilidad proporcional al TD error en lugar de uniformemente. Importance sampling correction obligatorio (de ahí el peso $w_i$). Estructura de datos: sum-tree.
- **Bellemare et al. (2017) — "A Distributional Perspective on Reinforcement Learning"** → [arXiv:1707.06887](https://arxiv.org/abs/1707.06887). C51: aprender la **distribución** del retorno, no solo su esperanza. Conceptualmente importante: si tu tesis se mueve hacia uncertainty-aware RL, este paper es seminal.
- **Hessel et al. (2018) — "Rainbow: Combining Improvements in Deep RL"** → [arXiv:1710.02298](https://arxiv.org/abs/1710.02298). La síntesis. Combina Double + Dueling + PER + Multi-step + Distributional + NoisyNets. Excelente paper para ver qué *ablation studies* serios parecen.

### Refuerzo

- **CleanRL — `c51.py`, y los notebooks de Rainbow components**. Para Rainbow completo, mira repos especializados como **rainbow-is-all-you-need** ([GitHub](https://github.com/Curt-Park/rainbow-is-all-you-need)) — una serie de notebooks pedagógicos que añaden cada componente uno a uno.

### Temas mínimos

**Maximization bias**: por qué $\max E[X] \leq E[\max X]$ implica sobreestimación, y la intuición de cómo Double Q-Learning lo evita. **Dueling architecture**: cabeza de valor + cabeza de advantage, con normalización (típicamente sustraer media) para identificabilidad. **PER**: prioridad $p_i = |\delta_i| + \epsilon$, probabilidad de muestreo $P(i) = p_i^\alpha / \sum_k p_k^\alpha$, peso IS $w_i = (N \cdot P(i))^{-\beta}$ con $\beta$ que sube linealmente hacia 1. **Sum-tree** como estructura de datos eficiente para muestreo proporcional. **n-step returns en deep**: el mismo concepto que en Fase 1, ahora con neural Q-function. **NoisyNets** como alternativa a ε-greedy.

### Ejercicio

Toma tu DQN de Pong del bloque 2.2. Añade **una mejora a la vez** y compara curvas:
1. Baseline (vanilla DQN)
2. + Double DQN
3. + Dueling
4. + PER
5. + n-step (n=3)

Cada experimento con ≥3 seeds (idealmente 5) y bandas de confianza. Esta es **tu primera tabla de ablación** — y la tabla de ablación es **la unidad estructural** de los papers de RL. Hazla bien.

Si te queda energía, implementa Rainbow completo en un entorno más rápido (LunarLander) y compara contra DQN vanilla.

### Checklist 2.3

- [ ] Sé explicar el maximization bias y cómo Double DQN lo arregla con un ejemplo numérico
- [ ] Entiendo la arquitectura Dueling y por qué hace falta normalizar la cabeza de advantage
- [ ] Implementé PER con sum-tree y entiendo el papel del importance sampling weight
- [ ] Tengo una tabla de ablación con ≥3 seeds en Pong: vanilla → +Double → +Dueling → +PER
- [ ] Leí Rainbow entero y entiendo qué hacen los seis componentes

---

## 2.4 — Policy Gradients: REINFORCE

### Por qué importa

Hasta aquí, todo era **value-based**: estimar $Q$ y derivar política $\pi(a|s) = \arg\max_a Q(s,a)$. Pero hay un problema: en **espacios de acción continuos** (control de robots, MuJoCo), $\arg\max_a Q(s,a)$ requiere optimización en cada paso, lo cual es intratable. **La solución es parametrizar la política directamente** y optimizarla por gradient ascent sobre el retorno esperado. Eso es REINFORCE (Williams 1992), y es la base de toda la rama que va a culminar en PPO/SAC en la Fase 3.

Hay otra razón doctoral: en Offline RL (Fase 4), muchos métodos modernos son policy-based (BCQ, BEAR, TD3+BC, IQL todavía estima Q pero la política se extrae por *advantage-weighted regression*). Y en Federated RL (Fase 6), federar gradientes de política (PAvg) es una de las dos estrategias canónicas, junto con federar Q-values (QAvg). Sin entender bien policy gradients, te quedas fuera de la mitad de la literatura.

### Recurso principal

- **Sutton & Barto, capítulo 13 entero**. Es relativamente corto y contiene la derivación del policy gradient theorem desde cero. **Sigue la derivación con bolígrafo** — es el paso clave.
- **Berkeley CS285 — Lecture 5 (Policy Gradients)** y **Lecture 6 (Actor-Critic)** → [Curso](http://rail.eecs.berkeley.edu/deeprlcourse/). Levine pasa media hora justificando por qué el gradiente tiene esa forma, incluyendo el log-derivative trick (REINFORCE trick). Es la mejor explicación pedagógica.

### Refuerzo

- **OpenAI Spinning Up — Policy Gradients intro** → [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). La derivación más limpia que existe en formato corto. Imprime esta página.
- **Lilian Weng — "Policy Gradient Algorithms"** → [Blog](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). La taxonomía completa de policy gradient methods, con derivaciones. Una referencia universal.

### Lectura histórica

- **Williams (1992) — "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"** → el paper original de REINFORCE. Léelo: es 30 años más antiguo que DQN y todavía es relevante.
- **Sutton, McAllester, Singh & Mansour (2000) — "Policy Gradient Methods for Reinforcement Learning with Function Approximation"** → el paper que formaliza el Policy Gradient Theorem.

### Temas mínimos

El **Policy Gradient Theorem**: $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)]$. **Log-derivative trick**: $\nabla \log p = \nabla p / p$ y por qué eso te permite estimar el gradiente por Monte Carlo. **REINFORCE algoritmo**: rollout completo, calcular $G_t$ por episodio, aplicar el gradiente. **Por qué REINFORCE tiene altísima varianza**: porque multiplica por el retorno completo del episodio (escala con horizonte). **Políticas estocásticas**: Categorical para acciones discretas (softmax sobre logits), Normal para acciones continuas (output: media y std de una gaussiana). **`log_prob` en `torch.distributions`**: el método que vas a usar mil veces. **Entropía** de la política y por qué es útil como regularizador (bonus para mantener exploración).

### Ejercicio

Implementa **REINFORCE puro** (sin baseline aún) en CartPole-v1. Verás que aprende, pero con **una varianza brutal**: a veces converge en 200 episodios, a veces 2000, a veces no converge. Esto es **deliberadamente** lo que se busca ver: prepara el terreno para el bloque 2.5, donde añadir un baseline va a transformar las curvas.

Después, REINFORCE con política gaussiana en Pendulum-v1 (continuo). Compara con tu intuición de "qué es una buena política" para ese problema.

### Checklist 2.4

- [ ] Derivé el Policy Gradient Theorem en papel sin mirar
- [ ] Implementé REINFORCE en PyTorch con `torch.distributions.Categorical`
- [ ] Mi REINFORCE resuelve CartPole-v1 (aunque con alta varianza entre seeds)
- [ ] Implementé REINFORCE con política gaussiana en Pendulum-v1
- [ ] Sé explicar qué hace `log_prob(action)` línea por línea

---

## 2.5 — Variance reduction: baselines, advantages, GAE

### Por qué importa — y este es el bloque que te separa de un practicante de policy gradients

REINFORCE puro tiene varianza tan alta que es prácticamente inútil en problemas no triviales. **Toda la rama policy-based de los últimos 30 años es esencialmente trucos de reducción de varianza**: baselines, advantages, GAE, TRPO, PPO. Si saltas este bloque, PPO te parecerá magia con un parámetro $\lambda$ misterioso y un *clipping* de ratio inexplicable. Si lo dominas, PPO es trivial.

Y específicamente: **GAE (Schulman et al. 2016) es el blueprint que hace que PPO funcione de verdad**. Es la generalización natural de las n-step returns de la Fase 1 al setting de policy gradients. Si la Fase 1 fue la antesala, esto es donde la conexión cierra.

### Lectura obligatoria

- **Schulman et al. (2016) — "High-Dimensional Continuous Control Using Generalized Advantage Estimation"** → [arXiv:1506.02438](https://arxiv.org/abs/1506.02438). **GAE**. Este paper es relativamente corto y absolutamente central. Léelo dos veces. La ecuación 16 (la fórmula recursiva de GAE) la vas a implementar a mano. Pasa el rato entendiendo el trade-off bias-variance que parametriza $\lambda$: $\lambda=0$ es TD(0) (bajo varianza, alto bias), $\lambda=1$ es MC (alta varianza, no bias).
- **Sutton & Barto, sección 13.4** sobre REINFORCE with baseline.

### Refuerzo

- **Berkeley CS285 — Lecture 6 (Actor-Critic Algorithms)**. Levine deriva paso a paso por qué un baseline no cambia la esperanza pero reduce la varianza, y conecta directamente con advantages.
- **Spinning Up — Vanilla Policy Gradient (VPG)** → [spinningup.openai.com/.../vpg](https://spinningup.openai.com/en/latest/algorithms/vpg.html). Implementación de referencia, con explicación pedagógica.

### Temas mínimos

**Baseline**: cualquier función $b(s)$ que **no depende de $a$** puede restarse del retorno sin introducir bias en el gradiente: $\nabla J = \mathbb{E}[\nabla \log \pi \cdot (G - b(s))]$. Demostración de por qué el bias es cero. **Baseline óptimo**: el que minimiza la varianza es la **función de valor** $V(s)$, intuitivamente. **Advantage**: $A(s,a) = Q(s,a) - V(s)$. Mide "cuánto mejor es la acción $a$ que la media bajo $\pi$". **Estimación del advantage**: las opciones canónicas son $A \approx G_t - V(s_t)$ (MC), $A \approx r + \gamma V(s') - V(s)$ (TD), o **GAE** (interpolación geométrica entre ellas). **GAE fórmula**: $\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ donde $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Implementación recursiva eficiente (de derecha a izquierda en el rollout). **Importance sampling** en policy gradients para off-policy correction (aparecerá en PPO en Fase 3, anticipa la conexión).

### Ejercicio

Sobre el REINFORCE de CartPole del bloque anterior, añade **iterativamente**:
1. REINFORCE + baseline constante (media de retornos)
2. REINFORCE + baseline $V(s)$ aprendido por una red de valor (esto **ya es A2C esencialmente**)
3. Lo anterior + GAE con $\lambda \in \{0, 0.5, 0.95, 1.0\}$

Compara curvas: deberías ver una reducción **drástica** de varianza entre 1 y 2, y un sweet spot empírico en GAE con $\lambda \approx 0.95$.

### Checklist 2.5

- [ ] Demostré en papel que un baseline $b(s)$ no introduce bias en el gradiente
- [ ] Implementé GAE recursivamente y verifiqué que con $\lambda=0$ recupero TD y con $\lambda=1$ recupero MC
- [ ] Mi REINFORCE+baseline+GAE en CartPole tiene varianza visiblemente menor que REINFORCE puro
- [ ] Sé explicar por qué el advantage es preferible al retorno crudo para policy gradients
- [ ] Anticipo cómo GAE aparece en PPO y por qué $\lambda \approx 0.95$ es el default casi universal

---

## 2.6 — Actor-Critic: A2C / A3C

### Por qué importa

Actor-Critic es **literalmente** la combinación de las dos ramas: un *actor* que es una red política $\pi_\theta(a|s)$ entrenada con policy gradients, y un *critic* que es una red de valor $V_\phi(s)$ entrenada con TD. El actor decide qué hacer, el critic le dice cuán bueno es lo que ha hecho. **Casi todos los algoritmos SOTA de Deep RL son actor-critic**: A2C, A3C, PPO, TRPO, SAC, TD3, DDPG.

A2C/A3C son el **puente directo a Fase 3**. Si los dominas, PPO en Fase 3 será extender A2C con un trust region; SAC será A2C en continuo con entropía maximizada; TD3 será DDPG con trucos de DQN. La inversión aquí paga toda la Fase 3.

### Recurso principal

- **Mnih et al. (2016) — "Asynchronous Methods for Deep Reinforcement Learning"** → [arXiv:1602.01783](https://arxiv.org/abs/1602.01783). El paper de A3C (y por extensión A2C, la versión síncrona que en la práctica es más usada hoy). La idea: paralelizar la recolección de experiencia entre múltiples workers para *descorrelacionar* las muestras sin necesidad de replay buffer.
- **Berkeley CS285 — Lecture 6 (Actor-Critic)** otra vez, pero ahora con foco en la parte de actor-critic.
- **CleanRL — `ppo.py` y los notebooks de A2C**. Aunque sea de PPO, el esqueleto es A2C.

### Temas mínimos

Arquitectura: actor y critic son **dos cabezas de la misma red** (típicamente, con CNN/MLP compartido) o dos redes separadas. **Loss combinada**: $L = L_{actor} + c_1 L_{critic} - c_2 H(\pi)$, donde $H(\pi)$ es entropía y $c_2$ es el coeficiente del entropy bonus. **Por qué los tres términos**: el actor mejora la política, el critic mejora la estimación de valor, la entropía mantiene exploración. **Síncrono vs. asíncrono**: A3C tiene workers asíncronos que actualizan parámetros globales con locks. A2C tiene workers síncronos que recolectan en paralelo y se sincronizan en el step de actualización — más simple, igual de bueno en la práctica (Wu et al. 2017 mostraron que la asincronía no aporta tanto). **Vectorized environments**: `gym.vector.SyncVectorEnv` y `AsyncVectorEnv`. **N-step rollouts y bootstrapping al final**: en lugar de episodios completos, recolectas N pasos y bootstrapeas con $V(s_N)$ al final.

### Ejercicio

Implementa **A2C** en CartPole-v1 y LunarLander-v2 con `gym.vector.SyncVectorEnv` y 8 environments paralelos. Compara contra tu REINFORCE+baseline+GAE del bloque anterior. Debería:
- Aprender más rápido en wall-clock time (gracias al paralelismo)
- Tener varianza menor entre seeds
- Funcionar en LunarLander, donde REINFORCE puro fallaba

### Checklist 2.6

- [ ] Implementé A2C con environments paralelos en PyTorch
- [ ] Sé explicar los tres términos de la loss y para qué sirve cada uno
- [ ] Mi A2C resuelve LunarLander-v2 en <2M timesteps
- [ ] Entiendo la diferencia entre A2C y A3C, y por qué la comunidad converge hacia A2C
- [ ] Soy capaz de dibujar un diagrama del flujo de datos en actor-critic vectorizado

---

## 2.7 — Reproducibilidad y disciplina experimental

### Por qué importa — el bloque que más rentabiliza durante el doctorado

Hay una **crisis de reproducibilidad en RL**, documentada por Henderson et al. en 2017 y peor desde entonces. Resultados publicados en venues top a veces no se reproducen con seeds distintas. Métricas mal usadas (mean sin error bars) producen conclusiones falsas. Implementaciones que difieren en detalles minúsculos producen curvas radicalmente distintas. **Esto no es un problema "del campo"; es un problema que vas a tener tú, en tu primera tabla**, si no aprendes la disciplina ahora.

Y para tu tesis específicamente: en Federated RL la varianza experimental es **aún mayor** (cada cliente añade ruido), y en Offline RL es **aún más sutil** (overestimation silenciosa). Las herramientas y la disciplina las construyes aquí, en Fase 2, no en Fase 6.

### Lectura obligatoria

- **Henderson et al. (2018) — "Deep Reinforcement Learning that Matters"** → [arXiv:1709.06560](https://arxiv.org/abs/1709.06560). **Léelo entero**, dos veces. Es el paper que cambió la cultura experimental del campo. Te enseñará a desconfiar de gráficas con 3 seeds y media sin desviación.
- **Agarwal et al. (2021) — "Deep RL at the Edge of the Statistical Precipice"** (NeurIPS Outstanding Paper) → [arXiv:2108.13264](https://arxiv.org/abs/2108.13264). El sucesor moderno: propone métricas robustas (Interquartile Mean, optimality gap, performance profiles) y la librería **`rliable`** que las computa. Esto **es** el estado del arte actual en reporting de resultados de RL. **Lectura obligatoria**.
- **Engstrom et al. (2020) — "Implementation Matters in Deep RL"** → [arXiv:2005.12729](https://arxiv.org/abs/2005.12729). Demuestra que **detalles de implementación** (clipping de observaciones, normalización de advantages, scheduling de learning rate) explican más diferencia de rendimiento que la elección entre PPO vs. TRPO.

### Herramientas

- **`rliable`** ([github.com/google-research/rliable](https://github.com/google-research/rliable)): la librería de Agarwal et al. para reporting estadísticamente robusto. **Úsala desde el primer experimento de Pong**.
- **Weights & Biases** ([wandb.ai](https://wandb.ai)): logging y tracking. Gratis para uso académico. Aprende sus `wandb.init`, `wandb.log`, y *sweeps* para hyperparameter search.
- **TensorBoard**: alternativa local, integrada en PyTorch. Más simple pero menos potente para colaboración.
- **Hydra** ([hydra.cc](https://hydra.cc)): config management con composición jerárquica. Esencial cuando los configs se multiplican.

### Disciplina mínima — y esto es **no negociable**

- **≥5 seeds por curva**. Idealmente 10 si el experimento no es caro. Nunca una sola seed para nada que vaya a un paper o a tu tesis.
- **Bandas de confianza visualizadas** (típicamente bootstrap CI 95% o IQR). No "media sola".
- **Métricas robustas**: usa IQM (Interquartile Mean) en lugar de mean cuando hay outliers, que en RL los hay siempre.
- **Hiperparámetros idénticos entre baselines y método propio**. Cualquier comparación injusta se ve a kilómetros.
- **Reportar varianza entre runs y entre seeds explícitamente** en tablas.
- **Almacenar las seeds usadas** en un fichero versionado del repo.
- **Test estadístico mínimo** en comparaciones (Welch's t-test o Mann-Whitney U). No es overkill: es lo que separa una afirmación de una creencia.

### Ejercicio

Coge tu DQN de Pong del bloque 2.2 y haz un **mini-paper interno** de 2 páginas:
- Tabla comparando vanilla DQN, Double DQN, Dueling DQN, los tres con ≥5 seeds
- Gráficas con IQM y bootstrap CI 95% usando `rliable`
- Performance profiles
- Test estadístico de si las diferencias son significativas

Este mini-paper interno es **tu primer artefacto pre-doctoral con estándares reales**. Guárdalo. Cuando empieces el doctorado, el primer paper que escribirás será una versión más larga de esto.

### Checklist 2.7

- [ ] Leí Henderson et al. 2018 y Agarwal et al. 2021 enteros
- [ ] Uso `rliable` para reportar métricas en todos mis experimentos
- [ ] Mis curvas siempre tienen ≥5 seeds y bandas de confianza
- [ ] Tengo un mini-paper interno de 2 páginas comparando variantes de DQN en Pong
- [ ] Sé qué es el IQM y por qué es preferible al mean para curvas de RL

---

## 2.8 — Ingeniería: vectorización, wrappers, logging

### Por qué importa

Esto es lo que **nadie te enseña en cursos** y todo el mundo descubre por dolor en su primer mes de doctorado. Saber cómo paralelizar 16 envs en CPU, cómo evitar el GIL en Python, cómo estructurar un replay buffer eficiente en memoria, qué wrappers aplicar en qué orden — todo esto es la diferencia entre un experimento que tarda 6h y uno que tarda 6 días para el mismo resultado. En la Fase 6 (Federated RL), esto se vuelve **vital**: cada experimento federado simula N clientes en paralelo y el coste compose.

### Temas mínimos

**Vectorized envs**: `gym.vector.SyncVectorEnv` vs. `AsyncVectorEnv`, cuándo cada uno. **Wrappers de Gymnasium**: `FrameStack`, `ResizeObservation`, `GrayscaleObservation`, `ClipReward`, `EpisodicLifeEnv`, `NoopResetEnv`. El **orden importa** (frame-stack debe ir DESPUÉS de resize, no antes). **Replay buffer eficiente**: numpy arrays preasignados en lugar de listas; almacenar `uint8` para imágenes Atari (no `float32`) reduce memoria 4×. **CUDA tensor management**: cuándo mover a GPU, batched `.to(device, non_blocking=True)`. **Mixed precision** con `torch.cuda.amp` para acelerar entrenamiento. **Perfilado**: `torch.profiler` y `cProfile` para encontrar bottlenecks. **Checkpointing**: guardar modelo cada N pasos, recuperar de crash. **Determinismo**: `torch.manual_seed`, `np.random.seed`, `env.reset(seed=...)`, `torch.backends.cudnn.deterministic = True` — y aceptar que **determinismo perfecto en GPU es prácticamente imposible** en deep RL.

### Recursos

- **CleanRL como referencia de estilo**: nota cómo estructuran cada algoritmo en un solo fichero, sin abstracciones innecesarias. Aprende ese estilo: vas a leerlo y replicarlo cientos de veces.
- **Stable-Baselines3** ([github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)) como referencia de calidad industrial. Más abstracto que CleanRL pero útil para ver "cómo se hace en producción".
- **RLlib** ([docs.ray.io/en/latest/rllib](https://docs.ray.io/en/latest/rllib/)) para escalado masivo (cientos de workers). Aún no lo necesitas, pero sabe que existe.

### Ejercicio

Refactoriza tu DQN de Pong con:
- 8 envs paralelos en lugar de 1
- Replay buffer con `uint8` en lugar de `float32`
- Mixed precision
- Logging completo a W&B (no solo curvas de reward: también loss del actor, loss del critic, magnitudes de gradientes, normas de pesos, fracción de exploración)
- Checkpoint cada 100k steps con resume automático

Mide el speedup wall-clock vs. tu DQN original. Deberías ver al menos 2–3× en el mismo hardware. Y lo más importante: **acostúmbrate a ver siempre los logs detallados**. Una loss del critic que explota y un advantage que diverge te avisan de un bug **antes** de que la curva de reward se rompa.

### Checklist 2.8

- [ ] Mi DQN usa 8 envs paralelos y replay buffer en uint8
- [ ] Logueo a W&B losses, gradientes, weights, y curvas de reward — todo
- [ ] Mi pipeline soporta checkpointing y resume tras crash
- [ ] Sé identificar el bottleneck de mi training (env step vs. GPU step) con `torch.profiler`
- [ ] He leído `cleanrl/dqn_atari.py` línea a línea y entiendo cada decisión de implementación

---

## 2.9 — Proyecto integrador: repositorio público de Deep RL

### Por qué importa

Es **tu segunda pieza de portfolio doctoral**, y la primera con código de Deep RL serio. Lo que tu director potencial verá si te aplica al laboratorio. Lo que un revisor de beca verá en tu CV. **La diferencia entre "he hecho Deep RL en algún curso" y "tengo un repo público con DQN, A2C, ablaciones, ≥5 seeds, métricas robustas y un mini-paper de 4 páginas" es lo que decide candidaturas doctorales.**

### Estructura mínima del repo

```
deep-rl-from-scratch/
├── README.md                    # narrativo, gráficas embebidas, links a W&B
├── envs/                        # wrappers custom si los hay
├── agents/
│   ├── dqn.py                   # vanilla DQN (CartPole + Atari)
│   ├── double_dqn.py
│   ├── dueling_dqn.py
│   ├── per_dqn.py
│   ├── reinforce.py
│   ├── a2c.py
│   └── shared/                  # replay buffer, networks, schedulers
├── experiments/                 # scripts que reproducen cada figura
├── configs/                     # YAMLs con hiperparámetros (Hydra)
├── figures/                     # generadas por rliable
├── reports/
│   └── mini_paper.pdf           # tu informe interno de 4 páginas
├── tests/                       # unit tests (replay buffer, GAE, etc.)
├── requirements.txt
└── notebooks/                   # exploraciones interactivas
```

### Buenas prácticas que NO son opcionales

- **Reproducibilidad bit-a-bit imposible** en GPU, pero **reproducibilidad estadística** (mismas conclusiones con mismas seeds) sí. Demuéstrala.
- **README en inglés**, con badges (CI, license), gráficas embebidas, y enlaces a runs de W&B públicos.
- **Mini-paper en `reports/`**: 4 páginas en formato NeurIPS/ICML, con abstract, intro, método, experimentos, conclusión. Sí, aunque sea "solo" una replicación de DQN. Te entrena en el formato y te da un primer producto.
- **Unit tests** específicos: comprobar que tu implementación de GAE coincide con un cálculo manual para un caso pequeño; comprobar que el replay buffer respeta orden y prioridades.
- **CI con GitHub Actions** ejecutando los tests y un smoke run de cada agente en CartPole.

### Checklist 2.9

- [ ] Repo público en GitHub, MIT/Apache 2.0
- [ ] README narrativo en inglés con gráficas y enlaces a W&B
- [ ] Implementaciones de DQN, Double, Dueling, PER-DQN, REINFORCE, A2C
- [ ] Mini-paper de 4 páginas en `reports/` con métricas robustas (`rliable`)
- [ ] CI básico verde
- [ ] Documentado el setup completo (CUDA, gymnasium versión, AutoROM, etc.)

---

## Pre-final: lo que se ve cuando se cierra la Fase 2

Cuando termines esto, deberías poder:
- Abrir un paper de Deep RL del 2018–2024 (DQN family, A2C family, hasta SAC anterior a tus lecturas de Fase 3) y **leerlo entero** sin tropezarte con notación o trucos básicos.
- Implementar desde cero, sin pseudocódigo, DQN, Double DQN, REINFORCE y A2C en una tarde por algoritmo.
- Defender en una pizarra qué es la Deadly Triad, por qué cada truco de DQN la mitiga, y qué pasaría si quitaras cada uno.
- Reportar experimentos con la disciplina que un revisor de NeurIPS aceptaría: ≥5 seeds, IQM con bandas de confianza, tests estadísticos básicos.
- Mirar PPO (que entra en Fase 3) y reconocer que es A2C + clipping de ratio + GAE. No te parecerá un algoritmo nuevo; te parecerá un actor-critic con dos trucos. **Ese es el sello de que Fase 2 está cerrada.**

---

## Plan sugerido de 10 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 2.1 Deadly Triad + counterexample de Baird | Lectura S&B cap. 9–11 |
| 2 | 2.2 DQN: paper + CartPole funcionando | Setup gymnasium + ALE |
| 3–4 | 2.2 DQN en Atari Pong | Disciplina experimental incipiente |
| 5 | 2.3 Double + Dueling + PER | Ablación primera con 3 seeds |
| 6 | 2.4 REINFORCE + 2.5 GAE (parte 1) | Lectura S&B cap. 13 |
| 7 | 2.5 GAE (parte 2) + 2.6 A2C | A2C en LunarLander |
| 8 | 2.7 Reproducibilidad: Henderson + Agarwal | Migración a `rliable` |
| 9 | 2.8 Ingeniería: vectorización + profiling | Refactor DQN con 8 envs |
| 10 | 2.9 Mini-paper + repo pulido | Submit a mentor/director |

Si Pong te resiste más de 3 semanas, **no es tu implementación, son tus hiperparámetros**. Compara contra CleanRL seed-por-seed y copia los suyos exactamente. Es la única forma de aislar si el problema es algorítmico o de configuración.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:
- Derivaciones desarrolladas (especialmente el Policy Gradient Theorem y la fórmula de GAE)
- Código PyTorch comentado línea a línea de cada algoritmo
- Análisis de los papers críticos (Mnih, Schulman GAE, Henderson, Agarwal) con notas propias
- Debugging guide específico de Deep RL: los errores típicos (target no detachado, advantage no normalizado, broadcasting silencioso) y cómo identificarlos

Mi recomendación de orden:
1. **2.1 + 2.2 en un solo bloque dedicado** (Deadly Triad y DQN), porque son la unidad conceptual de la rama value-based.
2. **2.4 + 2.5 en otro bloque dedicado** (REINFORCE y GAE), por la misma razón en la rama policy-based.
3. **2.7 + 2.8 en un bloque "investigador serio"**, porque la disciplina experimental e ingenieril son lo que más te va a rentabilizar de toda la fase durante el doctorado.

Y un aviso final: **no llegues a Fase 3 con dudas sobre la Deadly Triad o sobre por qué advantages reducen varianza**. PPO y SAC son fáciles si esos dos conceptos te resultan naturales; son infranqueables si no.
