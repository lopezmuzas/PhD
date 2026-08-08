---
title: "Fase 1 — RL Tabular y Fundamentos Teóricos"
tags: [rl, refuerzo]
status: borrador
updated: 2026-08-08
---

# Fase 1 — RL Tabular y Fundamentos Teóricos

> **Objetivo de la fase**: que las ecuaciones de Bellman y los algoritmos de Q-Learning / SARSA dejen de ser fórmulas memorizadas y se conviertan en parte de tu intuición. Saber **demostrar** (no solo enunciar) por qué value iteration y TD(0) convergen en el caso tabular. Salir con un repositorio público propio — tu primer pieza de portfolio doctoral — con todos los algoritmos clásicos implementados desde cero.
>
> **Tu situación de partida**: vienes de la Fase 0 con Sutton & Barto capítulos 1–3 ya digeridos, MDPs claros, esperanzas y cadenas de Markov dominadas, NumPy y PyTorch a punto. Aquí PyTorch aún no entra; **todo se hace con NumPy puro**.
>
> **Tiempo estimado realista**: 6–8 semanas a ritmo de 1–2 h/día. Es una fase corta en contenido nuevo pero **densa en implementación y en demostraciones**. La prisa aquí se paga en la Fase 2 con bugs silenciosos en DQN que no sabrás depurar.

---

## Cómo usar esta guía

Misma lógica que la guía de Fase 0: índice maestro organizado por bloques, cada uno con:

- **Por qué importa** → la conexión directa con el resto del roadmap (Deep RL, Offline RL, Federated RL)
- **Recursos principales** → curso central por el que te guías
- **Refuerzo en vídeo** → tu formato preferido
- **Lectura en inglés** → para integrar el idioma técnico y los papers
- **Ejercicios mínimos** → implementación obligatoria, no opcional
- **Checklist** → criterios concretos para saber que ese bloque está cerrado

Cuando termines un bloque, vuelves aquí, lo marcas, y me pides el siguiente con ejemplos, demostraciones desarrolladas y código comentado.

---

## Mapa de la Fase 1

| Bloque | Tema | Peso |
|---|---|---|
| 1.1 | MDPs finitos y ecuaciones de Bellman | 🔥🔥🔥🔥🔥 La base de todo |
| 1.2 | Programación Dinámica (Policy/Value Iteration) | 🔥🔥🔥🔥 Crítico — y demostrable |
| 1.3 | Métodos Monte Carlo | 🔥🔥🔥 Puente entre DP y TD |
| 1.4 | TD Learning: SARSA y Q-Learning | 🔥🔥🔥🔥🔥 El corazón de la fase |
| 1.5 | n-step methods, TD(λ) y eligibility traces | 🔥🔥🔥 No saltar |
| 1.6 | Planning y Dyna-Q | 🔥🔥🔥 Ancestro del model-based moderno |
| 1.7 | Exploración: ε-greedy, UCB, optimismo | 🔥🔥🔥 Más profundo de lo que parece |
| 1.8 | Teoría: convergencia y sample complexity | 🔥🔥🔥🔥 El sello doctoral |
| 1.9 | Proyecto integrador: repositorio público | 🔥🔥🔥🔥🔥 Tu primera pieza de portfolio |

---

## 1.1 — MDPs finitos y ecuaciones de Bellman

### Por qué importa

Toda la teoría de RL — tabular, profundo, offline, federado — se construye sobre el formalismo del MDP y las ecuaciones de Bellman. Cuando en la Fase 6 leas "Federated Q-Learning con heterogeneidad de MDPs", esa frase está saturada de implicaciones que solo tienen sentido si las ecuaciones de Bellman te resultan tan naturales como una derivada. Esto no es repaso de la Fase 0: es **subir de marcha** desde "los he visto" a "los uso para razonar".

### Recurso principal

- **Sutton & Barto, capítulos 3 y 4** → [PDF gratis legal](http://incompleteideas.net/book/RLbook2020.pdf). Si en Fase 0 los hojeaste, **ahora los lees en serio, haciendo todos los ejercicios al final del capítulo**. Especialmente los del 3.

### Vídeo

- **David Silver — UCL RL Course, Lecture 2 (MDPs)** → [YouTube](https://www.youtube.com/watch?v=lfHX2hHRMVQ). Silver explica MDPs con una claridad excepcional. Es **innegociable** verla antes de pasar al bloque 1.2.
- **Mutual Information — serie RL** → [YouTube](https://www.youtube.com/@Mutual_Information). Los primeros 3–4 vídeos sobre MDPs y Bellman son refuerzo visual al estilo 3Blue1Brown.

### Lectura adicional

- **Puterman — "Markov Decision Processes: Discrete Stochastic Dynamic Programming"**. El libro de referencia matemática. No lo leas entero; tenlo como diccionario para cuando algo en Sutton & Barto te quede corto en formalismo. Especialmente útil más adelante para Fase 4 y 6.

### Temas mínimos que tienes que dominar

La tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ y qué representa cada elemento. Política $\pi$: determinista vs. estocástica, y cómo se escribe formalmente $\pi(a|s)$. Retornos $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ y por qué el descuento $\gamma < 1$ es **matemáticamente** necesario (no solo una conveniencia). Funciones de valor $V^\pi(s)$ y $Q^\pi(s,a)$, y la relación $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$. Ecuación de Bellman de **expectativa** (para evaluar una política dada) y de **optimalidad** (para encontrar la mejor). Política óptima $\pi^*$ y por qué siempre existe una determinista óptima en MDPs finitos. Operador de Bellman como **operador en el espacio de funciones de valor** — esto te lo verás más adelante en la demostración de contracción de Banach.

### Checklist 1.1

- [ ] Escribo la ecuación de Bellman para $V^\pi$ y $Q^\pi$ sin mirar, y sé pasar de una a otra
- [ ] Sé explicar por qué $V^* (s) = \max_a Q^*(s,a)$ y por qué eso implica una política greedy óptima
- [ ] Resuelvo a mano el ejemplo del Student MDP de Silver (calcular $V^\pi$ para una política dada)
- [ ] Entiendo qué pasaría con $\gamma = 1$ en un MDP de horizonte infinito y por qué se rompe

---

## 1.2 — Programación Dinámica: Policy y Value Iteration

### Por qué importa

DP es el caso "ideal": asume modelo conocido ($P$ y $R$ dados), planificación pura. **Pero es el único algoritmo del que vas a poder demostrar convergencia limpiamente** mediante el teorema del punto fijo de Banach. Esa demostración es la que después aplicas (con muletas) a Q-Learning, y es el tipo de razonamiento que tu director esperará de ti en un comité doctoral. Aquí construyes la base teórica de toda la fase.

### Recurso principal

- **Sutton & Barto, capítulo 4 completo**. Incluye policy evaluation, policy improvement, policy iteration, value iteration, y la discusión sobre asincronía y generalized policy iteration (GPI).
- **David Silver — Lecture 3 (Planning by Dynamic Programming)** → [YouTube](https://www.youtube.com/watch?v=Nd1-UUMVfz4). La acompañas a Sutton & Barto.

### Lectura teórica

- **Agarwal, Jiang, Kakade & Sun — "Reinforcement Learning: Theory and Algorithms"** → [PDF gratis](https://rltheorybook.github.io/). **Capítulo 1 entero**. Aquí ves la demostración formal de que el operador de Bellman es una $\gamma$-contracción en norma infinito, y por tanto value iteration converge geométricamente al punto fijo $V^*$. **Este es el primer paper-book de tu vida doctoral; léelo aunque te cueste**, con papel y lápiz al lado.
- **Bertsekas — "Dynamic Programming and Optimal Control" Vol. I**. Tenlo como referencia. Bertsekas es el otro pilar (junto a Puterman) del formalismo DP.

### Temas mínimos

Policy evaluation iterativa y su criterio de parada. Policy improvement theorem y por qué garantiza mejora estricta o convergencia. Policy iteration: alternar evaluación e improvement hasta punto fijo. Value iteration como caso particular donde se trunca la evaluación a un solo paso. **Demostración de contracción del operador de Bellman** $\mathcal{T}^\pi$ y del operador óptimo $\mathcal{T}^*$ en norma sup, con factor $\gamma$. Convergencia geométrica: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$. Versión asíncrona y por qué sigue convergiendo si todos los estados se actualizan infinitamente a menudo. Generalized Policy Iteration (GPI) — el patrón que verás luego en todo Actor-Critic.

### Ejercicio obligatorio

Implementa **value iteration y policy iteration desde cero en NumPy** para un GridWorld 4×4 hecho a mano (con paredes, recompensas terminales y un par de estados peligrosos). Grafica:
- $\|V_k - V_{k-1}\|_\infty$ vs. iteración (tiene que ser geométrica)
- La política óptima como flechas sobre el grid
- Sensibilidad de la convergencia a $\gamma \in \{0.5, 0.9, 0.99\}$

### Checklist 1.2

- [ ] Implementé value iteration y policy iteration en NumPy y obtuve la misma política óptima con ambos
- [ ] Sé escribir la demostración de contracción del operador de Bellman, paso a paso, en una pizarra
- [ ] Entiendo por qué la convergencia es geométrica y por qué el ritmo depende de $\gamma$
- [ ] He leído entero el capítulo 1 de Agarwal-Jiang-Kakade-Sun (aunque haya tenido que releer secciones)

---

## 1.3 — Métodos Monte Carlo

### Por qué importa

DP asume modelo conocido. MC es el primer salto a **aprender sin modelo**, usando solo experiencia muestreada. El precio: alta varianza, episodios completos. La ganancia: aplicabilidad real. Más importante para ti: **MC es el "extremo" del eje bias-variance** del que TD es el otro extremo. n-step y TD(λ) interpolan entre ambos. Sin entender bien MC, no entiendes ni GAE (Fase 2) ni la varianza en gradientes de política, que es **el** problema central de policy gradients.

### Recurso principal

- **Sutton & Barto, capítulo 5 completo**. Cubre MC prediction, MC control con exploring starts, y MC off-policy con importance sampling.
- **David Silver — Lecture 4 (Model-Free Prediction)** → [YouTube](https://www.youtube.com/watch?v=PnHCvfgC_ZA). Primera mitad cubre MC.

### Lectura adicional

- **Sutton & Barto, sección 5.5–5.7** sobre **importance sampling**. Léelo dos veces. El concepto de IS reaparece en Fase 3 (PPO usa un *ratio* de importance sampling) y en Fase 4 (Offline RL es **literalmente** un problema de IS sobre la política de comportamiento).

### Temas mínimos

First-visit MC y every-visit MC para prediction. MC control on-policy con ε-soft policies. Importance sampling ordinario vs. ponderado, y por qué el ponderado tiene menos varianza pero está sesgado. El problema del importance ratio: si las políticas difieren mucho, la varianza explota — **esto es exactamente lo que Offline RL intenta resolver**. Por qué MC tiene cero bias pero alta varianza, y por qué eso lo hace impráctico en horizontes largos.

### Ejercicio obligatorio

Implementa en NumPy:
- Every-visit MC prediction sobre tu GridWorld (estimar $V^\pi$ para una política aleatoria)
- MC control con exploring starts en Blackjack (el ejemplo canónico de Sutton & Barto, sección 5.3)
- MC off-policy con weighted importance sampling sobre el mismo Blackjack

Compara la varianza de las estimaciones MC vs. el valor exacto obtenido por DP en el GridWorld. Es una de esas gráficas que te enseñan más que diez papers.

### Checklist 1.3

- [ ] Implementé MC prediction y MC control en NumPy, sin pseudocódigo a la vista
- [ ] Sé explicar por qué MC es no sesgado pero de alta varianza
- [ ] Entiendo el importance sampling ratio y sé predecir cuándo va a explotar la varianza
- [ ] He resuelto el ejemplo de Blackjack y mi política se parece a la "tabla óptima" que aparece en S&B

---

## 1.4 — TD Learning: SARSA y Q-Learning

### Por qué importa

**Este es el bloque central de la fase**. TD(0) es la idea más importante de RL clásico según el propio Sutton: actualizar estimaciones **a partir de otras estimaciones** (bootstrapping), sin esperar al final del episodio. Q-Learning es el algoritmo más famoso de la historia de RL. Su salto a deep (DQN) define toda la Fase 2. Y la **distinción on-policy / off-policy** que aparece aquí es exactamente la que define qué algoritmos puedes usar en Offline RL (Fase 4) y cómo se federan en Fase 6.

Si solo dominas un bloque de esta fase a fondo, que sea este.

### Recurso principal

- **Sutton & Barto, capítulo 6 entero**. Es probablemente el capítulo más importante del libro.
- **David Silver — Lectures 4 y 5 (Model-Free Prediction & Control)** → [YouTube](https://www.youtube.com/watch?v=PnHCvfgC_ZA) y [YouTube](https://www.youtube.com/watch?v=0g4j2k_Ggc4). La 5 es donde explica Q-Learning vs. SARSA con el ejemplo del Cliff Walking, que tienes que reproducir tú mismo.

### Refuerzo

- **Mutual Information — vídeos sobre TD Learning y Q-Learning**. Visualizaciones de cómo se propaga el valor por el grid; te ayudan a "ver" la actualización de Bellman.
- **Stanford CS234 — Lectures 3 y 4 (Brunskill)** → [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX). Más énfasis teórico que Silver. Recomendado si te orientas a tesis teórica.

### Lectura

- **Watkins & Dayan (1992) — "Q-Learning"**. El paper original, 4 páginas. [PDF](https://link.springer.com/article/10.1007/BF00992698). Léelo: es corto, histórico, y contiene la demostración de convergencia que aprenderás en el bloque 1.8.
- **Rummery & Niranjan (1994) — "On-line Q-learning using connectionist systems"**. El paper donde aparece SARSA por primera vez.

### Temas mínimos

TD(0) prediction: la actualización $V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$ y por qué el término entre corchetes se llama TD error. Bias-variance: TD tiene bias (porque bootstrapea de estimaciones imperfectas) pero menos varianza que MC. SARSA: actualización con $(s, a, r, s', a')$ — necesita la acción siguiente **real** según la política actual, por eso es **on-policy**. Q-Learning: actualización con $\max_{a'} Q(s', a')$ — usa la **mejor** acción siguiente, no la real, por eso es **off-policy**. Expected SARSA y por qué reduce varianza respecto a SARSA tradicional. Por qué en el Cliff Walking, Q-Learning aprende la política óptima pero SARSA aprende una política más conservadora — y por qué eso es **un feature, no un bug** según el contexto (seguridad, exploración).

### Ejercicio obligatorio — el central de la fase

Implementa en NumPy puro, **sin mirar pseudocódigo después de la primera lectura**:
1. SARSA tabular
2. Q-Learning tabular
3. Expected SARSA tabular

Sobre dos entornos:
- **Tu GridWorld** custom
- **FrozenLake-v1** de Gymnasium (slippery=True y slippery=False)

Para cada combinación entrega:
- Curva de retorno medio vs. episodios (≥5 seeds, banda de confianza, no solo media)
- Política aprendida visualizada sobre el grid
- Sensibilidad a $\alpha \in \{0.01, 0.1, 0.5\}$, $\gamma \in \{0.9, 0.99\}$, $\epsilon$ con decay
- **Reproducción del experimento del Cliff Walking** comparando Q-Learning y SARSA — el gráfico canónico de Sutton & Barto figura 6.5

Esto es **tu primer "experimento de RL como Dios manda"**: bandas de confianza, seeds múltiples, sensibilidad a hiperparámetros. Acostúmbrate desde el día uno, porque la crisis de reproducibilidad en RL (Henderson et al., 2017) la lees en Fase 2 pero la previenes desde aquí.

### Checklist 1.4

- [ ] Implementé SARSA, Q-Learning y Expected SARSA en NumPy desde cero, sin mirar pseudocódigo
- [ ] Sé explicar por qué Q-Learning es off-policy y SARSA es on-policy, y predigo qué política aprenderá cada uno en el Cliff Walking
- [ ] Mis gráficas llevan ≥5 seeds y bandas de confianza, no medias sueltas
- [ ] Q-Learning resuelve FrozenLake-v1 slippery=False con política óptima; en slippery=True alcanza ≥70% de éxito
- [ ] Leí el paper original de Watkins & Dayan (1992)

---

## 1.5 — n-step methods, TD(λ) y eligibility traces

### Por qué importa

n-step interpola entre TD(0) y MC; TD(λ) interpola **suavemente** entre todos los n-step. Es la base de **GAE (Generalized Advantage Estimation)**, que vas a implementar en Fase 2 para PPO. Si saltas este bloque, GAE te parecerá magia con un parámetro $\lambda$ misterioso. Si lo dominas, GAE es trivial.

### Recurso principal

- **Sutton & Barto, capítulo 7 entero** (n-step bootstrapping) y **capítulo 12** (Eligibility Traces). El 12 es denso; la primera vez basta con leer hasta TD(λ) forward view y backward view, y entender la equivalencia.
- **David Silver — Lecture 4 (segunda mitad)** sobre TD(λ).

### Temas mínimos

n-step TD prediction: $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$. n-step SARSA. Cómo escoger n en la práctica (típicamente entre 3 y 10). λ-return: combinación geométricamente ponderada de todos los n-step returns. Forward view de TD(λ) (teórica) vs. backward view (algorítmica, con eligibility traces). Eligibility traces como "memoria corta de qué estados se visitaron". SARSA(λ) y Q(λ), y por qué Q(λ) tiene un problema sutil cuando se combina con off-policy (lo verás otra vez en deep RL).

### Ejercicio

Extiende tu Q-Learning a **n-step Q-Learning** y compara curvas de aprendizaje con $n \in \{1, 3, 5, 10, \infty\}$ ($\infty$ = MC). Reproduce la figura 7.2 de Sutton & Barto sobre Random Walk. Verás visualmente la curva en U típica del trade-off bias-variance.

### Checklist 1.5

- [ ] Implementé n-step SARSA y veo cómo cambia la velocidad de aprendizaje con n
- [ ] Sé explicar la equivalencia entre forward y backward view de TD(λ)
- [ ] Entiendo que TD(λ=0) es TD(0) y TD(λ=1) es MC
- [ ] Anticipo que esto reaparecerá en Fase 2 como GAE y sé por qué

---

## 1.6 — Planning y Dyna-Q

### Por qué importa

Dyna-Q es el ancestro directo de todos los métodos **model-based** modernos: World Models, MuZero, Dreamer. La idea — *aprender un modelo del entorno con la experiencia real, y usarlo para generar experiencia simulada extra* — es la misma de fondo. Además, **es fundamental para tu tesis**: en Offline RL (Fase 4) y en Federated RL con comunicación cara (Fase 6), generar experiencia sintética a partir de un modelo aprendido es una estrategia central. El capítulo 8 de S&B es **innegociable** aunque tu propuesta original lo saltaba.

### Recurso principal

- **Sutton & Barto, capítulo 8 entero**. Especialmente las secciones 8.1–8.4 (Dyna-Q, Dyna-Q+) y la 8.6 (Trajectory Sampling).
- **David Silver — Lecture 8 (Integrating Learning and Planning)** → [YouTube](https://www.youtube.com/watch?v=ItMutbeOHtc).

### Temas mínimos

Modelo del entorno: $\hat{P}(s'|s,a)$ y $\hat{R}(s,a)$ aprendidos a partir de transiciones observadas. Dyna-Q: alterna learning real con planning sobre el modelo. Dyna-Q+: bonus de exploración por novedad temporal (precursor de los métodos de curiosity-driven exploration modernos). Prioritized sweeping: priorizar las actualizaciones por magnitud del TD error — la misma idea que **Prioritized Experience Replay** (PER) en DQN. Real-time DP y trajectory sampling. **Decision-time planning** vs. **background planning** — la distinción que separa MCTS de Dyna.

### Ejercicio

Implementa Dyna-Q en tu GridWorld y compara curvas de aprendizaje contra Q-Learning puro, con $n \in \{0, 5, 50\}$ pasos de planning por step real. Reproduce la figura 8.2 de Sutton & Barto. Verás un speedup brutal: ese es el "valor del modelo".

Bonus: implementa Dyna-Q+ y prueba con un GridWorld donde a mitad de entrenamiento **mueves una pared** (cambias el entorno). Q-Learning puro tarda mucho en adaptarse; Dyna-Q+ se recupera rápido. Es uno de los experimentos más reveladores de toda la fase.

### Checklist 1.6

- [ ] Implementé Dyna-Q en NumPy y veo el speedup vs. Q-Learning puro
- [ ] Implementé el experimento del entorno cambiante con Dyna-Q+
- [ ] Entiendo la conexión conceptual entre prioritized sweeping y PER en DQN
- [ ] Sé argumentar por qué los métodos model-based son atractivos en settings de datos caros (Offline, Federado)

---

## 1.7 — Exploración: ε-greedy, UCB, optimismo en la inicialización

### Por qué importa

La exploración es **el problema fundamental no resuelto de RL**. ε-greedy es lo que vas a usar el 90% del tiempo, pero es **trivial y subóptimo**. UCB y optimismo en la inicialización son los primeros métodos "principled" — y son los antecesores conceptuales de los métodos de exploración profunda (RND, ICM, BYOL-Explore) que aparecerán en Fase 3+. Para una tesis seria, exploración no puede ser un misterio.

### Recurso principal

- **Sutton & Barto, capítulo 2 entero** (multi-armed bandits) — sí, el cap. 2, que probablemente solo hojeaste en Fase 0. **Ahora léelo a fondo**. Es ahí donde aparecen UCB, gradient bandits, y el análisis de regret.
- **David Silver — Lecture 9 (Exploration and Exploitation)** → [YouTube](https://www.youtube.com/watch?v=sGuiWX07sKw).

### Lectura adicional (anticipo)

- **Lattimore & Szepesvári — "Bandit Algorithms"** → [PDF gratis](https://tor-lattimore.com/downloads/book/book.pdf). Es el libro canónico de bandits. No lo leas entero ahora; lee los capítulos 1–3 y 7–8. **Te va a servir toda la vida**, especialmente si tu tesis se mueve hacia la rama teórica.

### Temas mínimos

Trade-off exploración / explotación. ε-greedy con $\epsilon$ constante vs. decaying schedules. Por qué $\epsilon$ constante implica un regret lineal (sub-óptimo) y por qué un decay adecuado puede dar regret logarítmico. UCB1: $a_t = \arg\max_a \left[ \hat{Q}(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]$ — entiende cada término. Optimismo en la inicialización: por qué inicializar $Q_0$ con valores altos fuerza exploración exhaustiva durante un tiempo. Thompson Sampling como alternativa bayesiana (lectura introductoria; no implementarlo todavía). Por qué en MDPs con horizonte largo, ε-greedy **falla catastróficamente** en entornos sparse-reward, y por qué eso motiva métodos como ICM/RND que verás más adelante.

### Ejercicio

Implementa un **10-armed bandit** y compara empíricamente:
- ε-greedy con $\epsilon \in \{0, 0.01, 0.1\}$
- ε-greedy con decay $\epsilon_t = 1/\sqrt{t}$
- UCB1 con $c \in \{1, 2\}$
- Optimistic initialization con $Q_0 = 5$

Reproduce las figuras 2.2 y 2.4 de Sutton & Barto. Y en tu GridWorld original, añade UCB como exploración para Q-Learning y compara contra ε-greedy.

### Checklist 1.7

- [ ] Implementé ε-greedy, UCB y optimistic init en el bandit y obtengo las curvas de S&B
- [ ] Sé explicar el término de exploración de UCB1 y de dónde sale (Hoeffding)
- [ ] Entiendo por qué ε-greedy es suficiente en entornos densos pero falla en sparse
- [ ] Hojeé los primeros capítulos de Lattimore & Szepesvári

---

## 1.8 — Teoría: convergencia y sample complexity

### Por qué importa — y este es el bloque que te separa de un practicante

Cualquiera puede implementar Q-Learning siguiendo pseudocódigo. **Pocas personas pueden demostrar por qué converge**, qué hipótesis hacen falta, y cuál es su complejidad muestral. Tu director te va a evaluar precisamente en esa zona. Este bloque es **donde se nota si estás haciendo un doctorado o un curso online intensivo**.

### Recurso principal

- **Agarwal, Jiang, Kakade & Sun — "RL: Theory and Algorithms"** → [PDF gratis](https://rltheorybook.github.io/). **Capítulos 2 y 3** completos. El 2 cubre PAC-MDP y sample complexity. El 3 cubre análisis de Q-Learning tabular.
- **Bertsekas — "Reinforcement Learning and Optimal Control"**. Capítulos sobre stochastic approximation. Es donde formalmente vive Q-Learning como instancia de Robbins-Monro.

### Vídeo (más opcional aquí)

- **Stanford CS234 — Lectures sobre PAC-MDP y exploration theory (Brunskill)** → en la [playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX). Brunskill es referente en RL teórico y dedica clases enteras a sample complexity.

### Lectura de papers (selecta, no exhaustiva)

- **Jaakkola, Jordan & Singh (1994) — "On the Convergence of Stochastic Iterative Dynamic Programming Algorithms"**. La demostración canónica de convergencia de Q-Learning. Denso pero formativo.
- **Even-Dar & Mansour (2003) — "Learning rates for Q-Learning"**. Análisis fino de qué schedules de $\alpha$ dan qué tasas de convergencia.
- **Kakade (2003) — "On the sample complexity of reinforcement learning"** (tesis doctoral). La piedra fundacional de PAC-MDP. **No la leas entera ahora**; léete la intro y el capítulo 1.

### Temas mínimos

Stochastic approximation y condiciones de Robbins-Monro sobre $\alpha_t$: $\sum \alpha_t = \infty$ y $\sum \alpha_t^2 < \infty$. Por qué esas dos condiciones son **exactamente** lo que pide la demostración de convergencia de Q-Learning. Operador de Bellman como $\gamma$-contracción en norma sup. Convergencia casi segura de Q-Learning bajo las hipótesis: visitas infinitas a todos los pares $(s,a)$, schedule adecuado de $\alpha$, recompensas acotadas. Sample complexity: ¿cuántas muestras necesitas para garantizar con probabilidad $\geq 1-\delta$ que tu política está a $\epsilon$ de la óptima? Definición de PAC-MDP. Diferencia entre **regret** y **sample complexity** y por qué a veces se confunden.

### Ejercicio

Esto no es código, es **demostración**: escribe en LaTeX (o a mano, escaneado) la demostración paso a paso de la $\gamma$-contracción del operador de Bellman óptimo, y la convergencia de value iteration a partir de ahí. Es **3–4 páginas bien hechas**. Guárdalo: es tu primer documento técnico-formal de RL, y vas a referenciarlo durante todo el doctorado.

Bonus: lee la demostración de convergencia de Q-Learning de Jaakkola-Jordan-Singh y haz un resumen propio de **una página**. Es el tipo de ejercicio que tu director te va a pedir en el primer mes.

### Checklist 1.8

- [ ] Escribí la demostración de contracción del operador de Bellman en 3–4 páginas
- [ ] Sé enunciar las condiciones de Robbins-Monro y explicar por qué cada una es necesaria
- [ ] Leí el capítulo 2 de Agarwal-Jiang-Kakade-Sun y entendí la noción de sample complexity PAC-MDP
- [ ] Resumí en una página el paper de Jaakkola-Jordan-Singh

---

## 1.9 — Proyecto integrador: repositorio público

### Por qué importa

Es **tu primera pieza de portfolio doctoral**. Si tu director potencial quiere ver cómo trabajas, le mandas este repo. Si aplicas a un programa de doctorado o a una beca, este repo aparece en tu CV. La diferencia entre "he hecho los ejercicios de S&B" y "tengo un repo limpio, documentado, con experimentos reproducibles" es **enorme** en términos de credibilidad.

### Estructura mínima del repo

```
rl-tabular-from-scratch/
├── README.md          # explicación clara, gráficas embebidas, badges
├── envs/              # GridWorld custom + wrappers de FrozenLake
├── agents/            # un fichero por algoritmo (vi, pi, mc, sarsa, qlearning, expected_sarsa, n_step, dyna_q)
├── experiments/       # scripts que generan cada figura del README
├── figures/           # outputs versionados (PNG/PDF)
├── notebooks/         # exploraciones interactivas
├── tests/             # unit tests de las actualizaciones de Bellman
├── requirements.txt
└── theory/            # tus demostraciones del bloque 1.8 en PDF/LaTeX
```

### Buenas prácticas que NO son opcionales

- **Seeds explícitas y reproducibles** en todos los experimentos. ≥5 seeds por curva, bandas de confianza visualizadas.
- **Hyperparámetros via CLI o YAML**, no hardcoded.
- **Logging con `wandb` o `tensorboard`** — empieza ya, lo usarás toda tu carrera.
- **README narrativo**: explica qué hace cada algoritmo, muestra gráficas, enlaza a las secciones de S&B. Escríbelo en inglés. Es **tu producto visible**.
- **Tests unitarios mínimos**: que la actualización de Bellman computada por tu código coincida con la calculada a mano para un caso pequeño.
- **Licencia abierta** (MIT o Apache 2.0).

### Checklist 1.9

- [ ] Repo público en GitHub con la estructura descrita
- [ ] README escrito en inglés con explicación clara y gráficas embebidas
- [ ] Todos los algoritmos del bloque 1.2–1.6 implementados, testeados y comparados
- [ ] Carpeta `theory/` con tu demostración del bloque 1.8 en PDF
- [ ] CI básico (GitHub Actions) que corre los tests al hacer push

---

## Pre-final: lo que se ve cuando se cierra la Fase 1

Cuando termines esto, deberías poder:
- Sentarte delante de un MDP arbitrario y, sin pseudocódigo, escribir Q-Learning y SARSA en NumPy de memoria en 30 minutos.
- Demostrar en una pizarra por qué value iteration converge, mencionando explícitamente la contracción y el factor $\gamma$.
- Defender por qué Q-Learning es off-policy frente a alguien que afirme lo contrario, con un ejemplo concreto (Cliff Walking).
- Leer el paper de Mnih et al. 2015 (DQN) y reconocer que la ecuación 2 es **exactamente** la actualización de Q-Learning de Sutton & Barto, ahora con $Q$ parametrizado por una red neuronal y dos cabezas (online y target). Ese reconocimiento es **la prueba de que la Fase 1 está cerrada**: la Fase 2 es Fase 1 + función aproximada + trucos para que no explote.

---

## Plan sugerido de 8 semanas

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 1.1 MDPs + Bellman a fondo | Lectura Silver Lecture 2 |
| 2 | 1.2 DP: VI y PI + demostración de contracción | Implementación en GridWorld |
| 3 | 1.3 Monte Carlo + importance sampling | Blackjack experiment |
| 4 | 1.4 SARSA + Q-Learning (parte 1) | Lectura paper Watkins & Dayan |
| 5 | 1.4 (parte 2): FrozenLake + Cliff Walking | Empezar 1.5 (n-step) |
| 6 | 1.5 n-step / TD(λ) + 1.6 Dyna-Q | Reproducir figura 8.2 |
| 7 | 1.7 Exploración + 1.8 Teoría (parte 1) | Lectura Agarwal cap. 2 |
| 8 | 1.8 Teoría (parte 2) + 1.9 Repo limpio | Escribir demostración formal |

Si encuentras que un bloque te cuesta más — sobre todo el 1.4 y el 1.8 — **estíralo sin culpa**. La densidad real de la fase está en esos dos.

---

## Cómo seguimos

Cuando estés listo, dime qué bloque quieres profundizar primero y abrimos un documento dedicado con:
- Demostraciones desarrolladas paso a paso (especialmente bloque 1.2 y 1.8)
- Ejercicios concretos resueltos con código comentado
- Código NumPy de cada algoritmo, comentado línea a línea
- Análisis de las figuras canónicas de Sutton & Barto reproducidas en tus propios datos

Mi recomendación: empieza por **1.1 + 1.2 en un solo bloque dedicado**, porque la teoría de Bellman y la demostración de DP son una unidad conceptual. Después, el monstruo del bloque **1.4 (SARSA / Q-Learning)** merece su propio documento extenso con código. Y reserva un documento entero para el bloque **1.8 (teoría de convergencia)**: ese es el que más rentabiliza durante el resto del doctorado.
