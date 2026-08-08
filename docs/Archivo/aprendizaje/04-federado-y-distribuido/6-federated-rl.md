---
title: "Fase 6 — Federated Reinforcement Learning (el terreno de tu tesis)"
tags: [federado, distribuido, data-spaces]
status: borrador
updated: 2026-08-08
---

# Fase 6 — Federated Reinforcement Learning (el terreno de tu tesis)

> **Objetivo de la fase**: cerrar el círculo. Combinar **Offline RL** (Fase 4) con **Federated Learning clásico** (Fase 5) sobre **dataspaces Pontus-X / Compute-to-Data** para entrar en el campo donde vives los próximos 3–4 años. Dominar la **estructura del campo de FRL** en sus tres ejes ortogonales (heterogeneidad de entornos, qué se federa, online vs. offline), saber leer y reproducir los papers fundacionales (QAvg/PAvg de Jin et al. 2022, FedFormer, Federated Ensemble-Directed Offline RL de NeurIPS 2024), y **producir tu primer artefacto doctoral original**: un benchmark de Offline FRL sobre D4RL particionado, un primer paper de workshop, o ambos. Salir con una vía de tesis **elegida, defendible y ya empezada experimentalmente**.
>
> **Tu situación de partida**: vienes de Fase 4 con CQL/IQL replicados, un mini-paper de Offline RL y un `notes_for_supervisor.md` que ya cruza Offline con Federated. Vienes de Fase 5 con FedAvg/FedProx/FedOpt funcionando en Flower, DP-SGD integrado con Opacus, y un *Hello World* corriendo en Pontus-X testnet. La pregunta a partir de aquí ya no es "¿cómo se hace X?", es **"¿cuál de los cinco huecos de la literatura es el que quiero atacar?"**. Esta fase es **donde dejas de estudiar el campo y empiezas a contribuir a él**.
>
> **Tiempo estimado realista**: **no aplica como las anteriores**. Esta fase no termina en 6 ni 8 semanas — **es el doctorado**. Lo que cubre este documento es **el on-ramp de 8–12 semanas** que te lleva de "fin de Fase 5" a "primer experimento original con valor de publicación". Después de eso, el calendario lo marca tu director y tu vena propia de investigación, no un roadmap.

---

## Cómo usar esta guía

Misma lógica que las fases anteriores, con cuatro avisos específicos:

**No eres alumno aquí; eres investigador**. En Fase 4 leías el survey de Levine para aprender el campo. Aquí lees el survey de Qi 2021 para **identificar lo que falta** del campo. El cambio mental es enorme. Si te encuentras consumiendo papers como si fueran un manual, estás haciendo Fase 6 mal: tu trabajo es **encontrar el hueco**, no llenarlo desde el principio. Llenar viene después.

**El estado del arte aquí es honestamente delgado**. Cuando llegues a la intersección "Offline RL × Federated × dataspaces", vas a ver que **el número total de papers publicados con resultados serios cabe en una mano**. Eso es a la vez **el problema y la oportunidad**. Asume que muchas decisiones (qué baseline canónico usar, qué benchmark, qué métrica) **no están establecidas todavía** — y que **establecerlas es una contribución doctoral perfectamente válida** (los workshops de NeurIPS y ICLR sobre benchmarks aceptan esto explícitamente).

**Pontus-X es tu diferenciador real, no tu obstáculo**. La mayoría de papers de FRL están en simulación pura (5 clientes Ray sobre una GPU). Tú llegas a la conversación doctoral con **un deployment funcional en testnet, conocimiento operacional de Ocean Protocol y deltaDAO**, y la posibilidad de hacer experimentos sobre **infraestructura realista** que el 99% de los autores no han tocado. Esto importa. Si tu tesis no se monta sobre esta diferencia, has malgastado Fase 5.

**Hay que decidir, no hay que cubrirlo todo**. Las cinco vías del roadmap maestro (Offline FRL non-IID, FRL+DP, Personalized FRL, comunicación-eficiente, Sim2Real federado) **no son un menú de degustación**: son **alternativas mutuamente excluyentes para la tesis**. Vas a elegir una (o un cruce muy específico de dos) y vas a defender esa elección. Este documento no te dice cuál — te da la información para decidir con tu supervisor.

---

## Mapa de la Fase 6

| Bloque | Tema | Peso |
|---|---|---|
| 6.1 | Estructura del campo: los tres ejes (entornos, qué se federa, online/offline) | 🔥🔥🔥🔥🔥 La taxonomía mental |
| 6.2 | El paper fundacional: QAvg, PAvg, DQNAvg y la heterogeneidad de entornos | 🔥🔥🔥🔥🔥 No negociable |
| 6.3 | Convergencia y garantías teóricas en FRL | 🔥🔥🔥🔥 La rama teórica del campo |
| 6.4 | Offline Federated RL — la intersección de tu tesis | 🔥🔥🔥🔥🔥 El corazón del documento |
| 6.5 | Personalized FRL y meta-RL federado | 🔥🔥🔥 Una de las 5 vías de tesis |
| 6.6 | Comunicación-eficiente FRL — economía de tokens en Pontus-X | 🔥🔥🔥🔥 Vía con valor de negocio |
| 6.7 | Privacy en FRL: DP user-level con trayectorias | 🔥🔥🔥🔥 Vía regulatoria |
| 6.8 | Off-Policy Evaluation federado | 🔥🔥🔥🔥🔥 Subcampo vacío, donde puede estar tu hueco |
| 6.9 | Disciplina experimental específica de FRL | 🔥🔥🔥🔥 No es disciplina genérica |
| 6.10 | Elección de vía de tesis: matriz de decisión | 🔥🔥🔥🔥🔥 La decisión central |
| 6.11 | Proyecto inaugural: benchmark Offline-FRL + primer paper de workshop | 🔥🔥🔥🔥🔥 Tu primer artefacto original |
| 6.12 | Hábitos de investigador: reading group, arXiv, conferencias | 🔥🔥🔥 Infraestructura intelectual |

---

## 6.1 — Estructura del campo: los tres ejes

### Por qué importa — y esta es la taxonomía mental que define toda la fase

Federated Reinforcement Learning no es **un** problema sino **una familia** de problemas. Sin una taxonomía mental clara, todo paper te parecerá una variante del anterior y te perderás. **Tres ejes ortogonales** organizan el campo, y necesitas tenerlos tatuados:

**Eje 1 — Heterogeneidad de entornos**. ¿Los $K$ clientes comparten el mismo MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ o tienen MDPs distintos? El caso *homogéneo* es relativamente fácil — es esencialmente "Federated Q-Learning con linear speed-up en $K$". El caso *heterogéneo* es **donde está la frontera**, porque la política óptima depende de cada $P_k$ y $R_k$, y un único modelo global puede no existir como objeto óptimo. **Pontus-X es heterogéneo por construcción**: cada participante del dataspace genera datos desde su propio proceso (cada fábrica, cada hospital, cada banco).

**Eje 2 — Qué se federa**. Hay al menos cinco objetos compartibles, con costes y garantías muy distintos:
1. **Gradientes / pesos** del actor o crítico (FedAvg-style: QAvg, PAvg, DQNAvg, DDPGAvg).
2. **Modelos aprendidos del entorno** $\hat{P}(s'|s,a)$ (cercano a model-based federado, casi virgen).
3. **Representaciones latentes** o embeddings (FedFormer y trabajos relacionados, 2023+).
4. **Trajectories** o datasets completos (poco práctico salvo en setups muy específicos; viola privacy por defecto).
5. **Información de exploración** (mapas de visita, estimaciones de incertidumbre) en setups online.

**Eje 3 — Online vs. Offline**. ¿Cada cliente **interactúa** con su entorno durante el training (Online FRL) o solo tiene un **dataset fijo** $\mathcal{D}_k$ generado por su política histórica $\pi_\beta^{(k)}$ (Offline FRL)?

**Para Pontus-X y Compute-to-Data, tu setup canónico es Offline FRL con heterogeneidad de entornos, donde se federan gradientes (de empezar simple)**. Tres ejes "máximamente difíciles" a la vez. **Esta intersección está particularmente poco estudiada todavía**, y es donde probablemente está tu contribución original.

### Recurso principal — la pieza canónica

- **Qi, Zhu, Wei & Yang (2021) — "Federated Reinforcement Learning: Techniques, Applications, and Open Challenges"** → [arXiv:2108.11887](https://arxiv.org/abs/2108.11887). **El survey clásico de FRL**. Estructura el campo en Horizontal FRL (HFRL) y Vertical FRL (VFRL) por analogía con FL clásico. **Lectura obligatoria entera**, con notas. Es la cita que vas a poner en la introducción de cada paper que escribas en los próximos años.
- **Survey actualizado (2025) — "A Survey of Multi-Agent Reinforcement Learning: Federated Learning and Cooperative and Noncooperative Decentralized Regimes"** → [arXiv:2507.06278](https://arxiv.org/abs/2507.06278). El más reciente y didáctico, **explícitamente cubre la frontera 2023–2025**. Usa los pseudocódigos de QAvg/PAvg/DQNAvg que da aquí como referencia.

### Lecturas complementarias para el contexto

- **Kairouz et al. 2021 (FL clásico, ya conocido de Fase 5)** — repasa la sección donde mencionan RL como dirección abierta. Útil para entender por qué FRL queda fuera del survey principal de FL.
- **Reading lista de NeurIPS / ICLR / ICML workshops 2023–2025** sobre Federated Learning + RL. Los workshops ("Federated Learning in the Age of Foundation Models", "Workshop on Algorithmic Decision-Making with Heterogeneous Data") son donde aparecen los papers tesis-relevantes antes que en el track principal.

### Temas mínimos

**Horizontal FRL (HFRL)**: clientes con espacios $(\mathcal{S}, \mathcal{A})$ iguales o casi iguales pero **trayectorias** distintas. Análogo a "horizontal FL": mismas features, distintos samples. **Pontus-X cae aquí casi siempre**. **Vertical FRL (VFRL)**: clientes con espacios distintos que comparten **estados o instantes temporales**. Mucho menos común; aparece en setups financieros (banco + aseguradora ven al mismo cliente desde ángulos distintos). **Federated multi-task RL**: cada cliente resuelve una **tarea distinta** del mismo entorno (objetivos distintos sobre dinámicas similares). **Federated meta-RL**: el objetivo es un **agente que se adapta rápido** a la tarea de cada cliente (conexión directa con MAML, Fallah et al. 2020). **Cooperative vs. competitive**: ¿los clientes ganan o pierden si el modelo global es mejor? En Pontus-X es **cooperative** por contrato (la federación es voluntaria y beneficia a todos), pero con cuidado: **puede haber free-riders** que extraen valor sin aportar dato útil.

### Ejercicio "wow"

Antes de implementar nada, **dibuja un mapa**: en una pizarra física o digital, traza los tres ejes y **ubica cada paper que has leído hasta ahora** (Jin et al. 2022 sobre eje "heterogeneidad alta + online + se federa Q", Rengarajan et al. 2024 sobre eje "offline + ensemble + se federan políticas", etc.). Verás dos cosas: (1) la mayoría de papers ocupan dos o tres puntos repetidos; (2) **enormes regiones del espacio están vacías**. Esas regiones vacías **son tu hueco potencial**. Guarda ese mapa — vas a actualizarlo cada mes durante el doctorado.

### Checklist 6.1

- [ ] Leí Qi 2021 entero, con notas
- [ ] Leí el survey de 2025 al menos hasta la sección sobre QAvg/PAvg
- [ ] Sé enunciar los tres ejes en una frase cada uno
- [ ] Sé clasificar correctamente cualquier paper de FRL que leo en los tres ejes
- [ ] Tengo el "mapa del campo" dibujado y ubico mi setting Pontus-X en él

---

## 6.2 — El paper fundacional: QAvg, PAvg, DQNAvg

### Por qué importa — y este es el "McMahan 2017" de FRL

Si en FL clásico **todo se entiende en función de FedAvg**, en FRL **todo se entiende en función de QAvg y PAvg**. Jin et al. 2022 son los autores que (1) formalizaron el setting de Federated Reinforcement Learning con Environment Heterogeneity (FRL-EH), (2) propusieron las extensiones naturales de FedAvg a Q-Learning (QAvg) y a Policy Gradient (PAvg), (3) **demostraron formalmente que ambos algoritmos convergen a un punto subóptimo** donde la suboptimalidad depende cuantitativamente de la heterogeneidad de entornos, y (4) introdujeron **personalized variants** (PerQAvg, PerPAvg) que permiten que cada cliente mantenga partes del modelo locales.

**Tu lectura crítica de este paper define el resto de Fase 6**. Si entiendes por qué QAvg converge **a un punto subóptimo** y no al óptimo, entiendes el problema central de FRL. Si entiendes la prueba (no solo el enunciado), tienes la base para evaluar cada paper teórico que viene después.

### Lectura obligatoria — la pieza canónica

- **Jin, Peng, Yang, Wang & Zhang (2022) — "Federated Reinforcement Learning with Environment Heterogeneity"** → [arXiv:2204.02634](https://arxiv.org/abs/2204.02634). **El paper fundacional**. Léelo **tres veces**: una para el setting, una para los algoritmos, una para las pruebas. Las secciones de QAvg (Algorithm 1), PAvg (Algorithm 2) y DQNAvg / DDPGAvg (Algorithm 3) son tu pseudocódigo de referencia para implementar la primera línea base honesta. La figura sobre redes con "embedding layers locales" (la base de PerQAvg) es la idea que conecta con personalized FRL.

### Lecturas complementarias

- **Wang, Yang & Su (2024) — "On the Convergence Rates of Federated Q-Learning across Heterogeneous Environments"** → [arXiv:2409.03897](https://arxiv.org/abs/2409.03897). **Análisis moderno con resultado contraintuitivo**: linear speedup en $K$ (más clientes = más rápido) pero **$E > 1$ degrada significativamente** la convergencia en entornos heterogéneos. **Si tu paper de tesis dice $E$, tienes que citar este**.
- **Hwang & Hong (2025) — "Federated Reinforcement Learning in Heterogeneous Environments"** → [arXiv:2507.14487](https://arxiv.org/abs/2507.14487). Propone **estrategias robustas de local update** que extienden QAvg con técnicas inspiradas en FedProx/SCAFFOLD. Es la lectura "lo que viene después de QAvg" en el camino fix-the-heterogeneity.
- **Mitra, Pappas & Hassani (2024) — "Towards Fast Rates for Federated and Multi-Task Reinforcement Learning"** → [arXiv:2409.05291](https://arxiv.org/abs/2409.05291). Análisis teórico moderno con garantías de convergencia mejoradas. Si tu rama va por la teoría, este es tu paper.

### Temas mínimos

**QAvg**: cada cliente $k$ mantiene una tabla (o red) Q local; en cada ronda corre $E$ steps de Q-Learning local; el servidor agrega $\bar{Q} = \frac{1}{K}\sum_k Q_k$. **PAvg**: misma estructura pero sobre policy gradients; tras los $E$ steps locales, el servidor agrega las políticas y **proyecta** sobre el simplex de probabilidad (ProjPAvg) o aplica softmax (SoftPAvg). **DQNAvg / DDPGAvg**: extensiones a deep RL — comparten **parámetros de la red** en lugar de la tabla Q. La estructura de la red importa: típicamente se comparten todas las capas excepto la **embedding layer** del estado, que se mantiene local cuando los espacios de estado tienen pequeñas variaciones. **El teorema clave**: bajo $L$-smoothness y heterogeneidad medida por $\kappa$ (una norma de diferencia entre $P_k$), QAvg converge a una vecindad del óptimo cuyo radio es $O(\kappa^2)$ — es decir, **el sesgo no desaparece** aunque $K \to \infty$. **Local updates rompen el algoritmo cuando heterogeneidad es alta**: $E=1$ converge bien, $E$ grande converge a un punto peor cuando $\kappa$ crece. **Esta es la patología central del campo** y la motivación para FedProx-style fixes (Hwang 2025) o personalized variants.

### Ejercicio obligatorio

**Replica QAvg sobre el GridWorld de Fase 1** (sí, el tabular). Cinco clientes, cada uno con un GridWorld ligeramente distinto (recompensa en posición distinta, o probabilidad de "slip" distinta). Tres regímenes:

1. **Homogéneo** ($\kappa = 0$): los cinco GridWorlds son idénticos.
2. **Heterogéneo moderado** ($\kappa$ pequeño): pequeñas variaciones en transiciones.
3. **Heterogéneo alto** ($\kappa$ grande): recompensas en posiciones distintas.

Para cada régimen, compara:
- QAvg con $E=1$
- QAvg con $E=10$
- Q-Learning local-only (sin federar)
- Q-Learning sobre la unión de datos (centralizado, "techo")

**Reproduce la figura del paper de Jin et al. 2022 sobre el efecto de $E$**. Cuando lo veas con tus propios ojos, **el campo de FRL se vuelve concreto**: no es matemática abstracta, es una patología que ves en una curva.

Después, **escala a Deep RL**: DQNAvg sobre $K=5$ instancias de `CartPole-v1` con `gravity` distinta entre clientes (puedes modificar el env de Gymnasium para esto). Compara DQNAvg vs. DQN local-only vs. DQN centralizado sobre el merged buffer.

### Checklist 6.2

- [ ] Leí Jin et al. 2022 tres veces, incluyendo el apéndice con la prueba de QAvg
- [ ] Implementé QAvg tabular sobre 5 GridWorlds heterogéneos
- [ ] Reproduje la figura del efecto de $E$ vs. heterogeneidad
- [ ] Implementé DQNAvg sobre CartPole con `gravity` heterogéneo
- [ ] Sé enunciar el teorema de convergencia de QAvg con sus supuestos
- [ ] Leí al menos uno de Wang 2024 / Hwang 2025 / Mitra 2024 para el contexto reciente

---

## 6.3 — Convergencia y garantías teóricas en FRL

### Por qué importa

FRL es **uno de los pocos campos modernos de ML donde los papers teóricos siguen siendo aceptados en venues top sin requerir resultados experimentales SOTA**. La razón es honesta: el espacio de algoritmos es relativamente pequeño y los resultados de convergencia son **algo nuevo bajo el sol** cuando los combinas con privacy o heterogeneidad. Si tu tesis tiene una rama teórica, **este bloque define tu día a día**.

Incluso si tu tesis es 100% empírica, **necesitas saber leer los teoremas**: cada paper experimental respetable cita una garantía de convergencia para justificar por qué su algoritmo no es magia.

### Lecturas

- **Khodadadian, Doan, Romberg & Maguluri (2022) — "Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling"** → [arXiv:2206.10185](https://arxiv.org/abs/2206.10185). El primer paper que prueba **linear speedup** en $K$ para Federated TD learning bajo sampling markoviano (no IID por construcción). La técnica de prueba es la base de muchos análisis posteriores.
- **Woo, Joshi & Chi (2023) — "The Blessing of Heterogeneity in Federated Q-Learning: Linear Speedup and Beyond"** → [arXiv:2305.10697](https://arxiv.org/abs/2305.10697). **Resultado provocador**: bajo cierta condición de "cobertura conjunta", la heterogeneidad **ayuda** porque la unión de los datasets cubre más del espacio estado-acción. Importante para tu intuición sobre el setup Pontus-X.
- **Mitra, Pappas & Hassani 2024 (ya citado)**: garantías de convergencia para FRL y federated multi-task RL.
- **Doan, Maguluri & Romberg (2019) — "Finite-Time Analysis of Distributed TD(0) with Linear Function Approximation"** → [arXiv:1902.07393](https://arxiv.org/abs/1902.07393). Análisis pionero. Base del aparato matemático.

### Temas mínimos

**Linear speedup en $K$**: bajo el modelo "todos los clientes muestrean del mismo MDP", el error del algoritmo federado escala como $O(1/\sqrt{KT})$ en vez de $O(1/\sqrt{T})$ centralizado. **Concentrability coefficient $C_\pi$**: cuánto se aleja la distribución de las trayectorias de $\pi_\beta$ de la inducida por $\pi$. **Heterogeneity coefficient $\kappa$**: norma de diferencia entre $P_k$ o entre $R_k$ a través de clientes. Las cotas de error suelen tener la forma $O(1/\sqrt{KT} + \kappa^2)$ — el segundo término **no decrece** con más datos. **Local updates ($E > 1$)**: amplifican el bias por $E$ veces aproximadamente; las cotas se degradan como $O(E\kappa^2)$. **Markovian sampling**: en RL las muestras no son IID — están **correlacionadas temporalmente**. Las técnicas de prueba que usan ergodicidad / mixing time son la herramienta canónica. **Contracción de Bellman** sigue siendo la garantía base, pero compuesta con la dinámica federada.

### Ejercicio

**No demuestres teoremas desde cero**. Sí: lee la prueba de **Khodadadian 2022** (la del linear speedup en TD federado) **con lápiz**, anotando dónde aparece cada supuesto y dónde se usa. **No necesitas saber re-demostrarla**, necesitas saber:

1. Qué supuestos hace.
2. Cuáles **se rompen** en tu setting Pontus-X (mixing time real desconocido, heterogeneidad alta, sampling no markoviano por offline).
3. Cómo se modifican típicamente para tu setting.

Esto es **el ejercicio de un investigador, no de un alumno**. Si lo haces bien, esta lectura te lleva 1–2 semanas; **es tiempo bien invertido**.

### Checklist 6.3

- [ ] Leí la prueba completa de al menos un teorema de FRL (Khodadadian o Mitra)
- [ ] Sé enumerar 4–5 supuestos típicos de los análisis teóricos de FRL
- [ ] Identifiqué qué supuestos se rompen específicamente en mi setting Pontus-X
- [ ] Sé explicar en pizarra qué es "linear speedup en $K$" y qué condiciones requiere
- [ ] Tengo intuición de la forma típica de las cotas $O(1/\sqrt{KT} + \kappa^2)$

---

## 6.4 — Offline Federated RL — la intersección de tu tesis

### Por qué importa — y este es el corazón del documento

**Aquí es donde tu Fase 4 y tu Fase 5 se encuentran**. Offline Federated RL combina:
- **Distributional shift** (Fase 4): cada cliente tiene un dataset generado por su política $\pi_\beta^{(k)}$, y la política aprendida puede alucinar valores Q para acciones OOD.
- **Heterogeneidad** (Fase 5 y 6.2): cada cliente tiene **distinto $\pi_\beta^{(k)}$ y distinto MDP $\mathcal{M}_k$**. El distributional shift no es solo entre $\pi_\beta$ y $\pi$ — **es entre cada $\pi_\beta^{(k)}$ por separado**.
- **Compute-to-Data**: los datasets **no se mueven** del cliente. Solo viajan parámetros del modelo. Esta es la realidad Pontus-X.

**Esta intersección está particularmente poco estudiada**. A fecha de 2025, hay quizás 10–20 papers con resultados serios. El de Rengarajan et al. (NeurIPS 2024) es probablemente el más visible. **Esto es a la vez un problema (no hay tradición establecida) y una oportunidad (puedes establecerla tú)**.

### Lectura obligatoria — el primer paper "tesis-relevante" del bloque

- **Rengarajan, Banerjee, Sundararajan, Mishra, Kalathil, Shakkottai & Caramanis (2024) — "Federated Ensemble-Directed Offline Reinforcement Learning"** → [proceedings.neurips.cc / NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b99315234cc95e6ef281f9155b68832-Paper-Conference.pdf). **El paper que mejor representa el estado del arte de Offline FRL hoy**. El setting de su Figura 1 — **cinco clientes con `hopper-expert-v2` y cinco con `hopper-medium-v2`, sin conocimiento de la calidad de cada uno** — es **literalmente** un proxy de un dataspace heterogéneo donde nadie te dice qué política generó cada dataset. **Léelo entero, con foco en (1) el setup experimental, (2) por qué centralizar los datasets falla en su Figura 1, (3) qué propone el algoritmo ensemble-directed**. Es el paper que vas a **extender o desafiar** en tu primer experimento original.

### Lecturas complementarias

- **Qiao & Wang (2022) — "Offline Reinforcement Learning with Differential Privacy"** → [arXiv:2206.00810](https://arxiv.org/abs/2206.00810). Offline RL + DP en el caso centralizado, con garantías formales. **Las trayectorias se tratan como unidades de privacy** (user-level), análogo a Fase 5.6 pero adaptado al setting secuencial. Necesario antes de pensar en DP-Offline-FRL.
- **Mathieu, Marfoq, Neglia & Lozzo (2024) — "Differentially Private Deep Model-Based Reinforcement Learning (PriMORL)"** → [arXiv:2402.05525](https://arxiv.org/abs/2402.05525). **Adaptación honesta de DP-FedAvg al setting de RL** con trayectorias como units de privacy. Operativo más que teórico.
- **Cheng, Yang & Cai (2023) — "Federated Offline Reinforcement Learning"** (varias propuestas con este título; busca en arXiv. Una versión: [arXiv:2206.05581](https://arxiv.org/abs/2206.05581)). Primer intento sistemático de combinar CQL con FedAvg.
- **Papers de workshop NeurIPS 2024 / ICLR 2025** sobre Offline FRL. **Búsqueda mensual obligatoria** en arXiv CS.LG con query `"offline federated reinforcement learning"` y filtros por año.

### Temas mínimos

**El setting formal**: $K$ clientes, cada uno con dataset $\mathcal{D}_k = \{(s, a, r, s')\}$ generado por $\pi_\beta^{(k)}$ sobre MDP $\mathcal{M}_k$. Objetivo: aprender una política $\pi$ (global o personalized) **sin acceso al entorno y sin compartir los datasets**. **Composición de dificultades**: distributional shift **dentro** de cada cliente (Fase 4) + heterogeneidad **entre** clientes (Fase 5/6.2). **Approaches naïves y por qué fallan**:

1. **CQL local + FedAvg de pesos**: cada cliente entrena CQL local, el servidor promedia. **Falla** porque el término conservativo se computa contra distribuciones locales $\pi_\beta^{(k)}$ distintas — el promedio de penalizaciones no garantiza el lower bound global.
2. **IQL local + FedAvg de pesos**: análogo. **Funciona mejor** en práctica porque IQL no requiere queries OOD, pero la expectile regression con $\tau$ fijo puede degradarse si la calidad de los datasets varía mucho entre clientes (Rengarajan et al. 2024 explotan exactamente esto).
3. **TD3+BC federado**: el término BC se vuelve **incoherente entre clientes** si las $\pi_\beta^{(k)}$ son muy distintas — empuja a la política global hacia una "media de comportamientos" que puede no ser bueno en ningún cliente.

**Approaches no triviales**: ensemble-directed (Rengarajan 2024 — entrena ensembles y dirige el aprendizaje hacia regiones donde el ensemble está confiado **y** alineado entre clientes); personalized (cada cliente mantiene su propio CQL/IQL pero comparte una parte del modelo, conexión directa con 6.5); cluster-based (agrupa clientes por similitud de $\pi_\beta$ inferida y entrena un modelo por cluster).

**OPE federado** (Fase 6.8): cómo evaluar una política aprendida en el setting offline-federated sin acceso al entorno y sin agregar datasets. **Probablemente el subproblema más vacío del campo hoy**.

### Ejercicio obligatorio — el central de la fase

**Replica el setup de Rengarajan 2024 sobre tu pipeline de Fase 4**. Específicamente:

1. **Toma tu IQL y tu CQL** del repo de Fase 4 (ya replicados, ya con ≥5 seeds e IQM).
2. **Construye un setup federado mixto sobre D4RL**: $K=10$ clientes, 5 con `hopper-expert-v2`, 5 con `hopper-medium-v2`. Cada cliente solo ve **su** dataset.
3. **Tres baselines obligatorios**:
   - **Local-only**: cada cliente entrena CQL/IQL solo sobre su dataset.
   - **Centralized**: entrena CQL/IQL sobre la **unión** de los 10 datasets (esto **no es un baseline desplegable**, pero es el "techo" teórico... que en Offline FRL **no siempre es el mejor**, como muestra Rengarajan).
   - **FedCQL / FedIQL naïve**: FedAvg de pesos sobre los CQL/IQL locales, sin más sofisticación. Este es el **baseline crítico**: tienes que ver cómo se degrada en heterogeneidad.
4. **Reporta resultados con disciplina de Fase 4 (≥5 seeds, IQM, bootstrap CI 95%)** y compara contra los números de Rengarajan 2024 cuando aplique.

**Spoiler**: vas a ver que (a) Local-only es el suelo, (b) Centralized es a veces peor que esperarías, y (c) FedCQL/FedIQL naïve cae entre los dos pero a menudo más cerca de Local-only que de Centralized. **Esa observación experimental es la motivación de tu primer paper original** — y de la mitad del campo.

### Checklist 6.4

- [ ] Leí Rengarajan et al. 2024 entero, con foco en setup y figuras
- [ ] Leí Qiao & Wang 2022 (Offline RL + DP centralizado)
- [ ] Reproduje el setup `hopper-expert + hopper-medium` con 10 clientes
- [ ] Tengo tabla comparativa Local-only / Centralized / FedCQL / FedIQL en al menos 2 entornos
- [ ] Mis números reproducen las tendencias de Rengarajan 2024 (no necesariamente los valores absolutos)
- [ ] Sé explicar por qué FedAvg naïve sobre CQL pierde garantías de lower bound
- [ ] Tengo una **hipótesis propia** sobre por qué IQL podría ser más federable que CQL

---

## 6.5 — Personalized FRL y meta-RL federado

### Por qué importa — y aquí hay conexión directa con una de las 5 vías de tesis

Si los entornos son **realmente distintos** entre clientes (cada fábrica tiene una termodinámica ligeramente distinta, cada hospital tiene una población distinta), **un modelo global único es objetivamente peor** que un modelo por cliente. Personalized FRL acepta esto y construye:

- O un **modelo global como "punto de partida"** que cada cliente afina (meta-learning, Per-FedAvg-style).
- O un **modelo por cliente con backbone compartido** (FedPer-style).
- O un **mixture-of-experts** federado donde cada cliente tiene su propio experto.

**Esta es la vía 3 del roadmap maestro**. Es **atractiva** porque (a) hay literatura limpia de pFL clásico de la que tirar (Per-FedAvg, pFedMe, Ditto), (b) la conexión con MAML es directa, (c) en Pontus-X cada participante **quiere** un modelo adaptado a sus datos. Es **arriesgada** porque (a) requiere implementar bien tanto FRL como meta-RL, (b) puede ser difícil distinguir "mi algoritmo personaliza bien" de "mi algoritmo es Local-only con extra steps".

### Lecturas

- **Jin et al. 2022 (ya citado)** — la última sección introduce **PerQAvg y PerPAvg**, que mantienen **la capa de embedding local** y comparten el resto. Es el punto de partida más simple.
- **Fallah, Mokhtari & Ozdaglar (2020) — "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach" (Per-FedAvg)** → [arXiv:2002.07948](https://arxiv.org/abs/2002.07948). **Imprescindible, ya conocido de Fase 5.9**. La estructura MAML aplicada a FL. Para RL, sustituyes "supervised loss" por "RL loss" y los teoremas siguen aplicando con ajustes.
- **Liu, Xu, Wu et al. (2024) — "Federated Meta Reinforcement Learning for Personalized Tasks"** ([Tsinghua Science and Technology 29(3):911-926](https://www.sciopen.com/article/10.26599/TST.2023.9010066)). FMRL combina PPO local con FedAdam server-side para personalización en RL. **Lectura aplicada importante** porque muestra una receta concreta funcionando.
- **Vettoruzzo, Bouguelia & Rögnvaldsson (2023) — "Personalized Federated Learning with Contextual Modulation and Meta-Learning"** → [arXiv:2312.15191](https://arxiv.org/abs/2312.15191). pFL con modulación contextual; útil como inspiración arquitectónica.
- **Revisión: Wang et al. 2024 — "A Review of Personalized Federated Reinforcement Learning"** (ResearchGate). Survey corto de la rama, útil para mapear el campo.

### Temas mínimos

**Personalized FRL como problema bi-nivel**: aprende un meta-parámetro $\theta$ tal que $\theta_k = \theta - \alpha \nabla F_k(\theta)$ resuelve bien la tarea del cliente $k$ con pocos steps locales. Es **exactamente MAML** con $F_k$ siendo el RL objective del cliente. **Two-loop training**: outer loop = federated rounds; inner loop = local adaptation (uno o varios gradient steps). **Computational cost**: el segundo gradiente del MAML es caro; **FOMAML** (first-order approximation, Reptile-style) lo evita. **Per-FedAvg para RL**: aplica MAML con PPO o SAC en el inner loop. Cuidado: en RL los gradientes son ruidosos, y el segundo gradiente de MAML puede explotar — Liu et al. 2024 lo resuelven evitando MAML estricto y usando algo más parecido a Reptile + FedAdam. **PerQAvg / PerDQNAvg**: mucho más simple — comparte todas las capas excepto la embedding del estado, que se mantiene local. **Una excelente baseline antes de cualquier MAML**.

**El problema de evaluación**: en pFRL, **¿qué reportas?** Accuracy/return promedio por cliente (con varianza), distribución completa, o accuracy del peor cliente (worst-case). En tesis aplicada, **reportar los tres** es la disciplina honesta.

### Ejercicio (opcional pero recomendado si tu vía es ésta)

Sobre tu QAvg de 6.2:

1. **Implementa PerQAvg** (capas compartidas + embedding local). 50 líneas extra sobre QAvg. Compara accuracy promedio-por-cliente vs. QAvg vanilla en tu setup de GridWorlds heterogéneos.
2. **Implementa fine-tuning local post-QAvg**: tras la convergencia federada, cada cliente hace $E_{adapt}$ steps locales adicionales. **Esto es el Reptile-de-pobre** y a menudo es competitivo. Repórtalo como baseline.
3. **Si te animas**: replica PerDDPGAvg sobre los CartPoles con `gravity` heterogénea. Compara contra DDPGAvg y DDPG local-only.

**No te metas en MAML completo** salvo que estés ya seguro de que personalized FRL es tu vía de tesis. Los gradientes de segundo orden con políticas estocásticas son una rathole que devora semanas.

### Checklist 6.5

- [ ] Leí Per-FedAvg (Fallah 2020) y la sección de personalización de Jin et al. 2022
- [ ] Sé enunciar la diferencia entre Per-FedAvg (MAML federado) y PerQAvg (embedding local)
- [ ] Implementé al menos uno: PerQAvg, FedPer-RL o fine-tuning local post-QAvg
- [ ] Sé reportar accuracy global, accuracy promedio por cliente, y accuracy worst-case
- [ ] Tengo una opinión informada sobre si "personalized FRL" es mi vía o no

---

## 6.6 — Comunicación-eficiente FRL — economía de tokens en Pontus-X

### Por qué importa — y aquí hay valor de negocio directo

En FL clásico, reducir bytes comunicados es **optimización**. En Pontus-X, **es economía**: cada transacción tiene coste fijo (gas/tokens), cada byte movido off-chain a través de provider nodes consume recursos, y los costes se acumulan ronda a ronda. Si tu agente entrena en 1000 rondas con un modelo de 50M parámetros, **la diferencia entre enviar gradientes completos vs. comprimidos a 8-bit es real en EUROe**.

Esto significa que **comunicación-eficiente FRL no es una rama académica más; es la rama con valor económico mensurable** para tu setting. Es la **vía 4 del roadmap maestro**.

Hay más: en setups con muchos clientes y baja conectividad (algunos dataspaces industriales), **la frecuencia de comunicación** es un parámetro de diseño. $E$ grande reduce comunicación pero **rompe convergencia en heterogeneidad** (Wang 2024 lo demuestra formalmente). Encontrar el sweet spot $E^*$ específico de un dataspace es un problema concreto.

### Lecturas

- **Konečný et al. 2016 (ya conocido de Fase 5.5)** — sobre compresión de gradientes en FL.
- **Sattler et al. 2019 (ya conocido)** — STC, sparsification.
- **Lan, Tang, Yan & Du (2023) — "FedFormer: Contextual Federation with Attention in Reinforcement Learning"** → buscar arXiv. Uso de transformers para agregar embeddings entre clientes; conceptualmente, **comunica representaciones, no gradientes**. Lectura inspiradora.
- **Wang, Yang & Su 2024 (ya citado)** — análisis teórico del trade-off comunicación-vs-convergencia en FRL específicamente.
- **Yue et al. (2021) — "Efficient Federated Meta-Learning over Multi-Access Wireless Networks"** → [arXiv:2108.06453](https://arxiv.org/abs/2108.06453). Setting wireless, pero las técnicas de selección de clientes y compresión son trasladables.

### Temas mínimos

**Compresión de gradientes/pesos**: cuantización (8-bit, 4-bit, 1-bit), sparsification (top-k), sketching, low-rank factorization. **Trade-off canónico**: factor de compresión vs. degradación de retorno. En RL, **más sensible que en clasificación** — un par de epochs con gradientes corrompidos pueden hacer divergir la política. **Local updates como compresión implícita**: $E$ grande = menos rondas = menos comunicación, pero degrada convergencia en heterogeneidad. **Selección estratégica de clientes por ronda**: en lugar de todos los $K$, muestrear $S_t \subset [K]$ con $|S_t| = C \cdot K$, $C < 1$. Reduce comunicación por factor $1/C$. Estrategia clásica: muestrear $\propto n_k$; estrategia inteligente: muestrear clientes con **mayor pérdida** o **mayor incertidumbre**. **Compresión en el upstream** (cliente → servidor) y **downstream** (servidor → cliente). En FL el upstream suele dominar; en FRL **depende del tamaño del modelo y la red local del cliente**.

**Compute-to-Data específico de Pontus-X**: las transacciones on-chain solo llevan **metadata y hashes** del modelo. El modelo en sí viaja off-chain por canales tradicionales (HTTPS, S3-like storage). **Esto significa que el cuello de botella económico no es siempre el tamaño del modelo** — puede ser la **frecuencia de compute jobs** (cada job tiene un coste fijo en EUROe). Reducir **el número de rondas** ($T$) puede ser más rentable que reducir **el tamaño por ronda**.

### Ejercicio

1. **Cuantización 8-bit sobre tus updates de QAvg/DQNAvg**. Mide retorno final vs. bytes comunicados. Curva trade-off explícita.
2. **Top-k sparsification (k=10%)** sobre los mismos updates. Misma medición.
3. **Estudio de $E$**: con tu setup de 6.2/6.4, varía $E \in \{1, 2, 5, 10, 50\}$ y reporta retorno final vs. número de rondas hasta convergencia. **Esta es exactamente la figura del paper de Wang 2024**, reproducida sobre tu pipeline.
4. **Estimación de coste en Pontus-X**: con los datos de tu deploy de Fase 5.11, **estima cuánto costaría en EUROe** entrenar QAvg con cada compresión sobre 100 rondas con $K=10$ clientes. **Esta estimación es un experimento "operacional" único que casi nadie en FL puede hacer** — es exactamente el tipo de figura que un supervisor potencial agradece ver.

### Checklist 6.6

- [ ] Implementé cuantización 8-bit y top-k sobre updates federados de RL
- [ ] Tengo la curva trade-off compresión vs. retorno
- [ ] Reproduje la curva $E$ vs. convergencia en heterogeneidad
- [ ] Hice una estimación de coste en EUROe de un training federado realista en Pontus-X
- [ ] Conozco al menos 3 técnicas de selección de clientes y cuándo aplicar cada una

---

## 6.7 — Privacy en FRL: DP user-level con trayectorias

### Por qué importa — y aquí está la vía regulatoria

DP en supervised FL (Fase 5.6) tiene una unidad de privacy clara: **un sample** ($x_i, y_i$) o **un usuario** completo (todos sus samples). En RL, la unidad natural es **una trayectoria** $\tau = (s_0, a_0, r_0, \ldots, s_T)$. Las trayectorias son **correlacionadas internamente** (los $(s_t, a_t)$ siguientes dependen de los anteriores), **largas** (potencialmente miles de pasos), y **el atacante puede inferir información de cualquier segmento**.

Esto rompe varias suposiciones de DP-SGD estándar:
- **Per-sample clipping** no es lo natural; **per-trajectory clipping** sí lo es, pero las trayectorias tienen tamaño variable.
- **Privacy amplification by sampling** (clave para DP-SGD eficiente) no aplica trivialmente cuando muestreas segmentos correlacionados.
- **Composition de privacy a lo largo de muchos updates** es más cara porque cada trayectoria contribuye a múltiples updates de Bellman.

**Para Pontus-X en aplicaciones reguladas (médicas, financieras, industriales bajo Data Governance Act y AI Act)**, DP a nivel de trayectoria/cliente es **probablemente requisito legal**, no opcional. Esto es la **vía 2 del roadmap maestro**, y combinada con la vía 1 (Offline FRL non-IID) compone una tesis muy concreta y diferenciable.

### Lecturas obligatorias

- **Qiao & Wang 2022 (ya citado)** — Offline RL + DP centralizado, con garantías formales sobre tabular y MDPs lineales. Tu **punto de partida teórico**.
- **Mathieu et al. 2024 (PriMORL, ya citado)** — DP en RL model-based, **con DP-FedAvg adaptado a unidades de trayectoria**. La sección donde discuten "per-trajectory clipping" es la que tienes que asimilar.
- **McMahan, Ramage, Talwar & Zhang (2018) — "Learning Differentially Private Recurrent Language Models" (DP-FedAvg)** → ya conocido de Fase 5.6. **Releer con la pregunta concreta**: ¿qué cambia si la unidad de un cliente es una trayectoria larga y no una secuencia de samples?
- **Garcelon, Perchet, Pike-Burke & Pirotta (2021) — "Local Differential Privacy for Regret Minimization in Reinforcement Learning"** → NeurIPS 2021 ([proceedings](https://proceedings.neurips.cc/paper/2021/hash/580760fb5def6e2ca8eaf601236d5b08-Abstract.html)). LDP en online RL. Útil para entender el "menú completo" de garantías privacy en RL.
- **"Preserving Expert-Level Privacy in Offline Reinforcement Learning"** → [arXiv:2411.13598](https://arxiv.org/abs/2411.13598). Trabajo reciente (2024) sobre privacy específica del experto que generó las trayectorias. Lectura relevante para tu setting Pontus-X donde la "política experta" del cliente es información sensible.

### Temas mínimos

**Niveles de privacy en FRL**:
1. **Sample-level**: garantía sobre una transición $(s, a, r, s')$ individual. Demasiado débil para la mayoría de aplicaciones reales.
2. **Trajectory-level**: garantía sobre una trayectoria entera. Adecuado para "una sesión de paciente", "una ejecución industrial". **Probablemente la elección canónica para Pontus-X**.
3. **Client-level (user-level)**: garantía sobre la **participación completa** de un cliente. La más fuerte. La que reguladores europeos van a pedir.

**Per-trajectory clipping**: $\|g_\tau\|_2 \leq C$ donde $g_\tau$ es el gradiente acumulado a lo largo de la trayectoria. **Cambio respecto a DP-SGD vanilla**: no clipas cada gradiente individual, clipas el agregado por trayectoria. **Ruido Gaussiano** se añade al agregado del cliente. **Privacy accounting**: RDP o Gaussian DP composition. **Mecanismo concreto (DP-FedAvg trayectoria-level)**:

```
for round t:
    for client k:
        for trajectory tau in D_k:
            g_tau = sum of per-step gradients over tau
            g_tau = g_tau / max(1, ||g_tau|| / C)   # per-trajectory clip
        client_update = sum(g_tau) over trajectories
    aggregated = sum_k client_update / K + noise(sigma)
```

**Curva privacy-utility de FRL**: análoga a la de FL clásico pero con $\epsilon$ típicamente requiriendo **más ruido** para misma garantía porque cada trayectoria afecta a múltiples updates de Bellman. **DP user-level** + **Secure Aggregation** sigue siendo el estándar de oro para "privacy seria", **misma receta que Fase 5**.

### Ejercicio

1. **Implementa DP-FedAvg trayectoria-level sobre tu QAvg/DQNAvg** de bloque 6.2. Usa Opacus como base pero con clipping per-trajectory en vez de per-sample.
2. **Curva privacy-utility federada**: $\epsilon \in \{0.5, 1, 2, 4, 8, \infty\}$ sobre tu setup `hopper-expert + hopper-medium`. Reporta retorno medio vs. $\epsilon$. Esta es la curva canónica de tu tesis si vas por la vía 2.
3. **Compara contra DP-FedAvg sample-level** (clip por transición individual, como Opacus haría por defecto). Muestra que sample-level **subestima la privacy real** que necesitas en producción.

**Documenta el privacy accounting completo**: $\epsilon$, $\delta$, level (trayectoria/cliente), mecanismo de composición. **Sin esto, "DP" en tu tesis es marketing**.

### Checklist 6.7

- [ ] Leí Qiao & Wang 2022 y Mathieu et al. 2024 (PriMORL)
- [ ] Sé distinguir privacy sample-, trajectory- y client-level en RL
- [ ] Implementé DP-FedAvg con per-trajectory clipping sobre QAvg
- [ ] Dibujé curva privacy-utility federada para mi setup
- [ ] Sé qué reportar exactamente para que un revisor crea mi garantía DP
- [ ] Tengo claro qué exige el Data Governance Act / AI Act sobre Pontus-X (al menos a nivel de keywords)

---

## 6.8 — Off-Policy Evaluation federado

### Por qué importa — y este es probablemente el subcampo más vacío

En Fase 4.8 viste que **OPE** (evaluar una política sin desplegarla) es crítico en Offline RL. En FRL, OPE se vuelve **doblemente crítico y triplemente difícil**:

- **Crítico** porque en Pontus-X **no puedes desplegar** la política aprendida en el "entorno real" del cliente (es Compute-to-Data: no hay entorno accesible, solo datasets históricos). **La única forma de evaluar es OPE**. Sin OPE federado robusto, **tu paper no puede afirmar que el algoritmo federado X es mejor que el Y** — solo puede afirmar que es mejor en tu métrica de evaluación, **que es exactamente la cosa cuya validez está en duda**.
- **Difícil** porque (a) cada cliente puede hacer OPE local sobre su dataset, pero (b) la composición de estimadores OPE locales en uno global no es trivial; (c) las garantías de varianza de IS-style estimators se degradan compuestamente; (d) el FQE federado (Fitted Q Evaluation distribuida) requiere su propio análisis de convergencia.

**Resultado**: la literatura de OPE federado a fecha de 2025 es **esencialmente inexistente**. Hay 2–3 papers que lo tocan tangencialmente. **Esto significa dos cosas**: (a) es un riesgo de tesis (si nadie está trabajando en ello, puede haber razones técnicas); (b) **es probablemente donde está el hueco más grande del campo entero**.

Si tu personalidad investigadora te pide trabajar donde el campo está más vacío, **OPE federado puede ser tu tesis entera**.

### Lecturas

- **Toda la sección 4.8 de tu documento de Offline RL** — releer con la pregunta "¿qué cambia en federado?".
- **Voloshin et al. 2019 (ya conocido)** — survey de OPE centralizado.
- **Fu et al. 2021 (DOPE benchmark, ya conocido)** — punto de partida para construir un benchmark federado.
- **Búsqueda activa en arXiv** con queries `"federated off-policy evaluation"`, `"distributed offline policy evaluation"`. Trabajos recientes son escasos pero relevantes; mantén alerta de las nuevas publicaciones.
- **Levine et al. 2020 (tutorial fundacional, ya conocido)** — sección de OPE, leerla con ojo federado.

### Temas mínimos

**Estimadores OPE federados posibles** (todos relativamente nuevos):
1. **FQE federado**: cada cliente entrena su FQE local; servidor agrega los Q-models por FedAvg; la estimación final $\hat{V}^\pi$ se computa como expectación sobre la distribución de inicio federada. **Garantías de convergencia: abierto en la literatura**.
2. **IS federado (per-decision)**: cada cliente computa importance ratios sobre su dataset; el servidor agrega ponderando por tamaño de dataset. **Varianza altamente sensible a heterogeneidad** de $\pi_\beta^{(k)}$.
3. **Doubly Robust federado**: combinación de IS y FQE, hereda la robustez de DR centralizado. Análisis abierto.
4. **Model-based OPE federado**: cada cliente aprende su modelo del entorno $\hat{P}_k$; el servidor agrega los modelos. Permite **rollouts sintéticos federados** — pero el error compuesto del modelo es no trivial.

**Métricas de calidad de OPE federado**: bias, varianza, RMSE entre $\hat{V}^\pi$ y $V^\pi$ real (cuando hay acceso al ground truth en simulación), intervalos de confianza honestos en condiciones federadas.

**Conexión con Pontus-X**: en la práctica, **OPE federado puede ser el "producto" del compute job**. Es decir, los algoritmos publicados en Pontus-X podrían incluir no solo "entrena este modelo" sino **"evalúa esta política sobre tus datos y devuelve el estimador OPE local"**. Esto es operacionalmente más limpio que entrenar federado.

### Ejercicio (más exploratorio que los anteriores)

**Este bloque tiene menos receta porque el campo está más abierto**. Sugerencias:

1. **Sobre tu setup `hopper-expert + hopper-medium` (10 clientes)**, implementa FQE local en cada cliente. Estima $\hat{V}_k^\pi$ por cliente y agrega como $\hat{V}^\pi = \sum_k w_k \hat{V}_k^\pi$.
2. **Compara con ground truth online**: para tu setup D4RL, sí puedes ejecutar $\pi$ en el simulador y obtener $V^\pi$ real. Reporta bias y varianza del estimador federado.
3. **Estudia cómo se degrada** la calidad de FQE federado en función de la heterogeneidad (más expert vs. medium en el split).

Si haces esto bien, **tienes un primer paper de workshop**. En serio. El campo está tan vacío que **una replicación cuidadosa y un análisis honesto del comportamiento del FQE federado es ya una contribución**.

### Checklist 6.8

- [ ] Releí la sección 4.8 con ojo federado
- [ ] Implementé FQE federado sobre mi setup `hopper-expert + hopper-medium`
- [ ] Comparé el estimador federado contra ground truth online y reporté bias/varianza
- [ ] Tengo una intuición sobre cómo la heterogeneidad de $\pi_\beta$ afecta la calidad del OPE federado
- [ ] Tengo identificado al menos un sub-problema concreto y bien delimitado en OPE federado

---

## 6.9 — Disciplina experimental específica de FRL

### Por qué importa

La disciplina de Fase 2 (≥5 seeds, IQM, `rliable`), la de Fase 4 (specifics de offline RL) y la de Fase 5 (specifics de FL) **siguen aplicando todas**. Aquí se **componen** y aparecen **patologías nuevas** que solo emergen en FRL:

- **El "techo centralizado" no siempre es el techo**: en Offline FRL con heterogeneidad alta, **entrenar sobre la unión de datasets puede ser peor** que entrenar federado bien-personalizado, porque la unión mezcla regímenes que no son combinables (Rengarajan 2024 lo muestra). **Reportar centralizado como "techo trivial" es engañoso**.
- **La heterogeneidad no es un número, es un vector**: heterogeneidad de $\pi_\beta$ (qué generó los datos), de $P_k$ (dinámica del entorno), de $R_k$ (recompensas), y de $n_k$ (tamaño del dataset). Reportar "heterogeneidad alta" sin especificar **qué tipo** es como decir "non-IID" sin especificar Dirichlet $\alpha$ en Fase 5.
- **Hiperparámetros explosionados**: ahora tienes los de RL ($\gamma$, lr, replay buffer), los de Offline RL (CQL $\alpha$, IQL $\tau$, $\beta$), los de FL ($E$, $C$, lr server-side), los de DP ($\epsilon$, $\delta$, clip $C_{clip}$). **Tuneo honesto es brutalmente caro**; ablations honestas son aún peor.
- **Reporting de $E$ en heterogeneidad**: si tu paper dice "QAvg con $E=10$" sin más, no es honesto. $E=10$ con $\kappa$ pequeño converge bien; con $\kappa$ grande, diverge. **Reporta siempre la pareja $(E, \kappa)$**.
- **Ground truth en OPE federado**: la mayoría de papers reportan online evaluation como ground truth pero **esto no se puede hacer en Pontus-X**. Si tu paper afirma "esto generaliza a un dataspace real" pero solo lo evaluaste online, **tu evaluación no es fiel al setting que dices estudiar**.

### Lecturas

- **Toda la disciplina de Fase 4.9** — releer.
- **Toda la disciplina de Fase 5.12** — releer.
- **Henderson et al. 2017 "Deep RL that Matters"** (ya conocido de Fase 2) — sigue siendo el aviso transversal.

### Disciplina mínima específica de FRL

- **Especifica los tres ejes** en cada experimento: nivel de heterogeneidad (cómo se mide), qué se federa (gradientes/embeddings/modelos), online vs. offline.
- **Reporta resultados con tres "techos" y tres "suelos"**: techo = centralized + oracle policy access; techo realista = mejor algoritmo federado conocido (FedCQL/FedIQL); suelo = Local-only; suelo realista = random policy.
- **Cuando uses $E > 1$, reporta también $E = 1$** como ablation obligatoria. Si $E = 10$ gana a $E = 1$ por 2%, **la mejora es ruido**; si gana por 30%, es real.
- **Distingue evaluación online (en simulador) de OPE pura**. Si tu paper se vende como "federated offline RL para dataspaces", la evaluación online es **una conveniencia experimental, no la prueba de que funciona** en el setting real.
- **Privacy accounting completo en FRL**: $\epsilon$, $\delta$, level (sample/trajectory/client), mecanismo, librería.
- **Resultados por cliente, no solo agregados**: la varianza entre clientes es información, no ruido. Reporta accuracy promedio + worst-case.

### Checklist 6.9

- [ ] Especifico los tres ejes en cada experimento de FRL que reporto
- [ ] Reporto siempre Local-only y al menos un baseline federado básico (QAvg/FedAvg-IQL)
- [ ] Reporto $E=1$ junto con $E$ elegido como ablation
- [ ] Distingo evaluación online de OPE en mis tablas
- [ ] Privacy accounting siempre completo
- [ ] Reporto accuracy/return promedio por cliente, no solo agregado
- [ ] Sé enumerar 5 errores comunes en papers de FRL para evitarlos

---

## 6.10 — Elección de vía de tesis: matriz de decisión

### Por qué importa — y esto es probablemente la decisión más importante del documento

Las cinco vías del roadmap maestro son **alternativas, no menú de degustación**. Vas a elegir **una** (o un cruce de dos) como tu tesis. Esta sección te ayuda a decidir con criterio.

**Recordatorio de las cinco vías**:
1. **Offline FRL en non-IID extremo**: CQL/IQL + FedAvg cuando cada cliente tiene $\pi_\beta^{(k)}$ distinta. Distributional shift agravado por heterogeneidad.
2. **FRL con privacy garantizada**: DP-SGD trajectory/user-level + secure aggregation sobre updates de Q-values.
3. **Personalized FRL**: meta-RL federado, cada cliente con MDP ligeramente distinto.
4. **Comunicación-eficiente FRL**: compresión, sketching, selección estratégica, optimización del trade-off $(E, T, K)$ específico de Pontus-X.
5. **Sim2Real federado**: combinar datos reales y simuladores publicados en el dataspace.

### Matriz de decisión

| Vía | Madurez del campo | Riesgo técnico | Encaje Pontus-X | Posibilidad de teoría | Diferenciación |
|---|---|---|---|---|---|
| 1. Offline FRL non-IID | Baja-media (10–20 papers) | Bajo (tienes Fases 4+5) | Altísimo | Alta | Alta |
| 2. FRL + DP | Baja | Medio (DP en RL es duro) | Alto (regulación EU) | Muy alta | Muy alta |
| 3. Personalized FRL | Media | Medio-alto (meta-RL inestable) | Alto | Media | Media |
| 4. Comunicación-eficiente | Media-alta | Bajo | **Único de Pontus-X** | Media | Muy alta (operacional) |
| 5. Sim2Real federado | Baja | Alto (compone tres mundos) | Depende del dataspace | Baja | Alta |

**Lecturas clave para decidir**:
- Las **introducciones** (no los technicalia) de los 5 papers más recientes de cada vía. Te sitúan en qué pregunta concreta están haciendo en cada rama.
- **Tu propio `notes_for_supervisor.md`** de Fase 4 y Fase 5: la matriz de gaps que construiste allí señala dónde **tú** ves el hueco.
- **Una conversación de 1h con tu supervisor potencial**, después de mandarle el mini-paper de Fase 5 + `notes_for_supervisor.md` + el experimento de 6.4. Su respuesta te dice qué le interesa a él/ella, lo cual es la mitad de tu decisión.

### Recomendación honesta

**La combinación más natural para tu perfil parece ser 1 + 4**: Offline FRL non-IID (vía 1) con foco en comunicación-eficiente sobre Pontus-X (vía 4). Razones:

- **Encajas en lo que ya sabes**: Fase 4 te dio Offline RL, Fase 5 te dio FL + Pontus-X. Vía 1+4 explota directamente las dos inversiones.
- **Diferenciador inmediato**: nadie en el campo está combinando Offline FRL con coste de Pontus-X como métrica. Es un hueco operacional con valor académico claro.
- **Riesgo bajo**: vía 1 ya tiene baseline establecido (Rengarajan 2024), vía 4 tiene literatura clásica de FL aplicada.
- **Posibilidad de tesis sólida en 3 papers**: (a) benchmark Offline-FRL sobre dataspaces, (b) análisis comunicación-eficiencia en Pontus-X, (c) método propio (compresión adaptativa o cluster-based federation) que mejora el trade-off.

**Si tu director va por la rama teórica, 2 (FRL+DP) es la vía natural**: hay garantías abiertas en privacy de trayectorias y el trabajo de Qiao-Wang ofrece base teórica. Pero el coste de entrada es más alto.

**No elijas 3 (Personalized FRL) ni 5 (Sim2Real federado) salvo que tu director te las pida explícitamente**. La 3 es atractiva pero tiene riesgo medio-alto (meta-RL es inestable empíricamente) y diferenciación media; la 5 compone demasiados problemas para una primera tesis.

### Checklist 6.10

- [ ] Releí mi `notes_for_supervisor.md` de Fase 4 y Fase 5
- [ ] Leí las introducciones de los 5 papers más recientes de cada vía
- [ ] Tuve al menos una conversación dirigida con un supervisor potencial sobre la elección
- [ ] Escribí en mi `notes_for_supervisor.md` la **decisión preliminar** de vía con justificación
- [ ] La decisión está **defendida con criterios**, no con preferencia estética

---

## 6.11 — Proyecto inaugural: benchmark Offline-FRL + primer paper de workshop

### Por qué importa

**Este es el primer artefacto original de tu carrera**. Hasta ahora todo era replicación: replicaste CQL/IQL en Fase 4, replicaste FedAvg en Fase 5, replicaste QAvg en 6.2. **Aquí produces algo que no existía antes**.

El objetivo es **un primer paper sometido a un workshop** de un venue top (NeurIPS Workshop on Federated Learning, ICLR Workshop on Trustworthy ML, NeurIPS Workshop on Offline RL). **Rechazo posible, no pasa nada**: el valor está en el artefacto, no en la aceptación.

### Dos formatos posibles para el primer paper

**Formato A — Benchmark paper (recomendado si tu vía es 1 o 4)**: presentas **el primer benchmark riguroso de Offline-FRL sobre dataspaces**. Construyes:

- Un protocolo de particionado de D4RL en regímenes federados controlados ($\kappa$ medido, $K$ variable, mix expert/medium/random).
- Implementación reproducible de Local-only, Centralized, FedCQL-naïve, FedIQL-naïve sobre ese protocolo.
- Tablas con ≥5 seeds, IQM, performance profiles à la `rliable`.
- Análisis de qué algoritmo de Offline RL es "más federable" empíricamente.
- **Bonus**: una versión del benchmark desplegable sobre Pontus-X testnet — esto es lo que **te diferencia absolutamente** de cualquier otro benchmark paper.

**Formato B — Method paper**: presentas un **método nuevo** (por ejemplo: "FedIQL-Personalized con embedding local" o "DP-FedCQL trajectory-level") y muestras que mejora sobre baselines en el benchmark del Formato A (que tú habrás construido o tomado prestado).

**Recomendación**: para tu primer paper, **Formato A**. Razones: (a) el campo necesita benchmarks decentes y los workshops aceptan benchmarks de calidad, (b) construirlo te da el laboratorio sobre el que iterar el resto de la tesis, (c) requiere menos "saltos creativos" — la disciplina es la que hace el paper.

### Estructura mínima del repo

```
federated-offline-rl-bench/
├── README.md                          # narrativo, gráficas, link a repos Fase 4 y Fase 5
├── benchmark/
│   ├── partitioning/
│   │   ├── kappa_partition.py         # particionado D4RL con kappa controlado
│   │   ├── mix_quality_partition.py   # expert/medium/random mixing
│   │   └── README.md                  # protocolo del benchmark
│   └── metrics/
│       ├── rliable_metrics.py
│       └── ope_federated.py           # FQE federado de 6.8
├── algorithms/
│   ├── local_only.py                  # baseline trivial
│   ├── centralized.py                 # "techo" oracle
│   ├── fed_cql.py                     # FedAvg sobre CQL
│   ├── fed_iql.py                     # FedAvg sobre IQL
│   ├── fed_td3bc.py                   # FedAvg sobre TD3+BC
│   └── per_fed_iql.py                 # opcional: tu primer método propio
├── flower_app/                        # implementación Flower deployable
│   ├── client_app.py
│   ├── server_app.py
│   └── strategy_offline_rl.py
├── pontus_x/                          # ← diferenciador
│   ├── README.md
│   ├── algorithm_local_train_iql/
│   ├── notes_operations.md
│   └── docker/
├── experiments/
│   ├── e1_kappa_sweep.py              # algoritmo vs kappa
│   ├── e2_K_sweep.py                  # algoritmo vs número de clientes
│   ├── e3_E_sweep.py                  # algoritmo vs epochs locales
│   ├── e4_ope_federated.py            # bias/varianza de FQE federado
│   └── e5_pontusx_cost.py             # coste estimado en EUROe
├── configs/                           # Hydra YAMLs
├── figures/                           # generadas por rliable
├── reports/
│   ├── workshop_paper.pdf             # 8 páginas + apéndice
│   └── workshop_paper.tex
├── thesis_relevant/
│   ├── README.md
│   ├── thesis_outline.md              # ← nuevo: outline de 3 papers
│   └── notes_for_supervisor.md        # actualizado con decisión final
├── tests/
├── requirements.txt
└── notebooks/
```

### Tabla obligatoria — la que va al paper

| Algoritmo | $\kappa=0$ | $\kappa$ bajo | $\kappa$ medio | $\kappa$ alto | $\kappa$ extremo |
|---|---|---|---|---|---|
| Local-only | | | | | |
| Centralized | | | | | |
| FedCQL | | | | | |
| FedIQL | | | | | |
| FedTD3+BC | | | | | |
| Tu método (opc) | | | | | |

≥5 seeds, IQM, bootstrap CI 95%. Reportar también accuracy worst-client. **Esta tabla es la columna principal de tu paper de workshop**.

### Mini-paper de workshop (8 páginas + apéndice)

Formato NeurIPS Workshop. Estructura:

1. **Intro**: motivación (Pontus-X / Compute-to-Data como setting real), gap en la literatura (Offline FRL es delgado), contribución (benchmark + análisis sistemático).
2. **Background**: 1.5 páginas. Offline RL, FL clásico, FRL. Cita Levine 2020, McMahan 2017, Qi 2021, Jin 2022, Rengarajan 2024.
3. **Benchmark design**: 2 páginas. Protocolo de particionado D4RL con $\kappa$ controlado. Algoritmos baseline. Métricas (`rliable` + OPE federado).
4. **Experimentos**: 2 páginas. Tabla principal + figures (curva $\kappa$, curva $E$, performance profiles).
5. **Pontus-X deployment**: 0.5 páginas. Demostración de que el benchmark corre sobre testnet real con estimación de coste.
6. **Discussion**: 1 página. Cuál algoritmo es más federable y por qué. Limitaciones. Trabajo futuro (que es **tu tesis**).
7. **Apéndice**: hiperparámetros completos, prompts a CORL, código.

### `thesis_relevant/thesis_outline.md`

Documento de **3 páginas** que escribes **al final de Fase 6**. Contiene:

- **Pregunta central de la tesis**, en una frase. Ejemplo: "Cómo combinar Offline Reinforcement Learning con federación heterogénea sobre dataspaces minimizando degradación por distributional shift compuesto y coste comunicacional".
- **Los 3 papers que componen la tesis**, con un párrafo cada uno:
  - Paper 1: el benchmark (lo que acabas de hacer; va a workshop ahora, a venue principal después de iterar).
  - Paper 2: tu primer método nuevo (probablemente nace del experimento del workshop como "lo siguiente que querría probar").
  - Paper 3: la contribución mayor (típicamente combina dos vías o aporta análisis teórico).
- **Calendario realista por años**: año 1 paper 1, año 2 paper 2, año 3 paper 3, año 4 defensa.
- **Riesgos identificados** y planes B.

Este documento **es para tu supervisor, no para publicar**. Es la pieza que abre la conversación "esto es lo que quiero hacer en los próximos 3 años, y aquí está la evidencia de que sé hacerlo".

### Checklist 6.11

- [ ] Repo público con Local-only / Centralized / FedCQL / FedIQL implementados sobre D4RL particionado
- [ ] Tabla con $\kappa$-sweep completa, ≥5 seeds, IQM, CI 95%
- [ ] FQE federado implementado y comparado contra ground truth
- [ ] Despliegue del benchmark en Pontus-X testnet, documentado
- [ ] Mini-paper de 8 páginas en `reports/`
- [ ] `thesis_outline.md` con la pregunta central y los 3 papers
- [ ] `notes_for_supervisor.md` con decisión final de vía
- [ ] **Submission a un workshop top en marcha o ya enviada**

---

## 6.12 — Hábitos de investigador: reading group, arXiv, conferencias

### Por qué importa

Los bloques 6.1 a 6.11 te dan el contenido. Este bloque te da **la infraestructura intelectual** sobre la que vivirás 3+ años. Sin estos hábitos, te dispersarás o te quemarás.

### Hábitos no negociables

**Reading group semanal o quincenal**. Mejor con compañeros del laboratorio; si no, con peers online (Discord de RL, EleutherAI server, ML Collective). Una lectura por sesión, una hora máximo, **siempre la misma estructura**: alguien presenta 20 min, todos discuten 30 min, queda escrito qué se ha aprendido y qué queda abierto. **Sin reading group, la lectura solitaria te agota y no rinde**.

**Lectura dirigida, no exploración pura**. arXiv-sanity y arXiv-feeds están bien para inspiración, pero **no para tu día a día**. Tu día a día son las **referencias que cita el paper que estás extendiendo** y los **papers que citan ese paper**. Eso te da una bola de nieve dirigida en lugar de un océano de ruido.

**Pre-print first, peer review second**. En ML, casi todo lo importante sale primero en arXiv. Espera a que algo lleve 1–3 meses circulando antes de invertir tiempo serio en él — te ahorras los papers que se demuestran erróneos en semanas.

**Una conferencia al año, presencial si puedes**. NeurIPS, ICML, ICLR son las top de ML. EuroSys y SOSP cubren la cara de sistemas. PETS cubre privacy. **Si tu tesis es FRL sobre dataspaces, considera también workshops de Federated Learning en NeurIPS y ICLR específicamente** — son donde tu primer paper irá. Asistir presencialmente vale por el networking; presentar (incluso un poster en workshop) vale **mucho más**.

**Open notebook mental**. Un archivo `research_journal.md` donde anotas cada semana: qué leíste, qué experimentaste, qué te sorprendió, qué te frustró, qué pregunta nueva tienes. **Tres años de doctorado producen miles de microdecisiones; sin journal, las olvidas**.

**Reproducibilidad como hábito**: cada experimento commiteado con seed, config Hydra, git hash, run en W&B. **No** "lo arreglo después". Después es nunca.

### Recursos canónicos para el día a día

- **Papers With Code** para implementaciones de referencia, **incluso con cautela** (la mayoría no replican el paper exactamente).
- **W&B Reports** públicos de la gente del campo (algunos researchers de DeepMind, Anthropic, Berkeley publican sus runs).
- **Twitter/X y Bluesky de la comunidad RL/FL**: cuentas como Sergey Levine, Yann Dubois, Jeff Ding, Tara Sainath, Peter Kairouz. **Modo lectura disciplinada, no scroll infinito**.
- **arXiv-sanity** (Karpathy) con suscripción a keywords como `federated reinforcement learning`, `offline reinforcement learning`, `differential privacy reinforcement learning`. Un email diario filtrado.
- **Newsletter "The Gradient", "Import AI" (Jack Clark)** para mantener perspectiva amplia.
- **Slack/Discord del laboratorio** + **GitHub Issues activos** de los frameworks (Flower, CORL, d3rlpy, Minari). Donde se discuten los bugs reales.

### Checklist 6.12

- [ ] Tengo reading group establecido (semanal o quincenal)
- [ ] Lectura dirigida configurada (citas + cited-by + alertas arXiv en keywords)
- [ ] `research_journal.md` actualizado al menos semanalmente
- [ ] Una conferencia/workshop ya identificada para someter el paper del bloque 6.11
- [ ] Reproducibilidad por defecto: W&B + Hydra + seeds + git hash en cada experimento
- [ ] Sigo a al menos 10 cuentas relevantes en X/Bluesky, en modo disciplinado

---

## Stack técnico canónico (resumen)

Para que no busques estas decisiones cada vez:

- **RL base**: PyTorch + Gymnasium (Farama). Tu stack de Fases 1–3.
- **Offline RL**: **CORL** ([tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)) sigue siendo tu referencia. **d3rlpy** ([takuseno/d3rlpy](https://github.com/takuseno/d3rlpy)) como segunda opción industrial.
- **Datasets offline**: **Minari** (sucesor oficial de D4RL bajo Farama) + **D4RL legacy** para reproducir resultados de papers anteriores.
- **FL**: **Flower 1.x** ([flower.ai](https://flower.ai/)) — tu framework de federación.
- **DP**: **Opacus** con per-trajectory clipping custom para el setting RL.
- **Dataspace**: **Pontus-X** + **Nautilus** ([nautilus.delta-dao.com](https://nautilus.delta-dao.com/)).
- **Evaluación estadística**: **`rliable`** sigue siendo el estándar transversal.
- **Tracking**: **W&B** + Hydra configs.
- **OPE**: tu propia implementación de FQE federado (no hay librería estándar todavía — **esto puede ser parte de tu contribución open-source**).
- **Visualización**: matplotlib + seaborn para curvas, `rliable.plot_utils` para performance profiles.

---

## Pre-final: lo que se ve cuando se cierra Fase 6 (la versión "fin del on-ramp")

Cuando termines este documento, deberías poder:

- Leer cualquier paper de FRL del 2022–2026 sin tropezar con la notación, los acrónimos (QAvg/PAvg/DQNAvg, HFRL/VFRL, PerQAvg) o los baselines estándar. Clasificarlo en los tres ejes en menos de 2 minutos.
- Implementar QAvg/DQNAvg desde cero en una tarde, FedCQL/FedIQL en dos tardes, FQE federado en una semana.
- Defender en pizarra el problema central de FRL (heterogeneidad de entornos + distributional shift compuesto) y por qué los baselines naïve fallan en cada vía.
- Tener una **vía de tesis elegida con criterio**, **un benchmark propio** corriendo, y **un paper de workshop sometido o en cola para enviar**.
- Conocer Pontus-X operacionalmente al nivel de poder onboardear a otro investigador en una tarde — **esto vale como activo de tesis por sí solo**.
- Tener una **conversación de pares** con un investigador establecido del campo sin sentirte "alumno" — porque ya no lo eres.

**Lo que ya no se ve y es lo importante**: Fase 6 no se cierra en un evento concreto. Lo que se cierra aquí es el **on-ramp**. A partir de ahora vives en bucle paper-experimento-paper-experimento durante el resto del doctorado. Si los hábitos de 6.12 están en pie y el artefacto de 6.11 está sometido a un workshop, **el on-ramp está cumplido y empieza la carretera real**.

---

## Plan sugerido de 8–12 semanas (el on-ramp)

| Semanas | Foco principal | Foco secundario |
|---|---|---|
| 1 | 6.1 Estructura del campo + Qi 2021 survey | Mapa del campo dibujado |
| 2 | 6.2 QAvg/PAvg/DQNAvg + Jin 2022 paper | Replicación QAvg tabular sobre GridWorlds |
| 3 | 6.2 cont. — DQNAvg sobre CartPole heterogéneo | Reproducción figura $E$ vs. heterogeneidad |
| 4 | 6.4 Offline FRL — Rengarajan 2024 + setup | Implementación FedCQL/FedIQL naïve |
| 5 | 6.4 cont. — reproducción `hopper-expert + medium` | Tabla Local/Centralized/FedCQL/FedIQL |
| 6 | 6.6 Comunicación-eficiente + 6.7 DP | Curva trade-off comp/DP-utility |
| 7 | 6.8 OPE federado — FQE federado + ground truth | Bias/varianza del estimador federado |
| 8 | 6.10 Decisión de vía de tesis | Conversación con supervisor potencial |
| 9 | 6.11 Construcción del benchmark formal | Particionado con $\kappa$ controlado |
| 10 | 6.11 cont. — experimentos completos $\kappa$-sweep | Tablas + performance profiles |
| 11 | 6.11 cont. — Pontus-X deploy + figura coste | `notes_operations.md` actualizado |
| 12 | 6.11 cont. — mini-paper + thesis_outline + submission | Workshop submission enviada |

Si el bloque 6.11 te lleva más de 4 semanas, **estíralo sin culpa**. Es **el primer artefacto original de tu carrera**: vale el tiempo que necesites.

Si te quedas atascado en 6.10 (decisión de vía), **es señal de que necesitas una conversación dirigida con tu supervisor potencial antes de seguir solo**. No es debilidad, es buena ingeniería de carrera.

---

## Cosas que **no** entran en este documento y que conviene aclarar

**RLHF, DPO, GRPO**: pertenecen al ecosistema de LLMs (alignment, post-training de modelos de lenguaje). **Tocan tangencialmente RL** pero los problemas son distintos. Apártalos salvo que tu director te los pida explícitamente. Meterlos en tu tesis sin que tu director esté detrás te dispersa.

**MARL (Multi-Agent RL)**: hay superposición conceptual con FRL pero son **campos distintos**. MARL = agentes **comparten entorno** y se ven entre sí (juegos, mercados, equipos de robots); FRL = agentes en **entornos separados** que comparten **parámetros**. Si tu tesis es FRL, MARL es lectura lateral, no central.

**Foundation models federados / LLMs federados**: subcampo creciente y caliente, pero **no es FRL en el sentido estricto**. Es FL aplicado a fine-tuning de LLMs. Si tu director va por ahí, es otra fase, no esta.

**Competiciones NeurIPS**: optativas. Una buena competición da visibilidad; mala gestión te come 6 meses sin producir paper. **No las elijas como primera contribución doctoral**.

**Sim2Real puro (no federado)**: relevante si tu tesis es robótica. Para dataspaces médicos/financieros, lateral.

---

## Cómo seguimos

Cuando estés listo, podemos abrir documentos dedicados sobre cualquiera de:

- **Implementación comentada de QAvg/DQNAvg** desde cero en PyTorch, con experimentos sobre GridWorlds y CartPole heterogéneos paso a paso.
- **Walkthrough de Rengarajan 2024**: leer juntos el paper, replicar las figuras, discutir las decisiones de diseño y los gaps que deja abiertos.
- **Diseño operativo del benchmark de 6.11**: protocolo de particionado, métricas, código de orquestación en Flower, despliegue concreto en Pontus-X testnet con DIDs reales.
- **Análisis crítico de un paper específico**: si quieres ir a fondo en cualquiera de los citados, lo discutimos línea por línea con el ojo de "qué de esto va a aparecer en tu tesis".
- **Conversación dirigida sobre la elección de vía de tesis**: matriz de decisión, lectura de las introducciones de los papers candidatos, simulación de la conversación con tu supervisor.
- **Estructura del primer paper de workshop**: outline, escritura iterada, revisión peer-style antes de enviar.

Mi recomendación de orden:

1. **6.1 + 6.2 como bloque dedicado**: la taxonomía mental + QAvg. Sin esto sólido, todo el resto es ruido.
2. **6.4 como otro bloque dedicado**: Offline FRL es donde vive tu tesis. Inviértele tiempo asimétrico.
3. **6.10 como conversación, no como ejercicio**: la elección de vía es la decisión más importante del año; trátala con la seriedad que merece.
4. **6.11 como bloque final, largo**: el primer artefacto original. Sin prisa pero sin pausa.

Y un aviso final, que vale repetir: **Fase 6 no termina**. Termina el on-ramp; empieza el doctorado. Si al cierre de las 8–12 semanas tienes un benchmark sometido a workshop, una vía elegida y defendible, y los hábitos de 6.12 en pie, **has cumplido el roadmap entero**. Todo lo que viene después es la tesis — y la tesis se construye paper a paper, no fase a fase.

Bienvenido al campo. Ahora ya no eres alumno.
