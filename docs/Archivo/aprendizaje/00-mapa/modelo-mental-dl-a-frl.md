---
title: "Modelo Mental: Del Deep Learning al Aprendizaje Federado con Refuerzo (FRL)"
tags: [mapa, deep-learning]
status: borrador
updated: 2026-08-08
---

# Modelo Mental: Del Deep Learning al Aprendizaje Federado con Refuerzo (FRL)

> **Objetivo del documento:** construir, capa a capa, un modelo mental que parta de los fundamentos del aprendizaje automático y llegue hasta el **aprendizaje federado por refuerzo (Federated Reinforcement Learning, FRL)** sobre **espacios de datos (data spaces)**.

---

## 1. El mapa general: dónde encaja cada cosa

**Relación de campo (subconjuntos):** `Inteligencia Artificial (IA) ⊃ Machine Learning (ML) ⊃ Deep Learning (DL)`.
El DL es simplemente la parte del ML que usa redes neuronales profundas.

**El punto clave del modelo mental:** construir un sistema de ML **no** es bajar por una jerarquía, sino **elegir, de forma independiente, en tres ejes**. Los paradigmas de aprendizaje **no cuelgan de DL**: cuelgan de ML, porque existen igual con o sin redes profundas (un SVM es supervisado; K-Means, no supervisado). El DL es una opción del eje "con qué modelo", no el contenedor de todo lo demás.

```
Inteligencia Artificial (IA)
└── Machine Learning (ML) ── "aprender de los datos en vez de programar reglas"
        │
        │   Un sistema de ML se define eligiendo en 3 EJES INDEPENDIENTES.
        │   Casi cualquier combinación es válida → el FRL es una de ellas.
        │
        ├── ① CÓMO aprende ───── PARADIGMA  (¿de dónde viene la señal de aprendizaje?)
        │     ├── Supervisado ............. etiquetas humanas (x → y)
        │     ├── Semi / débilmente sup. .. pocas etiquetas, o etiquetas ruidosas/imprecisas
        │     ├── Auto-supervisado ........ etiqueta derivada del propio dato (base de los foundation models)
        │     ├── No supervisado .......... sin etiquetas, solo estructura
        │     ├── Por refuerzo (RL) ....... feedback EVALUATIVO por interacción
        │     │                             (ejes internos: online/offline · on/off-policy · model-free/based)
        │     └── Mixtos / híbridos ....... imitation learning, RLHF, active learning, SSL+fine-tuning
        │
        ├── ② CON QUÉ se implementa ── FAMILIA DE MODELO  (el tipo de función que ajustas)
        │     ├── Clásicos ............... regresión, SVM, k-NN, Naive Bayes, árboles de decisión
        │     ├── Ensembles (combinan varios modelos base):
        │     │     ├── Bagging (paralelo, ↓varianza) ...... Random Forest
        │     │     ├── Boosting (secuencial, ↓sesgo) ...... AdaBoost, Gradient Boosting, XGBoost
        │     │     └── Stacking (un meta-modelo los combina)
        │     └── Deep Learning (DL) ..... redes profundas: MLP, CNN, RNN/LSTM, Transformers
        │            └── (aquí es donde "encaja" el DL: como FAMILIA DE MODELO, no como nivel superior)
        │
        └── ③ DÓNDE están los datos ── ARQUITECTURA DE ENTRENAMIENTO
              ├── Centralizado ......... todos los datos en un servidor
              ├── Distribuido .......... datos repartidos, pero un mismo dueño
              └── FEDERADO ............. datos de dueños distintos que NO se comparten
                    └── Data Spaces .... la gobernanza que lo hace posible (soberanía, contratos, conectores)
```

**Cómo leer el mapa:** eliges un valor de cada eje y obtienes un sistema concreto.
- *CNN + supervisado + centralizado* = clasificador de imágenes clásico.
- *XGBoost + supervisado + federado* = scoring crediticio entre banco y telco (SecureBoost).
- **Tu objetivo, el FRL = `① Por refuerzo` + `② redes profundas` + `③ federado`, sobre Data Spaces.**

> **Idea clave:** los tres ejes son **independientes y ortogonales**. El paradigma (CÓMO), la familia de modelo (CON QUÉ) y la arquitectura (DÓNDE) se combinan libremente. Por eso "casi todo se puede federar": federar es una decisión del eje ③ que no obliga a cambiar los ejes ① y ②.
>
> Estos tres ejes son, además, el esqueleto del resto del documento: **eje ① → Sección 3**, **eje ② → Secciones 2 y 4**, **eje ③ → Sección 5**.

---

## 2. ¿Qué es Machine Learning y qué es Deep Learning?

### 2.1 Machine Learning (ML)
Es el campo que construye sistemas que **mejoran su desempeño con la experiencia (datos)** sin ser programados explícitamente para cada caso. Un modelo de ML es una función parametrizada `f(x; θ)` donde:
- `x` = entrada (imagen, texto, sensor...)
- `θ` = parámetros que se ajustan durante el entrenamiento
- El "aprendizaje" = encontrar los `θ` que minimizan una **función de pérdida** (error).

### 2.2 Deep Learning (DL)
Es un **subconjunto del ML** que usa **redes neuronales profundas**: composiciones de muchas capas de transformaciones no lineales.

| Aspecto | ML clásico | Deep Learning |
|---|---|---|
| Features (características) | Diseñadas a mano por expertos | **Aprendidas automáticamente** capa a capa |
| Datos necesarios | Pocos/medios | Muchos |
| Cómputo | Bajo | Alto (GPUs/TPUs) |
| Interpretabilidad | Mayor | Menor ("caja negra") |
| Ejemplos | Regresión logística, SVM, Random Forest | CNN, RNN/LSTM, Transformers |

**La relación:** todo sistema de DL es ML, pero no al revés. El DL automatiza la extracción de representaciones, lo que lo hace dominante en visión, lenguaje y control.

### 2.3 El ciclo de entrenamiento de una red profunda
1. **Forward pass:** la entrada atraviesa las capas → predicción.
2. **Pérdida:** se compara la predicción con el objetivo.
3. **Backpropagation:** se calcula el gradiente de la pérdida respecto a cada parámetro.
4. **Optimización:** se actualizan los pesos (SGD, Adam...).
5. Repetir con muchos lotes (batches) de datos.

> 🔑 **Retén esto:** "entrenar = calcular gradientes y actualizar pesos". El aprendizaje federado consiste, esencialmente, en **compartir esos gradientes/pesos en lugar de los datos**.

---

## 3. Los paradigmas de aprendizaje (el eje "CÓMO")

> **Nota de modelo mental:** los paradigmas son **ortogonales** a la distinción ML↔DL. Existen igual en ML clásico (un SVM es supervisado; K-Means, no supervisado) y en DL. Por eso en el mapa de la sección 1 cuelgan de **ML** (eje ①), no de DL: el DL es solo una opción del eje "con qué modelo" (eje ②). Cualquier paradigma se combina además con cualquier arquitectura del eje "DÓNDE" (eje ③).

### 3.0 La idea organizadora: el espectro de la supervisión

Antes de listar los paradigmas conviene entender **qué los distingue de verdad**. La pregunta clave no es "¿qué algoritmo usa?", sino: **¿de dónde sale la señal que le dice al modelo si va bien o mal?** (la *señal de aprendizaje*). Esa señal puede ser más o menos rica, y eso ordena casi todo:

```
MÁS supervisión humana ◄──────────────────────────────► MENOS supervisión humana

 Supervisado  →  Semi-sup.  →  Débilmente sup.  →  Auto-sup.  →  No supervisado
 (etiqueta       (pocas         (etiquetas          (etiqueta      (sin etiquetas,
  exacta por      etiquetas +    ruidosas /          derivada       solo estructura
  cada dato)      muchos sin     imprecisas /        del propio     de los datos)
                  etiquetar)     incompletas)        dato)

 ══════════════ EJE DISTINTO: ¿qué TIPO de feedback recibe? ══════════════
 Instructivo  → "la respuesta correcta era y"  → todos los paradigmas de arriba
 Evaluativo   → "esa acción valió r"           → APRENDIZAJE POR REFUERZO (RL)
```

Dos matices que evitan los errores conceptuales más comunes:

1. **El RL vive en otro eje.** Los demás paradigmas reciben feedback **instructivo** (se les da la respuesta correcta). El RL recibe feedback **evaluativo**: solo se le dice *cuánto* valió lo que hizo, no *qué* debería haber hecho, y normalmente con **retardo**. Por eso el RL no es "más" ni "menos" supervisado que los otros: es una categoría aparte.
2. **No son cajas estancas.** Los sistemas modernos **encadenan** paradigmas (un LLM se auto-supervisa, luego se ajusta supervisado, luego se refina con RL). A esos encadenamientos se dedica la sección **3.7 (paradigmas mixtos)** — que suele ser donde está la acción real hoy.

---

### 3.1 Aprendizaje supervisado

- **Datos:** pares `(entrada, etiqueta)` → `(x, y)`. La etiqueta la pone un humano o un proceso fiable.
- **Objetivo:** aprender la función `x → y` que **generalice** a datos nuevos (no memorizar el dataset).
- **Señal de aprendizaje:** el error entre la predicción y la etiqueta verdadera, codificado en una **función de pérdida**.
- **Pregunta que responde:** *"¿Qué es esto / cuánto vale esto?"*

**Subtipos por la naturaleza de la salida:**

| Tarea | Salida | Pérdida típica | Ejemplo |
|---|---|---|---|
| **Clasificación binaria** | 1 de 2 clases | Entropía cruzada binaria | ¿Spam o no? |
| **Clasificación multiclase** | 1 de N clases | Entropía cruzada (softmax) | Dígito 0–9 |
| **Clasificación multietiqueta** | varias clases a la vez | Sigmoide por clase | Etiquetar una foto: {playa, perro, sol} |
| **Regresión** | valor(es) continuo(s) | MSE / MAE / Huber | Predecir precio, demanda |
| **Predicción estructurada** | secuencia / estructura | CE por elemento, CTC, seq2seq | Traducción, etiquetado de secuencias |
| **Ranking / aprendizaje a ordenar** | orden relativo | pérdidas pairwise/listwise | Resultados de búsqueda |

**Conceptos clave que conviene tener anclados:**
- **Generalización vs memorización:** el objetivo es el error en datos *no vistos*, no en los de entrenamiento.
- **Sobreajuste (overfitting) / infraajuste (underfitting)** y el **compromiso sesgo–varianza** (*bias–variance trade-off*).
- **Regularización** (L1/L2, dropout, early stopping) para controlar la varianza.
- **Validación** (train / validation / test, validación cruzada) para estimar honestamente la generalización.
- **Cuello de botella práctico:** etiquetar es **caro**. De ahí nacen casi todos los demás paradigmas (intentan aprender con *menos* etiquetas humanas).

---

### 3.2 Aprendizaje no supervisado

- **Datos:** solo entradas `x`, sin etiquetas.
- **Objetivo:** descubrir **estructura oculta** en los datos.
- **Señal de aprendizaje:** propiedades intrínsecas de los datos (distancias, densidad, reconstrucción, verosimilitud).
- **Pregunta que responde:** *"¿Cómo se organiza esto?"*

**Subtipos / tareas principales:**

| Tarea | Qué busca | Métodos representativos |
|---|---|---|
| **Clustering** | agrupar datos similares | K-Means (particional), DBSCAN (densidad), aglomerativo (jerárquico), GMM (basado en modelo) |
| **Reducción de dimensionalidad** | comprimir conservando información | PCA (lineal), t-SNE / UMAP (visualización no lineal), autoencoders (no lineal) |
| **Estimación de densidad** | modelar la distribución `p(x)` | KDE, modelos de mezcla, *normalizing flows* |
| **Detección de anomalías** | encontrar lo atípico | Isolation Forest, one-class SVM, autoencoders por error de reconstrucción |
| **Reglas de asociación** | co-ocurrencias frecuentes | Apriori, FP-Growth ("quien compra X compra Y") |
| **Modelado generativo** | aprender a *generar* datos nuevos | VAE, GAN, modelos de difusión |

> Frontera difusa: el **modelado generativo** se cuenta tradicionalmente como no supervisado, pero hoy muchos de sus métodos más potentes (LLMs, difusión guiada) se entrenan con objetivos **auto-supervisados**. La etiqueta de "paradigma" importa menos que entender **qué señal se está usando**.

---

### 3.3 Aprendizaje auto-supervisado (self-supervised, SSL) — el motor de los *foundation models*

Es la idea que ha redefinido el DL de la última década, así que merece su propia sección. **No hay etiquetas humanas; el modelo fabrica su propia tarea supervisada a partir de la estructura del dato.** Se oculta una parte de la entrada y se entrena al modelo a reconstruirla o a predecirla. Eso convierte un problema *sin* etiquetas en uno *con* etiquetas "gratis" (derivadas del propio dato).

**La gran ventaja:** permite **preentrenar** con cantidades enormes de datos **no etiquetados** (todo Internet, todos los historiales, todas las señales de sensor) y aprender **representaciones** reutilizables.

**Familias de tareas pretexto (pretext tasks):**

| Familia | Idea | Ejemplos |
|---|---|---|
| **Autoregresiva / generativa** | predecir el **siguiente** elemento dado el pasado | GPT (siguiente token), PixelCNN, WaveNet |
| **Enmascarada (masked)** | ocultar partes y reconstruirlas usando el contexto | BERT (palabras), MAE (parches de imagen), masked autoencoding de audio |
| **Contrastiva** | acercar dos "vistas" del mismo dato y alejar las de datos distintos | SimCLR, MoCo; **CLIP** (texto↔imagen) |
| **Auto-distilación (no contrastiva)** | dos vistas del dato sin necesitar negativos | BYOL, DINO |

**La receta dominante hoy** (y por qué importa para tu objetivo):
```
SSL (preentrenar con datos SIN etiquetar)  →  representaciones de propósito general
        ▼
Adaptación a la tarea final con POCAS etiquetas:
   fine-tuning supervisado  |  linear probing  |  prompting / in-context  |  few-shot
```
Esto es lo que hace a los modelos fundacionales tan potentes: el grueso del conocimiento se adquiere sin etiquetas, y la especialización cuesta poco. **En federado es especialmente atractivo:** se puede preentrenar SSL sobre datos privados de muchos participantes sin que nadie tenga que etiquetar nada.

---

### 3.4 Variantes con supervisión parcial: semi-supervisado y débilmente supervisado

Ocupan el tramo intermedio del espectro: hay *algo* de etiqueta, pero imperfecta o escasa.

**a) Semi-supervisado** — pocas etiquetas + muchos datos sin etiquetar.
Métodos típicos:
- **Self-training / pseudo-etiquetado:** el modelo etiqueta los datos sin etiqueta en los que está seguro y reentrena con ellos.
- **Regularización por consistencia:** la predicción debe ser estable ante pequeñas perturbaciones de la entrada (Mean Teacher, FixMatch).
- **Métodos basados en grafos:** propagar etiquetas por similitud entre ejemplos.

**b) Débilmente supervisado (weak supervision)** — las etiquetas existen pero son de **baja calidad**. Tres sabores clásicos:
- **Incompleta:** solo una parte de los datos está etiquetada (≈ semi-supervisado).
- **Inexacta:** la etiqueta es de **grano grueso** (p. ej., "esta imagen contiene un tumor" sin decir dónde → *multiple instance learning*).
- **Inexacta/ruidosa:** las etiquetas tienen **errores** (etiquetado automático, *crowdsourcing*).
- **Supervisión programática / distante:** generar etiquetas con reglas heurísticas o bases de conocimiento (p. ej., **Snorkel**), en vez de a mano.

> Pista de modelo mental: semi-supervisado y débilmente supervisado **no son paradigmas "nuevos"**, son **supervisado con la señal degradada**. Por eso se sitúan entre supervisado y auto-supervisado en el espectro.

> 🔑 **Por qué esto es una oportunidad concreta para FL:** en el mundo real, los clientes federados (usuarios finales, dispositivos, hospitales) generan enormes cantidades de datos pero casi nunca los etiquetan a mano — etiquetar cuesta tiempo y expertise. Combinar FL con semi-supervisado (self-training/pseudo-etiquetado local, regularización por consistencia) permite que cada nodo aprenda de su masa de datos **sin etiquetar**, usando solo las pocas etiquetas que sí tenga como ancla. Es una de las líneas más activas hoy para que el aprendizaje federado sea práctico fuera de los casos de laboratorio con datasets ya etiquetados.

---

### 3.5 Aprendizaje por refuerzo (RL) — el paradigma clave para tu objetivo

Aquí no hay un dataset estático: hay un **agente** que **interactúa** con un **entorno**. La señal es **evaluativa** (recompensa), no instructiva.

```
        acción aₜ
Agente ──────────► Entorno
   ▲                  │
   └──────────────────┘
   estado sₜ₊₁, recompensa rₜ₊₁
```

**Componentes formales (Proceso de Decisión de Markov, MDP):**
- **Estado (s):** lo que el agente observa del entorno.
- **Acción (a):** lo que el agente decide hacer.
- **Recompensa (r):** señal escalar que indica qué tan buena fue la acción.
- **Política π(a|s):** la "estrategia" del agente; mapea estados a acciones. **Esto es lo que se aprende.**
- **Función de valor V(s) / Q(s,a):** estimación de la recompensa futura acumulada.
- **Objetivo:** maximizar la recompensa acumulada (descontada) a largo plazo.

**Dos dificultades propias del RL que conviene nombrar:**
- **Dilema exploración vs explotación:** probar cosas nuevas vs usar lo que ya funciona.
- **Asignación de crédito (credit assignment):** una recompensa tardía puede deberse a una acción tomada mucho antes; atribuir mérito/culpa es difícil.
- **Recompensas densas vs escasas (sparse):** cuanto más rara es la recompensa, más difícil el problema (de ahí técnicas de *reward shaping*).

#### Clasificaciones importantes dentro del RL

**a) Online vs. Offline**

| | RL **Online** | RL **Offline** (batch RL) |
|---|---|---|
| Interacción | El agente interactúa con el entorno **en tiempo real** mientras aprende | Aprende **solo de un dataset fijo** de experiencias pasadas (sin interactuar) |
| Riesgo | Puede tomar malas decisiones reales durante el aprendizaje | Seguro: no actúa sobre el mundo real durante el entrenamiento |
| Problema típico | Coste/peligro de explorar | *Distribution shift*: el dataset no cubre todas las situaciones |
| Cuándo usarlo | Simuladores, juegos, robots con entorno seguro | Sanidad, conducción, industria: donde explorar en vivo es inaceptable |

**b) On-policy vs. Off-policy**
- **On-policy:** aprende de la experiencia generada por **su propia política actual** (ej. PPO, A2C). Más estable, menos eficiente en datos.
- **Off-policy:** puede aprender de experiencia generada por **otras políticas** (ej. Q-Learning, DQN, SAC). Reutiliza datos (replay buffer) → más eficiente. **Es el puente natural hacia el RL offline y el federado.**

**c) Model-free vs. Model-based**
- **Model-free:** aprende directamente la política o los valores sin modelar el entorno.
- **Model-based:** aprende un modelo del entorno y lo usa para planificar/simular (más eficiente en datos, más complejo).

---

### 3.6 Paradigmas mixtos e híbridos (donde está la frontera)

Casi ningún sistema real usa un paradigma "puro". Esta es la parte que más conviene dominar para entender los modelos actuales — y varios de estos híbridos **cruzan precisamente supervisado y RL**, que es el territorio del FRL.

**a) Aprendizaje por imitación (Imitation Learning) — el puente supervisado ↔ RL**
Aprender a actuar a partir de **demostraciones de un experto**, sin necesitar una función de recompensa.
- **Behavioral Cloning (BC):** trata los pares `(estado, acción del experto)` como un problema **supervisado** puro. Simple, pero sufre *distribution shift* (acumula error al salirse de la distribución del experto).
- **DAgger:** corrige eso pidiendo al experto que etiquete los estados que el agente visita.
- Muy relevante en **conducción y robótica**, y como *arranque en frío* antes de afinar con RL.

**b) RL inverso (Inverse RL):** en vez de la política, **infiere la función de recompensa** que explica el comportamiento del experto, y luego optimiza esa recompensa. Útil cuando "qué premiar" es difícil de especificar a mano.

**c) RLHF / RLAIF (RL a partir de feedback humano/de IA) — la receta tras ChatGPT**
Encadena **tres** paradigmas:
```
1. SFT (supervisado): afinar con ejemplos de buenas respuestas.
2. Modelo de recompensa (supervisado): aprender a puntuar respuestas a partir de
   COMPARACIONES humanas ("A es mejor que B").
3. RL (PPO): optimizar la política del modelo contra ese modelo de recompensa.
```
- **DPO (Direct Preference Optimization)** es una simplificación que logra un efecto parecido **sin** el bucle de RL, tratándolo casi como un problema supervisado sobre preferencias.

**d) Aprendizaje activo (Active Learning):** el modelo **elige qué datos quiere que un humano etiquete** (los más informativos/inciertos), minimizando el coste de etiquetado. Es supervisado con un humano *en el bucle*.

**e) Aprendizaje contrastivo:** ya citado en SSL; vive entre lo no supervisado y lo auto-supervisado, y es la base de modelos multimodales como CLIP.

**f) Otros mixtos útiles de conocer:**
- **Curriculum learning:** ordenar los ejemplos de fácil a difícil.
- **Aprendizaje multitarea (multi-task):** una red aprende varias tareas a la vez compartiendo representación.

---

### 3.7 Ajustes transversales (modifican a cualquier paradigma, no son paradigmas en sí)

Conviene no confundirlos con los paradigmas: son **estrategias de cómo se reutiliza o reparte el aprendizaje**, y se combinan con supervisado, RL, etc.

- **Transfer learning:** reaprovechar un modelo entrenado en una tarea/dominio para otra (la base del *fine-tuning*).
- **Meta-learning ("aprender a aprender"):** entrenar para **adaptarse rápido** a tareas nuevas con pocos ejemplos → habilita **few-shot / zero-shot**.
- **Aprendizaje continuo / lifelong:** aprender tareas en secuencia sin **olvidar** las anteriores (*catastrophic forgetting*).
- **Online learning vs batch learning:** actualizar el modelo dato a dato y en streaming, frente a hacerlo sobre un lote fijo. (Ojo: "online" aquí significa *en streaming*, distinto del "online" del RL, que significa *interactuando con el entorno*.)

> Por qué importa para el FL: la **personalización federada** (sección 7.3) es esencialmente *transfer / multi-task learning* aplicado por cliente; y la heterogeneidad no-IID se ataca con ideas de *meta-learning*.

---

### 3.8 Tabla comparativa de los paradigmas

| | Supervisado | No supervisado | Auto-supervisado | Por refuerzo |
|---|---|---|---|---|
| **Señal de aprendizaje** | Etiquetas humanas correctas | Estructura de los datos | Etiquetas derivadas del propio dato | Recompensa (evaluativa, retardada, escasa) |
| **Tipo de feedback** | Instructivo | Ninguno directo | Instructivo (auto-generado) | **Evaluativo** |
| **Datos** | Dataset estático etiquetado | Dataset estático sin etiquetar | Dataset sin etiquetar (etiqueta implícita) | Experiencia secuencial (s, a, r, s') |
| **Coste de etiquetado** | Alto | Nulo | Nulo | N/A (necesita entorno/recompensa) |
| **Pregunta que responde** | "¿Qué es esto?" | "¿Cómo se organiza esto?" | "¿Puedo predecir una parte con el resto?" | "¿Qué debo hacer?" |
| **Ejemplo estrella** | Clasificación de imágenes | Clustering de clientes | Preentrenar un LLM | Robot que aprende a caminar |

---

## 4. Tipos de algoritmos (el catálogo)

### 4.1 Algoritmos supervisados
- **Clásicos:** regresión lineal/logística, árboles de decisión, Random Forest, Gradient Boosting (XGBoost), SVM, k-NN.
- **Deep:** MLP (perceptrón multicapa), **CNN** (imágenes), **RNN/LSTM/GRU** (secuencias), **Transformers** (lenguaje, y hoy casi todo).

> 🔑 **Nota FL — qué algoritmos son "federables" de forma natural:** una red neuronal aprende ajustando **pesos y sesgos** mediante descenso de gradiente. Esos pesos son exactamente lo que cada cliente entrena localmente y sube al servidor en una ronda de FedAvg (ver 5.2). Por eso los algoritmos **paramétricos** —regresión logística, MLP, CNN, Transformers— son candidatos ideales para FL: su "conocimiento" cabe en un vector de parámetros que se puede **promediar**.
>
> **Contraejemplo — k-NN es un mal candidato para FL estándar:** k-NN no aprende ninguna ecuación ni ajusta parámetros; simplemente **memoriza** todo el dataset de entrenamiento y busca vecinos cercanos en el momento de predecir. No hay "pesos" que subir al servidor: para que un servidor calcule distancias necesitaría los datos crudos de los clientes (violando la privacidad) o protocolos de cómputo seguro multiparte (SMPC) muy costosos. Es un buen ejemplo de que **no todo algoritmo se federa igual de bien**: la federabilidad depende de si el modelo es paramétrico (agregable) o basado en instancias (no agregable sin fugar datos).

### 4.2 Algoritmos no supervisados
- **Clustering:** K-Means, DBSCAN, clustering jerárquico.
- **Reducción de dimensionalidad:** PCA, t-SNE, UMAP.
- **Deep:** Autoencoders (y VAE), GANs y modelos de difusión (generativos), modelos de auto-supervisión contrastiva (SimCLR).

### 4.3 Algoritmos de RL (los que luego federaremos)

```
RL profundo (Deep RL)
├── Basados en VALOR (value-based) ── aprenden Q(s,a), política implícita
│   ├── Q-Learning (tabular, clásico)
│   ├── DQN (Deep Q-Network) ── Q-Learning + red profunda + replay buffer
│   └── Variantes: Double DQN, Dueling DQN, Rainbow
│
├── Basados en POLÍTICA (policy-based) ── aprenden π directamente
│   ├── REINFORCE (policy gradient básico)
│   └── PPO, TRPO (con restricciones de estabilidad) ── PPO es el estándar de facto
│
├── ACTOR-CRITIC ── híbridos: un actor (política) + un crítico (valor)
│   ├── A2C / A3C
│   ├── DDPG, TD3 (acciones continuas, off-policy)
│   └── SAC (Soft Actor-Critic, máxima entropía)
│
└── RL OFFLINE ── aprender de datasets fijos
    ├── BCQ, CQL (Conservative Q-Learning)
    └── Decision Transformer (RL como modelado de secuencias)
```

**Regla mental rápida:**
- ¿Acciones discretas y pocas? → DQN.
- ¿Quieres robustez y simplicidad? → PPO.
- ¿Acciones continuas (robótica, control)? → SAC / TD3.
- ¿Solo tienes datos históricos, no puedes interactuar? → RL offline (CQL).

### 4.4 Métodos de ensamblado (transversales: combinan varios modelos base)

No son un paradigma (la señal de aprendizaje no cambia) ni una arquitectura: son una **meta-estrategia para construir un modelo fuerte combinando varios "débiles"**. Por eso varios de los algoritmos de 4.1 (Random Forest, XGBoost) son en realidad *ensembles*.

```
Ensembles
├── Bagging (Bootstrap Aggregating) ── modelos en PARALELO, independientes
│       sobre submuestras (bootstrap) → se PROMEDIA / vota
│       → reduce VARIANZA (combate el overfitting). Ej.: Random Forest
├── Boosting ── modelos en SECUENCIA, cada uno CORRIGE los errores del anterior
│       → reduce SESGO (combate el underfitting). Ej.: AdaBoost, Gradient Boosting, XGBoost
└── Stacking ── un meta-modelo aprende a combinar las salidas de los modelos base
```

**Regla mental:** *bagging ataca la varianza, boosting ataca el sesgo* (enlaza con el compromiso sesgo–varianza de la Sección 3.1).

> **Por qué aparece en un documento sobre FL:** el bagging es el **antepasado conceptual de la agregación federada**. Bagging = repartir datos en subconjuntos → entrenar modelos locales → **agregar**; FedAvg hace lo mismo, pero con datos de **dueños distintos**. La diferencia técnica: el bagging promedia *predicciones*, mientras que FedAvg promedia *pesos* (la variante de FL que promedia predicciones se llama **ensemble federado**). El boosting también se ha federado: **SecureBoost / federated XGBoost** (ver Sección 8).

---

## 5. El eje "DÓNDE": del entrenamiento centralizado al federado

### 5.1 El problema
El DL necesita muchos datos, pero los datos valiosos suelen estar:
- **Repartidos** entre organizaciones (hospitales, bancos, fábricas) o dispositivos (móviles, vehículos).
- **Protegidos** por privacidad (RGPD en Europa), competencia comercial o regulación sectorial.

➡️ **No se pueden centralizar.** Solución: **mover el modelo a los datos, no los datos al modelo.**

### 5.2 ¿Qué es el aprendizaje federado (Federated Learning, FL)?

Es un paradigma de entrenamiento **colaborativo y distribuido** en el que múltiples participantes (**clientes**: hospitales, móviles, empresas...) entrenan un **modelo global compartido** sin que sus datos locales salgan nunca de su poder.

**El ciclo de una ronda federada (algoritmo FedAvg, el fundacional):**

```
       ┌──────────── Servidor (coordinador) ────────────┐
       │   1. Envía el modelo global wₜ a los clientes  │
       └────────┬───────────────┬───────────────┬───────┘
                ▼               ▼               ▼
           Cliente A       Cliente B       Cliente C
        2. Cada uno entrena LOCALMENTE con SUS datos
           (varias épocas de SGD) → wᴬ, wᴮ, wᶜ
                │               │               │
                └───────────────┼───────────────┘
                                ▼
       3. Suben SOLO los pesos/gradientes (nunca los datos)
       4. El servidor AGREGA: wₜ₊₁ = Σ (nₖ/n) · wᵏ  (media ponderada)
       5. Volver al paso 1 hasta converger
```

> **Puente con lo que ya sabes:** la idea de *"entrenar submodelos sobre porciones de datos y luego agregar"* no es nueva. El **bagging** (Sección 4.4) ya lo hacía dentro de una sola máquina, sobre submuestras de un mismo dataset. El FL lleva esa misma intuición al caso difícil: porciones que pertenecen a **dueños distintos que no pueden compartirlas**. Cambia *qué* se agrega (pesos en vez de predicciones) y *por qué* (privacidad/soberanía, no solo reducir varianza), pero el esqueleto es el mismo.

### 5.3 Tipos de aprendizaje federado (según cómo se reparten los datos)

| Tipo | Qué comparten los clientes | Ejemplo |
|---|---|---|
| **Horizontal (HFL)** | Mismas **features**, distintos **individuos** | Dos hospitales con los mismos tipos de análisis pero pacientes distintos |
| **Vertical (VFL)** | Mismos **individuos**, distintas **features** | Un banco y una telco con los mismos clientes pero datos diferentes |
| **Federated Transfer Learning** | Poco solapamiento en ambos ejes | Organizaciones de sectores distintos que comparten conocimiento de modelo |

También se distingue por escala:
- **Cross-device:** millones de dispositivos pequeños y poco fiables (móviles — el teclado de Google fue el caso pionero).
- **Cross-silo:** pocas organizaciones grandes y estables (hospitales, empresas). **Es el escenario natural de los data spaces.**

### 5.4 Retos del FL (y sus contramedidas)

| Reto | Descripción | Mitigación |
|---|---|---|
| **Datos no-IID** | Cada cliente tiene distribuciones distintas (heterogeneidad estadística) | FedProx, SCAFFOLD, personalización |
| **Privacidad residual** | Los gradientes pueden filtrar información (ataques de inversión) | **Privacidad diferencial (DP)**, **agregación segura (SMPC)**, **cifrado homomórfico** |
| **Comunicación** | Subir/bajar modelos es costoso | Compresión, cuantización, menos rondas |
| **Clientes maliciosos** | Envenenamiento del modelo (poisoning) | Agregación robusta (Krum, mediana) |
| **Heterogeneidad de sistemas** | Dispositivos con capacidades distintas | Selección de clientes, modelos adaptativos |

> 🔑 **FL ≠ privacidad perfecta por sí solo.** FL minimiza la exposición (los datos no viajan), pero necesita técnicas adicionales (DP, agregación segura) para garantías fuertes.

> ⚠️ **El no-IID no es solo un reto de datos, también es un reto de elección de modelo.** El video asume implícitamente datos centralizados y balanceados; en FL cada cliente tiene su propio sesgo. Si el algoritmo elegido es propenso al overfitting o sensible al ruido local —un **árbol de decisión individual** sin podar, o un **SVM sin un margen robusto**—, el modelo local diverge con fuerza del óptimo global, y esa divergencia se propaga (y se amplifica) en la agregación del servidor. Cuanto más "memoriza" un modelo las particularidades de su cliente, peor agrega. Por eso, cuando la heterogeneidad no-IID es alta, conviene combinar las mitigaciones de la tabla anterior (FedProx, SCAFFOLD, personalización) con arquitecturas más estables ante el ruido (ensembles como Random Forest, o redes profundas con regularización) en lugar de modelos clásicos frágiles.

---

## 6. Data Spaces (espacios de datos): la capa de gobernanza

### 6.1 ¿Qué es un data space?
Es una **infraestructura federada de compartición de datos** entre organizaciones, basada en estándares comunes de **soberanía, confianza e interoperabilidad**. Es un concepto impulsado fuertemente en Europa (iniciativas como **Gaia-X**, **IDSA — International Data Spaces Association**, y los espacios de datos sectoriales europeos: salud/EHDS, movilidad, industria/Catena-X, agro...).

**Principio fundamental: soberanía del dato.** Cada participante:
- Mantiene el **control total** sobre sus datos.
- Define **políticas de uso** (quién, para qué, cuánto tiempo) mediante contratos digitales.
- Comparte datos (o resultados de cómputo) **solo bajo esas condiciones**, a través de **conectores** estandarizados (ej. Eclipse Dataspace Connector).

### 6.2 Componentes típicos
- **Conectores:** software estandarizado en cada participante que negocia y ejecuta el intercambio.
- **Catálogo / broker:** descubrimiento de qué datos/servicios existen.
- **Identidad y confianza:** certificación de participantes.
- **Contratos y políticas de uso (usage control):** reglas ejecutables sobre el uso del dato.
- **Clearing house:** registro auditable de las transacciones.

### 6.3 La sinergia perfecta: Data Spaces + Federated Learning

| El data space aporta | El FL aporta |
|---|---|
| Gobernanza, contratos, identidad, confianza | El mecanismo técnico para entrenar sin mover datos |
| "¿Bajo qué condiciones colaboramos?" | "¿Cómo colaboramos computacionalmente?" |

En un data space, **el "data sharing" no implica necesariamente mover datos crudos**: puede ser *compute-to-data* (el cómputo viaja al dato). El FL es exactamente eso aplicado al entrenamiento de modelos: **lo que se comparte son actualizaciones de modelo, bajo las políticas de uso del espacio de datos.**

```
Data Space (gobernanza)
├── Participante A (hospital)  ──┐
├── Participante B (hospital)  ──┼── Conectores + contratos de uso
├── Participante C (aseguradora)─┘
└── Servicio de FL (orquestador federado)
    └── Rondas de FedAvg respetando las políticas del espacio
```

---

## 7. Federated Reinforcement Learning (FRL): tu objetivo final

### 7.1 La idea
Combinar ambos mundos: **varios agentes de RL**, cada uno interactuando con **su propio entorno local** (o con su propio dataset de experiencias), que colaboran para aprender una **política compartida mejor**, sin compartir sus trayectorias de experiencia (que pueden ser sensibles: datos de conducción, consumos energéticos, comportamiento de pacientes...).

```
   Entorno A          Entorno B          Entorno C
      ▲ ▼                ▲ ▼                ▲ ▼
   Agente A           Agente B           Agente C
   (política πᴬ)      (política πᴮ)      (política πᶜ)
      │                  │                  │
      └────── suben pesos de la política/Q-network ──────┐
                                                          ▼
                                              Servidor: agrega → π global
                                              y la redistribuye
```

### 7.2 Qué se federa en FRL
Depende del algoritmo de RL elegido:
- **Con DQN:** se federan los pesos de la **Q-network** (Horizontal FRL clásico).
- **Con PPO / A2C:** se federan los pesos del **actor** (y opcionalmente el crítico).
- **Con SAC/TD3:** actor + crítico(s).
- **Con RL offline:** cada cliente entrena con su dataset local de trayectorias (CQL, etc.) y se agregan los modelos → muy relevante en dominios regulados.

**Dos grandes familias:**
- **HFRL (Horizontal FRL):** agentes en entornos *similares* (misma tarea, distintos datos). Ej.: muchos vehículos autónomos aprendiendo a conducir, cada uno en sus carreteras.
- **VFRL (Vertical FRL):** los agentes observan **partes distintas del mismo entorno** (observación parcial repartida). Menos maduro, más complejo.

### 7.3 Retos específicos del FRL (además de los del FL)
1. **Heterogeneidad de entornos:** las dinámicas locales difieren (no-IID llevado al extremo: distintos MDPs).
2. **Exploración descoordinada:** cada agente explora por su cuenta; agregar políticas en fases distintas de aprendizaje puede desestabilizar.
3. **Estabilidad de la agregación:** promediar pesos de políticas RL es más frágil que en supervisado (las pérdidas de RL son no estacionarias).
4. **Recompensas distintas o privadas** entre clientes.
5. **Sincronía:** episodios de distinta duración → agregación asíncrona.

**Estrategias habituales:** agregar cada N episodios, ponderar por desempeño, federar solo partes del modelo (capas de representación compartidas + cabezas locales = **personalización**), usar algoritmos off-policy (reutilizan experiencia y toleran mejor la mezcla).

### 7.4 Casos de uso de FRL en data spaces
- **Energía:** comunidades energéticas optimizando consumo/almacenamiento sin revelar perfiles de consumo (espacio de datos de energía).
- **Movilidad:** flotas/vehículos aprendiendo políticas de conducción o de gestión de tráfico (espacio de datos de movilidad).
- **Industria 4.0:** robots/fábricas optimizando procesos sin exponer secretos de producción (Catena-X, Manufacturing-X).
- **Sanidad:** políticas de tratamiento dinámico aprendidas con **RL offline federado** sobre historiales clínicos de varios hospitales (EHDS).
- **Telecomunicaciones / edge:** asignación de recursos de red en estaciones base.

---

## 8. Respuesta directa: ¿qué sistemas de aprendizaje se pueden implementar con data sharing federado?

**Prácticamente todos los paradigmas se pueden federar.** Lo que cambia es la madurez y la dificultad:

| Paradigma | ¿Federable? | Madurez | Qué se comparte | Ejemplo en data space |
|---|---|---|---|---|
| **Supervisado** | ✅ Sí | ⭐⭐⭐⭐⭐ (el caso estándar de FL) | Pesos/gradientes del modelo | Diagnóstico médico multi-hospital |
| **No supervisado** | ✅ Sí | ⭐⭐⭐ | Pesos (autoencoders, clustering federado) | Detección de anomalías/fraude entre bancos |
| **Auto-supervisado** | ✅ Sí | ⭐⭐⭐ | Pesos del modelo de representación | Pre-entrenar modelos de lenguaje/series sobre datos privados |
| **Semi-supervisado** | ✅ Sí | ⭐⭐⭐ | Pesos + pseudo-etiquetado local | Clientes con pocas etiquetas |
| **RL online** | ✅ Sí (HFRL) | ⭐⭐ | Pesos de política/Q-network | Flotas, robots, redes de telecom |
| **RL offline** | ✅ Sí | ⭐⭐ (creciendo rápido) | Pesos entrenados sobre trayectorias locales | Políticas clínicas, conducción |
| **Analítica clásica (GBM, árboles)** | ✅ Sí | ⭐⭐⭐⭐ | Estadísticos/histogramas (SecureBoost, federated XGBoost) | Scoring crediticio banco+telco (VFL) |

**Y técnicas complementarias que conviene conocer:**
- **Federated Analytics:** estadísticas agregadas sin entrenar modelos.
- **Split Learning:** la red se parte entre cliente y servidor (útil en VFL).
- **Knowledge Distillation federada (FedMD):** se comparten *predicciones* sobre un dataset público en lugar de pesos → permite modelos heterogéneos.
- **Transfer learning federado:** para participantes con datos muy distintos.

---

## 9. Tu modelo mental en una página (resumen)

1. **Tres ejes independientes, no una jerarquía:** un sistema de ML se define eligiendo en ① **CÓMO** aprende (paradigma), ② **CON QUÉ** se implementa (clásico / ensemble / **DL**), y ③ **DÓNDE** están los datos (centralizado / distribuido / federado). El DL es solo una opción del eje ②, no el contenedor de los paradigmas. **El FRL = `RL` (①) + `redes profundas` (②) + `federado` (③).**
2. **El espectro de la supervisión:** lo que separa a los paradigmas es **de dónde sale la señal de aprendizaje**: supervisado (etiquetas humanas) → semi/débilmente sup. (etiquetas escasas o ruidosas) → auto-supervisado (etiqueta derivada del dato; base de los *foundation models*) → no supervisado (solo estructura). El **RL** está en otro eje: feedback **evaluativo** (recompensa) por interacción, con sus ejes online/offline y on/off-policy. Y lo real suele ser **mixto** (imitation learning, RLHF, SSL+fine-tuning).
3. **Algoritmos de RL:** valor (DQN), política (PPO), actor-critic (SAC), offline (CQL).
4. **FL = "el modelo viaja, los datos no":** rondas de entrenamiento local + agregación (FedAvg). Tipos: horizontal/vertical, cross-device/cross-silo. Retos: no-IID, privacidad, comunicación.
5. **Data spaces = la gobernanza** (soberanía, contratos, conectores) que hace viable la colaboración entre organizaciones; el FL es su mecanismo técnico natural de *compute-to-data*.
6. **FRL = agentes RL locales + agregación federada de políticas.** Hereda los retos del FL y añade los de RL (entornos heterogéneos, estabilidad). El **RL offline federado** es la vía más prometedora en dominios regulados.
7. **Casi todo es federable**; el supervisado es lo maduro, el FRL es la frontera.

---

## 10. Ruta de aprendizaje sugerida (orden de estudio)

1. Fundamentos de redes neuronales y backpropagation.
2. RL básico: MDPs, Q-Learning tabular → DQN → PPO (libro de referencia: *Sutton & Barto, Reinforcement Learning: An Introduction*).
3. RL offline (paper survey de Levine et al., *Offline Reinforcement Learning*).
4. FL: paper original de FedAvg (McMahan et al., 2017) + survey *Advances and Open Problems in Federated Learning* (Kairouz et al.).
5. Frameworks prácticos: **Flower**, TensorFlow Federated, FedML (FL) + Gymnasium, Stable-Baselines3 (RL). Combínalos para un prototipo de FRL.
6. Data spaces: documentación de **IDSA** (Reference Architecture Model) y **Gaia-X**; Eclipse Dataspace Components para los conectores.
7. Surveys de FRL: *Federated Reinforcement Learning: Techniques, Applications, and Open Challenges* (Qi et al.).
