---
title: Índice maestro de Deep Learning
tags: [deep-learning, mapa, indice]
status: revisado
updated: 2026-08-08
---
# Índice maestro de Deep Learning



> Índice de términos y conceptos para ubicar cualquier paper.
> La numeración es estable: úsala como identificador de bloque.

---

# ÁRBOL COMPLETO

```none
DEEP LEARNING
│
├── 0. UBICACIÓN — qué es y dónde encaja
│    ├── 0.1  Qué es el deep learning
│    ├── 0.2  Taxonomía: de la IA al DL
│    ├── 0.3  Los tres ejes de un sistema de ML
│    └── 0.4  Cómo usar este índice
│
├── 1. LA NEURONA BIOLÓGICA
│    ├── 1.1  Anatomía de la neurona
│    ├── 1.2  La señal eléctrica
│    ├── 1.3  La sinapsis
│    ├── 1.4  Codificación de la información
│    ├── 1.5  Plasticidad: cómo aprende el cerebro
│    ├── 1.6  Organización en circuitos
│    ├── 1.7  De la neurona biológica a la artificial
│    ├── 1.8  Dónde se rompe la analogía
│    └── 1.9  Corrientes de vuelta: NeuroAI
│
├── 2. FUNDAMENTOS MATEMÁTICOS
│    ├── 2.1  Álgebra lineal
│    ├── 2.2  Cálculo y diferenciación automática
│    ├── 2.3  Probabilidad y estadística
│    ├── 2.4  Teoría de la información
│    ├── 2.5  Optimización
│    └── 2.6  Computación numérica
│
├── 3. ANATOMÍA DE UNA RED
│    ├── 3.1  La unidad básica
│    ├── 3.2  Funciones de activación
│    ├── 3.3  Topología y flujo de información
│    ├── 3.4  Normalización
│    ├── 3.5  Inicialización
│    ├── 3.6  Regularización
│    └── 3.7  Entrada y salida
│
├── 4. MECÁNICA DEL APRENDIZAJE
│    ├── 4.1  El bucle de entrenamiento
│    ├── 4.2  Funciones de pérdida
│    ├── 4.3  Backpropagation
│    ├── 4.4  Optimizadores
│    ├── 4.5  Tasa de aprendizaje
│    ├── 4.6  Hiperparámetros
│    ├── 4.7  Patologías del entrenamiento
│    └── 4.8  Evaluación y metodología
│
├── 5. DATOS EN REJILLA — imágenes (CNN)
│    ├── 5.1  Del perceptrón al MLP
│    ├── 5.2  La convolución
│    ├── 5.3  Genealogía de las CNN
│    ├── 5.4  Tareas de visión
│    └── 5.5  Transferencia
│
├── 6. DATOS EN SECUENCIA — lenguaje (RNN → Transformer)
│    ├── 6.1  Redes recurrentes
│    ├── 6.2  El problema del gradiente y sus parches
│    ├── 6.3  Seq2seq
│    ├── 6.4  Mecanismo de atención
│    ├── 6.5  El Transformer
│    ├── 6.6  Las tres familias de Transformer
│    ├── 6.7  Eficiencia y evolución
│    └── 6.8  El Transformer fuera del texto
│
├── 7. CREAR DATOS NUEVOS — modelos generativos
│    ├── 7.1  El problema generativo
│    ├── 7.2  Autoencoders y VAE
│    ├── 7.3  Redes adversariales (GAN)
│    ├── 7.4  Modelos de flujo
│    ├── 7.5  Modelos autoregresivos
│    ├── 7.6  Modelos de difusión
│    └── 7.7  Multimodalidad
│
├── 8. ESTRUCTURA IRREGULAR — grafos y simetrías
│    ├── 8.1  Datos no euclidianos
│    ├── 8.2  Redes sobre grafos (GNN)
│    ├── 8.3  Redes equivariantes
│    ├── 8.4  Geometric DL como marco unificador
│    └── 8.5  Aplicaciones científicas
│
├── 9. MODELOS BASE Y ESCALA
│    ├── 9.1  El cambio de modelo de trabajo
│    ├── 9.2  Pre-entrenamiento auto-supervisado
│    ├── 9.3  Leyes de escalado
│    ├── 9.4  Ciclo de vida de un modelo grande
│    ├── 9.5  Adaptación eficiente (PEFT)
│    ├── 9.6  Compresión e inferencia
│    ├── 9.7  Uso en inferencia
│    └── 9.8  Comportamiento emergente
│
├── 10. REFUERZO Y ALINEAMIENTO
│    ├── 10.1 Marco formal
│    ├── 10.2 Ejes de clasificación
│    ├── 10.3 Familias de algoritmos
│    ├── 10.4 Aprendizaje a partir de preferencias humanas
│    └── 10.5 Agentes
│
├── 11. ARQUITECTURAS DE ENTRENAMIENTO — ¿dónde están los datos?
│    ├── 11.1  Por qué este eje es independiente
│    ├── 11.2  Centralizado
│    ├── 11.3  Distribuido (un solo dueño)
│    ├── 11.4  Aprendizaje federado
│    ├── 11.5  Aprendizaje descentralizado
│    ├── 11.6  Entrenamiento descentralizado de modelos grandes
│    ├── 11.7  Split learning y aprendizaje en el borde
│    ├── 11.8  Privacidad y seguridad
│    ├── 11.9  Data spaces y gobernanza del dato
│    ├── 11.10 Evaluación, benchmarks y frameworks
│    └── 11.11 Preguntas abiertas de investigación
│
├── RUTA DE APRENDIZAJE — 12 bloques ejecutables
├── ANEXO A — TEORÍA ABIERTA
├── ANEXO B — CRONOLOGÍA
└── ANEXO C — TÉRMINOS DE INGENIERÍA (compacto)
```

---

# PARTE 0 — UBICACIÓN

## 0.1 Qué es el deep learning
- **0.1.1 Definición operativa** — programación diferenciable: un grafo computacional diferenciable parametrizado, ajustado por descenso de gradiente vía diferenciación automática.
- **0.1.2 Definición insuficiente** — "redes con muchas capas": circular, no explica nada.
- **0.1.3 Sesgo inductivo** — supuestos sobre la estructura de los datos codificados en la arquitectura; es lo único que distingue una familia de otra.
- **0.1.4 Representation learning** — la red aprende las características en vez de recibirlas diseñadas a mano.
- **0.1.5 Composicionalidad jerárquica** — capas sucesivas construyen abstracciones de complejidad creciente.

## 0.2 Taxonomía: de la IA al DL

```none
Inteligencia Artificial (IA)
├── IA simbólica (GOFAI) ....... reglas, lógica, búsqueda, planificación
│      └── vigente en SAT/SMT, planners, sistemas neuro-simbólicos
└── Machine Learning (ML) ...... aprender de los datos en vez de programar reglas
        └── Deep Learning (DL) .. una FAMILIA DE MODELO dentro de ML,
                                  no un nivel superior
```

## 0.3 Los tres ejes de un sistema de ML

### 0.3.1 Eje ① — Paradigma: de dónde viene la señal de aprendizaje
- **Supervisado** — pares (x, y) etiquetados por humanos.
- **Semi-supervisado** — pocas etiquetas + mucho dato sin etiquetar.
- **Débilmente supervisado** — etiquetas ruidosas, imprecisas o a nivel de bolsa.
- **Auto-supervisado** — la etiqueta se deriva del propio dato.
  - *Predictivo/generativo* — next-token, masked modeling.
  - *Contrastivo* — SimCLR, MoCo, CLIP, InfoNCE.
  - *Destilación no contrastiva* — BYOL, DINO.
- **No supervisado** — clustering, reducción de dimensión, estimación de densidad.
- **Por refuerzo** — feedback evaluativo por interacción (→ Parte 10).
- **Híbridos** — imitation learning, RLHF, active learning, curriculum, SSL + fine-tuning.

### 0.3.2 Eje ② — Familia de modelo: qué tipo de función ajustas

```none
├── Clásicos ............ regresión lineal/logística, SVM y kernels, k-NN,
│                         Naive Bayes, árboles de decisión
├── Probabilísticos ..... HMM, CRF, redes bayesianas, GMM, procesos gaussianos
├── Ensembles
│     ├── Bagging (paralelo, ↓varianza) ....... Random Forest
│     ├── Boosting (secuencial, ↓sesgo) ....... AdaBoost, GBM, XGBoost, LightGBM
│     └── Stacking .......................... meta-modelo que combina base learners
└── Deep Learning ....... MLP, CNN, RNN/LSTM, Transformer, GNN, difusión
```

### 0.3.3 Eje ③ — Arquitectura de entrenamiento: dónde están los datos
- **Centralizado · Distribuido · Federado · Descentralizado** → Parte 11.

## 0.4 Cómo usar este índice
- **0.4.1 No es lineal** — sistema de coordenadas, no temario secuencial.
- **0.4.2 Partes 5–8 en paralelo** — cuatro formas de dato (rejilla, secuencia, generación, estructura irregular), no cuatro etapas.
- **0.4.3 Parte 2 en espiral** — las matemáticas se revisitan, no se agotan antes de empezar.
- **0.4.4 Parte 11 ortogonal** — cualquier modelo de las partes 5–9 puede entrenarse en cualquier modalidad de la 11.
- **0.4.5 Criterio de hueco** — si no sabes situar un paper en los tres ejes de 0.3, ahí está tu laguna.

---

# PARTE 1 — LA NEURONA BIOLÓGICA

## 1.1 Anatomía de la neurona
- **1.1.1 Soma** — el cuerpo celular; integra las señales recibidas.
- **1.1.2 Dendritas** — el árbol de entrada; reciben contactos de miles de neuronas.
- **1.1.3 Axón** — el cable de salida, único por neurona.
- **1.1.4 Mielina y nodos de Ranvier** — aislamiento que acelera la conducción (saltatoria).
- **1.1.5 Terminal axónico y botón sináptico** — donde se libera el neurotransmisor.
- **1.1.6 Escala** — ~86 000 millones de neuronas, ~10³–10⁴ sinapsis por neurona.
- **1.1.7 Tipos neuronales** — piramidales, interneuronas, estrelladas, de Purkinje.
- **1.1.8 Glía** — astrocitos, oligodendrocitos, microglía; modulan la transmisión, no son soporte pasivo.

## 1.2 La señal eléctrica
- **1.2.1 Potencial de reposo** — ~−70 mV; gradientes iónicos y bomba Na⁺/K⁺.
- **1.2.2 Canales iónicos** — dependientes de voltaje o de ligando.
- **1.2.3 Potenciales postsinápticos** — EPSP (despolariza) e IPSP (hiperpolariza).
- **1.2.4 Integración espacial y temporal** — sumar entradas de distintas dendritas e instantes.
- **1.2.5 Umbral de disparo** — ~−55 mV; decide el cono axónico.
- **1.2.6 Potencial de acción** — todo-o-nada: despolarización, repolarización, hiperpolarización.
- **1.2.7 Periodo refractario** — absoluto y relativo; impone frecuencia máxima.
- **1.2.8 Ley del todo o nada** — la información no está en la amplitud, sino en cuándo y cuántos.
- **1.2.9 Modelo de Hodgkin-Huxley (1952)** — las ecuaciones que describen lo anterior.
- **1.2.10 Modelos simplificados** — integrate-and-fire, Izhikevich.
- **1.2.11 Cómputo dendrítico** — las dendritas hacen operaciones no lineales locales; una piramidal se aproxima mejor por una red de 2–3 capas que por una unidad.

## 1.3 La sinapsis
- **1.3.1 Sinapsis química** — vesículas, hendidura sináptica, receptores.
- **1.3.2 Sinapsis eléctrica** — uniones gap; rápidas y bidireccionales, minoritarias.
- **1.3.3 Neurotransmisores** — glutamato, GABA, dopamina, serotonina, acetilcolina.
- **1.3.4 Receptores** — ionotrópicos (AMPA, NMDA, GABA-A) y metabotrópicos.
- **1.3.5 El receptor NMDA** — detector de coincidencia; base molecular de Hebb.
- **1.3.6 Neuromodulación** — cambia el régimen de toda una red, no un peso concreto.
- **1.3.7 Peso sináptico** — la eficacia de la conexión; lo único que el modelo artificial conservó.
- **1.3.8 Estocasticidad de la liberación** — la sinapsis es ruidosa por diseño.
- **1.3.9 Ley de Dale** — una neurona es excitatoria o inhibitoria, no ambas.

## 1.4 Codificación de la información
- **1.4.1 Rate coding** — información en la frecuencia media de disparo.
- **1.4.2 Temporal / spike coding** — información en el instante exacto del pico.
- **1.4.3 Codificación poblacional** — el mensaje está en el patrón conjunto.
- **1.4.4 Codificación dispersa** — pocas neuronas activas; eficiencia y separabilidad.
- **1.4.5 Variabilidad neuronal** — el mismo estímulo produce respuestas distintas.
  - *Ruido vs señal* — parte de la variabilidad codifica incertidumbre, no es error.
  - *Correlaciones de ruido* — la variabilidad compartida limita la información poblacional.
  - *Hipótesis del muestreo* — el cerebro representaría distribuciones muestreando.
- **1.4.6 Actividad espontánea** — el cerebro nunca está en reposo.
- **1.4.7 Campos receptivos** — Hubel & Wiesel: células simples y complejas; ancestro de la convolución.
- **1.4.8 Predictive coding** — se transmitiría error de predicción, no la señal completa.

## 1.5 Plasticidad: cómo aprende el cerebro
- **1.5.1 Regla de Hebb (1949)** — la coactivación refuerza la conexión.
- **1.5.2 LTP y LTD** — potenciación y depresión a largo plazo; sustrato experimental de Hebb.
- **1.5.3 STDP** — depende del orden temporal: pre antes de post refuerza; al revés debilita.
- **1.5.4 Plasticidad homeostática** — escalado sináptico que estabiliza sin borrar.
- **1.5.5 Plasticidad estructural** — crecimiento y poda de sinapsis y espinas.
- **1.5.6 Neurogénesis** — limitada en adultos.
- **1.5.7 Periodos críticos** — ventanas de plasticidad máxima en el desarrollo.
- **1.5.8 Plasticidad intermodal (cross-modal)** — el córtex se reasigna cuando falta una entrada.
  - *Caso canónico* — el córtex visual de personas ciegas procesa braille y lenguaje.
  - *Implicación* — el sustrato cortical parece relativamente genérico: la función la fija la entrada.
  - *Paralelo en DL* — la misma arquitectura sirve para texto, imagen, audio y proteínas.
- **1.5.9 Consolidación** — de corto a largo plazo; papel del sueño y del hipocampo.
- **1.5.10 Replay hipocampal** — reproducción de secuencias en reposo; inspiración del experience replay en RL.

## 1.6 Organización en circuitos
- **1.6.1 Córtex y sus seis capas** — patrón de conectividad estereotipado.
- **1.6.2 Columnas corticales** — hipótesis del microcircuito repetido.
- **1.6.3 Jerarquía visual** — V1 → V2 → V4 → IT; la analogía más directa con una CNN.
- **1.6.4 Feedforward, lateral y feedback** — el feedback descendente es masivo y el DL apenas lo usa.
- **1.6.5 Balance excitación/inhibición** — condición de estabilidad.
- **1.6.6 Inhibición lateral** — competencia local; produce esparsidad y realce de contrastes.
- **1.6.7 Oscilaciones y sincronía** — theta, gamma; coordinación entre áreas.
- **1.6.8 Modularidad y mundo pequeño** — topología de la conectividad.
- **1.6.9 Eficiencia energética** — ~20 W para todo el cerebro.

## 1.7 De la neurona biológica a la artificial
- **1.7.1 McCulloch & Pitts (1943)** — la neurona como unidad lógica de umbral.
- **1.7.2 Qué se conservó** — suma ponderada, umbral no lineal, peso modificable, conectividad masiva.
- **1.7.3 Qué se descartó** — tiempo, picos, química, morfología dendrítica, neuromodulación, glía, energía.

```none
COMPARATIVA
                        NEURONA BIOLÓGICA            UNIDAD ARTIFICIAL
  señal .............   picos discretos en el        número real continuo
                        tiempo
  salida ............   todo o nada + frecuencia     valor real (activación)
  tiempo ............   dinámica continua real       ausente o discretizado
  cómputo dendrítico    no lineal, local             ninguno (suma simple)
  aprendizaje .......   local (Hebb, STDP)           global (backprop)
  señal de error ....   local / neuromoduladora      gradiente propagado hacia atrás
  pesos .............   ~10⁴ por neurona, ruidosos   exactos y deterministas
  signo .............   excitatoria O inhibitoria    un peso puede cambiar de signo
  energía ...........   ~20 W (cerebro completo)     10⁵–10⁷ W (entrenamiento)
  estructura ........   se reorganiza físicamente    fija durante el entrenamiento
```

## 1.8 Dónde se rompe la analogía
- **1.8.1 Backprop no es biológicamente plausible** — cuatro objeciones:
  - *Weight transport problem* — el backward exige los pesos exactos del forward, en sentido inverso.
  - *Señal de error global* — no hay evidencia de un error escalar distribuido a todo el cerebro.
  - *Separación de fases* — el cerebro no alterna forward y backward limpiamente.
  - *Linealidad del backward* — no corresponde a ninguna vía neural conocida.
- **1.8.2 Feedback alignment** — pesos aleatorios en el backward que aun así aprenden.
- **1.8.3 Predictive coding como algoritmo** — aproxima backprop con cómputo local.
- **1.8.4 Forward-forward (Hinton, 2022)** — dos pasadas hacia delante, sin backward.
- **1.8.5 Target propagation, equilibrium propagation, synthetic gradients** — otras alternativas locales.
- **1.8.6 Conclusión** — el DL se inspiró en el cerebro; no lo modela ni lo explica.

## 1.9 Corrientes de vuelta: NeuroAI
- **1.9.1 Redes de picos (SNN)** — cómo entrenarlas: surrogate gradients.
- **1.9.2 Hardware neuromórfico** — Loihi, SpiNNaker, TrueNorth; cómputo dirigido por eventos.
- **1.9.3 DL como modelo de neurociencia** — CNN entrenadas predicen respuestas de IT.
- **1.9.4 Alineamiento representacional** — RSA, CKA: comparar espacios biológicos y artificiales.
- **1.9.5 Ideas ya adoptadas** — atención, memoria episódica, replay, normalización, dropout, curriculum.
- **1.9.6 Ideas aún no adoptadas** — aprendizaje local, plasticidad continua sin olvido, feedback descendente, eficiencia energética.

---

# PARTE 2 — FUNDAMENTOS MATEMÁTICOS

## 2.1 Álgebra lineal
- **2.1.1 Matriz como aplicación lineal** — no como tabla de números.
- **2.1.2 Producto matricial** — composición de aplicaciones; el batch es un eje extra.
- **2.1.3 Espacios, base, rango, núcleo** — rango efectivo y colapso de representaciones.
- **2.1.4 Normas y productos internos** — L1, L2, L∞, Frobenius; similitud coseno.
- **2.1.5 Proyecciones y ortogonalidad** — base de la atención, PCA y whitening.
- **2.1.6 Autovalores y descomposición espectral** — estabilidad y dinámica recurrente.
- **2.1.7 SVD y aproximación de rango bajo** — PCA, compresión, LoRA.
- **2.1.8 Condicionamiento numérico** — explica gradientes que explotan o desaparecen.
- **2.1.9 Tensores** — forma, broadcasting, contracción, notación de Einstein.

## 2.2 Cálculo y diferenciación automática
- **2.2.1 Gradiente, jacobiana, hessiana**.
- **2.2.2 Regla de la cadena** — backpropagation *es* la regla de la cadena sobre un grafo.
- **2.2.3 Derivadas matriciales** — convención numerador vs denominador.
- **2.2.4 Diferenciación automática**
  - *Modo forward (JVP)* — eficiente si la entrada es de baja dimensión.
  - *Modo reverso (VJP)* — eficiente si la salida lo es; el caso de una loss escalar.
  - *vs simbólica y numérica* — ni expansión de expresiones ni error de truncamiento.
- **2.2.5 Grafo computacional** — nodos = operaciones; define el orden del backward.
- **2.2.6 Serie de Taylor** — justifica el paso de gradiente y los métodos de segundo orden.

## 2.3 Probabilidad y estadística
- **2.3.1 Variables aleatorias, esperanza, varianza, covarianza**.
- **2.3.2 Distribuciones clave** — Bernoulli, categórica, gaussiana, exponencial, Dirichlet.
- **2.3.3 Verosimilitud, MLE y MAP** — toda pérdida es un −log p disfrazado.
- **2.3.4 Teorema de Bayes** — prior, verosimilitud, posterior; regularización = prior.
- **2.3.5 Divergencias** — KL, Jensen-Shannon, Wasserstein.
- **2.3.6 Estimación de Monte Carlo** — muestrear para aproximar esperanzas intratables.
- **2.3.7 Sesgo, varianza y su descomposición**.
- **2.3.8 Maldición de la dimensionalidad y concentración de la medida**.
- **2.3.9 Hipótesis de la variedad** — los datos reales viven en una subvariedad de baja dimensión.

## 2.4 Teoría de la información
- **2.4.1 Entropía** — incertidumbre media de una distribución.
- **2.4.2 Entropía cruzada** — coste de codificar p usando q; la pérdida de clasificación.
- **2.4.3 Divergencia KL** — cross-entropy menos entropía.
- **2.4.4 Información mutua** — objetivo de los métodos contrastivos.
- **2.4.5 Longitud mínima de descripción** — compresión como generalización.

## 2.5 Optimización
- **2.5.1 Convexidad** — y por qué el DL no lo es.
- **2.5.2 Descenso de gradiente** — batch, estocástico, mini-batch.
- **2.5.3 Lipschitz y tasas de convergencia**.
- **2.5.4 Puntos críticos** — en alta dimensión dominan las sillas, no los mínimos locales.
- **2.5.5 Geometría del paisaje de pérdida** — valles, mesetas, mínimos planos vs agudos.
- **2.5.6 Momentum y aceleración**.
- **2.5.7 Métodos de segundo orden** — Newton, cuasi-Newton, K-FAC, Shampoo.
- **2.5.8 Multiplicadores de Lagrange y optimización restringida**.

## 2.6 Computación numérica
- **2.6.1 Punto flotante** — FP32, FP16, BF16, FP8; rango vs precisión.
- **2.6.2 Estabilidad numérica** — log-sum-exp, softmax estable, epsilon en denominadores.
- **2.6.3 Underflow, overflow y NaN**.
- **2.6.4 Determinismo y semillas** — y sus límites en GPU.

---

# PARTE 3 — ANATOMÍA DE UNA RED

## 3.1 La unidad básica
- **3.1.1 Capa afín** — `y = Wx + b`.
- **3.1.2 Pesos y sesgos** — los parámetros aprendibles.
- **3.1.3 Por qué apilar lineales no sirve** — la composición de lineales es lineal.
- **3.1.4 Capa densa** — todo con todo, sin supuestos estructurales.

## 3.2 Funciones de activación
- **3.2.1 Escalón y signo** — el perceptrón original; no diferenciable.
- **3.2.2 Sigmoide y tanh** — saturan y matan el gradiente en profundidad.
- **3.2.3 ReLU (2011)** — gradiente constante en el semieje positivo; el desbloqueo práctico.
- **3.2.4 Variantes** — LeakyReLU, PReLU, ELU, SELU.
- **3.2.5 Suaves modernas** — GELU, SiLU/Swish, Mish.
- **3.2.6 Con puerta** — GLU, GeGLU, SwiGLU; estándar en LLM actuales.
- **3.2.7 De salida** — softmax, sigmoide, identidad.

## 3.3 Topología y flujo de información
- **3.3.1 Profundidad vs anchura**.
- **3.3.2 Conexiones residuales (ResNet, 2015)** — `y = x + f(x)`; hacen entrenable la profundidad.
- **3.3.3 Skip connections** — U-Net: recuperar resolución espacial.
- **3.3.4 Dense connections** — DenseNet.
- **3.3.5 Gating** — Highway Networks, LSTM.
- **3.3.6 Cuellos de botella** — forzar compresión para inducir abstracción.
- **3.3.7 Compartición de pesos** — base de CNN y RNN.
- **3.3.8 Esparsidad y mezcla de expertos (MoE)**.

## 3.4 Normalización
- **3.4.1 Por qué normalizar** — estabilizar escalas y suavizar el paisaje.
- **3.4.2 BatchNorm (2015)** — depende del batch; falla con batch pequeño o secuencias.
- **3.4.3 LayerNorm (2016)** — estándar en Transformers.
- **3.4.4 RMSNorm** — sin centrado, más barato.
- **3.4.5 GroupNorm, InstanceNorm**.
- **3.4.6 Pre-norm vs post-norm** — determina la estabilidad del entrenamiento profundo.
- **3.4.7 Normalización de la entrada** — estandarización, whitening.

## 3.5 Inicialización
- **3.5.1 Por qué importa** — decide si la señal sobrevive en profundidad.
- **3.5.2 Criterio de varianza** — preservar varianza de activaciones y gradientes.
- **3.5.3 Xavier/Glorot** — para tanh y sigmoide.
- **3.5.4 He/Kaiming** — para ReLU.
- **3.5.5 Ortogonal e identidad** — recurrentes y redes muy profundas.
- **3.5.6 Inicialización a cero** — rompe la simetría entre neuronas y con ella el aprendizaje.

## 3.6 Regularización
- **3.6.1 Explícita** — weight decay/L2, L1, dropout, label smoothing.
- **3.6.2 Por datos** — augmentation, mixup, CutMix, RandAugment.
- **3.6.3 Por entrenamiento** — early stopping, ruido del SGD.
- **3.6.4 Arquitectónica** — compartición de pesos, cuellos de botella, restricciones de simetría.
- **3.6.5 Normalización como regularizador**.

## 3.7 Entrada y salida
- **3.7.1 Tokenización** — la frontera entre lo simbólico y el tensor.
  - *Nivel* — carácter, palabra, subpalabra.
  - *Algoritmos* — BPE, WordPiece, SentencePiece, Unigram.
  - *Tokens especiales* — `[CLS]`, `[MASK]`, `<eos>`, padding.
- **3.7.2 Embeddings** — word2vec, GloVe, fastText; contextuales (ELMo, BERT).
- **3.7.3 Codificación posicional** — sinusoidal, aprendida, relativa, RoPE, ALiBi.
- **3.7.4 Cabezales de salida** — clasificación, regresión, detección, contrastivo, LM.
- **3.7.5 Espacios latentes** — lo que realmente se manipula.

---

# PARTE 4 — MECÁNICA DEL APRENDIZAJE

## 4.1 El bucle de entrenamiento

```none
┌──────────────────────────────────────────────────────┐
│ 1. FORWARD    ŷ = f_θ(x)                             │
│ 2. PÉRDIDA    L = ℓ(ŷ, y)                            │
│ 3. BACKWARD   ∇_θ L    ← autodiff en modo reverso    │
│ 4. UPDATE     θ ← optimizador(θ, ∇_θ L)              │
└──────────────────────────────────────────────────────┘
```

- **4.1.1 Época, batch, iteración**.
- **4.1.2 Tamaño de batch** — ruido del gradiente vs eficiencia de cómputo.
- **4.1.3 Acumulación de gradiente** — simular batches grandes con poca memoria.
- **4.1.4 Modo train vs eval** — dropout y BatchNorm cambian de comportamiento.

## 4.2 Funciones de pérdida
- **4.2.1 Toda pérdida es un supuesto probabilístico**.
- **4.2.2 Regresión** — MSE, MAE, Huber.
- **4.2.3 Clasificación** — cross-entropy, BCE, focal loss.
- **4.2.4 Contrastivas** — InfoNCE, triplet.
- **4.2.5 Generativas** — ELBO, adversarial, score matching.
- **4.2.6 De preferencia** — Bradley-Terry, DPO.
- **4.2.7 Multitarea** — ponderación, pérdidas auxiliares, equilibrado de escalas.

## 4.3 Backpropagation
- **4.3.1 Las cuatro ecuaciones del error retropropagado**.
- **4.3.2 Coste computacional** — el backward cuesta ~2× el forward.
- **4.3.3 Coste en memoria** — hay que guardar las activaciones.
- **4.3.4 Gradient checkpointing** — recomputar en vez de almacenar.
- **4.3.5 Detach y stop-gradient** — cortar el flujo deliberadamente.
- **4.3.6 Straight-through estimator** — atravesar operaciones no diferenciables.

## 4.4 Optimizadores

```none
SGD
 ├─► + Momentum ──► + Nesterov
 ├─► Adaptativos: AdaGrad ──► RMSProp ──► Adam (2014)
 │                                          └─► AdamW ★ estándar actual
 ├─► Segundo orden aproximado: K-FAC · Shampoo · Muon
 └─► Especializados: LAMB · Lion · Adafactor
```

- **4.4.1 SGD** — su ruido generaliza mejor de lo esperable.
- **4.4.2 Momentum y Nesterov**.
- **4.4.3 Métodos adaptativos** — tasa por parámetro según historia del gradiente.
- **4.4.4 Adam y AdamW** — el desacoplo del weight decay importa.
- **4.4.5 Estado del optimizador** — 2 tensores extra por parámetro.
- **4.4.6 Gradient clipping** — por norma o por valor.

## 4.5 Tasa de aprendizaje
- **4.5.1 El hiperparámetro nº1**.
- **4.5.2 Warmup** — imprescindible en Transformers.
- **4.5.3 Decaimiento** — step, exponencial, cosine, lineal.
- **4.5.4 One-cycle y políticas cíclicas**.
- **4.5.5 LR range test**.
- **4.5.6 Relación batch–LR** — escalado lineal y raíz cuadrada.

## 4.6 Hiperparámetros
- **4.6.1 Parámetros vs hiperparámetros**.
- **4.6.2 Los que importan** — LR, batch, weight decay, arquitectura, duración.
- **4.6.3 Estrategias de búsqueda** — grid < random < bayesiana < bandits (ASHA, Hyperband).
- **4.6.4 Transferencia de hiperparámetros** — μP y escalado predecible.

## 4.7 Patologías del entrenamiento
- **4.7.1 Vanishing gradient** — residuales, ReLU, normalización.
- **4.7.2 Exploding gradient** — clipping, mejor inicialización.
- **4.7.3 Dead ReLU**.
- **4.7.4 Sobreajuste** — brecha train/val creciente.
- **4.7.5 Subajuste**.
- **4.7.6 Colapso de rango y de representación**.
- **4.7.7 Colapso de modo** — típico de GAN.
- **4.7.8 Olvido catastrófico** — el fine-tuning destruye lo aprendido.
- **4.7.9 Data leakage** — el error silencioso más caro.
- **4.7.10 Desbalanceo de clases**.

## 4.8 Evaluación y metodología
- **4.8.1 Particiones** — el test se toca una sola vez.
- **4.8.2 Validación cruzada** — k-fold, estratificada, temporal.
- **4.8.3 Métricas de clasificación** — accuracy, precisión, recall, F1, ROC-AUC, PR-AUC.
- **4.8.4 Métricas de regresión** — MSE, MAE, R².
- **4.8.5 Métricas por dominio** — IoU, mAP, BLEU, perplejidad, FID.
- **4.8.6 Calibración**.
- **4.8.7 Curvas de aprendizaje** — diagnóstico de sesgo y varianza.
- **4.8.8 Ablations** — quitar componentes para atribuir el efecto.
- **4.8.9 Significancia** — un solo run no es un resultado.
- **4.8.10 Benchmarks y contaminación**.

---

# PARTE 5 — DATOS EN REJILLA: IMÁGENES

## 5.1 Del perceptrón al MLP
- **5.1.1 Perceptrón (Rosenblatt, 1958)** — clasificador lineal con regla de actualización propia.
- **5.1.2 El límite** — Minsky & Papert (1969): no resuelve XOR → primer invierno.
- **5.1.3 MLP** — capas ocultas; el XOR se resuelve con una capa intermedia.
- **5.1.4 Backpropagation** — Linnainmaa (1970) inventa el autodiff reverso; Werbos (1974) lo aplica a redes; Rumelhart, Hinton & Williams (1986) lo popularizan.
- **5.1.5 Teorema de aproximación universal** — garantiza existencia, no aprendibilidad.

## 5.2 La convolución
- **5.2.1 Motivación biológica** — campos receptivos (→ 1.4.7).
- **5.2.2 Sesgo inductivo** — localidad, compartición de pesos, equivarianza a traslación.
- **5.2.3 Anatomía** — kernel, stride, padding, canales, dilatación.
- **5.2.4 Campo receptivo** — crece con la profundidad.
- **5.2.5 Pooling** — max, average, global.
- **5.2.6 Variantes** — 1×1, separable en profundidad, transpuesta, dilatada.
- **5.2.7 Coste** — por qué una CNN es barata frente a un MLP equivalente.

## 5.3 Genealogía de las CNN

```none
Neocognitron (Fukushima, 1980) ..... estructura convolucional SIN backprop
   ▼
LeNet-5 (LeCun, 1989/1998) ......... CNN + backprop; funciona en dígitos
   │
   │  ⛔ 1998–2012: SEGUNDO INVIERNO
   │     Dominan SVM y boosting. Las redes profundas no entrenan.
   │     Se desbloquea con: ReLU · dropout · GPU · ImageNet
   │     · pre-entrenamiento no supervisado (Hinton, 2006)
   ▼
AlexNet (2012) ★ el punto de inflexión real del deep learning
   ├── VGG (2014) ............ profundidad uniforme con kernels 3×3
   ├── Inception (2014) ...... procesamiento multiescala en paralelo
   ├── ResNet (2015) ★ ....... conexiones residuales → cientos de capas
   ├── DenseNet (2016)
   ├── MobileNet / ShuffleNet  eficiencia en dispositivo
   ├── EfficientNet (2019) ... escalado compuesto
   └── ConvNeXt (2022) ....... la CNN reentrenada con recetas de Transformer
```

## 5.4 Tareas de visión
- **5.4.1 Clasificación**.
- **5.4.2 Detección de objetos** — R-CNN, Faster R-CNN, YOLO, SSD, DETR.
- **5.4.3 Segmentación** — semántica, de instancia, panóptica; U-Net, Mask R-CNN, SAM.
- **5.4.4 Estimación de pose y puntos clave**.
- **5.4.5 Vídeo** — convoluciones 3D, flujo óptico.
- **5.4.6 Visión 3D** — nubes de puntos, voxels, NeRF, gaussian splatting.

## 5.5 Transferencia
- **5.5.1 Transfer learning** — reutilizar representaciones de otro dominio.
- **5.5.2 Fine-tuning vs feature extraction**.
- **5.5.3 Congelación por capas y descongelado progresivo**.
- **5.5.4 Adaptación de dominio** — cuando train y test difieren de distribución.

---

# PARTE 6 — DATOS EN SECUENCIA: LENGUAJE

## 6.1 Redes recurrentes
- **6.1.1 Motivación** — entradas de longitud variable con dependencia temporal.
- **6.1.2 Estado oculto** — memoria comprimida del pasado.
- **6.1.3 Compartición de pesos en el tiempo**.
- **6.1.4 BPTT** — desplegar la red en el tiempo y retropropagar.
- **6.1.5 Truncated BPTT** — acotar el coste.
- **6.1.6 Variantes topológicas** — bidireccional, apilada, one-to-many, many-to-many.

## 6.2 El problema del gradiente y sus parches
- **6.2.1 Vanishing/exploding en el tiempo** — el gradiente se multiplica ~T veces.
- **6.2.2 LSTM (1997)** — celda de estado + puertas de entrada, olvido y salida.
- **6.2.3 GRU (2014)** — puertas de actualización y reinicio.
- **6.2.4 Limitación de fondo** — las puertas parchean; el cómputo sigue siendo secuencial.

## 6.3 Seq2seq
- **6.3.1 Encoder-decoder (Sutskever, 2014)**.
- **6.3.2 Vector de contexto** — todo el significado en un solo vector.
- **6.3.3 El cuello de botella** — la calidad cae con la longitud.
- **6.3.4 Teacher forcing** — y el exposure bias que produce.
- **6.3.5 Decodificación** — greedy, beam search, muestreo, top-k, top-p, temperatura.

## 6.4 Mecanismo de atención
- **6.4.1 Idea central** — acceso ponderado y dinámico a todos los estados.
- **6.4.2 Bahdanau (2014)** — atención aditiva.
- **6.4.3 Luong (2015)** — atención multiplicativa.
- **6.4.4 Query, key, value** — recuperación asociativa diferenciable.
- **6.4.5 Scaled dot-product** — el √d evita saturar el softmax.
- **6.4.6 Interpretabilidad** — los mapas de atención sugieren alineamiento, no explicación.

## 6.5 El Transformer (2017)
- **6.5.1 "Attention Is All You Need"** — sin recurrencia: paralelizable en la longitud.
- **6.5.2 Self-attention** — cada posición atiende a todas las de su propia secuencia.
- **6.5.3 Multi-head attention** — varios subespacios de relación en paralelo.
- **6.5.4 Bloque feed-forward** — donde vive la mayoría de los parámetros.
- **6.5.5 Residual + LayerNorm** — lo que hace entrenable la pila.
- **6.5.6 Codificación posicional** — sin ella el modelo es invariante a permutación.
- **6.5.7 Máscara causal** — impide ver el futuro.
- **6.5.8 Cross-attention** — el decoder consulta al encoder.
- **6.5.9 Coste cuadrático** — O(n²) en longitud: la limitación central.

## 6.6 Las tres familias de Transformer

```none
├── ENCODER-ONLY ..... bidireccional, para COMPRENDER
│     BERT, RoBERTa, DeBERTa, ELECTRA · masked language modeling
├── DECODER-ONLY ..... autoregresivo, para GENERAR   ★ familia dominante
│     GPT, LLaMA, Mistral, Claude, Gemini, DeepSeek · next-token prediction
└── ENCODER-DECODER .. de secuencia a secuencia
      T5, BART, Whisper
```

## 6.7 Eficiencia y evolución
- **6.7.1 Atención eficiente** — Linformer, Performer, Longformer, atención dispersa.
- **6.7.2 FlashAttention** — exacta y más rápida por gestión de memoria.
- **6.7.3 Multi-query y grouped-query attention** — reducir la KV cache.
- **6.7.4 Extensión de contexto** — interpolación posicional, RoPE escalado.
- **6.7.5 Mezcla de expertos** — Switch Transformer, Mixtral, DeepSeek-V3.
- **6.7.6 State Space Models** — S4, Mamba (2023): la recurrencia vuelve, paralelizable.
- **6.7.7 Arquitecturas híbridas** — combinar atención y recurrencia por capas.
- **6.7.8 Memoria explícita y aprendizaje continuo** — Titans, Nested Learning: módulos de memoria que se actualizan a distintas frecuencias para mitigar el olvido.

## 6.8 El Transformer fuera del texto
- **6.8.1 ViT (2020)** — imagen como secuencia de parches.
- **6.8.2 Whisper, wav2vec** — audio.
- **6.8.3 AlphaFold2** — estructura de proteínas.
- **6.8.4 Decision Transformer** — RL como modelado de secuencias.
- **6.8.5 Conclusión** — con datos y cómputo suficientes, el sesgo inductivo mínimo gana.

---

# PARTE 7 — CREAR DATOS NUEVOS: MODELOS GENERATIVOS

## 7.1 El problema generativo
- **7.1.1 Discriminativo vs generativo** — modelar p(y|x) frente a modelar p(x).
- **7.1.2 El trilema** — calidad, diversidad y velocidad: elige dos.
- **7.1.3 Verosimilitud explícita vs implícita** — poder evaluar p(x) o solo muestrear.
- **7.1.4 Espacio latente** — variable oculta de baja dimensión que genera el dato.

## 7.2 Autoencoders y VAE
- **7.2.1 Autoencoder** — comprimir y reconstruir; el latente no es muestreable.
- **7.2.2 Denoising autoencoder** — antecedente directo de la difusión.
- **7.2.3 RBM y Deep Belief Nets (Hinton, 2006)** — el pre-entrenamiento capa a capa que reabrió el campo.
- **7.2.4 VAE (2013)** — latente probabilístico con prior gaussiano.
- **7.2.5 ELBO** — reconstrucción + término KL.
- **7.2.6 Truco de reparametrización** — hacer diferenciable el muestreo.
- **7.2.7 Posterior collapse** — el latente se ignora.
- **7.2.8 VQ-VAE** — latente discreto; base de la generación por tokens.

## 7.3 Redes adversariales (GAN)
- **7.3.1 GAN (2014)** — juego minimax entre generador y discriminador.
- **7.3.2 Equilibrio de Nash** — el objetivo es un punto de silla, no un mínimo.
- **7.3.3 Patologías** — colapso de modo, inestabilidad.
- **7.3.4 WGAN y penalización de gradiente**.
- **7.3.5 Genealogía** — DCGAN, pix2pix, CycleGAN, StyleGAN, BigGAN.
- **7.3.6 Estado actual** — desplazadas por la difusión salvo donde importa la velocidad.

## 7.4 Modelos de flujo
- **7.4.1 Normalizing flows** — transformaciones invertibles con jacobiano tratable.
- **7.4.2 Verosimilitud exacta** — su ventaja frente a VAE y GAN.
- **7.4.3 RealNVP, Glow**.
- **7.4.4 Continuous flows / Neural ODEs**.

## 7.5 Modelos autoregresivos
- **7.5.1 Factorización en cadena** — p(x) como producto de condicionales.
- **7.5.2 PixelCNN, WaveNet**.
- **7.5.3 Los LLM como modelos generativos** — el mismo principio sobre tokens.
- **7.5.4 Coste** — calidad alta, muestreo secuencial lento.

## 7.6 Modelos de difusión
- **7.6.1 Idea** — aprender a invertir un proceso de ruido gaussiano progresivo.
- **7.6.2 Proceso forward** — añadir ruido hasta destruir la señal; sin parámetros.
- **7.6.3 Proceso reverso** — la red aprende a eliminar ruido paso a paso.
- **7.6.4 DDPM (2020)** — la formulación que lo hizo funcionar.
- **7.6.5 Score matching y SDE** — el marco unificador continuo.
- **7.6.6 Muestreo acelerado** — DDIM, solvers de EDO, destilación de pasos.
- **7.6.7 Guía** — classifier guidance y classifier-free guidance.
- **7.6.8 Latent Diffusion (2022)** — difundir en el latente de un VAE; Stable Diffusion.
- **7.6.9 DiT (2023)** ★ — la U-Net sustituida por un Transformer.
- **7.6.10 Flow matching / rectified flow** — formulación más simple; el estándar emergente.
- **7.6.11 Difusión descentralizada** — expertos de difusión entrenados sobre particiones del dataset y combinados después (→ 11.6).

## 7.7 Multimodalidad
- **7.7.1 CLIP (2021)** — alineación contrastiva de texto e imagen.
- **7.7.2 Texto a imagen** — DALL·E, Imagen, Stable Diffusion.
- **7.7.3 Texto a vídeo, audio y 3D**.
- **7.7.4 Modelos de mundo** — predecir el futuro como objetivo de aprendizaje; JEPA, Genie, Dreamer.

---

# PARTE 8 — ESTRUCTURA IRREGULAR: GRAFOS Y SIMETRÍAS

## 8.1 Datos no euclidianos
- **8.1.1 Más allá de la rejilla** — grafos, conjuntos, mallas, variedades.
- **8.1.2 Invarianza vs equivarianza** — la salida no cambia / cambia igual que la entrada.
- **8.1.3 Grupos de simetría** — traslación, rotación, permutación, escala, gauge.

## 8.2 Redes sobre grafos (GNN)
- **8.2.1 Message passing** — agregar información del vecindario y actualizar el nodo.
- **8.2.2 Agregadores permutación-invariantes** — suma, media, máximo.
- **8.2.3 GCN (2016)** — convolución espectral simplificada.
- **8.2.4 GraphSAGE** — muestreo de vecindario para escalar.
- **8.2.5 GAT (2018)** — atención sobre las aristas.
- **8.2.6 Patologías** — over-smoothing, over-squashing, límites de expresividad (Weisfeiler-Lehman).
- **8.2.7 Pooling y readout** — representación global del grafo.
- **8.2.8 GNN federadas** — el grafo repartido entre dueños; aristas que cruzan silos.

## 8.3 Redes equivariantes
- **8.3.1 Deep Sets** — funciones sobre conjuntos.
- **8.3.2 Equivarianza E(3)/SE(3)** — rotación y traslación en el espacio físico.
- **8.3.3 Steerable CNNs, armónicos esféricos**.
- **8.3.4 Gauge equivariance** — el caso general sobre variedades.

## 8.4 Geometric DL como marco unificador

```none
BRONSTEIN et al. (2021) — "el programa de Erlangen del deep learning"

   La ARQUITECTURA se deduce de la SIMETRÍA del dominio de los datos.

   dominio               simetría                 arquitectura resultante
   ─────────────────────────────────────────────────────────────────────
   rejilla espacial  →   traslación           →   CNN            (Parte 5)
   secuencia         →   traslación temporal  →   RNN            (Parte 6)
   conjunto          →   permutación          →   Transformer    (Parte 6)
   grafo             →   permutación de nodos →   GNN            (Parte 8)
   variedad / 3D     →   rotación, gauge      →   red equivariante

   ⇒ Las cuatro familias no son cuatro inventos independientes:
     son el MISMO principio bajo cuatro grupos de simetría distintos.
```

- **8.4.1 Los cinco bloques (5G)** — grids, groups, graphs, geodesics, gauges.
- **8.4.2 Plantilla común** — capa equivariante local + no linealidad + agregación global invariante.
- **8.4.3 Consecuencia de diseño** — elegir arquitectura = identificar la simetría del problema.

## 8.5 Aplicaciones científicas
- **8.5.1 Química y materiales** — moléculas como grafos.
- **8.5.2 Biología estructural** — AlphaFold2/3.
- **8.5.3 Física** — simulación, dinámica de partículas, PDE.
- **8.5.4 Physics-informed neural networks** — imponer leyes físicas en la pérdida.
- **8.5.5 Neural operators** — FNO, DeepONet.

---

# PARTE 9 — MODELOS BASE Y ESCALA

## 9.1 El cambio de modelo de trabajo
- **9.1.1 Antes** — un modelo por tarea, desde cero, con datos etiquetados.
- **9.1.2 Ahora** — un modelo base pre-entrenado sobre datos masivos + adaptación barata.
- **9.1.3 Definición de modelo base (foundation model)** — modelo general adaptable a muchas tareas.
- **9.1.4 Consecuencia económica** — el coste se concentra en el pre-entrenamiento; adaptar es marginal.
- **9.1.5 Consecuencia científica** — la unidad de investigación pasa a ser la receta, no la arquitectura.

## 9.2 Pre-entrenamiento auto-supervisado
- **9.2.1 Next-token prediction** — el objetivo dominante en lenguaje.
- **9.2.2 Masked modeling** — BERT, MAE.
- **9.2.3 Contrastivo** — SimCLR, MoCo, CLIP.
- **9.2.4 No contrastivo** — BYOL, DINO, SimSiam.
- **9.2.5 Curación de corpus** — deduplicación, filtrado de calidad, mezcla de dominios.

## 9.3 Leyes de escalado
- **9.3.1 Kaplan et al. (2020)** — la pérdida decae como ley de potencias en parámetros, datos y cómputo.
- **9.3.2 Chinchilla (2022)** — corrección: los modelos estaban infra-entrenados en datos.
- **9.3.3 Cómputo óptimo** — repartir el presupuesto entre tamaño y tokens.
- **9.3.4 Escalado en inferencia** — gastar cómputo en tiempo de test; modelos de razonamiento.
- **9.3.5 El muro de los datos** — agotamiento de texto de calidad; datos sintéticos.
- **9.3.6 Leyes de escalado para entrenamiento con poca comunicación** — cómo escalan DiLoCo y familia (→ 11.6).

## 9.4 Ciclo de vida de un modelo grande

```none
pre-entrenamiento → continual → SFT → alineamiento
      → evaluación → cuantización → despliegue → monitorización
```

- **9.4.1 Pre-entrenamiento** — la fase cara; donde se adquiere el conocimiento.
- **9.4.2 Continual pretraining** — especialización de dominio sin empezar de cero.
- **9.4.3 SFT** — fine-tuning por instrucciones: enseñar el formato de respuesta.
- **9.4.4 Alineamiento** — RLHF, DPO, IA constitucional (→ 10.4).
- **9.4.5 RL sobre razonamiento** — recompensa verificable en tareas con solución comprobable (matemáticas, código); GRPO.
- **9.4.6 Destilación de razonamiento** — trasladar capacidad de razonamiento a modelos pequeños.
- **9.4.7 Evaluación** — benchmarks, evaluación humana, red-teaming.

## 9.5 Adaptación eficiente (PEFT)
- **9.5.1 Motivación** — el fine-tuning completo es inviable en memoria y almacenamiento.
- **9.5.2 LoRA** — actualización de rango bajo `ΔW = BA`; aquí reaparece la SVD.
- **9.5.3 QLoRA** — LoRA sobre un modelo base cuantizado.
- **9.5.4 Adapters, prefix tuning, prompt tuning**.
- **9.5.5 Fusión y composición de adaptadores**.

## 9.6 Compresión e inferencia
- **9.6.1 Cuantización** — INT8, INT4, GPTQ, AWQ.
- **9.6.2 Pruning** — estructurado y no estructurado.
- **9.6.3 Destilación de conocimiento**.
- **9.6.4 KV cache** — el cuello de botella de memoria en generación.
- **9.6.5 Decodificación especulativa**.
- **9.6.6 Batching continuo, paged attention**.

## 9.7 Uso en inferencia
- **9.7.1 Prompting** — zero-shot, few-shot, in-context learning.
- **9.7.2 Razonamiento explícito** — chain-of-thought, self-consistency, thinking tokens.
- **9.7.3 RAG** — recuperación + generación; bases vectoriales, chunking, reranking.
- **9.7.4 Uso de herramientas** — llamada a funciones, ejecución de código, protocolos de contexto.
- **9.7.5 Salida estructurada** — decodificación restringida, gramáticas.
- **9.7.6 Ventana de contexto** — límites, coste, gestión.

## 9.8 Comportamiento emergente
- **9.8.1 La afirmación** — capacidades que aparecen bruscamente a cierta escala.
- **9.8.2 La objeción** — Schaeffer et al. (2023): artefacto de métricas discontinuas.
- **9.8.3 Estado** — pregunta abierta; tratar con cautela metodológica.

---

# PARTE 10 — REFUERZO Y ALINEAMIENTO

## 10.1 Marco formal
- **10.1.1 MDP** — estados, acciones, transiciones, recompensa, descuento.
- **10.1.2 Política, valor, función Q, ventaja**.
- **10.1.3 Ecuación de Bellman**.
- **10.1.4 Exploración vs explotación** — ε-greedy, UCB, bonus de entropía.
- **10.1.5 Asignación de crédito temporal** — el problema difícil del RL.

## 10.2 Ejes de clasificación

```none
├── online  vs  offline .......... interactúa con el entorno / dataset fijo
├── on-policy vs off-policy ...... aprende de su política / de otra
└── model-free vs model-based .... sin modelo del entorno / aprendiendo la dinámica
```

## 10.3 Familias de algoritmos
- **10.3.1 Basados en valor** — Q-learning, SARSA, DQN.
- **10.3.2 Basados en política** — REINFORCE, gradiente de política.
- **10.3.3 Actor-crítico** — A2C/A3C, DDPG, TD3, SAC.
- **10.3.4 Región de confianza** — TRPO, PPO, GRPO.
- **10.3.5 Model-based** — Dyna, MuZero, Dreamer.
- **10.3.6 Offline RL** — CQL, IQL, Decision Transformer.

## 10.4 Aprendizaje a partir de preferencias humanas
- **10.4.1 Imitation learning y behavioral cloning**.
- **10.4.2 Inverse RL** — inferir la recompensa desde el comportamiento.
- **10.4.3 Modelo de recompensa** — predictor de preferencia humana.
- **10.4.4 RLHF** — SFT + modelo de recompensa + PPO.
- **10.4.5 DPO y variantes** — optimizar la preferencia sin RL explícito.
- **10.4.6 IA constitucional / RLAIF** — feedback generado por el modelo bajo principios.
- **10.4.7 Recompensa verificable (RLVR)** — la señal viene de comprobar la solución, no de un humano.
- **10.4.8 Reward hacking** — optimizar la métrica destruyendo el objetivo real.

## 10.5 Agentes
- **10.5.1 Bucle percepción–acción**.
- **10.5.2 Memoria y estado**.
- **10.5.3 Uso de herramientas y entornos**.
- **10.5.4 RL multiagente y colaborativo** — enjambres de agentes que se critican y revisan.
- **10.5.5 Evaluación de agentes** — tasa de éxito, coste, robustez.

---

# PARTE 11 — ARQUITECTURAS DE ENTRENAMIENTO: DÓNDE ESTÁN LOS DATOS

## 11.1 Por qué este eje es independiente

```none
                   ¿QUIÉN CONTROLA LOS DATOS Y EL CÓMPUTO?
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
   UN SOLO DUEÑO                                     VARIOS DUEÑOS
        │                                                   │
   ┌────┴────────┐                        ┌─────────────────┼─────────────────┐
   │             │                        │                 │                 │
CENTRALIZADO  DISTRIBUIDO            FEDERADO        DESCENTRALIZADO     DATA SPACES
un servidor   varios nodos,          servidor        sin servidor        la gobernanza
              mismo perímetro        agregador;      central;            que hace posible
              (11.3)                 los datos no    topología de        el intercambio
                                     salen (11.4)    pares (11.5)        con soberanía (11.9)
                                                           │
                                              ENTRENAMIENTO DESCENTRALIZADO
                                              DE MODELOS GRANDES (11.6)
```

- **11.1.1 Distribuido ≠ federado** — en distribuido reparto porque quiero (rendimiento); en federado no puedo juntar los datos (legal, comercial o físicamente).
- **11.1.2 Federado ≠ descentralizado** — el federado clásico tiene servidor agregador; el descentralizado no tiene ningún punto central de confianza ni de fallo.
- **11.1.3 Motivaciones para no centralizar** — regulación, secreto comercial, volumen, latencia, coste de transferencia, soberanía, disponibilidad de GPU dispersa.
- **11.1.4 Qué viaja en cada caso** — datos brutos / gradientes / pseudo-gradientes / pesos / activaciones / estadísticos agregados / nada.
- **11.1.5 El compromiso central** — utilidad ↔ privacidad ↔ coste de comunicación. No se maximizan los tres.
- **11.1.6 La magnitud que lo domina todo** — el ancho de banda entre nodos. Dentro de un centro de datos hay cientos de Gb/s; entre centros, del orden de cientos de Mb/s a 1 Gb/s. Ese salto de tres órdenes de magnitud es lo que separa el 11.3 del 11.5–11.6.

## 11.2 Centralizado
- **11.2.1 El caso base** — todos los datos en un servidor o clúster; todo lo demás se compara contra esto.
- **11.2.2 Ventajas** — control de la mezcla, barajado global, depuración simple, máximo rendimiento por dato.
- **11.2.3 Límites** — almacenamiento, cumplimiento normativo, punto único de fallo y de fuga.
- **11.2.4 Cuándo sigue siendo la respuesta correcta** — la mayoría de las veces; descentralizar sin necesidad es coste puro.

## 11.3 Distribuido (un solo dueño)
- **11.3.1 Motivación** — el modelo o los datos no caben, o se quiere ir más rápido.
- **11.3.2 Data parallelism** — replicar el modelo, repartir el batch, promediar gradientes cada paso.
- **11.3.3 Model / tensor parallelism** — repartir los pesos de una misma capa entre dispositivos.
- **11.3.4 Pipeline parallelism** — repartir capas; burbujas y micro-batching.
- **11.3.5 Sequence / context parallelism** — repartir la longitud de la secuencia.
- **11.3.6 Expert parallelism** — repartir expertos de un MoE.
- **11.3.7 ZeRO / FSDP** — fragmentar estado del optimizador, gradientes y parámetros.
- **11.3.8 Paralelismo 3D / 4D** — combinar datos + tensor + pipeline (+ contexto); lo que usan los modelos frontera.
- **11.3.9 Comunicación colectiva** — all-reduce, all-gather, reduce-scatter; NCCL, ring vs tree.
- **11.3.10 Interconexión** — NVLink, InfiniBand, RoCE; topología y congestión.
- **11.3.11 Síncrono vs asíncrono** — barreras y stragglers frente a gradientes obsoletos (staleness).
- **11.3.12 MFU (model FLOPs utilization)** — la métrica que mide cuánto del hardware se aprovecha de verdad.
- **11.3.13 Tolerancia a fallos** — checkpointing, reinicio elástico, malla de dispositivos elástica.
- **11.3.14 Ley de Amdahl en la práctica** — pasado cierto punto, la comunicación domina.

## 11.4 Aprendizaje federado

### 11.4.1 Fundamentos
- **Definición** — múltiples dueños entrenan un modelo común sin mover los datos; solo se comparten actualizaciones.
- **Origen** — McMahan et al. (Google, 2016-17): teclado predictivo en móviles.
- **El bucle federado** — el servidor envía el modelo → los clientes entrenan localmente → devuelven actualizaciones → el servidor agrega → repetir.
- **Ronda de comunicación** — la unidad de coste; sustituye a la época como magnitud a minimizar.

### 11.4.2 Taxonomía por partición de datos

```none
                    características (columnas)
                  ┌──────────────────────────────┐
   muestras   ┌───┤ HORIZONTAL: mismas columnas, │
   (filas)    │   │ distintas filas              │  → hospitales distintos,
              │   │ (el caso más común)          │    mismos campos clínicos
              │   ├──────────────────────────────┤
              │   │ VERTICAL: mismas filas,      │  → banco y aseguradora
              │   │ distintas columnas           │    con los mismos clientes
              │   ├──────────────────────────────┤
              └───┤ FEDERATED TRANSFER: poco     │  → dominios distintos,
                  │ solape en ambos ejes         │    se transfiere conocimiento
                  └──────────────────────────────┘
```

### 11.4.3 Taxonomía por escala
- **Cross-device** — millones de clientes pequeños, poco fiables, disponibilidad intermitente.
- **Cross-silo** — decenas de organizaciones grandes y fiables, siempre disponibles.
- **Consecuencia** — son problemas casi distintos: en cross-device manda la comunicación y el muestreo; en cross-silo, la heterogeneidad y la gobernanza.

### 11.4.4 Algoritmos de agregación
- **FedSGD** — un paso local por ronda; base teórica, comunicación prohibitiva.
- **FedAvg** — varios pasos locales antes de agregar; el algoritmo de referencia.
- **FedProx** — término proximal que limita la deriva del cliente.
- **SCAFFOLD** — variables de control que corrigen el sesgo del cliente.
- **FedNova** — normalización por número desigual de pasos locales.
- **FedOpt / FedAdam** — optimizador adaptativo en el lado del servidor.
- **Agregación por matching** — FedMA, PFNM: alinear neuronas antes de promediar, porque el promedio ingenuo ignora la permutación.

### 11.4.5 El problema central: heterogeneidad
- **Datos no-IID** — cada cliente tiene una distribución distinta; rompe el supuesto que justifica promediar.
  - *Tipos de shift* — de etiquetas, de características, de cantidad, temporal.
  - *Client drift* — los modelos locales divergen durante los pasos locales.
  - *Simulación* — particiones por Dirichlet o por etiqueta; cómo se reporta en la literatura.
- **Heterogeneidad de sistema** — dispositivos con capacidad, red y batería desiguales.
- **Stragglers y participación parcial** — solo una fracción de clientes responde cada ronda.
- **Equidad (fairness)** — buen rendimiento medio y malo para clientes minoritarios.

### 11.4.6 Personalización
- **Fine-tuning local** — la línea base más simple y sorprendentemente fuerte.
- **Meta-aprendizaje** — Per-FedAvg: aprender una inicialización fácil de adaptar.
- **Capas compartidas + cabezal personal** — FedPer, FedRep.
- **Clustering de clientes** — agrupar por similitud de distribución.
- **Multi-task federado** — cada cliente como tarea relacionada.
- **Destilación federada** — compartir logits en vez de pesos; permite arquitecturas distintas por cliente.

### 11.4.7 Eficiencia de comunicación
- **Cuantización de actualizaciones** — menos bits por parámetro.
- **Esparsificación** — enviar solo el top-k de los gradientes.
- **Compresión con memoria de error** — acumular lo no enviado para no perder señal.
- **Actualizaciones de rango bajo** — LoRA federado: enviar solo adaptadores.
- **Menos rondas** — más cómputo local por ronda, hasta donde lo permita la deriva.
- **Selección de clientes** — muestreo por disponibilidad, importancia o contribución.

### 11.4.8 Federado y modelos base
- **Fine-tuning federado de LLM** — el caso de uso emergente; PEFT lo hace viable.
- **LoRA federado y sus variantes** — agregación exacta de adaptadores, rangos heterogéneos por cliente, jerarquías de adaptadores.
- **Federated prompt tuning** — compartir prompts en vez de pesos.
- **Restricciones prácticas** — memoria del cliente, tamaño del modelo base, licencias.

## 11.5 Aprendizaje descentralizado

> Sin servidor central: no hay agregador, ni punto único de fallo, ni entidad
> que tenga que ser de confianza. La coordinación emerge de la topología.

### 11.5.1 Fundamentos
- **Diferencia con federado** — el federado sustituye "datos centralizados" por "agregación centralizada"; el descentralizado elimina también esa.
- **Modelo de comunicación** — cada nodo habla solo con sus vecinos en un grafo.
- **Consenso** — todos los nodos deben converger al mismo modelo (o a modelos cercanos) sin coordinador.
- **Matriz de mezcla (gossip matrix)** — describe con qué peso cada nodo promedia con sus vecinos.
- **Spectral gap** — la propiedad espectral de esa matriz que gobierna la velocidad de consenso; el análogo descentralizado del ancho de banda.

### 11.5.2 Algoritmos base
- **Gossip learning** — intercambio de modelos entre vecinos; la información se difunde por la red.
- **D-PSGD** — descenso estocástico paralelo descentralizado: la formulación de referencia.
- **Gossip asíncrono** — sin rondas globales; cada nodo avanza a su ritmo.
- **Choco-SGD y compresión descentralizada** — cuantizar o esparsificar lo que viaja entre vecinos.
- **Push-sum y grafos dirigidos** — cuando la comunicación no es simétrica.
- **Promedio con momentum descentralizado** — corregir la deriva entre vecinos.

### 11.5.3 Topología
- **Familias** — anillo, malla, toro, expander, grafo aleatorio, jerárquico.
- **Compromiso** — más conectividad acelera el consenso pero encarece cada ronda.
- **Topología dinámica** — cambiar los vecinos entre rondas mejora la mezcla.
- **Descubrimiento de nodos** — tablas hash distribuidas (DHT) para redes abiertas.

### 11.5.4 Confianza y coordinación sin servidor
- **Swarm learning** — coordinación mediante blockchain; el registro sustituye al agregador.
- **Redes permissionless** — cualquiera puede unirse; hace falta verificación e incentivos.
- **Verificación de trabajo** — probar que un nodo entrenó lo que dice haber entrenado.
- **Seguimiento de contribución** — medir el aporte para repartir crédito o recompensa.
- **Cómputo voluntario** — el linaje de proyectos de ciencia distribuida aplicado a DL.

### 11.5.5 Dificultades específicas
- **Convergencia más lenta** — el consenso cuesta rondas adicionales frente al all-reduce global.
- **Análisis teórico más difícil** — la garantía depende de la topología, no solo del algoritmo.
- **Heterogeneidad amplificada** — sin agregador global, la deriva entre regiones del grafo persiste.
- **Nodos maliciosos** — sin autoridad central, las defensas deben ser locales.
- **Fallos y entradas/salidas** — la red cambia durante el entrenamiento.

## 11.6 Entrenamiento descentralizado de modelos grandes

> La línea de trabajo más activa: entrenar modelos de miles de millones de
> parámetros repartidos entre centros de datos, nubes o incluso voluntarios,
> unidos por enlaces lentos.

### 11.6.1 El problema
- **El cuello de botella** — el data parallelism clásico sincroniza cada paso; imposible con enlaces de 1 Gb/s.
- **La pregunta** — ¿cuánto se puede reducir la frecuencia de sincronización sin perder calidad?
- **La respuesta corta** — mucho más de lo que se pensaba: en el orden de cientos de pasos locales.

### 11.6.2 DiLoCo y familia
- **DiLoCo (2023)** — dos bucles de optimización: uno interno con cientos de pasos locales de AdamW, y uno externo que intercambia pseudo-gradientes (la diferencia entre pesos inicial y final del tramo local) con SGD y momentum de Nesterov. Reduce la comunicación entre nodos en varios órdenes de magnitud manteniendo curvas de pérdida muy próximas a las del entrenamiento colocalizado.
- **Parentesco con FedAvg** — es esencialmente FedAvg llevado al pre-entrenamiento de LLM, con optimizador externo.
- **Pseudo-gradiente** — el objeto central: no es un gradiente, es un desplazamiento de pesos tratado como tal.
- **OpenDiLoCo** — implementación abierta; pasos externos asíncronos para que los rezagados no bloqueen.
- **DiLoCoX y variantes** — extensiones para clústeres descentralizados con ancho de banda limitado.
- **Leyes de escalado para DiLoCo** — cómo se comporta el método al crecer el modelo.
- **Límites conocidos** — sobrecoste de reloj de pared, deriva acumulada entre pasos externos que se agrava con el tamaño del modelo, e incompatibilidad con arquitecturas que necesitan estadísticas de batch sincronizadas.

### 11.6.3 Sistemas y ejecuciones de referencia
- **INTELLECT-1 (2024)** — primer modelo de 10 000 millones de parámetros entrenado con recursos descentralizados; agregación jerárquica, cuantización int8 de lo transmitido, malla de dispositivos elástica y recuperación de checkpoints entre pares.
- **INTELLECT-2 (2025)** — modelo de razonamiento entrenado mediante RL descentralizado a escala global.
- **Enjambres de RL descentralizado** — post-entrenamiento colaborativo en el que cada nodo ejecuta un modelo local y participa en ciclos de respuesta, crítica y revisión.
- **Difusión descentralizada** — expertos entrenados sobre particiones del dataset y compuestos después.
- **Inferencia descentralizada** — paralelismo de modelo sobre nodos voluntarios; expertos distribuidos y enrutamiento por DHT.

### 11.6.4 Ejes de diseño de un sistema descentralizado
- **Frecuencia de sincronización** — el parámetro maestro: cada paso ↔ cada centenar de pasos.
- **Qué se transmite** — gradientes, pseudo-gradientes, pesos, o representaciones comprimidas.
- **Compresión** — cuantización, esparsificación, proyecciones de rango bajo.
- **Tolerancia a la heterogeneidad** — nodos con GPU distintas y enlaces distintos.
- **Elasticidad** — nodos que entran y salen sin reiniciar el entrenamiento.
- **Aprovechamiento económico** — combinar créditos de varias nubes o precios spot en vez de un único proveedor.

### 11.6.5 Cuándo NO conviene
- Cuando existe un plazo duro y el ancho de banda completo del centro de datos está disponible.
- Cuando el modelo es muy grande y requiere paralelismo de pipeline estrecho, porque la deriva entre pasos externos crece con el tamaño.
- Cuando la arquitectura necesita estadísticas de batch sincronizadas entre nodos.

## 11.7 Split learning y aprendizaje en el borde
- **11.7.1 Split learning** — cortar la red: el cliente ejecuta las primeras capas, el servidor el resto.
- **11.7.2 Qué viaja** — activaciones intermedias y sus gradientes, no los datos ni el modelo completo.
- **11.7.3 Ventaja e inconveniente** — cliente muy ligero; pero comunicación por muestra y no por ronda.
- **11.7.4 SplitFed** — híbrido entre paralelismo federado y corte de split.
- **11.7.5 Riesgo de reconstrucción** — las activaciones intermedias pueden filtrar la entrada.
- **11.7.6 TinyML y edge** — inferencia y entrenamiento en dispositivo: cuantización, memoria, energía.
- **11.7.7 Aprendizaje continuo en el borde** — adaptación local sin olvido catastrófico.
- **11.7.8 Inferencia en el borde** — motivación de privacidad: el dato sensible nunca sale del dispositivo.

## 11.8 Privacidad y seguridad
- **11.8.1 El error de partida** — no compartir datos **no** es privacidad; los gradientes filtran información.
- **11.8.2 Privacidad diferencial**
  - *Definición* — garantía formal (ε, δ) mediante ruido calibrado a la sensibilidad.
  - *Local vs central* — ruido en el cliente o en el agregador; distinto modelo de confianza.
  - *DP-SGD* — recorte de gradiente por muestra + ruido gaussiano.
  - *Contabilidad de privacidad* — composición, moments accountant; el presupuesto ε se agota.
  - *Coste* — pérdida de utilidad, sobre todo en clases minoritarias.
- **11.8.3 Agregación segura** — el servidor ve la suma, nunca la contribución individual.
- **11.8.4 Cifrado homomórfico** — operar sobre datos cifrados; coste computacional alto.
- **11.8.5 Computación multiparte (MPC)** — cálculo conjunto sin revelar entradas.
- **11.8.6 Entornos de ejecución confiables (TEE)** — enclaves hardware; confianza desplazada al fabricante.
- **11.8.7 Datos sintéticos** — compartir una muestra generada; y sus límites de garantía.
- **11.8.8 Ataques**
  - *Inversión de gradiente* — reconstruir la muestra original a partir del gradiente.
  - *Inferencia de pertenencia* — determinar si un registro estuvo en el entrenamiento.
  - *Inferencia de propiedades* — deducir atributos agregados de un cliente.
  - *Envenenamiento de datos y de modelo* — degradar el modelo global.
  - *Puertas traseras (backdoors)* — comportamiento malicioso activado por un disparador.
  - *Free-riding* — beneficiarse sin aportar cómputo ni datos.
  - *Sybil* — un atacante simula muchos nodos; crítico en redes abiertas.
- **11.8.9 Defensas** — agregación robusta (Krum, trimmed mean, median), detección de anomalías, recorte de normas, validación cruzada entre clientes.
- **11.8.10 Auditoría y trazabilidad** — demostrar quién contribuyó qué y bajo qué condiciones.

## 11.9 Data spaces y gobernanza del dato
- **11.9.1 Qué problema resuelven** — lo federado resuelve lo técnico; los data spaces resuelven lo institucional: bajo qué reglas dos organizaciones colaboran.
- **11.9.2 Soberanía del dato** — el dueño conserva el control sobre el uso incluso después de compartirlo.
- **11.9.3 Arquitectura de referencia**
  - *Conectores* — el componente que media todo intercambio (IDS Connector, Eclipse EDC).
  - *Identidad y confianza* — certificación de participantes, credenciales verificables.
  - *Catálogo y descubrimiento* — publicar qué datos existen sin exponerlos.
  - *Vocabularios y ontologías* — interoperabilidad semántica; sin esto no hay federación real.
- **11.9.4 Contratos de uso** — políticas ejecutables (ODRL): qué se puede hacer, cuánto tiempo, por quién.
- **11.9.5 Iniciativas** — IDSA, Gaia-X, FIWARE; espacios sectoriales (salud EHDS, movilidad, industria, agroalimentario).
- **11.9.6 Marco regulatorio** — RGPD, Data Act, Data Governance Act, AI Act; anonimización vs seudonimización.
- **11.9.7 Modelos de incentivos** — valoración de la contribución (Shapley), reparto de beneficios, mercados de datos.
- **11.9.8 La brecha real** — la literatura federada asume cooperación perfecta; los data spaces existen precisamente porque no la hay.

## 11.10 Evaluación, benchmarks y frameworks
- **11.10.1 Qué se mide** — precisión global, precisión por cliente, varianza entre clientes, rondas hasta converger, bytes transmitidos, MFU, energía.
- **11.10.2 Evaluación justa** — cómo particionar, cómo reportar no-IID, por qué muchos resultados no son comparables.
- **11.10.3 Benchmarks** — LEAF, FedScale, FLamby (médico).
- **11.10.4 Frameworks federados** — Flower, FedML, NVIDIA FLARE, TensorFlow Federated, OpenFL, Substra.
- **11.10.5 Frameworks descentralizados** — implementaciones abiertas de DiLoCo y protocolos de entrenamiento distribuido entre nubes.
- **11.10.6 Simulación vs despliegue real** — casi toda la investigación es simulada; el hueco con la práctica es grande.
- **11.10.7 Reproducibilidad** — semillas, partición, orden de clientes y topología: cuatro fuentes de varianza adicionales.

## 11.11 Preguntas abiertas de investigación
- **11.11.1 Heterogeneidad extrema** — garantías de convergencia con no-IID severo y participación muy parcial.
- **11.11.2 Frontera privacidad-utilidad-comunicación** — caracterizarla como superficie de Pareto, no como puntos aislados.
- **11.11.3 Equidad frente a clientes minoritarios** — modelos globales que no penalicen la cola de la distribución.
- **11.11.4 Leyes de escalado descentralizadas** — cómo cambian los exponentes de escalado cuando la sincronización es infrecuente.
- **11.11.5 Deriva en pasos externos** — por qué crece con el tamaño del modelo y cómo corregirla.
- **11.11.6 Federado vertical a escala** — alineamiento privado de entidades (private set intersection) y su coste.
- **11.11.7 Robustez frente a participantes maliciosos** — defensas que no sacrifiquen a los honestos, en redes abiertas y sin agregador.
- **11.11.8 Incentivos y valoración de contribución** — mecanismos veraces y eficientes de reparto de crédito.
- **11.11.9 Verificabilidad del entrenamiento** — probar criptográficamente que un nodo hizo el trabajo que declara.
- **11.11.10 Puente federado ↔ data spaces** — traducir contratos de uso en restricciones ejecutables sobre el entrenamiento. Poca literatura, mucha demanda industrial.
- **11.11.11 Sostenibilidad y ubicación** — coste energético y de carbono del entrenamiento repartido geográficamente; cuándo compensa de verdad.
- **11.11.12 Geometría del promedio de pesos** — por qué promediar modelos entrenados por separado funciona, y cuándo deja de hacerlo.

---

# RUTA DE APRENDIZAJE

| # | Bloque | Secciones | Entregable |
|---|---|---|---|
| **0** | La neurona biológica y la ruptura de la analogía | Parte 1 | Nota conceptual + tabla comparativa |
| **1** | Perceptrón, MLP, la neurona artificial | 0.1, 3.1–3.2, 5.1 | MLP desde cero en NumPy |
| **2** | Backprop = regla de la cadena; autodiff reverso | 2.1–2.2, 4.3 | Motor de autodiff propio |
| **3** | Optimización, pérdidas, regularización, diagnóstico | 2.3–2.5, 3.4–3.6, 4.1–4.8 | Entrenamiento estable + curvas + ablations |
| **4** | Datos en rejilla: CNN | 3.3, 5.2–5.5 | CNN en PyTorch |
| **5** | Datos en secuencia: RNN → LSTM → seq2seq | 6.1–6.3 | Modelo de secuencias |
| **6** | Atención y Transformer | 3.7, 6.4–6.8 | Transformer desde cero |
| **7** | Modelos base y escala | Parte 9 | Fine-tuning con LoRA |
| **8** | Modelos generativos | Parte 7 | Difusión mínima |
| **9** | Grafos y simetrías | Parte 8 | GNN + síntesis unificadora |
| **10** | Refuerzo y alineamiento | Parte 10 | Agente PPO |
| **11** | **Arquitecturas de entrenamiento, federado y descentralizado** ★ | Parte 11 | FedAvg propio + experimento no-IID + DiLoCo mínimo sobre dos nodos + revisión de literatura |
| **∞** | Teoría abierta | Anexo A | Línea de investigación |

**Nota sobre el bloque 11.** Si la tesis va por ahí, se puede adelantar justo
después del bloque 4: basta saber entrenar una CNN para federarla. Los bloques
5–10 se retoman después. El orden de la tabla es pedagógico, no obligatorio.

**Criterio de avance:** (1) implementarlo desde cero sin copiar, y
(2) explicar por qué la pieza anterior era insuficiente.

---

# ANEXO A — TEORÍA ABIERTA

## A.1 Generalización
- **A.1.1 La paradoja** — modelos sobreparametrizados que no sobreajustan.
- **A.1.2 Doble descenso** — el error baja, sube y vuelve a bajar con la capacidad.
- **A.1.3 Regularización implícita del SGD**.
- **A.1.4 Mínimos planos vs agudos** — y la disputa sobre si predicen generalización.
- **A.1.5 Límites de la teoría clásica** — VC y Rademacher predicen mal aquí.
- **A.1.6 Grokking** — generalización que aparece mucho después de memorizar.

## A.2 Optimización
- **A.2.1 Por qué el SGD funciona en un paisaje no convexo**.
- **A.2.2 Conectividad de modos** — los mínimos están conectados por caminos de baja pérdida.
- **A.2.3 Neural Tangent Kernel** — el régimen de anchura infinita y sus límites.
- **A.2.4 Dinámica del entrenamiento** — fases, transiciones, edge of stability.

## A.3 Expresividad y arquitectura
- **A.3.1 Aproximación universal** — existencia, no aprendibilidad.
- **A.3.2 Ventaja de la profundidad** — separaciones exponenciales.
- **A.3.3 Límites de expresividad de GNN** — jerarquía Weisfeiler-Lehman.
- **A.3.4 Qué computa realmente un Transformer** — expresividad formal y límites de razonamiento.

## A.4 Esparsidad y compresión
- **A.4.1 Lottery ticket hypothesis** — subredes entrenables desde la inicialización.
- **A.4.2 Pruning** — hasta dónde se puede recortar.
- **A.4.3 Destilación** — por qué el estudiante puede igualar al profesor.

## A.5 Interpretabilidad
- **A.5.1 Interpretabilidad mecanicista** — ingeniería inversa de circuitos.
- **A.5.2 Superposición** — más características que neuronas.
- **A.5.3 Autoencoders dispersos** — características monosemánticas.
- **A.5.4 Sondas y edición causal** — activation patching, ablación causal.
- **A.5.5 Atribución** — saliency, gradientes integrados, SHAP; y sus fallos.
- **A.5.6 Fidelidad de la cadena de razonamiento** — si el razonamiento mostrado es el real.

## A.6 Robustez y fiabilidad
- **A.6.1 Ejemplos adversariales**.
- **A.6.2 Entrenamiento adversarial y certificación**.
- **A.6.3 Fuera de distribución** — detección, generalización, shift.
- **A.6.4 Incertidumbre** — epistémica vs aleatoria; ensembles, conformal.
- **A.6.5 Alucinación** — causas, medición, mitigación.
- **A.6.6 Atajos y correlaciones espurias**.

## A.7 Aprendizaje distribuido y descentralizado
- **A.7.1 Convergencia bajo no-IID** — garantías realistas, no asintóticas.
- **A.7.2 Límites fundamentales de privacidad** — cuánta utilidad cuesta cada ε.
- **A.7.3 Teoría del consenso** — cómo la topología determina la velocidad de convergencia.
- **A.7.4 Por qué funciona promediar pesos** — y en qué régimen deja de funcionar.
- **A.7.5 Teoría de incentivos** — mecanismos veraces entre competidores.

## A.8 Aprendizaje continuo y memoria
- **A.8.1 Olvido catastrófico** — por qué el gradiente sobrescribe.
- **A.8.2 Estabilidad vs plasticidad** — el compromiso fundamental, también en biología (→ 1.5.4).
- **A.8.3 Memorias a múltiples escalas temporales** — módulos que se actualizan a distintas frecuencias.
- **A.8.4 Aprendizaje en contexto vs en pesos** — dos memorias distintas y cuándo usar cada una.

## A.9 Alineamiento, ética y sociedad
- **A.9.1 Especificación de objetivos** — la brecha entre lo medido y lo deseado.
- **A.9.2 Sesgos** — de datos, de representación, de despliegue; métricas de equidad.
- **A.9.3 Evaluación de capacidades peligrosas**.
- **A.9.4 Privacidad y memorización** — extracción de datos de entrenamiento.
- **A.9.5 Impacto ambiental y restricciones energéticas**.
- **A.9.6 Gobernanza** — regulación, auditoría, trazabilidad.

---

# ANEXO B — CRONOLOGÍA

```none
┌─ RAÍCES BIOLÓGICAS Y PRIMERAS IDEAS ─────────────────────────────────┐
│                                                                       │
1943  McCulloch & Pitts .................. neurona lógica de umbral
  │
1949  Hebb ............................... regla de plasticidad
  │
1952  Hodgkin & Huxley .................. modelo del potencial de acción
  │
1958  Rosenblatt ........................ PERCEPTRÓN
  │
1959  Hubel & Wiesel .................... campos receptivos
  │
1969  Minsky & Papert ................... XOR ⛔ PRIMER INVIERNO
  │
└───────────────────────────────────────────────────────────────────────┘
  │
┌─ LA MAQUINARIA DEL APRENDIZAJE ──────────────────────────────────────┐
  │
1970  Linnainmaa ........................ autodiff en modo reverso
  │
1974  Werbos ............................ backprop aplicada a redes
  │
1980  Fukushima ......................... Neocognitron (CNN sin backprop)
  │
1986  Rumelhart, Hinton & Williams ...... backprop popularizada
  │
1989  LeCun ............................. LeNet: CNN + backprop
  │
1997  Hochreiter & Schmidhuber .......... LSTM
  │
1998  ................................... ⛔ SEGUNDO INVIERNO
  │                                          (dominan SVM y boosting)
2006  Hinton ............................ deep belief nets: el deshielo
2006  Dwork et al. ...................... privacidad diferencial
  │
2011  Glorot et al. ..................... ReLU
  │
└───────────────────────────────────────────────────────────────────────┘
  │
┌─ LA ERA MODERNA ─────────────────────────────────────────────────────┐
  │
2012  Krizhevsky et al. ................. ALEXNET ★ punto de inflexión
  │
  ├── VISIÓN ────────────────────────────────────────────────
  │   2014  VGG · Inception
  │   2015  He et al. ..................... ResNet ★ residuales
  │   2015  Ioffe & Szegedy ............... BatchNorm
  │
  ├── GENERATIVO ────────────────────────────────────────────
  │   2013  Kingma & Welling .............. VAE
  │   2014  Goodfellow .................... GAN
  │   2020  Ho et al. ..................... DDPM (difusión)
  │   2021  Radford et al. ................ CLIP
  │   2022  Rombach et al. ................ Latent Diffusion
  │
  ├── SECUENCIAS ────────────────────────────────────────────
  │   2014  Sutskever ..................... Seq2seq
  │   2014  Bahdanau ...................... atención
  │   2017  Vaswani et al. ................ TRANSFORMER ★
  │   2018  Devlin / OpenAI ............... BERT y GPT
  │   2020  Dosovitskiy et al. ............ ViT
  │
  ├── GRAFOS ────────────────────────────────────────────────
  │   2016  Kipf & Welling ................ GCN
  │   2021  Bronstein et al. .............. Geometric Deep Learning
  │
  ├── ESCALA ────────────────────────────────────────────────
  │   2014  Kingma & Ba ................... Adam
  │   2020  Kaplan et al. ................. leyes de escalado
  │   2022  Hoffmann et al. ............... Chinchilla
  │   2022  Ouyang et al. ................. InstructGPT / RLHF
  │
  └── DESCENTRALIZADO ───────────────────────────────────────
      2016  McMahan et al. ............... FedAvg ★ aprendizaje federado
      2016  Abadi et al. ................. DP-SGD
      2017  Bonawitz et al. .............. agregación segura
      2019  Kairouz et al. ............... "Open Problems in Federated Learning"
  │
└───────────────────────────────────────────────────────────────────────┘
  │
┌─ 2023 EN ADELANTE ───────────────────────────────────────────────────┐
  │
2023  ├── Gu & Dao ....................... Mamba / SSM: vuelve la recurrencia
      ├── Peebles & Xie .................. DiT ★ difusión sobre Transformer
      ├── Rafailov et al. ................ DPO: alineamiento sin RL explícito
      ├── Lipman et al. .................. flow matching
      ├── LLaMA · Mixtral ................ modelos abiertos y MoE
      └── Douillard et al. ............... DiLoCo ★ entrenamiento con poca
  │                                        comunicación (FedAvg para LLM)
2024  ├── Nobel de Física ................ Hopfield y Hinton
      ├── Nobel de Química ............... Hassabis, Jumper y Baker (AlphaFold)
      ├── AlphaFold 3 .................... interacciones biomoleculares
      ├── Llama 3 (405B) ................. escala abierta y 4D parallelism
      ├── o1 ............................. cómputo en tiempo de inferencia
      ├── Titans ......................... memoria a largo plazo aprendida
      ├── OpenDiLoCo ..................... DiLoCo abierto y asíncrono
      └── INTELLECT-1 .................... primer modelo de 10B entrenado
  │                                        con recursos descentralizados
2025  ├── DeepSeek-R1 .................... razonamiento por RL, abierto y
      │                                    documentado; punto de inflexión
      ├── RL con recompensa verificable .. GRPO y destilación de razonamiento
      ├── Scaling laws for DiLoCo ........ el método escala de forma predecible
      ├── INTELLECT-2 .................... RL descentralizado a escala global
      ├── Enjambres de RL ................ post-entrenamiento colaborativo
      ├── Difusión descentralizada ....... expertos por partición del dataset
      └── Nested Learning / HOPE ......... memoria continua contra el olvido
  │
2026  ├── Modelos de mundo ............... JEPA y sistemas tipo Genie:
      │                                    aprender física por predicción
      ├── Aprendizaje continuo ........... primeros prototipos fiables
      ├── Arquitecturas híbridas ......... atención + recurrencia + memoria
      │                                    desplazan al Transformer puro
      ├── Inteligencia en el borde ....... razonamiento destilado en dispositivo
      ├── DiLoCoX y sucesores ............ entrenamiento entre clústeres
      │                                    con ancho de banda limitado
      └── Restricción energética ......... la potencia eléctrica como límite
  │                                        de planificación, no el cómputo
  ▼
2027  HORIZONTE ABIERTO (líneas activas, aún sin resultado consolidado)
      · agentes de largo horizonte con memoria persistente
      · aprendizaje continuo sin olvido en producción
      · modelos de mundo como sustrato de la robótica
      · entrenamiento descentralizado como alternativa real al centro de datos
      · verificabilidad e incentivos en redes de entrenamiento abiertas
      · gobernanza ejecutable del dato (data spaces operativos)
```

---

# ANEXO C — TÉRMINOS DE INGENIERÍA (compacto)

Solo para ubicación al leer papers de sistemas.

- **Datos** — versionado (DVC), deduplicación, curación, formatos columnar (Parquet, Arrow, WebDataset), dataloaders y cuellos de I/O.
- **Cómputo** — precisión mixta (BF16/FP8), gradient checkpointing, MFU, perfilado, compilación y fusión de kernels (`torch.compile`, Triton).
- **Experimentación** — tracking (MLflow), visualización (TensorBoard), configuración declarativa (Hydra), búsqueda de hiperparámetros (Optuna, ASHA).
- **Reproducibilidad** — experimento = código + datos + configuración + semilla + entorno; contenedores y lockfiles; determinismo limitado en GPU.
- **Despliegue** — serving y batching, exportación (ONNX, safetensors), deriva de datos y de concepto, coste por inferencia.
