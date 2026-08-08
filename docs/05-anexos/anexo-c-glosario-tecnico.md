---
title: "Anexo C — Glosario técnico"
tags: [glosario, referencia]
status: revisado
---

# Anexo C — Glosario técnico

> **Para qué sirve:** lees un paper, aparece un término que no ubicas, lo
> buscas aquí. Cada entrada dice qué es en una línea y a qué sección de la
> guía pertenece. Organizado por familias, no alfabéticamente, para que los
> términos vecinos se aprendan juntos.

## C.1 Activaciones (→ 3.1, 5.3.1)
| Término | Qué es |
|---|---|
| sigmoide, tanh | Las históricas; saturan y matan el gradiente en profundidad |
| **ReLU** | max(0,x); el desbloqueo práctico de 2011 |
| LeakyReLU, PReLU, ELU, SELU | Variantes que evitan la "neurona muerta" |
| GELU, SiLU/Swish, Mish | Las suaves modernas; estándar en Transformers |
| GLU, GeGLU, **SwiGLU** | Con puerta; el bloque feed-forward de los LLM actuales |
| softmax | Convierte puntuaciones en probabilidades; capa de salida de clasificación |

## C.2 Inicialización (→ 3.1)
| Término | Qué es |
|---|---|
| Xavier/Glorot | Calibrada para tanh/sigmoide; preserva la varianza de la señal |
| He/Kaiming | Calibrada para ReLU |
| ortogonal | Para recurrentes y redes muy profundas |
| inicialización a cero | Rompe el aprendizaje: todas las neuronas quedan idénticas (simetría) |

## C.3 Normalización (→ 3.4.6)
| Término | Qué es |
|---|---|
| **BatchNorm** (2015) | Normaliza sobre el lote; falla con lotes pequeños o secuencias |
| **LayerNorm** (2016) | Normaliza sobre las características; estándar en Transformers |
| RMSNorm | LayerNorm sin centrado; más barata, usada en LLaMA |
| GroupNorm, InstanceNorm | Alternativas para visión y transferencia de estilo |
| pre-norm vs post-norm | Dónde va la norma respecto al bloque; decide la estabilidad en profundidad |

## C.4 Optimizadores y planificación (→ 3.3)
| Término | Qué es |
|---|---|
| SGD, momentum, Nesterov | La base y sus dos mejoras de inercia |
| AdaGrad → RMSProp → Adam | La cadena de los adaptativos (tasa por parámetro) |
| **AdamW** | Adam con weight decay desacoplado; el estándar actual |
| LAMB, Lion, Adafactor, Muon | Especializados: lotes enormes, memoria, segundo orden aproximado |
| K-FAC, Shampoo | Aproximaciones de segundo orden (curvatura) |
| warmup | Subir la tasa de aprendizaje gradualmente al inicio; imprescindible en Transformers |
| cosine decay, one-cycle | Formas de bajar la tasa a lo largo del entrenamiento |
| gradient clipping | Techo al tamaño del paso para evitar explosiones |
| μP (muP) | Transferir hiperparámetros de un modelo pequeño a uno grande |
| grid / random / bayesiana / ASHA | Estrategias de búsqueda de hiperparámetros, de peor a mejor |

## C.5 Regularización y patologías (→ 3.4, 3.5)
| Término | Qué es |
|---|---|
| weight decay / L2, L1 | Penalizar pesos grandes / inducir esparsidad |
| **dropout** | Apagar neuronas al azar en entrenamiento |
| label smoothing | Suavizar las etiquetas para no dar confianza extrema |
| mixup, CutMix, RandAugment | Aumentos de datos modernos |
| early stopping | Parar cuando validación deja de mejorar |
| vanishing / exploding gradient | La señal de error se apaga / se desboca en profundidad |
| dead ReLU | Neurona permanentemente a cero |
| colapso de modo | El generador (GAN) produce poca variedad |
| colapso de representación | Todas las salidas convergen al mismo vector |
| olvido catastrófico | Aprender lo nuevo borra lo viejo |
| data leakage | Información del test se cuela en el entrenamiento |
| exposure bias | El modelo entrenó viendo la verdad y en inferencia ve sus propios errores |

## C.6 Evaluación (→ 3.6)
| Término | Qué es |
|---|---|
| precision, recall, F1 | Acierto en lo dicho / cobertura de lo real / su media armónica |
| ROC-AUC, PR-AUC | Área bajo curvas; comparar clasificadores sin fijar umbral |
| IoU, mAP | Métricas de detección y segmentación en visión |
| BLEU, ROUGE | Métricas de traducción y resumen |
| perplejidad | Cómo de "sorprendido" queda un modelo de lenguaje; menor es mejor |
| FID | Distancia entre imágenes generadas y reales |
| calibración | Que un 80 % de confianza acierte el 80 % de las veces |
| k-fold, estratificada, temporal | Variantes de validación cruzada |
| ablation | Quitar una pieza para medir su aporte |

## C.7 Piezas de arquitectura (→ 5.3)
| Término | Qué es |
|---|---|
| capa afín / densa / fully-connected | `y = Wx + b`; todo con todo |
| **conexión residual** (ResNet) | `y = x + f(x)`; la autopista que permite la profundidad |
| skip connection | Saltos entre capas (U-Net) para recuperar detalle |
| gating | Puertas que regulan el flujo (LSTM, Highway) |
| bottleneck | Estrechar a propósito para forzar compresión |
| **MoE** (mezcla de expertos) | Activar solo una parte de la red por token; parámetros baratos |
| kernel, stride, padding, dilatación | Los mandos de la convolución |
| pooling (max, average, global) | Reducir resolución quedándose con lo importante |
| campo receptivo | Cuánto de la entrada "ve" una neurona |
| BPTT | Backprop desplegando la red en el tiempo (RNN) |
| teacher forcing | Entrenar la RNN alimentándole la verdad en vez de su propia salida |
| beam search, top-k, top-p, temperatura | Estrategias para generar texto |
| query / key / value | La abstracción de la atención: qué busco / qué ofrezco / qué llevo |
| multi-head attention | Varias atenciones en paralelo sobre subespacios distintos |
| máscara causal | Prohibir mirar al futuro en generación |
| cross-attention | El decodificador consulta al codificador |
| positional encoding, **RoPE**, ALiBi | Formas de inyectar el orden en el Transformer |
| tokenización, **BPE**, SentencePiece | Trocear texto en unidades; la frontera texto↔tensor |
| embedding | Vector denso que representa un token o entidad |
| FlashAttention | Atención exacta reescrita para la jerarquía de memoria de la GPU |
| MQA, GQA | Compartir claves/valores entre cabezas para reducir la KV cache |
| KV cache | Guardar claves y valores ya computados al generar; el cuello de memoria |
| SSM, **Mamba**, S4 | Modelos de espacio de estados: recurrencia paralelizable |
| Titans, Nested Learning | Módulos de memoria a varias frecuencias contra el olvido |

## C.8 Modelos generativos (→ 5.3.5)
| Término | Qué es |
|---|---|
| ELBO | La cota que optimiza un VAE (reconstrucción + KL) |
| truco de reparametrización | Hacer diferenciable el muestreo del latente |
| posterior collapse | El VAE ignora su latente |
| VQ-VAE | Latente discreto; base de generar por tokens |
| discriminador / generador | Los dos jugadores de una GAN |
| WGAN | GAN con distancia de Wasserstein; más estable |
| StyleGAN, CycleGAN, pix2pix | GANs célebres: caras, traducción de dominio |
| DDPM, DDIM | Difusión original / su muestreo acelerado |
| score matching | La formulación continua que unifica la difusión |
| classifier-free guidance | Controlar cuánto obedece la difusión al prompt |
| latent diffusion | Difundir en el latente comprimido (Stable Diffusion) |
| **DiT** | Difusión con Transformer en vez de U-Net |
| flow matching / rectified flow | La formulación simplificada emergente |
| normalizing flow | Transformaciones invertibles con probabilidad exacta |
| autoregresivo | Generar pieza a pieza condicionando en lo anterior |
| JEPA, world model, Dreamer, Genie | Aprender un modelo del mundo prediciendo el futuro |

## C.9 Grafos y simetrías (→ 5.3.6)
| Término | Qué es |
|---|---|
| message passing | Cada nodo agrega info de sus vecinos y se actualiza |
| GCN, GraphSAGE, GAT | Los tres modelos base de grafos |
| over-smoothing / over-squashing | Con profundidad, los nodos se igualan / la info se comprime demasiado |
| Weisfeiler-Lehman | El test que marca el límite de lo que una GNN distingue |
| invarianza / equivarianza | La salida no cambia / cambia igual que la entrada |
| E(3)/SE(3)-equivariante | Redes que respetan rotaciones y traslaciones 3D (moléculas) |
| Deep Sets | Funciones sobre conjuntos sin orden |
| PINN, neural operator (FNO) | Redes con física en la pérdida / que aprenden operadores |

## C.10 Modelos gigantes (→ 5.3.7)
| Término | Qué es |
|---|---|
| foundation model | Modelo base genérico adaptable a muchas tareas |
| leyes de escalado, Chinchilla | El error baja predeciblemente con tamaño/datos/cómputo; y la receta corregida |
| SFT | Fine-tuning supervisado por instrucciones |
| RLHF, **DPO**, RLAIF, constitucional | Familias de alineamiento con preferencias |
| GRPO, RLVR | RL sobre razonamiento con recompensa verificable |
| **LoRA**, QLoRA, adapters, prompt tuning | Adaptar tocando pocos parámetros (PEFT) |
| destilación | Un modelo pequeño imita al grande |
| cuantización, GPTQ, AWQ, INT8/INT4 | Comprimir los pesos para inferencia |
| pruning | Podar pesos o estructuras enteras |
| decodificación especulativa | Un modelo pequeño propone, el grande verifica |
| in-context learning, few-shot | Aprender de ejemplos puestos en el prompt, sin entrenar |
| chain-of-thought, test-time compute | Razonar por pasos; gastar cómputo al responder |
| RAG | Buscar documentos y generar con ellos |
| contaminación de benchmarks | El test ya estaba en el corpus de pre-entrenamiento |

## C.11 Refuerzo (→ 4.4)
| Término | Qué es |
|---|---|
| MDP, política, valor, Q, ventaja | El vocabulario formal del RL |
| ecuación de Bellman | La recursión que define el valor óptimo |
| ε-greedy, UCB | Estrategias de exploración |
| Q-learning, SARSA, DQN | Métodos basados en valor (DQN = Q-learning + red + replay) |
| experience replay | Buffer de experiencias pasadas para reutilizar (→ 1.2.5, replay biológico) |
| REINFORCE, policy gradient | Optimizar la política directamente |
| actor-crítico, A2C, SAC, TD3 | Política (actor) + evaluador (crítico) |
| TRPO, **PPO** | Limitar cuánto cambia la política por paso; PPO es el estándar |
| model-based, MuZero, Dreamer | Aprender el modelo del entorno y planificar dentro |
| **CQL, IQL** | Offline RL: pesimismo ante acciones no vistas |
| Decision Transformer | RL como modelado de secuencias |
| OOD actions | Acciones fuera de la distribución del dataset; el problema del offline |
| behavioral cloning, inverse RL | Imitar demostraciones / inferir la recompensa del comportamiento |
| reward hacking | Optimizar la métrica destruyendo el objetivo |

## C.12 Distribuido, federado y descentralizado (→ 6)
| Término | Qué es |
|---|---|
| data / tensor / pipeline / expert parallelism | Las cuatro formas de repartir el entrenamiento |
| ZeRO, FSDP | Fragmentar optimizador, gradientes y parámetros entre GPUs |
| all-reduce, all-gather, NCCL | Las operaciones y la librería de comunicación colectiva |
| NVLink, InfiniBand, RoCE | Las interconexiones físicas |
| MFU | Fracción del hardware realmente aprovechada |
| straggler, staleness | Nodo rezagado / gradiente que llega viejo |
| **FedAvg** | Promediar modelos tras pasos locales; el algoritmo de referencia |
| FedSGD, FedProx, SCAFFOLD, FedNova | La familia: base teórica y correcciones al drift |
| FedOpt / FedAdam | Optimizador adaptativo en el lado servidor |
| **no-IID** | Cada cliente tiene distribución distinta; EL problema del federado |
| client drift | Los modelos locales se desvían durante los pasos locales |
| partición de Dirichlet | Cómo se simula el no-IID en los papers |
| cross-device / cross-silo | Millones de móviles / decenas de organizaciones |
| horizontal / vertical / transfer FL | Según cómo se solapan filas y columnas entre dueños |
| Per-FedAvg, FedPer, FedRep | Familias de personalización |
| destilación federada | Compartir salidas en vez de pesos; permite modelos distintos por cliente |
| gossip, D-PSGD | Aprendizaje sin servidor: intercambio entre vecinos |
| matriz de mezcla, spectral gap | La topología y su propiedad que gobierna la convergencia |
| swarm learning | Coordinación por blockchain, sin agregador |
| **DiLoCo**, OpenDiLoCo, DiLoCoX | Pre-entrenar LLM con sincronización cada cientos de pasos |
| pseudo-gradiente | El desplazamiento de pesos de un tramo local, tratado como gradiente |
| INTELLECT-1/-2 | Los hitos del entrenamiento con recursos dispersos (2024-25) |
| split learning, SplitFed | Cortar la red entre cliente y servidor; viajan activaciones |
| federated LoRA | Federar solo los adaptadores; el caso LLM+federado |

## C.13 Privacidad y seguridad (→ 6.5)
| Término | Qué es |
|---|---|
| **privacidad diferencial**, (ε, δ) | La garantía formal y su presupuesto |
| DP-SGD | Recorte por muestra + ruido gaussiano |
| moments accountant | Contabilizar cuánto presupuesto ε se ha gastado |
| **secure aggregation** | El servidor ve la suma, no las partes |
| cifrado homomórfico, MPC, TEE | Computar cifrado / conjunto sin revelar / enclave hardware |
| gradient inversion | Reconstruir el dato desde el gradiente |
| membership inference | ¿Estuvo este registro en el entrenamiento? |
| poisoning, backdoor | Envenenar el entrenamiento / puerta trasera con disparador |
| free-riding, Sybil | Beneficiarse sin aportar / suplantar muchos nodos |
| Krum, trimmed mean, median | Agregaciones robustas frente a maliciosos |
| private set intersection | Alinear entidades entre dueños sin revelar las listas (FL vertical) |

## C.14 Data spaces (→ 6.6)
| Término | Qué es |
|---|---|
| soberanía del dato | Control del dueño sobre el uso, incluso tras compartir |
| conector (EDC, IDS) | El componente que media todo intercambio |
| ODRL | Lenguaje de políticas de uso ejecutables |
| IDSA, **Gaia-X**, FIWARE | Las iniciativas de referencia europeas |
| Pontus-X, Ocean, compute-to-data | Tu stack: el cómputo viaja al dato, no al revés |
| EHDS | El espacio de datos sanitarios europeo |
| RGPD, Data Act, AI Act | El marco legal |
| valor de Shapley | Repartir el mérito de la contribución de cada participante |
| credencial verificable | Identidad certificada criptográficamente |

## C.15 Ingeniería (mínimo para ubicar papers de sistemas)
| Término | Qué es |
|---|---|
| FP32/FP16/**BF16**/FP8, mixed precision | Formatos numéricos y entrenar mezclándolos |
| gradient checkpointing | Recomputar activaciones en vez de guardarlas |
| torch.compile, Triton, fusión de kernels | Compilar y fusionar operaciones para la GPU |
| DVC, lockfile, contenedor | Versionar datos, fijar entornos, reproducibilidad |
| MLflow, TensorBoard, Hydra, Optuna | Tracking, visualización, configuración, búsqueda |
| ONNX, safetensors | Formatos de exportación de modelos |
| Flower, FedML, NVIDIA FLARE, OpenFL | Frameworks federados |
| LEAF, FedScale, FLamby | Benchmarks federados (FLamby: médico) |
| deriva de datos / de concepto | El mundo cambia y el modelo desplegado se degrada |
