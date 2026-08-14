---
title: "Deep Learning — Base de Conocimiento"
---

# Deep Learning — Índice Maestro

Bienvenido al repositorio de conocimiento del doctorado, organizado en **5 bloques temáticos**. Navega por las pestañas superiores o usa los enlaces de abajo como mapa rápido.

---

## I. Fundamentos

Contexto, matemáticas y biología que dan base al Deep Learning.

- [1. Qué es y de dónde viene](01-fundamentos/01-que-es-y-de-donde-viene/index.md)
- [2. Las matemáticas necesarias](01-fundamentos/02-las-matematicas-necesarias/index.md)
- [3. Cómo se entrena una red](01-fundamentos/03-como-se-entrena-una-red/index.md)

---

## II. Deep Learning

Los paradigmas de aprendizaje, las familias de modelos y dónde están los datos.

- [4. Formas de aprender](02-deep-learning/04-formas-de-aprender/index.md)
- [5. Tipos de modelo](02-deep-learning/05-tipos-de-modelo/index.md)
- [6. Dónde están los datos](02-deep-learning/06-donde-estan-los-datos/index.md)

---

## III. Data Spaces

La capa de gobernanza que hace posible el aprendizaje federado entre organizaciones distintas.

- [1. Introducción — Por qué los datos no se pueden centralizar](03-data-spaces/01-introduccion/index.md)
- [2. Aprendizaje Federado](03-data-spaces/02-aprendizaje-federado/index.md)
- [3. Data Spaces y Gobernanza](03-data-spaces/03-data-spaces-gobernanza/index.md)
- [4. Paradigmas FL en Data Spaces](03-data-spaces/04-paradigmas-fl/index.md)
- [5. Retos de Investigación](03-data-spaces/05-retos-investigacion/index.md)

---

## IV. Proyecto

El laboratorio práctico: combinaciones, guía de entorno, notebooks y diario de experimentación.

- [7. Combinaciones](04-proyecto/07-combinaciones/index.md)
- [8. Guía de Ejecución](04-proyecto/08-guia-entorno/index.md)
- [9. Índice de Notebooks](04-proyecto/09-indice-notebooks/index.md)
- [10. Diario / Bitácora](04-proyecto/10-bitacora-experimentos/index.md)
- [11. Federado sobre Data Spaces](04-proyecto/11-federado-sobre-data-spaces/11-indice.md)
- [12. Federado con FELT Labs](04-proyecto/12-federado-con-felt-labs/12-indice.md)
- [13. Herramientas y variantes](04-proyecto/13-herramientas-y-variantes/13-indice.md)
- [14. Validación por silo retenido](04-proyecto/14-validacion-por-silo-retenido/14-indice.md)
- [15. Laboratorio de modelos en el editor](04-proyecto/15-laboratorio-de-modelos-en-el-editor/15-indice.md)

---

## V. Anexos

- [A. Lo que nadie sabe todavía](05-anexos/anexo-a-preguntas-abiertas.md)
- [B. Línea temporal](05-anexos/anexo-b-linea-temporal.md)
- [C. Glosario técnico](05-anexos/anexo-c-glosario-tecnico.md)
- [D. Búsqueda bibliográfica WoS](05-anexos/anexo-d-guia-busqueda-wos.md)
- [E. Sistemas emergentes y redes neuronales](05-anexos/anexo-e-sistemas-emergentes.md)

---

## Mapa del repositorio

```none
docs/
├── 01-fundamentos/                          ──► I. Fundamentos
│   ├── 01-que-es-y-de-donde-viene/          │   ├─► 1.1 DL dentro de la informática
│   │                                        │   ├─► 1.2 ML antes del DL
│   │                                        │   ├─► 1.3 La neurona biológica
│   │                                        │   ├─► 1.4 Del perceptrón a GPT
│   │                                        │   ├─► 1.5 Las tres preguntas
│   │                                        │   └─► 1.6 RL Offline — el problema
│   ├── 02-las-matematicas-necesarias/       │   ├─► 2.0 Notación y formas
│   │                                        │   ├─► 2.1 Álgebra lineal
│   │                                        │   ├─► 2.2 Derivadas y regla de la cadena
│   │                                        │   ├─► 2.3 Probabilidad
│   │                                        │   ├─► 2.4 Optimización
│   │                                        │   ├─► 2.5 Teoría de la información
│   │                                        │   ├─► 2.6 Estadística e inferencia
│   │                                        │   ├─► 2.7 Geometría en alta dimensión
│   │                                        │   └─► 2.8 MDP y Bellman
│   └── 03-como-se-entrena-una-red/          │   ├─► 3.0 Los datos
│                                            │   ├─► 3.1 El ciclo de entrenamiento
│                                            │   ├─► 3.2 Backpropagation
│                                            │   ├─► 3.3 Pérdidas y optimizadores
│                                            │   ├─► 3.4 Regularización
│                                            │   ├─► 3.5 Precisión y memoria
│                                            │   ├─► 3.6 Escalar el entrenamiento
│                                            │   ├─► 3.7 Transferencia y fine-tuning
│                                            │   ├─► 3.8 Problemas típicos
│                                            │   └─► 3.9 Cómo medir si funciona
│
├── 02-deep-learning/                        ──► II. Deep Learning
│   ├── 04-formas-de-aprender/               │   ├─► 4.1 Supervisado
│   │                                        │   ├─► 4.2 No supervisado
│   │                                        │   ├─► 4.3 Auto-supervisado
│   │                                        │   ├─► 4.4 Imitación
│   │                                        │   ├─► 4.5 Preferencias
│   │                                        │   ├─► 4.6 Refuerzo
│   │                                        │   ├─► 4.7 Destilación
│   │                                        │   ├─► 4.8 El pipeline real
│   │                                        │   └─► 4.9 Modelos de razonamiento
│   ├── 05-tipos-de-modelo/                  │   ├─► 5.0 Baselines y heurísticas
│   │                                        │   ├─► 5.1 Modelos clásicos
│   │                                        │   ├─► 5.2 Árboles y boosting
│   │                                        │   ├─► 5.3 Modelos probabilísticos
│   │                                        │   ├─► 5.4 Redes neuronales (historia)
│   │                                        │   │   ├─► 5.4.1 Perceptrón y MLP
│   │                                        │   │   ├─► 5.4.2 CNN para imágenes
│   │                                        │   │   ├─► 5.4.3 RNN y LSTM
│   │                                        │   │   ├─► 5.4.4 Atención y Transformer
│   │                                        │   │   ├─► 5.4.5 Modelos generativos
│   │                                        │   │   ├─► 5.4.6 Redes sobre grafos
│   │                                        │   │   └─► 5.4.7 Modelos gigantes
│   │                                        │   ├─► 5.5 Modelos preentrenados
│   │                                        │   └─► 5.6 Ensembles
│   └── 06-donde-estan-los-datos/            │   ├─► 6.1 Centralizado
│                                            │   ├─► 6.2 Distribuido
│                                            │   ├─► 6.3 Federado
│                                            │   ├─► 6.4 Descentralizado
│                                            │   ├─► 6.5 Privacidad y seguridad
│                                            │   └─► 6.6 Data Spaces
│
├── 03-data-spaces/                          ──► III. Data Spaces
│   ├── 01-introduccion/
│   ├── 02-aprendizaje-federado/
│   ├── 03-data-spaces-gobernanza/
│   ├── 04-paradigmas-fl/
│   └── 05-retos-investigacion/
│
├── 04-proyecto/                             ──► IV. Proyecto (Laboratorio)
│   ├── 07-combinaciones/                    │   ├─► 7.1 Cómo se mezclan
│   │                                        │   ├─► 7.2 Refuerzo + Federado
│   │                                        │   ├─► 7.3 Offline + Federado + Data Spaces
│   │                                        │   ├─► 7.4 Tabla de experimentos
│   │                                        │   ├─► 7.5 Aprendizaje federado en C2D
│   │                                        │   └─► 7.6 Guía práctica (fedlab + Ocean)
│   ├── 08-guia-entorno/                     │   └─► Guía Docker / Makefile
│   ├── 09-indice-notebooks/                 │   └─► Índice de notebooks
│   ├── 10-bitacora-experimentos/            │   └─► Diario de experimentos
│   ├── 11-federado-sobre-data-spaces/       │   ├─► 11.0–11.10 (ocean-node / C2D)
│   ├── 12-federado-con-felt-labs/           │   ├─► 12.0–12.8 (FELT Labs / doble ciego)
│   ├── 13-herramientas-y-variantes/         │   ├─► 13.0–13.9 (frameworks, LoRA, one-shot…)
│   ├── 14-validacion-por-silo-retenido/     │   ├─► 14.0–14.7 (LOSO, cobertura, OPE)
│   └── 15-laboratorio-de-modelos-en-el-editor/ │   └─► 15.1–15.7 (proxy, router, LLM lab…)
│
└── 05-anexos/                               ──► V. Anexos
    ├── anexo-a-preguntas-abiertas.md
    ├── anexo-b-linea-temporal.md
    ├── anexo-c-glosario-tecnico.md
    ├── anexo-d-guia-busqueda-wos.md
    └── anexo-e-sistemas-emergentes.md
```
