# 🎓 Base de Conocimiento, Estrategia y Laboratorio de Doctorado

> **Tema de Investigación:** Aprendizaje por Refuerzo Offline y Federado (*Federated Offline RL*) sobre Espacios de Datos Soberanos (*Data Spaces* / Gaia-X / Ocean Protocol / Pontus-X).

Este repositorio es la infraestructura central del doctorado. Combina la **estrategia de investigación**, una **base de conocimiento interactiva estructurada como libro**, el **laboratorio de experimentos reproducibles (PyTorch / CUDA)** y el **entorno de análisis bibliométrico (R / Biblioshiny)**.

---

## 🎯 1. Estrategia de Tesis (`estrategia-tesis-2026.md`)

El documento maestro que guía la línea de investigación y toma de decisiones de la tesis:

📄 **[estrategia-tesis-2026.md](estrategia-tesis-2026.md)** — *Documento vivo de la fase de exploración y posicionamiento.*

### Ejes de la Estrategia:
- **Parte A — Diagnóstico del campo:** Análisis del estado del arte en Aprendizaje Federado (FL), RL Offline y Data Spaces.
- **Parte B — Diez ideas de enfoque:** Alternativas de investigación evaluadas por factibilidad e impacto.
- **Parte B-bis & B-ter — Validación:** Definición de la evaluación offline en FL y validación por silo retenido (*Leave-One-Site-Out* / LOSO).
- **Parte C & D — Posicionamiento de impacto:** Qué está resuelto en la literatura y dónde residen las contribuciones originales (el cruce de decisiones $\text{Offline RL} \times \text{Federado} \times \text{Data Spaces}$).
- **Parte E & F — Recomendación y Hoja de Ruta:** Plan de acción prioritario y tareas acotadas.

---

## 📚 2. Base de Conocimiento (`docs/`)

La base de conocimiento está redactada en formato de libro técnico interactivo, con **navegación secuencial (`← Anterior` / `Siguiente →`)** a lo largo de sus **+116 capítulos**, organizada en 5 bloques conceptuales:

### 📖 Mapa de la Base de Conocimiento

#### [I. Fundamentos](docs/01-fundamentos/index.md)
Contexto histórico, marco conceptual de las decisiones de diseño de IA y base matemática:
- **[1. Qué es y de dónde viene](docs/01-fundamentos/01-que-es-y-de-donde-viene/index.md):** Del perceptrón a GPT, las 4 decisiones de diseño de IA (① Señal, ② Modelo, ③ Ubicación de datos, ④ Inferencia / *Test-time compute*) y la problemática del RL Offline.
- **[2. Las matemáticas necesarias](docs/01-fundamentos/02-las-matematicas-necesarias/index.md):** Álgebra lineal, cálculo multivariable, probabilidad, optimización, teoría de la información y MDPs / Ecuación de Bellman.
- **[3. Cómo se entrena una red](docs/01-fundamentos/03-como-se-entrena-una-red/index.md):** Ciclo de entrenamiento, backpropagation, pérdidas/optimizadores, regularización, precisión mixta, fine-tuning y métricas de evaluación.

#### [II. Deep Learning](docs/02-deep-learning/index.md)
Taxonomía detallada de aprendizaje, familias de modelos y arquitectura de datos:
- **[4. Formas de aprender](docs/02-deep-learning/04-formas-de-aprender/index.md):** Supervisado, No supervisado, Auto-supervisado, Imitación, Preferencias (RLHF/DPO), Refuerzo (RLVR), Destilación, El pipeline real y Coda: Modelos de razonamiento (*System 2 / Test-Time Compute*).
- **[5. Tipos de modelo](docs/02-deep-learning/05-tipos-de-modelo/index.md):** Baselines, modelos clásicos, boosting, probabilísticos, evolución de arquitecturas (MLP, CNN, RNN/LSTM, Transformers, Generativos, GNNs) y Ensembles.
- **[6. Dónde están los datos](docs/02-deep-learning/06-donde-estan-los-datos/index.md):** Centralizado, distribuido, federado, descentralizado, privacidad/seguridad y Data Spaces.

#### [III. Data Spaces](docs/03-data-spaces/index.md)
Capa de gobernanza, soberanía de datos y arquitecturas decentralizadas:
- **Introducción & FL:** Por qué los datos no se pueden centralizar y fundamentos de Aprendizaje Federado.
- **Gobernanza & Retos:** Arquitectura de Data Spaces (Pontus-X / IDS), Compute-to-Data (C2D) y la matriz de retos de investigación R1–R15.

#### [IV. Proyecto & Laboratorio Práctico](docs/04-proyecto/index.md)
Implementación experimental y guías del laboratorio:
- **7. Combinaciones:** La matriz 3D ($\text{Señal} \times \text{Modelo} \times \text{Ubicación}$), FRL, Offline + FL + Data Spaces y guías de Flower + Ocean C2D.
- **8 a 10. Entorno y Bitácora:** Guía de ejecución Docker/Makefile, índice de notebooks Jupyter y la bitácora diaria de experimentos.
- **11 & 12. Infraestructura Federada:** Pipeline C2D con `ocean-node` y protocolo de doble ciego con FELT Labs.
- **13 & 14. Herramientas y Validación:** Frameworks (Flower, TRL, vLLM), adaptadores LoRA, one-shot FL y protocolo de validación por silo retenido (LOSO / OPE).

#### [V. Anexos](docs/05-anexos/anexo-a-preguntas-abiertas.md)
- **[Anexo A — Preguntas Abiertas](docs/05-anexos/anexo-a-preguntas-abiertas.md):** Registro vivo de interrogantes abiertas clasificadas por madurez, coste y dependencias (A1–A20).
- **Anexos B a E:** Línea temporal del Deep Learning, glosario técnico, guía de búsqueda bibliográfica en Web of Science (WoS) y teoría de sistemas emergentes.

---

## 📁 Estructura del Repositorio

```none
PhD/
├── estrategia-tesis-2026.md  # Estrategia general y plan de investigación de la tesis
├── compose.yaml              # Orquestador Docker Compose (MkDocs + JupyterLab + MLflow)
├── mkdocs.yml                # Configuración del visor web MkDocs Material
├── Makefile                  # Automatización de comandos (build, docs, lab, sync)
├── docs/                     # BASE DE CONOCIMIENTO (Páginas Markdown enlazadas)
│   ├── 01-fundamentos/       # Bloque I: Historia, Matemáticas y Entrenamiento
│   ├── 02-deep-learning/     # Bloque II: Paradigmas, Arquitecturas y Razonamiento
│   ├── 03-data-spaces/       # Bloque III: Gobernanza, FL y Retos en Data Spaces
│   ├── 04-proyecto/          # Bloque IV: Experimentos, C2D, FELT y Bitácora
│   └── 05-anexos/            # Bloque V: Preguntas abiertas, Glosario y WoS
├── labs/
│   └── dl-lab/               # LABORATORIO DE COMPUTO (PyTorch / CUDA / MLflow)
│       ├── notebooks/        # Cuadernos Jupyter (.ipynb <-> .py sincronizados)
│       ├── src/              # Módulos Python reutilizables (dllab)
│       └── experiments/      # Ficheros YAML de configuración experimental
├── bibliometrix/             # ANÁLISIS BIBLIOMÉTRICO (R / Biblioshiny)
│   ├── data/                 # Datasets .bib exportados de Web of Science y Scopus
│   └── run_biblioshiny.R     # Script de ejecución de la interfaz R Shiny
└── docs-stack/               # Contenedor y recursos del servidor MkDocs
```

---

## 🛠️ Inicio Rápido y Comandos

### 1. Inicialización

```bash
cp .env.example .env
make init
```

### 2. Comandos habituales (`Makefile`)

| Comando | Servicio | URL / Descripción |
|---|---|---|
| `make up` | **Docs + Jupyter (CPU)** | Levanta el visor MkDocs y el entorno JupyterLab. |
| `make up-gpu` | **Docs + Jupyter (GPU)** | Levanta el entorno con soporte NVIDIA CUDA. |
| `make docs` | **Base de Conocimiento** | [http://localhost:8000](http://localhost:8000) (Visor interactivo MkDocs con recarga en vivo). |
| `make lab` | **JupyterLab** | [http://localhost:8888/lab?token=dev](http://localhost:8888/lab?token=dev) (Entorno de código). |
| `make bibliometrix` | **Bibliometrix** | Lanza la app local de R Biblioshiny con las búsquedas de `bibliometrix/data/`. |
| `make mlflow` | **MLflow Server** | [http://localhost:5000](http://localhost:5000) (Seguimiento de experimentos y artefactos). |
| `make tb` | **TensorBoard** | [http://localhost:6006](http://localhost:6006) (Visualizador de métricas de entrenamiento). |
| `make sync` | **Jupytext** | Sincroniza bidireccionalmente cuadernos `.ipynb` con scripts `.py`. |
| `make lint` | **Calidad de código** | Linter `ruff` para Python y `markdownlint` para documentación. |
| `make down` | **Apagar** | Detiene de forma limpia todos los servicios activos. |

---

## 🌐 Puertos de Servicios Locales

| Puerto | Servicio | Función |
|---|---|---|
| **8000** | **MkDocs Material** | Servidor web de la Base de Conocimiento interactiva. |
| **8888** | **JupyterLab** | Entorno para prototipado de experimentos y scripts. |
| **5000** | **MLflow** | Registro centralizado de parámetros y métricas de modelos. |
| **6006** | **TensorBoard** | Gráficas de convergencia y visualización de gradientes. |
| *Host* | **Biblioshiny** | Aplicación Shiny en R para mapas bibliométricos y análisis de citas. |

---

## 🔗 Control de Versiones

- **Repositorio Remoto:** [https://github.com/lopezmuzas/PhD](https://github.com/lopezmuzas/PhD)
- **Compilación estática de la web de conocimiento:** `make docs-build` (genera la distribución estática en `.site/`).
