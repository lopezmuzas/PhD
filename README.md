# 🎓 Base de Conocimiento y Laboratorio de Doctorado

> **Área de Investigación:** Aprendizaje por Refuerzo Offline y Federado (*Federated Offline RL*) sobre Espacios de Datos Soberanos (*Data Spaces* / Gaia-X / Ocean Protocol / Pontus-X).

Este repositorio centraliza la **base de conocimiento interactiva**, el **laboratorio de cómputo en PyTorch (CPU/GPU)**, las herramientas de **análisis bibliométrico en R** y la **bitácora de experimentos** del doctorado.

---

## 📂 Estructura del Proyecto

```none
PhD/
├── compose.yaml              # Orquestador raíz Docker Compose (docs + laboratorio)
├── mkdocs.yml                # Configuración del sitio MkDocs Material (awesome-pages)
├── Makefile                  # Automatización de tareas de desarrollo y despliegue
├── docs/                     # BASE DE CONOCIMIENTO (Ficheros Markdown fuente)
│   ├── 01-fundamentos/       # I. Fundamentos (Matemáticas, Álgebra, Cálculo, Biología)
│   ├── 02-deep-learning/     # II. Deep Learning (Paradigmas, Arquitecturas, Modelos)
│   ├── 03-data-spaces/       # III. Data Spaces (Gobernanza, FL, Pontus-X, Retos R1–R15)
│   ├── 04-proyecto/          # IV. Proyecto (Combinaciones 3D, Guía, Notebooks, Bitácora)
│   ├── 05-anexos/            # V. Anexos (Línea temporal, Glosario, WoS, Sistemas Emergentes)
│   └── Archivo/              # Material e investigaciones históricas respaldadas
├── labs/
│   └── dl-lab/               # LABORATORIO DE DEEP LEARNING (PyTorch / CUDA / MLflow)
│       ├── notebooks/        # Cuadernos Jupyter (.ipynb <-> .py sincronizados con Jupytext)
│       ├── src/              # Código fuente modular (dllab)
│       └── experiments/      # Configuraciones de experimentos (YAML)
├── bibliometrix/             # ANÁLISIS BIBLIOMÉTRICO (R / Biblioshiny local)
│   ├── data/                 # Registros .bib exportados de Web of Science y Scopus
│   └── run_biblioshiny.R     # Script de lanzamiento nativo en R
└── docs-stack/               # Imagen Docker y configuración del visor MkDocs
```

---

## 🚀 Inicio Rápido

### 1. Inicializar configuración local
```bash
cp .env.example .env
make init
```

### 2. Comandos habituales (Makefile)

| Comando | Servicio | URL / Descripción |
|---|---|---|
| `make up` | **Docs + Jupyter (CPU)** | Levanta la documentación y el laboratorio en modo CPU (Recomendado). |
| `make up-gpu` | **Docs + Jupyter (GPU)** | Levanta el laboratorio con aceleración CUDA. |
| `make docs` | **Documentación** | [http://localhost:8000](http://localhost:8000) (MkDocs con recarga en vivo). |
| `make lab` | **JupyterLab** | [http://localhost:8888/lab?token=dev](http://localhost:8888/lab?token=dev) |
| `make bibliometrix` | **Bibliometrix** | Lanza la interfaz web de Biblioshiny en R local con los datos de `bibliometrix/data/`. |
| `make mlflow` | **MLflow Server** | [http://localhost:5000](http://localhost:5000) (Tracking de métricas y modelos). |
| `make tb` | **TensorBoard** | [http://localhost:6006](http://localhost:6006) (Visualización de entrenamiento). |
| `make sync` | **Jupytext** | Sincroniza cuadernos `.ipynb` con scripts de Python `.py`. |
| `make lint` | **Calidad de código** | Linter Ruff para Python y markdownlint para documentación. |
| `make down` | **Apagar** | Detiene de forma limpia todos los contenedores y servicios. |

---

## 🌐 Servicios y Puertos en Local

| Puerto | Servicio | Descripción |
|---|---|---|
| **8000** | **MkDocs Material** | Sitio web de la Base de Conocimiento. |
| **8888** | **JupyterLab** | Entorno de desarrollo para cuadernos y experimentos. |
| **6006** | **TensorBoard** | Visualizador de curvas de aprendizaje sobre `outputs/runs/`. |
| **5000** | **MLflow** | Servidor de registro de experimentos y artefactos. |
| **3000** | **Wiki (SilverBullet)** | Visor/Editor ágil de notas (opcional). |
| *Host* | **Biblioshiny** | Aplicación web Shiny local en R para análisis bibliométrico. |

---

## 🔗 Control de Versiones e Integración

El proyecto se gestiona en un repositorio privado en GitHub:
- **Repositorio Remoto**: [https://github.com/lopezmuzas/PhD](https://github.com/lopezmuzas/PhD)
- **Generar construcción estática**: `make docs-build` (genera el paquete estático en `.site/`).
