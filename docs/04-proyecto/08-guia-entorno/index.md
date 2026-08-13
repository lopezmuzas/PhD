---
title: "8. Guía de Ejecución"
tags: [guia, entorno, docker, bibliometrix]
status: revisado
---

# 8. Guía de Ejecución y Entorno de Desarrollo

Este documento sirve como manual rápido para operar la infraestructura local del proyecto. Todos los comandos se ejecutan mediante el [Makefile](file:///Users/lopezmuzas/Developer/PhD/Makefile) en la raíz del proyecto.

---

## 🚀 Inicio Rápido (Quickstart)

### 1. Inicializar el entorno
Crea el archivo `.env` local si no existe y prepara los directorios para almacenar datasets, checkpoints y logs de entrenamiento:
```bash
make init
```

### 2. Construir contenedores
- **CPU (Por defecto)**: `make build`
- **GPU (Con soporte CUDA)**: `make build-gpu`

### 3. Levantar servicios básicos (Docs + JupyterLab)
Levanta la documentación en local y el laboratorio de cómputo:
```bash
make up       # Para entorno CPU
# o bien
make up-gpu   # Para entorno con aceleración GPU
```

---

## 🛠️ Servicios Disponibles en Local

Una vez levantado el entorno, puedes acceder a los siguientes servicios desde tu navegador:

| Servicio | URL / Comando | Activación | Descripción |
| :--- | :--- | :--- | :--- |
| **Documentación** | [http://localhost:8000](http://localhost:8000) | `make up` / `make docs` | Este sitio web de MkDocs. |
| **JupyterLab** | [http://localhost:8888/lab?token=dev](http://localhost:8888/lab?token=dev) | `make up` / `make lab` | Entorno interactivo principal de Python. |
| **TensorBoard** | [http://localhost:6006](http://localhost:6006) | `make tb` | Visualización de curvas de aprendizaje sobre `outputs/runs/`. |
| **MLflow Server** | [http://localhost:5000](http://localhost:5000) | `make mlflow` | Registro y tracking de experimentos y modelos. |
| **Wiki** | [http://localhost:3000](http://localhost:3000) | `make wiki` | Edición ágil de notas (SilverBullet). |
| **Bibliometrix (R)** | Interfaz Web Shiny local | `make bibliometrix` | Análisis bibliométrico en R local (evita sobrecarga de Docker en macOS). |

---

## 📊 Análisis Bibliométrico con Bibliometrix / Biblioshiny

Para evitar la sobrecarga de emulación de Docker en macOS, la herramienta de análisis bibliométrico corre **de forma nativa en R / RStudio** sobre el sistema host.

### 📁 Almacenamiento de Ficheros `.bib`
Todos los registros exportados de Web of Science / Scopus se guardan centralizadamente en la carpeta del proyecto: [bibliometrix/data/](file:///Users/lopezmuzas/Developer/PhD/bibliometrix/data)
- `savedrecsAntonio.bib`: Búsquedas generales de doctorado.
- `savedrecsAntonioFLDT.bib`: Búsquedas de Federated Learning & Data Spaces.
- `savedrecsFLRLDS.bib`: Búsquedas de FRL + RL Offline + Data Spaces.

### 🚀 Cómo ejecutar Biblioshiny
Puedes iniciarlo de dos formas:

1. **Desde la consola con el Makefile**:
   ```bash
   make bibliometrix
   ```
2. **Desde RStudio**:
   Abre y ejecuta [bibliometrix/run_biblioshiny.R](file:///Users/lopezmuzas/Developer/PhD/bibliometrix/run_biblioshiny.R) o lanza en la consola de R:
   ```R
   library(bibliometrix)
   biblioshiny()
   ```

3. **Cargar datos**: En la aplicación web Shiny, ve a **Import raw files** $\rightarrow$ seleccciona **BibTeX** y elige cualquiera de los ficheros ubicados en `bibliometrix/data/`.

---

## 💻 Comandos del Día a Día

### Entrar al contenedor interactivo (bash)
```bash
make shell
```

### Sincronización de Jupyter Notebooks con Jupytext
```bash
make sync
```

### Ejecutar tests y linter
```bash
make test  # Lanza la suite de pruebas unitarias
make lint  # Aplica formateador y linter de código (Ruff, markdownlint)
```

### Apagar el entorno
```bash
make down
```

<!-- nav-start -->

---

← Anterior: [7.6 Guía práctica: Flower + Ocean C2D](../07-combinaciones/7.6-guia-practica-flower-ocean-c2d.md)  
Siguiente: [index.md](../09-indice-notebooks/index.md) →

<!-- nav-end -->
