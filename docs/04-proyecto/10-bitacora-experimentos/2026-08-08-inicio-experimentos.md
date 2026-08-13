---
title: "2026-08-08 Puesta a punto del entorno"
tags: [bitacora, docker, setup]
status: revisado
---

# Puesta a punto del entorno de desarrollo e infraestructura de computación

**Fecha**: 08 de Agosto de 2026  
**Autor**: Antonio  
**Categoría**: Infraestructura / Configuración Inicial

---

## 🎯 Objetivo de la Sesión
Establecer un entorno de desarrollo reproducible basado en contenedores Docker con soporte completo para la GPU, y configurar la base de conocimiento en MkDocs para documentar las iteraciones del doctorado.

---

## ⚙️ Trabajo Realizado

### 1. Reestructuración de la Base de Conocimiento (MkDocs)
- Se movieron todos los archivos históricos a la sección `Archivo/`.
- Se configuró la estructura de 5 categorías horizontales principales con subcarpetas explícitas.
- Se ajustó `mkdocs.yml` para mapear `docs_dir` a `docs/docs` y corregir los problemas de visualización del menú superior.

### 2. Comprobación del Soporte CUDA
Se levantó el contenedor del laboratorio con el comando `make up-gpu` y se lanzó el notebook interactivo de Smoke Test (`others/00_smoke_test.ipynb`). 

**Resultado**:
- PyTorch detecta correctamente el driver de NVIDIA a través de WSL2/macOS docker context.
- Dispositivo CUDA mapeado y memoria VRAM disponible para entrenamiento.

---

## 💡 Conclusiones y Aprendizajes
- **Orden de navegación**: La separación en carpetas con archivos `.pages` estructurados es mucho más limpia que declarar todas las páginas manualmente en el `mkdocs.yml`, permitiendo escalabilidad a medida que añada nuevos notebooks.
- **Formato Jupytext**: El uso de Jupytext sincronizando `.ipynb` <-> `.py` previene el commit de metadatos pesados de Jupyter en el repositorio de Git, facilitando la legibilidad en diffs de código.

---

## 📌 Próximos Pasos
- [ ] Ejecutar el notebook de entrenamiento básico del Perceptrón en la GPU para medir latencia de transferencia de tensores.
- [ ] Configurar el primer experimento formal con MLflow para validar el registro local de métricas de entrenamiento.

<!-- nav-start -->

---

← Anterior: [index.md](index.md)  
Siguiente: [11. Índice general](../11-federado-sobre-data-spaces/11-indice.md) →

<!-- nav-end -->
