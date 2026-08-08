---
title: "Bibliometrix Explorer — Proyecto Dockerizado"
tags: [guia]
status: borrador
updated: 2026-08-08
---

# Bibliometrix Explorer — Proyecto Dockerizado

Aplicación Shiny para análisis bibliométrico basada en el paquete R [`bibliometrix`](https://www.bibliometrix.org/).

---

## 🚀 Inicio rápido

### Requisitos
- [Docker](https://www.docker.com/) instalado y en ejecución
- [Docker Compose](https://docs.docker.com/compose/) (incluido en Docker Desktop)

### Levantar la aplicación

```bash
docker compose up --build
```

> La primera vez tardará unos minutos mientras se instalan las dependencias de R.  
> Las siguientes veces usará la caché y será inmediato.

Una vez levantado, abre en tu navegador:

```
http://localhost:3838/bibliometrix/
```

### Detener la aplicación

```bash
docker compose down
```

---

## 📂 Estructura del proyecto

```
bibliometrix-project/
├── docker-compose.yml       # Orquestación de contenedores
├── Dockerfile               # Imagen R + Shiny + bibliometrix
├── shiny-app/
│   └── app.R                # Aplicación Shiny completa
├── data/                    # Carpeta para tus datos bibliográficos
└── README.md
```

---

## 📥 Cómo usar la aplicación

1. Exporta registros desde **Web of Science**, **Scopus**, **PubMed** u otra fuente compatible.
2. En la app, ve a **Cargar datos** → sube tu fichero.
3. Selecciona la base de datos origen y el formato del fichero.
4. Pulsa **Cargar** y explora los análisis.

### Formatos soportados

| Base de datos | Formato recomendado |
|---------------|---------------------|
| Web of Science | Texto plano (.txt) |
| Scopus | BibTeX (.bib) o CSV (.csv) |
| PubMed | Texto plano (.txt) |
| Lens.org | CSV (.csv) |
| OpenAlex | CSV (.csv) |

---

## 🔧 Personalización

- Añade tus archivos de datos en la carpeta `data/` (se monta en el contenedor).
- Edita `shiny-app/app.R` para añadir nuevos análisis.
- Cambia el puerto en `docker-compose.yml` si el 3838 está ocupado.

---

## 📦 Paquetes R incluidos

- `bibliometrix` — análisis bibliométrico
- `shiny` + `shinydashboard` — interfaz web
- `DT` — tablas interactivas
- `ggplot2` + `plotly` — visualizaciones
- `dplyr` — manipulación de datos
