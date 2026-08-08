# 📊 Bibliometrix / Biblioshiny — Análisis Bibliométrico Local

Este espacio almacena los datos de búsqueda bibliográfica (`.bib`) y las instrucciones para ejecutar la suite de análisis bibliométrico **Bibliometrix** mediante **RStudio / R local** de forma nativa en macOS.

> ⚠️ **Nota de rendimiento**: Se utiliza R/RStudio nativo en el sistema host en lugar de Docker para evitar la degradación de rendimiento y la sobrecarga de emulación en macOS.

---

## 📂 Estructura del Directorio

```
bibliometrix/
├── data/                         # Ficheros .bib exportados de Web of Science / Scopus
│   ├── savedrecsAntonio.bib      # Búsquedas generales de doctorado
│   ├── savedrecsAntonioFLDT.bib  # Búsquedas de Federated Learning & Data Spaces
│   └── savedrecsFLRLDS.bib       # Búsquedas de FRL + RL Offline + Data Spaces
├── run_biblioshiny.R             # Script en R para lanzar la interfaz web local
└── README.md                     # Este manual de uso
```

---

## 🚀 Cómo Ejecutar Biblioshiny

### Opción A: Desde la Terminal con Makefile (Recomendado)
Desde la raíz del proyecto, ejecuta:
```bash
make bibliometrix
```
*(Lanza `run_biblioshiny.R` utilizando tu instalación local de R).*

### Opción B: Desde RStudio
1. Abre **RStudio**.
2. Abre el archivo `bibliometrix/run_biblioshiny.R`.
3. Haz clic en **Source** o ejecuta en la consola de R:
   ```R
   source("bibliometrix/run_biblioshiny.R")
   ```

### Opción C: Desde la Consola de R directamente
```R
library(bibliometrix)
biblioshiny()
```

---

## 📥 Cargar los Datos `.bib` en la Aplicación

1. Una vez abierta la interfaz de **Biblioshiny** en el navegador:
2. Ve al menú lateral: **Import or Load Files** $\rightarrow$ **Import raw files**.
3. Selecciona la fuente: **Web of Science** (o **Scopus** según el fichero).
4. Elige el formato: **BibTeX**.
5. Haz clic en **Browse** y navega hasta la carpeta del proyecto:
   `bibliometrix/data/`
6. Selecciona el archivo `.bib` deseado (ej. `savedrecsFLRLDS.bib`).
7. Haz clic en **Start** para procesar los registros e iniciar los análisis de co-citación, mapas temáticos y redes de colaboración.
