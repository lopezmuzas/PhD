# 📂 Estructura de Notebooks por Temática

Para facilitar el aprendizaje y la experimentación, los cuadernos de laboratorio (`notebooks/`) están organizados siguiendo **la misma estructura temática** que la documentación del proyecto (`docs/`). De este modo, puedes correlacionar directamente la teoría con el código práctico.

## 🗺️ Mapa de Contenidos y Notebooks

A continuación se muestra la correspondencia entre los módulos temáticos y los cuadernos interactivos actuales:

```text
notebooks/
├── 01-fundamentos/
│   ├── 01-que-es-y-de-donde-viene/
│   │   ├── 01_01_perceptron_rosenblatt_1958.ipynb   # El Perceptrón clásico (Rosenblatt 1958)
│   │   └── 01_02_arboles_bosques_y_boosting.ipynb   # ML clásico: árboles, RF, boosting, XGBoost/LightGBM
│   ├── 02-las-matematicas-necesarias/
│   │   └── .gitkeep
│   └── 03-como-se-entrena-una-red/
│       └── .gitkeep
├── 02-deep-learning/
│   ├── 04-formas-de-aprender/
│   │   └── .gitkeep                                 # (Preparado para notebooks de aprendizaje supervisado/refuerzo)
│   ├── 05-tipos-de-modelo/
│   │   └── .gitkeep
│   └── 06-donde-estan-los-datos/
│       └── .gitkeep                                 # (Preparado para notebooks de datos distribuidos/federados)
├── 03-data-spaces/
│   └── [01 al 05]/.gitkeep                          # (Preparado para experimentación en gobernanza y espacios de datos)
├── 04-proyecto/
│   ├── 07-combinaciones/.gitkeep
│   ├── 08-guia-entorno/
│   │   └── 00_smoke_test.ipynb                      # Test de entorno: CUDA, GPU y aceleración MPS
│   ├── 09-indice-notebooks/.gitkeep
│   └── 10-bitacora-experimentos/.gitkeep
├── 05-anexos/
│   └── .gitkeep
└── 06-others/
    └── nielsen/                                     # Serie basada en Neural Networks and Deep Learning (M. Nielsen)
        ├── 01_el_problema.ipynb                     # Serie Nielsen 01: El problema MNIST e introducción
        ├── 02_descenso_del_gradiente.ipynb          # Serie Nielsen 02: Descenso de Gradiente (SGD)
        ├── 03_backpropagation.ipynb                 # Serie Nielsen 03: Ecuaciones y Backprop manual
        ├── 04_red_completa_mnist.ipynb              # Serie Nielsen 04: Red completa entrenada desde cero
        ├── 05_mejoras_del_aprendizaje.ipynb         # Serie Nielsen 05: Regularización L2 y Entropía Cruzada
        └── 06_puente_a_pytorch.ipynb                # Serie Nielsen 06: Migración de red MLP a PyTorch y PyTorch CNN
```

---

## ⚡ Emparejado automático con Jupytext

Cada archivo `.ipynb` mantiene un archivo `.py` gemelo en su mismo directorio usando `jupytext`.
* Al editar o guardar un notebook interactivo en JupyterLab, su script `.py` correspondiente se actualiza de forma automática para facilitar el control de versiones limpio en `git`.
* Si editas un script `.py` directamente en tu editor de código o IDE favorito, puedes sincronizar y regenerar el notebook ejecutando desde la raíz del laboratorio:
  ```bash
  make sync
  ```
  *(Este comando ejecutará de forma interna `find notebooks -name "*.ipynb" -exec jupytext --sync {} +` dentro del contenedor).*

---

## 🚀 Ejecución en Google Colab

Cada cuaderno cuenta con un bloque de **Bootstrap Automático** en sus celdas iniciales que detecta de forma dinámica si se está ejecutando en un entorno de Google Colab o local (Docker). En caso de Colab, clona el repositorio del proyecto e instala la librería local `dllab` en modo editable para que todas las importaciones y carga de datos funcionen de inmediato.
