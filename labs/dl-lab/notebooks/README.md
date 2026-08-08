# 📂 Estructura de Notebooks por Módulos y Temas

Para evitar acumular decenas de notebooks sueltos en la raíz, los cuadernos de laboratorio están organizados secuencialmente por módulos pedagógicos y temas:

```text
notebooks/
├── others/
│   └── 00_smoke_test.ipynb           # Verificación de entorno, PyTorch, GPU/MPS y dllab
├── 01_perceptron_y_fundamentos/
│   ├── 01_01_perceptron_rosenblatt_1958.ipynb   # El Perceptrón original (AND, OR, fracaso en XOR)
│   └── 01_02_adaline_regla_delta.ipynb          # ADALINE y descenso de gradiente (Widrow-Hoff)
├── 02_mlp_y_clasificacion/
│   ├── 02_01_mlp_clasificacion_sintetica.ipynb  # Superando XOR y frontera Make Moons con MLP
│   └── 02_02_mlp_mnist_numeros.ipynb             # Clasificación de dígitos manuscritos (MNIST)
└── 03_cnn_y_vision/
    └── 03_01_introduccion_convnet.ipynb         # Primeras convoluciones para visión por computador
```

---

## 🏷️ Convenciones de Nomenclatura

`MM_NN_tema_descripcion.ipynb`
* **`MM` (Módulo)**: Categoría principal (`others` para tests/utilidades sueltas, `01` Fundamentos, `02` MLP, `03` CNN...)
* **`NN` (Orden)**: Número secuencial dentro del módulo (`01`, `02`, `03`...)

| Módulo | Enfoque Pedagógico |
|---|---|
| `others` | Smoke tests, verificaciones de entorno PyTorch, comprobación de GPU/MPS y utilidades varias. |
| `01_perceptron_y_fundamentos` | Modelos lineales de una sola capa, Rosenblatt, ADALINE, fronteras de decisión rectilíneas y separación lineal. |
| `02_mlp_y_clasificacion` | Redes multicapa (MLP), funciones de activación (ReLU/Sigmoid), backpropagation y datasets reales (MNIST/Fashion-MNIST). |
| `03_cnn_y_vision` | Convoluciones, Pooling, aumento de datos y visión por computador. |

---

## ⚡ Emparejado automático con Jupytext

Cada `.ipynb` mantiene un archivo `.py` gemelo gracias a `jupytext`.
* Al editar o ejecutar el notebook en JupyterLab, la versión `.py` se sincroniza automáticamente para facilitar el control de versiones en `git`.
* Para sincronizar manualmente todos los subdirectorios:
  ```bash
  jupytext --sync notebooks/*/*.ipynb
  ```

---

## 🚀 Ejecución en Google Colab

Cada notebook cuenta en sus primeras celdas con el bloque de **Bootstrap Automático**, el cual detecta si se ejecuta en Colab o en Docker local para clonar el repositorio e instalar la librería local `dllab` en modo editable.
