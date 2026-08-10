---
title: "9. Índice de Notebooks"
tags: [codigo, jupyter, laboratorios]
status: borrador
---

# Índice de Notebooks y Código de Laboratorio

Aquí se centraliza el acceso a los cuadernos interactivos del laboratorio, ubicados en `labs/dl-lab/notebooks/`.

El contenedor monta `labs/dl-lab/` como `/workspace`, así que los paths en JupyterLab parten desde ahí. Los botones de abajo **abren en una pestaña nueva** directamente en [http://localhost:8888](http://localhost:8888/lab?token=dev).

---

## 🔬 1. Fundamentos — Perceptrón de Rosenblatt (1958)

Implementación desde cero del perceptrón clásico y prueba de convergencia.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/01-fundamentos/01-que-es-y-de-donde-viene/01_01_perceptron_rosenblatt_1958.ipynb?token=dev" target="_blank">📓 Abrir Notebook (.ipynb)</a> | <a href="http://localhost:8888/lab/tree/notebooks/01-fundamentos/01-que-es-y-de-donde-viene/01_01_perceptron_rosenblatt_1958.py?token=dev" target="_blank">🐍 Ver Script (.py)</a> |

---

## 🌳 1-bis. Fundamentos — Árboles, bosques y boosting (el ML que no es DL)

Laboratorio práctico de la nota [1.2](../../01-fundamentos/01-que-es-y-de-donde-viene/1.2-machine-learning-antes-del-dl.md):
árboles de decisión, Random Forest, gradient boosting **implementado a mano**, XGBoost y LightGBM.
Cada sección **comprueba experimentalmente** una afirmación de la teoría — incluida una que resulta
ser falsa en datos limpios (la ventaja de los árboles depende de que la tabla sea *irregular*).

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/01-fundamentos/01-que-es-y-de-donde-viene/01_02_arboles_bosques_y_boosting.ipynb?token=dev" target="_blank">📓 Abrir Notebook (.ipynb)</a> | <a href="http://localhost:8888/lab/tree/notebooks/01-fundamentos/01-que-es-y-de-donde-viene/01_02_arboles_bosques_y_boosting.py?token=dev" target="_blank">🐍 Ver Script (.py)</a> |

---

## 📚 2. Serie Nielsen — *Neural Networks and Deep Learning*

Recreación y ampliación práctica del libro clásico de Michael Nielsen, implementando redes desde cero hacia PyTorch.

### 01. El Problema

Introducción a las redes neuronales y planteamiento del problema MNIST.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/01_el_problema.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/01_el_problema.py?token=dev" target="_blank">🐍 Ver Script</a> |

### 02. Descenso de Gradiente

Concepto y optimización interactiva del descenso estocástico.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/02_descenso_del_gradiente.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/02_descenso_del_gradiente.py?token=dev" target="_blank">🐍 Ver Script</a> |

### 03. Backpropagation

Implementación manual de la regla de la cadena y propagación del error.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/03_backpropagation.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/03_backpropagation.py?token=dev" target="_blank">🐍 Ver Script</a> |

### 04. Red Completa sobre MNIST

Integración de todas las piezas sin librerías de alto nivel.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/04_red_completa_mnist.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/04_red_completa_mnist.py?token=dev" target="_blank">🐍 Ver Script</a> |

### 05. Mejoras del Aprendizaje

Cross-entropy, regularización L2 y técnicas avanzadas de optimización.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/05_mejoras_del_aprendizaje.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/05_mejoras_del_aprendizaje.py?token=dev" target="_blank">🐍 Ver Script</a> |

### 06. Puente a PyTorch

Migración del código artesanal a tensores y módulos de PyTorch.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/06_puente_a_pytorch.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/nielsen/06_puente_a_pytorch.py?token=dev" target="_blank">🐍 Ver Script</a> |

---

## 📘 3. Serie UDL — *Understanding Deep Learning* (Simon J.D. Prince)

Los 69 cuadernos oficiales del libro, copiados **tal cual** de
[udlbook/udlbook](https://github.com/udlbook/udlbook/tree/main/Notebooks) (licencia MIT).
Cada uno aísla un concepto y lo plantea como ejercicio: el código está escrito salvo unas
celdas con `TODO` que hay que completar.

Índice detallado por capítulo, licencia y dependencias extra en
<a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/README.md?token=dev" target="_blank">📄 su propio README</a>.

| Capítulo | Cuadernos |
|---|---|
| 1. Introducción | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap01?token=dev" target="_blank">📂 Chap01</a> — matemáticas de base |
| 2. Aprendizaje supervisado | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap02?token=dev" target="_blank">📂 Chap02</a> — modelo 1D, pérdida y ajuste |
| 3. Redes superficiales | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap03?token=dev" target="_blank">📂 Chap03</a> — regiones lineales, activaciones |
| 4. Redes profundas | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap04?token=dev" target="_blank">📂 Chap04</a> — composición y plegado del espacio |
| 5. Funciones de pérdida | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap05?token=dev" target="_blank">📂 Chap05</a> — mínimos cuadrados, entropía cruzada |
| 6. Ajuste de modelos | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap06?token=dev" target="_blank">📂 Chap06</a> — GD, SGD, momento, Adam |
| 7. Gradientes e inicialización | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap07?token=dev" target="_blank">📂 Chap07</a> — backprop e inicialización de He |
| 8. Medir el rendimiento | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap08?token=dev" target="_blank">📂 Chap08</a> — sesgo–varianza, doble descenso |
| 9. Regularización | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap09?token=dev" target="_blank">📂 Chap09</a> — L2, implícita, ensamblados, aumento |
| 10. Redes convolucionales | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap10?token=dev" target="_blank">📂 Chap10</a> — convolución 1D/2D, MNIST |
| 11. Redes residuales | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap11?token=dev" target="_blank">📂 Chap11</a> — skip connections, batch norm |
| 12. Transformers | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap12?token=dev" target="_blank">📂 Chap12</a> — atención, tokenización, decodificación |
| 13. Redes sobre grafos | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap13?token=dev" target="_blank">📂 Chap13</a> — GNN y graph attention |
| 15. GANs | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap15?token=dev" target="_blank">📂 Chap15</a> — GAN de juguete, Wasserstein |
| 16. Normalizing flows | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap16?token=dev" target="_blank">📂 Chap16</a> — flujos 1D y autorregresivos |
| 17. Autoencoders variacionales | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap17?token=dev" target="_blank">📂 Chap17</a> — variable latente, reparametrización |
| 18. Modelos de difusión | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap18?token=dev" target="_blank">📂 Chap18</a> — encoder, difusión 1D, familias |
| 19. Aprendizaje por refuerzo | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap19?token=dev" target="_blank">📂 Chap19</a> — MDP, Monte Carlo, TD |
| 20. ¿Por qué funciona el DL? | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap20?token=dev" target="_blank">📂 Chap20</a> — lottery tickets, ataques adversarios |
| 21. Ética | <a href="http://localhost:8888/lab/tree/notebooks/06-others/udlbook/Chap21?token=dev" target="_blank">📂 Chap21</a> — sesgos y explicabilidad |

!!! warning "Dependencias extra"
    Nueve cuadernos usan `mnist1d` y tres usan `networkx`, que **no** vienen en la imagen del
    laboratorio; `12_4` necesita `transformers` (extra `nlp`). Cada notebook trae su propia celda
    `!pip install ...` al principio, así que funcionan sin tocar nada — también en Colab.

    El capítulo 14 (*Unsupervised learning*) no tiene cuadernos en el repositorio original.

---

## 🎬 4. Serie Karpathy — *Neural Networks: Zero To Hero*

### Let's build GPT — de bigrama a Transformer

Companion notebook de la clase de nanoGPT de Andrej Karpathy: construye un GPT de caracteres
sobre *tiny shakespeare* partiendo de un modelo bigrama, derivando la auto-atención paso a
paso (media acumulada → producto matricial enmascarado → softmax → *scaled dot-product*) y
terminando en un Transformer decoder completo de 4 capas que entrena en minutos sobre CPU.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/karpathy/gpt_dev.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/karpathy/README.md?token=dev" target="_blank">📄 Procedencia y notas</a> |

!!! warning "La primera celda usa `wget`"
    `wget` no está instalado en la imagen del laboratorio (sí en Colab). Cambia esa celda por
    `!curl -sLO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
    y el resto funciona tal cual.

### nanoGPT — el mismo modelo, como proyecto ejecutable

Copia de [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT) en
`notebooks/06-others/karpathy/nanoGPT/`: la versión de trabajo de lo que el cuaderno anterior
construye a mano, con checkpoints, evaluación, carga de pesos de GPT-2 y entrenamiento
distribuido. Incluye dos cuadernos propios sobre leyes de escalado y dimensionado de
transformers.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/06-others/karpathy/nanoGPT-como-probarlo.md?token=dev" target="_blank">📄 Guía de ejecución</a> | <a href="http://localhost:8888/lab/tree/notebooks/06-others/karpathy/nanoGPT/scaling_laws.ipynb?token=dev" target="_blank">📓 scaling_laws.ipynb</a> |

!!! danger "nanoGPT se ejecuta **fuera** del contenedor"
    Docker Desktop en macOS no expone la GPU integrada: dentro del laboratorio no hay MPS.
    Para usar `--device=mps` hay que lanzarlo nativamente desde una terminal del Mac, desde la
    raíz de `nanoGPT/`. Todas las dependencias necesarias ya están en el Python del sistema.
    Detalles y límites de memoria en la guía.

---

## 🧪 5. Tests y Utilidades

### Smoke Test de Cómputo

Verificación rápida de disponibilidad de CUDA, tensor operations y aceleración GPU.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/04-proyecto/08-guia-entorno/00_smoke_test.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/04-proyecto/08-guia-entorno/00_smoke_test.py?token=dev" target="_blank">🐍 Ver Script</a> |
