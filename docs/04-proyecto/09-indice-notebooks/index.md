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

## 🧪 3. Tests y Utilidades

### Smoke Test de Cómputo

Verificación rápida de disponibilidad de CUDA, tensor operations y aceleración GPU.

| | |
|---|---|
| <a href="http://localhost:8888/lab/tree/notebooks/04-proyecto/08-guia-entorno/00_smoke_test.ipynb?token=dev" target="_blank">📓 Abrir Notebook</a> | <a href="http://localhost:8888/lab/tree/notebooks/04-proyecto/08-guia-entorno/00_smoke_test.py?token=dev" target="_blank">🐍 Ver Script</a> |
