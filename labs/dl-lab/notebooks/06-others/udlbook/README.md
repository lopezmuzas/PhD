# 📘 Serie UDL — *Understanding Deep Learning* (Simon J.D. Prince)

Copia de los cuadernos oficiales del libro
[*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (MIT Press, 2023),
tomados de [github.com/udlbook/udlbook](https://github.com/udlbook/udlbook/tree/main/Notebooks).

Son los notebooks referenciados en los márgenes del texto: cada uno aísla **un concepto**
y lo plantea como ejercicio — el código está escrito salvo unas pocas celdas con `TODO`
que hay que completar. Se conservan **tal cual vienen de upstream** (misma estructura de
carpetas, mismos nombres, con sus salidas de referencia) para poder re-sincronizarlos
con el repositorio original sin conflictos.

- **Licencia**: MIT — ver [LICENSE](LICENSE). Copyright 2023 Simon Prince.
- **Snapshot**: rama `main`, descargada el 2026-08-08.
- **Sin gemelos `.py`**: a diferencia del resto del laboratorio, aquí no se versionan los
  scripts de jupytext. Si abres y guardas uno en JupyterLab, jupytext generará su `.py`
  local (la configuración de `jupytext.toml` es global); no hace falta commitearlo.

---

## ⚠️ Dependencias que **no** están en la imagen del laboratorio

La mayoría de los cuadernos solo necesitan `numpy`, `matplotlib` y `torch`, que ya vienen
instalados. Estos son los que piden algo extra:

| Paquete | Notebooks que lo usan | Cómo instalarlo |
|---|---|---|
| `mnist1d` | 8_1, 8_3, 9_5, 10_2, 11_2, 11_3, 20_1, 20_2, 20_3 | `pip install git+https://github.com/greydanus/mnist1d` |
| `networkx` | 13_1, 13_2, 13_3 | `pip install networkx` |
| `transformers` | 12_4 | `pip install -e '.[nlp]'` (extra ya declarado en `pyproject.toml`) |

Los propios notebooks incluyen la celda `!pip install ...` al principio, así que también
funcionan tal cual en Google Colab.

---

## 🗂️ Índice por capítulo

> El capítulo 14 (*Unsupervised learning*) no tiene cuadernos en el repositorio original.

### Cap. 1 — Introducción

| Notebook | Tema |
|---|---|
| [1_1_BackgroundMathematics.ipynb](Chap01/1_1_BackgroundMathematics.ipynb) | Repaso de las matemáticas de base |

### Cap. 2 — Aprendizaje supervisado

| Notebook | Tema |
|---|---|
| [2_1_Supervised_Learning.ipynb](Chap02/2_1_Supervised_Learning.ipynb) | Modelo 1D, pérdida y ajuste a mano |

### Cap. 3 — Redes superficiales

| Notebook | Tema |
|---|---|
| [3_1_Shallow_Networks_I.ipynb](Chap03/3_1_Shallow_Networks_I.ipynb) | Red de una capa oculta |
| [3_2_Shallow_Networks_II.ipynb](Chap03/3_2_Shallow_Networks_II.ipynb) | Entradas y salidas multidimensionales |
| [3_3_Shallow_Network_Regions.ipynb](Chap03/3_3_Shallow_Network_Regions.ipynb) | Número de regiones lineales |
| [3_4_Activation_Functions.ipynb](Chap03/3_4_Activation_Functions.ipynb) | Funciones de activación |

### Cap. 4 — Redes profundas

| Notebook | Tema |
|---|---|
| [4_1_Composing_Networks.ipynb](Chap04/4_1_Composing_Networks.ipynb) | Componer dos redes superficiales |
| [4_2_Clipping_functions.ipynb](Chap04/4_2_Clipping_functions.ipynb) | Recorte de funciones y plegado del espacio |
| [4_3_Deep_Networks.ipynb](Chap04/4_3_Deep_Networks.ipynb) | Redes profundas y notación matricial |

### Cap. 5 — Funciones de pérdida

| Notebook | Tema |
|---|---|
| [5_1_Least_Squares_Loss.ipynb](Chap05/5_1_Least_Squares_Loss.ipynb) | Mínimos cuadrados como máxima verosimilitud |
| [5_2_Binary_Cross_Entropy_Loss.ipynb](Chap05/5_2_Binary_Cross_Entropy_Loss.ipynb) | Entropía cruzada binaria |
| [5_3_Multiclass_Cross_entropy_Loss.ipynb](Chap05/5_3_Multiclass_Cross_entropy_Loss.ipynb) | Entropía cruzada multiclase |

### Cap. 6 — Ajuste de modelos

| Notebook | Tema |
|---|---|
| [6_1_Line_Search.ipynb](Chap06/6_1_Line_Search.ipynb) | Búsqueda en línea |
| [6_2_Gradient_Descent.ipynb](Chap06/6_2_Gradient_Descent.ipynb) | Descenso de gradiente |
| [6_3_Stochastic_Gradient_Descent.ipynb](Chap06/6_3_Stochastic_Gradient_Descent.ipynb) | SGD |
| [6_4_Momentum.ipynb](Chap06/6_4_Momentum.ipynb) | Momento y momento de Nesterov |
| [6_5_Adam.ipynb](Chap06/6_5_Adam.ipynb) | Adam |

### Cap. 7 — Gradientes e inicialización

| Notebook | Tema |
|---|---|
| [7_1_Backpropagation_in_Toy_Model.ipynb](Chap07/7_1_Backpropagation_in_Toy_Model.ipynb) | Backprop en un modelo de juguete |
| [7_2_Backpropagation.ipynb](Chap07/7_2_Backpropagation.ipynb) | Backprop en una red profunda |
| [7_3_Initialization.ipynb](Chap07/7_3_Initialization.ipynb) | Inicialización de He y explosión/desvanecimiento |

### Cap. 8 — Medir el rendimiento

| Notebook | Tema |
|---|---|
| [8_1_MNIST_1D_Performance.ipynb](Chap08/8_1_MNIST_1D_Performance.ipynb) | Train/test error sobre MNIST-1D |
| [8_2_Bias_Variance_Trade_Off.ipynb](Chap08/8_2_Bias_Variance_Trade_Off.ipynb) | Compromiso sesgo–varianza |
| [8_3_Double_Descent.ipynb](Chap08/8_3_Double_Descent.ipynb) | Doble descenso |
| [8_4_High_Dimensional_Spaces.ipynb](Chap08/8_4_High_Dimensional_Spaces.ipynb) | Rarezas de la alta dimensión |

### Cap. 9 — Regularización

| Notebook | Tema |
|---|---|
| [9_1_L2_Regularization.ipynb](Chap09/9_1_L2_Regularization.ipynb) | Regularización L2 |
| [9_2_Implicit_Regularization.ipynb](Chap09/9_2_Implicit_Regularization.ipynb) | Regularización implícita del SGD |
| [9_3_Ensembling.ipynb](Chap09/9_3_Ensembling.ipynb) | Ensamblados |
| [9_4_Bayesian_Approach.ipynb](Chap09/9_4_Bayesian_Approach.ipynb) | Enfoque bayesiano |
| [9_5_Augmentation.ipynb](Chap09/9_5_Augmentation.ipynb) | Aumento de datos |

### Cap. 10 — Redes convolucionales

| Notebook | Tema |
|---|---|
| [10_1_1D_Convolution.ipynb](Chap10/10_1_1D_Convolution.ipynb) | Convolución 1D |
| [10_2_Convolution_for_MNIST_1D.ipynb](Chap10/10_2_Convolution_for_MNIST_1D.ipynb) | CNN sobre MNIST-1D |
| [10_3_2D_Convolution.ipynb](Chap10/10_3_2D_Convolution.ipynb) | Convolución 2D |
| [10_4_Downsampling_and_Upsampling.ipynb](Chap10/10_4_Downsampling_and_Upsampling.ipynb) | Submuestreo y sobremuestreo |
| [10_5_Convolution_For_MNIST.ipynb](Chap10/10_5_Convolution_For_MNIST.ipynb) | CNN sobre MNIST |

### Cap. 11 — Redes residuales

| Notebook | Tema |
|---|---|
| [11_1_Shattered_Gradients.ipynb](Chap11/11_1_Shattered_Gradients.ipynb) | Gradientes fragmentados |
| [11_2_Residual_Networks.ipynb](Chap11/11_2_Residual_Networks.ipynb) | Conexiones residuales |
| [11_3_Batch_Normalization.ipynb](Chap11/11_3_Batch_Normalization.ipynb) | Normalización por lotes |

### Cap. 12 — Transformers

| Notebook | Tema |
|---|---|
| [12_1_Self_Attention.ipynb](Chap12/12_1_Self_Attention.ipynb) | Auto-atención |
| [12_2_Multihead_Self_Attention.ipynb](Chap12/12_2_Multihead_Self_Attention.ipynb) | Atención multi-cabeza |
| [12_3_Tokenization.ipynb](Chap12/12_3_Tokenization.ipynb) | Tokenización (BPE) |
| [12_4_Decoding_Strategies.ipynb](Chap12/12_4_Decoding_Strategies.ipynb) | Estrategias de decodificación |

### Cap. 13 — Redes sobre grafos

| Notebook | Tema |
|---|---|
| [13_1_Graph_Representation.ipynb](Chap13/13_1_Graph_Representation.ipynb) | Representación de grafos |
| [13_2_Graph_Classification.ipynb](Chap13/13_2_Graph_Classification.ipynb) | Clasificación de grafos |
| [13_3_Neighborhood_Sampling.ipynb](Chap13/13_3_Neighborhood_Sampling.ipynb) | Muestreo de vecindarios |
| [13_4_Graph_Attention_Networks.ipynb](Chap13/13_4_Graph_Attention_Networks.ipynb) | Graph attention networks |

### Cap. 15 — GANs

| Notebook | Tema |
|---|---|
| [15_1_GAN_Toy_Example.ipynb](Chap15/15_1_GAN_Toy_Example.ipynb) | GAN de juguete |
| [15_2_Wasserstein_Distance.ipynb](Chap15/15_2_Wasserstein_Distance.ipynb) | Distancia de Wasserstein |

### Cap. 16 — Normalizing flows

| Notebook | Tema |
|---|---|
| [16_1_1D_Normalizing_Flows.ipynb](Chap16/16_1_1D_Normalizing_Flows.ipynb) | Flujos 1D |
| [16_2_Autoregressive_Flows.ipynb](Chap16/16_2_Autoregressive_Flows.ipynb) | Flujos autorregresivos |
| [16_3_Contraction_Mappings.ipynb](Chap16/16_3_Contraction_Mappings.ipynb) | Aplicaciones contractivas |

### Cap. 17 — Autoencoders variacionales

| Notebook | Tema |
|---|---|
| [17_1_Latent_Variable_Models.ipynb](Chap17/17_1_Latent_Variable_Models.ipynb) | Modelos de variable latente |
| [17_2_Reparameterization_Trick.ipynb](Chap17/17_2_Reparameterization_Trick.ipynb) | Truco de reparametrización |
| [17_3_Importance_Sampling.ipynb](Chap17/17_3_Importance_Sampling.ipynb) | Muestreo por importancia |

### Cap. 18 — Modelos de difusión

| Notebook | Tema |
|---|---|
| [18_1_Diffusion_Encoder.ipynb](Chap18/18_1_Diffusion_Encoder.ipynb) | Codificador de difusión |
| [18_2_1D_Diffusion_Model.ipynb](Chap18/18_2_1D_Diffusion_Model.ipynb) | Modelo de difusión 1D |
| [18_3_Reparameterized_Model.ipynb](Chap18/18_3_Reparameterized_Model.ipynb) | Modelo reparametrizado |
| [18_4_Families_of_Diffusion_Models.ipynb](Chap18/18_4_Families_of_Diffusion_Models.ipynb) | Familias de modelos de difusión |

### Cap. 19 — Aprendizaje por refuerzo

| Notebook | Tema |
|---|---|
| [19_1_Markov_Decision_Processes.ipynb](Chap19/19_1_Markov_Decision_Processes.ipynb) | Procesos de decisión de Markov |
| [19_2_Dynamic_Programming.ipynb](Chap19/19_2_Dynamic_Programming.ipynb) | Programación dinámica |
| [19_3_Monte_Carlo_Methods.ipynb](Chap19/19_3_Monte_Carlo_Methods.ipynb) | Métodos Monte Carlo |
| [19_4_Temporal_Difference_Methods.ipynb](Chap19/19_4_Temporal_Difference_Methods.ipynb) | Diferencias temporales |
| [19_5_Control_Variates.ipynb](Chap19/19_5_Control_Variates.ipynb) | Variables de control |

### Cap. 20 — ¿Por qué funciona el deep learning?

| Notebook | Tema |
|---|---|
| [20_1_Random_Data.ipynb](Chap20/20_1_Random_Data.ipynb) | Ajustar datos aleatorios |
| [20_2_Full_Batch_Gradient_Descent.ipynb](Chap20/20_2_Full_Batch_Gradient_Descent.ipynb) | Descenso por lote completo |
| [20_2_Full_Batch_Gradient_Descent_GPU.ipynb](Chap20/20_2_Full_Batch_Gradient_Descent_GPU.ipynb) | Ídem, versión GPU |
| [20_3_Lottery_Tickets.ipynb](Chap20/20_3_Lottery_Tickets.ipynb) | Hipótesis del billete de lotería |
| [20_4_Adversarial_Attacks.ipynb](Chap20/20_4_Adversarial_Attacks.ipynb) | Ataques adversarios |

### Cap. 21 — Ética del deep learning

| Notebook | Tema |
|---|---|
| [21_1_Bias_Mitigation.ipynb](Chap21/21_1_Bias_Mitigation.ipynb) | Mitigación de sesgos |
| [21_2_Explainability.ipynb](Chap21/21_2_Explainability.ipynb) | Explicabilidad |

---

## 🔄 Actualizar el snapshot

```bash
# desde una carpeta temporal
curl -sL https://github.com/udlbook/udlbook/archive/refs/heads/main.tar.gz -o udl.tar.gz
tar xzf udl.tar.gz --strip-components=2 udlbook-main/Notebooks
rsync -a --delete Chap* /ruta/al/repo/labs/dl-lab/notebooks/06-others/udlbook/
```

El tarball completo pesa ~370 MB porque el repositorio incluye las figuras del libro;
de ahí solo interesa `Notebooks/` (~1,3 MB).
