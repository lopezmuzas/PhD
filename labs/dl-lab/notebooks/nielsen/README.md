# Serie: redes neuronales desde cero

Recorrido guiado por [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/)
de Michael Nielsen, adaptado para ejecutarse en este laboratorio.

El objetivo no es que la red funcione — eso lo consigue cualquiera copiando
código — sino que entiendas **por qué** funciona, habiendo escrito cada pieza.
Al final del recorrido habrás implementado backpropagation a mano, comprobado
numéricamente que tu gradiente es correcto, y visto de dónde sale cada punto
porcentual de acierto.

## El recorrido

| # | Notebook | Capítulo del libro | Qué construyes | Tiempo |
|---|---|---|---|---|
| 1 | `01_el_problema` | 1 | MNIST, perceptrón, neurona sigmoide, arquitectura | ~2 min |
| 2 | `02_descenso_del_gradiente` | 1 | Función de coste, gradiente, SGD | ~2 min |
| 3 | `03_backpropagation` | 2 | Las cuatro ecuaciones, implementadas y verificadas | ~3 min |
| 4 | `04_red_completa_mnist` | 1 | Red completa entrenada: **~95%** | ~20 min |
| 5 | `05_mejoras_del_aprendizaje` | 3 | Entropía cruzada, L2, inicialización: **~98%** | ~25 min |
| 6 | `06_puente_a_pytorch` | — | La misma red en PyTorch, luego CNN: **~99%** | ~10 min |

Los tiempos son de CPU. Los notebooks 4 y 5 son lentos a propósito: son numpy
puro con un bucle por ejemplo, igual que en el libro. Esa lentitud es parte de la
lección — cuando en el notebook 6 la misma red entrena en segundos, entiendes
exactamente qué te está dando el framework.

**Hazlos en orden.** Cada uno da por sabido el anterior.

## Cómo ejecutarlos

### En local

```bash
make up                     # desde la raíz del doctorado
```

Abre http://localhost:8888 y navega a `notebooks/nielsen/`. La primera vez que
ejecutes el notebook 1 se descargará MNIST (~11 MB) en `data/raw/`.

### En Colab

Sube el repositorio a GitHub y abre:

```
https://colab.research.google.com/github/TU_USUARIO/dl-lab/blob/main/notebooks/nielsen/04_red_completa_mnist.ipynb
```

La primera celda de cada notebook detecta Colab y prepara el entorno sola.
Útil si quieres lanzar los notebooks largos mientras haces otra cosa.

## El código de apoyo

Los notebooks explican; el código reutilizable vive en `src/dllab/nielsen/`:

| Fichero | Contenido |
|---|---|
| `data.py` | Carga de MNIST en formato de vectores columna |
| `network.py` | `Red`: sigmoides, coste cuadrático, SGD, backprop (caps. 1-2) |
| `improved.py` | `RedMejorada`: entropía cruzada, L2, mejor init (cap. 3) |
| `viz.py` | Dígitos, curvas de aprendizaje, matriz de confusión, pesos |

Está pensado para leerse. `network.py` cabe en una sentada y no tiene ningún
truco: si algo del notebook 3 no te queda claro, ábrelo y sigue el flujo.

## Progresión de resultados

| Método | Acierto en test |
|---|---|
| Azar | 10% |
| Plantilla media por clase | ~82% |
| Red 784→30→10, coste cuadrático | ~95% |
| + entropía cruzada, L2, init escalada | ~98% |
| Convolucional en PyTorch | ~99% |

## Qué queda fuera

El libro tiene seis capítulos; esta serie cubre bien los tres primeros y asoma al
resto. Si quieres seguir:

- **Cap. 4** — demostración visual de que una red con una capa oculta puede
  aproximar cualquier función. Corto y muy visual.
- **Cap. 5** — el problema del gradiente evanescente, que ya asoma en el
  notebook 3. Explica por qué las redes profundas eran inentrenables.
- **Cap. 6** — convolucionales en profundidad y hacia dónde va el campo.

## Nota sobre autoría

El libro de Nielsen está publicado bajo licencia
[CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/deed.es). El
código y las explicaciones de estos notebooks son originales: siguen su
recorrido pedagógico, que es lo verdaderamente valioso, pero no reproducen su
texto ni su código. Lee el original — es gratuito, está muy bien escrito y estos
notebooks funcionan mejor como complemento que como sustituto.
