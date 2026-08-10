# 🎬 Serie Karpathy — *Neural Networks: Zero To Hero*

Cuadernos de la serie de clases de Andrej Karpathy
([karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html)).

| Contenido | Clase | Qué construye |
|---|---|---|
| [gpt_dev.ipynb](gpt_dev.ipynb) | *Let's build GPT: from scratch, in code, spelled out* | Un GPT de caracteres sobre *tiny shakespeare*, desde el modelo bigrama hasta un Transformer decoder completo |
| [nanoGPT/](nanoGPT/) | *ídem* — repositorio de código | La misma arquitectura como proyecto ejecutable: checkpoints, evaluación, carga de pesos de GPT-2, DDP |

El notebook es el **paseo didáctico**; nanoGPT es la **versión de trabajo** del mismo modelo.
Para ejecutar nanoGPT en este equipo → [nanoGPT-como-probarlo.md](nanoGPT-como-probarlo.md)
(dependencias, MPS, límites de memoria en 8 GB). Incluye además dos cuadernos propios,
`nanoGPT/scaling_laws.ipynb` y `nanoGPT/transformer_sizing.ipynb`.

## Procedencia y licencia

`gpt_dev.ipynb` es el **companion notebook** de la clase de nanoGPT, publicado por el autor
como [Colab compartido](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing).
Copia fiel, con sus salidas originales y sin `.py` gemelo de jupytext.

No existe en ningún repositorio de GitHub: el código equivalente vive en
[karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) (`bigram.py`, `gpt.py`),
publicado bajo **licencia MIT**, igual que
[karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero), que recoge el resto
de clases de la serie (micrograd y makemore). El notebook en sí no lleva cabecera de licencia.

`nanoGPT/` es una copia de [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), rama
`master`, **licencia MIT** — ver [nanoGPT/LICENSE](nanoGPT/LICENSE). Se conserva íntegro,
incluidos su `README.md` y su `.gitignore` originales.

Ambos descargados el 2026-08-09.

## ⚠️ Antes de ejecutarlo en el laboratorio

La primera celda descarga el dataset con `wget`, que **no está instalado** en la imagen del
laboratorio (sí lo está en Colab). Sustitúyela por `curl`, que sí viene:

```python
!curl -sLO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

El resto funciona tal cual. `input.txt` (~1,1 MB) se descarga en esta misma carpeta y está
excluido de git.

## Recorrido del cuaderno

1. **Dataset y tokenizador** (celdas 1–8) — tiny shakespeare, vocabulario de caracteres,
   codificación a tensores y split train/val.
2. **Modelo bigrama** (9–16) — línea base entrenable con `nn.Embedding` y muestreo.
3. **El truco matemático de la auto-atención** (17–24) — de la media acumulada al producto
   matricial con máscara triangular y softmax, en cuatro versiones equivalentes.
4. **Escalado, LayerNorm y encoder/decoder** (25–35) — por qué se divide por `√head_size`,
   normalización por capa frente a batch norm, y la diferencia entre atención causal y
   bidireccional.
5. **Código final** (37) — el Transformer completo: `batch_size=16`, `block_size=32`,
   4 capas, 4 cabezas, `n_embd=64`, 5000 iteraciones. Es pequeño a propósito: entrena en
   pocos minutos en CPU y usa GPU automáticamente si la detecta (`torch.cuda.is_available()`).
