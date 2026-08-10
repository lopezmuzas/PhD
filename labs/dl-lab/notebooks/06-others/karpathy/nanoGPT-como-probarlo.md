# 🛠️ nanoGPT — cómo probarlo en este equipo

Guía de ejecución para la copia de [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
que vive en [`nanoGPT/`](nanoGPT/) (licencia MIT, snapshot de `master` del 2026-08-09).

Es la versión "de producción" de lo que [`gpt_dev.ipynb`](gpt_dev.ipynb) construye a mano en
el notebook: mismo modelo, pero como repositorio ejecutable con checkpoints, evaluación,
carga de pesos de GPT-2 y entrenamiento distribuido.

---

## ⚠️ Esto se ejecuta **fuera** del contenedor

Docker Desktop en macOS **no da acceso a la GPU integrada**: dentro del contenedor del
laboratorio no existe MPS y `torch` solo verá la CPU. Para usar `--device=mps` hay que
lanzarlo **nativamente en el Mac**, desde una terminal normal.

## 1. Dependencias — ya las tienes

Comprobado en el Python del sistema de este equipo:

| Paquete | Estado | ¿Hace falta para Shakespeare? |
|---|---|---|
| `torch` 2.2.2 (MPS disponible) | ✅ instalado | sí |
| `numpy`, `requests`, `tiktoken`, `tqdm` | ✅ instalados | sí (`tiktoken` lo importa `sample.py` aunque no se use en char-level) |
| `transformers` | ✅ instalado | no — solo para cargar pesos de GPT-2 |
| `datasets` | ❌ falta | no — solo para OpenWebText |
| `wandb` | ❌ falta | no — el logging remoto viene desactivado (`wandb_log = False`) |

**No hace falta instalar nada** para el recorrido de esta guía. Si algún día quieres los dos
que faltan:

```bash
pip install datasets wandb
```

## 2. Preparar los datos

```bash
cd labs/dl-lab/notebooks/06-others/karpathy/nanoGPT
python data/shakespeare_char/prepare.py
```

Descarga *tiny shakespeare* y lo tokeniza a nivel de carácter: genera `train.bin`
(1 003 854 tokens), `val.bin` (111 540) y `meta.pkl` con el vocabulario.

> **Ojo con el directorio de trabajo:** `train.py` resuelve el dataset como `data/<dataset>`
> **relativo al cwd**, así que todos los comandos siguientes hay que lanzarlos desde la raíz
> de `nanoGPT/`.

## 3. Entrenar

Primero una pasada corta para confirmar que todo arranca (un par de minutos):

```bash
python train.py config/train_shakespeare_char.py \
  --device=mps --compile=False --dtype=float32 \
  --max_iters=500 --lr_decay_iters=500 --eval_interval=100 --eval_iters=20
```

Y ya el entrenamiento completo de la config oficial (6 capas, 6 cabezas, `n_embd=384`,
contexto de 256 caracteres, 5000 iteraciones ≈ 10,6 M de parámetros):

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --dtype=float32
```

Por qué esos tres flags:

- **`--device=mps`** — usa la GPU integrada del M1. Karpathy reporta 2-3× frente a CPU.
- **`--compile=False`** — `torch.compile` sigue siendo inestable en macOS/MPS.
- **`--dtype=float32`** — no es imprescindible, pero evita un aviso feo: sin CUDA el valor por
  defecto de `dtype` cae en `float16`, y `train.py` instancia entonces un
  `torch.cuda.amp.GradScaler(enabled=True)` en una máquina sin CUDA. PyTorch avisa y lo
  desactiva solo, así que funciona igual — pero en MPS el entrenamiento va en float32 de todas
  formas, y pedirlo explícitamente deja el log limpio.

El checkpoint se guarda en `out-shakespeare-char/ckpt.pt`, y solo cuando mejora la pérdida de
validación (`always_save_checkpoint = False`).

Si vas justo de memoria o de paciencia, la receta reducida del README original (4 capas,
4 cabezas, `n_embd=128`, contexto 64, batch 12) también funciona en MPS:

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --dtype=float32 \
  --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=20 --log_interval=1
```

## 4. Generar texto

```bash
python sample.py --out_dir=out-shakespeare-char --device=mps
```

Otros parámetros útiles: `--start="ROMEO:"` para condicionar el arranque,
`--num_samples`, `--max_new_tokens`, `--temperature`.

---

## 📏 Hasta dónde llega este hardware (M1, 8 GB unificados)

La cuenta que manda es la del **optimizador**: AdamW guarda, por cada parámetro, el peso
(4 B en fp32), su gradiente (4 B) y dos momentos (8 B) → **~16 bytes por parámetro**, antes de
contar activaciones.

| Modelo | Parámetros | Estado del optimizador | Veredicto en 8 GB |
|---|---|---|---|
| `train_shakespeare_char` | ~10,6 M | ~170 MB | ✅ cómodo |
| GPT-2 small | 124 M | ~2 GB | ✅ entrenable / fine-tuning, con batch pequeño |
| GPT-2 medium | 350 M | ~5,6 GB | ❌ no cabe junto al sistema y las activaciones |
| GPT-2 large / XL | 774 M / 1,5 B | ~12 GB / ~24 GB | ❌ descartado |

Es decir: modelos pequeños desde cero y fine-tuning de GPT-2 small, sí. De medium para
arriba, no — y no es cuestión de esperar más tiempo, es que no entra en memoria.

Para el fine-tuning de GPT-2 small sobre Shakespeare (a nivel de token BPE, no de carácter):

```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py --device=mps --compile=False --dtype=float32
```

---

## 🧹 Qué queda fuera de git

El `.gitignore` que trae el propio nanoGPT ya excluye `*.bin`, `*.pkl`, `*.pt` e `input.txt`,
así que los datos tokenizados y los checkpoints que generes **no se versionan**. No hay que
tocar nada.

## 🔄 Actualizar el snapshot

```bash
curl -sL https://github.com/karpathy/nanoGPT/archive/refs/heads/master.tar.gz -o nanogpt.tar.gz
tar xzf nanogpt.tar.gz
rsync -a --delete nanoGPT-master/ /ruta/al/repo/labs/dl-lab/notebooks/06-others/karpathy/nanoGPT/
```
