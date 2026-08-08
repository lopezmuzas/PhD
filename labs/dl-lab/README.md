# dl-lab — tu Colab en local

Laboratorio dockerizado para diseñar, entrenar y comparar redes neuronales profundas.
JupyterLab dentro de Docker, código reutilizable en `src/`, experimentos reproducibles
desde YAML, y notebooks que se abren tal cual en Google Colab.

## Arranque rápido

```bash
cp .env.example .env
make build      # imagen CPU   (o: make build-gpu)
make up         # JupyterLab en http://localhost:8888  (o: make gpu)
```

El token está en `.env` (`JUPYTER_TOKEN=dev`). Abre `notebooks/00_smoke_test.ipynb`
y ejecútalo entero: si termina, el entorno está bien.

```bash
make help       # todos los comandos disponibles
```

## Estructura

```
dl-lab/
├── compose.yaml            # perfiles: cpu | gpu | tracking
├── docker/Dockerfile       # una sola imagen, el índice de wheels decide CPU/CUDA
├── pyproject.toml          # dependencias + dllab instalado como paquete editable
├── jupytext.toml           # emparejado .ipynb <-> .py
├── Makefile
│
├── src/dllab/              # TODO el código reutilizable vive aquí
│   ├── config.py           # dataclasses de configuración + carga de YAML
│   ├── data/               # datasets y dataloaders
│   ├── models/             # arquitecturas + registro build_model()
│   ├── training/loop.py    # bucle de entrenamiento, early stopping, TensorBoard
│   └── utils/              # semillas, selección de dispositivo
│
├── notebooks/              # exploración; ver notebooks/README.md
├── experiments/            # un YAML por experimento
├── scripts/train.py        # runs reproducibles desde CLI
├── tests/                  # pytest sobre src/
├── data/                   # raw / interim / processed  (ignorado por git)
└── outputs/                # runs, checkpoints, logs    (ignorado por git)
```

La clave del diseño: `src/dllab` se instala en modo editable (`pip install -e .`),
así que desde cualquier notebook haces `from dllab.models.mlp import MLP` sin tocar
`sys.path`, y editar el fichero en tu IDE se refleja al instante en el kernel
(con `%autoreload 2`).

## Los dos modos de trabajo

**Explorar** → notebook. Iteras rápido, ves gráficas, te equivocas barato.

**Consolidar** → `scripts/train.py`. Cuando una idea funciona, la fijas en un YAML:

```bash
docker compose --profile cpu exec lab python scripts/train.py \
    --config experiments/mlp_moons.yaml

# barrido rápido sin tocar el YAML
for lr in 1e-2 1e-3 1e-4; do
  docker compose --profile cpu exec lab python scripts/train.py \
      --config experiments/mlp_moons.yaml --override train.lr=$lr
done
```

Cada run crea `outputs/runs/<nombre>-<timestamp>/` con `config.json`, `summary.json`,
el mejor checkpoint y los eventos de TensorBoard.

## GPU

`compose.yaml` tiene un perfil `gpu` que instala las wheels `cu124`. Requisitos:

| Sistema | ¿Funciona la GPU en Docker? |
|---------|-----------------------------|
| Linux + NVIDIA | Sí — instala [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Windows + NVIDIA | Sí — vía WSL2, con el driver de Windows y el toolkit dentro de WSL |
| macOS (Apple Silicon) | **No** — Docker no expone Metal/MPS. Usa el perfil `cpu` para todo, y un venv nativo (`uv venv`) cuando necesites MPS |

Comprobar:

```bash
make gpu
docker compose --profile gpu exec lab-gpu python -c "import torch; print(torch.cuda.is_available())"
```

Si usas `num_workers > 0` en el DataLoader, sube `shm_size` en `compose.yaml`
(ya está en 8 GB para el perfil GPU); es el fallo silencioso más habitual.

## Seguimiento de experimentos

- **TensorBoard** (por defecto, cero configuración): `make tb` → http://localhost:6006
- **MLflow** (cuando quieras comparar decenas de runs con hiperparámetros):
  `make mlflow` → http://localhost:5000

## Calidad

```bash
make test    # pytest
make lint    # ruff
pre-commit install    # limpia outputs de notebooks y sincroniza jupytext al commitear
```

## Colab

Ver `notebooks/README.md`. Resumen: sube el repo a GitHub, abre el notebook con la
URL `colab.research.google.com/github/...`, y la celda de bootstrap clona el repo e
instala `dllab` dentro de Colab. El mismo notebook, los mismos imports, dos entornos.
