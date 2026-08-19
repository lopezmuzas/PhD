# dl-notebooks · El itinerario práctico

Los 31 notebooks del itinerario, de N00 a N30.

| Dónde | Qué |
|---|---|
| `docs/00-metodo/itinerario.md` | Los principios, el dataset y el porqué |
| `docs/04-proyecto/09-indice-notebooks/notebooks.md` | La lista de trabajo, con los bloques para el LLM |
| `docs/04-proyecto/10-bitacora-experimentos/` | La bitácora |

## Arranque

```bash
pip install -r requirements-dev.txt
make test      # comprueba que el arnés funciona
make lab       # abre jupyter
```

## Estructura

```
dl-notebooks/
├── lab/              código reutilizable, importado por los notebooks
│   ├── harness.py    el arnés de experimentos (N00)
│   └── data.py       los generadores sintéticos (N01)
├── notebooks/        N00 … N30
├── tests/            que el arnés no se rompa sin avisar
└── runs/             resultados (ignorado por git)
```

## Convenios

**Idioma.** El código va en inglés: funciones, métodos y variables. Los comentarios y
docstrings son bilingües — inglés para *qué hace*, español para *por qué*.

**Nombres semánticos.** `build_line_dataset`, `irreducible_error`, `final_loss`. Si hace
falta un comentario para saber qué guarda una variable, el nombre está mal.

**El código reutilizable vive en `lab/`, nunca en una celda.** Los notebooks lo escriben
con `%%writefile` y lo importan. Así no hay copias que diverjan.

**Todo notebook empieza con este arranque**, para que funcione desde cualquier carpeta y
los resultados caigan siempre en el mismo `runs/`:

```python
import os, sys
from pathlib import Path
while not (Path.cwd() / "lab").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
sys.path.insert(0, str(Path.cwd()))
```

## La API del arnés

> 📖 **Documentación:**
> - [`lab/HARNESS.md`](lab/HARNESS.md): Referencia de la API del arnés, estructura del config, salidas y persistencia.
> - [`lab/PYTORCH_EN_EL_ARNES.md`](lab/PYTORCH_EN_EL_ARNES.md): Manual de usuario de las funciones de PyTorch utilizadas internamente por el arnés.

```python
from lab import harness as H

H.run_experiment(config, seed=0)              # una ejecución, guardada en disco
H.sweep(config, "optimizer_args.lr", [...])   # variar UNA clave
H.compare_runs(run_ids)                       # tabla comparativa
H.plot_runs(run_ids)                          # curvas superpuestas
H.repeat_with_seeds(config, n_seeds=5)        # dispersión entre semillas

@H.datasets.register("name")    # añadir componentes sin tocar el arnés
@H.models.register("name")
@H.optimizers.register("name")
```

Un experimento es un diccionario:

```python
config = {
    "name": "n00",
    "dataset": "line",  "dataset_args": {"noise_std": 0.5},
    "model": "mlp",     "model_args": {"hidden_size": 16},
    "optimizer": "adam", "optimizer_args": {"lr": 1e-2},
    "epochs": 30, "loss": "mse_loss",
}
```
