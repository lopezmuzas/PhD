# 🛠️ Documentación de la API: `lab/harness.py`

> **Propósito:** El arnés de experimentos (*Experiment Harness*) estandariza la ejecución, persistencia, trazabilidad y comparación de modelos.  
> **Regla de diseño:** *Todo lo que **cambia** entre experimentos vive en el `config`. Todo lo que **no cambia** vive en el arnés.*  
> 📘 *Ver también:* [`PYTORCH_EN_EL_ARNES.md`](PYTORCH_EN_EL_ARNES.md) (manual de las funciones internas de PyTorch empleadas en el arnés).

---

## 1. Anatomía del `base_config`

Un experimento en el arnés se define íntegramente mediante un diccionario de Python serializable a JSON. 

### Ejemplo de Configuración Estándar

```python
base_config = {
    "name": "n00",
    "dataset": "line",
    "dataset_args": {
        "n_samples": 512,
        "noise_std": 0.1
    },
    "model": "mlp",
    "model_args": {
        "hidden_size": 16,
        "n_hidden_layers": 1
    },
    "optimizer": "adam",
    "optimizer_args": {
        "lr": 1e-2
    },
    "epochs": 30,
    "loss": "mse_loss",
    # Campos opcionales:
    "seed": 0,
    "device": "cpu"  # O "cuda", si se omite auto-detecta GPU
}
```

### Especificación de Campos del Config

| Campo | Tipo | Obligatorio | Descripción |
|---|---|:---:|---|
| `name` | `str` | Sí | Nombre base del experimento (prefijo para el `run_id`). |
| `dataset` | `str` | Sí | Nombre del generador/dataset registrado en `harness.datasets`. |
| `dataset_args` | `dict` | No | Argumentos (`kwargs`) que recibe la función generadora del dataset. |
| `model` | `str` | Sí | Nombre del modelo registrado en `harness.models`. |
| `model_args` | `dict` | No | Argumentos (`kwargs`) para el constructor de la arquitectura. |
| `optimizer` | `str` | No | Optimizador (`"adam"`, `"sgd"`, etc. Default: `"adam"`). |
| `optimizer_args` | `dict` | No | Parámetros del optimizador (ej. `{"lr": 0.01, "weight_decay": 1e-4}`). |
| `epochs` | `int` | Sí | Número total de épocas completas de entrenamiento. |
| `loss` | `str` | No | Nombre de la función en `torch.nn.functional` (Default: `"mse_loss"`). |
| `seed` | `int` | No | Semilla aleatoria (Default: `0` si no se especifica en la llamada). |
| `device` | `str` | No | Dispositivo de cómputo (`"cpu"`, `"cuda"`). Auto-detectado si no se indica. |

### 💡 Funciones de Pérdida Disponibles (`loss`)

El campo `"loss"` en el `config` acepta cualquier nombre de función matemática de pérdida disponible en el módulo [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html) de PyTorch (que es resuelta dinámicamente usando `getattr`). 

Aquí tienes las más utilizadas en la práctica y sus casos de uso recomendados:

| Valor en `"loss"` | Función de PyTorch | Tipo de Tarea | Descripción / Requisito del modelo |
|---|---|---|---|
| `"mse_loss"` (Default) | `F.mse_loss` | Regresión | Error Cuadrático Medio ($L_2$). Penaliza con fuerza los errores grandes. |
| `"l1_loss"` | `F.l1_loss` | Regresión | Error Absoluto Medio ($L_1$). Más robusto ante valores atípicos (*outliers*). |
| `"huber_loss"` | `F.huber_loss` | Regresión | Transición suave entre MSE (para errores pequeños) y L1 (para errores grandes). |
| `"cross_entropy"` | `F.cross_entropy` | Clasificación Multiclase | Entropía cruzada. **Espera recibir logits** (salidas sin activar) y etiquetas de clase como enteros ($0, 1, 2, ...$). |
| `"binary_cross_entropy_with_logits"` | `F.binary_cross_entropy_with_logits` | Clasificación Binaria | Entropía cruzada binaria numéricamente estable. **Espera recibir logits** (sin activación final). |
| `"binary_cross_entropy"` | `F.binary_cross_entropy` | Clasificación Binaria | Entropía cruzada binaria estándar. **Requiere** que el modelo termine en una capa de activación `Sigmoid`. |
| `"nll_loss"` | `F.nll_loss` | Clasificación Multiclase | Pérdida de verosimilitud logarítmica negativa. **Requiere** que el modelo termine en una activación `LogSoftmax`. |

---

## 2. Salida y Resultados de un Experimento

Al ejecutar:
```python
result = H.run_experiment(base_config, seed=0)
```

Se obtienen dos cosas:
1. **Un objeto en memoria (`ExperimentResult`)** devuelto por la función.
2. **Un directorio en disco (`runs/<run_id>/`)** con la persistencia completa del experimento.

---

### A. Estructura del Objeto `ExperimentResult` (en memoria)

```python
@dataclass
class ExperimentResult:
    run_id: str             # Identificador único (ej: "n00_s0_20260819-083000")
    config: dict            # Diccionario de configuración exacto usado
    seed: int               # Semilla utilizada
    history: list[dict]     # Métricas por época [{'epoch': 0, 'train_loss': ..., 'val_loss': ...}, ...]
    model: nn.Module        # Instancia del modelo de PyTorch entrenado
    elapsed_seconds: float  # Tiempo total de entrenamiento en segundos
    scratch: dict           # Diccionario para datos de diagnóstico de callbacks
```

#### Métodos y Propiedades de `ExperimentResult`:
* `result.final_metrics -> dict`: Devuelve el diccionario de métricas de la última época (ej. `{"epoch": 29, "train_loss": 0.012, "val_loss": 0.014}`).
* `result.metric(name="val_loss") -> float`: Devuelve el valor final de una métrica concreta.
* `result.model`: Acceso directo a los pesos y arquitectura del modelo para inferencia o inspección.

---

### B. Estructura en Disco (`runs/<run_id>/`)

Cada ejecución genera una carpeta aislada e inmutable dentro de `runs/`:

```none
runs/
└── n00_s0_20260819-083000/
    ├── config.json   # Copia exacta del diccionario de configuración
    ├── metrics.csv   # Histórico época a época (DataFrame exportado)
    ├── weights.pt    # Pesos del modelo (torch.save(model.state_dict()))
    └── meta.json     # Metadatos del entorno y reproducibilidad
```

#### Contenido de `meta.json`:
```json
{
  "run_id": "n00_s0_20260819-083000",
  "seed": 0,
  "date": "2026-08-19 08:30:00",
  "commit": "a1b2c3d",
  "elapsed_seconds": 1.42,
  "torch": "2.4.0",
  "python": "3.11.8",
  "host": "macbook-pro"
}
```

---

## 3. Catálogo de Clases y Funciones

### 🏷️ Sistema de Registros (`Registry`)

Permite desacoplar el string del `config` de las clases de Python, manteniendo el config 100 % serializable en JSON.

```python
from lab.harness import models, datasets, optimizers

# 1. Registrar un dataset
@datasets.register("line")
def build_line_dataset(n_samples=512, noise_std=0.1, batch_size=32):
    ...
    return train_loader, val_loader

# 2. Registrar un modelo
@models.register("mlp")
def build_mlp(hidden_size=16, n_hidden_layers=1, in_features=1, out_features=1):
    ...
    return model

# 3. Construir manualmente mediante el registro
model = models.build("mlp", hidden_size=32)
```

---

### 🎲 Reproducibilidad y Semillas (`seed`)

En informática el azar real no existe: se utilizan **generadores de números pseudoaleatorios** basados en fórmulas matemáticas deterministas. La **semilla (`seed`)** es el número inicial que alimenta esas fórmulas. A misma semilla, la secuencia de números generados es **100 % idéntica**.

#### ¿Qué decide la semilla durante el entrenamiento?
1. **Inicialización de pesos ($W, b$):** Decide en qué punto exacto del espacio de parámetros arranca la red.
2. **Barajado de datos (*Data Shuffling*):** Decide en qué orden se entregan los lotes (*batches*) en cada época dentro del `DataLoader`.
3. **División de datos (*Train/Val Split*):** Decide qué ejemplos específicos caen en entrenamiento y cuáles en validación.
4. **Técnicas estocásticas:** Decide qué neuronas se apagan en `Dropout` o qué transformaciones se aplican en `Data Augmentation`.

#### `set_seed(seed: int, deterministic: bool = True) -> None`
Sincroniza todos los generadores aleatorios a la vez:
* `random.seed(seed)` (Python estándar)
* `np.random.seed(seed)` (NumPy)
* `torch.manual_seed(seed)` (PyTorch CPU)
* `torch.cuda.manual_seed_all(seed)` (PyTorch GPU / CUDA)
* `torch.backends.cudnn.deterministic = True` (Fuerza operaciones de álgebra deterministas en GPU).

> ⚠️ **El concepto de "Ruido de Fondo":** Cambiar solo la semilla de `0` a `1` produce resultados ligeramente distintos (dispersión natural). Antes de concluir que una nueva técnica funciona, debes verificar con `H.repeat_with_seeds()` que la mejora obtenida es significativamente mayor que la dispersión natural entre semillas ($> 2\sigma$).

#### `current_git_commit() -> str`
Obtiene el hash corto de Git del commit actual para asegurar la trazabilidad del código en `meta.json`.

---

### 🔄 Bucle Principal y Entrenamiento

#### `run_experiment(config, seed=None, callbacks=None, save=True, verbose=True) -> ExperimentResult`
Ejecuta un experimento completo de principio a fin:
1. Resuelve la semilla y el dispositivo (`resolve_device`).
2. Instancia datos, modelo, optimizador y función de pérdida (`build_components`).
3. Ejecuta el bucle de épocas llamando a `train_one_epoch` y `evaluate`.
4. Dispara los eventos correspondientes de `Callback`.
5. Si `save=True`, guarda los artefactos en `runs/<run_id>/`.
6. Devuelve el `ExperimentResult`.

#### `build_components(config: dict, seed: int) -> Components`
Construye y empaqueta en una dataclass `Components`:
* `train_loader`, `val_loader`: DataLoaders generados por el builder de dataset.
* `model`: `nn.Module` instanciado.
* `optimizer`: Optimizador configurado sobre `model.parameters()`.
* `loss_fn`: Función de pérdida resuelta desde `torch.nn.functional`.

#### `train_one_epoch(components, state, callbacks) -> float`
Ejecuta un pase completo sobre `components.train_loader`, calcula pérdidas, actualiza pesos con backprop y reporta el `loss` promedio de entrenamiento.

#### `evaluate(components, device) -> dict`
Bajo contexto `@torch.no_grad()`, evalúa el modelo en `components.val_loader` y devuelve `{"loss": float}`.

---

### 🪝 Sistema de Diagnóstico (`Callback` & `TrainingState`)

Permite inyectar sondas, métricas y diagnósticos en tiempo de ejecución sin modificar el bucle principal:

```python
class Callback:
    def on_train_start(self, state: TrainingState) -> None: ...
    def on_batch_end(self, state: TrainingState, loss: float) -> None: ...
    def on_epoch_end(self, state: TrainingState) -> None: ...
    def on_train_end(self, state: TrainingState) -> None: ...
```

`TrainingState` expone durante el bucle:
* `state.model`, `state.optimizer`, `state.device`
* `state.epoch`, `state.step`
* `state.history`
* `state.scratch`: diccionario para que los callbacks almacenen información intermedia.

---

### 💾 Persistencia y Consulta de Runs

#### `save_run(result: ExperimentResult) -> Path`
Escribe en `runs/<run_id>/` los 4 ficheros (`config.json`, `metrics.csv`, `weights.pt`, `meta.json`).

#### `load_run(run_id: str) -> dict`
Carga desde disco un run previo devolviendo un diccionario con:
`{"run_id", "config", "meta", "history"}` (donde `history` es un `pd.DataFrame`).

#### `list_runs(pattern: str = "") -> list[str]`
Devuelve la lista ordenada de `run_id`s guardados en la carpeta `runs/` que coincidan con `pattern`.

---

### 📊 Comparación, Gráficos y Barridos (Sweeps)

#### `compare_runs(run_ids: list[str], metric: str = "val_loss") -> pd.DataFrame`
Genera una tabla comparativa en Pandas donde cada fila es un run con su configuración aplanada, métrica final, métrica óptima (`best`) y duración.

#### `plot_runs(run_ids: list[str], metrics=("train_loss", "val_loss"), log_scale=True, ax=None)`
Dibuja las curvas de aprendizaje superpuestas de múltiples ejecuciones para comparar su convergencia.

#### `sweep(base_config: dict, key: str, values: list, seeds=(0,)) -> list[str]`
Ejecuta múltiples experimentos variando **un único parámetro** a través de una lista de valores y semillas. Soporta claves anidadas con notación por puntos (ej. `"optimizer_args.lr"` o `"model_args.hidden_size"`).

```python
# Ejemplo de barrido de Learning Rate
run_ids = H.sweep(base_config, "optimizer_args.lr", [1e-1, 1e-2, 1e-3, 1e-4])
H.plot_runs(run_ids)
```

#### `repeat_with_seeds(config: dict, n_seeds: int = 5, metric: str = "val_loss") -> pd.DataFrame`
Ejecuta el mismo experimento con $N$ semillas distintas. Imprime la media, desviación estándar ($\sigma$) y rango del resultado, estableciendo el **umbral de ruido / credibilidad**:
$$\text{Diferencia significativa} > 2 \times \sigma$$
