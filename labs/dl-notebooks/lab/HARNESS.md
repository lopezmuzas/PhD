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

---

## 4. Los límites del arnés: el caso del Aprendizaje por Refuerzo

El arnés cubre bien el ~95% de los problemas basados en datasets: regresión, clasificación,
series temporales, visión, contrastivo y preentrenamiento autosupervisado de texto. Todos
comparten una misma forma: **los datos existen antes de empezar a entrenar.**

El Aprendizaje por Refuerzo rompe eso, pero conviene ser quirúrgico sobre **dónde** lo rompe.
No es "otro mundo entero": son exactamente cuatro suposiciones, y solo una de ellas es grave.

### 4.1 Las cuatro suposiciones que RL viola

| # | Suposición del arnés | Dónde vive | Qué exige el refuerzo | Gravedad |
|---|---|---|---|---|
| 1 | Los loaders se construyen **una vez**, antes del bucle | [harness.py:170](harness.py#L170) | Los datos dependen de la política *actual* → caducan cada época | 🔴 **Estructural** |
| 2 | Un lote es la tupla `(inputs, targets)` | [harness.py:193](harness.py#L193) | Es la terna `(estado, acción, recompensa)` | 🟡 Cosmética |
| 3 | La pérdida es `f(pred, target)` sacada de `torch.nn.functional` | [harness.py:178](harness.py#L178) | Es $-\log \pi(a\|s) \cdot A$: lleva un **peso escalar por muestra** | 🟡 Cosmética |
| 4 | `val_loss` mide el progreso | `evaluate()` | La métrica es el **retorno acumulado**; la pérdida puede subir mientras el agente mejora | 🟡 Cosmética |

Las tres cosméticas se resuelven sin tocar el arnés: un `Dataset` de tres columnas
(problema 2), una función de pérdida registrada a mano en lugar de un nombre de `F.*`
(problema 3) y un `Callback` que registre el retorno en `state.scratch` (problema 4).

### 4.2 El único cerrojo real

El problema 1 no tiene salida elegante, y la razón es un detalle muy concreto del diseño
actual: en [harness.py:240](harness.py#L240) el `TrainingState` recibe `model`, `optimizer`
y `device`, **pero no los loaders**.

```python
state = TrainingState(config=config, model=components.model,
                      optimizer=components.optimizer, device=device)
#                     ↑ falta: components
```

La consecuencia es precisa: un `Callback` **puede diagnosticar** el entrenamiento (ve los
pesos, los gradientes, el historial) pero **no puede alimentarlo**. No tiene forma de
sustituir `components.train_loader` entre épocas.

Y ahí está toda la cuestión, porque el gradiente de política **no necesita nada más**. Si
un callback pudiera rellenar el loader en `on_epoch_end()` muestreando de la política
actual, REINFORCE entraría en el arnés por la puerta de siempre, **sin un solo
`if config["is_rl"]`**.

### 4.3 Los tres niveles (y hasta dónde tiene sentido llegar)

**Nivel 1 — `harness.py` con el cerrojo abierto.**
Añadir `components` al `TrainingState` (una línea) y escribir un `RolloutCallback` que
regenere el `train_loader` en cada época a partir del modelo actual. Cubre REINFORCE y
RLHF didáctico. El mensaje de fondo: *el arnés ya era más general de lo que parecía; solo
le faltaba dejar que alguien le cambiara los datos.*

**Nivel 2 — `harness_rl.py` como módulo hermano.** ✅ *Esbozado en [harness_rl.py](harness_rl.py).*
Necesario en cuanto entran en juego `gymnasium.Env` y `env.step()`, *replay buffers*, red
de valor (crítico $V(s)$), ventajas (GAE), redes objetivo o corrección *off-policy*. Aquí
el bucle deja de parecerse a "una época sobre un dataset" y no queda nada reutilizable del
bucle de entrenamiento — pero **sí se reutiliza toda la infraestructura**: `Registry`,
`save_run`, `load_run`, `compare_runs`, `plot_runs`, `sweep`, `repeat_with_seeds`.
La comunicación entre ambos arneses es por **checkpoints en disco**: `harness.py` produce
el `weights.pt` base, y `harness_rl.py` lo carga como política inicial.

**Nivel 3 — no construirlo.**
PPO con *clipping*, *value clipping*, normalización de ventajas, *entropy bonus*. Aquí se
usa `CleanRL`, `TorchRL` o `TRL` y se cita. Marcar explícitamente dónde acaba el
laboratorio y empieza la librería es honestidad documental, no una carencia.

| Paradigma | Librería estándar de la industria | Equivalente en el laboratorio |
|---|---|---|
| Supervisado / Autosupervisado / LLMs | PyTorch, HuggingFace `Trainer`, Lightning | `lab/harness.py` |
| Refuerzo con entorno (DQN, PPO, A2C) | Stable-Baselines3, CleanRL, TorchRL | `lab/harness_rl.py` (Nivel 2) |
| Alineación de LLMs (RLHF / DPO) | TRL (*Transformer Reinforcement Learning*) | Nivel 1 sobre `harness.py` |

#### El esbozo que existe hoy: `harness_rl.py`

Hay una versión mínima escrita, en el Nivel 1½: implementa **REINFORCE** (gradiente de
política puro) con la contabilidad completa —rollouts, retornos hacia adelante, tres modos
de ponderación, bonus de entropía, evaluación determinista— y **nada más**. Reutiliza al
100% `Registry`, `Callback`, `TrainingState`, `save_run`, `load_run` y `plot_runs`; lo único
que sustituye es el bucle.

| Función | Papel | Análogo en `harness.py` |
|---|---|---|
| `envs` | Registro nuevo (el único) | `datasets` |
| `collect_episode()` | **Genera** los datos con la política actual | `datasets.build()` — pero corre en cada iteración |
| `returns_to_go()` | Reparte el mérito hacia atrás en el tiempo | *(no existe)* |
| `compute_weights()` | Convierte retornos en el peso de cada muestra | *(no existe)* |
| `policy_loss()` | $-(\log \pi \cdot w)$: supervisado ponderado | `getattr(F, config["loss"])` |
| `train_one_iteration()` | Genera datos **y** da UN paso | `train_one_epoch()` |
| `run_rl_experiment()` | Devuelve un `H.ExperimentResult` normal | `run_experiment()` |
| `compare_rl_runs()` | Igual, pero `best` = **máximo** | `compare_runs()` (usa mínimo) |
| `repeat_rl_with_seeds()` | Obligatorio en RL, no opcional | `repeat_with_seeds()` |

Trae dos entornos de juguete, cada uno aislando **una** idea:

* **`two_armed_bandit`** — un estado, dos acciones, un paso. No hay tiempo ni crédito que
  repartir: solo "¿qué palanca paga más?". Es la prueba de humo del gradiente.
* **`corridor`** — cobrar poco ya (izquierda) o aguantar por el premio grande (derecha).
  Cambiando **solo `gamma`** se invierte la política óptima: con `0.99` aguanta, con `0.6`
  cobra. Demuestra que gamma no es un detalle de implementación sino parte de la
  definición del problema.

El entorno interesante —LM que genera, clasificador que puntúa— **no está en el módulo a
propósito**: depende del vocabulario y del checkpoint de un notebook concreto, así que se
registra desde allí con `@rl.envs.register(...)`. Mismo patrón que los datasets.

> ⚠️ Todo el archivo lleva marcadores `# ALTERNATIVA:` en cada bifurcación de diseño, y
> cierra con un mapa (§🔀) de lo que deliberadamente **no** hace —crítico $V(s)$, GAE, PPO,
> replay buffers, entornos vectorizados, penalización KL, DPO— con la referencia de a dónde
> ir en cada caso. La decisión en cada bifurcación fue **la más fácil de leer, no la mejor**.

### 4.4 Aviso sobre las semillas en RL

Todo lo dicho en `repeat_with_seeds()` sobre el umbral de credibilidad ($2\sigma$) se
vuelve **crítico** en refuerzo. El estimador de REINFORCE tiene varianza altísima: con
corpus pequeños las curvas bailan de forma salvaje entre semillas y es trivial "demostrar"
un resultado que era ruido.

Eso no es un defecto de la implementación: **es la razón por la que existen los
*baselines*, los críticos y las ventajas normalizadas.** Un experimento de RL con una sola
semilla no dice nada. Con 5 semillas y el abanico de curvas dibujado, se ve de golpe por
qué el refuerzo necesita maquinaria que el aprendizaje supervisado no necesita.

### 4.5 El pipeline de tres etapas y de dónde sale la recompensa

La pregunta que decide el diseño de la Fase 3 no es qué algoritmo usar, sino **de dónde
sale la recompensa escalar $R$**. En un videojuego la da el entorno gratis; en texto no
existe ninguna función matemática que puntúe una frase. La respuesta de RLHF: se entrena un
clasificador con juicios humanos y **ese clasificador se convierte en la función de
recompensa**.

```none
FASE 1 ──► LM preentrenado ──────────────────┐
   (predice siguiente palabra)               │
                                             ▼
FASE 2 ──► Clasificador de sentimiento ──► ES EL REWARD MODEL
   (pocos ejemplos etiquetados)              │
                                             │ puntúa
                                             ▼
FASE 3 ──► El LM de Fase 1 GENERA texto ──► recompensa ──► REINFORCE
   (política = LM, acción = token)
```

Implicaciones para el arnés:

* La Fase 2 **no es un peldaño intermedio**: es el ingrediente que hace posible la Fase 3.
* El checkpoint de la Fase 1 se carga **dos veces**: como inicialización del clasificador
  (rama supervisada) y como política a afinar (rama de refuerzo).
* La Fase 3 necesita **tres modelos vivos a la vez**: la política (entrenable), el reward
  model (congelado, `eval()`) y una copia congelada de la Fase 1 para el término KL. El
  `config` de un experimento de RL, por tanto, apunta a **varios** `run_id` anteriores, no
  a uno solo. Eso es una diferencia real de trazabilidad respecto a los configs actuales.
* Aparecerá *reward hacking*: la política aprende a engañar al juez (`"excelente maravilla
  excelente maravilla"`) en vez de a escribir bien. No es un bug, es
  [Goodhart](https://en.wikipedia.org/wiki/Goodhart%27s_law), y es la mejor demostración
  posible de por qué RLHF real necesita la penalización KL contra el modelo original.

> 📖 El desarrollo conceptual completo de este pipeline (incluyendo DPO como colapso de las
> Fases 2 y 3) está en `docs/01-fundamentos/03-como-se-entrena-una-red/3.7-transferencia-y-finetuning.md`.
> La reinterpretación de la pérdida de política como *"supervisado ponderado"* está en
> [PYTORCH_EN_EL_ARNES.md](PYTORCH_EN_EL_ARNES.md), sección de los 4 paradigmas.
