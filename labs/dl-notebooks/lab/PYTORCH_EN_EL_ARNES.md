# 📘 Guía básica de PyTorch en el arnés (`harness.py`)

> **Objetivo:** Este documento es un manual de referencia rápido para entender **qué piezas de PyTorch usa `harness.py` internamente**, para qué sirve cada una y cómo encajan en el ciclo de entrenamiento sin entrar en complejidades innecesarias.

---

## 1. Mapa de componentes de PyTorch utilizados

En `harness.py`, PyTorch se utiliza en **4 áreas clave**:

```none
  ┌───────────────────────────────────────────────────────────┐
  │ 1. MODELOS Y CAPAS (torch.nn)                             │
  │    • nn.Module: Estructura de la red                      │
  │    • model.train() / model.eval(): Modo de trabajo        │
  │    • model.parameters(): Los pesos a optimizar            │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ 2. DATOS (torch.utils.data.DataLoader)                    │
  │    • DataLoader: Entrega lotes (inputs, targets)          │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ 3. EL BUCLE DE CÁLCULO Y OPTIMIZACIÓN                     │
  │    • optimizer.zero_grad(): Limpiar gradientes viejos     │
  │    • loss_fn(pred, y): Medir el error (torch.nn.functional│
  │    • loss.backward(): Calcular derivadas (Autograd)       │
  │    • optimizer.step(): Ajustar pesos                      │
  │    • @torch.no_grad(): Evaluación rápida sin memoria extra│
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ 4. DISPOSITIVO Y PERSISTENCIA                             │
  │    • torch.device / .to(device): Enviar a CPU o GPU (CUDA)│
  │    • torch.save(model.state_dict(), ...): Guardar pesos   │
  │    • torch.manual_seed(): Reproducibilidad                │
  └───────────────────────────────────────────────────────────┘
```

---

## 2. Los métodos de PyTorch explicados uno a uno

### ① Modelos y Arquitectura (`torch.nn`)

#### ¿Qué es `nn.Module`?
Es la **clase base universal** para cualquier red neuronal o capa en PyTorch. No es simplemente una clase abstracta: actúa como una máquina de estados que gestiona automáticamente el árbol de parámetros entrenables ($W$ y $b$), las transiciones de hardware (CPU/GPU), el modo de ejecución (train/eval) y la serialización a disco.

#### ¿Qué hace en el arnés (`harness.py`)?
El arnés trata a cualquier modelo como una **caja negra con interfaz estandarizada**. Al heredar de `nn.Module`, el arnés puede entrenar, evaluar y persistir cualquier arquitectura (MLP, CNN, Transformer, etc.) sin conocer sus detalles internos:

```python
# Operaciones que el arnés delega directamente en la interfaz de nn.Module:
model.to(device)                                        # Enviar arquitectura completa a CPU/GPU
optimizer = torch.optim.Adam(model.parameters(), lr)    # Registrar los pesos a optimizar
model.train()                                           # Activar modo entrenamiento
model.eval()                                            # Activar modo evaluación (desactiva Dropout, fija BatchNorm)
preds = model(inputs)                                   # Ejecutar pase hacia adelante (forward con hooks)
torch.save(model.state_dict(), "weights.pt")            # Extraer diccionario de pesos para guardarlos
```

---

#### Funciones y Métodos Base de `nn.Module` Explicados

* **`__init__()` y `super().__init__()`**:
  * **Qué hace:** Inicializa los diccionarios internos del módulo (`_parameters`, `_modules`, `_buffers`).
  * **Regla obligatoria:** Toda subclase debe invocar `super().__init__()` al principio de su constructor; de lo contrario, PyTorch no podrá registrar las capas ni los parámetros.

* **`forward(*inputs)`**:
  * **Qué hace:** Define la lógica matemática y el flujo de datos desde la entrada $X$ hasta la salida $\hat{y}$.
  * **Uso:** Aquí se aplican las operaciones, capas intermedias, activaciones o saltos residuales.

* **`__call__(*inputs)` (invocar `model(inputs)`)**:
  * **Qué hace:** Es el método especial de Python que se ejecuta al llamar al modelo como una función.
  * **Por qué importa:** `__call__` ejecuta internamente `forward()`, pero además gestiona *hooks* de depuración, comprobaciones de estado y perfiles de autograd. **Nunca llames a `model.forward(x)` directamente; usa siempre `model(x)`.**

* **`model.train()` y `model.eval()`**:
  * **Qué hacen:** Conmutan una bandera booleana interna en cascada para el modelo y todos sus submódulos hijos.
  * **Por qué importa:** 
    * En `train_one_epoch()` se activa `model.train()` para permitir el aprendizaje y el comportamiento estocástico de capas como `nn.Dropout` y la actualización de medias en `nn.BatchNorm`.
    * En `evaluate()` se pone `model.eval()` para desactivar `Dropout` y usar las medias acumuladas en `BatchNorm`, garantizando predicciones deterministas.

* **`model.parameters()` y `model.named_parameters()`**:
  * **Qué hace:** Devuelve un generador iterador con todos los tensores `nn.Parameter` que tienen `requires_grad=True`.
  * **Por qué importa:** Se le pasa al optimizador para que sepa exactamente qué matrices de pesos y vectores de sesgos debe ajustar durante el gradiente descendente.

* **`model.to(device)`**:
  * **Qué hace:** Transfiere recursivamente todos los pesos, sesgos y buffers del modelo a la memoria del dispositivo especificado (`"cuda"`, `"cpu"`, `"mps"`).

* **`model.state_dict()` y `model.load_state_dict(...)`**:
  * **Qué hace:** `state_dict()` exporta un diccionario estándar de Python `{nombre_capa: tensor_pesos}` sin datos del grafo de autograd. `load_state_dict()` restaura dichos pesos en la estructura del modelo.

* **`register_buffer(name, tensor)`**:
  * **Qué hace:** Registra un tensor que forma parte del estado del modelo y debe guardarse en el `state_dict()` y moverse con `.to(device)`, pero **no es un parámetro optimizable por gradiente** (ej. medias móviles o matrices de posición fija).

* **`model.zero_grad()`**:
  * **Qué hace:** Pone a cero los gradientes acumulados en todos los parámetros del modelo (`param.grad = None` o `0`).

---

#### Opciones y Formas de Construir Modelos (`nn.Module`)

PyTorch ofrece varias alternativas para estructurar redes según la complejidad requerida:

##### 1. Subclassing Personalizado (Máximo Control y Flexibilidad)
Heredar directamente de `nn.Module` implementando `__init__` y `forward`. Es el estándar para arquitecturas complejas con conexiones residuales (*Skip Connections*), ramas paralelas o bucles.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.relu(out + residual)  # Conexión residual
```

##### 2. `nn.Sequential` (Flujo Lineal Directo)
Encadena capas en orden estricto donde la salida de la capa $i$ es la entrada inmediata de la capa $i+1$.

```python
# Ideal para bloques sencillos sin bifurcaciones
simple_mlp = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 10)
)
```

##### 3. `nn.ModuleList` (Listas Dinámicas de Capas)
Si guardas capas en una lista estándar de Python (`[nn.Linear(...)]`), PyTorch **no las rastreará** en `model.parameters()` ni en `.to(device)`. `nn.ModuleList` asegura el registro formal de cada capa.

```python
class DeepMLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        # Permite crear N capas ocultas de forma dinámica
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
```

##### 4. `nn.ModuleDict` (Diccionarios de Subredes o Cabezales)
Permite indexar submódulos por clave de texto, ideal para arquitecturas multi-tarea o cabezales intercambiables.

```python
class MultiTaskNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.backbone = nn.Linear(in_dim, 64)
        self.heads = nn.ModuleDict({
            "classification": nn.Linear(64, 2),
            "regression": nn.Linear(64, 1)
        })

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        features = F.relu(self.backbone(x))
        return self.heads[task](features)
```

##### 5. Composición Jerárquica (Módulos dentro de Módulos)
Un `nn.Module` puede contener instancias de otros `nn.Module`, construyendo un árbol jerárquico modular.

```python
class FullClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.block = ResNetBlock(in_features=64, hidden_features=128)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.block(x)
        return self.head(x)
```

---

#### ¿Por qué importa este diseño?
1. **Desacoplamiento total:** Puedes sustituir un MLP por un Transformer en `@models.register("mi_modelo")` sin modificar una sola línea de `harness.py`.
2. **Cero fugas de memoria o parámetros:** La gestión centralizada garantiza que ningún peso quede fuera del optimizador ni en un dispositivo equivocado.
3. **Reutilización de bloques:** Facilita la creación de arquitecturas complejas a partir de bloques pequeños y testeables.

---

### ② Carga de Datos (`torch.utils.data.DataLoader`)

* **`DataLoader`**:
  * **Qué hace:** Envuelve un conjunto de datos (`Dataset`) y los agrupa en pequeños lotes (*batches* o minilotes), gestionando el barajado (*shuffling*), el orden y la carga paralela con múltiples hilos (`num_workers`).
  * **En el arnés:** En cada iteración del bucle `for inputs, targets in components.train_loader:`, el `DataLoader` entrega tensores con las entradas $X$ y las dianas esperadas $y$.

#### 🧭 ¿Cómo sabe la red "contra qué predecir"? Los 4 Paradigmas en PyTorch

La red neuronal es ciega: no sabe qué representan los números ni de dónde sale la verdad fundamental. **Quien define la diana ($y$) es el diseño de tu `DataLoader` y tu función de pérdida**:

##### 1. Aprendizaje Supervisado (Clasificación / Regresión)
El dataset entrega directamente la tupla `(X, y)` donde $y$ es una **etiqueta generada por un anotador humano**.

```python
# Dataset: [(foto_gato, 0), (foto_perro, 1), (foto_pajaro, 2)]
for inputs, targets in dataloader:
    logits = model(inputs)                          # Salida de la red [batch, num_classes]
    loss = F.cross_entropy(logits, targets)         # Compara contra la etiqueta humana
    loss.backward()
    optimizer.step()
```

##### 2. Aprendizaje Autosupervisado (LLMs tipo GPT / Next-Token Prediction)
No hay humanos etiquetando. **El propio dato se supervisa a sí mismo** desfasando la secuencia una posición hacia la derecha.

```python
# Texto crudo: "El gato bebe leche fresca" -> Tokens: [12, 45, 89, 34, 78]
for sequence in dataloader:
    inputs  = sequence[:, :-1]                      # "El gato bebe leche"
    targets = sequence[:, 1:]                       # "gato bebe leche fresca" (+1 posición)
    
    logits = model(inputs)                          # [batch, seq_len, vocab_size]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        targets.view(-1)
    )                                               # Se mide el acierto de la siguiente palabra
    loss.backward()
    optimizer.step()
```

##### 3. Aprendizaje No Supervisado / Contrastivo (SimCLR / InfoNCE)
No existen etiquetas $y$. El dataset entrega datos crudos y el código genera **dos vistas aumentadas** de cada muestra. La pérdida fuerza a que ambas vistas tengan representaciones cercanas y se alejen del resto del lote.

```python
# Dataset: [imagen1, imagen2, imagen3] (sin etiquetas)
for raw_images in dataloader:
    view_A = transform_crop(raw_images)             # Vista A (recorte aleatorio)
    view_B = transform_color(raw_images)            # Vista B (filtro de color)
    
    emb_A = model(view_A)                           # Vector latente A
    emb_B = model(view_B)                           # Vector latente B
    
    # Pérdida matemática de distancia relativa (ej: InfoNCE / Contrastive)
    loss = contrastive_loss(emb_A, emb_B)
    loss.backward()
    optimizer.step()
```

##### 4. Aprendizaje por Refuerzo (Policy Gradient / RLHF)
No hay un dataset estático de entrenamiento; la red interactúa con un **entorno** y recibe una **recompensa escalar ($R$)**.

```python
# Interacción con un simulador / entorno
state = env.reset()

action_probs = model(state)                         # Probabilidades de cada acción [p_izq, p_der]
action = torch.multinomial(action_probs, 1)         # Muestrear una acción según la probabilidad
next_state, reward = env.step(action.item())        # El entorno devuelve una recompensa (+1 o -1)

# Pérdida de política: fomenta la acción si R > 0, la desalienta si R < 0
loss = -torch.log(action_probs[action]) * reward
loss.backward()
optimizer.step()
```

###### 🔑 La reinterpretación clave: el refuerzo **es** supervisado ponderado

Mirando el bloque anterior, la tentación es concluir que RL es otro algoritmo distinto.
No lo es. Compara la pérdida de política con la entropía cruzada de toda la vida:

```python
# Supervisado:   la etiqueta la pone un humano, y pesa siempre 1
loss = -torch.log(probs[etiqueta_humana])

# Refuerzo:      la "etiqueta" es lo que el propio modelo hizo, y pesa R
loss = -torch.log(probs[accion_que_yo_elegi]) * recompensa
```

Son **la misma fórmula** con dos cambios:

1. La diana ya no viene de un anotador: **es la propia salida del modelo** (`accion`).
2. Cada muestra lleva un **peso escalar** ($R$) en lugar de contar todas por igual.

De ahí la formulación en una frase que conviene memorizar:

> **El gradiente de política es aprendizaje supervisado sobre tus propias muestras,
> ponderado por lo bien que salieron.**

Si la acción salió bien ($R > 0$), el signo empuja su probabilidad **hacia arriba**: "haz
más de esto". Si salió mal ($R < 0$), el signo se invierte y la empuja **hacia abajo**:
"haz menos de esto". Eso es el algoritmo **REINFORCE** completo, sin crítico, sin
ventajas, sin GAE. Lo demás que verás en PPO o A2C son técnicas para **reducir la
varianza** de este mismo estimador, no algoritmos distintos.

###### ¿Y de dónde sale $R$ cuando no hay simulador?

En un videojuego $R$ es gratis: la da el entorno. En texto no existe ninguna función
matemática que puntúe una frase. La respuesta de RLHF: **se entrena un clasificador con
juicios humanos y ese clasificador pasa a ser la función de recompensa.**

```python
# El reward model NO es un entorno: es un clasificador entrenado en la fase anterior
reward_model = load_classifier("runs/fase2_finetune/weights.pt")
reward_model.eval()                                  # Congelado: es un juez, no un alumno

texto_generado = politica.generate(prompt)           # La política (el LM) actúa
with torch.no_grad():
    R = reward_model(texto_generado).softmax(-1)[1]  # P(positivo) → recompensa escalar
```

Esto encadena los paradigmas de forma muy concreta: **el modelo autosupervisado (§2) es la
política, y el modelo supervisado (§1) es la recompensa.** Los cuatro paradigmas de esta
sección no son cuatro cajones separados; en un pipeline moderno se acoplan unos a otros.

###### Las tres cosas que rompen el `for inputs, targets in loader`

Por eso este paradigma no encaja en el bucle canónico del arnés, y merece la pena ser
preciso sobre **qué** exactamente se rompe:

| Suposición del bucle estándar | Qué exige el refuerzo |
|---|---|
| Un lote es una tupla `(inputs, targets)` | Es una **terna** `(estado, acción, recompensa)` |
| La pérdida es `f(pred, target)` de `torch.nn.functional` | Lleva un **peso escalar por muestra**: no existe en `F.*` |
| El `DataLoader` se construye **una vez**, antes de entrenar | Los datos los genera la **política actual** → caducan en cada época |

Las dos primeras son cosméticas (se resuelven con un `Dataset` de tres columnas y una
pérdida propia). **La tercera es estructural**: obliga a regenerar los datos a mitad del
entrenamiento, y eso ya no es un detalle de formato sino un cambio en la forma del bucle.
Es la razón real por la que la industria usa herramientas separadas
(`Stable-Baselines3`, `CleanRL`, `TorchRL`, `TRL`) en lugar de un `Trainer` supervisado.

---

### ③ El Bucle de Entrenamiento (Los 4 Pasos Sagrados)

Dentro de `train_one_epoch()` se ejecutan exactamente las 4 órdenes canónicas de PyTorch:

```python
# 1. Limpiar gradientes acumulados
components.optimizer.zero_grad()

# 2. Pase hacia adelante (Forward) y cálculo de pérdida
loss = components.loss_fn(components.model(inputs), targets)

# 3. Retropropagación (Backward)
loss.backward()

# 4. Actualización de pesos
components.optimizer.step()
```

| Función de PyTorch | ¿Qué hace internamente? |
|---|---|
| **`optimizer.zero_grad()`** | Pone a cero los gradientes (`weight.grad = 0`). PyTorch por defecto los acumula en cada iteración; si no los limpias, se sumarían a los del lote anterior. |
| **`loss_fn(...)`** | Compara la predicción con el objetivo usando funciones como `torch.nn.functional.mse_loss` o `cross_entropy` y devuelve un tensor escalar con el error. |
| **`loss.backward()`** | El motor de diferenciación automática (*Autograd*) recorre el grafo de operaciones hacia atrás y calcula $\frac{\partial \text{loss}}{\partial w}$ para cada peso del modelo. |
| **`optimizer.step()`** | Modifica los pesos aplicando la fórmula del optimizador seleccionado (ej. $w \leftarrow w - \eta \cdot \text{grad}$ en SGD/Adam). |

---

### ④ Modo de Evaluación (`@torch.no_grad()`)

* **`@torch.no_grad()`**:
  * **Qué hace:** Desactiva temporalmente el motor de cálculo de gradientes (*Autograd*).
  * **Por qué se usa en `evaluate()`:** Cuando solo queremos medir el error en el conjunto de validación, no necesitamos calcular derivadas. Desactivarlo reduce drásticamente el uso de memoria RAM/VRAM y acelera la ejecución.
* **`loss.item()`**:
  * **Qué hace:** Extrae el valor numérico (un float de Python normal) de un tensor que contiene un solo número. Permite acumular el error para estadísticas sin arrastrar el grafo de memoria de PyTorch.

---

### ⑤ Dispositivo y Aceleración (`torch.device` y `.to(device)`)

* **`torch.cuda.is_available()`**:
  * Comprueba si el equipo cuenta con una GPU NVIDIA con soporte CUDA disponible.
* **`torch.device("cuda" | "cpu")`**:
  * Representa el destino de cómputo donde vivirán los tensores.
* **`tensor.to(device)` / `model.to(device)`**:
  * Mueve los datos y los pesos a la memoria de la GPU (o CPU).  
  * *Regla de oro de PyTorch:* Tanto las entradas como el modelo deben estar en el **mismo dispositivo**; de lo contrario, saltará un error de tipo `RuntimeError: Expected all tensors to be on the same device`.

---

### ⑥ Guardar y Restaurar Modelos (`torch.save` y `.state_dict()`)

* **`model.state_dict()`**:
  * Devuelve un diccionario estándar de Python que mapea el nombre de cada capa con su tensor de pesos actual.
* **`torch.save(model.state_dict(), "weights.pt")`**:
  * Serializa y guarda los pesos entrenados en el fichero `weights.pt` dentro de la carpeta `runs/<run_id>/` para que puedan reutilizarse o inspeccionarse más tarde.
* **`model.load_state_dict(torch.load("weights.pt"))`**:
  * Restaura los pesos guardados en la estructura del modelo.

#### 💡 ¿Por qué es la pieza clave del Entrenamiento Multi-Etapa?
En proyectos reales (como entrenar un LLM o un modelo de visión), el aprendizaje se divide en fases separadas (**Preentrenamiento $\rightarrow$ Fine-Tuning $\rightarrow$ Refuerzo**):
1. **No se usa un único bucle gigante con `if epoch < 10`:** Aunque matemáticamente haría lo mismo, si el script falla en la época final se perderían semanas de cómputo.
2. **Desacoplamiento con Checkpoints:** Guardar el `state_dict()` permite que la Fase 1 (Preentrenamiento masivo) se guarde en `modelo_base.pt`, y luego cualquier investigador pueda lanzar 20 experimentos distintos de Fine-Tuning cargando esos pesos en 2 segundos, usando optimizadores e hiperparámetros diferentes sin re-entrenar la base.

---

### ⑦ Semillas y Reproducibilidad (`set_seed`)

Una red neuronal usa números aleatorios en múltiples momentos críticos:
1. **Pesos iniciales:** Al crear una capa (`nn.Linear`, `nn.Conv2d`), sus pesos $W$ y sesgos $b$ se rellenan con valores aleatorios.
2. **Barajado de datos:** `DataLoader(..., shuffle=True)` reordena los ejemplos aleatoriamente al inicio de cada época.
3. **Capas estocásticas:** `nn.Dropout` apaga un porcentaje aleatorio de neuronas en cada pase.

La **semilla (`seed`)** es el número inicial que fija el punto de partida de estos generadores pseudoaleatorios:

* **`torch.manual_seed(seed)`**:
  * Fija el generador aleatorio de PyTorch para operaciones en CPU.
* **`torch.cuda.manual_seed_all(seed)`**:
  * Fija el generador aleatorio para todas las GPUs disponibles.
* **`torch.backends.cudnn.deterministic = True`**:
  * Fuerza a los algoritmos de bajo nivel de NVIDIA (cuDNN) a usar implementaciones deterministas, evitando pequeñas variaciones numéricas por paralelismo.

> **¿Qué implica fijar la semilla?** Permite que si repites el entrenamiento hoy y mañana con `seed=42`, la red empiece con los mismos pesos y procese los datos en el mismo orden, garantizando **reproducibilidad exacta** y permitiendo aislar el efecto de cualquier cambio de hiperparámetros.

---

### ⑧ Relación entre la última capa de activación y la pérdida (Y cómo lo gestiona el arnés)

La última capa de la red neuronal refleja directamente el objetivo físico o matemático que se desea predecir. Por tanto, la función de activación de salida y la función de pérdida están fuertemente vinculadas.

#### ¿Cómo lo gestiona `harness.py`?
En el arnés, **el modelo no define la pérdida de forma rígida**. El arnés resuelve dinámicamente la función de pérdida basándose en la clave `"loss"` del diccionario de configuración (config), buscándola en `torch.nn.functional` mediante `getattr` (línea 178 de [`harness.py`](file:///Users/lopezmuzas/Developer/PhD/labs/dl-notebooks/lab/harness.py#L178)):
```python
loss_fn = getattr(torch.nn.functional, config.get("loss", DEFAULT_LOSS))
```
Por lo tanto, la arquitectura de tu red (registrada en `@models.register`) debe estar alineada con la función de pérdida que indiques en el config. Por ejemplo, en clasificación, la red debe escupir *logits* (sin activación final) porque la entropía cruzada de PyTorch aplica el Softmax internamente.

#### Tabla de correspondencias según el objetivo de la red

| Objetivo / Tipo de Tarea | Rango Deseado | Última Activación en el Modelo | Pérdida en Config (en `harness.py`) | Función interna de PyTorch |
|---|---|---|---|---|
| **Regresión libre** (ej: temperatura, seno) | $(-\infty, +\infty)$ | **Ninguna** (Lineal) | `"mse_loss"` o `"l1_loss"` | `F.mse_loss` / `F.l1_loss` |
| **Regresión acotada positiva** (ej: distancias, precios > 0) | $[0, +\infty)$ | **ReLU** o **Softplus** | `"mse_loss"` | `F.mse_loss` |
| **Regresión acotada en rango** (ej: porcentajes) | $[0, 1]$ o $[-1, 1]$ | **Sigmoid** o **Tanh** | `"mse_loss"` o `"binary_cross_entropy"` | `F.mse_loss` / `F.binary_cross_entropy` |
| **Clasificación Binaria** (ej: Spam / No Spam) | Probabilidad $0$ a $1$ | **Ninguna** (Lineal / Logits) | `"binary_cross_entropy_with_logits"` | `F.binary_cross_entropy_with_logits` |
| **Clasificación Multiclase** (ej: Gato / Perro / Pájaro) | Probabilidades que suman $1$ | **Ninguna** (Lineal / Logits) | `"cross_entropy"` | `F.cross_entropy` |

---

## 3. Resumen de funciones clave en una sola tabla

| Función en `harness.py` | ¿Para qué sirve? |
|---|---|
| `torch.device(dev)` | Define si usamos procesador (`cpu`) o tarjeta gráfica (`cuda`). |
| `model.to(device)` | Envía la red neuronal a la memoria de cálculo correspondiente. |
| `model.train()` | Activa el modo de entrenamiento en la red. |
| `model.eval()` | Activa el modo de inferencia/evaluación. |
| `optimizer.zero_grad()` | Borra los gradientes de la pasada anterior. |
| `loss.backward()` | Calcula las derivadas automáticas de cada parámetro. |
| `optimizer.step()` | Mueve los pesos en la dirección que reduce la pérdida. |
| `@torch.no_grad()` | Acelera la evaluación evitando guardar grafos de cálculo. |
| `loss.item()` | Convierte el tensor de pérdida en un simple número `float`. |
| `torch.save(...)` | Guarda los pesos de la red en el disco (`weights.pt`). |
