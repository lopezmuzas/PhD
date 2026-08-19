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

* **`nn.Module`**:
  * **Qué es:** Es la clase padre obligatoria de cualquier red neuronal en PyTorch.
  * **En el arnés:** Cuando registras un modelo (`@models.register("mlp")`), la función constructora devuelve un objeto que hereda de `nn.Module`.
* **`model.train()` y `model.eval()`**:
  * **Qué hacen:** Cambian el "modo" del modelo.
  * **Por qué importa:** En `train_one_epoch()` se activa `model.train()` para permitir el aprendizaje y comportamiento de capas como Dropout. En `evaluate()` se pone `model.eval()` para congelar esas capas y obtener predicciones deterministas.
* **`model.parameters()`**:
  * **Qué hace:** Devuelve un iterador con todos los tensores de pesos ($W$) y sesgos ($b$) que la red puede aprender. Se le pasa al optimizador para que sepa qué variables debe modificar.
* **`model(inputs)`**:
  * **Qué hace:** Ejecuta el pase hacia adelante (*forward pass*) para producir las predicciones a partir de los datos de entrada.

---

### ② Carga de Datos (`torch.utils.data.DataLoader`)

* **`DataLoader`**:
  * **Qué hace:** Envuelve un conjunto de datos y los agrupa en pequeños lotes (*batches* o minilotes).
  * **En el arnés:** En cada iteración del bucle `for inputs, targets in components.train_loader:`, el `DataLoader` entrega tensores con las entradas $X$ y las dianas esperadas $y$.

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
