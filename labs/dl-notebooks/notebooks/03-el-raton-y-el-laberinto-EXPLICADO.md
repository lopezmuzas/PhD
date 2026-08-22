# 📖 El ratón y el laberinto, línea a línea

**Guía de acompañamiento de** `03-el-raton-y-el-laberinto.ipynb`

---

## Cómo usar este documento

El notebook cuenta la **historia**. Este documento explica el **código**: cada línea, por
qué está escrita así, y qué habría pasado si la hubiéramos escrito de otra forma.

Ábrelos en dos ventanas, uno al lado del otro. Los apartados de aquí van numerados igual
que las celdas de código del notebook, en orden: **Celda 0**, **Celda 1**, etc.

Tres tipos de aviso que verás:

> 🐍 **Python.** Un truco o una construcción del lenguaje que conviene reconocer.

> 🔥 **PyTorch.** Algo específico de la librería de redes neuronales.

> ⚠️ **Trampa.** Un sitio donde es fácil equivocarse, o una decisión discutible.

---

## Índice

| Celda | Qué hace |
|---|---|
| [0](#celda-0--arranque) | Arranque: encontrar el proyecto e importar |
| [1](#celda-1--el-mapa-del-laberinto) | El mapa, los muros y las acciones |
| [2](#celda-2--lo-que-ve-el-ratón-one-hot) | Lo que ve el ratón (*one-hot*) |
| [3](#celda-3--las-reglas-del-mundo-el-entorno) | Las reglas del mundo: el entorno |
| [4](#celda-4--el-ratón-novato) | Un ratón sin entrenar |
| [5](#celda-5--contar-los-aciertos-por-casualidad) | Contar los aciertos por casualidad |
| [6](#celda-6--el-config-y-el-entrenamiento) | El `config` y el entrenamiento |
| [7](#celda-7--la-gráfica-del-aprendizaje) | La gráfica del aprendizaje |
| [8](#celda-8--la-chuleta-de-flechas) | La chuleta de flechas |
| [9](#celda-9--comprobar-la-chuleta-casilla-por-casilla) | Comprobar la chuleta |
| [10](#celda-10--el-dibujo-bueno-matplotlib) | El dibujo bueno (matplotlib) |
| [11](#celda-11--movemos-el-queso) | Movemos el queso |
| [12](#celda-12--dibujar-la-travesura) | Dibujar la travesura |
| [13](#celda-13--poner-el-gato) | Poner el gato |
| [14](#celda-14--reentrenar-con-gato) | Reentrenar con gato |
| [15](#celda-15--comparar-los-dos-lado-a-lado) | Comparar los dos |
| [16](#celda-16--cinco-ratones-cinco-semillas) | Cinco ratones |
| [17](#celda-17--los-cinco-dibujos) | Los cinco dibujos |
| [A](#apéndice-a--qué-pasa-dentro-de-las-funciones-del-arnés) | **Apéndice A:** dentro del arnés |
| [B](#apéndice-b--chuleta-de-python-y-pytorch) | **Apéndice B:** chuleta de Python/PyTorch |

---
---

# Celda 0 · Arranque

```python
# ── ARRANQUE ──
import os, sys
from pathlib import Path

while not (Path.cwd() / "lab").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lab import harness as H
from lab import harness_rl as rl

CPU = torch.device("cpu")
print("✅ Listo. Vamos a jugar.")
```

### El bucle raro del principio

```python
while not (Path.cwd() / "lab").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
```

Esto resuelve un problema muy tonto y muy real: **el notebook no sabe dónde está**.

Si lo abres desde `dl-notebooks/`, la carpeta `lab/` está justo al lado. Si lo abres desde
`dl-notebooks/notebooks/`, hay que subir un nivel. Y si lo abres desde otro sitio, quién
sabe.

Lo que hace el bucle, traducido: *"mientras no haya una carpeta llamada `lab` aquí, sube un
nivel"*.

* `Path.cwd()` = *current working directory*, la carpeta en la que estamos.
* `Path.cwd() / "lab"` — el operador `/` en `pathlib` **une rutas**. No divide. Es azúcar
  sintáctico para `os.path.join(...)`, y se lee mucho mejor.
* `.exists()` — ¿existe esa carpeta?
* `os.chdir("..")` — subir un nivel.

La segunda condición es el **freno de emergencia**:

```python
Path.cwd() != Path.cwd().parent
```

En la raíz del disco (`/`), el padre de `/` es `/` otra vez. Sin esta condición, si nunca
encontrara la carpeta `lab`, el bucle subiría para siempre. Con ella, al llegar arriba del
todo se para.

> ⚠️ **Trampa.** Este bucle **cambia el directorio de trabajo del proceso**. Si ejecutas la
> celda dos veces no pasa nada (ya está en el sitio bueno, el `while` no entra). Pero si
> tienes otro código que abre ficheros con rutas relativas, ojo.

### El camino de búsqueda de módulos

```python
sys.path.insert(0, str(Path.cwd()))
```

`sys.path` es la **lista de carpetas donde Python busca cuando escribes `import`**. Es una
lista normal, y aquí le metemos la carpeta actual **en la posición 0** — la primera, la de
mayor prioridad.

Sin esta línea, `from lab import harness` fallaría con `ModuleNotFoundError`: Python no
tiene ninguna razón para mirar en nuestra carpeta.

* ¿Por qué `insert(0, ...)` y no `append(...)`? Para que nuestra versión **gane** si
  hubiera otro paquete instalado que también se llamara `lab`.
* `str(...)` porque `sys.path` quiere cadenas de texto, no objetos `Path`.

### Los imports

```python
from lab import harness as H
from lab import harness_rl as rl
```

`as H` y `as rl` son **alias**. Escribir `H.set_seed(0)` en vez de
`harness.set_seed(0)` cincuenta veces. Es la misma costumbre que `import numpy as np`.

Los dos son ficheros del proyecto:

* `lab/harness.py` — el arnés de experimentos normal (datos → modelo → pérdida).
* `lab/harness_rl.py` — el arnés de refuerzo. Importa el anterior y reutiliza casi todo.

```python
from matplotlib.patches import Rectangle
```

Solo necesitamos una clase de `matplotlib.patches` (para pintar los muros como cuadrados
grises), así que la importamos directamente en vez de traer el módulo entero.

### El dispositivo

```python
CPU = torch.device("cpu")
```

> 🔥 **PyTorch.** Un `device` le dice a PyTorch **dónde vivir**: en el procesador (`"cpu"`)
> o en la tarjeta gráfica (`"cuda"`, `"mps"` en Mac). Aquí forzamos CPU a propósito: el
> laberinto es tan pequeño que la GPU no ayudaría — mover datos a la tarjeta y traerlos de
> vuelta costaría **más** que hacer la cuenta.

Lo guardamos en una variable en mayúsculas por convención: en Python, `MAYÚSCULAS` significa
*"esto es una constante, no lo cambies"*. Python no lo impide, es un acuerdo entre
programadores.

---
---

# Celda 1 · El mapa del laberinto

```python
MAPA = [
    "R....",
    ".###.",
    ".....",
    ".###.",
    "....Q",
]

FILAS, COLS = len(MAPA), len(MAPA[0])
N_CASILLAS  = FILAS * COLS

MUROS  = {(r, c) for r, fila in enumerate(MAPA)
                 for c, ch in enumerate(fila) if ch == "#"}
INICIO = (0, 0)
QUESO  = (4, 4)
```

### El mapa como texto

`MAPA` es simplemente **una lista de cinco cadenas de cinco caracteres**. Nada de arrays ni
matrices: texto que se puede leer con los ojos.

Esa es toda la ventaja. Si quieres cambiar el laberinto, editas los dibujitos. Compara con
la alternativa:

```python
# Lo mismo, escrito de forma ilegible:
MUROS = {(1,1), (1,2), (1,3), (3,1), (3,2), (3,3)}
```

Funciona igual, pero suerte encontrando el error si te equivocas en una coordenada.

### Contar filas y columnas

```python
FILAS, COLS = len(MAPA), len(MAPA[0])
```

> 🐍 **Python.** Esto es **desempaquetado de tuplas**: a la derecha se construye la tupla
> `(5, 5)` y a la izquierda se reparte, una a cada variable. Equivale a dos líneas, pero se
> lee de golpe.

* `len(MAPA)` = 5 → cuántas cadenas hay → cuántas **filas**.
* `len(MAPA[0])` = 5 → cuántos caracteres tiene **la primera** cadena → cuántas **columnas**.

> ⚠️ **Trampa.** Solo mira la primera fila. Si te dejaras una fila con cuatro caracteres en
> vez de cinco, el código no se quejaría aquí y explotaría mucho más tarde, en un sitio que
> no tiene nada que ver. Para un laberinto de juguete es aceptable; en código serio pondrías
> un `assert all(len(f) == COLS for f in MAPA)`.

### Encontrar los muros de una sola vez

```python
MUROS = {(r, c) for r, fila in enumerate(MAPA)
                for c, ch in enumerate(fila) if ch == "#"}
```

Esto es una **comprensión de conjunto** (*set comprehension*), y hace de golpe lo que en un
lenguaje sin este truco serían seis líneas:

```python
# Exactamente lo mismo, escrito a mano:
MUROS = set()
for r in range(len(MAPA)):        # r = número de fila
    fila = MAPA[r]
    for c in range(len(fila)):    # c = número de columna
        if fila[c] == "#":
            MUROS.add((r, c))
```

Pieza por pieza:

* `enumerate(MAPA)` produce pares `(0, "R....")`, `(1, ".###.")`, ... Es decir: **el índice
  y el valor a la vez**. Sin `enumerate` tendrías que llevar un contador a mano.
* `for r, fila in ...` — desempaqueta cada par en dos variables.
* El segundo `for` recorre los caracteres de esa fila, otra vez con su índice.
* `if ch == "#"` — el filtro: solo nos quedamos con los muros.
* `(r, c)` — lo que guardamos: una tupla con la coordenada.

> 🐍 **Python.** El orden de los `for` es **el mismo que si los escribieras anidados**: el
> primero es el de fuera. Es la única regla que hay que recordar y la que más se confunde.

> 🐍 **¿Por qué llaves `{}` y no corchetes `[]`?** Con `{}` y sin dos puntos sale un
> **conjunto** (`set`). Y eso es una decisión de rendimiento, no de estilo:
>
> | | `lista` | `conjunto` |
> |---|---|---|
> | ¿Cuánto tarda `x in coleccion`? | Recorre todo: **lento** | Directo: **instantáneo** |
> | ¿Permite repetidos? | Sí | No |
> | ¿Mantiene el orden? | Sí | No (nos da igual) |
>
> Y esa comprobación (`destino not in MUROS`) se hace **en cada paso de cada partida**.
> Echa la cuenta: 120 rondas × 48 partidas × hasta 40 pasos ≈ **230.000 comprobaciones**.
> Con una lista de 6 elementos tampoco se notaría, pero la costumbre es la correcta y en un
> laberinto de 100×100 sería la diferencia entre segundos y minutos.

### Las cuatro acciones

```python
ACCIONES = {0: (-1, 0),    # arriba
            1: (+1, 0),    # abajo
            2: (0, -1),    # izquierda
            3: (0, +1)}    # derecha
FLECHAS  = {0: "↑", 1: "↓", 2: "←", 3: "→"}
```

Un diccionario que traduce **un número** (lo que la red neuronal sabe producir) a **un
desplazamiento** (lo que el mundo entiende).

Cada valor es una tupla `(cuánto cambia la fila, cuánto cambia la columna)`.

> ⚠️ **Trampa importante: "arriba" es −1.** La fila 0 está **arriba**, como en una tabla o
> una matriz. Así que subir significa **restar** a la fila. Es al revés que en las
> matemáticas del colegio, donde la *y* crece hacia arriba, y es la causa del 90% de los
> errores de signo al programar cuadrículas.

La red neuronal, por cierto, no sabe nada de esto. Escupe cuatro números y nosotros
decidimos que el número 3 significa "derecha". Podríamos haberlos ordenado al revés y
aprendería exactamente igual de bien: **el significado de las acciones lo pone el entorno,
no el modelo.**

### La función que lo pinta

```python
def dibujar_texto(raton=INICIO, queso=QUESO, gato=None):
    lineas = []
    for r in range(FILAS):
        fila = []
        for c in range(COLS):
            if   (r, c) in MUROS:        fila.append("⬛")
            elif (r, c) == queso:        fila.append("🧀")
            elif gato and (r, c) == gato: fila.append("🐱")
            elif (r, c) == raton:        fila.append("🐭")
            else:                        fila.append("··")
        lineas.append(" ".join(fila))
    return "\n".join(lineas)
```

Recorre las 25 casillas y decide qué emoji poner en cada una.

* **`raton=INICIO, queso=QUESO, gato=None`** son **valores por defecto**: si llamas a
  `dibujar_texto()` sin argumentos, usa esos. Si llamas a `dibujar_texto(gato=(2,4))`,
  cambia solo el gato. Por eso más adelante podemos reutilizar la misma función con y sin
  gato.
* **El orden de los `if` es una prioridad.** Si una casilla es a la vez muro y queso, gana
  el muro, porque se pregunta primero. Aquí no pasa, pero es bueno saber que la cadena
  `if/elif` **se para en el primero que acierta**.
* **`if gato and (r, c) == gato`** — aquí hay una sutileza. `gato` vale `None` por defecto,
  y en Python `None` es **falso** en un contexto booleano. Así que si no hay gato, la
  condición se corta en el primer `and` y ni se molesta en comparar. Se llama
  **cortocircuito**, y evita comparar contra `None`.
* **`" ".join(fila)`** — pega los elementos de la lista poniendo un espacio entre medias.
  Hay que hacerlo así (y no `fila + " "` dentro del bucle) porque `join` es muchísimo más
  rápido y no deja un espacio suelto al final.
* **`"\n".join(lineas)`** — lo mismo, pero pegando las filas con saltos de línea. El
  resultado es **una sola cadena** con el laberinto entero dentro, lista para `print`.

> 🐍 **Python.** ¿Por qué `lineas = []` y luego `.append()`, en vez de ir sumando cadenas?
> Porque las cadenas en Python son **inmutables**: cada `cadena + "x"` crea una cadena nueva
> y copia todo. Con listas + `join` se construye una vez. Con 25 casillas da igual; la
> costumbre es la que importa.

### La última línea

```python
print(f"Casillas libres: {N_CASILLAS - len(MUROS)} de {N_CASILLAS}")
```

> 🐍 **Python.** Una **f-string**: el prefijo `f` antes de la comilla permite meter código
> Python dentro de `{}` y que se evalúe. `25 - 6 = 19` casillas libres.

---
---

# Celda 2 · Lo que ve el ratón (*one-hot*)

```python
def donde_estoy(posicion):
    '''Los 25 números que ve el ratón: todo ceros menos su casilla.'''
    v = torch.zeros(N_CASILLAS)
    v[posicion[0] * COLS + posicion[1]] = 1.0
    return v
```

Tres líneas, y son de las más importantes del notebook: **aquí se decide qué información
recibe el ratón**, y por tanto qué es capaz de aprender.

### `torch.zeros(25)`

> 🔥 **PyTorch.** Crea un **tensor** de 25 ceros. Un tensor es como una lista de números,
> pero preparada para que PyTorch haga cuentas rápidas con él y para poder derivar. Por
> defecto son `float32` (números decimales de 32 bits), que es lo que las redes esperan.

### La fórmula del índice

```python
v[posicion[0] * COLS + posicion[1]] = 1.0
```

Esta línea convierte **una coordenada de dos números en una posición de una lista**. Es la
fórmula de *aplanar* una cuadrícula, y aparece constantemente en programación:

```none
índice = fila × (cuántas columnas hay) + columna
```

Con nuestro laberinto de 5 columnas:

| Casilla (fila, col) | Cuenta | Índice |
|---|---|---|
| (0, 0) — la salida | 0 × 5 + 0 | **0** |
| (0, 4) | 0 × 5 + 4 | **4** |
| (1, 0) | 1 × 5 + 0 | **5** |
| (4, 3) | 4 × 5 + 3 | **23** |
| (4, 4) — el queso | 4 × 5 + 4 | **24** |

Y eso es exactamente lo que sale al ejecutar la celda: para `(4, 3)` el `1` aparece en la
posición 23 de los 25 números. Puedes contarlo en la salida.

La intuición: es como numerar las casillas de un tablero de ajedrez del 0 al 63 leyendo de
izquierda a derecha y de arriba abajo.

### ¿Por qué 25 números y no 2?

Esta es **la** pregunta de la celda. Podríamos haberle pasado al ratón simplemente
`(fila, columna)`, dos números, y ahorrarnos 23. ¿Por qué no?

Se llama codificación ***one-hot*** ("uno caliente": todo apagado menos un bit encendido) y
la diferencia es profunda:

| | `(fila, columna)` — 2 números | *One-hot* — 25 números |
|---|---|---|
| ¿Puede aprender una acción distinta en cada casilla? | Con esfuerzo | **Sí, trivialmente** |
| ¿Cree que las casillas vecinas se parecen? | **Sí** (3 está cerca de 4) | No, todas son ajenas |
| ¿Cuántas entradas para un laberinto 100×100? | 2 | **10.000** 😬 |
| ¿Puede aprovechar lo aprendido en una casilla para otra? | Sí | **No** |

Con `(fila, columna)`, la red tiene que aprender una **función suave**: algo como "si la
fila es alta y la columna es alta, ve a la derecha". Pero un laberinto no es suave — dos
casillas pegadas pueden requerir acciones opuestas si hay un muro entre ellas. Forzar a la
red a inventarse una curva que pase por todos esos saltos es pedirle un trabajo innecesario.

Con *one-hot*, cada casilla activa **su propio grupo de pesos**, independiente del resto. La
red se convierte prácticamente en una **tabla**: para cada casilla, cuatro números. Es lo
más fácil de aprender que hay.

> ⚠️ **El precio.** Justo por eso el ratón **no puede generalizar**. Lo que aprende sobre la
> casilla (2,2) no le sirve absolutamente de nada para la (2,3): son entradas distintas con
> pesos distintos. Y ese es, literalmente, el motivo de que en la sección 6 del notebook el
> ratón se estampe contra la esquina vacía cuando movemos el queso.
>
> No es un bug que se pueda arreglar con más entrenamiento. **Es una consecuencia directa de
> esta línea de código.** Si quisieras un ratón que generalizara, tendrías que cambiar
> `donde_estoy` para que le diera información *relativa* — por ejemplo "el queso está 2
> casillas a tu derecha y 3 abajo" — y entrenarlo en muchos laberintos distintos.

Ahí tienes la lección más útil del notebook entero: **lo que un modelo puede aprender está
limitado por lo que le dejas ver**, y eso lo decides tú en tres líneas de código, no el
algoritmo.

---
---

# Celda 3 · Las reglas del mundo: el entorno

```python
@rl.envs.register("laberinto")
def construir_laberinto(gato=None, queso=QUESO, coste_paso=0.05, **kwargs):
    '''El mundo del ratón. Tres métodos: empezar, dar un paso, y ya está.'''

    class Laberinto:
        obs_dim   = N_CASILLAS
        n_actions = 4

        def reset(self):
            self.pos = INICIO
            self.camino = [INICIO]
            return donde_estoy(self.pos)

        def step(self, accion):
            df, dc = ACCIONES[accion]
            destino = (self.pos[0] + df, self.pos[1] + dc)

            dentro = 0 <= destino[0] < FILAS and 0 <= destino[1] < COLS
            if dentro and destino not in MUROS:
                self.pos = destino

            self.camino.append(self.pos)

            if self.pos == queso:
                return donde_estoy(self.pos), 1.0 - coste_paso, True
            if gato and self.pos == gato:
                return donde_estoy(self.pos), -1.0, True
            return donde_estoy(self.pos), -coste_paso, False

    return Laberinto()
```

La celda más densa del notebook. Vamos por capas.

### La primera línea: `@rl.envs.register("laberinto")`

> 🐍 **Python.** Eso de arriba es un **decorador**. Un decorador es una función que recibe
> la función escrita justo debajo y hace algo con ella.

Traducido a español: *"apunta esta función en la libreta de entornos, bajo el nombre
`laberinto`"*.

Es equivalente a escribir esto:

```python
def construir_laberinto(...):
    ...

rl.envs["laberinto"] = construir_laberinto     # <- lo que hace el decorador
```

Y por dentro (`lab/harness.py`) el registro es literalmente un diccionario con dos métodos:

```python
class Registry(dict):
    def register(self, name):
        def decorator(builder):
            self[name] = builder      # guarda la función
            return builder            # y la devuelve sin tocarla
        return decorator
    def build(self, name, **kwargs):
        return self[name](**kwargs)   # la busca y la llama
```

**¿Y para qué tanto lío?** Para que el `config` de la celda 6 pueda ser un diccionario de
texto plano:

```python
config = {"env": "laberinto", ...}   # ← una CADENA, no una clase
```

Y eso importa porque un diccionario de cadenas y números **se puede guardar en disco**
(`runs/<id>/config.json`). Si el config guardara la clase `Laberinto` directamente, no
habría forma de escribirlo en un fichero de texto ni de reproducir el experimento seis meses
después. Todo el sistema de trazabilidad del arnés depende de este detalle.

### El truco de la fábrica: una clase dentro de una función

Fíjate en la estructura: `construir_laberinto` es una **función** que define una **clase**
dentro y devuelve **un objeto** de esa clase. Se llama *función fábrica*.

¿Por qué no definir la clase directamente, al nivel del fichero? Por esto:

```python
def construir_laberinto(gato=None, queso=QUESO, coste_paso=0.05, **kwargs):
    class Laberinto:
        def step(self, accion):
            if self.pos == queso:        # ← usa 'queso', que es un parámetro de la función
                ...
```

Los métodos de la clase usan `queso`, `gato` y `coste_paso`, **que son parámetros de la
función de fuera**. Eso funciona porque Python crea una **clausura** (*closure*): la clase
"se lleva puestas" las variables del entorno donde nació.

La alternativa sería guardarlas como atributos (`self.queso = queso`, `self.gato = gato`,
`self.coste = coste_paso`) y escribir `self.queso` en todas partes. Funciona igual y es más
explícito; esto es más corto. Las dos son defendibles.

**La ventaja práctica** es que cada llamada crea un mundo independiente con sus propias
reglas:

```python
env_a = rl.envs.build("laberinto")                       # sin gato
env_b = rl.envs.build("laberinto", gato=(2, 4))          # con gato
env_c = rl.envs.build("laberinto", queso=(4, 0))         # queso movido
```

Tres mundos distintos, a la vez, sin pisarse. Es justo lo que hacen las celdas 11 y 14.

### `**kwargs`: el cajón de sastre

```python
def construir_laberinto(gato=None, queso=QUESO, coste_paso=0.05, **kwargs):
```

> 🐍 **Python.** `**kwargs` recoge **cualquier argumento con nombre que no encaje en los
> anteriores** y lo mete en un diccionario llamado `kwargs`. Aquí no lo usamos para nada: es
> un **amortiguador**.

El arnés construye el entorno así:

```python
env = envs.build(config["env"], **config.get("env_args", {}))
```

Si algún día el arnés decidiera pasar un argumento extra que a este entorno no le interesa,
sin `**kwargs` la llamada explotaría con `TypeError: unexpected keyword argument`. Con
`**kwargs`, se lo traga y sigue. Es una convención de todo el arnés: **todos** los
constructores registrados lo llevan.

### `obs_dim` y `n_actions`: el contrato

```python
class Laberinto:
    obs_dim   = N_CASILLAS   # 25
    n_actions = 4
```

Estos dos no son decoración: son **el contrato** que el arnés lee para construir la red
neuronal del tamaño correcto. En `lab/harness_rl.py`:

```python
env = envs.build(config["env"], **config.get("env_args", {}))
policy = H.models.build(config["model"],
                        obs_dim=env.obs_dim,        # ← los lee de aquí
                        n_actions=env.n_actions,
                        **config.get("model_args", {}))
```

O sea: **primero se construye el mundo, y el mundo dicta el tamaño del cerebro.** No al
revés. Si cambiaras el laberinto a 8×8, `N_CASILLAS` pasaría a 64 y la red se construiría con
64 entradas automáticamente, sin que tú toques nada más.

> 🐍 **Python.** Están escritos **dentro de la clase pero fuera de los métodos**, así que
> son **atributos de clase**: compartidos por todos los objetos, no propios de cada uno.
> Aquí es lo correcto, porque el tamaño del laberinto no cambia entre partidas.

### `reset()`: empezar una partida

```python
def reset(self):
    self.pos = INICIO
    self.camino = [INICIO]
    return donde_estoy(self.pos)
```

* `self.pos = INICIO` — el ratón vuelve a la esquina de salida.
* `self.camino = [INICIO]` — una lista vacía... bueno, con la posición inicial dentro. Esto
  **no forma parte del contrato de refuerzo**: es un extra nuestro para poder dibujar por
  dónde pasó. El arnés no lo usa ni lo conoce.
* `return donde_estoy(self.pos)` — devuelve lo que el ratón ve. Quien llama a `reset()`
  necesita saber en qué estado empieza.

> ⚠️ Fíjate en que `self.pos` y `self.camino` **se crean aquí**, no en un `__init__`. Eso
> significa que si alguien llamara a `step()` antes de `reset()`, el programa fallaría con
> `AttributeError`. Es aceptable porque `collect_episode` **siempre** llama a `reset()`
> primero — pero es una suposición no escrita, y en código de producción pondrías un
> `__init__` que las inicialice.

### `step()`: el corazón del mundo

Esta función es el mundo entero. Recibe **una acción** y devuelve **tres cosas**.

```python
df, dc = ACCIONES[accion]
destino = (self.pos[0] + df, self.pos[1] + dc)
```

* `ACCIONES[accion]` busca en el diccionario. Si `accion` es `3`, devuelve `(0, +1)`.
* `df, dc = ...` desempaqueta en *delta fila* y *delta columna*.
* `destino` es **a dónde quiere ir**. Ojo: querer no es poder. Todavía no se ha movido.

```python
dentro = 0 <= destino[0] < FILAS and 0 <= destino[1] < COLS
```

> 🐍 **Python.** `0 <= x < FILAS` es una **comparación en cadena**, y es una de las cosas
> más bonitas del lenguaje. Significa `0 <= x and x < FILAS`, se lee como en matemáticas, y
> además evalúa `x` una sola vez. En casi cualquier otro lenguaje tendrías que escribir la
> versión larga.

Aquí comprobamos que no se sale del tablero: ni fila negativa, ni fila 5, ni columna
negativa, ni columna 5.

```python
if dentro and destino not in MUROS:
    self.pos = destino
# si no, se queda donde estaba
```

**La única línea que mueve al ratón.** Y la clave está en lo que *no* hay: no hay `else`.

Si el destino está fuera del tablero o es un muro, simplemente **no se hace nada**. El ratón
se queda donde estaba, y ha gastado su turno. Eso es "chocarse".

> 🔑 Y esto es lo que explica esa salida de la celda 4 con posiciones repetidas
> (`(0,0) → (0,0) → (0,0)`): cada repetición es un choque. El ratón **no sabe que el muro
> está ahí** hasta que lo intenta, porque su única información son los 25 números de
> `donde_estoy`, que no dicen nada sobre muros.

```python
self.camino.append(self.pos)
```

Se apunta la posición **siempre**, incluso si no se movió. Por eso se pueden contar los
choques después.

### Las tres salidas, y por qué el orden importa

```python
if self.pos == queso:
    return donde_estoy(self.pos), 1.0 - coste_paso, True    # ¡PREMIO!
if gato and self.pos == gato:
    return donde_estoy(self.pos), -1.0, True                # ¡ay!
return donde_estoy(self.pos), -coste_paso, False            # paso normal
```

Las tres devuelven una **tupla de tres elementos**, que es el contrato que espera
`collect_episode`:

```none
( qué ve ahora ,  qué ha ganado ,  ¿se acabó la partida? )
      tensor          float              bool
```

**El orden de los `if` es una decisión.** Se pregunta primero por el queso, así que si
alguien pusiera el gato encima del queso, ganaría el queso. Con otro orden, el mismo
laberinto se comportaría distinto. No es un detalle: es una regla del juego escondida en el
orden de dos líneas.

> 🐍 Fíjate también en que no hay `elif` ni `else`: se usa **`return` temprano**. Como cada
> rama devuelve, no hace falta encadenar condiciones. Se lee mejor y se anida menos.

### La cuenta del `1.0 - coste_paso`

¿Por qué llegar al queso da `0.95` y no `1.0`?

Porque **llegar también es dar un paso**, y todos los pasos cuestan. El premio es `1.0`,
menos los `0.05` del paso que te llevó allí.

Y ahora podemos verificar el número que sale en el notebook. El camino óptimo son 8 pasos:

```none
pasos 1 al 7  →  7 × (−0,05)  =  −0,35     (pasos normales)
paso 8        →  1,00 − 0,05  =  +0,95     (llega al queso)
                                 ────────
                        TOTAL  =  +0,60
```

**+0,60.** Que es exactamente la nota que el ratón alcanza en la celda 6 y la línea verde de
la gráfica. La cuenta cierra.

> 💡 Y si `coste_paso` fuera 0, todos los caminos valdrían 1,00 y al ratón **le daría igual**
> tardar 8 pasos o 38. Ese `0.05` es lo único que le hace tener prisa. Pruébalo: pon
> `env_args={"coste_paso": 0.0}` y mira cuántos pasos da.

---
---

# Celda 4 · El ratón novato

```python
H.set_seed(0)
raton_novato = H.models.build("mlp_policy", obs_dim=N_CASILLAS,
                              n_actions=4, hidden=64)

env = rl.envs.build("laberinto")
rl.collect_episode(raton_novato, env, CPU, max_steps=40)
```

### `H.set_seed(0)`

Fija el azar. Por dentro (`lab/harness.py`):

```python
def set_seed(seed, deterministic=True):
    random.seed(seed)              # el azar de Python
    np.random.seed(seed)           # el de numpy
    torch.manual_seed(seed)        # el de PyTorch (pesos iniciales, muestreo)
    torch.cuda.manual_seed_all(seed)
    ...
```

Hay **tres generadores de números aleatorios distintos** en juego y hay que sembrarlos
todos, porque no se hablan entre ellos. Si te olvidas de uno, tu experimento parece
reproducible hasta que un día deja de serlo.

Consecuencia práctica: **ejecutar esta celda dos veces da exactamente el mismo ratón novato y
exactamente el mismo paseo.** Eso es lo que hace que las cifras del notebook sean las mismas
para ti que para mí.

### Construir el cerebro

```python
raton_novato = H.models.build("mlp_policy", obs_dim=25, n_actions=4, hidden=64)
```

`H.models.build` busca `"mlp_policy"` en el registro de modelos y lo llama. Está definido en
`lab/harness_rl.py`:

```python
@H.models.register("mlp_policy")
def build_mlp_policy(obs_dim, n_actions, hidden=32, **kwargs):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, n_actions),
    )
```

O sea, el "cerebro" del ratón es esto:

```none
 25 números          64 números         64 números          4 números
(dónde estoy)   →   (capa oculta)  →   (aplastados)   →   (una nota por acción)
              Linear             Tanh              Linear
```

* **`nn.Linear(25, 64)`** — multiplica por una matriz de 25×64 y suma 64 números. Son
  `25×64 + 64 = 1.664` parámetros ajustables.
* **`nn.Tanh()`** — aplasta cada número al rango [−1, +1]. Sin algo así, dos capas lineales
  seguidas equivaldrían a una sola y la red no podría aprender nada curvo.
* **`nn.Linear(64, 4)`** — `64×4 + 4 = 260` parámetros. Salen **cuatro números**, uno por
  acción.
* **Total: 1.924 números** que el entrenamiento va a ajustar. Ese es "el cerebro del ratón".

> 🔥 **PyTorch: no hay Softmax al final.** La red escupe cuatro números crudos que pueden ser
> cualquier cosa (`[2.1, -0.4, 0.8, 5.3]`). Se llaman **logits**. Convertirlos en
> probabilidades es trabajo de quien los usa:
>
> ```python
> Categorical(logits=policy(estado))   # ← esto aplica el Softmax por dentro
> ```
>
> Se hace así por **estabilidad numérica**: calcular el logaritmo de un Softmax en un solo
> paso evita desbordamientos que aparecerían haciéndolo en dos. Es la misma regla que en
> clasificación, donde `F.cross_entropy` espera logits y no probabilidades.

> ⚠️ **`nn.Tanh()` y no `nn.ReLU()`.** En refuerzo se usa más `Tanh`, y no es superstición:
> las entradas no pasan por un `DataLoader` que las normalice, así que pueden venir con
> cualquier escala, y `Tanh` las acota. Está anotado en `harness_rl.py`.

### Soltar al ratón

```python
env = rl.envs.build("laberinto")
rl.collect_episode(raton_novato, env, CPU, max_steps=40)
```

`collect_episode` es **la función más importante del arnés de refuerzo**, porque es la que
*fabrica los datos*. Simplificada:

```python
def collect_episode(policy, env, device, max_steps=100, greedy=False):
    trajectory = Trajectory()
    state = env.reset().to(device)                     # ① empezar

    for _ in range(max_steps):                         # ② bucle de pasos
        distribution = Categorical(logits=policy(state))
        action = distribution.probs.argmax() if greedy else distribution.sample()

        trajectory.log_probs.append(distribution.log_prob(action))
        trajectory.entropies.append(distribution.entropy())

        state, reward, done = env.step(int(action.item()))
        state = state.to(device)
        trajectory.rewards.append(reward)
        if done:                                       # ③ salir si acabó
            break

    return trajectory
```

Lo que hace en cada paso: mira el estado → la red da cuatro notas → se **sortea** una acción
según esas notas → el entorno responde → se apunta todo.

> 🔑 **Aquí está la diferencia entera con el aprendizaje normal.** En `harness.py` los datos
> se cargan una vez de un fichero. Aquí **el modelo es un ingrediente de la generación de
> datos**: cambia el modelo, cambian los datos. Por eso RL no cabe en el arnés supervisado.

Fíjate en que **no guardamos el valor que devuelve**. Nos da igual la trayectoria: lo que
queremos son `env.camino` y `env.pos`, que el propio entorno se ha ido apuntando. Es un
atajo cómodo para inspeccionar.

> ⚠️ **Detalle real:** `Trajectory` guarda `log_probs`, `rewards` y `entropies`, pero **no
> guarda las acciones**. Es una limitación del arnés mínimo. Por eso hay que leer el camino
> del entorno y no de la trayectoria.

### Las cuentas de la salida

```python
print(f"El ratón dio {len(env.camino) - 1} pasos.")
```

**Por qué el `- 1`:** `camino` empieza conteniendo la posición inicial. Con 8 pasos hay 9
posiciones apuntadas. Es el clásico problema de los postes de una valla.

```python
print(f"¿Encontró el queso? {'SÍ 🧀' if env.pos == QUESO else 'NO 😐'}")
```

> 🐍 **Python.** Dentro de las llaves de una f-string cabe una **expresión condicional**
> (`A if condicion else B`). Es el "operador ternario" de Python, y se lee de izquierda a
> derecha como una frase: *"SÍ, si llegó, y si no NO"*.

```python
print("  " + " → ".join(str(p) for p in env.camino[:14]) + " → ...")
```

* `env.camino[:14]` — **rodaja**: los 14 primeros elementos. Si hay menos, no falla, coge
  los que haya.
* `str(p) for p in ...` — una **expresión generadora**. Hace falta porque `join` solo sabe
  pegar cadenas, y los elementos son tuplas.
* Sin corchetes alrededor: no se construye una lista intermedia, se van generando de uno en
  uno. Con 14 elementos es irrelevante; con un millón, no.

### El contador de choques (mi idiom favorito de la celda)

```python
repetidas = sum(1 for a, b in zip(env.camino, env.camino[1:]) if a == b)
```

Esta línea cuenta **cuántas veces el ratón se quedó en el sitio**. Merece desmontarse.

`zip(lista, lista[1:])` es el truco para recorrer **parejas de vecinos**:

```none
camino       = [ A ,  B ,  C ,  C ,  D ]
camino[1:]   = [ B ,  C ,  C ,  D ]
                 ↓    ↓    ↓    ↓
zip(...)     = (A,B) (B,C) (C,C) (C,D)
                            ↑
                     ¡aquí no se movió!
```

* `zip` empareja elemento a elemento y **se para en la lista más corta**, así que no hay que
  preocuparse por el final.
* `if a == b` — la posición no cambió → se chocó.
* `sum(1 for ...)` — suma un 1 por cada coincidencia. Es la forma idiomática de contar en
  Python (también valdría `sum(a == b for a, b in ...)`, porque `True` vale 1).

---
---

# Celda 5 · Contar los aciertos por casualidad

```python
exitos = 0
INTENTOS = 200
for _ in range(INTENTOS):
    rl.collect_episode(raton_novato, env, CPU, max_steps=40)
    if env.pos == QUESO:
        exitos += 1
```

Celda cortita y conceptualmente la más importante del notebook.

* **`for _ in range(200)`** — el `_` es una convención: *"necesito repetir 200 veces pero no
  me importa el número de vuelta"*. Es una variable normal, pero llamarla `_` avisa al lector
  de que no se usa.
* **Reutilizamos el mismo `env`.** ¿No hay que reiniciarlo? No: `collect_episode` llama a
  `env.reset()` por dentro, en su primera línea. Cada llamada es una partida limpia.
* **Y sobre todo: aquí NO hay entrenamiento.** No hay optimizador, no hay `backward()`, no
  hay ajuste de pesos. Es **el mismo ratón novato** 200 veces. Lo único que hacemos es
  *contar*.

El resultado (14%) es el número que hace posible todo lo demás:

> Si ese porcentaje fuera **0**, no habría ninguna partida buena que reforzar y el ratón
> **nunca** aprendería nada. Ni con un millón de rondas.

Eso tiene nombre: **el problema de la exploración**. Y explica por qué el refuerzo funciona
en un laberinto 5×5 y sufre horrores en un videojuego donde hay que encadenar 300 acciones
correctas antes de que aparezca el primer premio. En esos casos el algoritmo no es que
aprenda despacio: es que no tiene **nada** de lo que aprender.

> 🧪 **Experimento que puedes hacer:** cambia `max_steps=40` a `max_steps=10` y vuelve a
> contar. Con 10 pasos es imposible llegar (el mínimo son 8, y andando al azar no vas
> derecho), así que el porcentaje caerá casi a cero. Luego entrena con `max_steps=10` y verás
> que el ratón no aprende. No es un bug del algoritmo: le has quitado la materia prima.

---
---

# Celda 6 · El `config` y el entrenamiento

```python
config = {
    "name": "raton_laberinto",
    "env": "laberinto",
    "model": "mlp_policy",
    "model_args": {"hidden": 64},
    "optimizer": "adam",
    "optimizer_args": {"lr": 0.05},

    "iterations": 120,
    "episodes_per_iteration": 48,
    "max_steps": 40,

    "gamma": 0.99,
    "advantage": "normalized",
    "entropy_coef": 0.01,
    "grad_clip": 1.0,
    "eval_episodes": 10,
    "seed": 0,
}

resultado = rl.run_rl_experiment(config)
```

### Por qué un diccionario y no argumentos

Podríamos haber escrito `rl.run_rl_experiment(env="laberinto", lr=0.05, iterations=120, ...)`.
El diccionario se elige por una razón concreta: **se puede guardar en disco**.

Al terminar, el arnés escribe:

```none
runs/raton_laberinto_s0_20260822-110536/
├── config.json     ← este diccionario, tal cual
├── metrics.csv     ← una fila por ronda
├── weights.pt      ← el cerebro entrenado
└── meta.json       ← fecha, commit de git, versión de torch, máquina
```

Dentro de seis meses podrás abrir `config.json`, ver exactamente con qué parámetros se
entrenó, y repetirlo. Con argumentos de función eso se pierde en cuanto cierras el notebook.
Ese es todo el motivo, y es la idea central del arnés.

### Las claves, una por una

| Clave | Qué controla | Por qué este valor |
|---|---|---|
| `name` | Prefijo de la carpeta en `runs/` | Para reconocerlo luego |
| `env` | Qué mundo construir | La cadena que registramos en la celda 3 |
| `model` | Qué cerebro construir | `mlp_policy`, de `harness_rl.py` |
| `model_args` | Extras para el cerebro | `hidden: 64` → la capa oculta. `obs_dim` y `n_actions` **no** van aquí: los pone el entorno |
| `optimizer` | Cómo ajustar los pesos | Adam, el estándar |
| `optimizer_args` | `lr` = tasa de aprendizaje | **0.05** es altísimo para visión, normal para un RL de juguete |
| `iterations` | Rondas de práctica | 120. Converge sobre la 40, así que va sobrado |
| `episodes_per_iteration` | Partidas por ronda | 48. Es el "lote": más = menos ruido, más lento |
| `max_steps` | Tope de pasos por partida | 40, para que un ratón perdido no dé vueltas eternas |
| `gamma` | Cuánta paciencia | 0.99 → casi no descuenta el futuro |
| `advantage` | Cómo puntuar cada paso | `"normalized"` (ver abajo) |
| `entropy_coef` | Premio por seguir dudando | 0.01 → un empujoncito a explorar |
| `grad_clip` | Tope al tamaño del ajuste | 1.0, para que una partida con suerte no rompa nada |
| `eval_episodes` | Partidas de examen por ronda | 10, jugadas **sin** azar |
| `seed` | La semilla del azar | 0 |

### `"iterations"` y no `"epochs"`

Detalle deliberado. Una **época** significa "una pasada completa por el conjunto de datos".
Aquí **no hay conjunto de datos por el que pasar** — los datos se fabrican en cada ronda. Así
que la palabra no aplica, y `harness_rl.py` usa otra a propósito.

### `"advantage": "normalized"`

Esta es la única clave que merece explicación de verdad, porque decide **cómo se puntúa cada
paso**. En `harness_rl.py`:

```python
def compute_weights(returns, mode="normalized"):
    if mode == "return":
        return returns                                        # el retorno crudo
    if mode == "baseline":
        return returns - returns.mean()                        # "¿mejor que la media?"
    if mode == "normalized":
        return (returns - returns.mean()) / (returns.std() + 1e-8)
```

El problema que resuelve: si un paso acabó dando `+0,60`, **¿eso es bueno?** No se puede
saber sin comparar con algo.

* `"return"` — usa el `+0,60` a pelo. Como es positivo, **refuerza** ese paso. Pero si todas
  las partidas dan algo positivo, refuerza todas, incluidas las malas. Aprende, pero lento y
  a tumbos.
* `"baseline"` — resta la media del lote. Ahora "bueno" significa **"mejor que lo normal"**, y
  las partidas peores que la media reciben peso **negativo** y se desalientan de verdad.
* `"normalized"` — además divide por la desviación, dejando los pesos alrededor de ±1. Así la
  tasa de aprendizaje funciona igual de bien tanto si tus recompensas van de 0 a 1 como de 0
  a 10.000.

### `resultado = rl.run_rl_experiment(config)`

Una llamada, y por dentro el bucle completo. Simplificado:

```python
for iteration in range(config["iterations"]):          # 120 rondas
    # ── 1. FABRICAR los datos con la política de AHORA ──
    trajectories = [collect_episode(policy, env, ...)
                    for _ in range(48)]

    # ── 2. REPARTIR el mérito ──
    returns = [returns_to_go(t.rewards, gamma) for t in trajectories]
    weights = compute_weights(returns, "normalized").detach()

    # ── 3. UN paso de optimizador (los 4 pasos sagrados de PyTorch) ──
    loss = -(log_probs * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Y la pérdida de la penúltima línea es toda la teoría:

```python
loss = -(log_probs * weights).mean()
```

> 🔑 Compárala con la clasificación normal:
>
> ```python
> supervisado:  loss = -log π(etiqueta_de_un_humano)        # peso implícito: 1
> refuerzo:     loss = -log π(accion_propia) * peso         # peso: qué tal salió
> ```
>
> **Es la misma fórmula.** Cambian dos cosas: la diana la eligió el propio modelo, y cada
> muestra lleva un peso. Ese `* weights` es literalmente todo lo que separa el aprendizaje
> supervisado del gradiente de política.
>
> Y el signo lo hace todo: si el peso es positivo, minimizar `−log(p)·w` **sube** la
> probabilidad de esa acción. Si es negativo, la **baja**.

El `.detach()` de los pesos no es decorativo: dice *"esto es un juicio sobre lo que pasó, no
una cantidad a optimizar"*. Sin él, el gradiente intentaría "mejorar la nota" en vez de
mejorar al ratón.

### Lo que devuelve

`resultado` es un `ExperimentResult` con:

* `resultado.model` — el cerebro entrenado (lo usamos en las celdas 8, 9, 10, 11).
* `resultado.history` — una lista de diccionarios, uno por ronda.
* `resultado.run_id` — el nombre de la carpeta en `runs/`.

---
---

# Celda 7 · La gráfica del aprendizaje

```python
historia = H.load_run(resultado.run_id)["history"]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(historia["epoch"], historia["reward_mean"], lw=2, color="steelblue",
        label="nota media del ratón")
ax.axhline(0.60, color="seagreen", ls="--", lw=2, label="lo máximo posible (+0,60)")
ax.axhline(0.0, color="grey", ls=":", alpha=0.6)
ax.set_xlabel("rondas de práctica")
ax.set_ylabel("nota (recompensa)")
ax.set_title("El ratón aprendiendo")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

### ¿Por qué leer del disco si ya tenemos `resultado`?

```python
historia = H.load_run(resultado.run_id)["history"]
```

Tenemos `resultado.history` en memoria. Pero es una **lista de diccionarios**, incómoda para
graficar. `H.load_run` lee el `metrics.csv` del disco y lo devuelve como un **DataFrame de
pandas**, que permite `historia["reward_mean"]` para sacar una columna entera.

Los dos caminos son válidos:

```python
resultado.history[-1]["reward_mean"]              # lista de diccionarios
historia["reward_mean"].iloc[-1]                  # DataFrame de pandas
```

Y hay un efecto secundario útil: al leer del disco estás **comprobando que se guardó bien**.

> ⚠️ Solo funciona porque en la celda 6 no pusimos `save=False`. En la celda 16 sí lo
> ponemos, y allí esta técnica no serviría.

### La columna se llama `"epoch"` aunque sean rondas

Es una fricción real de reutilizar el arnés supervisado: `H.plot_runs` busca una columna
llamada `epoch`, así que `harness_rl.py` la llama así aunque conceptualmente sea una
iteración. Está anotado en el código, a la vista, en vez de escondido.

### Matplotlib, pieza a pieza

```python
fig, ax = plt.subplots(figsize=(9, 4.5))
```

Devuelve **dos** objetos y hay que tener clara la diferencia:

* **`fig`** (*figure*) = la hoja de papel entera.
* **`ax`** (*axes*) = **un** gráfico dentro de esa hoja, con sus ejes.

Casi todo lo que quieres hacer se le pide al `ax`. La `fig` solo se usa para guardar o para
ajustar el conjunto. `figsize` va en pulgadas.

| Línea | Qué hace |
|---|---|
| `ax.plot(x, y, ...)` | La línea de datos |
| `lw=2` | *linewidth*, grosor |
| `label="..."` | El texto que aparecerá en la leyenda |
| `ax.axhline(0.60, ...)` | Una línea **h**orizontal de referencia en y=0,60 |
| `ls="--"` | *linestyle*: `"--"` discontinua, `":"` de puntos |
| `alpha=0.6` | Transparencia (0 invisible, 1 opaco) |
| `ax.legend()` | Dibuja la leyenda con todas las `label` |
| `ax.grid(alpha=0.3)` | La rejilla, muy tenue |
| `plt.tight_layout()` | Recoloca todo para que no se solapen los textos |
| `plt.show()` | Píntalo ya |

La línea verde en `0.60` es la que convierte la gráfica en algo interpretable: sin una
referencia, una curva que sube no dice si va bien o si le falta muchísimo.

### Los `print` del final

```python
print(f"Nota al empezar : {historia['reward_mean'].iloc[0]:+.3f}")
```

* **`.iloc[0]`** — *integer location*: el primer elemento **por posición**. En pandas hay que
  distinguir posición (`iloc`) de etiqueta (`loc`), porque el índice puede no ser 0,1,2...
* **`:+.3f`** — el formato: `+` fuerza a mostrar el signo siempre (útil para comparar cosas
  que pueden ser negativas), `.3f` son tres decimales.

> 🐍 **Ojo a las comillas.** Dentro de una f-string con comillas dobles, la clave del
> diccionario va con comillas **simples**: `f"{historia['reward_mean']}"`. Mezclar el mismo
> tipo cierra la cadena antes de tiempo. (Desde Python 3.12 ya se permite repetirlas, pero la
> costumbre de alternar sigue siendo la más portable.)

```python
print(f"(el máximo con 4 opciones es ln(4) = {np.log(4):.3f}: ...)")
```

La **entropía** mide cuánto reparte el ratón sus probabilidades, o sea cuánto duda. Con 4
opciones el máximo posible es `ln(4) = 1,386`, que corresponde a `[0.25, 0.25, 0.25, 0.25]`:
ninguna preferencia. Al empezar sale ≈1,385 (duda de todo) y al final 0,000 (ha decidido).

Ese cero es una buena noticia y una mala a la vez: ha decidido, sí, pero **ya no explora**.
Si el mundo cambiara, no se enteraría. Es el dilema exploración/explotación, visible en un
número.

---
---

# Celda 8 · La chuleta de flechas

```python
def chuleta(raton, gato=None, queso=QUESO):
    lineas = []
    for r in range(FILAS):
        fila = []
        for c in range(COLS):
            if   (r, c) in MUROS:         fila.append("⬛")
            elif (r, c) == queso:         fila.append("🧀")
            elif gato and (r, c) == gato:  fila.append("🐱")
            else:
                with torch.no_grad():
                    mejor = int(raton(donde_estoy((r, c))).argmax())
                fila.append(f" {FLECHAS[mejor]}")
        lineas.append("  ".join(fila))
    return "\n".join(lineas)
```

La estructura es la misma que `dibujar_texto`. Lo nuevo son estas tres líneas:

```python
with torch.no_grad():
    mejor = int(raton(donde_estoy((r, c))).argmax())
```

### `with torch.no_grad():`

> 🔥 **PyTorch.** Por defecto, PyTorch **va apuntando todas las operaciones** que haces con
> tensores, para poder calcular derivadas después. Eso se llama el *grafo de cómputo*, y
> cuesta memoria y tiempo.
>
> `torch.no_grad()` lo apaga dentro de ese bloque. Se usa **siempre que solo estás
> preguntando y no vas a entrenar**: evaluación, inferencia, o esto.

Aquí no es solo optimización: es una declaración de intenciones. *"Estoy leyéndole el
pensamiento al ratón, no enseñándole nada."*

> ⚠️ **Trampa clásica.** Si en refuerzo la recompensa la calcula **otro modelo** (un juez, un
> clasificador), y te olvidas del `no_grad()`, el gradiente se escapa hacia el juez y acabas
> "mejorando al examinador" en lugar de al alumno. Es uno de los errores más caros y
> silenciosos de RLHF.

### Las tres operaciones encadenadas

Se leen de dentro hacia fuera:

```python
donde_estoy((r, c))          # ① los 25 números de esa casilla
raton( ... )                 # ② el cerebro: 25 números → 4 notas
      .argmax()              # ③ ¿cuál es la nota más alta?
int( ... )                   # ④ de tensor a número de Python
```

* **②** — llamar a un `nn.Module` como si fuera una función ejecuta su `forward()`. Es azúcar
  de PyTorch: `raton(x)` es mejor que `raton.forward(x)` porque además dispara los *hooks*
  internos.
* **③ `argmax()`** — devuelve **el índice del máximo**, no el máximo. Si las notas son
  `[0.1, 0.3, 0.2, 4.0]`, `argmax()` da `3`. Es "cuál elegiría", no "cuánto vale".
* **④ `int(...)`** — `argmax()` devuelve un **tensor** de un solo elemento, y con un tensor no
  se puede indexar un diccionario de Python. `int()` lo baja a entero normal.

> 🔑 **`argmax` es la versión determinista.** Durante el entrenamiento el ratón **sortea** su
> acción (`distribution.sample()`), para explorar. Aquí le preguntamos por su **favorita**.
> Son dos comportamientos distintos sacados de los mismos pesos, y por eso el arnés registra
> dos métricas: `reward_mean` (jugando con azar) y `eval_reward` (jugando a lo seguro).
>
> Y ojo: en el aprendizaje normal no existe nada parecido. `model.eval()` no cambia lo que el
> modelo *cree*, solo apaga `Dropout` y `BatchNorm`. Aquí la aleatoriedad está en **cómo
> elegimos**, que es una decisión nuestra, no una capa de la red.

---
---

# Celda 9 · Comprobar la chuleta casilla por casilla

```python
def seguir_flechas(desde, raton, max_pasos=40, queso=QUESO):
    pos = desde
    for _ in range(max_pasos):
        if pos == queso:
            return True
        with torch.no_grad():
            a = int(raton(donde_estoy(pos)).argmax())
        df, dc = ACCIONES[a]
        destino = (pos[0] + df, pos[1] + dc)
        if 0 <= destino[0] < FILAS and 0 <= destino[1] < COLS and destino not in MUROS:
            pos = destino
    return False
```

Suelta al ratón en una casilla cualquiera y le hace seguir sus propias flechas, a ver si
acaba en el queso.

* **`return True` en cuanto llega** — sale del bucle inmediatamente.
* **`return False` al final** — si el `for` termina sin llegar, se rindió. Nótese que está
  **fuera** del bucle: solo se ejecuta si el `for` acabó sin `return`.
* **Los 40 pasos de tope** son necesarios: si las flechas forman un círculo, sin tope el
  programa se colgaría para siempre.

> ⚠️ **Una pega honesta de este código.** Las reglas de movimiento están escritas **dos
> veces**: aquí y en `Laberinto.step()`. Si cambiaras el laberinto para que los muros se
> pudieran atravesar en diagonal, tendrías que acordarte de tocar los dos sitios. Es lo que se
> llama tener **dos fuentes de la verdad**, y es un olor a código.
>
> Se ha hecho así **a propósito** para que la función se lea sola, sin tener que entender el
> entorno. En código serio harías que `seguir_flechas` usara el entorno de verdad. La versión
> limpia sería casi la misma llamada que ya usa la celda 10:
> `rl.collect_episode(raton, env, CPU, greedy=True)`.

### El bloque de comprobación

```python
env_ruta = rl.envs.build("laberinto")
rl.collect_episode(resultado.model, env_ruta, CPU, max_steps=40, greedy=True)
su_ruta = set(env_ruta.camino)

libres = [(r, c) for r in range(FILAS) for c in range(COLS) if (r, c) not in MUROS]
llegan = [p for p in libres if seguir_flechas(p, resultado.model)]
fallan = [p for p in libres if p not in llegan]
```

* **Construimos un entorno nuevo y jugamos una partida `greedy=True`** para saber por dónde
  pasa el ratón *de verdad*. Es importante hacerlo **aquí** y no reutilizar el `env` de antes:
  ese `env` guarda el paseo del ratón **novato**, que se paseó por medio laberinto y daría una
  respuesta falsa. (Este bug estaba en la primera versión del notebook.)
* **`set(env_ruta.camino)`** — lo convertimos en conjunto porque solo vamos a preguntar
  "¿está aquí?" y en un conjunto es instantáneo. Además elimina repetidos.
* **`libres`** — comprensión de lista con **dos `for` y un `if`**: las 19 casillas que no son
  muro.
* **`llegan`** — las que sí funcionan. Cada elemento llama a `seguir_flechas`, que simula
  hasta 40 pasos. Son 19 × 40 = 760 pasadas por la red como máximo: instantáneo.
* **`fallan = [p for p in libres if p not in llegan]`** — la diferencia.

> ⚠️ **Detalle de eficiencia.** `p not in llegan` recorre una **lista** cada vez, así que esto
> es O(n²). Con 19 elementos no se nota. Lo correcto en general sería
> `fallan = [p for p in libres if p not in set(llegan)]`, o directamente trabajar con
> conjuntos. Lo dejo así porque es más legible y aquí no importa — pero conviene saber
> distinguir "está mal" de "no importa aquí".

### El resultado, y por qué es la mejor lección del notebook

De 19 casillas, **18 llevan al queso**. La que falla es `(3, 0)`, y su flecha apunta contra
un muro.

Y lo verificamos: **el ratón nunca pisa esa casilla.** Ahí está el porqué. Esa flecha
**nunca se corrigió**: sigue puesta donde la dejó el azar de la inicialización, porque para
arreglarla haría falta haber pasado por allí, haberse chocado, y haber notado que era mala
idea.

> 🔑 Eso es lo que significa que REINFORCE sea ***on-policy***: **solo mejora la política
> donde el agente va**. No aprende nada de los sitios que no pisa.
>
> Y no es un detalle académico. Significa que si tu agente encuentra pronto una forma
> razonable de resolver algo, **dejará de mirar el resto del mapa** — y nunca sabrás si había
> algo mejor un poco más allá.

---
---

# Celda 10 · El dibujo bueno (matplotlib)

La celda más larga, pero es todo dibujo. Vamos por bloques.

### El truco de reutilizar el eje

```python
solo = ax is None
if solo:
    _, ax = plt.subplots(figsize=(5.4, 5.4))
```

Este patrón hace la función utilizable de dos formas:

```python
dibujar(modelo)                    # crea su propia figura
dibujar(modelo, ax=mi_eje)         # dibuja en un eje que ya existe
```

Y es lo que permite que las celdas 15 y 17 pongan dos y cinco laberintos en una misma
figura. Sin él, harían falta dos funciones casi idénticas.

* **`ax is None`** y no `ax == None`. `is` compara **identidad** (¿es el mismo objeto?), `==`
  compara **valor** (y puede estar redefinido por la clase). Con `None` se usa siempre `is`.
* **`_, ax = plt.subplots(...)`** — el `_` descarta la figura, que aquí no nos hace falta.
* **`solo`** se guarda porque más abajo decide si poner leyenda: en una figura de cinco
  paneles, cinco leyendas es ruido.

### Los muros

```python
if (r, c) in MUROS:
    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#3a3a44"))
```

> ⚠️ **La trampa número uno de dibujar cuadrículas: el orden de las coordenadas se
> intercambia.**
>
> | | Nuestro código | Matplotlib |
> |---|---|---|
> | Primera coordenada | **fila** (`r`) | **x** (horizontal) |
> | Segunda coordenada | **columna** (`c`) | **y** (vertical) |
>
> Así que la fila va al eje **y** y la columna al eje **x**. De ahí que se escriba
> `(c, r)` y no `(r, c)`. Si te equivocas, el laberinto sale **transpuesto** y te vuelves loco
> buscando el error.

* **`Rectangle((x, y), ancho, alto)`** — el punto que se le pasa es la **esquina inferior
  izquierda**, no el centro.
* **El `- .5`** centra el cuadrado en la casilla. La casilla `(1,1)` está centrada en el punto
  `(1,1)`, así que su cuadrado de lado 1 va desde `(0.5, 0.5)` hasta `(1.5, 1.5)`.
* **`ax.add_patch(...)`** — las formas geométricas no se "dibujan", se **añaden** al eje.
* **`"#3a3a44"`** — color en hexadecimal: dos dígitos de rojo, dos de verde, dos de azul. Un
  gris azulado oscuro.

### Las flechas

```python
elif raton is not None and (r, c) != queso and (r, c) != gato:
    with torch.no_grad():
        a = int(raton(donde_estoy((r, c))).argmax())
    df, dc = ACCIONES[a]
    ax.arrow(c - dc * .22, r - df * .22, dc * .42, df * .42,
             head_width=.16, head_length=.14,
             fc="#8899aa", ec="#8899aa", lw=1.4)
```

La condición triple: dibuja flecha **solo si** nos pasaron un ratón, **y** no es la casilla
del queso, **y** no es la del gato — porque ahí van los marcadores y una flecha debajo
quedaría sucia.

**La aritmética de la flecha** merece explicarse, porque parece arbitraria y no lo es.

`ax.arrow(x, y, dx, dy)` quiere **el punto de salida** y **cuánto avanza**. Lo obvio sería
salir del centro de la casilla:

```python
ax.arrow(c, r, dc * .42, df * .42)     # ← se ve descentrada
```

Pero entonces la flecha ocupa solo la mitad derecha de la casilla y visualmente parece
desplazada. La solución: **retroceder un poco antes de empezar**.

```none
Para una flecha "→" (dc = +1, df = 0) en la casilla de centro c:

     inicio: c − 1×0,22 = c − 0,22        (un poco a la izquierda del centro)
     avance:     +1×0,42 = 0,42
     final:  c − 0,22 + 0,42 = c + 0,20   (un poco a la derecha)

     ├───────────────┼───────────────┤
   c−0,5           c              c+0,5
           ●────────────►
        c−0,22        c+0,20        ← queda centrada en la casilla
```

Multiplicar por `dc` y `df` hace que funcione en las cuatro direcciones automáticamente: para
"↑" (`df = −1`) el desplazamiento se invierte solo. Los números `.22` y `.42` son puro gusto
visual — cámbialos y verás flechas más largas o más cortas.

* **`head_width` / `head_length`** — el tamaño de la punta.
* **`fc` / `ec`** — *facecolor* y *edgecolor*: relleno y borde.

### El camino recorrido

```python
if camino:
    ys, xs = zip(*camino)
    ax.plot(xs, ys, color="crimson", lw=2.6, alpha=.85, zorder=3,
            marker="o", ms=4, label=f"camino ({len(camino)-1} pasos)")
```

> 🐍 **Python: `zip(*lista)` es el truco para transponer.** Es de los más útiles del lenguaje
> y de los que más cuesta ver la primera vez.
>
> ```none
> camino = [(0,0), (0,1), (0,2), (1,2)]      lista de PAREJAS
>
> zip(*camino)  ≡  zip((0,0), (0,1), (0,2), (1,2))
>                    ↑ el * "desparrama" la lista como argumentos sueltos
>
> resultado:   (0, 0, 0, 1)      ← todos los primeros elementos  = las FILAS
>              (0, 1, 2, 2)      ← todos los segundos            = las COLUMNAS
> ```
>
> De pasar de "una lista de puntos" a "una lista de x y una lista de y", que es lo que
> `ax.plot` necesita.

Y otra vez el intercambio: `ys` recoge las filas y `xs` las columnas, así que la llamada es
`ax.plot(xs, ys)` — **primero las columnas**. Los nombres de las variables están puestos para
que la línea se lea bien y no te confundas.

* **`if camino:`** — una lista vacía es falsa en Python, así que esto cubre a la vez el caso
  `None` y el caso "lista vacía".
* **`zorder=3`** — el orden de apilado. Lo que tiene `zorder` más alto se dibuja **encima**.
  Aquí: muros (por defecto, ~1) < camino (3) < marcadores (4) < la letra R (5).
* **`marker="o", ms=4`** — puntitos en cada casilla del camino, tamaño 4.

### Los marcadores

```python
ax.plot(INICIO[1], INICIO[0], marker="o", ms=17, mfc="#4a7fd0",
        mec="white", mew=1.6, zorder=4)
ax.text(INICIO[1], INICIO[0], "R", ha="center", va="center", zorder=5,
        color="white", fontsize=9, fontweight="bold")
```

Usamos `ax.plot` con **un solo punto** para dibujar un marcador. Es lo idiomático en
matplotlib, aunque suene raro "trazar una línea de un punto".

* **`INICIO[1], INICIO[0]`** — otra vez el intercambio: columna primero.
* **`mfc` / `mec` / `mew`** — abreviaturas de `markerfacecolor`, `markeredgecolor`,
  `markeredgewidth`: relleno, borde y grosor del borde. El borde blanco es lo que hace que el
  círculo se vea aunque caiga encima de la línea roja.
* **`ha` / `va`** — `horizontalalignment` y `verticalalignment`. Con `"center"` en los dos, el
  texto queda centrado **en** el punto en vez de empezar en él.

> 🎨 **Y aquí está el motivo de que no haya emojis en el dibujo.** La primera versión usaba
> `ax.text(..., "🧀")` y matplotlib avisaba:
>
> ```none
> UserWarning: Glyph 129472 (\N{CHEESE WEDGE}) missing from font(s) DejaVu Sans.
> ```
>
> Matplotlib usa por defecto la fuente **DejaVu Sans**, que no tiene emojis. El queso habría
> salido como un **cuadradito vacío** justo en el gráfico más importante del notebook. Los
> emojis se quedan en las salidas de texto (donde los pinta el terminal o el navegador, que sí
> los tienen) y en los dibujos usamos marcadores nativos.

### Los ejes

```python
ax.set_xticks(range(COLS)); ax.set_yticks(range(FILAS))
ax.set_xlim(-.5, COLS - .5); ax.set_ylim(FILAS - .5, -.5)
ax.set_aspect("equal"); ax.grid(color="#cccccc", lw=.8)
```

* **`set_xticks(range(5))`** — marcas en 0,1,2,3,4 y no en los decimales que matplotlib
  elegiría solo.
* **`set_xlim(-.5, 4.5)`** — que el borde caiga justo en el borde de las casillas.
* **`set_ylim(FILAS - .5, -.5)`** — **fíjate en que está al revés**: `4.5` primero y `-0.5`
  después. Eso **invierte el eje vertical**, para que la fila 0 salga **arriba**, como cuando
  lees una matriz. Es la alternativa compacta a llamar a `ax.invert_yaxis()`.
* **`set_aspect("equal")`** — que una unidad en x mida lo mismo que una en y. Sin esto las
  casillas saldrían rectangulares y las flechas diagonales torcidas.

> 🐍 Los `;` juntando dos instrucciones en una línea son legales en Python. En general se
> evitan; aquí agrupan pares que van conceptualmente juntos.

### El uso al final de la celda

```python
env = rl.envs.build("laberinto")
rl.collect_episode(resultado.model, env, CPU, max_steps=40, greedy=True)

dibujar(resultado.model, camino=env.camino,
        titulo="Las flechas que aprendió, y el camino que sigue")
```

`greedy=True` es la clave: jugamos **sin azar**, siempre la acción favorita. Es lo que hay que
hacer para ver "lo que ha aprendido" en vez de "lo que le salió esta vez".

---
---

# Celda 11 · Movemos el queso

```python
QUESO_NUEVO = (4, 0)

env_travesura = rl.envs.build("laberinto", queso=QUESO_NUEVO)
rl.collect_episode(resultado.model, env_travesura, CPU, max_steps=40, greedy=True)
```

Tres líneas, y son la mejor lección del notebook.

**`rl.envs.build("laberinto", queso=QUESO_NUEVO)`** — el `queso=` se le pasa a
`construir_laberinto`, sustituyendo su valor por defecto. Es exactamente por esto que el
entorno se escribió como una fábrica con parámetros: **crear una variante del mundo cuesta
un argumento**.

Y lo importante es lo que **no** aparece en la celda:

* No hay `run_rl_experiment`.
* No hay optimizador.
* No hay entrenamiento.

Usamos **el mismo `resultado.model` de siempre**, tal cual. Solo ha cambiado el mundo.

El resultado: camina hasta `(4,4)`, la esquina donde ya no hay nada, y se queda ahí
40 pasos chocándose. Porque **su chuleta dice "ve a esa esquina"**, y su chuleta es todo lo
que tiene.

> 🔑 Y esto conecta directamente con la celda 2. El ratón nunca vio el queso: sus 25 números
> solo dicen *dónde está él*. Nada en su entrada le informa de dónde está el premio. Así que
> era **imposible** que aprendiera "busca queso"; solo podía aprender "ve a esa casilla".
>
> **La limitación no está en el algoritmo, está en `donde_estoy()`.** Si quisieras un ratón
> que generalizara, tendrías que cambiar esa función para darle información relativa ("el queso
> está 2 a la derecha y 3 abajo") y entrenarlo en muchos laberintos. El algoritmo sería el
> mismo.

```python
print("  " + " → ".join(str(p) for p in env_travesura.camino[-8:]))
```

**`[-8:]`** — rodaja con índice negativo: **los 8 últimos**. Los índices negativos cuentan
desde el final (`-1` es el último). Aquí sirven para ver que se quedó atascado repitiendo la
misma casilla.

---
---

# Celda 12 · Dibujar la travesura

```python
dibujar(resultado.model, queso=QUESO_NUEVO, camino=env_travesura.camino,
        titulo="Mismo ratón, queso movido: se va a la esquina de siempre")
```

Una sola llamada, reutilizando la función de la celda 10. Le pasamos `queso=QUESO_NUEVO` para
que la estrella se pinte en el sitio nuevo — así se ve de un golpe que el camino rojo va a un
lado y la estrella está en el otro.

Nótese que las **flechas siguen siendo las mismas**: son del modelo, y el modelo no ha
cambiado.

---
---

# Celda 13 · Poner el gato

```python
GATO = (2, 4)
print("El laberinto ahora:\n")
print(dibujar_texto(gato=GATO))
```

Aquí solo se **mira**. `dibujar_texto` ya aceptaba un parámetro `gato` desde la celda 1,
aunque hasta ahora no lo habíamos usado: se escribió previendo esto.

**¿Por qué la casilla (2,4)?** No es al azar. Acuérdate de que el laberinto tiene **dos
pasillos** para bajar, por la columna 0 y por la columna 4. La casilla (2,4) es el **cuello de
botella del pasillo derecho**: bloquearla obliga a usar el izquierdo.

Y como los dos pasillos miden lo mismo, **el rodeo sale gratis**: el ratón seguirá llegando en
8 pasos. Eso hace la lección limpia (se ve el efecto sin que se mezcle con "y además tarda
más").

> ⚠️ Elegir bien esa casilla costó varios intentos. Si el gato bloqueara el **único** camino y
> el rodeo fuera mucho más largo, este ratón **no lo encontraría**: lo medí y **0 de 3
> semillas** lo consiguen, porque descubrir por casualidad un camino de 12 pasos es mucho más
> difícil que uno de 8. Está avisado en el notebook, y es el problema de la exploración otra
> vez.

---
---

# Celda 14 · Reentrenar con gato

```python
config_gato = dict(config, name="raton_con_gato", env_args={"gato": GATO})
resultado_gato = rl.run_rl_experiment(config_gato, verbose=False)
```

### `dict(config, clave=valor)`: copiar con cambios

> 🐍 **Python.** `dict(d, k=v)` crea un **diccionario nuevo** con todo lo de `d` y además (o
> en lugar de) `k=v`. **No modifica `d`.**

Eso último es crítico aquí:

```python
# ✅ Lo que hacemos: config sigue intacto
config_gato = dict(config, name="raton_con_gato", env_args={"gato": GATO})

# ❌ Lo que NO hay que hacer:
config["name"] = "raton_con_gato"        # ¡esto rompe config para siempre!
config["env_args"] = {"gato": GATO}
```

Si mutáramos `config`, la celda 16 (los cinco ratones) entrenaría **con gato sin saberlo**, y
los resultados no tendrían sentido. Es un bug clásico y silencioso: nada falla, solo que los
números están mal.

### `env_args`: cómo llegan los parámetros al mundo

`env_args` es el sobre en el que el arnés mete los argumentos del entorno. Por dentro:

```python
env = envs.build(config["env"], **config.get("env_args", {}))
#                                 ↑ se desparrama como argumentos con nombre
```

Con `env_args={"gato": GATO}`, la llamada acaba siendo
`construir_laberinto(gato=(2, 4))`. Y por eso `construir_laberinto` tenía que aceptar `gato`
como parámetro: es la única vía para configurar el mundo desde el config.

* **`config.get("env_args", {})`** — `.get` con valor por defecto: si la clave no existe,
  devuelve `{}` en vez de fallar. Así `env_args` es opcional.
* **`verbose=False`** — que no imprima las notas de cada ronda. Ya vimos el proceso en la
  celda 6; aquí solo interesa el resultado.

### La comprobación

```python
print(f"¿Pasó por el gato? {'SÍ 😿' if GATO in env_gato.camino else 'NO, lo esquivó ✓'}")
```

**`GATO in env_gato.camino`** — busca la tupla en la lista de posiciones. Es la comprobación
directa: si el gato no aparece en el camino, lo esquivó.

Y lo notable es **cómo** lo aprendió: nadie le dijo que hubiera un gato. Los 25 números de
`donde_estoy` no mencionan gatos. Simplemente, las partidas que pasaban por ahí acababan con
`−1,00`, ese peso negativo bajaba la probabilidad de ir en esa dirección, y a fuerza de
repetirlo el pasillo derecho dejó de ser una opción.

---
---

# Celda 15 · Comparar los dos, lado a lado

```python
fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 5.4))

env_sin = rl.envs.build("laberinto")
rl.collect_episode(resultado.model, env_sin, CPU, max_steps=40, greedy=True)
dibujar(resultado.model, camino=env_sin.camino,
        titulo="Sin gato: baja por la derecha", ax=izq)

dibujar(resultado_gato.model, gato=GATO, camino=env_gato.camino,
        titulo="Con gato: las flechas se dan la vuelta", ax=der)
```

* **`plt.subplots(1, 2)`** — una fila, dos columnas. Devuelve la figura y un **array de dos
  ejes**, que desempaquetamos con `(izq, der)`.
* **`ax=izq`, `ax=der`** — aquí se cobra el trabajo del `solo = ax is None` de la celda 10: la
  misma función dibuja en el panel que le digas.
* **Volvemos a jugar la partida sin gato** (`env_sin`) porque `env` se reutilizó en celdas
  intermedias y su `camino` podría ser de otra cosa. Regenerarlo es más barato que rastrear
  qué contiene.
* **Dos modelos distintos**: `resultado.model` a la izquierda, `resultado_gato.model` a la
  derecha. Son dos ratones, entrenados por separado.

El resultado visual es la moraleja: **todas las flechas se han dado la vuelta**.

---
---

# Celda 16 · Cinco ratones, cinco semillas

```python
ratones = {}
for semilla in range(5):
    res = rl.run_rl_experiment(dict(config, seed=semilla, name=f"raton_s{semilla}"),
                               verbose=False, save=False)
    env_s = rl.envs.build("laberinto")
    rl.collect_episode(res.model, env_s, CPU, max_steps=40, greedy=True)
    ratones[semilla] = (res, env_s.camino, env_s.pos)
```

* **`dict(config, seed=semilla, ...)`** — otra vez copiar-con-cambios. Cinco configs
  distintos, `config` intacto.
* **`f"raton_s{semilla}"`** — nombres distintos para no confundirlos.
* **`save=False`** — **no** escribe en `runs/`. Cinco entrenamientos son cinco carpetas de
  basura; aquí solo queremos comparar en memoria. (Efecto secundario: en esta celda no se puede
  usar `H.load_run`, porque no hay nada guardado.)
* **`ratones[semilla] = (res, camino, pos)`** — guardamos una **tupla de tres cosas** por cada
  semilla. Un diccionario de tuplas es la forma más rápida de agrupar resultados
  heterogéneos sin definir una clase.

### Detectar por qué pasillo bajó

```python
columnas_medias = [p[1] for p in camino if p[0] == 2]
lado = "izquierda" if columnas_medias and min(columnas_medias) == 0 else "derecha"
```

Truco sencillo: la **fila 2** es la única totalmente abierta que cruza el laberinto. Mirando
por qué columnas pasó el ratón al estar en la fila 2, sabemos qué pasillo usó.

* **`[p[1] for p in camino if p[0] == 2]`** — de las posiciones cuya **fila** (`p[0]`) es 2,
  quédate con la **columna** (`p[1]`).
* **`min(...) == 0`** — si en algún momento estuvo en la columna 0 de la fila 2, bajó por la
  izquierda.
* **`columnas_medias and ...`** — el guardián. Si la lista está vacía, `min()` **explotaría**
  con `ValueError`. Con el `and` delante, el cortocircuito evita la llamada.

> ⚠️ **Es una heurística, no una verdad.** Funciona en *este* laberinto porque los dos
> pasillos son las columnas 0 y 4. Si cambiaras el mapa, esta línea mentiría sin avisar. Es
> el tipo de código que está bien en un notebook exploratorio y que no debe acabar en una
> librería.

### El formateo de la tabla

```python
print(f"    {semilla}   │   {len(camino)-1:2d}  │  {'SÍ' if fin == QUESO else 'NO':2s}   │ {lado}")
```

* **`:2d`** — entero en un ancho de 2 caracteres, con espacio delante si hace falta. Es lo
  que mantiene las columnas alineadas.
* **`:2s`** — lo mismo para cadenas.
* **`{'SÍ' if ... else 'NO':2s}`** — 🐍 aquí hay un detalle exótico: se aplica un formato **al
  resultado de una expresión condicional**, dentro de una f-string. Los dos puntos separan la
  expresión del formato, así que se evalúa primero el `if/else` y luego se formatea. Legal, y
  sorprendentemente útil para tablas.
* **Los caracteres `│` y `┼`** son de dibujo de cajas Unicode. Quedan mejor que `|` y `+`.

---
---

# Celda 17 · Los cinco dibujos

```python
fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
for ax, (semilla, (res, camino, fin)) in zip(axes, ratones.items()):
    dibujar(res.model, camino=camino, titulo=f"ratón {semilla}", ax=ax)
    ax.set_xticklabels([]); ax.set_yticklabels([])
```

* **`zip(axes, ratones.items())`** — empareja el eje 0 con el ratón 0, el 1 con el 1, etc. Es
  el patrón estándar para rellenar una rejilla de gráficos.
* **`(semilla, (res, camino, fin))`** — 🐍 **desempaquetado anidado**. `ratones.items()` da
  pares `(clave, valor)`, y el valor es a su vez una tupla de tres. Los paréntesis internos
  desmontan las dos capas de una vez. Es muy Python, y muy legible cuando te acostumbras.
* **`set_xticklabels([])`** — quita los **números** de los ejes pero deja la rejilla. Con
  cinco paneles pequeños, los números serían ruido.

Y aquí es donde `solo = ax is None` de la celda 10 se cobra la segunda vez: como pasamos
`ax`, la función no dibuja cinco leyendas.

---
---

# Apéndice A · Qué pasa dentro de las funciones del arnés

El notebook llama a tres funciones de `harness_rl.py` como si fueran cajas negras. Abrámoslas.

### `rl.collect_episode(politica, env, device, max_steps, greedy)`

Juega **una** partida y devuelve lo que hizo. Es la función que **no existe** en el arnés
supervisado, porque allí su equivalente (`datasets.build()`) corre una sola vez antes del
entrenamiento.

```python
trajectory = Trajectory()
state = env.reset().to(device)

for _ in range(max_steps):
    distribution = Categorical(logits=policy(state))       # ① las 4 notas → probabilidades
    action = (distribution.probs.argmax() if greedy        # ② elegir
              else distribution.sample())

    trajectory.log_probs.append(distribution.log_prob(action))   # ③ apuntar CON grafo
    trajectory.entropies.append(distribution.entropy())

    state, reward, done = env.step(int(action.item()))     # ④ el mundo responde
    trajectory.rewards.append(reward)                      #    apuntar SIN grafo
    if done:
        break
```

Los dos detalles que importan:

* **`log_probs` llevan grafo de gradiente; `rewards` son floats puros.** El gradiente solo
  puede viajar por los `log_probs`. Las recompensas son un veredicto, no algo derivable.
* **`greedy` cambia de política sin cambiar los pesos.** Muestrear = explorar (entrenamiento).
  `argmax` = explotar (evaluación). Dos comportamientos, un solo cerebro.

### `rl.run_rl_experiment(config)`

El bucle completo. Por cada una de las 120 rondas:

```none
① collect_episode × 48          →  FABRICAR datos con la política de ahora
② returns_to_go + compute_weights → REPARTIR el mérito hacia atrás en el tiempo
③ policy_loss, backward, step    →  UN paso de optimizador
④ evaluate_greedy × 10           →  el examen, sin azar
```

Y al acabar guarda `config.json`, `metrics.csv`, `weights.pt` y `meta.json`.

> 🔑 **Los datos son de un solo uso.** Después del paso ③ la política ha cambiado, así que
> esas 48 partidas describen a un ratón que **ya no existe** y se tiran. Eso es *on-policy*, y
> es la ineficiencia fundamental del gradiente de política. Los métodos *off-policy* (DQN, SAC)
> existen precisamente para poder reciclar experiencia vieja.

### `rl.returns_to_go(recompensas, gamma)`

Convierte "lo que gané en cada paso" en "lo que valió cada paso mirando su futuro".

```python
returns, running = [], 0.0
for reward in reversed(rewards):       # ← HACIA ATRÁS
    running = reward + gamma * running
    returns.append(running)
return list(reversed(returns))
```

Va **al revés** porque el mérito de un paso depende de lo que vino **después**. Un paso se
juzga por su futuro, no por su pasado.

Con nuestro camino óptimo y `gamma=0.99`:

| Paso | Recompensa inmediata | Retorno $G_t$ |
|---|---|---|
| 1 | −0,05 | +0,55 |
| 2 | −0,05 | +0,60 |
| 3 | −0,05 | +0,66 |
| 4 | −0,05 | +0,72 |
| 5 | −0,05 | +0,77 |
| 6 | −0,05 | +0,83 |
| 7 | −0,05 | +0,89 |
| 8 | **+0,95** | **+0,95** |

Lee la columna de la derecha de abajo arriba y ves cómo el premio se va **propagando hacia
atrás**, perdiendo un poco en cada salto (ese poco es `gamma`). El primer paso, que
inmediatamente solo dio pérdidas, acaba valorado en **+0,55**.

Eso es lo que permite que el ratón aprenda que caminar valía la pena, aunque caminar, por sí
solo, únicamente diera pérdidas.

---
---

# Apéndice B · Chuleta de Python y PyTorch

Todo lo que aparece en el notebook, en una tabla.

### Python

| Construcción | Ejemplo del notebook | Qué hace |
|---|---|---|
| Desempaquetado de tuplas | `FILAS, COLS = len(MAPA), len(MAPA[0])` | Reparte una tupla en varias variables |
| Desempaquetado anidado | `for ax, (s, (r, c, f)) in zip(...)` | Desmonta varias capas de una vez |
| Comprensión de lista | `[p[1] for p in camino if p[0] == 2]` | Construye una lista filtrando |
| Comprensión de conjunto | `{(r, c) for ... if ch == "#"}` | Igual, pero conjunto: `in` instantáneo |
| Expresión generadora | `sum(1 for a, b in zip(...) if a == b)` | Como la lista, pero sin construirla en memoria |
| `enumerate` | `for r, fila in enumerate(MAPA)` | Índice **y** valor a la vez |
| `zip` | `zip(camino, camino[1:])` | Empareja; se para en la más corta |
| `zip(*lista)` | `ys, xs = zip(*camino)` | **Transpone**: de lista de puntos a listas de coordenadas |
| Comparación en cadena | `0 <= x < FILAS` | `0 <= x and x < FILAS` |
| Cortocircuito | `if gato and self.pos == gato` | Si el primero es falso, no evalúa el segundo |
| Expresión condicional | `'SÍ' if llegó else 'NO'` | El "ternario" de Python |
| f-string | `f"{valor:+.3f}"` | Interpola y formatea |
| Formato `:+.3f` | | Signo siempre, 3 decimales |
| Formato `:2d` / `:2s` | | Ancho fijo, para alinear tablas |
| Rodajas | `camino[:14]`, `camino[-8:]` | Los primeros N / los últimos N |
| `dict(d, k=v)` | `dict(config, seed=3)` | **Copia** con cambios (no muta `d`) |
| `.get(k, defecto)` | `config.get("env_args", {})` | Lectura que no falla si no existe |
| `**kwargs` | `def f(..., **kwargs)` | Recoge argumentos extra |
| `**diccionario` | `envs.build(nombre, **args)` | Desparrama un dict como argumentos |
| Decorador | `@rl.envs.register("laberinto")` | Registra la función de debajo |
| Clausura | clase dentro de una función | La clase "se lleva puestas" las variables de fuera |
| `_` | `for _ in range(200)` | "No me importa esta variable" |
| `is None` | `solo = ax is None` | Identidad, nunca `== None` |
| `MAYÚSCULAS` | `MUROS`, `QUESO` | Convención: es una constante |

### PyTorch

| Construcción | Ejemplo | Qué hace |
|---|---|---|
| `torch.zeros(n)` | `torch.zeros(25)` | Tensor de ceros (`float32`) |
| `torch.device("cpu")` | `CPU = ...` | Dónde viven los cálculos |
| `nn.Sequential(...)` | la política | Encadena capas en orden |
| `nn.Linear(a, b)` | `nn.Linear(25, 64)` | Multiplica por matriz y suma sesgo |
| `nn.Tanh()` | | Aplasta a [−1, +1]; sin esto, dos `Linear` = una |
| `modelo(x)` | `raton(donde_estoy(p))` | Ejecuta `forward()` |
| `.argmax()` | | El **índice** del máximo, no el máximo |
| `int(tensor)` | | De tensor de 1 elemento a entero de Python |
| `.item()` | `action.item()` | Igual, para cualquier escalar |
| `with torch.no_grad():` | | Apaga el registro de derivadas |
| `Categorical(logits=...)` | | Distribución sobre acciones; aplica Softmax por dentro |
| `.sample()` / `.log_prob()` | | Sortea una acción / su log-probabilidad |
| `.entropy()` | | Cuánto duda; máximo `ln(n_acciones)` |
| `.detach()` | `weights.detach()` | "Esto es un dato, no algo a optimizar" |
| `torch.manual_seed(s)` | dentro de `H.set_seed` | Fija el azar de PyTorch |

### Matplotlib

| Construcción | Qué hace |
|---|---|
| `fig, ax = plt.subplots()` | `fig` = la hoja, `ax` = un gráfico dentro |
| `plt.subplots(1, 5)` | Rejilla de 1×5; devuelve un array de ejes |
| `ax.plot(x, y)` | Línea (o un marcador, si es un punto) |
| `ax.arrow(x, y, dx, dy)` | Flecha: **desde** y **cuánto avanza** |
| `ax.add_patch(Rectangle(...))` | Añade una forma; el punto es la esquina inferior izquierda |
| `ax.axhline(v)` | Línea horizontal de referencia |
| `ax.set_ylim(alto, bajo)` | **Invertido** → la fila 0 arriba |
| `ax.set_aspect("equal")` | Casillas cuadradas |
| `zorder` | Quién se dibuja encima |
| `mfc` / `mec` / `mew` | Relleno / borde / grosor del borde del marcador |
| `ha` / `va` | Alineación del texto: centrado **en** el punto |

> ⚠️ **Y el aviso que más veces te va a morder:** en nuestro código las coordenadas son
> `(fila, columna)`; en matplotlib son `(x, y)` = `(columna, fila)`. **Van al revés.** Cada
> vez que pintes una cuadrícula, comprueba ese intercambio antes de buscar el error en otro
> sitio.

---

## Para seguir

| Si quieres... | Ve a |
|---|---|
| El mecanismo matemático con números | `notebooks/02-refuerzo-con-el-arnes-rl.ipynb` |
| Por qué RL no cabe en el arnés normal | `lab/HARNESS.md` §4 |
| Qué le falta a este arnés y qué usar en su lugar | `lab/harness_rl.py`, sección 🔀 |
| Cómo esto alinea modelos de lenguaje (RLHF) | `notebooks/01-aprendizaje-multi-etapa-arnes.ipynb` |
