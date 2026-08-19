---
title: "Lista de notebooks · N00 a N30"
tags: [notebooks, itinerario, specs, practica]
status: vivo
---

# Lista de notebooks · N00 → N30

> **Qué es esto.** La lista de trabajo. Se recorre **en orden**, del N00 al N30. Cada
> entrada trae su contexto y el bloque listo para pasarle a un LLM.
>
> **El porqué de todo esto** —principios, dataset, diseño del itinerario— está en
> [`00-metodo/itinerario.md`](../../00-metodo/itinerario.md). Este documento es para
> trabajar, aquel para entender.

---

## Cómo se usa cada notebook

```none
1. Leer el contexto de la entrada (arriba del bloque)
2. Escribir la HIPÓTESIS en la bitácora        ← antes de ejecutar nada
3. Copiar: PREÁMBULO + BLOQUE Nxx  →  al LLM
4. Construir, ajustar, romper lo que dice la sección "QUÉ ROMPO"
5. Cerrar la entrada de bitácora: resultado, sorpresa, siguiente paso
6. Marcar la casilla en la tabla de progreso
```

**Los marcados con ⭐ son los que más enseñan.** Si tienes que recortar por tiempo,
recorta en cualquier otro sitio.

---

## Tabla de progreso

| | Notebook | Pregunta | Bloque |
|---|---|---|---|
| ☐ | **[N00](#n00-el-arnés-de-experimentos)** El arnés de experimentos | ¿puedo lanzar una variación sin reescribir el bucle? | 0 |
| ☐ | **[N01](#n01-fábrica-de-datos-sintéticos)** Fábrica de datos sintéticos | ¿puedo generar problemas donde conozca la respuesta exacta? | 0 |
| ☐ | **[N02](#n02-el-perceptrón-y-su-muerte)** El perceptrón, y su muerte | ¿qué es exactamente una neurona, y dónde se rompe? | 1 |
| ☐ | **[N03](#n03-backprop-a-mano)** Backprop a mano | ¿sé calcular un gradiente sin `autograd`? | 1 |
| ☐ | **[N04](#n04-verificación-del-gradiente)** Verificación del gradiente ⭐ | ¿mi backprop es correcto? | 1 |
| ☐ | **[N05](#n05-el-mismo-modelo-en-pytorch)** El mismo modelo en PyTorch | ¿coinciden mis gradientes con los del framework? | 1 |
| ☐ | **[N06](#n06-la-caja-de-instrumentos)** La caja de instrumentos ⭐ | ¿puedo ver si el modelo está aprendiendo, y dónde? | 2 |
| ☐ | **[N07](#n07-parar-tocar-seguir)** Parar, tocar, seguir | ¿puedo intervenir en mitad del entrenamiento? | 2 |
| ☐ | **[N08](#n08-patologías-del-gradiente)** Patologías del gradiente | ¿cómo se ve un gradiente enfermo? | 3 |
| ☐ | **[N09](#n09-tasa-de-aprendizaje-y-optimizadores)** Tasa de aprendizaje y optimizadores | ¿cuánto importa el optimizador comparado con la tasa? | 3 |
| ☐ | **[N10](#n10-regularización-y-el-experimento-incómodo)** Regularización, y el experimento incómodo ⭐ | ¿la regularización impide memorizar? | 3 |
| ☐ | **[N11](#n11-normalización)** Normalización | ¿qué hace realmente BatchNorm? | 3 |
| ☐ | **[N12](#n12-precisión-y-memoria)** Precisión y memoria | ¿por qué no cabe, y por qué da NaN? | 3 |
| ☐ | **[N13](#n13-tu-ruido-de-fondo)** Tu ruido de fondo ⭐ | ¿cuánto se mueven mis resultados solo con la semilla? | 4 |
| ☐ | **[N14](#n14-métricas-que-engañan)** Métricas que engañan | ¿mi métrica ve el problema? | 4 |
| ☐ | **[N15](#n15-fuga-de-datos)** Fuga de datos | ¿mi split mide lo que creo? | 4 |
| ☐ | **[N16](#n16-mlp-como-aproximador-universal)** MLP como aproximador universal | ¿qué puede y qué no puede un MLP? | 5 |
| ☐ | **[N17](#n17-cnn-el-sesgo-inductivo-que-se-ve)** CNN: el sesgo inductivo que se ve | ¿cuánto vale compartir pesos? | 5 |
| ☐ | **[N18](#n18-rnn-y-lstm-memoria-y-explosión)** RNN y LSTM: memoria y explosión ⭐ | ¿cómo se procesa una secuencia, y por qué es inestable? | 5 |
| ☐ | **[N19](#n19-atención-desde-cero)** Atención desde cero | ¿qué es realmente Q, K, V? | 5 |
| ☐ | **[N20](#n20-transformer-mínimo)** Transformer mínimo | ¿puedo montar el bloque completo? | 5 |
| ☐ | **[N21](#n21-auto-supervisado-el-modelo-base)** Auto-supervisado: el modelo base | ¿qué aprende un modelo que solo predice el siguiente token? | 6 |
| ☐ | **[N22](#n22-contrastivo-y-el-colapso)** Contrastivo y el colapso | ¿cómo se aprende sin reconstruir nada? | 6 |
| ☐ | **[N23](#n23-transferencia-y-sondeo-lineal)** Transferencia y sondeo lineal | ¿cuánto valen las representaciones del modelo base? | 6 |
| ☐ | **[N24](#n24-lora)** LoRA ⭐ | ¿puedo ajustar tocando el 0,1 % de los pesos? | 6 |
| ☐ | **[N25](#n25-sft-enseñar-a-responder)** SFT: enseñar a responder ⭐ | ¿cómo pasa de autocompletar a responder? | 6 |
| ☐ | **[N26](#n26-preferencias-enseñar-criterio)** Preferencias: enseñar criterio | ¿puedo mejorar sin escribir la respuesta perfecta? | 6 |
| ☐ | **[N27](#n27-refuerzo-de-bandidos-a-política)** Refuerzo: de bandidos a política | ¿cómo se aprende sin que nadie te diga la respuesta? | 6 |
| ☐ | **[N28](#n28-rlvr-el-modelo-descubre-solo)** RLVR: el modelo descubre solo ⭐ | ¿puede el modelo aprender a razonar sin demostraciones? | 6 |
| ☐ | **[N29](#n29-destilación)** Destilación | ¿puedo transferir capacidad a un modelo pequeño? | 7 |
| ☐ | **[N30](#n30-el-pipeline-de-punta-a-punta)** El pipeline de punta a punta ⭐ | ¿qué aporta cada fase, medido? | 7 |

---

## El preámbulo común

**Pega esto delante de cualquier bloque.** No lo repito en cada entrada.

```text
CONTEXTO GENERAL
Estoy construyendo un itinerario de notebooks de deep learning, de lo más básico a un
pipeline completo de modelo de lenguaje. Soy programador con Python sólido y entiendo
backpropagation conceptualmente. Mi objetivo NO es obtener buenos resultados: es
ENTENDER los mecanismos y aprender a diagnosticar.

RESTRICCIONES INNEGOCIABLES
- Todo debe correr en CPU en menos de 10 minutos. Si algo tarda más, reduce el problema.
- Datos sintéticos generados en el propio notebook, salvo que el bloque diga lo contrario.
- Nada de descargas grandes ni de modelos preentrenados grandes.
- Código legible por encima de código eficiente. Prefiero bucles claros a vectorización
  incomprensible.
- Comenta el PORQUÉ, no el QUÉ. No expliques que `x += 1` incrementa x.

ESTRUCTURA OBLIGATORIA DE CADA NOTEBOOK
1. Celda markdown inicial: la PREGUNTA que responde el notebook, en una frase, y la
   HIPÓTESIS de qué va a pasar.
2. Imports y semilla fija (registrada en una variable visible).
3. Generación de datos.
4. El experimento principal.
5. SECCIÓN "QUÉ ROMPO AQUÍ": el fallo provocado a propósito que indica el bloque.
6. Gráficas de diagnóstico.
7. Celda markdown final: QUÉ APRENDÍ, QUÉ ME SORPRENDIÓ, SIGUIENTE PASO.

REGLAS DE CALIDAD
- Antes de confiar en cualquier entrenamiento: comprueba que el modelo puede sobreajustar
  10 muestras hasta pérdida ≈ 0. Si no, hay un bug.
- Comprueba que la pérdida inicial coincide con el valor teórico esperado.
- Nunca reportes un número sin decir de cuántas ejecuciones sale.
- Si un resultado sale demasiado bien, sospecha antes de celebrarlo y dime por qué.

FORMATO DE SALIDA
Dame el notebook como celdas numeradas, indicando si cada una es markdown o código.
```

---
---

# BLOQUE 0 · El andamio

> **¿Tengo con qué trabajar?**

---

## N00 · El arnés de experimentos

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo lanzar una variación sin reescribir el bucle?

**Construyes:** una función que acepte una configuración y devuelva métricas. Fija la
semilla, guarda configuración, métricas por época, pesos finales y el commit.

**Qué rompes:** nada. Es infraestructura.

**Criterio de terminado:** puedes lanzar tres configuraciones distintas cambiando solo un
diccionario, y los resultados quedan en disco separados.

→ **Enlaza con:** 3.9 (parte B, trazabilidad) · *Bitácora*
**Tiempo:** 1 sesión

> ⚠️ **La tentación es saltarse este notebook.** No lo hagas. Sin arnés harás diez
> experimentos en vez de cien, porque cada uno cuesta demasiado.

---

### 📋 Bloque para el LLM

```text
BLOQUE N00

OBJETIVO
Construir la infraestructura mínima para que lanzar un experimento nuevo cueste una
línea de configuración, no una copia del notebook anterior.

QUÉ CONSTRUIR
1. Una función `run(config: dict) -> dict` que:
   - fije todas las semillas (python, numpy, torch)
   - construya modelo, datos y optimizador a partir del dict
   - entrene y devuelva un dict de métricas por época
2. Persistencia en disco: cada ejecución crea una carpeta con
   `config.json`, `metrics.csv`, `weights.pt` y `meta.json` (semilla, fecha, commit git
   si existe).
3. Una función `compare(run_ids) -> DataFrame` que cargue varias ejecuciones y las
   ponga en una tabla.
4. Una función `plot_runs(run_ids, metric)` que superponga las curvas.

DATOS
Un problema trivial de juguete (regresión lineal con ruido) solo para probar el arnés.
El dataset real llega en N01.

QUÉ ROMPO AQUÍ
Nada. Es infraestructura. Pero incluye un test que compruebe que dos ejecuciones con la
misma semilla dan resultados idénticos, y que con semillas distintas NO los dan.

CRITERIO DE TERMINADO
Puedo lanzar tres configuraciones cambiando solo un diccionario, y comparar las tres
curvas en una sola gráfica sin escribir código nuevo.

NOTA DE DISEÑO
No construyas un framework. Un solo fichero de menos de 200 líneas. Si crece más, es que
me estoy desviando del objetivo.
```

---

## N01 · Fábrica de datos sintéticos

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo generar problemas donde conozca la respuesta exacta?

**Construyes:** un módulo con generadores parametrizables:

- Regresión lineal con ruido controlado → conoces los coeficientes verdaderos
- Clasificación separable / no separable → conoces la frontera óptima
- Espirales, medias lunas → no linealidad controlada
- Desbalanceo, ruido de etiqueta, duplicados → los parámetros que romperán cosas después
- **El lenguaje sintético del hilo conductor**, con su verificador

**Qué rompes:** genera un dataset donde la relación sea puro ruido, y guárdalo. Lo
usarás en N09 y N14 para ver a un modelo "aprender" algo que no existe.

**Criterio de terminado:** para cualquier dataset que generes, puedes calcular el
rendimiento del **modelo óptimo**. Ese número es tu techo, y sin él no sabes si un 0,87
es bueno o desastroso.

→ **Enlaza con:** 3.0 · 3.9
**Tiempo:** 2 sesiones

### 📋 Bloque para el LLM

```text
BLOQUE N01

OBJETIVO
Construir el generador de datos que usarán TODOS los notebooks siguientes. Tiene dos
partes independientes.

PARTE A · DATOS TABULARES Y 2D (para N02–N17)
Generadores, todos con semilla y con parámetros:
- recta_con_ruido(pendiente, sesgo, sigma, n)
- dos_gaussianas(separacion, n)
- xor(n)                        ← para N02
- espirales(vueltas, ruido, n)
- medias_lunas(separacion, n)
- ruido_puro(n_features, n)     ← SIN relación real entre X e y
- imagenes_formas(tamano, n)    ← círculos/cuadrados, para CNN

Modificadores aplicables a cualquiera:
- desbalanceo(proporcion)
- ruido_etiqueta(pct)
- duplicados(pct)               ← para provocar fuga de datos
- grupos(n_grupos)              ← estructura de sitio/sujeto
- deriva_temporal()

CRÍTICO: para cada dataset, una función `optimo(dataset) -> float` que devuelva el
rendimiento del mejor modelo posible. Sin ese número no sé si un 0.87 es bueno o malo.

PARTE B · EL LENGUAJE ARITMÉTICO (para N18–N30)
Vocabulario: dígitos 0-9, operadores + - * ( ), el símbolo =, y <pad> <eos>. 17 tokens.
Tokenización a nivel de CARÁCTER (cada dígito es un token independiente).

Cinco niveles de dificultad:
  nivel 0: un dígito           3+4=7
  nivel 1: 2-3 dígitos         47+38=85        ← con acarreo
  nivel 2: precedencia         3+4*2=11
  nivel 3: paréntesis          (3+4)*2=14
  nivel 4: enunciado en texto  (opcional, dejar sin implementar de momento)

Funciones requeridas:
- generar(nivel, n, semilla) -> lista de expresiones
- formatear(expr, modo)      -> 'crudo' | 'chat' | 'invertido'
    · 'chat': <|user|>¿cuánto es 3+4?<|assistant|>7<|end|>
    · 'invertido': la respuesta con los dígitos al revés (47+38=58)
- tokenizar(texto, modo)     -> 'char' | 'agrupado'
- verificar(problema, respuesta) -> bool
- resolver(problema)         -> str
- pasos(problema)            -> traza de referencia con pasos intermedios
- generar_preferencias(tipo, n) -> pares (preferida, rechazada)
    · tipo 'correccion':  "46" ≻ "45"
    · tipo 'longitud':    "46" ≻ "El resultado de sumar 12 y 34 es 46."
    · tipo 'honestidad':  "no lo sé" ≻ invención, para números fuera de rango

TRES TIPOS DE SPLIT (importante, implementar los tres):
- aleatorio        → cuidado: 47+38 en train y 38+47 en test
- por_resultado    → todos los que dan 85 en train, ninguno en test
- por_rango        → entrena con 1-3 dígitos, evalúa con 4

QUÉ ROMPO AQUÍ
Genera el dataset de ruido_puro y guárdalo. En N10 y N14 entrenaré sobre él para ver a un
modelo "aprender" algo que no existe.

CRITERIO DE TERMINADO
Con estas funciones, ningún notebook posterior necesita escribir código de datos.
```


# BLOQUE 1 · El mecanismo desnudo

> **¿Entiendo qué pasa por dentro?**

---

## N02 · El perceptrón, y su muerte

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué es exactamente una neurona, y dónde se rompe?

**Construyes:** un perceptrón a mano. Regla de actualización, frontera de decisión
dibujada época a época.

**Qué rompes:** **XOR.** Ejecuta hasta convencerte de que no converge nunca. Dibuja la
frontera e intenta separar los cuatro puntos con una recta.

**La lección:** ese fallo concreto paró el campo casi veinte años. No es anécdota
histórica: es la razón de que existan las capas ocultas.

→ **Enlaza con:** 5.4.1 · 1.4
**Tiempo:** 1 sesión

---

### 📋 Bloque para el LLM

```text
BLOQUE N02

OBJETIVO
Implementar un perceptrón desde cero en NumPy y encontrar su límite con las manos.

RESTRICCIÓN ESPECIAL
NumPy puro. Nada de PyTorch en este notebook.

QUÉ CONSTRUIR
1. Perceptrón con la regla de actualización clásica (no descenso de gradiente).
2. Entrenamiento sobre datos linealmente separables (dos_gaussianas de N01).
3. Visualización de la frontera de decisión ÉPOCA A ÉPOCA, en una animación o en una
   rejilla de subplots.

QUÉ ROMPO AQUÍ
XOR. Entrena sobre los 4 puntos de XOR y deja correr 1000 épocas.
- Dibuja la frontera final e intenta separar los puntos con una recta.
- Grafica el número de errores por época: verás que oscila y nunca llega a cero.
- Explica en markdown por qué NINGUNA recta puede separarlos.

MEDIR
- Épocas hasta convergencia en el caso separable.
- Errores por época en XOR (no converge).

CRITERIO DE TERMINADO
Puedo explicar sin mirar por qué el perceptrón falla en XOR y qué haría falta para
arreglarlo.

CONTEXTO HISTÓRICO
Añade una celda markdown corta: este fallo concreto frenó el campo casi 20 años. No es
anécdota, es la razón de que existan las capas ocultas.
```

---

## N03 · Backprop a mano

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿sé calcular un gradiente sin `autograd`?

**Construyes:** un MLP de dos capas en NumPy. Forward, pérdida, backward escrito por ti,
actualización. Resuelve XOR.

**Qué rompes:** inicializa todos los pesos a cero. Observa que **ninguna neurona se
diferencia de las demás**: la simetría no se rompe nunca y la red no aprende.

**El objetivo real:** hacerlo **una vez** para no volver a hacerlo nunca. A partir de
aquí `autograd` deja de ser magia.

→ **Enlaza con:** 3.2 · 2.2
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N03

OBJETIVO
Escribir backpropagation con mis propias manos, una vez, para no volver a hacerlo nunca.

RESTRICCIÓN ESPECIAL
NumPy puro. Prohibido autograd.

QUÉ CONSTRUIR
1. MLP de 2 capas (entrada → oculta → salida) con activación a elegir.
2. Forward paso a paso, guardando las activaciones intermedias.
3. Backward escrito por mí: derivada de la pérdida, regla de la cadena capa a capa.
4. Actualización de pesos.
5. Entrenar sobre XOR hasta resolverlo. (Es la respuesta a N02.)

ESTRUCTURA
Escribe el backward de forma EXPLÍCITA y verbosa, con una variable por gradiente
intermedio y comentarios que indiquen a qué corresponde cada término. La legibilidad
importa más que la elegancia.

QUÉ ROMPO AQUÍ
Inicializa TODOS los pesos a cero y entrena.
- Observa que la red no aprende nada.
- Imprime los pesos de la capa oculta tras varias épocas y comprueba que todas las
  neuronas son IDÉNTICAS.
- Explica en markdown la ruptura de simetría.

MEDIR
- Curva de pérdida en XOR, con inicialización buena y con ceros.
- Las activaciones de la capa oculta: ¿qué representa cada neurona?

CRITERIO DE TERMINADO
XOR resuelto, y sé decir qué hace cada línea del backward.
```

---

## N04 · Verificación del gradiente ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿mi backprop es correcto?

**Construyes:** comprobación numérica del gradiente. Compara tu derivada analítica con
$(f(x+\epsilon) - f(x-\epsilon)) / 2\epsilon$.

**Qué rompes:** introduce a propósito un error de signo en una capa. Comprueba que la red
**sigue entrenando y bajando la pérdida** — peor, pero baja. Ese es el punto.

**La lección:** un bug en backprop no da error. Da un modelo mediocre que parece
funcionar. Es el modo de fallo más caro que existe.

→ **Enlaza con:** 3.2 · 3.8
**Tiempo:** 1 sesión

---

### 📋 Bloque para el LLM

```text
BLOQUE N04

OBJETIVO
Comprobar que mi backprop de N03 es correcto, y descubrir que un bug ahí NO da error.

QUÉ CONSTRUIR
1. Función de comprobación numérica: para cada parámetro, calcular
   (f(x+eps) - f(x-eps)) / (2*eps) y compararlo con el gradiente analítico.
2. Reportar el error relativo por capa. Un error relativo < 1e-7 indica implementación
   correcta.
3. Aplicarlo al MLP de N03.

QUÉ ROMPO AQUÍ (lo más importante del notebook)
Introduce un error de SIGNO en el gradiente de una capa. Y entonces:
- Comprueba que la verificación numérica lo detecta inmediatamente.
- PERO entrena igualmente el modelo con el bug y grafica la curva de pérdida.
- Verás que la pérdida BAJA, más lentamente y peor, pero baja.
- Escribe en markdown la lección: un bug en backprop no lanza excepción, da un modelo
  mediocre que parece funcionar.

Prueba también con otros bugs plausibles:
- olvidar multiplicar por la derivada de la activación
- transponer mal una matriz
- usar la activación en vez de la preactivación

MEDIR
Error relativo por capa, con implementación correcta y con cada bug.

CRITERIO DE TERMINADO
Tengo una función de verificación reutilizable, y he visto con mis ojos que un modelo
con backprop roto sigue "entrenando".
```

---

## N05 · El mismo modelo en PyTorch

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿coinciden mis gradientes con los del framework?

**Construyes:** la traducción del MLP de N03 a PyTorch. Compara gradientes numéricamente
en el mismo lote con los mismos pesos.

**Qué rompes:** olvida `optimizer.zero_grad()`. Observa cómo los gradientes se acumulan
entre pasos y el entrenamiento se comporta de forma extraña sin dar ningún error.

→ **Enlaza con:** 3.1
**Tiempo:** 1 sesión

### 📋 Bloque para el LLM

```text
BLOQUE N05

OBJETIVO
Traducir el MLP de N03 a PyTorch y comprobar que los gradientes coinciden con los míos.

QUÉ CONSTRUIR
1. El mismo MLP en PyTorch, con la misma arquitectura.
2. Copiar los pesos de mi implementación NumPy a la de PyTorch.
3. Hacer un forward+backward con el MISMO lote en las dos.
4. Comparar gradientes parámetro a parámetro y reportar la diferencia máxima.
5. Explicar el bucle de entrenamiento canónico de PyTorch, indicando qué hace cada línea:
   zero_grad / forward / loss / backward / step.

QUÉ ROMPO AQUÍ
Quita `optimizer.zero_grad()` del bucle.
- Entrena y grafica la curva.
- Imprime la norma del gradiente por paso: verás que crece sin parar porque se acumula.
- Explica por qué PyTorch acumula gradientes por defecto (pista: acumulación de
  gradiente para simular lotes grandes) y por qué eso obliga a limpiarlos a mano.

Prueba también:
- llamar a `backward()` dos veces sin `zero_grad`
- olvidar `model.train()` / `model.eval()` con dropout activo

MEDIR
Diferencia máxima entre mis gradientes y los de PyTorch (debe ser ~1e-7).

CRITERIO DE TERMINADO
Autograd ha dejado de ser magia: sé qué está haciendo por debajo.
```


# BLOQUE 2 · Instrumentar

> **¿Puedo VER qué pasa por dentro?**

---

## N06 · La caja de instrumentos ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo ver si el modelo está aprendiendo, y dónde?

**Construyes** —y esto se queda en el arnés para siempre—:

| Instrumento | Qué revela |
|---|---|
| **Pérdida inicial esperada** | $\ln(K)$ para $K$ clases. Si no coincide, hay un bug antes de empezar |
| **Sobreajustar 10 muestras** | Si no llega a pérdida ~0, hay un bug. No es hiperparámetros |
| **Histograma de pesos por capa** | Antes / mitad / después. ¿Se mueven todas o solo las últimas? |
| **Norma del gradiente por capa** | ¿Llega la señal a las capas iniciales? |
| **Norma global por paso** | El detector precoz: su crecimiento precede al colapso |
| **Curva train + validación juntas** | El electrocardiograma |

**Qué rompes:** entrena con una capa congelada por accidente y comprueba que los
instrumentos lo detectan (su histograma no se mueve) aunque la pérdida baje.

**Criterio de terminado:** el test de sobreajuste a 10 muestras está automatizado y corre
con la suite.

→ **Enlaza con:** 3.8 · 3.1
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N06

OBJETIVO
Construir el instrumental de diagnóstico que se quedará en el arnés PARA SIEMPRE. Este es
el notebook que cambia mi forma de trabajar: dejo de entrenar a ciegas.

QUÉ CONSTRUIR
Seis instrumentos, cada uno como función reutilizable:

1. `perdida_inicial_esperada(n_clases)` → ln(K). Comparar con la real antes de entrenar.
   Si no coinciden, hay un bug antes de empezar.

2. `test_overfit(modelo, n=10)` → entrena sobre 10 muestras hasta pérdida ≈ 0.
   Si no lo consigue, hay un bug. Debe poder ejecutarse como test automático.

3. `histograma_pesos(modelo, momento)` → un histograma por capa. Llamarlo antes, a
   mitad y al final. Ponerlos en una rejilla comparable.

4. `normas_gradiente_por_capa(modelo)` → registrar en cada paso. Graficar en ESCALA
   LOGARÍTMICA, una línea por capa.

5. `norma_global(modelo)` → un escalar por paso. Es el detector precoz de divergencia.

6. `curvas(historial)` → train y validación en el MISMO gráfico, nunca separados.

INTEGRACIÓN
Todos deben poder activarse desde la config del arnés de N00, con un flag tipo
`diagnosticos=True`.

QUÉ ROMPO AQUÍ
Congela una capa intermedia por accidente (requires_grad=False) y entrena.
- La pérdida baja igualmente: el modelo compensa con las demás capas.
- PERO el histograma de esa capa no se mueve y su norma de gradiente es exactamente cero.
- Lección: la curva de pérdida no detecta este fallo. Los instrumentos sí.

CRITERIO DE TERMINADO
`test_overfit` está automatizado y corre con la suite de tests.
```

---

## N07 · Parar, tocar, seguir

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo intervenir en mitad del entrenamiento?

**Construyes:** un flujo de checkpoints y *hooks*. Y entonces juegas:

- Predice ejemplos a mano. ¿Cuáles falla? ¿Tienen algo en común?
- Pon a cero una capa. ¿Cuánto se degrada?
- Duplica la tasa a mitad de entrenamiento. ¿Qué pasa?
- Bifurca desde un checkpoint y compara las dos ramas.
- Inspecciona las activaciones de una capa oculta para dos entradas parecidas.

**Qué rompes:** todo lo que quieras. Es el notebook de manosear.

> **Este notebook no tiene criterio de terminado objetivo.** Está terminado cuando dejas
> de tener miedo a tocar un modelo entrenado. Es el que te falta.

→ **Enlaza con:** 3.8 · *Primeros pasos*
**Tiempo:** 2 sesiones, sin prisa

### 📋 Bloque para el LLM

```text
BLOQUE N07

OBJETIVO
Perder el miedo a intervenir en mitad de un entrenamiento. Es un notebook de
EXPLORACIÓN, no tiene un resultado concreto que alcanzar.

QUÉ CONSTRUIR
1. Sistema de checkpoints: guardar y restaurar estado completo (pesos, optimizador,
   época, semilla) en cualquier punto.
2. Hooks de PyTorch para capturar activaciones de capas intermedias sin modificar el
   modelo.
3. Función para bifurcar: cargar un checkpoint, cambiar algo, y seguir dos ramas en
   paralelo para compararlas.

LOS EXPERIMENTOS DE MANOSEO (uno por sección)
a) Entrena 5 épocas. Para. Predice 20 ejemplos a mano y examina cuáles falla. ¿Tienen
   algo en común? Visualízalos.
b) Pon a cero los pesos de una capa concreta. Mide cuánto se degrada. Repite con cada
   capa y haz una tabla: ¿qué capa es más crítica?
c) Duplica la tasa de aprendizaje a mitad de entrenamiento. Sigue 5 épocas. Grafica el
   antes y el después en la misma curva.
d) Desde un checkpoint, lanza dos ramas con configuraciones distintas y compáralas.
e) Captura las activaciones de la capa oculta para dos entradas muy parecidas y para dos
   muy distintas. ¿Se parecen las representaciones?
f) Añade ruido gaussiano a los pesos de una capa, con magnitud creciente. ¿Cuánto ruido
   aguanta el modelo antes de romperse?

QUÉ ROMPO AQUÍ
Todo. Es el notebook de romper cosas sin objetivo.

CRITERIO DE TERMINADO
Subjetivo: cuando pueda coger cualquier modelo entrenado y responder "¿qué pasa aquí
dentro?" sin buscar un tutorial.
```


# BLOQUE 3 · Romper a propósito

> **¿Reconozco los fallos?**

---

## N08 · Patologías del gradiente

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cómo se ve un gradiente enfermo?

**Provocas:** red de 20 capas con sigmoides (desvanecimiento) · pesos inicializados
grandes (explosión) · ReLU con tasa alta (neuronas muertas).

**Mides:** norma del gradiente **por capa**, en escala logarítmica. La caída de órdenes
de magnitud según retrocedes es lo que hay que ver con los ojos.

**Arreglas:** ReLU/GELU, inicialización adecuada, conexiones residuales, recorte.

→ **Enlaza con:** 3.8 · 3.4
**Tiempo:** 1–2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N08

OBJETIVO
Provocar y reconocer la firma de cada patología del gradiente.

QUÉ CONSTRUIR
Un modelo configurable en profundidad y activación, para poder barrer ambas.

LOS TRES FALLOS (uno por sección)
1. GRADIENTE DESVANECIDO
   Red de 20 capas con sigmoide. Grafica la norma del gradiente POR CAPA en escala log.
   Debe verse la caída de órdenes de magnitud al retroceder.
   Arreglo: cambiar a ReLU/GELU. Volver a graficar y comparar.

2. GRADIENTE EXPLOTANDO
   Inicializa los pesos con desviación grande. Observa la norma dispararse y luego NaN.
   Arreglo: inicialización adecuada (He/Xavier) y recorte por norma. Mide el efecto de
   cada uno por separado.

3. NEURONAS MUERTAS (dead ReLU)
   ReLU con tasa de aprendizaje alta. Cuenta qué porcentaje de neuronas queda a cero
   permanentemente para todas las entradas.
   Arreglo: LeakyReLU o GELU. Comparar el porcentaje.

Y EL ARREGLO ESTRUCTURAL
Añade conexiones residuales a la red de 20 capas con sigmoides. Compara las normas de
gradiente antes y después. Es la demostración de por qué existen las ResNets.

MEDIR
En todos los casos: norma del gradiente por capa, en escala logarítmica, antes y después
del arreglo. Es LA gráfica del notebook.

CRITERIO DE TERMINADO
Puedo mirar una gráfica de normas por capa y decir qué le pasa al modelo.
```

---

## N09 · Tasa de aprendizaje y optimizadores

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cuánto importa el optimizador comparado con la tasa?

**Provocas:** tasa 100× (NaN) · tasa 1000× menor (plano) · sin normalizar entradas.

**Construyes:** un *LR range test*, y compara SGD, SGD+momento, Adam y AdamW **con el
mismo presupuesto de ajuste** para cada uno.

**Qué rompes:** ajusta Adam a fondo y deja SGD con valores por defecto. Comprueba que
"ganas" — y que esa conclusión no vale nada. Es cómo se fabrica una mejora falsa.

→ **Enlaza con:** 3.3 · 3.9
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N09

OBJETIVO
Entender que la tasa de aprendizaje pesa más que la elección de optimizador, y ver cómo
se fabrica una comparación falsa.

QUÉ CONSTRUIR
1. LR RANGE TEST: entrenar unos cientos de pasos aumentando la tasa exponencialmente, y
   graficar pérdida frente a tasa. Identificar el rango útil.
2. Barrido de tasas: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5. Superponer las cinco curvas.
3. Comparativa de optimizadores: SGD, SGD+momento, Adam, AdamW.

LOS FALLOS
- Tasa 100× la buena → NaN. Anota en qué paso ocurre.
- Tasa 1000× menor → curva plana. Importante: distinguir esto de "no aprende por bug",
  y explicar cómo se distinguen.
- Sin normalizar las entradas → convergencia mucho peor con el mismo modelo. Compara.

QUÉ ROMPO AQUÍ (lo más importante)
Comparación injusta a propósito:
- Ajusta Adam probando 20 configuraciones de tasa.
- Deja SGD con su valor por defecto, sin ajustar.
- Reporta que "Adam gana".
- Luego dale a SGD el mismo presupuesto de 20 configuraciones y vuelve a comparar.
- Escribe en markdown la lección: acabo de fabricar una mejora falsa, y así es como
  aparecen en la literatura.

MEDIR
- Tabla: optimizador × presupuesto de ajuste → mejor resultado.
- Curvas superpuestas de las cinco tasas.

CRITERIO DE TERMINADO
Tengo un LR range test reutilizable, y he visto cómo una comparación desigual invierte
la conclusión.
```

---

## N10 · Regularización, y el experimento incómodo ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿la regularización impide memorizar?

**Provocas:** ⭐ **entrena con etiquetas barajadas al azar** (CIFAR-10 o tu dataset de
ruido de N01). Actívale weight decay, dropout y aumentación.

**Lo que verás:** las ajusta igual. La regularización **no lo impide**.

**Y después:** entrena el doble de épocas de lo razonable en un caso normal, buscando el
doble descenso. Si aparece, tu instinto sobre el *early stopping* cambia para siempre.

> **Si solo haces un notebook del bloque 3, que sea este.** Reencuadra todo lo que creías
> sobre generalización.

→ **Enlaza con:** 3.4
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N10

OBJETIVO
Descubrir que la regularización NO impide memorizar, y que la curva de validación puede
subir y volver a bajar.

EL EXPERIMENTO PRINCIPAL (el más importante del bloque 3)
1. Coge un dataset de clasificación (CIFAR-10 pequeño, o el sintético de N01).
2. BARAJA LAS ETIQUETAS AL AZAR. Ahora no hay relación entre X e y.
3. Entrena una red con capacidad suficiente hasta pérdida de entrenamiento ≈ 0.
4. Repite activando weight decay, dropout y aumentación.
5. Comprueba que las ajusta IGUALMENTE.
6. Escribe la conclusión: la red siempre puede memorizar. La regularización no lo impide;
   lo que hace es sesgar QUÉ solución encuentra entre las que interpolan.

EL SEGUNDO EXPERIMENTO: DOBLE DESCENSO
1. Vuelve a las etiquetas correctas, pero mete un 15% de ruido de etiqueta.
2. Entrena MUCHAS más épocas de las razonables (10× lo que parece necesario).
3. Grafica train y validación juntas, en escala amplia.
4. Busca si el error de validación sube y luego vuelve a bajar.
5. Si aparece: marca en la gráfica dónde habría parado un early stopping ingenuo.

TAMBIÉN
Barrido de la intensidad de cada regularizador por separado (weight decay, dropout,
aumentación) sobre el problema normal, para ver cuánto aporta cada uno.

QUÉ ROMPO AQUÍ
Las etiquetas, literalmente. Es el punto del notebook.

CRITERIO DE TERMINADO
Puedo explicar por qué "regularizar para que no memorice" es una descripción incorrecta.
```

---

## N11 · Normalización

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué hace realmente BatchNorm?

**Provocas:** BatchNorm con lote de 2 · el mismo caso con GroupNorm · olvidar
`model.eval()` y ver métricas de validación absurdas.

**Y en el lado transformer** (déjalo apuntado para N19): pre-norm frente a post-norm, y
si el warmup deja de hacer falta.

→ **Enlaza con:** 3.4 · 3.8
**Tiempo:** 1 sesión

---

### 📋 Bloque para el LLM

```text
BLOQUE N11

OBJETIVO
Ver qué hace realmente la normalización y cuándo se rompe.

QUÉ CONSTRUIR
1. Una red con BatchNorm, entrenada con tamaños de lote: 128, 32, 8, 2.
   Grafica el rendimiento frente al tamaño de lote. Debe degradarse con lotes pequeños.
2. La misma red con GroupNorm, mismos tamaños de lote. Superpón las curvas.
   Conclusión: separar el efecto del tamaño de lote del efecto de la técnica.
3. Visualiza la distribución de activaciones por capa, con y sin normalización.

QUÉ ROMPO AQUÍ
Olvida `model.eval()` antes de evaluar, con BatchNorm activo.
- Reporta la métrica de validación con y sin el fallo.
- Explica qué cambia: en train usa estadísticas del lote, en eval usa medias móviles.
- Comprueba que el error es mayor cuanto más pequeño es el lote de validación.

TAMBIÉN
Comprueba si BatchNorm regulariza como efecto secundario: entrena con BatchNorm y con
GroupNorm en un problema propenso al sobreajuste y compara la brecha train/validación.

DEJAR APUNTADO
Pre-norm vs post-norm se prueba en N20, cuando haya transformer.

MEDIR
Rendimiento frente a tamaño de lote, una línea por técnica de normalización.
```

---

## N12 · Precisión y memoria

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿por qué no cabe, y por qué da NaN?

**Construyes:** calcula a mano el presupuesto de memoria de tu modelo y compáralo con lo
que reporta el sistema.

**Provocas:** NaN en fp16 por *underflow* de gradientes, y comprueba que en bf16 no pasa.

**Mides:** el intercambio de *gradient checkpointing* — cuánta memoria ahorras y cuánto
tiempo pagas. Y la utilización de GPU: si está por debajo del 80 %, el cuello es el
pipeline de datos.

→ **Enlaza con:** 3.5
**Tiempo:** 1 sesión

### 📋 Bloque para el LLM

```text
BLOQUE N12

OBJETIVO
Entender por qué un modelo no cabe, por qué da NaN y por qué va lento.

QUÉ CONSTRUIR
1. CALCULADORA DE MEMORIA: función que, dado un modelo y un tamaño de lote, estime
   memoria de pesos, gradientes, estados del optimizador y activaciones.
   Compárala con la medición real. Explica las discrepancias.
2. Comparativa fp32 / bf16 / fp16 (con autocast) del mismo entrenamiento:
   tiempo, memoria pico, resultado final.
3. Gradient checkpointing: activar y medir el intercambio memoria/tiempo.
4. Acumulación de gradiente: simular un lote grande y verificar que da resultados
   equivalentes a usarlo de verdad.

QUÉ ROMPO AQUÍ
- Provoca NaN en fp16 (underflow de gradientes) y comprueba que en bf16 NO ocurre.
  Explica la diferencia de rango dinámico.
- Usa acumulación de gradiente SIN reajustar la tasa de aprendizaje, y compara con el
  caso reajustado. Es el error más común al activarla.

TAMBIÉN
Perfila el bucle: mide qué porcentaje del tiempo se va en datos, en cómputo y en
sincronización. Si la utilización de CPU/GPU es baja, localiza el cuello.

MEDIR
Tabla: precisión × (tiempo, memoria pico, métrica final).
```


# BLOQUE 4 · Medir

> **¿Me creo mis propios resultados?**

---

## N13 · Tu ruido de fondo ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cuánto se mueven mis resultados solo con la semilla?

**Construyes:** el mismo experimento, cinco semillas, intervalo de confianza.

**El resultado es un número que usarás durante años:** por debajo de esa dispersión,
cualquier diferencia que observes **no es un resultado**.

**Extra incómodo:** comprueba si fijar la semilla basta, o si el no-determinismo de las
operaciones sobre GPU sigue moviendo el resultado.

> **Es el notebook más barato del itinerario y el que más te protege.**

→ **Enlaza con:** 3.9
**Tiempo:** 1 sesión

---

### 📋 Bloque para el LLM

```text
BLOQUE N13

OBJETIVO
Averiguar cuánto se mueven mis resultados solo por azar. Es el número más importante de
todo el itinerario.

QUÉ CONSTRUIR
1. Ejecuta EXACTAMENTE el mismo experimento con 5 semillas distintas (luego con 10).
2. Reporta: media, desviación, mínimo, máximo y un intervalo de confianza por bootstrap.
3. Grafica los resultados individuales, no solo la media. Un gráfico de puntos.
4. Función reutilizable `run_seeds(config, n_seeds)` integrada en el arnés de N00.

EL EXPERIMENTO REVELADOR
Coge dos configuraciones que difieran poco (por ejemplo, dos tasas de aprendizaje
cercanas). Ejecuta ambas con 5 semillas.
- ¿Se solapan los intervalos?
- Si se solapan, la diferencia NO es un resultado.
- Escribe cuál es el tamaño mínimo de diferencia que puedo considerar real.

QUÉ ROMPO AQUÍ
- Reporta el resultado de UNA sola semilla, la mejor de las cinco. Compáralo con la
  media. Esa diferencia es lo que se gana haciendo cherry-picking.
- Comprueba si fijar la semilla basta: repite el mismo experimento dos veces con la MISMA
  semilla y verifica si los resultados son idénticos bit a bit. En GPU probablemente no
  lo sean; investiga por qué y documéntalo.

MEDIR
La dispersión. Anótala en la bitácora en grande: es tu umbral de credibilidad.
```

---

## N14 · Métricas que engañan

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿mi métrica ve el problema?

**Construyes:** un problema desbalanceado (de N01). Compara accuracy, matriz de
confusión, ROC-AUC y precisión-exhaustividad.

**Y las líneas base tontas:** clase mayoritaria, modelo aleatorio, regresión logística
sobre features crudas. Si tu red no las bate claramente, no has demostrado nada.

**Extra:** mide la **calibración**. Un modelo con 90 % de accuracy que dice "estoy 99 %
seguro" en todo es un modelo roto que la accuracy no detecta.

→ **Enlaza con:** 3.9
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N14

OBJETIVO
Descubrir que la métrica que miro puede ser ciega al problema.

QUÉ CONSTRUIR
1. Genera un problema con desbalanceo severo (95/5) usando N01.
2. Entrena un modelo y calcula: accuracy, precisión, exhaustividad, F1, ROC-AUC y
   PR-AUC. Ponlas todas en una tabla.
3. Muestra la matriz de confusión al lado.
4. Compara ROC-AUC y PR-AUC: la primera se ve bien y la segunda mal. Explica por qué.

LAS LÍNEAS BASE TONTAS (obligatorio)
Calcula el resultado de:
- predecir siempre la clase mayoritaria
- predecir al azar según la distribución de clases
- una regresión logística sobre las features crudas
Si mi red no bate claramente a las tres, no he demostrado nada.

CALIBRACIÓN
1. Grafica el diagrama de fiabilidad: confianza predicha frente a exactitud real.
2. Calcula el error de calibración esperado (ECE).
3. Aplica temperature scaling y vuelve a graficar.
4. Conclusión: un modelo con 90% de accuracy que dice "estoy 99% seguro" en todo está
   roto, y la accuracy no lo detecta.

QUÉ ROMPO AQUÍ
Entrena sobre el dataset de ruido_puro de N01 y reporta todas las métricas. Comprueba
qué métricas siguen dando números "razonables" sobre datos sin ninguna señal.
```

---

## N15 · Fuga de datos

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿mi split mide lo que creo?

**Provocas, uno por uno:**

- Normalizar antes de partir train/test
- Duplicados entre las dos particiones
- Split aleatorio sobre datos con estructura de grupo o temporal
- Seleccionar features mirando el conjunto completo

**Mides** cuánto sube el resultado con cada fuga. Verás subidas que en un paper
parecerían una contribución.

**Carril B (real):** hazlo también sobre un dataset descargado, donde no controlas la
generación. La diferencia es instructiva.

→ **Enlaza con:** 3.0 · 3.9
**Tiempo:** 2 sesiones

### 📋 Bloque para el LLM

```text
BLOQUE N15

OBJETIVO
Provocar las cuatro fugas de datos más comunes y medir cuánto inflan el resultado.

LAS CUATRO FUGAS (una sección cada una)
1. NORMALIZAR ANTES DE PARTIR
   Calcula media y desviación sobre el dataset completo, luego parte. Compara con
   calcularlas solo en train. Mide la diferencia en la métrica de test.

2. DUPLICADOS ENTRE PARTICIONES
   Usa el modificador `duplicados` de N01. Barre el porcentaje (0%, 5%, 20%) y grafica
   la métrica de test frente a ese porcentaje.

3. SPLIT ALEATORIO CON ESTRUCTURA DE GRUPO
   Usa el modificador `grupos`. Compara split aleatorio contra split por grupo.
   La diferencia es lo que estabas midiendo de más.

4. SELECCIÓN DE FEATURES MIRANDO TODO EL DATASET
   Selecciona las k mejores features usando todo el dataset, luego parte y entrena.
   Compara con hacerlo solo en train.

CARRIL B · DATOS REALES
Repite las fugas 1 y 3 sobre un dataset tabular real pequeño descargado. La diferencia
respecto al sintético es instructiva: en el real no controlas la generación.

EL EXPERIMENTO DE LA ARITMÉTICA
Usando el lenguaje de N01, compara los tres splits:
- aleatorio (47+38 en train, 38+47 en test)
- por_resultado
- por_rango
Comprueba cuánto cae el rendimiento en cada uno y explica qué mide cada split.

MEDIR
Tabla: tipo de fuga → métrica con fuga → métrica sin fuga → inflación.
Escribe cuáles de esas inflaciones parecerían una contribución publicable.
```


# BLOQUE 5 · Arquitecturas

> **¿Qué aporta cada familia?**

---

## N16 · MLP como aproximador universal

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué puede y qué no puede un MLP?

**Construyes:** aproxima funciones 1D cada vez más retorcidas. Visualiza cómo la red las
compone a trozos.

**Qué rompes:** dale una imagen desplazada dos píxeles. Observa que **no la reconoce**.
Ese fallo es exactamente el argumento de la sección siguiente.

→ **Enlaza con:** 5.4.1
**Tiempo:** 1 sesión

---

### 📋 Bloque para el LLM

```text
BLOQUE N16

OBJETIVO
Ver qué puede aproximar un MLP y dónde está su límite estructural.

QUÉ CONSTRUIR
1. Aproximar funciones 1D de complejidad creciente: recta, seno, seno de frecuencia alta,
   función escalonada, función con discontinuidad.
2. Para cada una: grafica la función real y la aproximación, superpuestas.
3. Barre el número de neuronas de la capa oculta (2, 8, 32, 128) y muestra cómo mejora
   la aproximación.
4. VISUALIZA cómo la red compone la función a trozos: grafica la salida de cada neurona
   oculta por separado y su suma ponderada.

QUÉ ROMPO AQUÍ
Entrena un MLP para clasificar imágenes pequeñas (formas de N01).
- Evalúa con las mismas imágenes DESPLAZADAS 2 píxeles.
- Comprueba que el rendimiento se desploma.
- Explica: el MLP no tiene ninguna noción de que un píxel esté al lado de otro.
- Ese fallo es exactamente el argumento de la CNN, que viene en el notebook siguiente.

MEDIR
- Error de aproximación frente a número de neuronas.
- Rendimiento con imágenes originales frente a desplazadas.
```

---

## N17 · CNN: el sesgo inductivo que se ve

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cuánto vale compartir pesos?

**Construyes:** CNN pequeña sobre imágenes. Visualiza los filtros de la primera capa y
los mapas de activación.

**Qué rompes:** ⭐ **quita el compartir pesos** — sustituye la convolución por capas
densas con el mismo número de parámetros. Compara con pocos datos y con muchos.

**La lección:** la arquitectura **es** regularización (→ 3.4, mando ⑤), y se puede medir.

**Carril B:** un dataset real pequeño (CIFAR-10) para ver el fenómeno sin trampa.

→ **Enlaza con:** 5.4.2 · 3.4
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N17

OBJETIVO
Medir cuánto vale compartir pesos, en lugar de aceptarlo como dogma.

QUÉ CONSTRUIR
1. CNN pequeña sobre imágenes (formas sintéticas de N01, y CIFAR-10 como carril B).
2. Visualiza los filtros aprendidos de la primera capa como imágenes.
3. Visualiza los mapas de activación para una entrada concreta, capa a capa.
4. Comprueba la invariancia a traslación que le faltaba al MLP de N16.

EL EXPERIMENTO CENTRAL (lo más importante)
Compara CNN contra MLP con EL MISMO NÚMERO DE PARÁMETROS:
- con 500 ejemplos de entrenamiento
- con 5.000
- con 50.000
Grafica ambas curvas frente a cantidad de datos.
La hipótesis: la ventaja de la CNN es enorme con pocos datos y se estrecha con muchos.
Explica por qué: el sesgo inductivo sustituye a los datos que no tienes.

QUÉ ROMPO AQUÍ
Sustituye la convolución por una capa densa "local" que NO comparte pesos entre
posiciones (locally connected). Mismo campo receptivo, sin compartir.
- Compara las tres: CNN, locally-connected, MLP.
- Aísla así exactamente cuánto aporta COMPARTIR, separado de la localidad.

MEDIR
Tabla de tres arquitecturas × tres tamaños de dataset.
```

---

## N18 · RNN y LSTM: memoria y explosión ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cómo se procesa una secuencia, y por qué es inestable?

**Construyes:** una RNN sobre secuencias sintéticas de tu lenguaje. Tarea de memoria
larga: recordar un token del principio.

**Qué rompes:** haz explotar el gradiente. Míralo en la norma. Aplica recorte y ve la
diferencia. Compara RNN simple contra LSTM en dependencias largas.

→ **Enlaza con:** 5.4.3 · 3.8
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N18

OBJETIVO
Procesar secuencias, y ver por qué las RNN simples fallan en dependencias largas.

DATOS
El lenguaje aritmético de N01, nivel 1 (2-3 dígitos con acarreo): 47+38=85

QUÉ CONSTRUIR
1. RNN simple, escrita de forma explícita (el bucle temporal a la vista).
2. LSTM (puedes usar la de PyTorch, pero explica qué hace cada puerta).
3. Entrenar ambas en la tarea de suma. Comparar exactitud por número de dígitos.

EL EXPERIMENTO DEL ACARREO
La suma con acarreo es una dependencia de largo alcance: un dígito de la derecha afecta
a otro de la izquierda.
- Mide la exactitud desglosada por: problemas CON acarreo y SIN acarreo.
- La diferencia entre esas dos cifras es la medida directa del problema.
- Compara RNN vs LSTM en esa métrica concreta.

EL EXPERIMENTO DE LA REPRESENTACIÓN (muy revelador)
Entrena la misma arquitectura con la respuesta en orden normal y en orden INVERTIDO
(usa `formatear(modo='invertido')` de N01): 47+38=58 en vez de 85.
- Compara curvas de aprendizaje.
- El invertido debería aprenderse mucho más rápido, porque el acarreo fluye en el mismo
  sentido que la generación.
- Lección: la representación del dato importa tanto como la arquitectura.

QUÉ ROMPO AQUÍ
Haz explotar el gradiente en la RNN con secuencias largas.
- Grafica la norma del gradiente por paso temporal.
- Aplica recorte por norma y compara.

MEDIR
Exactitud por longitud de secuencia y por presencia de acarreo.
```

---

## N19 · Atención desde cero

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué es realmente Q, K, V?

**Construyes:** atención de producto escalar a mano, sobre una tarea sintética donde
**sepas a qué debería atender** (copiar un token concreto, buscar un delimitador).

**Qué rompes:** quita el escalado por $\sqrt{d_k}$ y observa el softmax saturarse. Quita
el enmascarado causal y comprueba que el modelo "hace trampa" mirando el futuro.

**Mides:** los mapas de atención. ¿Miran lo que crees?

→ **Enlaza con:** 5.4.4
**Tiempo:** 2–3 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N19

OBJETIVO
Entender Q, K, V implementándolos, sobre una tarea donde SÉ a qué debería atender.

RESTRICCIÓN ESPECIAL
Implementa la atención a mano con operaciones básicas. Nada de `nn.MultiheadAttention`
en la primera parte.

DATOS
Dos tareas sintéticas donde la atención correcta es conocida de antemano:
a) COPIA SELECTIVA: en una secuencia, copiar el token que va después de un marcador.
   El modelo DEBE atender al marcador.
b) El lenguaje aritmético nivel 1: al generar cada dígito del resultado, debería atender
   a los dígitos correspondientes de los operandos.

QUÉ CONSTRUIR
1. Atención de producto escalar escalado, paso a paso, con variables intermedias
   visibles: scores, softmax, salida.
2. VISUALIZACIÓN DE MAPAS DE ATENCIÓN: matriz de calor de qué posición atiende a cuál.
3. Comprobar en la tarea (a) si el mapa coincide con lo esperado.
4. En la tarea (b), examinar si al generar el dígito de las unidades atiende a las
   unidades de ambos operandos.

QUÉ ROMPO AQUÍ
1. Quita el escalado por sqrt(d_k). Grafica la distribución de los scores antes del
   softmax y observa la saturación: el softmax se vuelve casi one-hot y el gradiente
   muere. Compara curvas de entrenamiento.
2. Quita el enmascarado causal en una tarea autoregresiva. Comprueba que el modelo
   alcanza una pérdida sospechosamente baja: está mirando el futuro. Es una fuga de datos
   arquitectónica.

MEDIR
Mapas de atención como imágenes, comparados con el patrón esperado.
```

---

## N20 · Transformer mínimo

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo montar el bloque completo?

**Construyes:** atención multi-cabeza + MLP + residuales + normalización. Pequeño, sobre
tu lenguaje sintético.

**Qué rompes:** ⭐ **pre-norm frente a post-norm.** Entrena ambas con tasa alta y sin
warmup. Verás que una diverge y la otra no. Es el experimento que cierra N11.

**Y también:** quita las residuales en un modelo profundo y mira el gradiente morir.

→ **Enlaza con:** 5.4.4 · 3.4
**Tiempo:** 3 sesiones

### 📋 Bloque para el LLM

```text
BLOQUE N20

OBJETIVO
Montar el bloque transformer completo y probar la decisión de diseño que más importa.

DATOS
Lenguaje aritmético, niveles 1 y 2.

QUÉ CONSTRUIR
1. Bloque transformer completo: atención multi-cabeza + MLP + residuales + normalización
   + embeddings posicionales.
2. Modelo pequeño: 2-4 capas, dimensión 64-128, 2-4 cabezas. Debe entrenar en minutos.
3. Entrenar en la tarea aritmética y comparar contra la LSTM de N18.

EL EXPERIMENTO CENTRAL: PRE-NORM VS POST-NORM
1. Implementa las dos variantes (normalización dentro del bloque residual y fuera).
2. Entrena ambas con tasa ALTA y SIN warmup.
3. Observa que post-norm diverge y pre-norm no.
4. Repite post-norm CON warmup y comprueba que ahora sí converge.
5. Grafica la norma del gradiente por capa en la inicialización para ambas variantes.
   Debería verse que en post-norm los gradientes son grandes cerca de la salida.
6. Conclusión: la necesidad de warmup no es una propiedad del optimizador, es
   consecuencia de dónde pusiste la normalización.

QUÉ ROMPO AQUÍ (además de lo anterior)
- Quita las conexiones residuales en un modelo de 8 capas. Mira el gradiente morir.
- Quita los embeddings posicionales. Comprueba que el modelo no puede resolver la suma:
  sin posición, "47+38" y "74+83" son el mismo conjunto de tokens.

MEDIR
Curvas de las 4 combinaciones (pre/post × con/sin warmup) en el mismo gráfico.
```


# BLOQUE 6 · Formas de aprender

> **¿De dónde puede salir la señal?**

---

## N21 · Auto-supervisado: el modelo base

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué aprende un modelo que solo predice el siguiente token?

**Construyes:** entrena tu transformer de N20 sobre el corpus del lenguaje sintético.
Predicción del siguiente token, sin una sola etiqueta.

**Mides:** ¿aprendió las reglas de la gramática? Genera y verifica cuántas salidas son
sintácticamente válidas.

**El momento clave:** hazle una **pregunta** al modelo base y observa que **autocompleta
en vez de responder**. Ese fallo justifica el notebook siguiente.

→ **Enlaza con:** 4.3
**Tiempo:** 3 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N21

OBJETIVO
Entrenar un modelo base con predicción del siguiente token, y ver qué aprende sin una
sola etiqueta.

DATOS
Corpus de expresiones aritméticas correctas, nivel 2. Formato CRUDO, sin estructura de
pregunta-respuesta: "3+4*2=11\n12-5=7\n..."

QUÉ CONSTRUIR
1. Entrenar el transformer de N20 con objetivo autoregresivo sobre el corpus.
2. Generación con muestreo (temperatura, top-k).
3. Métricas de evaluación PROPIAS de esta fase:
   - % de salidas SINTÁCTICAMENTE válidas (parseable)
   - % de salidas SEMÁNTICAMENTE correctas (el = es cierto)
   - perplejidad en un conjunto de retención

EL EXPERIMENTO CLAVE (el que justifica N25)
Hazle una PREGUNTA en lenguaje natural o en formato de instrucción:
   "¿cuánto es 12+34?"
Observa que AUTOCOMPLETA con otra expresión en vez de responder.
Documenta la salida literal. Ese fallo es el argumento entero del SFT.

TAMBIÉN
- Barre el tamaño del corpus (1k, 10k, 100k expresiones) y grafica corrección frente a
  cantidad de datos.
- Compara tokenización a nivel de carácter frente a dígitos agrupados. Explica por qué
  la agrupada empeora la aritmética.

QUÉ ROMPO AQUÍ
Entrena con un corpus que contenga un 20% de expresiones INCORRECTAS (3+4=8). Mide
cuánto baja la corrección del modelo. Es la versión aritmética del ruido de etiqueta.

MEDIR
% válidas y % correctas, en varios niveles de dificultad.
```

---

## N22 · Contrastivo y el colapso

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cómo se aprende sin reconstruir nada?

**Construyes:** un método contrastivo simple sobre datos sintéticos con "vistas" (dos
perturbaciones del mismo ejemplo).

**Qué rompes:** ⭐ **quita los negativos.** Observa la pérdida caer a cero mientras todas
las representaciones convergen al mismo vector. Media hora, y enseña más que tres papers.

**Extra:** busca el colapso **dimensional** mirando el espectro de valores singulares de
los embeddings. Es más sutil y la pérdida no lo delata.

→ **Enlaza con:** 4.3 · 3.8
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N22

OBJETIVO
Aprender representaciones sin reconstruir nada, y ver el fallo característico del método.

DATOS
Dos opciones, elige la más simple de implementar:
a) Imágenes sintéticas de N01 con dos aumentaciones distintas de la misma imagen.
b) Expresiones aritméticas equivalentes como "vistas" (3+4 y 4+3 son la misma cosa).

QUÉ CONSTRUIR
1. Encoder + cabeza de proyección.
2. Pérdida contrastiva tipo InfoNCE, con negativos dentro del lote.
3. Entrenar y evaluar mediante sondeo lineal (una capa lineal sobre representaciones
   congeladas).

EL EXPERIMENTO CENTRAL: EL COLAPSO
1. Quita los ejemplos negativos: deja SOLO el término que acerca las vistas positivas.
2. Entrena.
3. Observa que la pérdida cae a cero rápidamente.
4. Comprueba que TODAS las representaciones convergen al mismo vector: calcula la
   desviación estándar de los embeddings y verás que tiende a cero.
5. Grafica una proyección 2D: todos los puntos encima del mismo punto.

EL COLAPSO SUTIL
Calcula el espectro de valores singulares de la matriz de embeddings.
- En un modelo sano, los valores decaen suavemente.
- En colapso dimensional, la mayoría son casi cero: los embeddings viven en un
  subespacio pequeño.
- Este colapso NO se ve en la pérdida. Es el punto.

TAMBIÉN
Implementa una de las soluciones sin negativos (stop-gradient asimétrico tipo BYOL, o
un término de varianza tipo VICReg) y comprueba que evita el colapso.

MEDIR
Desviación estándar de los embeddings y espectro de valores singulares, con y sin
negativos.
```

---

## N23 · Transferencia y sondeo lineal

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cuánto valen las representaciones del modelo base?

**Construyes:** congela el tronco de N21, entrena solo una cabeza lineal encima con muy
pocas etiquetas. Compara contra entrenar desde cero.

**Mides:** la curva de rendimiento frente a número de etiquetas. El hueco entre las dos
curvas **es el valor del preentrenamiento**, cuantificado.

**Carril B:** repítelo con un modelo preentrenado real y un dataset pequeño.

→ **Enlaza con:** 3.7 · 4.3
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N23

OBJETIVO
Cuantificar cuánto valen las representaciones del modelo base.

QUÉ CONSTRUIR
1. Congela el tronco del modelo de N21.
2. Entrena SOLO una capa lineal encima, para una tarea nueva (por ejemplo: predecir si
   el resultado es par, o predecir el número de dígitos del resultado).
3. Compara contra el mismo modelo entrenado DESDE CERO en esa tarea.

LA CURVA QUE IMPORTA
Repite ambos con 10, 50, 100, 500, 5.000 ejemplos etiquetados.
Grafica las dos curvas juntas: rendimiento frente a número de etiquetas.
EL HUECO ENTRE LAS DOS CURVAS ES EL VALOR DEL PREENTRENAMIENTO, cuantificado.

TAMBIÉN
1. Sondeo lineal por capa: entrena una cabeza lineal sobre CADA capa intermedia por
   separado. ¿En qué capa vive la información útil? Grafica rendimiento frente a
   profundidad de la capa.
2. Compara sondeo lineal contra fine-tuning completo, midiendo también el tiempo.

CARRIL B · DATOS REALES
Repítelo con un modelo preentrenado pequeño de Hugging Face y un dataset de texto
pequeño. La estructura es idéntica.

QUÉ ROMPO AQUÍ
Haz fine-tuning completo con una tasa de aprendizaje alta (la del preentrenamiento) y
comprueba que destruye las representaciones: el resultado es PEOR que el sondeo lineal.
Es el olvido catastrófico, provocado a propósito.

MEDIR
Curva de rendimiento frente a etiquetas, tres condiciones: desde cero, sondeo lineal,
fine-tuning.
```

---

## N24 · LoRA ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo ajustar tocando el 0,1 % de los pesos?

**Construyes:** LoRA sobre tu transformer. Compara contra fine-tuning completo:
rendimiento, memoria y tiempo.

**Mides:** barre el rango $r$. ¿A partir de qué valor deja de mejorar?

**Qué rompes:** aplica LoRA solo a algunas proyecciones y compara con aplicarlo a todas.

> **Este notebook es la puerta a tu tesis:** lo que viaja por la red en un federado es el
> adaptador, no el modelo.

→ **Enlaza con:** 3.7 · 13.4
**Tiempo:** 2 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N24

OBJETIVO
Ajustar un modelo tocando el 0,1% de sus pesos, y entender exactamente qué se está
aprendiendo.

QUÉ CONSTRUIR
1. Implementa LoRA A MANO primero: para una capa lineal W, añade B@A con rango r,
   congelando W. Inicializa A aleatoria y B a cero (explica por qué B a cero).
2. Aplícalo al transformer de N21.
3. Entrena en una tarea nueva y compara contra fine-tuning completo:
   - parámetros entrenables
   - memoria pico
   - tiempo por época
   - rendimiento final

BARRIDOS
1. Rango r: 1, 2, 4, 8, 16, 32. Grafica rendimiento frente a r. ¿Dónde satura?
2. Qué módulos: solo Q y V, frente a todas las proyecciones lineales. Compara.
3. Alfa: comprueba el efecto de la escala.

FUSIÓN
Implementa la fusión: W' = W + B@A. Verifica que el modelo fusionado da EXACTAMENTE las
mismas salidas que el modelo con adaptador separado.

EL EXPERIMENTO QUE CONECTA CON MI TESIS
1. Entrena DOS adaptadores LoRA distintos sobre el mismo modelo base, con datos
   diferentes (por ejemplo, sumas y restas por separado).
2. PROMEDIA los adaptadores de dos formas:
   a) promediando A y B por separado: (A1+A2)/2 y (B1+B2)/2
   b) promediando el producto: (B1@A1 + B2@A2)/2
3. Comprueba que NO dan el mismo resultado.
4. Explica por qué: el producto de las medias no es la media de los productos.
5. Mide cuál de los dos funciona mejor en una tarea que combine ambas.

Este último experimento es una pregunta abierta de mi tesis. Documéntalo con cuidado.

MEDIR
Tabla: método × (parámetros entrenables, memoria, tiempo, rendimiento).
```

---

## N25 · SFT: enseñar a responder ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cómo pasa de autocompletar a responder?

**Construyes:**

1. Genera un dataset de instrucciones sobre tu lenguaje: `(pregunta, respuesta correcta)`.
2. Implementa la plantilla de chat con tokens especiales.
3. **El enmascarado de la pérdida**: solo sobre la respuesta.
4. Ajusta el modelo de N21.

**Mides:** la misma pregunta antes y después. El cambio de comportamiento es visible de
golpe.

**Qué rompes:** ⭐ **quita el enmascarado** y entrena sobre todos los tokens. Observa que
el modelo empieza a generar preguntas de usuario además de respuestas.

**Y el aviso ②:** mete en el dataset respuestas seguras a preguntas cuya respuesta el
modelo no puede saber. Comprueba si aumenta su tendencia a inventar.

→ **Enlaza con:** 4.4
**Tiempo:** 3 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N25

OBJETIVO
Convertir el modelo base de N21, que autocompleta, en uno que responde.

DATOS
Genera con N01 un dataset de instrucciones, nivel 2-3:
   <|user|>¿cuánto es 3+4*2?<|assistant|>11<|end|>
Unos 10.000 ejemplos. Que las preguntas tengan varias formulaciones para que no memorice
una plantilla única.

QUÉ CONSTRUIR
1. Añadir los tokens especiales al vocabulario: <|user|>, <|assistant|>, <|end|>.
2. Función que aplane un diálogo a texto plano con esas marcas.
3. EL ENMASCARADO DE LA PÉRDIDA: calcular la pérdida SOLO sobre los tokens de la
   respuesta, no sobre los de la pregunta. Muestra explícitamente la máscara.
4. Entrenar partiendo del modelo de N21 (tasa de aprendizaje 10-100× menor que en
   preentrenamiento; justifica el valor).

EL ANTES Y EL DESPUÉS
Haz la MISMA pregunta al modelo antes y después del SFT. Pon las dos salidas literales
en una celda markdown. El cambio de comportamiento debe ser evidente.

QUÉ ROMPO AQUÍ (dos experimentos)
1. QUITA EL ENMASCARADO: entrena con la pérdida sobre todos los tokens.
   - Genera varias muestras.
   - Comprueba que el modelo empieza a generar PREGUNTAS DE USUARIO además de
     respuestas, porque aprendió a producir ambos roles.

2. ENSEÑAR A ALUCINAR: añade al dataset de SFT respuestas seguras a preguntas cuya
   respuesta el modelo no puede saber (números fuera del rango de entrenamiento).
   - Mide la tasa de invención antes y después.
   - Comprueba la hipótesis: no aprende esos datos, aprende el PATRÓN de responder con
     seguridad sin saber.

TAMBIÉN
Barrido de calidad frente a cantidad: entrena con 500 ejemplos muy limpios y con 20.000
ruidosos. Compara. La hipótesis es que ganan los 500.

MEDIR
- % de respuestas con formato correcto.
- % de respuestas correctas.
- Tasa de invención en preguntas fuera de rango.
```

---

## N26 · Preferencias: enseñar criterio

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo mejorar sin escribir la respuesta perfecta?

**Construyes:** aquí es donde el lenguaje sintético paga su inversión. **Fabricas
preferencias objetivas sin anotar nada:**

```none
respuesta correcta  ≻  respuesta incorrecta
respuesta breve     ≻  respuesta correcta pero inflada
respuesta honesta   ≻  invención segura
```

Implementa DPO. Compara contra el modelo de N25.

**Mides:** la **longitud media** de las respuestas antes y después. Si creció sin mejorar
el contenido, acabas de reproducir el *reward hacking* con tus propios datos.

**Qué rompes:** entrena mucho más de lo razonable y observa cómo la puntuación sigue
subiendo mientras la calidad real baja. Varía la penalización KL y busca el punto donde
el freno estorba.

→ **Enlaza con:** 4.5
**Tiempo:** 3 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N26

OBJETIVO
Mejorar el modelo sin escribir ni una sola respuesta ideal, solo comparando.

DATOS
Genera pares de preferencia con N01, de los tres tipos, ~5.000 pares de cada:
  1. CORRECCIÓN:   "11" ≻ "10"
  2. LONGITUD:     "11" ≻ "El resultado de la operación 3+4*2 es, efectivamente, 11."
  3. HONESTIDAD:   "no lo sé" ≻ "483920"  (para números fuera del rango entrenado)

QUÉ CONSTRUIR
1. Implementa DPO. Explica cada término de la pérdida y qué papel juega el modelo de
   referencia congelado.
2. Entrena partiendo del modelo SFT de N25.
3. Evalúa por separado el efecto sobre cada uno de los tres tipos.

LAS TRES MÉTRICAS
1. Corrección: ¿mejora, empeora o se mantiene? (Hipótesis: apenas cambia.)
2. LONGITUD MEDIA de las respuestas antes y después. Es la métrica delatora.
3. Tasa de "no lo sé" en preguntas fuera de rango, antes y después.

QUÉ ROMPO AQUÍ: REWARD HACKING
1. Entrena SOLO con pares del tipo 1 (corrección), muchas más épocas de lo razonable.
2. Grafica: longitud media de respuesta frente a época.
3. Si crece sin que mejore la corrección, acabo de reproducir el reward hacking.
4. Barre la penalización KL (beta): 0.01, 0.1, 0.5. Grafica longitud y corrección frente
   a beta. Busca el punto donde el freno empieza a estorbar.

EXTRA (opcional pero muy instructivo)
Entrena un modelo de recompensa explícito con Bradley-Terry sobre los mismos pares.
Comprueba si su puntuación correlaciona con la corrección real. Donde no correlacione,
ahí está el hueco que el modelo puede explotar.

MEDIR
Tabla: tipo de preferencia × (corrección, longitud media, tasa de honestidad).
```

---

## N27 · Refuerzo: de bandidos a política

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿cómo se aprende sin que nadie te diga la respuesta?

**Construyes** —tres pasos, porque el salto es grande—:

1. **Bandidos.** Exploración frente a explotación en su forma más pura.
2. **Gridworld tabular.** Q-learning. Visualiza la tabla de valores llenándose. Aquí ves
   la **asignación de crédito** con los ojos.
3. **Policy gradient** sobre un entorno pequeño.

**Qué rompes:** una función de recompensa mal diseñada. Elige una que se pueda explotar y
observa al agente encontrarle el truco en vez de resolver la tarea.

→ **Enlaza con:** 4.6 · 2.8
**Tiempo:** 4 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N27

OBJETIVO
Entender el refuerzo desde la forma más simple posible, antes de aplicarlo a lenguaje.

TRES PARTES, EN ESTE ORDEN

PARTE 1 · BANDIDOS DE K BRAZOS
- k=10 brazos con recompensas estocásticas.
- Estrategias: greedy, epsilon-greedy, UCB.
- Grafica recompensa acumulada y % de acciones óptimas frente a paso.
- Es la exploración frente a explotación en su forma más pura, sin estados.

PARTE 2 · GRIDWORLD TABULAR CON Q-LEARNING
- Rejilla pequeña (5x5) con meta y trampa.
- Q-learning tabular.
- VISUALIZA LA TABLA Q como mapa de calor, época a época. Debe verse el valor
  propagarse hacia atrás desde la meta.
- Esta visualización es la asignación de crédito temporal, con los ojos.
- Barre el factor de descuento gamma y observa cómo cambia la política.

PARTE 3 · POLICY GRADIENT
- REINFORCE sobre el mismo gridworld o sobre CartPole.
- Explica por qué la recompensa no es diferenciable y qué hace el truco del gradiente
  logarítmico.
- Compara con y sin línea base (baseline). Mide la varianza del gradiente en ambos casos.

QUÉ ROMPO AQUÍ
Diseña una función de recompensa mal especificada, que se pueda explotar.
Ejemplo: premiar acercarse a la meta en línea recta en vez de llegar. El agente encontrará
una forma de acumular recompensa sin resolver la tarea.
- Documenta qué política degenerada aprende.
- Es reward hacking en su forma más pura, y conecta con N26.

MEDIR
Curvas de recompensa acumulada. Para el gridworld, el mapa de calor de la tabla Q.
```

---

## N28 · RLVR: el modelo descubre solo ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puede el modelo aprender a razonar sin demostraciones?

**Construyes:** el notebook estrella del bloque, y solo es posible por la decisión del
hilo conductor.

1. Verificador de tu lenguaje: dada una respuesta, ¿es correcta? → recompensa 0/1.
2. Genera $N$ intentos por problema.
3. Actualiza la política premiando los que aciertan (GRPO simplificado: la ventaja sale
   de comparar entre las $N$ generaciones, sin crítico).

**Mides:** ¿se alargan las salidas? ¿Aparecen pasos intermedios que nadie enseñó?

**Qué rompes:** ⭐ **quítale la base capaz.** Aplica RLVR a un modelo que acierta ~0 % por
azar. La recompensa es 0 siempre y no aprende nada. **Es la restricción 3 del pipeline,
demostrada.**

→ **Enlaza con:** 4.6 · 4.9 · 4.8
**Tiempo:** 4 sesiones

### 📋 Bloque para el LLM

```text
BLOQUE N28

OBJETIVO
Que el modelo descubra POR SÍ MISMO que le conviene razonar por pasos, sin que nadie se
lo enseñe. Es el notebook estrella del itinerario.

DATOS
Lenguaje aritmético NIVEL 3 (con paréntesis): (3+4)*2=14
Es importante el nivel 3: resolverlo de una pasada es difícil, por pasos es fácil. Esa
brecha es lo que hace que emerja el razonamiento.

QUÉ CONSTRUIR
1. EL VERIFICADOR (tres líneas):
   def recompensa(problema, respuesta):
       return 1.0 if respuesta == str(eval(problema)) else 0.0
2. Bucle de RL simplificado tipo GRPO:
   - para cada problema, generar N=8 intentos con muestreo
   - calcular la recompensa de cada uno
   - la ventaja de cada intento = su recompensa menos la media del grupo
     (así no hace falta modelo crítico)
   - actualizar la política con gradiente de política ponderado por esa ventaja
3. Partir del modelo SFT de N25.
4. Permitir que el modelo genere tokens intermedios antes de la respuesta final:
   define un formato donde pueda "pensar" antes del resultado, y verifica solo el
   resultado final.

QUÉ MEDIR (lo importante)
1. Exactitud frente a paso de entrenamiento.
2. LONGITUD MEDIA DE LA GENERACIÓN frente a paso. La hipótesis: se alarga sola.
3. Guarda muestras de generaciones en varios momentos del entrenamiento y ponlas en una
   celda markdown. Busca si aparecen pasos intermedios que nadie enseñó.

QUÉ ROMPO AQUÍ (la restricción del pipeline, demostrada)
Aplica exactamente el mismo RLVR a un modelo que acierta ~0% por azar (un modelo base sin
SFT, o uno demasiado pequeño).
- La recompensa será 0 en todos los intentos.
- La ventaja será 0 para todos.
- El gradiente será nulo y el modelo no aprenderá NADA.
- Explica: sin aciertos ocasionales no hay señal que reforzar. El RL no puede enseñar
  desde cero.

TAMBIÉN
Compara el resultado de RLVR contra el de SFT sobre las trazas de referencia (`pasos()`
de N01). ¿Descubre el modelo la misma estrategia que le habríamos enseñado?

RESTRICCIÓN
Esto puede ser lento. Usa un modelo muy pequeño y problemas cortos. Prioriza que corra
en minutos aunque el resultado sea modesto.
```


# BLOQUE 7 · El pipeline completo

> **¿Sé encadenarlo todo?**

---

## N29 · Destilación

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿puedo transferir capacidad a un modelo pequeño?

**Construyes:** dos vías, para ver que son la misma idea con distinto vestido:

1. **Clásica:** pérdida KL con temperatura desde el modelo grande. Barre $T$.
2. **Moderna:** el modelo de N28 genera trazas resueltas → SFT del pequeño sobre ellas.

**Mides:** ¿el alumno destilado bate al mismo modelo pequeño entrenado con RLVR directo?
(La hipótesis dice que sí.)

**Qué rompes:** ⭐ **el colapso por datos sintéticos.** Genera con el alumno, entrena a un
tercero sobre esa salida, repite tres veces. Observa la diversidad estrecharse.

**Y también:** comprueba si los patrones **raros** del lenguaje sobreviven a la
destilación, o si solo sobreviven los comunes.

→ **Enlaza con:** 4.7
**Tiempo:** 3 sesiones

---

### 📋 Bloque para el LLM

```text
BLOQUE N29

OBJETIVO
Transferir capacidad a un modelo pequeño por dos vías distintas, y provocar el colapso
por datos sintéticos.

PARTE 1 · DESTILACIÓN CLÁSICA (de logits)
1. Modelo maestro: el de N28 (o el mayor que tengas entrenado).
2. Modelo alumno: la mitad de tamaño.
3. Pérdida: KL entre las distribuciones softmax con temperatura T, combinada con la
   pérdida sobre etiquetas duras.
4. Barre T = 1, 2, 5, 10. Grafica rendimiento del alumno frente a T.
5. Compara contra el alumno entrenado desde cero con etiquetas duras. El hueco es lo que
   aporta el maestro.

DARK KNOWLEDGE A LA VISTA
Imprime la distribución completa del maestro para 10 ejemplos.
- ¿Tienen sentido las probabilidades secundarias?
- Por ejemplo, si la respuesta es 46, ¿asigna más probabilidad a 45 y 47 que a 12?
- Esa estructura es lo que la etiqueta dura no puede transmitir.

PARTE 2 · DESTILACIÓN MODERNA (por generación)
1. El maestro de N28 genera 20.000 problemas resueltos CON sus pasos intermedios.
2. Filtra: quédate solo con los verificados como correctos.
3. El alumno hace SFT sobre esas trazas.
4. Compara contra aplicar RLVR directamente al alumno pequeño.
   Hipótesis: la destilación gana, porque el alumno no tiene capacidad para DESCUBRIR
   pero sí para COPIAR.

QUÉ ROMPO AQUÍ: EL COLAPSO POR DATOS SINTÉTICOS
1. Modelo 1 entrenado con datos reales (generados por el verificador).
2. Modelo 1 genera un corpus. Modelo 2 se entrena SOLO con ese corpus.
3. Modelo 2 genera. Modelo 3 se entrena con eso. Repite 4-5 generaciones.
4. Mide en cada generación:
   - diversidad de las salidas (entropía, número de expresiones únicas)
   - cobertura: ¿siguen apareciendo los casos raros o solo los comunes?
   - corrección
5. Grafica las tres métricas frente al número de generación.
6. Repite el experimento ACUMULANDO datos reales y sintéticos en vez de sustituirlos.
   Comprueba si el colapso se evita.

PARTE 3 · LO QUE SE PIERDE
Comprueba si los patrones RAROS del lenguaje (operaciones poco frecuentes en el corpus)
sobreviven a la destilación, o si solo sobreviven los comunes.
```

---

## N30 · El pipeline de punta a punta ⭐

☐ **Pendiente** — marcar al terminar, y anotar la entrada de bitácora

**Pregunta:** ¿qué aporta cada fase, medido?

**Construyes:** encadena todo sobre el mismo lenguaje sintético, y **evalúa después de
cada fase con el mismo protocolo**:

```none
base (N21) ──▶ SFT (N25) ──▶ DPO (N26) ──▶ RLVR (N28) ──▶ destilado (N29)
```

**La tabla que produce este notebook** es el resultado de todo el itinerario:

| Fase | Corrección | Formato | Longitud media | Coste (tiempo, memoria) |
|---|---|---|---|---|
| Base | | | | |
| +SFT | | | | |
| +DPO | | | | |
| +RLVR | | | | |
| Destilado | | | | |

**Qué rompes:**

- **Salta el SFT** e intenta hacer DPO sobre el modelo base. Comprueba que no hay dos
  respuestas comparables: **no hay nada que preferir**.
- **Cambia el orden.** RLVR antes que SFT, como R1-Zero. ¿Funciona? ¿Qué se pierde?

**Y la pregunta que cierra el itinerario:** en tu problema, **¿en qué fase satura?** Con
un problema fácil puede que baste el SFT. Saberlo es lo que evita sobre-diseñar.

→ **Enlaza con:** 4.8 · 3.9
**Tiempo:** 4 sesiones

### 📋 Bloque para el LLM

```text
BLOQUE N30

OBJETIVO
Encadenar las cinco fases sobre el mismo problema y medir qué aporta cada una. Es el
notebook que resume el itinerario entero.

QUÉ CONSTRUIR
Un pipeline reproducible que ejecute en orden, partiendo del mismo modelo base:
  1. base (N21)      auto-supervisado sobre corpus de expresiones
  2. +SFT (N25)      formato de pregunta-respuesta
  3. +DPO (N26)      preferencias de corrección, longitud y honestidad
  4. +RLVR (N28)     recompensa del verificador
  5. destilado (N29) a un modelo la mitad de grande

EL PROTOCOLO DE EVALUACIÓN (fijarlo ANTES de ejecutar)
Un único conjunto de evaluación, aplicado idénticamente después de cada fase:
- % de respuestas con formato correcto
- % de respuestas correctas, desglosado por nivel de dificultad
- longitud media de la respuesta
- tasa de "no lo sé" en preguntas fuera de rango
- tiempo de inferencia por respuesta
- memoria del modelo

Ejecutar con 3 semillas y reportar intervalos, no puntos.

LA TABLA FINAL (el resultado del itinerario)
Fase | Formato | Corrección | Longitud | Honestidad | Coste
Rellénala y comenta cada salto: qué aportó esa fase y qué no.

QUÉ ROMPO AQUÍ (dos experimentos de orden)
1. SALTARSE EL SFT: intenta hacer DPO directamente sobre el modelo base.
   - Genera dos respuestas del modelo base a la misma pregunta.
   - Comprueba que son dos autocompletados, no dos respuestas.
   - Conclusión: no hay nada que preferir. La precondición es real.

2. CAMBIAR EL ORDEN: aplica RLVR ANTES del SFT (como hizo R1-Zero).
   - ¿Funciona?
   - ¿Qué se gana y qué se pierde respecto al orden estándar?
   - Compara el formato de las salidas en ambos casos.

LA PREGUNTA QUE CIERRA EL ITINERARIO
Para cada nivel de dificultad del lenguaje (0 a 3), determina EN QUÉ FASE SATURA:
- ¿En qué nivel basta el SFT?
- ¿A partir de qué nivel aporta el RLVR?
Grafica: nivel de dificultad frente a fase donde deja de mejorar.
Esa gráfica es el argumento contra sobre-diseñar un pipeline.
```

---
---

# Al terminar cada notebook

- [ ] La entrada de bitácora está cerrada, con **qué me sorprendió** relleno.
- [ ] La casilla de la tabla de progreso está marcada.
- [ ] Si salió una pregunta abierta → subida al **Anexo A**.
- [ ] Si la especificación resultó equivocada en algo → anotado aquí mismo, debajo del
      bloque. **Siempre cambia algo.**

---

# Si un notebook se atasca

| Síntoma | Qué hacer |
|---|---|
| Tarda más de 10 minutos | Reduce el problema. La velocidad de iteración manda |
| El bloque pide demasiado | Pártelo en dos: N19a y N19b. No renumeres el resto |
| No entiendo el resultado | Es material de bitácora, no un fracaso. Anótalo y sigue |
| Funciona a la primera | Sospecha. Comprueba que mides lo que crees |
| Llevo tres sesiones atascado | Sáltalo, marca el motivo, y vuelve cuando tengas más contexto |

---

# Después del N30

El itinerario **se repite**, no se amplía: los mismos notebooks del bloque 6 sobre datos
reales y con la GPU de la universidad. Ver la Parte V de
[`00-metodo/itinerario.md`](../../00-metodo/itinerario.md).