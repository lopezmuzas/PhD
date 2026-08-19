---
title: "Itinerario práctico · De cero a un pipeline completo"
tags: [metodo, notebooks, itinerario, practica, dataset]
status: vivo
---

# Itinerario práctico

> **Qué es esto.** El mapa completo de notebooks a construir, del perceptrón al pipeline
> de entrenamiento de un modelo de lenguaje. Incluye el dataset que los atraviesa todos.
>
> **No es un curso.** Es una secuencia de experimentos, cada uno con una pregunta y un
> fallo que provocar. No hay plazos: hay orden y dependencias.
>
> **La lista de notebooks** —con el bloque de especificación para el LLM de cada uno—
> está en `04-proyecto/09-indice-notebooks/notebooks.md`. Este documento se **lee** una
> vez; aquel se **recorre** del N00 al N30.

---

## Cómo navegar este documento

| Parte | Qué contiene | Cuándo se lee |
|---|---|---|
| **I · Principios** | Los cinco criterios de diseño y el hilo conductor | Una vez, al empezar |
| **II · El dataset** | Los generadores y el lenguaje aritmético | Al construir N01, y de consulta |
| **III · Los notebooks** | El mapa de los 31, y enlace a la lista de trabajo | Para orientarse |
| **IV · Cómo trabajar** | Reglas, advertencias y autodiagnóstico | Cuando dudes si vas bien |
| **V · Decisiones** | Lo que queda por fijar, y la fase 2 | Al cerrar un bloque |

**Y el documento hermano:** la lista de trabajo, del N00 al N30, con el bloque de
especificación de cada uno, está en
[`04-proyecto/09-indice-notebooks/notebooks.md`](../04-proyecto/09-indice-notebooks/notebooks.md).

---

## El diagnóstico

Tu problema no es de conocimiento. Es de **observabilidad**.

```none
   Lo que tienes                     Lo que falta
   ─────────────                     ────────────
   Sabes qué hace backprop           Ver el gradiente moverse
   Sabes qué es una pérdida          Saber si esa curva es sana o mentirosa
   Has entrenado modelos             Saber POR QUÉ funcionó
   Sabes la teoría de los paradigmas Haberlos roto con las manos
```

Y de ahí sale el orden de lo que viene: **primero instrumentos, después modelos.**

No sirve de nada avanzar a arquitecturas más grandes si sigues sin poder ver qué pasa
dentro. Un LSTM que no entiendes falla igual que un MLP que no entiendes, solo que tarda
más en fallar y cuesta más depurarlo.

---

## El principio que ordena todo

> **La intuición no viene de construir. Viene de romper.**

Un modelo que funciona te enseña poco: no sabes cuál de tus veinte decisiones fue la
buena. Un modelo que **rompes a propósito** te enseña el mecanismo, porque has aislado
la causa.

Esa es la diferencia entre seguir una guía y "hackearlo". Y es una habilidad que se
entrena deliberadamente, no que aparece con las horas.

---
---

---
---

# Parte I · Los principios

## Los cinco principios de diseño

### 1. Instrumentos antes que modelos

No se sube de arquitectura hasta poder ver qué pasa dentro de la actual. Un LSTM que no
entiendes falla igual que un MLP que no entiendes, pero tarda más y cuesta más depurarlo.

### 2. Cada notebook rompe algo a propósito

Un modelo que funciona no te dice cuál de tus veinte decisiones fue la buena. Uno que
rompes te enseña el mecanismo, porque has aislado la causa.

**Todos los notebooks tienen un apartado obligatorio: "qué rompo aquí".**

### 3. Datos sintéticos por defecto

Con datos sintéticos **conoces la respuesta verdadera**, y eso permite responder a
*"¿está bien?"* y no solo a *"¿mejora?"*. Datos reales solo cuando el fenómeno los exija.

### 4. Un notebook, una variable nueva

Si añades arquitectura, datos y optimizador a la vez y falla, no sabes cuál fue. Es el
mismo principio de aislamiento de → 3.8.

### 5. Se cierra con bitácora

Sin entrada de bitácora, el notebook no está terminado. Hipótesis antes de ejecutar,
resultado y sorpresa después (→ [Bitácora · El sistema](bitacora-el-sistema.md)).

---

## El hilo conductor: aritmética

La columna vertebral del itinerario es **un problema de juguete propio** que aparece una
y otra vez, cada vez atacado con más maquinaria.

**Decidido: aritmética en texto.** Cinco niveles de dificultad, de `3+4=7` a
`2*(8-3)+1=11`. Diecisiete tokens de vocabulario, verificador de tres líneas, y todo
corre en CPU.

→ La especificación completa está en la **Parte II** de este documento.

**Por qué esta decisión es la más importante del itinerario:**

| Ventaja | Consecuencia |
|---|---|
| Conoces la respuesta verdadera | Puedes medir el error absoluto, no solo la mejora |
| Tienes un **verificador** gratis | Sin él no hay RLVR posible en un portátil (N26) |
| Puedes fabricar **preferencias objetivas** | "Correcta ≻ incorrecta", sin anotar (N24) |
| Controlas la dificultad | Ves dónde se rompe cada arquitectura |
| Corre en CPU | Iteración de segundos, no de horas |

**Su límite, y cómo se compensa.** Un laboratorio de cristal enseña mecanismos y esconde
la suciedad del mundo real. Por eso el itinerario tiene **dos carriles**:

```none
  CARRIL A · sintético   ──▶ la columna vertebral, casi todos los notebooks
  CARRIL B · real        ──▶ solo donde el fenómeno lo exija (N09, N13, N14, N21)
```

---

## Mapa general

```none
BLOQUE 0 · EL ANDAMIO           N00–N01    ¿tengo con qué trabajar?
     ↓
BLOQUE 1 · EL MECANISMO         N02–N05    ¿entiendo qué pasa por dentro?
     ↓
BLOQUE 2 · INSTRUMENTAR         N06–N07    ¿puedo VER qué pasa por dentro?
     ↓
BLOQUE 3 · ROMPER               N08–N12    ¿reconozco los fallos?
     ↓
BLOQUE 4 · MEDIR                N13–N15    ¿me creo mis propios resultados?
     ↓
BLOQUE 5 · ARQUITECTURAS        N16–N20    ¿qué aporta cada familia?
     ↓
BLOQUE 6 · FORMAS DE APRENDER   N21–N28    ¿de dónde puede salir la señal?
     ↓
BLOQUE 7 · EL PIPELINE          N29–N30    ¿sé encadenarlo todo?
```

Los bloques 0–4 son **la inversión que hace rentable todo lo demás**. Son los menos
vistosos y los que no se pueden saltar.

---
---

# Parte II · El dataset

> Se define una vez, en N01, y se reutiliza en los treinta notebooks siguientes.
>
> **Principio rector:** cero complejidad añadida. Si tienes que explicar el dataset, está
> mal elegido.

### Dos partes, no una

El itinerario necesita dos tipos de dato, y conviene no mezclarlos:

```none
PARTE A · Tabular y 2D          N02 – N17
  puntos, espirales, imágenes pequeñas
  para: perceptrón, MLP, CNN, todo el bloque de romper y medir

PARTE B · El lenguaje           N18 – N30
  aritmética en texto
  para: RNN, atención, transformer, y TODO el bloque de formas de aprender
```

La parte A es convencional y no necesita justificación. **La parte B es la decisión de
diseño importante**, y es lo que ocupa el resto de este documento.

---
---

## PARTE A · Datos tabulares y 2D

Generadores parametrizables. Todos con **respuesta óptima calculable**, que es su razón
de ser.

| Generador | Parámetros | Para qué |
|---|---|---|
| **Recta con ruido** | pendiente, sesgo, $\sigma$ | Regresión. Conoces los coeficientes verdaderos |
| **Dos gaussianas** | separación, solapamiento | Clasificación. Conoces el clasificador de Bayes |
| **XOR** | — | **El fallo del perceptrón** (N02) |
| **Espirales** | vueltas, ruido | No linealidad graduable |
| **Medias lunas** | separación | Donde k-means se rompe |
| **Ruido puro** | — | ⭐ Sin ninguna relación real. Para N10 y N14 |
| **Imágenes mínimas** | tamaño, formas | CNN sin descargar nada (N17) |

### Los modificadores: lo que rompe cosas después

Cada generador acepta además:

```none
--desbalanceo      proporción de clases        → N14
--ruido-etiqueta   % de etiquetas mal          → N10
--duplicados       % repetidos entre splits    → N15
--grupos           estructura de grupo/sitio   → N15
--deriva-temporal  distribución que cambia     → N15
```

> **La regla:** todo lo que quieras romper después, tiene que ser un parámetro aquí.
> Si tienes que reescribir el generador para provocar un fallo, el generador está mal.

### Lo que hace especial a esta parte

**Para cada dataset generado, guarda también el rendimiento del modelo óptimo.**

Sin ese número, un 0,87 no significa nada: no sabes si te falta un 1 % o un 12 %. Con él,
cada experimento tiene techo conocido.

**Carril B (datos reales), solo tres veces:** MNIST o CIFAR-10 en N17, y un dataset
tabular real en N15 y N23. Nada más.

---
---

## PARTE B · El lenguaje: aritmética

### El formato base

```none
12+34=46
```

Eso es todo. Ocho caracteres, sin metadatos ni estructura oculta.

### Vocabulario

```none
0 1 2 3 4 5 6 7 8 9    ← 10 dígitos
+ - * ( )              ← 5 operadores y agrupación
=                      ← separador
<pad> <eos>            ← control
```

**17 tokens.** Un embedding de 17×64 son 1.088 parámetros. Cabe en cualquier sitio.

### Tokenización: a nivel de carácter

Cada dígito es un token independiente. **No fusiones dígitos.**

> ⚠️ **Y esto es una lección, no un detalle de implementación.** Los modelos reales
> tokenizan números de forma inconsistente —`1234` puede ser un token, dos o cuatro— y
> ese es uno de los motivos de que fallen en aritmética.
>
> En N01, tokeniza el mismo problema de las dos formas y guarda ambas versiones. En algún
> notebook posterior, entrena con cada una y compara. **Es el experimento que hace
> tangible el bloque del token de §4.3.**

---

### Los cinco niveles de dificultad

Un único parámetro `nivel` controla todo. Subes de nivel cuando el anterior se resuelve.

#### Nivel 0 · Un dígito

```none
3+4=7
8+1=9
```

**Para qué:** N16–N18. Es un problema de clasificación disfrazado: solo hay 100
combinaciones posibles y el modelo puede memorizarlas.

**Lo que enseña:** que memorizar no es aprender. Evalúa en combinaciones que no vio.

---

#### Nivel 1 · Dos y tres dígitos, con acarreo ⭐

```none
47+38=85
129+473=602
```

**Para qué:** N18–N21. **El nivel de trabajo principal.**

**Por qué es el bueno:** el **acarreo** es una dependencia de largo alcance. El resultado
del dígito de la derecha afecta al de la izquierda. Eso es exactamente lo que una RNN
simple hace mal, una LSTM mejor, y la atención bien.

Es tu experimento de N18 —memoria en secuencias— con contenido real en vez de artificial.

> **El truco que enseña más de lo que parece:** prueba a generar la respuesta **con los
> dígitos invertidos**:
>
> ```none
> 47+38=58        ← "85" escrito al revés
> ```
>
> Es mucho más fácil de aprender, porque el acarreo fluye en el mismo sentido que la
> generación. **La representación del dato importa tanto como la arquitectura**, y aquí
> se mide.

---

#### Nivel 2 · Varios operadores, con precedencia

```none
3+4*2=11
12-5+8=15
```

**Para qué:** N21–N25.

**Qué añade:** ya no basta con procesar de izquierda a derecha. Hay que **decidir el
orden**. Es lo primero que se parece a razonar, y es donde los mapas de atención de N19
empiezan a mostrar algo interesante.

---

#### Nivel 3 · Paréntesis ⭐

```none
(3+4)*2=14
2*(8-3)+1=11
```

**Para qué:** N26–N30. **El nivel donde emerge el razonamiento.**

**Por qué:** resolver esto de una pasada es difícil; resolverlo por pasos es fácil. Esa
brecha es lo que hace que el modelo, entrenado solo con recompensa por el resultado final,
**descubra por su cuenta** que le conviene escribir pasos intermedios.

---

#### Nivel 4 · Enunciado en texto (opcional)

```none
Ana tiene 12 manzanas y compra 34 más. ¿Cuántas tiene?=46
```

**Para qué:** solo si quieres realismo en SFT y preferencias. Añade variación lingüística
sin cambiar el verificador.

> **Aviso:** este nivel **sí añade complejidad**. Necesitas plantillas de redacción y el
> modelo tiene que aprender lenguaje además de aritmética. Déjalo para cuando el nivel 3
> funcione, y solo si te aporta algo.

---

### Cómo cada notebook usa el lenguaje

| Notebook | Nivel | Formato | Qué se mide |
|---|---|---|---|
| **N18** RNN/LSTM | 1 | `47+38=85` | ¿Aprende el acarreo? RNN vs LSTM |
| **N19** Atención | 1 | idem | ¿A qué dígito atiende al generar cada uno? |
| **N20** Transformer | 1–2 | idem | Pre-norm vs post-norm con este dato |
| **N21** Auto-supervisado | 2 | corpus de expresiones | % de salidas sintácticamente válidas |
| **N23** Sondeo lineal | 2 | representaciones congeladas | ¿Codifica el resultado en el embedding? |
| **N25** SFT | 2–3 | `<user>¿3+4*2?<assistant>11` | ¿Pasa de autocompletar a responder? |
| **N26** Preferencias | 3 | pares construidos | ¿Mejora la calibración sin tocar corrección? |
| **N28** RLVR | 3 | recompensa 0/1 del verificador | **¿Emergen pasos intermedios?** |
| **N29** Destilación | 3 | trazas del maestro | ¿Sobreviven los casos raros? |
| **N30** Pipeline | 3 | todo | La tabla final |

---

### Lo que desbloquea cada fase del pipeline

Aquí está el motivo real de esta elección. Con aritmética, **las cinco fases del capítulo
4 se pueden montar sin anotar un solo ejemplo a mano**.

### Auto-supervisado (N21)

Genera un corpus de expresiones correctas. Entrena predicción del siguiente token. Ya
está.

**Y el momento clave:** pregúntale `¿cuánto es 12+34?` y observa que **autocompleta con
otra expresión** en vez de responder. Ese fallo justifica el SFT.

### SFT (N25)

```none
<|user|>¿cuánto es 12+34?<|assistant|>46<|end|>
```

Con el enmascarado de la pérdida sobre la respuesta. El dataset se genera solo.

### Preferencias (N26) — donde la elección paga

Tres tipos de par, **todos fabricables sin humanos**:

```none
① CORRECCIÓN
   "46"  ≻  "45"
   
② CALIBRACIÓN DE LONGITUD
   "46"  ≻  "El resultado de sumar 12 y 34 es, efectivamente, 46."
   
③ HONESTIDAD
   "no lo sé"  ≻  "98234"     ← para números fuera del rango entrenado
```

El tercero es el más interesante y solo es posible aquí: **entrena con números de hasta
3 dígitos y pregunta con 6**. El modelo no puede saberlo. Si prefieres la respuesta
honesta a la inventada, le enseñas a admitir ignorancia — y puedes **medir si lo aprendió**,
cosa que con datos reales es casi imposible.

### RLVR (N28) — el notebook estrella

```python
# el verificador entero
def recompensa(problema, respuesta):
    return 1.0 if respuesta == str(eval(problema)) else 0.0
```

**Eso es todo.** Sin anotadores, sin modelo de recompensa, sin coste.

Y lo que esperas ver, con nivel 3:

```none
Al principio:
  (3+4)*2=13          ← falla

Después de mucho RLVR:
  (3+4)*2=<3+4=7, 7*2=14>14
                └── nadie le enseñó esto
```

**Es la historia de R1-Zero en miniatura**, corriendo en tu portátil.

### Destilación (N29)

El modelo de N28 genera 50.000 problemas resueltos **con sus pasos**. Un modelo pequeño
hace SFT sobre eso. Y compruebas si bate al mismo modelo pequeño entrenado con RLVR
directo.

---

### Lo que hay que implementar en N01

Un módulo pequeño. Estas son las funciones, no el código:

```none
generar(nivel, n, semilla)        → lista de expresiones
formatear(expr, modo)             → 'crudo' | 'chat' | 'invertido'
tokenizar(texto, modo)            → 'char' | 'agrupado'
verificar(problema, respuesta)    → bool
resolver(problema)                → la respuesta correcta
pasos(problema)                   → la traza de referencia (para N29)
generar_preferencias(tipo, n)     → pares (preferida, rechazada)
```

**El criterio de terminado:** con esas siete funciones, todos los notebooks del bloque 6
tienen sus datos resueltos. No se toca más.

### Y una decisión que hay que tomar ahí

**Cómo se parte train/test.** Con aritmética hay una trampa preciosa:

```none
Split ALEATORIO       →  47+38 en train, 38+47 en test
                         el modelo "generaliza"... o memorizó la conmutatividad

Split POR RESULTADO   →  todos los que dan 85 en train, ninguno en test
                         mucho más duro, y mucho más informativo

Split POR RANGO       →  entrena con 1-3 dígitos, evalúa con 4
                         mide extrapolación real
```

**Los tres son válidos y miden cosas distintas.** Impleméntalos los tres en N01 y usa el
que corresponda en cada notebook. Es la lección de → 3.9 aplicada desde el primer día.

---

### Cuándo dejar el juguete

Este dataset es un laboratorio de cristal: enseña mecanismos y esconde la suciedad real.
Tres momentos para salir:

1. **N15 y N17** — carril B, para ver fuga de datos y sesgo inductivo sobre datos reales.
2. **Cuando tengas GPU de la universidad** — repetir N21–N30 sobre un modelo
   preentrenado pequeño y un dataset real de instrucciones. La estructura del itinerario
   no cambia; solo la escala.
3. **Cuando el problema de la tesis esté definido** — sustituir la aritmética por
   trayectorias o registros del dominio real.

> **Pero no antes.** El valor de este dataset es que **puedes verificar todo**. En cuanto
> pases a datos reales, pierdes esa capacidad y empiezas a entrenar a ciegas otra vez.

---

### Resumen en una frase

> Un dataset que cabe en veinte líneas de código, corre en CPU, y permite montar las
> cinco fases del pipeline de un modelo de lenguaje sin anotar un solo ejemplo a mano.

---
---

# Parte III · Los notebooks

Los 31 notebooks, con su contexto y su bloque de especificación para el LLM, viven en un
documento aparte porque es **la lista de trabajo**, no material de lectura:

> ### 📋 [`04-proyecto/09-indice-notebooks/notebooks.md`](../04-proyecto/09-indice-notebooks/notebooks.md)
>
> Con tabla de progreso, preámbulo común y una entrada por notebook.

## El mapa, para referencia

| Bloque | Notebooks | Pregunta que responde |
|---|---|---|
| **0 · El andamio** | N00–N01 | ¿Tengo con qué trabajar? |
| **1 · El mecanismo desnudo** | N02–N05 | ¿Entiendo qué pasa por dentro? |
| **2 · Instrumentar** | N06–N07 | ¿Puedo **ver** qué pasa por dentro? |
| **3 · Romper a propósito** | N08–N12 | ¿Reconozco los fallos? |
| **4 · Medir** | N13–N15 | ¿Me creo mis propios resultados? |
| **5 · Arquitecturas** | N16–N20 | ¿Qué aporta cada familia? |
| **6 · Formas de aprender** | N21–N28 | ¿De dónde puede salir la señal? |
| **7 · El pipeline completo** | N29–N30 | ¿Sé encadenarlo todo? |

**Los bloques 0–4 son la inversión que hace rentable todo lo demás.** Son los menos
vistosos y los que no se pueden saltar.

---
---

# Parte IV · Cómo trabajar

## Reglas del itinerario

### Antes de cada notebook

- [ ] Escribir la **hipótesis** en la bitácora. Antes de ejecutar.
- [ ] Anotar qué es un **resultado esperado** y qué sería sorprendente.

### Dentro de cada notebook

- [ ] Sobreajustar 10 muestras antes de creerte nada.
- [ ] Comprobar la pérdida inicial contra el valor teórico.
- [ ] Al menos **una** cosa rota a propósito.

### Al cerrar

- [ ] Resultado, qué aprendí, **qué me sorprendió**, siguiente paso.
- [ ] Si aparece una pregunta abierta → sube al **Anexo A**.
- [ ] Una línea en el índice de la bitácora.

---

## Qué NO hacer

**No saltarse los bloques 0–4 porque son aburridos.** Son la inversión que hace rentable
todo lo demás. Sin instrumentos, el bloque 6 es tiempo perdido.

**No perseguir el estado del arte.** Ningún notebook aquí busca un buen resultado. Buscan
un buen **entendimiento**. Un 82 % explicado vale más que un 94 % inexplicado.

**No usar modelos grandes.** Todo cabe en CPU o en una GPU modesta. Si algo tarda más de
diez minutos, hazlo más pequeño: la velocidad de iteración manda sobre el realismo.

**No comparar sin igualar el presupuesto de ajuste.** Cada vez que compares dos cosas,
dales el mismo número de intentos. Es la trampa más fácil de caer y ya la habrás visto
en N09.

**No hacer un notebook sin bitácora.** El notebook es el experimento; la bitácora es el
aprendizaje. Se te olvidará el 80 % de lo que descubras si no lo escribes ese día.

---

---

## Cómo saber que vas bien

No por cuántos experimentos hayas hecho. Por si puedes responder a esto sin buscarlo:

- [ ] Miro una curva de pérdida y sé si es sana.
- [ ] Un modelo no aprende: sé en qué orden mirar.
- [ ] Sé cuánto se mueven mis resultados por azar.
- [ ] Puedo parar un entrenamiento, mirar dentro y decidir qué cambiar.
- [ ] Distingo "funciona" de "he demostrado que funciona".
- [ ] Cuando un resultado es demasiado bueno, sospecho antes de celebrarlo.

Los dos últimos son los que separan a alguien que entrena modelos de alguien que
investiga.

---

---

## Cinco advertencias

**1. No pases a la fase siguiente porque la anterior sea aburrida.** La fase 1 es la más
aburrida y la más importante.

**2. Los datasets simples esconden cosas.** Lo simétrico y lo convexo ocultan justo lo
que quieres estudiar. Cuando quieras ver un fenómeno de verdad, desbalancea, mete ruido,
rompe la simetría.

**3. Un experimento que funciona a la primera es sospechoso.** Suele significar que
mides algo más fácil de lo que crees.

**4. Escribe la hipótesis antes.** Siempre. Es lo único que impide construir la
explicación después de ver el resultado.

**5. Esto es un maratón, y un maratón no se corre con sueño.** La parte replicable se
puede hacer en huecos; la de criterio, no. Un bloque largo protegido al mes rinde más
que treinta noches sueltas.

---

---
---

# Parte V · Decisiones

## Puntos abiertos para decidir

Cosas que conviene fijar pronto y que aún no están decididas:

- [x] **¿Qué lenguaje sintético?** ✅ **Aritmética.** Ver la especificación del dataset.
- [ ] **¿Hasta dónde llega el carril B (datos reales)?** Propuesta: solo N15, N17, N23.
- [ ] **¿PyTorch puro o alguna capa por encima?** Recomendación: puro hasta N20, y
      considerar `trl`/`peft` a partir de N24 para no reimplementar DPO.
- [ ] **¿Dónde se guardan los notebooks?** `04-proyecto/09-indice-notebooks/` con el
      mismo esquema de numeración.
- [ ] **¿Hay algún bloque que quieras adelantar?** El orden es de dependencias, no
      dogma — pero el bloque 2 antes del 5 sí es innegociable.

---

## Después de esto: la fase 2

Cuando los treinta notebooks estén hechos, el itinerario **se repite** — no se amplía.

```none
FASE 1 (esta)        aritmética · CPU · portátil
                     objetivo: entender los MECANISMOS
        ↓
FASE 2 (después)     datasets reales · GPU de la universidad
                     objetivo: ver qué NO capturaba el juguete
```

La estructura de N21–N30 es idéntica con un modelo de 100k parámetros que con uno
preentrenado de 1B. Solo cambia la escala.

**Y ese es el argumento de fondo para hacerlo primero en juguete:** llegas al cómputo de
la universidad sabiendo **qué medir**, en vez de gastando horas de clúster aprendiendo
qué mirar.
