---
title: "4. Formas de aprender"
tags: [paradigmas, indice]
status: completo
eje: "¿de dónde sale la señal de aprendizaje?"
---

# 4. Formas de aprender

> **Eje de este capítulo:** ¿de dónde sale la señal de aprendizaje?

Una red neuronal no "aprende" en abstracto: ajusta sus pesos para reducir un error.
Ese error tiene que salir de algún sitio, y **de dónde sale es la decisión de diseño
más importante de todo el proyecto**, porque determina qué necesitas tener antes de
empezar: ¿anotadores humanos? ¿un corpus crudo? ¿un simulador? ¿un modelo maestro?

Esa es la pregunta que organiza las secciones de abajo.

---

## Aviso previo: las fronteras son convenio, no naturaleza

Si consultas tres libros vas a encontrar tres taxonomías distintas, y no es porque
uno esté mal. Las categorías se solapan por construcción:

- El **auto-supervisado es supervisado**: la única diferencia es que las etiquetas
  las fabrica el propio dato en vez de una persona. Durante años se clasificó dentro
  de "no supervisado", y aún verás esa convención en textos anteriores a ~2018.
- El **aprendizaje por preferencias entrena un modelo de recompensa de forma
  supervisada** y luego lo usa como señal de refuerzo. Vive a caballo entre dos filas.
- La **imitación** es formalmente supervisión (predecir la acción del experto) pero
  se estudia dentro de RL porque el problema que resuelve es de control secuencial.

Trata la tabla como un **mapa de fuentes de señal**, no como una clasificación
excluyente. Un sistema real usa varias, encadenadas (→ [§4.8](4.8-el-pipeline-real.md)).

---

## Vista general

| Forma | La señal viene de… | Cuello de botella | Ejemplo canónico |
|---|---|---|---|
| [Supervisado](4.1-con-etiquetas-supervisado.md) | Etiquetas humanas | Anotar es caro y no escala | ImageNet → ResNet |
| [No supervisado](4.2-sin-etiquetas-no-supervisado.md) | La estructura estadística | El objetivo es difuso | k-means, PCA |
| [Auto-supervisado](4.3-con-el-propio-dato-auto-supervisado.md) | El propio dato (tarea pretexto) | Necesitas escala brutal | Siguiente token → GPT |
| [Imitación](4.4-copiando-a-un-experto-imitacion.md) | Demostraciones de un experto | Conseguir al experto | Behavioral cloning |
| [Preferencias](4.5-comparando-opciones-preferencias.md) | Comparaciones humanas (A vs B) | Anotadores lentos y ruidosos | RLHF, DPO |
| [**Refuerzo**](4.6-por-prueba-y-error-refuerzo.md) ★ | Interactuar y recibir recompensa | Hace falta un entorno | AlphaGo, RLVR |
| [Destilación](4.7-de-otro-modelo-destilacion.md) | Otro modelo ya entrenado | Necesitas al maestro | Modelos "mini" |

★ **El refuerzo es la sección clave de este proyecto.** Las secciones 4.4 y 4.5 son
su antesala conceptual; conviene leerlas antes.

---

### Los modelos de razonamiento no son una fila de esta tabla

Merecen espacio en el capítulo, pero **no como octava fuente de señal**, porque no
lo son. Su aprendizaje ya está en la tabla: es la combinación de **Refuerzo sobre
recompensas verificables (§4.6)**, **Imitación (§4.4)** y **Destilación (§4.7)**.

Lo nuevo en ellos ocurre **después** del entrenamiento, cuando los pesos ya están
congelados: **cuánto cómputo se gasta al responder**. Eso es otra pregunta, y
merece un eje propio (→ [§4.9](4.9-modelos-de-razonamiento.md)).

```none
    Los siete de la tabla         Los modelos de razonamiento
    ─────────────────────         ───────────────────────────
    ¿de dónde sale el error       ¿cuánto cómputo se gasta
     que ajusta los pesos?         al producir la respuesta?
            │                              │
    eje ① de → 1.5                 eje ④, que → 1.5 no tenía
```

→ [§4.9 Coda: cuando el cómputo se mueve a la inferencia](4.9-modelos-de-razonamiento.md)

**Orden de lectura sugerido:** 4.1 → 4.2 → 4.3 (síntesis de las dos anteriores) →
4.4 → 4.5 → 4.6 → 4.7 → 4.8 → 4.9.

---

## 4.1 Con etiquetas: supervisado

**La señal:** un humano ha escrito la respuesta correcta para cada ejemplo. El modelo
predice, se compara con la etiqueta, y la diferencia es el error.

**Necesitas:** un dataset `(entrada, etiqueta)`. Nada más.

**Por qué se estudia primero:** es el caso más limpio para entender qué es una función
de pérdida y qué hace el descenso de gradiente. Todo lo demás son variaciones sobre
"de dónde saco esa etiqueta".

**El techo:** el coste de anotación crece linealmente con el tamaño del dataset, y la
calidad humana es un límite duro. No puedes anotar un billón de ejemplos. Por eso el
supervisado puro dejó de ser el motor principal de los modelos grandes.

**Dónde sigue siendo el rey:** diagnóstico médico, control de calidad industrial,
cualquier dominio donde el etiquetado sea barato o ya exista por otro motivo
(históricos, registros clínicos, logs).

→ [4.1-con-etiquetas-supervisado.md](4.1-con-etiquetas-supervisado.md)

---

## 4.2 Sin etiquetas: no supervisado

**La señal:** ninguna externa. Solo la estructura estadística de los propios datos.
El modelo busca agrupaciones, ejes de variación, densidades.

**Necesitas:** datos crudos.

**El problema real:** no hay respuesta correcta, así que **no hay métrica objetiva de
éxito**. Un clustering "bueno" lo es porque a alguien le resulta útil, no porque
minimice algo verificable. Esto lo convierte más en herramienta exploratoria que en
paradigma de entrenamiento para redes profundas.

**Ojo con la confusión clásica:** mucha literatura antigua mete aquí lo que hoy
llamamos auto-supervisado. Si un texto dice "no supervisado" y describe un autoencoder
o predicción de palabras, está usando la convención vieja.

→ [4.2-sin-etiquetas-no-supervisado.md](4.2-sin-etiquetas-no-supervisado.md)

---

## 4.3 Con el propio dato: auto-supervisado

**La señal:** el dato se etiqueta a sí mismo. Tapas una parte y pides al modelo que
la reconstruya. La parte tapada *es* la etiqueta, y es gratis.

**Necesitas:** una montaña de datos crudos y una **tarea pretexto** bien diseñada
(predecir el siguiente token, rellenar huecos, reconstruir una imagen enmascarada,
acercar dos vistas de la misma foto).

**Por qué lo cambió todo:** rompe el techo de 4.1. Si la etiqueta sale del dato, puedes
entrenar con todo internet. Es el motor del preentrenamiento de absolutamente todos los
modelos de fundación actuales.

**La idea clave a transmitir:** la tarea pretexto no importa por sí misma. A nadie le
interesa un modelo que prediga la siguiente palabra. Importa porque **para hacerlo bien
hay que entender el dominio**, y ese entendimiento queda en las representaciones
internas, listo para reutilizarse.

**El nuevo cuello de botella:** ya no son los anotadores, es el cómputo y la calidad
del corpus.

→ [4.3-con-el-propio-dato-auto-supervisado.md](4.3-con-el-propio-dato-auto-supervisado.md)

---

## 4.4 Copiando a un experto: imitación

**La señal:** trayectorias completas de alguien que ya sabe hacer la tarea. No etiquetas
sueltas, sino secuencias de `(situación → acción)`.

**Necesitas:** un experto dispuesto a ser grabado, o registros de su actividad.

**Por qué merece fila propia:** es el **puente entre supervisado y refuerzo**.
Técnicamente es supervisión (predice la acción del experto), pero el problema es
secuencial: tus errores cambian el mundo que verás después.

**El fallo característico — *distribution shift*:** el modelo solo ha visto estados por
los que pasa un experto. En cuanto se desvía un poco, entra en territorio desconocido,
se equivoca más, se desvía más. Los errores se componen. Este fallo es exactamente el
motivo por el que existe el refuerzo.

**Variantes que conviene nombrar:** *behavioral cloning* (copiar la acción directamente)
frente a *inverse reinforcement learning* (deducir qué recompensa explicaría la conducta
del experto, y luego optimizarla).

→ [4.4-copiando-a-un-experto-imitacion.md](4.4-copiando-a-un-experto-imitacion.md)

---

## 4.5 Comparando opciones: preferencias

**La señal:** un humano ve dos salidas del modelo y dice cuál prefiere. No puntúa, no
corrige: **compara**.

**Necesitas:** un modelo que ya genere salidas razonables (viene de 4.3) y anotadores
que las comparen.

**Por qué existe esta categoría:** para tareas subjetivas —"escribe una respuesta útil
y no ofensiva"— nadie sabe escribir la etiqueta correcta ni definir una recompensa
numérica. Pero **cualquiera sabe decir cuál de dos respuestas es mejor**. Es una fuente
de señal genuinamente distinta a las anteriores.

**Cómo se convierte en entrenamiento:**
- **RLHF** — con las comparaciones entrenas un *modelo de recompensa*, y con él haces
  refuerzo sobre el modelo principal. Dos etapas.
- **DPO y derivados** — optimizas directamente contra las preferencias, sin modelo de
  recompensa intermedio. Más simple y más estable.

**El riesgo a explicar sí o sí — *reward hacking*:** el modelo de recompensa es un
sustituto imperfecto del juicio humano. Si optimizas demasiado contra él, el modelo
aprende a explotar sus defectos en lugar de a mejorar de verdad (respuestas largas,
aduladoras y vacías que puntúan alto). Es el mismo fenómeno que verás en 4.6, aquí en
su forma más visible.

→ [4.5-comparando-opciones-preferencias.md](4.5-comparando-opciones-preferencias.md)

---

## 4.6 Por prueba y error: refuerzo ★

**La señal:** el modelo actúa en un entorno y recibe una recompensa. Nadie le dice cuál
era la acción correcta; solo si el resultado fue bueno o malo, y a menudo mucho después.

**Necesitas:** un **entorno** con el que interactuar (simulador, juego, verificador,
sistema real) y una función de recompensa.

**Qué lo hace distinto de todo lo anterior — tres dificultades acumuladas:**

1. **Asignación de crédito.** La recompensa llega al final. ¿Cuál de las 200 acciones
   previas fue la buena? Este es *el* problema del refuerzo.
2. **Exploración vs explotación.** Para descubrir algo mejor hay que probar cosas peores.
   Nadie te da los datos: te los generas tú, y si exploras mal nunca los verás.
3. **Los datos no son i.i.d.** Tu política determina qué datos recoges, y esos datos
   cambian tu política. El suelo se mueve mientras caminas. De ahí la inestabilidad
   crónica del RL frente a la placidez del supervisado.

**Por qué importa tanto ahora:** es el único paradigma que puede superar al humano en
lugar de imitarlo. En supervisado el techo es el anotador. En refuerzo el techo es el
entorno.

**RLVR (recompensa verificable)** — merece atención propia dentro de esta sección. Si
la tarea tiene una respuesta comprobable automáticamente (tests unitarios que pasan,
una demostración matemática que compila, un resultado numérico correcto), la recompensa
sale gratis y sin anotadores. Es donde está concentrada buena parte de la actividad
actual, y esquiva el cuello de botella de 4.5.

**Conceptos que la sección debe cubrir:** política, valor, recompensa, entorno, episodio;
on-policy vs off-policy; policy gradient frente a métodos basados en valor; por qué la
recompensa no es diferenciable y qué se hace al respecto; *reward hacking*.

→ [4.6-por-prueba-y-error-refuerzo.md](4.6-por-prueba-y-error-refuerzo.md)

---

## 4.7 De otro modelo: destilación

**La señal:** las salidas de un modelo maestro ya entrenado. El alumno no imita las
etiquetas duras, sino la **distribución completa** del maestro, que contiene mucha más
información ("esto es un 7, pero se parece bastante a un 1").

**Necesitas:** acceso al maestro. Ese es todo el cuello de botella.

**Por qué es una fuente distinta y no un truco de ingeniería:** el conocimiento no
proviene del mundo ni de un humano, sino de otra red. Esto abre posibilidades que no
existen en las demás filas —autodestilación, alumnos que superan al maestro, generación
de datos sintéticos— y también problemas propios: los errores del maestro se heredan, y
entrenar generaciones sucesivas sobre salidas sintéticas degrada la calidad.

**Usos típicos:** comprimir un modelo grande en uno desplegable; transferir capacidades
de un modelo caro a uno barato; generar corpus sintéticos de entrenamiento.

→ [4.7-de-otro-modelo-destilacion.md](4.7-de-otro-modelo-destilacion.md)

---

## 4.8 El pipeline real: cómo se encadenan

Ningún sistema serio usa un solo paradigma. La secuencia habitual en un modelo de
lenguaje moderno es:

```none
1. Auto-supervisado  (§4.3)  →  capacidad bruta, conocimiento del mundo
2. Imitación / SFT   (§4.4)  →  formato, seguir instrucciones
3. Preferencias      (§4.5)  →  utilidad, tono, seguridad
4. Refuerzo          (§4.6)  →  razonamiento, tareas verificables
   (Destilación      (§4.7)  →  transversal: comprimir o generar datos en cualquier fase)
```

Y la lectura importante de esa secuencia:

- **Cada etapa arregla el fallo de la anterior.** El auto-supervisado da capacidad pero
  no obediencia. La imitación da obediencia pero no criterio. Las preferencias dan
  criterio pero no capacidad nueva. El refuerzo es el único que puede añadir capacidad
  por encima de sus datos.
- **Hay restricciones duras de orden, pero no es una línea recta.** No puedes hacer
  RLHF sobre un modelo que no genera texto coherente: necesitas algo que comparar. Pero
  los pipelines de frontera **iteran** SFT → RL → SFT varias vueltas, y DeepSeek-R1-Zero
  llegó a aplicar RL directamente sobre el modelo base.
- **El coste es descendente en datos y ascendente en dificultad.** Billones de tokens
  crudos, luego miles de demostraciones, luego decenas de miles de comparaciones. Cada
  etapa usa muchos menos datos que la anterior, pero son mucho más difíciles de obtener.

Esta sección es el sitio natural para desmontar la idea de "elegir un paradigma": en la
práctica se **componen**.

→ [4.8-el-pipeline-real.md](4.8-el-pipeline-real.md)

---

## 4.9 Coda: cuando el cómputo se mueve a la inferencia

**No añade una fuente de señal.** Añade una pregunta nueva: ¿cuánto cómputo se gasta
al **responder**?

**Su aprendizaje:** RLVR (§4.6) + SFT (§4.4) + Destilación (§4.7). Nada nuevo.

**Su novedad:** el paso de **Sistema 1** (respuesta rápida autoregresiva, cómputo fijo)
a **Sistema 2** (deliberación, cadena de pensamiento, verificación, cómputo variable y
escalable en tiempo de inferencia).

**Por qué importa para este proyecto:** RLVR funciona porque el verificador **es un
entorno** barato e ilimitado. Es la mejor ilustración posible de lo que se consigue
cuando *tienes* entorno — y por contraste, de por qué el RL offline (→ 1.6) es duro.

→ [4.9-modelos-de-razonamiento.md](4.9-modelos-de-razonamiento.md)

---

## Errores frecuentes que conviene desactivar

| Confusión | Aclaración |
|---|---|
| "Auto-supervisado = no supervisado" | No. El auto-supervisado tiene etiquetas; las fabrica el dato. |
| "RLHF es refuerzo" | Parcialmente. La señal son preferencias humanas; el refuerzo es solo el mecanismo de optimización. |
| "Sin etiquetas no se puede aprender" | El grueso del cómputo de un LLM se gasta sin una sola etiqueta humana. |
| "El refuerzo es para juegos" | El juego es el laboratorio, no el objetivo. Lo relevante es que no existe respuesta correcta que copiar. |
| "Más recompensa = mejor modelo" | Solo si la recompensa mide lo que crees. Ver *reward hacking* en §4.5 y §4.6. |
| "Fine-tuning es un paradigma" | Es una fase del entrenamiento, no una fuente de señal. Cruza con este eje, no compite. |
| "Razonamiento = nuevo paradigma de aprendizaje" | No en el eje de fuente de señal. Es un eje distinto: **cuánto cómputo se gasta al responder** (§4.9). |
| "La cadena de pensamiento explica la respuesta" | **No necesariamente.** La traza visible puede no reflejar el cómputo real (§4.9). |

---

## Recursos transversales del capítulo

Los específicos van en cada sección. Estos cubren el mapa completo:

| Recurso | Formato | Nota |
|---|---|---|
| [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) | 📺🎮 | No cubre paradigmas, pero es el suelo conceptual de todo lo demás (→ 3.1) |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Simon Prince) | 📖 | **Gratuito.** Sus capítulos finales recorren supervisado, no supervisado y RL con figuras |
| [Dive into Deep Learning](https://d2l.ai) | 📖🛠️ | Libro con notebooks. Útil para tocar cada paradigma con código |
| [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) (OpenAI) | 📚🛠️ | Para §4.6. Su *Taxonomy of RL Algorithms* es el mapa que este capítulo imita |
| [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (Karpathy) | 📖 | El protocolo, transversal a cualquier paradigma (→ 3.8) |
| [RLHF Book](https://rlhfbook.com) (Nathan Lambert) | 📖 | **Gratuito y en desarrollo.** El mejor tratamiento unificado de SFT, preferencias, RLHF y RLVR. Cubre §4.4, §4.5, §4.6 y §4.8 de una vez |

---

## Recursos por sección — estado

- [x] §4.6 ★ — Spinning Up, Sutton & Barto, CS285 (→ 1.6)
- [x] §4.8 — Llama 3 report, DeepSeek-R1, alignment-handbook
- [x] §4.9 — visual guides, papers de test-time compute, open-r1
- [x] §4.1 — 3b1b, TF Playground, labelerrors.com, Teachable Machine
- [x] §4.2 — Setosa, Distill *Misread t-SNE*, PAIR *Understanding UMAP*
- [x] §4.3 — Lilian Weng, nanoGPT, Cookbook of SSL, Toolformer
- [x] §4.4 — CS285 clase 2, DAgger, LIMA, datasets de SFT
- [x] §4.5 — RLHF Book, InstructGPT, DPO, *Reward Model Overoptimization*
- [x] §4.7 — Hinton 2015, DistilBERT, colapso por datos sintéticos (Nature 2024)

**Nota:** las referencias están verificadas y enlazadas. Revísalas de todos modos antes
de citarlas en la tesis: los enlaces se mueven y los recursos envejecen.
