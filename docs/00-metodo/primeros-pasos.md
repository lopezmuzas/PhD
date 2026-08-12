---
title: "Primeros pasos · De entrenar a ciegas a hackear el entrenamiento"
tags: [metodo, doctorado, arranque]
status: vivo
---

# Primeros pasos

> **Qué es esto.** No es un plan de cinco años. Es por dónde empezar, dado un punto de
> partida concreto: sabes Python, entiendes backpropagation, has construido modelos
> siguiendo guías, y **entrenas a ciegas y confías**.

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

# Fase 0 · El andamio (una tarde)

Antes del primer experimento, dos cosas. Las dos son aburridas y las dos ahorran meses.

## 0.1 · La bitácora

Ver el documento del sistema. **Empieza con el primer experimento, no cuando "tengas
algo serio"** — el hábito se coge con lo trivial, no con lo importante.

## 0.2 · Un arnés de experimentos reutilizable

No un framework: **un fichero**. Algo que te permita lanzar variaciones sin reescribir el
bucle cada vez.

Lo mínimo que debe hacer:

- Aceptar una configuración (diccionario o YAML) y devolver métricas.
- Fijar la semilla y registrarla.
- Guardar en disco: configuración, métricas por época, y el `state_dict` final.
- Registrar el hash del commit.

Con eso, cada experimento es una línea de configuración y no una copia del notebook
anterior con dos números cambiados. **El coste de un experimento nuevo tiene que ser
cinco minutos, o no harás muchos.**

> Ya tienes MLflow y TensorBoard montados. El problema nunca fue la herramienta: es
> decidir **qué registrar**. Eso viene en la fase 2.

---
---

# Fase 1 · Hacer el modelo transparente

**Objetivo:** poder parar el entrenamiento en cualquier punto y responder *"¿qué está
pasando aquí dentro?"*.

Modelo: un MLP sobre MNIST o Fashion-MNIST. **Deliberadamente aburrido.** Lo interesante
no es el modelo, son los instrumentos.

## E1 · Predecir antes de entrenar

Antes de la primera época, pasa un lote por la red y mira la salida.

**Qué buscas:** con 10 clases y pesos aleatorios, la pérdida inicial debería ser
aproximadamente $\ln(10) \approx 2.30$. Si no lo es, hay algo mal antes de empezar.

**Por qué importa:** es tu primer "número esperado". A partir de aquí, cada vez que veas
una pérdida sabrás si es razonable o si el modelo está haciendo algo raro.

## E2 · Sobreajustar 10 muestras

Coge diez ejemplos y entrena hasta pérdida casi cero.

**Si no lo consigue, hay un bug.** No es un problema de hiperparámetros, no es que
"necesite más datos". Es un bug, y ahora sabes que existe antes de haber gastado horas.

**Conviértelo en un test automático.** Es la prueba que más veces te va a salvar la
semana.

## E3 · Mirar los pesos

Histograma de los pesos de cada capa: antes de entrenar, a mitad, al final.

**Qué buscas:**
- ¿Se mueven todas las capas o solo las últimas?
- ¿Alguna distribución se está yendo a los extremos?
- ¿Alguna se ha quedado exactamente donde empezó?

**Esto es "ver el aprendizaje".** Es lo que te falta, y son diez líneas de matplotlib.

## E4 · Mirar los gradientes

La norma del gradiente **por capa**, en cada paso.

**Qué buscas:** si la norma cae órdenes de magnitud según retrocedes hacia las capas
iniciales, estás viendo el gradiente desvanecerse con tus propios ojos.

Registra también la **norma global** en cada paso. Es el detector precoz: su crecimiento
sostenido suele preceder al colapso.

## E5 · Parar, tocar, seguir

El ejercicio que más se parece a lo que pides.

Entrena 5 épocas. **Para.** Y entonces:

- Predice unos cuantos ejemplos a mano. ¿Cuáles falla? ¿Tienen algo en común?
- Pon a cero los pesos de una capa. ¿Cuánto se degrada?
- Duplica la tasa de aprendizaje. Sigue 5 épocas más. ¿Qué cambia?
- Guarda el checkpoint, cambia una cosa, y compara las dos ramas.

**Aquí es donde se aprende a hackear un modelo.** No hay atajo ni tutorial: es sentarse
una tarde a manosear un modelo entrenado.

---

## Lo que sales sabiendo de la fase 1

- [ ] Qué pérdida es razonable al empezar.
- [ ] Si tu bucle funciona, antes de confiar en él.
- [ ] Cómo se mueven los pesos durante el entrenamiento.
- [ ] Cómo se ve un gradiente sano y uno enfermo.
- [ ] Que puedes intervenir en mitad del entrenamiento sin miedo.

**Duración estimada:** dos o tres sesiones. No corras: esta fase es el cimiento.

---
---

# Fase 2 · Romper a propósito

**Objetivo:** reconocer la firma de cada fallo antes de que te pase de verdad.

Cada uno de estos es una entrada de bitácora corta. Provocas el fallo, anotas su firma,
lo arreglas, anotas qué lo arregló.

| # | Provocar | Firma que buscas |
|---|---|---|
| E6 | Tasa de aprendizaje 100× | Pérdida a NaN. Cuánto tarda |
| E7 | Tasa 1000× más pequeña | Pérdida plana. Distinguirlo de "no aprende por bug" |
| E8 | Red profunda con sigmoides | Gradiente desvanecido. Comparar norma por capa |
| E9 | Sin normalizar las entradas | Convergencia mucho peor con el mismo modelo |
| E10 | Etiquetas barajadas al azar | **Que las ajuste igual.** Regularizar no lo impide |
| E11 | Dataset diminuto, modelo grande | Sobreajuste clásico. Y entrenar el doble para ver si hay doble descenso |
| E12 | Olvidar `model.eval()` | Métricas de validación absurdas |
| E13 | Normalizar antes de partir train/test | Fuga de datos. Resultados demasiado buenos |
| E14 | Duplicados entre train y test | Lo mismo, más silencioso |

**E10 es el más importante de la lista.** Ver con tus propios ojos que la red ajusta
etiquetas aleatorias reencuadra todo lo que creías sobre regularización. Es barato y no
se olvida.

**E13 y E14 son los más rentables a largo plazo**, porque son los que te van a pasar de
verdad y no avisan.

---

## Lo que sales sabiendo de la fase 2

Un catálogo mental de firmas. Cuando dentro de un año un entrenamiento se comporte raro,
vas a reconocer el patrón en vez de tantear.

**Duración estimada:** un mes de sesiones sueltas. Son experimentos de 30–60 minutos.

---
---

# Fase 3 · Aprender a medir

**Objetivo:** dejar de fiarte de un número.

Esta fase es la que más te va a diferenciar, y la que casi nadie hace.

## E15 · Cuánto se mueve todo con la semilla

Ejecuta **exactamente el mismo experimento** cinco veces cambiando solo la semilla.

**Anota la dispersión.** Ese número es tu ruido de fondo. A partir de ahora, cualquier
diferencia menor que eso **no es un resultado**.

Es el experimento más barato y el que más te va a proteger durante cinco años.

## E16 · La línea base tonta

Antes de celebrar un 92 % de accuracy:

- ¿Qué saca predecir siempre la clase mayoritaria?
- ¿Qué saca una regresión logística sobre los píxeles crudos?
- ¿Qué saca un modelo aleatorio?

**Si tu red profunda no bate claramente a eso, no has demostrado nada.** Y a veces no lo
bate, lo cual es un descubrimiento muy útil.

## E17 · Qué esconde la métrica

Coge un problema desbalanceado (o desbalancéalo tú).

- Mira la accuracy. Se ve bien.
- Mira la matriz de confusión. Se ve mal.
- Compara ROC-AUC con precisión-exhaustividad.

**La lección:** la métrica que todo el mundo mira puede ser ciega al fenómeno. Ya te
pasó una vez, con la divergencia entre modelos locales que se multiplicaba por nueve
mientras la accuracy no se movía.

## E18 · Presupuesto de ajuste igualado

Compara dos configuraciones probando **el mismo número de variantes de cada una**.

Luego repite dándole 20 intentos a una y 3 a la otra, y observa cómo cambia la
conclusión. **Eso es cómo se fabrica una mejora falsa**, y verlo una vez te vacuna.

---

## Lo que sales sabiendo de la fase 3

- [ ] Cuál es tu ruido de fondo.
- [ ] Que un número sin intervalo no es un resultado.
- [ ] Que la métrica es una decisión, no un dato.
- [ ] Cómo detectar una comparación injusta, incluida la tuya.

**Duración estimada:** dos o tres semanas.

---
---

# Fase 4 · Subir la escalera de modelos

Solo ahora. Con instrumentos y con criterio.

**La regla:** cada escalón nuevo, repite E2 (sobreajustar 10 muestras), E3 (mirar pesos)
y E4 (gradientes). El instrumental viaja contigo.

| Escalón | Qué añade | Qué romper aquí |
|---|---|---|
| MLP | Ya hecho | — |
| CNN | Compartir pesos, invariancia | Quitar el compartir: ¿cuánto empeora? |
| RNN / LSTM | Secuencia, memoria | Gradiente explotando. Ver el recorte funcionar |
| Transformer pequeño | Atención | Mirar los mapas de atención. ¿Miran lo que crees? |
| mini-GPT | Autoregresivo a escala | Ver la pérdida bajar sin una sola etiqueta |

**Sugerencia de recorrido:** construye un mini-GPT desde cero siguiendo material que
explique línea a línea, pero **con tu instrumental encima**. La diferencia entre seguir
un tutorial y aprender de él es exactamente eso: parar en el minuto 40, mirar los pesos,
y preguntarte por qué son así.

---
---

# Fase 5 · Mezclar formas de aprender

Cuando la escalera esté recorrida. Aquí la pregunta ya no es *"¿funciona?"* sino
*"¿qué aporta cada fase?"*.

| Experimento | Qué responde |
|---|---|
| Preentrenar auto-supervisado, luego sondeo lineal | ¿Cuánto valen las representaciones sin etiquetas? |
| El mismo modelo desde cero vs. preentrenado | ¿Cuánto aporta la fase 1? |
| Auto-supervisado → ajuste supervisado con pocas etiquetas | ¿Cuántas etiquetas me ahorro? |
| Ajustar solo la última capa vs. todo | ¿Dónde vive el conocimiento? |
| Medir el olvido: evaluar en la tarea vieja tras ajustar | ¿Qué se pierde al especializar? |

Y el que enlaza con todo lo demás: **entrenar por refuerzo algo minúsculo con un
verificador**, aunque sea un juego de tres estados. Ver la diferencia entre aprender de
etiquetas y aprender de recompensa con las manos, no en un diagrama.

---
---

# Cómo saber que vas bien

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

# Cinco advertencias

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

# El primer paso, concreto

Este fin de semana:

1. Crear `docs/04-proyecto/10-bitacora-experimentos/INDICE.md` con la tabla vacía.
2. Escribir `2026-08-XX-exp-001-primer-vistazo.md` con la plantilla.
3. Hacer **E1 y E2**: predecir antes de entrenar, y sobreajustar 10 muestras.
4. Cerrar la entrada con el "siguiente paso".

Dos horas. Y ya tienes el sistema en marcha, que es lo que de verdad cuesta arrancar.
