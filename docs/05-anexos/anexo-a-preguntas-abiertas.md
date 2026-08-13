---
title: "Anexo A · Preguntas abiertas"
tags: [anexo, preguntas-abiertas, tesis, registro]
status: vivo
---

# Anexo A · Preguntas abiertas

> **Qué es este documento.** El registro consolidado de todo lo que sabemos que **no
> está resuelto**, recogido de todas las secciones del libro. No hay teoría ni
> bibliografía aquí: para eso está cada sección. Aquí solo están las preguntas, con lo
> que ya sabemos de ellas y qué haría falta para atacarlas.
>
> **Es un documento vivo.** Cada vez que una sección deje una pregunta abierta, sube
> aquí. Cada vez que un experimento cierre una, márcala y anota el resultado.

---

## Cómo leer este anexo

Cada pregunta lleva tres etiquetas:

| Etiqueta | Significado |
|---|---|
| **Madurez** | 🌱 idea suelta · 🌿 acotada · 🌳 lista para atacar |
| **Coste** | 💧 tarde suelta · 💧💧 semanas · 💧💧💧 meses |
| **Dependencia** | Qué tiene que existir antes de poder tocarla |

Y un campo que importa más que los tres: **qué haría falta para responderla**. Si no
sabes escribirlo, la pregunta todavía no está acotada.

---
---

# Bloque 1 · Las preguntas de la tesis

Las tres que salieron de la fase de exploración. Ordenadas por lo prometedoras que
parecen hoy, no por lo definitivas que sean.

---

## A1 · ¿Cuándo basta una ronda?

**🌳 madura · 💧💧 semanas · sin dependencias**

### La pregunta

En un data space cada ronda de entrenamiento cuesta K trabajos de cómputo. ¿Cuántas
rondas necesita de verdad un modelo, y se puede saber **antes** de gastarlas?

### Qué sabemos ya

- Con modelos convexos, **una ronda basta**. Medido: treinta rondas no mejoraron nada
  sobre una sola, y la agregación en una ronda alcanzó el techo centralizado.
- Con redes profundas la garantía desaparece: las soluciones locales caen en cuencas
  distintas y promediarlas no tiene por qué funcionar.
- **La accuracy no avisa.** En el experimento de heterogeneidad, la accuracy apenas se
  movía mientras la divergencia entre modelos locales se multiplicaba por nueve.

### Qué falta

La otra mitad de la curva: **rondas → calidad con red profunda**, en función de
heterogeneidad y tamaño de silo. Y de ahí, un criterio *a priori*: mirar algo antes de
entrenar que prediga cuántas rondas harán falta.

### Por qué es nuestra

La motivación habitual de la literatura de una sola ronda es el ancho de banda en
móviles. La nuestra es el coste por trabajo y la gobernanza: son compromisos distintos,
y ese encuadre no lo ha hecho nadie.

### Qué haría falta

- [ ] Sustituir el modelo lineal por uno profundo en el laboratorio existente.
- [ ] Barrer heterogeneidad y medir la curva completa.
- [ ] Buscar un indicador precoz (divergencia entre locales, cobertura conjunta) que
      correlacione con "cuántas rondas hacen falta".

---

## A2 · Agregación robusta con cegado: ¿son compatibles?

**🌿 acotada · 💧💧💧 meses · toca criptografía**

### La pregunta

Los métodos de agregación robusta necesitan **comparar los modelos entre sí** para
detectar al malicioso. El cegado con ruido lo impide: quien agrega ve números sin
sentido. ¿Se pueden tener las dos cosas?

### Qué sabemos ya

- El cegado funciona y no cuesta calidad: la cancelación es exacta hasta precisión de
  máquina.
- El cegado protege de que alguien **lea** un modelo local. No protege de que alguien
  **envíe uno envenenado**: son amenazas distintas y no hay defensa común.
- Con el esquema actual, el que agrega no puede ni siquiera detectar un modelo absurdo.

### Qué falta

Una de dos salidas:

1. Un esquema de cegado que **preserve distancias** entre modelos, para poder comparar
   sin ver.
2. Un criterio de robustez que **no requiera comparar** — por ejemplo, basado en algo
   que el propio silo pueda demostrar de sí mismo.

### Aviso

Es la más difícil de las tres y la que más se aleja de nuestro terreno. Si se elige,
hace falta apoyo de alguien que sepa criptografía.

### Qué haría falta

- [ ] Formalizar exactamente qué necesita comparar cada método de agregación robusta.
- [ ] Probar si alguna transformación del cegado conserva lo mínimo necesario.
- [ ] Verificar si el problema ya está resuelto en la literatura de agregación segura.

---

## A3 · ¿Qué significa promediar funciones de valor sobre soportes distintos?

**🌱 idea · 💧💧💧 meses · es el núcleo conceptual**

### La pregunta

En aprendizaje por refuerzo offline, cada silo aprende una función de valor sobre la
región del espacio que **sus** datos cubren. Promediarlas coordenada a coordenada es lo
que hace el algoritmo estándar. ¿Significa eso algo?

### Qué sabemos ya

- El promedio de dos estimaciones pesimistas sobre regiones **diferentes** no es una
  estimación pesimista sobre la unión. Eso es casi seguro.
- El error de extrapolación se agrava al federar: no hay un desplazamiento de
  distribución, hay N desplazamientos distintos entre sí.
- Ni la privacidad, ni los adaptadores de bajo rango, ni el esquema de una sola ronda
  responden a esto. Solo lo ocultan mejor.

### El mismo problema, en otra forma

Aparece también con adaptadores de bajo rango: la media de dos matrices de rango *r* no
tiene rango *r*, y promediar los factores por separado no equivale a promediar el
producto. **Es la misma pregunta con otro ropaje**, y eso sugiere que hay algo general
debajo.

### Qué falta

Prácticamente todo. Es la más conceptual y la menos acotada. Un primer paso realista
sería el caso tabular: construir dos silos con soportes deliberadamente disjuntos y ver
qué produce el promedio comparado con lo que produciría un agregador consciente de la
cobertura.

### Qué haría falta

- [ ] Un ejemplo mínimo tabular donde el promedio ingenuo falle de forma visible.
- [ ] Enumerar qué alternativas a la media existen y cuáles son computables sin ver los
      datos ajenos.
- [ ] Decidir si la conexión con el problema de los adaptadores es real o una analogía.

---
---

# Bloque 2 · Evaluación

El hilo que atraviesa toda la tesis. Si no se resuelve, no se puede afirmar nada.

---

## A4 · Seleccionar política sin poder desplegarla

**🌳 madura · 💧💧 semanas**

### La pregunta

Se entrenan cuarenta políticas variando hiperparámetros. ¿Cuál se despliega, si no se
puede probar ninguna?

### Qué sabemos ya

- La pérdida **no sirve** como sustituto: una función de valor puede tener error bajísimo
  y producir una política pésima, porque el error se mide sobre las acciones que están
  en los datos y la política actúa sobre las que no.
- Hay una circularidad conocida: los estimadores de evaluación tienen a su vez
  hiperparámetros que habría que elegir evaluándolos.
- **Ordenar es más fácil que valorar.** Para elegir hiperparámetros solo hace falta el
  orden, no el número absoluto.

### Qué falta

Un protocolo defendible, con su condición de validez y su medida de confianza.

### Qué haría falta

- [ ] Fijar qué estimador se usa como base y justificarlo.
- [ ] Medir correlación de rangos entre el orden estimado y el verdadero, en un entorno
      donde el verdadero se conozca.
- [ ] Comparar contra líneas base honestas: elegir al azar, y elegir siempre el más
      conservador.

---

## A5 · ¿Sirve un silo retenido como conjunto de validación?

**🌳 madura · 💧💧 semanas · depende de A4**

### La pregunta

Con cuatro silos, entrenar con tres y evaluar en el cuarto. ¿Funciona, y hasta qué
heterogeneidad aguanta?

### Qué sabemos ya

- El silo retenido **no da una métrica**: da datos sobre los que ejecutar un estimador.
  Aporta independencia, no facilidad.
- **El problema del soporte no se resuelve, se mueve.** Antes había extrapolación al
  aprender; ahora la hay al evaluar.
- Hay una asimetría cruel: **cuanto mejor es la política, menos evaluable es**, porque
  más se aleja de lo que el silo evaluador vio.
- Al rotar los pliegues no se evalúa la misma política cuatro veces: se selecciona un
  procedimiento, no una política.

### Qué falta

El umbral. ¿A partir de qué divergencia entre silos deja de ser fiable el orden
estimado? Ese número es el resultado.

### Qué haría falta

- [ ] Un entorno sintético donde el orden verdadero se pueda calcular exactamente.
- [ ] Medir la correlación de rangos frente a la divergencia entre silos.
- [ ] Comprobar si la cobertura explica el fallo mejor que la divergencia.

---

## A6 · ¿Se puede evaluar con el resultado cegado?

**🌱 idea · 💧 tarde suelta · depende de A5**

### La pregunta

Si el trabajo de evaluación devuelve un escalar **cegado con ruido**, ¿se puede seguir
ordenando políticas?

### Por qué importa

Une los dos hilos: privacidad y evaluación. Si funciona, se puede comprar evaluación sin
que el evaluador sepa qué está evaluando ni el comprador vea el número intermedio.

### Qué haría falta

- [ ] Aplicar el esquema de cegado al escalar de salida y ver si el orden sobrevive.
- [ ] Es un experimento de una tarde sobre el laboratorio existente.

---

## A7 · ¿Cómo se elige el silo evaluador?

**🌱 idea · 💧💧 semanas · depende de A5**

### La pregunta

No todos los silos evalúan igual de bien la misma política. ¿Cuál se elige, y se pueden
combinar varios?

### La tensión

El silo cuya política de comportamiento más se parece a la evaluada es el que mejor la
evalúa — y también el que menos independencia aporta. Hay un compromiso, y no está
caracterizado.

### La conexión con el pago

Si el valor de un silo como evaluador depende de su cobertura, **la cobertura es lo que
se está comprando**. Eso conecta directamente con A8.

### Qué haría falta

- [ ] Medir si combinar varios silos evaluadores mejora la cobertura conjunta.
- [ ] Ver si ponderar por cobertura mejora el orden respecto a promediar sin más.

---
---

# Bloque 3 · Economía y gobernanza

Lo que la literatura de aprendizaje federado asume resuelto y no lo está.

---

## A8 · Atribución de contribución bajo cegado

**🌿 acotada · 💧💧💧 meses**

### La pregunta

Para pagar a cada participante hay que saber cuánto aportó. Medir la aportación exige
evaluar su modelo por separado. **El cegado lo impide.** ¿Cómo se reparte entonces?

### Qué sabemos ya

- Es la misma tensión que A2, en otra forma: privacidad contra la capacidad de
  distinguir contribuciones individuales.
- La literatura de valoración de datos lleva años siendo teórica porque no había dónde
  cobrar. Aquí sí hay raíles de pago reales.
- Con ruido calibrado fuerte, los métodos de atribución basados en gradientes dejan de
  ser fiables.

### Qué haría falta

- [ ] Enumerar qué métodos de atribución existen y cuáles necesitan ver el modelo.
- [ ] Probar si la atribución basada en **evaluación** (cuánto mejora el global al
      incluir a un silo) sobrevive al cegado, ya que solo requiere escalares.
- [ ] Diseñar una demostración mínima: tres silos, un reparto, verificable por todos.

---

## A9 · ¿Cómo se expresa un presupuesto de cómputo en un contrato?

**🌱 idea · 💧💧 semanas**

### La pregunta

Los lenguajes de política de uso de los espacios de datos expresan permisos de acceso.
¿Pueden expresar *"máximo N rondas"*, *"máximo N tokens de razonamiento"*, *"solo estas
herramientas"*?

### Por qué importa ahora

Con los modelos que gastan cómputo variable al responder, el coste de inferencia deja de
ser una decisión de producto y pasa a ser algo que hay que negociar. Nadie ha escrito
sobre eso en el contexto de espacios de datos.

### Qué haría falta

- [ ] Comprobar qué permite expresar la especificación actual de políticas de uso.
- [ ] Escribir un ejemplo de cláusula y ver si es verificable automáticamente.

---

## A10 · El derecho a evaluar frente al derecho a entrenar

**🌱 idea · 💧 tarde suelta**

### La observación

Un trabajo de entrenamiento devuelve un modelo; uno de evaluación devuelve un número. La
superficie de exposición es incomparable.

**Consecuencia:** hay muchos más participantes dispuestos a vender evaluación que
entrenamiento. Eso hace el mercado más líquido, y sugiere una vía de entrada a un
consorcio: empezar vendiendo evaluación y, cuando haya confianza, entrenamiento.

### Qué haría falta

- [ ] Verificar si esa distinción es expresable como dos permisos distintos.
- [ ] Es más una observación de diseño que un experimento, pero merece un párrafo en la
      motivación.

---

## A11 · Revocación: ¿se puede desaprender lo que aportó un silo?

**🌱 idea · 💧💧💧 meses**

### La pregunta

Si un participante revoca su permiso, hay que quitar su contribución del modelo. En
clasificación ya es difícil. **En una política aprendida de trayectorias, no está
estudiado.**

### Qué falta

Todo. Ni siquiera está claro qué significaría "desaprender" una trayectoria de una
función de valor.

### Qué haría falta

- [ ] Definir qué significa revocación en este contexto: ¿que el modelo no dependa de
      esos datos, o que se pueda demostrar que no depende?
- [ ] Ver qué hace la literatura de desaprendizaje en el caso supervisado y si algo
      traslada.

---

## A12 · ¿Se puede demostrar que un silo entrenó lo que dijo?

**🌱 idea · 💧💧 semanas · depende de verificar la infraestructura**

### La pregunta

Nada impide que un silo devuelva una actualización inventada. En federado convencional
esto es irresoluble sin criptografía cara. Pero la infraestructura de cómputo sobre datos
ajenos **puede atestiguar** que ejecutó una imagen concreta sobre unos datos concretos.

### Qué falta

Saber qué garantiza realmente la infraestructura. **Es una verificación, no una
investigación**, y hasta hacerla la pregunta no se puede acotar.

### Qué haría falta

- [ ] Leer qué firma exactamente el nodo y qué se puede verificar desde fuera.
- [ ] Si atestigua algo útil, formular qué propiedad de seguridad se obtendría.

---
---

# Bloque 4 · Preguntas del camino

Las que salieron construyendo la base y que **no son de la tesis**, pero merecen no
perderse. Muchas son experimentos de una tarde.

---

## A13 · ¿Qué esconde un experimento simétrico?

**💧 tarde suelta**

Dos veces ya, un fallo real quedó invisible porque el caso de prueba era simétrico:

- Un error de ponderación que no se manifestaba con silos del mismo tamaño.
- Una brecha de generalización que desaparecía al reajustar la tasa de aprendizaje.

**La pregunta general:** ¿qué otras propiedades de los experimentos de juguete ocultan
fallos sistemáticamente? Sería útil una lista de comprobación propia.

---

## A14 · ¿Cuándo la accuracy oculta el problema?

**💧 tarde suelta**

En el experimento de heterogeneidad la accuracy no se movía y la divergencia se
multiplicaba por nueve. **La métrica que todo el mundo mira era ciega al fenómeno.**

¿Qué otras métricas de diagnóstico deberían estar en el registro por defecto? Candidatas:
divergencia entre modelos locales, norma del gradiente, cobertura, dispersión entre
semillas.

---

## A15 · ¿Aparece el doble descenso en nuestros experimentos?

**💧 tarde suelta**

El error de validación puede subir y volver a bajar al entrenar más. Si es así, parar
en el primer repunte es parar en el peor sitio.

**Nunca lo hemos comprobado en nuestros propios modelos.** Es entrenar el doble de
épocas y mirar.

---

## A16 · ¿Cuánto se mueven nuestros resultados solo con la semilla?

**💧 tarde suelta**

Todas las conclusiones de los laboratorios se sacaron de una sola ejecución. **No sabemos
cuál es el ruido de fondo.**

Hasta tener ese número, cualquier diferencia observada entre dos métodos podría ser
ruido. Es la comprobación más barata y la que más protege.

---

## A17 · ¿Existe un verificador para nuestro dominio?

**💧 tarde suelta · pregunta conceptual**

Los modelos de razonamiento funcionan porque el verificador **es un entorno**: barato,
ilimitado, con verdad objetiva. Nuestro problema no lo tiene.

**La pregunta útil:** ¿hay alguna parte del problema que sí admita verificación
automática? Aunque sea parcial, cambiaría el planteamiento.

---

## A18 · ¿Qué tarea pretexto tendrían nuestros datos?

**💧💧 semanas**

El aprendizaje auto-supervisado fabrica etiquetas del propio dato. Con trayectorias,
series temporales o registros, ¿cuál sería la tarea pretexto?

Y la comprobación importante: **buscar activamente el atajo que la rompería**. Una tarea
pretexto que funciona demasiado bien y demasiado pronto suele estar enseñando el atajo.

---

## A19 · ¿Se puede evaluar la transferibilidad como resultado principal?

**💧💧 semanas · relacionada con A5**

Al rotar pliegues, la dispersión entre ellos es una señal de si la política aguanta fuera
del sitio donde se entrenó. Eso es **la pregunta que hace un comité clínico**, no la que
hace un revisor de aprendizaje automático.

¿Podría ser el resultado principal en vez de un subproducto?

---

## A20 · ¿Cuánto cuesta realmente nuestro experimento completo?

**💧 tarde suelta**

Nunca hemos calculado el coste en trabajos de cómputo de la rejilla completa: silos ×
configuraciones × pliegues × rondas.

Es una multiplicación, y puede que el resultado obligue a rediseñar el experimento antes
de lanzarlo. **Hacerla antes es gratis; después, no.**

---
---

# Registro de preguntas cerradas

> Cuando un experimento responda a una pregunta, muévela aquí con el resultado. Este
> apartado es el que demuestra avance, y el que se convierte en la sección de
> contribuciones.

*(Vacío. Primera entrada pendiente.)*

| # | Pregunta | Cerrada el | Resultado | Dónde está la evidencia |
|---|---|---|---|---|
| — | — | — | — | — |

---
---

# Cómo mantener este anexo

**Al abrir una pregunta:** entra en el bloque que le corresponda, con las tres etiquetas
y el campo de "qué haría falta". Si no sabes escribir ese campo, aún no está acotada:
déjala en 🌱 y anota solo la intuición.

**Al cerrar una:** muévela al registro de abajo con el resultado, **aunque sea
negativo**. Un negativo bien medido es un resultado.

**Cada tres meses:** relee el anexo entero. Las preguntas que lleven un año en 🌱 sin
moverse probablemente no eran preguntas, sino intuiciones. Bórralas sin pena.

**Regla de oro:** este documento crece hacia abajo (preguntas cerradas), no hacia arriba
(preguntas nuevas). Si al cabo de seis meses hay más arriba que abajo, se está explorando
en vez de investigando.

<!-- nav-start -->

---

← Anterior: [14.7-recursos.md](../04-proyecto/14-validacion-por-silo-retenido/14.7-recursos.md)  
Siguiente: [B. Línea temporal](anexo-b-linea-temporal.md) →

<!-- nav-end -->
