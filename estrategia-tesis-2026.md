---
title: "Estrategia de tesis · Federado sobre data spaces"
tags: [tesis, estrategia, decision, federado, rl-offline]
status: vivo
fecha: 2026-08-10
---

# Estrategia de tesis · Federado sobre data spaces

> **Qué es este documento.** El contexto completo de la fase de exploración:
> diagnóstico del campo, opciones de enfoque, ideas con potencial de impacto, y
> la recomendación concreta. Es un documento **vivo**: se relee al decidir, no se
> archiva.
>
> **Fecha de corte de la exploración: finales de agosto de 2026.**

---

## Índice

- [Parte A — Diagnóstico del campo](#parte-a--diagnóstico-del-campo)
- [Parte B — Diez ideas para enfocar la tesis](#parte-b--diez-ideas-para-enfocar-la-tesis)
- [Parte B-bis — Qué significa "evaluar" aquí](#parte-b-bis--qué-significa-evaluar-aquí)
- [Parte B-ter — Validación por silo retenido (leave-one-site-out)](#parte-b-ter--validación-por-silo-retenido-leave-one-site-out)
- [Parte C — Lo que ya está cogido](#parte-c--lo-que-ya-está-cogido)
- [Parte D — Seis ideas con potencial de impacto](#parte-d--seis-ideas-con-potencial-de-impacto)
- [Parte E — La recomendación](#parte-e--la-recomendación)
- [Parte F — Qué hacer antes de que acabe agosto](#parte-f--qué-hacer-antes-de-que-acabe-agosto)
- [Parte G — Evidencia y citas clave](#parte-g--evidencia-y-citas-clave)
- [Parte H — Mapa de lo ya construido](#parte-h--mapa-de-lo-ya-construido)
- [Parte I — Decisiones pendientes](#parte-i--decisiones-pendientes)

---
---

# Parte A — Diagnóstico del campo

## El tema es bueno; el campo es peligroso

Son dos cosas distintas y conviene no confundirlas.

### Lo peligroso

El aprendizaje federado lleva desde 2017 siendo "lo que va a desplegarse el año
que viene". Ocho años después, los despliegues reales que se pueden nombrar
caben en una mano: el teclado de Google y un puñado de consorcios médicos.

Mientras tanto se han publicado **miles** de papers llamados `FedAlgo` que son
FedAvg con un término extra, evaluados sobre CIFAR-10 particionado con Dirichlet.

> Si la tesis acaba pareciéndose a eso, será una tesis **correcta e irrelevante**.

### Lo bueno

La presión que hace falta para que el federado importe **por fin existe, y no es
tecnológica: es regulatoria**. Espacios europeos de datos, sanidad, industria,
AI Act.

Y ahí es donde estás tú, que es un sitio donde casi nadie de la comunidad
algorítmica está. El problema no es tener tema; **es no diluirlo**.

### La objeción que hay que responder de frente

En la era de los modelos fundacionales, el argumento por defecto es
**"centraliza y ajusta con LoRA"**. La tesis tiene que responder a eso
explícitamente, no ignorarlo. Y la respuesta es **legal y económica, no
técnica**: qué normativa concreta, qué coste concreto, qué caso concreto.

El capítulo de motivación es donde se gana o se pierde la tesis.

---

## La ventaja competitiva real

No es el algoritmo. Es que **tú tienes una economía y ellos no**.

| Lo que asume la literatura de federado | Lo que hay en tu escenario |
|---|---|
| Comunicación gratuita | Cada ronda cuesta K jobs C2D |
| Sin gobernanza | Contratos, políticas ODRL, permisos |
| Los participantes participan | Tienen que **querer**, y cobrar |
| Sin coste por ronda | El presupuesto es finito y contable |
| Evaluación disponible | En RL offline no existe |

**Cada una de esas filas es una restricción. Y una restricción es un problema
nuevo que nadie ha estudiado.** El wow no vendrá de un `FedAlgo` mejor: vendrá de
convertir una restricción en un recurso.

---
---

# Parte B — Diez ideas para enfocar la tesis

Estas son de **método y estrategia**, no de tema. Aplican a cualquiera de las
direcciones de la Parte D.

### 1. Anclar la tesis en la restricción, no en la técnica

El activo diferencial no es "hago federado". Es que trabajas bajo una restricción
concreta que casi nadie estudia: contenedor sin red, ejecución por jobs, coste
por ronda no despreciable. **Esa restricción genera problemas nuevos.** "Federado
en general" no.

### 2. Profundidad en un eje, superficialidad deliberada en los otros dos

Offline + federado + data spaces son tres campos. La trampa de la tesis triple es
acabar a nivel de máster en los tres. Elige uno donde vayas a contribuir de
verdad y trata los otros como infraestructura citada, **sin pedir perdón**.

### 3. La evaluación como contribución, no como capítulo

En RL offline federado no existe conjunto de validación. Eso lo convierte en *el*
problema, no en el trámite. Un protocolo de evaluación defendible es más
publicable y más citable que un `FedAlgo` más.

> **Ojo con la palabra.** "Evaluación" aquí **no** significa la fase de
> entrenamiento ni medir una pérdida. Significa responder a *"¿cuál de estas 40
> políticas despliego, y cuánto de buena es?"* sin poder ejecutar ninguna.
> Desarrollado entero en → **Parte B-bis**, que conviene leer antes que el resto.

### 4. Una pregunta como columna vertebral

Candidata: ***¿cuándo basta una ronda?*** Es concreta, medible, motivada por tu
infraestructura, y tiene respuesta empírica. Y da lugar a un criterio a priori
que otros pueden usar.

> Las tesis con **una idea** se defienden solas. Las que tienen "una serie de
> contribuciones" se defienden a duras penas.

### 5. El artefacto es un resultado

Infraestructura reproducible, con tests, datasets y protocolo. En un campo con
crisis de reproducibilidad, eso se cita. Ya lo tienes medio hecho (→ Parte H):
no lo escondas en un apéndice.

### 6. Guardar los negativos

"Probé X y no funcionó, y aquí está por qué" es la mitad de una buena discusión
de tesis, y la parte que nadie publica. En federado, donde todo el mundo reporta
mejoras, un resultado negativo bien medido destaca.

**Escríbelos el día que ocurren.** No se reconstruyen después.

### 7. Conseguir un consorcio real, aunque sea diminuto

Tres organizaciones con datos de verdad valen más que cien clientes simulados.
Cambia las preguntas que te haces, y es lo que separa una tesis de sistemas de un
ejercicio de simulación.

Si no es posible, **dilo en las limitaciones** en lugar de disimularlo.

### 8. No competir en variantes de agregación

Ese mercado está saturado y lo dominan grupos con más GPUs. Compite donde la
restricción del entorno **cambia** el problema: adaptadores heterogéneos,
agregación robusta compatible con cegado, evaluación sin validación.

### 9. Responder por escrito al contraargumento de la centralización

Ver Parte A. Un tribunal lo va a preguntar.

### 10. Publicar sistemas primero, algoritmo después

Un paper de sistemas —*"así se hace federado sobre C2D, esto es lo que se
rompe"*— es más fácil de publicar, te da presencia y te construye la
infraestructura para el paper algorítmico. Al revés, el algoritmo sin
infraestructura se queda en simulación.

---

### 11. (La que más importa) Ponerle fecha a la exploración

Entender bien el terreno se paga, y la base de conocimiento construida es real.
Pero **la documentación es infinitamente ampliable y los experimentos no**.
Ninguna sección más te acerca a una contribución.

> Fija el experimento de la curva rondas/calidad con redes profundas, córrelo, y
> deja que el resultado decida los dos años siguientes.

Esperar a entenderlo todo antes de medir nada es la forma más común de perder un
año sin darse cuenta.

---
---

# Parte B-bis — Qué significa "evaluar" aquí

> **Aviso de vocabulario.** Este apartado existe porque "evaluación" se confunde
> constantemente con la fase de entrenamiento o con calcular una métrica. No es
> ninguna de las dos cosas, y la distinción es la que sostiene la idea ① y buena
> parte de la tesis.

## Los tres verbos

En aprendizaje supervisado están tan pegados que parecen uno solo. En RL offline
se separan violentamente.

```none
① ENTRENAR    produce parámetros
              "dame una política"
              → en RL offline: RESUELTO. Hay decenas de algoritmos

② VALIDAR     elige entre candidatos
              "de estas 40 políticas que he entrenado, ¿cuál me llevo?"
              → en RL offline: PROBLEMA ABIERTO

③ EVALUAR     estima el rendimiento real
              "¿cuánto de buena es la que me llevo?"
              → en RL offline: PROBLEMA ABIERTO
```

En supervisado los tres son el mismo gesto repetido: entrenas, calculas accuracy
en validación para elegir, y en test para reportar. Por eso nadie los distingue.

**Cuando digo que "la evaluación es el problema", me refiero a ② y ③.** El ①
está resuelto y no es donde está la contribución.

---

## Por qué ② y ③ son duros

La pregunta que hay que responder en RL no es *"¿qué predice el modelo?"* sino:

> **"¿Cuánto retorno acumularía esta política si la ejecutara?"**

Y responder a eso, literalmente, exige ejecutarla. Que es exactamente lo que la
premisa del RL offline prohíbe.

### El caso concreto

Entrenas 40 políticas de dosificación en UCI, variando el coeficiente de
pesimismo de CQL, la tasa de aprendizaje y la arquitectura. Tienes 40 conjuntos
de pesos.

**¿Cuál despliegas?**

| Vía obvia | Por qué no sirve |
|---|---|
| Probarlas en pacientes | Es el motivo entero de hacerlo offline |
| Un simulador | Si lo tuvieras, harías RL online |
| Datos apartados como validación | Medir "cuánto acierta" **no es la pregunta**: la pregunta es qué habría pasado si el médico hubiera hecho lo que dice tu política y no lo que hizo. Contrafactual otra vez |

---

## La trampa que lo hace peor de lo que parece

El instinto es usar la pérdida: error de Bellman sobre unos datos apartados.

**No funciona, y este es el punto que más sorprende.**

> En RL offline **la pérdida no está bien correlacionada con la calidad de la
> política**. Una función $Q$ puede tener un error de Bellman bajísimo y producir
> una política pésima, porque el error se mide sobre las acciones **que están en
> el dataset** y la política actúa sobre las que **no**.

Es lo contrario de supervisado, donde una pérdida baja en validación sí te dice
algo. **Aquí el instrumento que crees que tienes está roto.**

---

## La consecuencia en cadena

```none
no hay evaluación fiable
        ↓
no puedes elegir hiperparámetros con criterio
        ↓
no puedes comparar tu método con el baseline honestamente
        ↓
no puedes afirmar que has mejorado nada
```

Ese último eslabón es el que convierte esto en un problema de tesis y no en un
detalle de ingeniería.

> **Si no puedes evaluar, no puedes hacer ciencia.** Solo puedes reportar que un
> número salió más alto, sin saber si es señal o ruido de semilla.

Y hay un pez que se muerde la cola: los métodos de evaluación off-policy tienen a
su vez hiperparámetros que hay que elegir, y para elegirlos necesitarías…
evaluarlos.

---

## Y al federar, peor

| | Centralizado | Tu caso |
|---|---|---|
| Evaluar globalmente | difícil | **imposible**: no hay un sitio con todos los datos |
| Evaluar localmente | — | sesgado: cada silo tiene su propia distribución |
| Saber qué cubre cada silo | — | **no puedes verlo** |
| Elegir hiperparámetros | difícil | sin criterio global |

---

## Qué significa entonces "la evaluación como contribución"

Que en la tesis, **diseñar y defender el procedimiento para responder a ② y ③
es el trabajo**, no el preámbulo del trabajo.

En concreto, un protocolo que responda:

- [ ] Cómo se **ordenan** políticas candidatas antes de desplegar ninguna.
- [ ] Cuánta **confianza** merece ese orden (intervalos, no puntos → 3.9).
- [ ] Cuántos datos —o cuántos **jobs C2D**— hacen falta para que el orden sea
      fiable.
- [ ] Qué se hace cuando dos políticas son indistinguibles con el presupuesto
      disponible.

---

## De aquí sale la idea ①

El razonamiento completo, en cuatro pasos:

```none
1. No tengo conjunto de validación.
2. Pero tengo vecinos con datos que NUNCA vieron mi política.
3. Los datos del silo B son, por tanto, un conjunto de retención genuino
   y fuera de distribución para la política entrenada en A.
4. El data space me da el mecanismo para usarlos sin verlos:
   un job C2D que devuelve UN ESCALAR, no un modelo.
```

**Comprar evaluación en el mercado.** Un job que devuelve un número tiene una
superficie de privacidad ridícula frente a uno que devuelve un modelo — lo que
además lo hace mucho más fácil de aprobar por la gobernanza.

> Sin resolver ② y ③, no sabes si lo que has construido funciona. Con ellos
> resueltos, tienes una contribución que sirve a todo el campo, no solo a tu
> pipeline.

---
---

# Parte B-ter — Validación por silo retenido (leave-one-site-out)

> **El desarrollo operativo de la idea ①.** Documentado entero en la sección
> **→ 14** del libro. Aquí queda el resumen que hace falta para decidir.

## El término

**Leave-one-site-out** (LOSO), o **validación externa**, viene del aprendizaje
automático médico multicéntrico: entrenas con todos los centros menos uno,
validas en el que dejas fuera, y rotas. Es **práctica estándar** en imagen médica
y federado clínico, porque un modelo puede aprender el **hospital** en vez de la
**enfermedad** y la partición aleatoria no lo detecta.

Es la misma idea que el *split por grupo* de → 3.0 y → 3.9, aplicada a
organizaciones.

## Qué cambia al traducirlo a RL offline

| | Supervisado | RL offline |
|---|---|---|
| Qué hay en el silo retenido | pares $(x,y)$ con la **etiqueta verdadera** | trayectorias de **otra** política |
| Cómo se mide | comparar $\hat y$ con $y$ | estimar el retorno **sin ejecutar** |
| Dificultad | trivial | **es la OPE** |

> **Conclusión ①.** El silo retenido **no te da una métrica. Te da datos sobre
> los que ejecutar un estimador de OPE.** Aporta independencia, no facilidad.

```none
silos 1,2,3 ──▶ entrenar ──▶ política π
                                  │
silo 4 (logs) ──▶ estimador OPE ──┴──▶ V̂(π)
```

---

## La cobertura: la condición de validez

La OPE sobre el silo 4 solo funciona **si el silo 4 cubre las acciones que tu
política quiere tomar**. Si $\pi$ propone dosis que nadie del hospital 4
administró, el estimador extrapola — el error de extrapolación de → 1.6, otra vez.

> **Conclusión ②.** Con un silo de retención **no has resuelto el problema del
> soporte: lo has movido.** Antes tenías extrapolación al aprender; ahora la
> tienes al evaluar.

### La asimetría cruel

> **Cuanto mejor es tu política —cuanto más se aleja de $\pi_\beta$ para
> mejorarla— menos evaluable es.** Lo que quieres medir es precisamente lo que no
> puedes medir.

**Regla operativa:** medir la cobertura **antes** de fiarte del número. Si es
baja, el resultado no significa nada y publicarlo sería peor que no tenerlo.
Reportar $\hat V(\pi)$ sin cobertura es como reportar accuracy sin decir el
tamaño del test.

---

## Qué obtienes: tres ambiciones distintas

| | Qué es | Dificultad |
|---|---|---|
| **Ordenar** | *"¿es π₇ mejor que π₂₃?"* | Difícil pero **abordable** |
| **Valorar** | *"¿π₇ obtiene 0,82?"* | Mucho más difícil |
| **Transferir** | *"¿aguanta fuera de su hospital?"* | **Quizá lo más valioso aquí** |

Para elegir hiperparámetros **solo necesitas ordenar**, y hay esperanza empírica:
Paine et al. encontraron que controlando bien los factores se pueden ordenar
políticas de forma fiable. Tres hallazgos suyos que estructuran el diseño:

1. Los algoritmos que **mantienen la política cerca de $\pi_\beta$ son más
   fáciles de evaluar**. → sesgo hacia la timidez, hay que declararlo.
2. **FQE** estima mejor que las estimaciones del propio algoritmo de
   entrenamiento.
3. El **valor medio del crítico en los estados iniciales** es el mejor
   estadístico de resumen.

Y el tercer producto, que la OPE centralizada no puede dar porque no tiene
sitios: **transferibilidad**. En un consorcio es *la* pregunta, y es
intrínsecamente federada.

---

## Rotar los cuatro pliegues

```none
entrena {1,2,3} → evalúa en 4  →  V̂ = 0.81
entrena {1,2,4} → evalúa en 3  →  V̂ = 0.79
entrena {1,3,4} → evalúa en 2  →  V̂ = 0.83
entrena {2,3,4} → evalúa en 1  →  V̂ = 0.42   ← ¿por qué este?
```

**La varianza entre pliegues es información, no ruido.** Un pliegue desviado
significa o bien que ese silo es muy distinto, o bien que su cobertura es pobre.
Distinguir ambas es obligatorio.

### El confounder que hay que declarar

> **Conclusión ③.** Al rotar no evalúas la misma política cuatro veces: evalúas
> cuatro políticas distintas. Seleccionas **un procedimiento** (una configuración
> de hiperparámetros), no una política.

```none
1. Leave-one-silo-out  →  elijo la CONFIGURACIÓN λ*
2. Reentreno con los 4 →  obtengo la política final π*
3. ¿Cuánto vale π*?    →  NO LO SÉ. No queda silo libre.
```

Idéntico al problema de la validación cruzada en supervisado, salvo que allí el
paso 3 lo resuelve un conjunto de test. **Con 5+ silos, reservar uno intocado
como test real** (→ 3.9).

### El coste en jobs

Con $K$ silos y $H$ configuraciones: $K \times H$ jobs de evaluación, más el
entrenamiento. **Si hay que recortar, recorta en configuraciones, no en
pliegues** — los jobs de evaluación son los baratos.

---

## Por qué en C2D esto es especialmente bueno

| | Job de entrenamiento | Job de evaluación |
|---|---|---|
| Devuelve | un modelo | **un número** |
| Superficie de privacidad | alta | mínima |
| Aprobación por gobernanza | difícil | **fácil** |

> **Un mercado de evaluación es mucho más fácil de vender a un consorcio que un
> mercado de entrenamiento.** Y el *derecho a evaluar* es distinto del *derecho a
> entrenar*: expresable en ODRL, y con más participantes dispuestos.

---

## El experimento (barato, ejecutable en septiembre)

**Pregunta:** ¿coincide el orden estimado sobre un silo retenido con el orden
verdadero, y hasta qué heterogeneidad aguanta?

```none
1. MDP tabular con dinámica CONOCIDA → permite calcular V_real exacto
2. 4 silos con políticas de comportamiento distintas
3. Entrenar con 3, barriendo el coeficiente de pesimismo → H candidatas
4. Estimar V̂ con FQE sobre el silo 4
5. Comparar el RANKING estimado con el RANKING verdadero (Spearman)
6. Rotar los 4 pliegues
7. REPETIR variando la similitud entre silos  ← el paso que da el resultado
```

**Publicable en cualquier dirección:**

| Resultado | Qué tienes |
|---|---|
| La correlación aguanta | **Un método** |
| Se rompe pronto | **Un resultado negativo con umbral cuantificado**, que nadie ha medido |

Tres gráficas: la curva Spearman frente a divergencia (**el umbral es el
resultado**), cobertura como variable explicativa, y el sesgo hacia políticas
conservadoras.

**Línea base que protege:** ¿qué correlación da elegir al azar? ¿Y elegir siempre
el $\beta$ más conservador? Si esta última ordena casi igual de bien, el
protocolo no aporta nada.

---

## Honestidad sobre la novedad

| Componente | Estado |
|---|---|
| LOSO en federado médico **supervisado** | **Estándar.** No es novedad |
| RL offline federado | **Existe** (ver Parte C) |
| Selección de política offline / OPE | **Área activa** |
| **La combinación** con el silo evaluador como servicio comprable | **No encontrado** |

> ⚠️ "No encontrado" no es "no existe". **Verificación bibliográfica: semana 1.**
> Buscar *federated off-policy evaluation*, *cross-site policy selection*,
> *external validation offline RL*, *multi-site OPE*.

**Precedente más cercano:** FDTR (JASA 2024) hace RL offline federado
multicéntrico con **una sola ronda intercambiando estadísticos resumen**. No usa
un sitio retenido para seleccionar hiperparámetros. Ahí sigue el hueco.

**Objeción que hay que responder:** *Offline RL Without Off-Policy Evaluation*
(Brandfonbrener et al., 2021) muestra que un paso de mejora sobre una $Q$
on-policy funciona sorprendentemente bien y es **más robusto a hiperparámetros**.
Si eso basta, tu protocolo pierde parte de su motivación.

---
---

# Parte C — Lo que ya está cogido

**Crítico: leer antes de elegir dirección.** El ángulo algorítmico obvio del RL
offline federado ya está ocupado, y por gente fuerte.

| Trabajo | Qué cubre |
|---|---|
| **Woo, Shi, Joshi & Chi**, *Federated Offline RL: Collaborative Single-Policy Coverage Suffices* (ICML 2024) — [arXiv:2402.05876](https://arxiv.org/abs/2402.05876) | Pesimismo en federado offline aprovechando la cobertura colaborativa, sin compartir datasets ni estimaciones de modelo |
| **FEDORA**, *Federated Ensemble-Directed Offline RL* (NeurIPS 2024) — [arXiv:2305.03097](https://arxiv.org/abs/2305.03097) | Heterogeneidad del ensemble, pesimismo entre clientes, deriva |
| **FDTR**, *Federated Offline Reinforcement Learning* (JASA 2024) | Régimen de tratamiento dinámico federado, con PEVI por sitio |
| **Woo et al.**, *The blessing of heterogeneity in federated Q-learning* (ICML 2023) | La heterogeneidad como ventaja, no como problema |

FEDORA lo formula ya explícitamente: *es vital capturar la sabiduría colectiva de
ese conjunto de políticas, no promediarlas*, y *entrenar el crítico local con el
término pesimista puede volverlo pesimista hacia acciones poco representadas en
su dataset pero bien representadas en los de otros clientes*.

### Pero fíjate en lo que TODOS asumen

```none
✗ comunicación gratuita
✗ sin gobernanza
✗ sin coste por ronda
✗ participantes que participan porque sí
✗ evaluación disponible de algún modo
```

**Ninguno de esos supuestos se sostiene en un data space real.** Ahí está el
hueco, y no es algorítmico.

> **Conclusión operativa:** no propongas "pesimismo federado" ni "agregación
> ponderada por cobertura" como contribución. Está hecho. Propón lo que ocurre
> cuando añades economía y gobernanza.

---
---

# Parte D — Seis ideas con potencial de impacto

Ordenadas por (impacto × viabilidad). ⭐ = potencial de impacto.

---

## ① Comprar evaluación en el mercado ⭐⭐⭐

> **Prerrequisito:** → **Parte B-bis**. Si no tienes clara la diferencia entre
> entrenar, validar y evaluar, esta idea no se entiende.

### La inversión

El RL offline no tiene conjunto de validación (→ Parte B-bis). Es *el* problema
abierto:
**uno de los mayores problemas abiertos del RL offline es el ajuste de
hiperparámetros**, y hay un pez que se muerde la cola — *para ajustar los
hiperparámetros de entrenamiento hay que ajustar los hiperparámetros de la OPE*.

Ahora mira tu escenario:

> **Los datos del silo B son un conjunto de validación fuera de distribución para
> la política del silo A.** Y el data space te da el mecanismo para acceder a
> ellos sin verlos: un job C2D que devuelve un escalar.

### La propuesta

**Evaluación federada off-policy como servicio comprable.** No entrenas con todos
los silos: entrenas con unos y **compras evaluación** a otros.

```none
   silos de entrenamiento          silos de evaluación
   ─────────────────────           ───────────────────
   A, B, C  ──▶ política π   ──▶   D, E  ──▶  V̂(π), un escalar
                                              │
                    selección de hiperparámetros ◀┘
```

### Por qué es fuerte

- Ataca el problema abierto **nº 1 del RL offline** con la capacidad **nº 1 de
  los data spaces**. Nadie ha conectado las dos cosas.
- Un job que devuelve **un número** tiene una superficie de privacidad ridícula
  comparada con uno que devuelve un modelo. **La gobernanza lo aprueba fácil.**
- Es medible: error de la estimación, coste en jobs, calidad de la selección.

### Riesgo: medio

La OPE sigue teniendo varianza alta. Pero incluso el resultado negativo —*"la
validación cruzada federada no basta para seleccionar hiperparámetros en offline
RL, y aquí está por qué"*— es publicable y útil.

### Referencias de partida

- [Hyperparameter Selection for Offline RL](https://arxiv.org/abs/2007.09055) (Paine et al., 2020)
- [Towards Hyperparameter-free Policy Selection](https://arxiv.org/abs/2110.14000) (Zhang & Jiang, NeurIPS 2021)
- [When is Offline Policy Selection Sample Efficient?](https://arxiv.org/abs/2312.02355) (2026)
- [A Survey on Offline RL](https://arxiv.org/abs/2203.01387) (Prudencio et al.), §V sobre OPE

---

## ② Incentivos exigibles: pagar por contribución verificada ⭐⭐⭐

### El hueco, señalado por la propia literatura

Un survey de junio de 2026 lo dice sin rodeos: **la ausencia de mecanismos de
incentivos exigibles es probablemente la barrera estructural más infraestudiada
para una federación real**. Los clientes racionales, sean instituciones o
dispositivos, **no contribuirán voluntariamente datos y cómputo de calidad si la
recompensa es incierta, no verificable o insuficiente** para compensar el coste
de participar.

### Por qué es tuyo

La literatura de valor de Shapley federado lleva años siendo **teórica porque no
hay dónde cobrar**. Ocean tiene raíles de pago reales.

### La tensión que lo hace investigación y no ingeniería

**Atribuir contribución exige evaluar modelos individuales. Tu cegado (→ 12.2) lo
impide.** Privacidad y atribución se pelean.

Y la versión con DP también está identificada como problema abierto: *la DP con
ε<1 vuelve poco fiables los métodos de atribución basados en gradientes, con más
del 30 % de error, creando una tensión dura entre privacidad y explicabilidad*.

Resolver **atribución bajo cegado** es una contribución con nombre propio.

### El wow de demo

Un consorcio donde **el hospital pequeño cobra por participar y todos pueden
verificar el reparto**. Eso se entiende en pantalla, no solo en un paper.

### Riesgo: medio-bajo

Hay literatura de la que partir y el mecanismo existe.

### Referencias de partida

- [Unfederated: Open Challenges, Deployment Gaps, and Emerging Directions in FL](https://link.springer.com/article/10.1007/s11831-026-10696-3) (2026) — **la cita del hueco**
- [A Comprehensive Survey of Incentive Mechanism for FL](https://arxiv.org/abs/2106.15406)
- Zheng, Cao & Yoshikawa, *Secure Shapley Value for Cross-Silo FL* (VLDB 2023)
- [Owen Sampling Accelerates Contribution Estimation in FL](https://arxiv.org/abs/2508.21261)
- Wang et al., *A principled approach to data valuation for federated learning*

---

## ③ La política de uso, compilada dentro del entrenamiento ⭐⭐⭐

### La idea

Hoy ODRL expresa *"este activo solo puede usarse para entrenar un modelo"*. Es
binario. Conviértelo en **restricción del problema de optimización**:

| Cláusula del contrato | Consecuencia algorítmica |
|---|---|
| `solo con ε ≤ 1` | Presupuesto de DP verificado por job |
| `revocable en 30 días` | **Desaprendizaje federado exigible por contrato** |
| `no para modelos desplegados fuera de UE` | Restricción de despliegue trazable |
| `máximo 3 rondas` | Presupuesto de cómputo que **fuerza one-shot** |

**Aprendizaje federado gobernado por contrato ejecutable.** Cada cláusula te
obliga a un algoritmo distinto.

### El sub-ángulo más maduro: desaprendizaje

Las regulaciones emergentes exigen que los modelos puedan **olvidar** los datos
aprendidos, con problemas abiertos en escalabilidad, equidad y desaprendizaje en
modelos fundacionales. En federado esto tiene nombre propio —*federated
unlearning*— y **nadie lo ha atado a contratos de data space**, donde un
participante puede revocar.

Y en RL offline, desaprender las trayectorias de un silo de una política ya
aprendida es **difícil y no estudiado**.

### Por qué es fuerte

Es exactamente la agenda europea (AI Act, espacios de datos), es interdisciplinar
de verdad, y **ningún grupo de ML puro puede hacerlo porque no tiene el data
space**.

### Riesgo: medio

Es amplio. Hay que acotar a una o dos cláusulas, no a las cuatro.

### Referencias de partida

- [Survey on Federated Unlearning: Challenges and Opportunities](https://www.computer.org/csdl/journal/bd/5555/01/11415662/2es6IzzXbEI) (2026)
- Especificación ODRL y Dataspace Protocol (→ 13.7)
- [A Service Architecture for Dataspaces](https://arxiv.org/abs/2507.07979) (2025)

---

## ④ El modelo tampoco se mueve ⭐⭐

### La inversión completa

Hoy los datos no viajan pero **el modelo sí** — y el modelo es propiedad
intelectual. Apheris ya lo tiene identificado: protegen **la propiedad
intelectual del modelo**, no solo los datos.

> **Compute-to-Data mutuo.** El proveedor de datos publica un dataset que no se
> descarga. El dueño del modelo publica un modelo que tampoco se descarga. Se
> encuentran en un tercer lugar atestiguado donde ninguno ve lo del otro.

### Qué desbloquea

Consorcios donde **nadie confía en nadie** y aun así se entrena. Con LoRA es
tratable: el adaptador es el artefacto pequeño que hay que proteger.

### Riesgo: medio-alto

Requiere atestación o entorno de ejecución confiable (TEE).

---

## ⑤ Demostrar que entrenaste lo que dijiste ⭐⭐

C2D asume que **tu algoritmo** es hostil. Dale la vuelta: **el proveedor también
puede serlo.** Nada impide que un silo devuelva una actualización inventada, o
entrenada sobre otros datos.

En federado normal esto es irresoluble sin criptografía cara. **Pero C2D te da
algo que la literatura de federado no tiene: un sustrato de ejecución que puede
atestiguar.** El nodo puede firmar *"ejecuté esta imagen sobre este DID"*.

**Aprendizaje federado atestiguado:** una versión débil pero práctica de las
pruebas de entrenamiento, apoyada en infraestructura que ya existe.

### Riesgo: medio-alto

Depende de qué garantice `ocean-node` de verdad. **Verificar antes de
comprometerse.**

---

## ⑥ El paper que nadie escribe: cuándo NO federar ⭐⭐

Un **procedimiento de decisión** con modelo de coste real: jobs, euros, latencia,
riesgo regulatorio. Con la respuesta honesta de que en muchos casos es
*"centraliza y ajusta con LoRA"*.

No suena espectacular. Pero es lo más citado a cinco años vista, porque todo el
que quiera desplegar necesita ese cálculo y nadie lo ha hecho.

**Y es antifrágil: no depende de que el federado triunfe.**

---
---

# Parte E — La recomendación

## Combinar ① + ②, con ⑥ como capítulo de cierre

### La historia, en un párrafo

> *En un espacio de datos, la evaluación y el pago son el mismo problema.
> Compruebas cuánto vale una política comprando su evaluación a otros silos; y
> ese mismo mecanismo te dice cuánto ha contribuido cada uno y cuánto cobra. Y
> como el cegado impide evaluar modelos individuales, privacidad y atribución hay
> que resolverlas juntas.*

### Por qué esta combinación

- Es una tesis con **una idea**, no con una lista de contribuciones.
- Ataca el problema abierto del **RL offline** (evaluación) y el problema abierto
  del **federado real** (incentivos) **con la misma herramienta**.
- Termina en una demo que un tribunal entiende en dos minutos: tres silos, una
  política, un reparto de pagos verificable.
- **Cada pieza es medible**: error de la evaluación federada, coste en jobs,
  desviación del reparto respecto al Shapley exacto. No es un manifiesto, es un
  sistema con números.
- **Tiene plan B:** si ① falla porque la OPE federada es demasiado ruidosa, ese
  resultado negativo sigue siendo un paper, y ② sobrevive solo.

### Estructura tentativa de la tesis

| Capítulo | Contenido | De dónde sale |
|---|---|---|
| 1 | Motivación: por qué no centralizar (legal + económico) | Parte A |
| 2 | Fundamentos: federado, RL offline, data spaces | → 1.5, → 1.6 |
| 3 | Sistema: federado sobre C2D, y qué se rompe | → 11, → 12 |
| 4 | **Evaluación federada off-policy** | Idea ① |
| 5 | **Atribución y pago bajo cegado** | Idea ② |
| 6 | Cuándo federar: procedimiento de decisión con coste | Idea ⑥ |
| 7 | Limitaciones, negativos y trabajo futuro | Idea 6 de la Parte B |

---
---

# Parte F — Qué hacer antes de que acabe agosto

Quedan ~3 semanas. **El objetivo no es entender más: es cerrar la exploración con
una decisión defendible y un experimento corrido.**

## Semana 1 — Verificar supuestos

- [ ] **Correr el experimento de la curva rondas/calidad con red profunda.**
      Sustituir el learner NumPy por PyTorch en → 11 y medir. Es barato, ya
      tienes todo, y su respuesta condiciona el resto.
- [ ] Comprobar qué garantiza `ocean-node` sobre atestación (condiciona ⑤).
- [ ] Verificar el estado de FELT (→ 12.7) y anotar la fecha.
- [ ] Buscar si existe ya *federated off-policy evaluation*, *cross-site policy
      selection*, *external validation offline RL* o *multi-site OPE*. **Si
      existe, la idea ① cambia de forma.** (→ Parte B-ter, y sección 14)
- [ ] Leer FDTR (JASA 2024) entero: es el precedente más cercano.
- [ ] Leer Brandfonbrener et al. 2021: es la objeción a responder.

## Semana 2 — Acotar

- [ ] Leer los cuatro papers de partida de ① y los cinco de ②.
- [ ] Escribir **media página** por cada una de las ideas ①, ② y ③, con: qué
      se contribuye, qué se mide, qué se necesita, qué puede salir mal.
- [ ] Estimar el coste en jobs C2D del experimento completo de cada una.
- [ ] Hablar con el director con esas tres medias páginas delante.

## Semana 3 — Decidir y arrancar

- [ ] Elegir. Escribir la **pregunta única** de la tesis en una frase.
- [ ] Escribir el párrafo de motivación (respuesta a "¿por qué no centralizar?").
- [ ] Definir el protocolo experimental: semillas, métricas, presupuesto de
      ajuste (→ 3.9, parte B).
- [ ] **Congelar la documentación.** A partir de aquí solo crece cuando un
      experimento lo justifica.

## La regla de agosto

> Si el 31 de agosto no tienes una pregunta escrita en una frase y un experimento
> corrido, la exploración ha fracasado — **por muy buena que sea la base de
> conocimiento**.

---
---

# Parte G — Evidencia y citas clave

Las afirmaciones sobre las que se apoya todo lo anterior, con su fuente, para no
tener que reconstruirlas.

## Sobre el hueco de los incentivos

> *La ausencia de mecanismos de incentivos exigibles es probablemente la barrera
> estructural más infraestudiada para una federación real. Los clientes
> racionales, sean instituciones o dispositivos, no contribuirán voluntariamente
> datos y cómputo de calidad si la recompensa es incierta, no verificable o
> insuficiente para compensar el coste de participar.*

— *Unfederated: Open Challenges, Deployment Gaps, and Emerging Directions in
Federated Learning*, Archives of Computational Methods in Engineering, junio 2026.
[Enlace](https://link.springer.com/article/10.1007/s11831-026-10696-3)

El mismo trabajo identifica la tensión **privacidad ↔ atribución**: con
(ε,δ)-DP y ε<1, los métodos de atribución basados en gradientes se vuelven poco
fiables, con más del 30 % de error.

## Sobre el hueco de la evaluación

> *Uno de los mayores problemas abiertos del RL offline es el ajuste de
> hiperparámetros.*

— Prudencio et al., *A Survey on Offline RL*, IEEE TNNLS.
[arXiv:2203.01387](https://arxiv.org/abs/2203.01387), §V

> *Para ajustar los hiperparámetros de entrenamiento hay que ajustar los
> hiperparámetros de la OPE, creando una situación de pez que se muerde la cola.*

— Zhang & Jiang, *Towards Hyperparameter-free Policy Selection*.
[arXiv:2110.14000](https://arxiv.org/abs/2110.14000)

Y sobre por qué la pérdida no sirve como sustituto (→ Parte B-bis): un pipeline
de selección de modelo en RL offline **requiere ejecutar la política en el
entorno real**, lo que en un contexto sanitario suele ser inviable — de ahí que
se recurra a la OPE como sustituto de la validación, con todos sus problemas.

— Referencias: *Model Selection for Offline RL: Practical Considerations for
Healthcare Settings*; Paine et al., [Hyperparameter Selection for Offline RL](https://arxiv.org/abs/2007.09055);
Levine et al., [Offline RL Tutorial](https://arxiv.org/abs/2005.01643), §6.

## Sobre lo que ya está resuelto (no repetir)

> *Es vital capturar la sabiduría colectiva de ese conjunto de políticas, no
> promediarlas.*
>
> *Entrenar el crítico local con el término pesimista puede volverlo pesimista
> hacia acciones poco representadas en su dataset pero bien representadas en los
> de otros clientes.*

— FEDORA, NeurIPS 2024. [arXiv:2305.03097](https://arxiv.org/abs/2305.03097)

## Sobre la restricción de C2D

Verificado leyendo el fuente de `ocean-node` (→ 11.6): **el contenedor arranca sin
pila de red** salvo que el operador active `enableNetwork`, con capacidades
retiradas, sin escalada de privilegios y con usuario no-root. **El diseño asume
que tu algoritmo es hostil.**

De ahí se deriva todo el patrón asíncrono por ficheros, y de ahí sale la idea ⑤.

## Sobre el error de promediado de LoRA federado

> *FedIT promedia A y B de forma independiente, lo que introduce errores
> matemáticos en el LoRA global.*

— FLoRA, NeurIPS 2024. El producto de las medias no es la media de los productos.

---
---

# Parte H — Mapa de lo ya construido

Qué hay hecho, y qué idea soporta cada pieza.

| Sección | Contenido | Soporta |
|---|---|---|
| **1.5** | Las 3 decisiones al diseñar un sistema de IA | Marco general |
| **1.6** | RL Offline: por qué es difícil + analogía biológica | ①, y la pregunta de fondo |
| **2** | Matemáticas necesarias (índice ampliado) | Base |
| **3.0–3.9** | Cómo se entrena una red: datos, ciclo, backprop, pérdidas, regularización, precisión, escalado, transferencia, problemas, medición | ⑥ y todo el método |
| **11** | Federado sobre data spaces: 4 labs sobre C2D, código ejecutable | ①, ②, ⑤ |
| **12** | FELT Labs: protocolo de doble ciego, cegado con ruido | **②** (la tensión atribución/cegado) |
| **13** | Herramientas y variantes: frameworks, agregadores, LoRA federado, one-shot, privacidad, data spaces | Todo |
| **14** | **Validación por silo retenido**: LOSO, cobertura, rotación, protocolo y experimento | **①**, y es el desarrollo completo de la recomendación |

## Resultados propios ya medidos

| Hallazgo | Dónde | Por qué importa |
|---|---|---|
| Con reparto Dirichlet, la accuracy no se mueve pero la **divergencia se multiplica por 9** | → 11 | La accuracy oculta el problema. Vale para toda la tesis |
| El centro pequeño pasa de **0,647 a 0,922** sin mover datos | → 11 | La tabla que justifica el proyecto |
| Con modelo convexo, **30 rondas no mejoran sobre 1** | → 12.3 | Media respuesta a "¿cuándo basta una ronda?" |
| El cegado cancela **exactamente** (error 2×10⁻¹³) | → 12.2 | No hay coste en calidad por la privacidad |
| El bug de la ponderación: invisible con silos iguales | → 12.2 | Los tests simétricos ocultan errores de peso |

**Falta la otra mitad:** la curva rondas/calidad con **red profunda**. Es el
experimento de la semana 1.

---
---

# Parte I — Decisiones pendientes

Lo que hay que cerrar antes del 31 de agosto:

- [ ] **¿Idea ①, ② o ③?** (o la combinación recomendada ① + ②)
- [ ] ¿Hay consorcio real posible, o la tesis es de simulación? Si es lo segundo,
      decidir cómo se defiende.
- [ ] ¿Ocean/Pontus-X es definitivo, o hay que evaluar EDC? (→ 13.7)
- [ ] ¿La analogía biológica (→ 1.6) entra como marco conceptual o solo como
      motivación en la introducción?
- [ ] ¿Se adopta el cegado en el pipeline de la 11? (→ 12.6)
- [ ] ¿Qué parte de los laboratorios 11–13 entra en la tesis y qué parte se queda
      como material de aprendizaje?
- [ ] Política de semillas y de presupuesto de ajuste, fijada para todo el
      proyecto (→ 3.9)
- [ ] **¿Qué estimador de OPE se usa como base?** (FQE, importance sampling
      marginalizada, error de Bellman tipo BVFT). Condiciona toda la idea ①

---

## Recordatorio final

> **El wow no viene de una técnica nueva. Viene de reencuadrar un problema.**
>
> Tienes una restricción que la literatura no tiene (coste por job), una
> infraestructura que la literatura no tiene (gobernanza y pagos), y un problema
> abierto que nadie ha conectado con ellas (evaluación en RL offline).
>
> Eso es suficiente. Lo que falta es elegir y empezar a medir.