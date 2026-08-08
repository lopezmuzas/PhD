---
title: "De la neurona biológica al aprendizaje en redes neuronales"
tags: [fundamentos, matematicas, neurociencia]
status: borrador
updated: 2026-08-08
---

# De la neurona biológica al aprendizaje en redes neuronales

> Guía de referencia. Recorre el funcionamiento de la neurona biológica (recepción,
> integración, disparo, plasticidad y neuromodulación) y su traducción formal a los
> algoritmos de aprendizaje usados en IA (TD-Learning, Actor-Critic, RLHF y las
> alternativas locales al backpropagation).
>
> **Hilo conductor:** el cerebro *predice* de forma continua (aprendizaje autosupervisado)
> y usa una señal escalar global (dopamina) para validar o invalidar las conexiones que
> participaron en esa predicción (aprendizaje por refuerzo).

---

## Índice

**Parte I — La neurona biológica**
1. [Dendritas: recepción de la señal](#1-dendritas-recepción-de-la-señal)
2. [Integración en el soma y decisión en el cono axónico](#2-integración-en-el-soma-y-decisión-en-el-cono-axónico)
3. [El umbral de disparo no es universal](#3-el-umbral-de-disparo-no-es-universal)
4. [Por qué existen sinapsis excitadoras e inhibidoras](#4-por-qué-existen-sinapsis-excitadoras-e-inhibidoras)
5. [Principio de Dale: una neurona, un neurotransmisor](#5-principio-de-dale-una-neurona-un-neurotransmisor)
6. [Dónde y cómo se liberan los neurotransmisores](#6-dónde-y-cómo-se-liberan-los-neurotransmisores)
7. [Cómo se decide qué neurona conecta con cuál](#7-cómo-se-decide-qué-neurona-conecta-con-cuál)
8. [Plasticidad: LTP, LTD y poda sináptica](#8-plasticidad-ltp-ltd-y-poda-sináptica)
9. [Memoria y aprendizaje como fuerza de conexión](#9-memoria-y-aprendizaje-como-fuerza-de-conexión)
10. [La dopamina como neuromodulador](#10-la-dopamina-como-neuromodulador)
11. [Error de Predicción de Recompensa (RPE)](#11-error-de-predicción-de-recompensa-rpe)

**Parte II — Traducción a inteligencia artificial**

12. [TD-Learning: la ecuación equivalente al RPE](#12-td-learning-la-ecuación-equivalente-al-rpe)
13. [Mapeo directo biología ↔ IA](#13-mapeo-directo-biología--ia)
14. [Arquitectura Actor-Critic](#14-arquitectura-actor-critic)
15. [Aclaración: arquitectura (atención) ≠ algoritmo de aprendizaje (RL)](#15-aclaración-arquitectura-atención--algoritmo-de-aprendizaje-rl)
16. [Dónde se cruzan Atención y Refuerzo: RLHF y Decision Transformers](#16-dónde-se-cruzan-atención-y-refuerzo-rlhf-y-decision-transformers)
17. [Plasticidad de tres factores: refuerzo sin cascada](#17-plasticidad-de-tres-factores-refuerzo-sin-cascada)
18. [Arquitecturas que implementan esta idea](#18-arquitecturas-que-implementan-esta-idea)
19. [Glosario de términos para investigar](#19-glosario-de-términos-para-investigar)

---

# Parte I — La neurona biológica

## 1. Dendritas: recepción de la señal

**Las dendritas no eligen ni piden carga de forma autónoma.** La presencia de carga
positiva o negativa depende exclusivamente de dos cosas:

1. El **tipo de neurotransmisor** que reciben de otras neuronas.
2. El **receptor específico** al que se une en la membrana dendrítica.

Cuando una neurona presináptica se activa, libera un neurotransmisor en la hendidura
sináptica que se acopla a los receptores de la dendrita. La naturaleza del neurotransmisor
determina qué canal iónico se abre.

### Señal positiva — Potencial Excitatorio Postsináptico (PEPS)

- **Mecanismo:** neurotransmisores excitadores (p. ej. **glutamato**) se unen a receptores
  que abren canales de **sodio (Na⁺)** o **calcio (Ca²⁺)**.
- **Efecto:** entran iones positivos, reduciendo la negatividad interna →
  **despolarización** de la membrana dendrítica.

### Señal negativa — Potencial Inhibitorio Postsináptico (PIPS)

- **Mecanismo:** neurotransmisores inhibidores (p. ej. **GABA**) se unen a receptores que
  abren canales de **cloro (Cl⁻)** o **potasio (K⁺)**.
- **Efecto:** entra carga negativa (Cl⁻) o sale carga positiva (K⁺), haciendo el interior
  aún más negativo → **hiperpolarización**.

> **Nota clave:** las dendritas conducen de forma **graduada y pasiva**, a diferencia del
> axón, que transmite impulsos de "todo o nada".

---

## 2. Integración en el soma y decisión en el cono axónico

### Conducción electrotónica

Las cargas que entran en la dendrita fluyen físicamente por el citoplasma hacia el soma,
**decayendo en intensidad** conforme avanzan (por distancia y resistencia).

### Suma en el soma

El soma actúa como **integrador algebraico**. Suma:

- **Espacialmente:** entradas simultáneas de miles de dendritas.
- **Temporalmente:** frecuencia de llegada de las señales.

Suma todos los PEPS (+) y PIPS (−).

### Decisión binaria en el cono axónico

Si la suma neta en la base del axón alcanza el **umbral de despolarización**
(≈ **−55 mV**, partiendo de un reposo de ≈ **−70 mV**), la neurona dispara un **potencial
de acción**. Si el saldo neto es negativo o no alcanza el umbral, la neurona permanece en
reposo.

### Secuencia completa del disparo

| Paso | Fenómeno |
|------|----------|
| 1. **Decisión binaria** | Suma algebraica ≥ umbral (≈ −55 mV) → se genera potencial de acción. Si no, 0 % de señal. Principio de **"todo o nada"**: no existen disparos parciales. |
| 2. **Propagación eléctrica** | El potencial de acción viaja como impulso autosostenido a lo largo del axón hasta los terminales axónicos (botones sinápticos). |
| 3. **Conversión eléctrica → química** | La llegada de carga abre canales de **Ca²⁺ dependientes de voltaje**. La entrada de calcio hace que las vesículas sinápticas se fusionen con la membrana y expulsen los neurotransmisores. |
| 4. **Recepción dendrítica** | Los neurotransmisores difunden por la hendidura sináptica y se unen a los receptores de la siguiente neurona. |

> **Resumen:** la transmisión es **eléctrica y binaria** dentro de la neurona (a lo largo
> del axón), pero **química y proporcional** al llegar a las dendritas de la receptora.

---

## 3. El umbral de disparo no es universal

El umbral **no es un valor fijo** ni entre personas ni entre neuronas del mismo cerebro.
Los −55 mV de los libros de texto son una referencia general; la cifra exacta varía
dinámicamente.

### 3.1 Diferencias según el tipo de neurona

- **Sensoriales vs. interneuronas:** una interneurona del córtex visual que procesa
  frecuencias altas de información puede tener un umbral **más bajo** (fácilmente
  excitable) que una motoneurona espinal grande, que requiere un impulso entrante mucho
  más fuerte.
- **Geometría del cono axónico:** la densidad de canales de sodio dependientes de voltaje
  (Na_v) donde nace el axón determina la facilidad de disparo. **A mayor densidad, menor
  umbral.**

### 3.2 Modulación dinámica (plasticidad)

- **Acomodación (historial de actividad):** si una neurona recibe estimulación lenta y
  continua sin llegar a disparar, los canales de sodio se inactivan progresivamente. Esto
  **eleva el umbral**, volviendo a la neurona momentáneamente resistente a activarse.
- **Neuromoduladores:** dopamina, serotonina o noradrenalina alteran la conductancia de
  los canales, desplazando el umbral arriba o abajo según el estado mental (atención,
  estrés, sueño).

### 3.3 Factores sistémicos y metabólicos

- **Electrólitos en sangre:** alteraciones de Ca²⁺, K⁺ o Na⁺ modifican el potencial de
  reposo. Ejemplo: la **hipocalcemia** acerca el potencial de reposo al umbral →
  hiperexcitabilidad y espasmos.
- **pH cerebral:** la **acidosis** (pH bajo) disminuye la excitabilidad (sube el umbral);
  la **alcalosis** (pH alto) la aumenta (baja el umbral).
- **Genética:** variaciones en genes de canales iónicos (**canalopatías**) hacen que
  algunas personas tengan un sistema nervioso inherentemente más excitable (predisposición
  a migrañas o epilepsia) o menos sensible.

---

## 4. Por qué existen sinapsis excitadoras e inhibidoras

Existen porque el sistema nervioso requiere **control, filtrado y procesamiento**, no solo
propagación ciega de señales. Sin inhibición, el cerebro entraría en sobreexcitación
constante (similar a una crisis epiléptica) y la información perdería todo su sentido.

### 4.1 Origen biológico

La naturaleza de la sinapsis **no se decide al azar en cada disparo**; viene determinada
desde el desarrollo celular:

- **Especialización genética:** durante el desarrollo embrionario las neuronas se
  diferencian para producir un neurotransmisor concreto. Las **neuronas piramidales** de
  la corteza se especializan en glutamato (excitador); las **interneuronas**, en GABA
  (inhibidor).
- **Compatibilidad de receptores:** la presináptica libera aquello para lo que está
  programada, y la membrana postsináptica expresa los receptores iónicos específicos que
  reaccionan abriendo canales de Na⁺ o de Cl⁻.

### 4.2 Razón funcional

| Función | Excitación (+) | Inhibición (−) |
|---|---|---|
| **Rol principal** | Transmitir datos y propagar la señal por la red | Modular, frenar y dar forma al mensaje |
| **Gating (filtro)** | Permite que el estímulo avance si supera un umbral | Bloquea el ruido de fondo e interrupciones irrelevantes |
| **Inhibición lateral** | Activa la vía del estímulo directo | Apaga las neuronas adyacentes para afinar la resolución (distinguir un punto exacto en la piel o la vista) |
| **Protección** | Genera la respuesta activa | Previene la **excitotoxicidad** (muerte celular por exceso de entrada de Ca²⁺) |

> **En resumen:** la excitación es el **acelerador** que mueve la información; la
> inhibición es el **freno y la dirección** que evitan el colapso y dan precisión al
> procesamiento.

---

## 5. Principio de Dale: una neurona, un neurotransmisor

Como regla general, **cada neurona libera un solo tipo principal de neurotransmisor
clásico**: es excitadora o inhibidora, no ambas a la vez. Este concepto se conoce como
**Principio de Dale**.

### 5.1 Por qué se especializa

- **Maquinaria enzimática específica:** para liberar un neurotransmisor la neurona debe
  sintetizarlo y empaquetarlo en vesículas, lo que requiere enzimas y transportadores
  determinados por su expresión genética:
  - **Excitadoras (glutamatérgicas):** expresan el transportador **VGLUT**, empaquetando
    glutamato → PEPS.
  - **Inhibidoras (GABAérgicas):** expresan la enzima **GAD** (transforma glutamato en
    GABA) y el transportador **VGAT** → PIPS.
- **Identidad de red rígida:** las ramas del axón de una misma neurona contactan con
  cientos o miles de dendritas postsinápticas, pero **en todas sus terminales libera la
  misma sustancia**. No puede enviar señal positiva por una rama y negativa por otra.

### 5.2 Excepciones y matices

- **Cotransmisión con neuropéptidos:** muchas neuronas liberan su neurotransmisor
  principal (glutamato, acetilcolina) junto con un **neuropéptido modulador** (sustancia
  P, endorfinas, neuropéptido Y) para ajustar intensidad o duración de la respuesta.
- **Co-liberación dual (casos raros):** en circuitos muy específicos (habénula lateral, o
  fases tempranas del desarrollo embrionario) se han descubierto neuronas capaces de
  co-liberar glutamato y GABA simultáneamente. En el cerebro adulto estándar la
  segregación es **casi absoluta**, para evitar interferencia de señales.

### 5.3 La cadena resultante, en dos pasos

1. **La emisora decide el "qué":** si es glutamatérgica libera glutamato; si es GABAérgica
   libera GABA.
2. **La receptora reacciona alterando su voltaje:**
   - **Subir la carga (excitación / despolarización):** el glutamato abre canales que
     dejan entrar cationes (Na⁺, Ca²⁺). El voltaje sube desde −70 mV hacia valores más
     positivos, **acercando** al umbral.
   - **Bajar la carga (inhibición / hiperpolarización):** el GABA abre canales que dejan
     entrar aniones (Cl⁻) o salir cationes (K⁺). El voltaje baja (p. ej. −80 mV),
     **alejando** del umbral.

> **Quien emite fija la polaridad; quien recibe la suma a su saldo eléctrico actual.**

---

## 6. Dónde y cómo se liberan los neurotransmisores

Los neurotransmisores se liberan **exclusivamente por los botones sinápticos**, en el
extremo de las terminales axónicas. **No se sueltan de forma generalizada sobre las capas
del cerebro.** Hay dos escalas microscópicas bien definidas:

### 6.1 Sinapsis clásica — comunicación punto a punto

- **Destino:** el axón libera el neurotransmisor en la **hendidura sináptica**, un espacio
  de apenas **20 nanómetros** entre el botón axónico y la espina dendrítica receptora.
- **Alcance:** la señal es **privada**. Cruza el espacio en microsegundos y se une a los
  receptores de la neurona directamente conectada.
- **Reciclaje inmediato:** enzimas degradadoras en el espacio sináptico y transportadores
  que reabsorben el sobrante hacia el propio axón impiden que el químico se extienda.

### 6.2 Transmisión de volumen — difusión local

La excepción, que funciona como una "emisión de radio" local:

- **Mecanismo:** ciertas neuronas (sobre todo las que liberan **dopamina, serotonina o
  noradrenalina**) presentan abultamientos a lo largo del axón (**varicosidades**) que
  liberan el neurotransmisor directamente al líquido extracelular.
- **Alcance:** el químico difunde por el espacio intercelular y afecta a un grupo de
  neuronas cercanas con los receptores adecuados, **aunque no exista sinapsis
  estructurada** con ellas.

---

## 7. Cómo se decide qué neurona conecta con cuál

**No es aleatorio.** La conectividad está regida por un programa genético de guía molecular
durante el desarrollo, más un ajuste fino posterior basado en la experiencia.

### 7.1 Guía axónica: el "GPS" químico (fase embrionaria)

La neurona emite su axón, en cuya punta hay una estructura exploradora: el **cono de
crecimiento**. Navega hacia su destino siguiendo señales químicas del tejido.

- **Quimiotaxis (atrayentes y repelentes):** las células del entorno secretan moléculas
  (**Netrinas, Semaforinas, Slits**). El cono detecta **gradientes de concentración**:
  avanza hacia los atrayentes y se aleja de los repelentes.
- **Adhesión al sustrato:** proteínas de superficie (**Cadherinas, CAMs**) actúan como
  "vías de tren" físicas por las que el axón se desplaza.

### 7.2 Reconocimiento y compatibilidad de la sinapsis

- **Proteínas de acoplamiento específico:** pares de moléculas de adhesión sináptica
  (**Neurexinas** en el axón, **Neuroliginas** en la dendrita) encajan como llave y
  cerradura.
- **Especialización espacial estratégica:**
  - Las sinapsis **excitadoras** (glutamatérgicas) suelen conectarse en las **espinas
    dendríticas** (periferia).
  - Las sinapsis **inhibidoras** (GABAérgicas) se conectan frecuentemente en el **soma o
    cerca del cono axónico**. Motivo: una señal negativa aplicada en la base del cuerpo
    celular puede **vetar de golpe** todas las señales positivas entrantes antes de que
    lleguen al axón.

### 7.3 Poda sináptica: ajuste final mediante el uso

La genética crea el mapa inicial **con exceso de cableado**. Después, la actividad
eléctrica filtra y consolida el circuito definitivo.

- **Regla de Hebb** — *"neuronas que disparan juntas, se conectan juntas"*: si la conexión
  se usa con frecuencia y es eficaz, la sinapsis se refuerza (potenciación a largo plazo).
- **Poda (pruning):** las sinapsis inútiles, redundantes o sin actividad regular son
  eliminadas bioquímicamente por las células gliales (**microglía**).

### 7.4 ¿Todos los cerebros son iguales?

| Nivel | Grado de similitud |
|---|---|
| **Macro (anatomía y vías)** | **Altamente similares.** Mismas estructuras (córtex, hipocampo, cerebelo, amígdala), mismas vías principales (la vía visual conecta retina y lóbulo occipital igual en todos), mismo balance general excitación/inhibición (≈ **80 % excitadoras / 20 % inhibidoras** en corteza). |
| **Micro (conectoma)** | **Absolutamente únicos**, más que una huella dactilar. |

Causas de la unicidad a nivel micro:

- **Genética individual:** pequeñas variaciones del ADN alteran densidad de neuronas o
  receptores.
- **Entorno:** cada vivencia, memoria, idioma o habilidad modifica la fuerza de conexiones
  específicas y reconecta redes locales.
- **Gemelos idénticos:** incluso los monocigóticos (100 % del código genético compartido)
  desarrollan conectomas distintos, porque las experiencias y las micro-decisiones de
  guiado axónico durante la gestación y la vida nunca son idénticas.

> **Reparto de responsabilidades:**
> - **Fase embrionaria (macroestructura):** los genes dictan qué neuronas nacen en cada
>   región, qué neurotransmisor producirán y hacia qué zonas extenderán sus axones.
> - **Tras el nacimiento (microestructura):** experiencia, aprendizaje y entorno moldean
>   las conexiones finas mediante poda y plasticidad.

---

## 8. Plasticidad: LTP, LTD y poda sináptica

Principio fundamental: **plasticidad estructural**, o *"lo que no se usa, se elimina"*.

### 8.1 Crecimiento físico — Potenciación a Largo Plazo (LTP)

Cuando una sinapsis se activa con frecuencia:

- **Crece en tamaño:** la espina dendrítica (punto de contacto) se expande físicamente.
- **Aumenta su densidad:** se anclan más receptores en la membrana postsináptica, captando
  el neurotransmisor con mayor eficacia.
- **Produce más señal:** el axón presináptico aumenta su capacidad de liberación de
  vesículas.

### 8.2 Encogimiento y poda — Depresión a Largo Plazo (LTD)

Si la conexión no transmite información relevante durante un tiempo:

- La espina dendrítica pierde receptores y reduce su volumen gradualmente.
- La **microglía** detecta la sinapsis inactiva, la marca químicamente y la elimina.

### 8.3 Aplica por igual a (+) y a (−)

La naturaleza del neurotransmisor **no altera la regla de uso**. Lo único que cambia es el
efecto funcional del refuerzo:

- **Reforzar una conexión excitadora (+):** crea un **acelerador más potente** (el
  estímulo desencadena la respuesta con más facilidad y velocidad).
- **Reforzar una conexión inhibidora (−):** crea un **freno más potente** (clave para
  afinar precisión motora, enfocar la atención o suprimir respuestas no deseadas).

> **Conclusión:** el cerebro no almacena datos de forma estática, sino **remodelando
> físicamente la fuerza y el volumen** de sus trillones de conexiones en función de la
> actividad.

---

## 9. Memoria y aprendizaje como fuerza de conexión

La fuerza de las conexiones sinápticas y el patrón físico en que se organizan constituyen
**la base biológica del aprendizaje y la memoria**.

### 9.1 Aprender = modificar conexiones

Aprender **no consiste en añadir neuronas nuevas**, sino en alterar la fuerza de la
comunicación entre las que ya existen:

1. Al experimentar algo nuevo o practicar una habilidad, un grupo específico de neuronas
   se activa secuencialmente.
2. Si la activación se repite, esas sinapsis sufren **LTP**: más receptores
   postsinápticos, mayor volumen de la espina dendrítica.
3. **Resultado práctico:** la resistencia eléctrica de ese camino baja. En adelante hace
   falta mucho menos esfuerzo o estímulo para que la señal fluya por esa ruta.

### 9.2 La memoria es la facilidad para reactivar un circuito

La memoria no se guarda como un archivo en un lugar concreto, sino como un **engrama**: la
huella física o red de conexiones fortalecidas.

- **Recordar** = volver a encender el mismo patrón de neuronas que se activó al aprender o
  vivir la experiencia.
- Como el camino se hizo físicamente más fuerte y eficiente, activar **solo un extremo** de
  la red (un olor, una palabra clave, una imagen) desencadena automáticamente la activación
  del resto del circuito.

### 9.3 La repetición consolida la estructura

- **Uso frecuente:** la estimulación repetida indica a la célula que debe sintetizar
  proteínas estructurales para hacer la conexión permanente (**memoria a largo plazo**).
- **Falta de uso:** si el circuito no se reactiva, la conexión se debilita mediante **LTD**
  hasta desmantelarse — el proceso biológico del **olvido**.

---

## 10. La dopamina como neuromodulador

### 10.1 Alcance: transmisión por volumen

La dopamina es el ejemplo clásico de **transmisión por volumen** (difusión
extrasináptica):

- Los axones dopaminérgicos presentan **miles de varicosidades** a lo largo de sus ramas.
- Liberan dopamina directamente al espacio extracelular. El químico difunde como una
  **"nube" local** que baña el tejido circundante, modulando simultáneamente cientos de
  neuronas vecinas con receptores dopaminérgicos.

### 10.2 ¿Positiva o negativa? Depende del receptor

La dopamina **no es intrínsecamente excitadora ni inhibidora**. A diferencia del glutamato
(que casi siempre suma carga) o del GABA (que siempre la resta), es un **neuromodulador**.
Su efecto depende de la familia de receptores acoplados a proteína G:

| Familia | Receptores | Vía | Efecto |
|---|---|---|---|
| **Tipo D1** | D1, D5 | Proteína **Gs** | **Excitador (+):** aumenta la sensibilidad de la neurona y facilita la entrada de cargas positivas, acercándola al umbral. |
| **Tipo D2** | D2, D3, D4 | Proteína **Gi** | **Inhibidor (−):** reduce el AMP cíclico interno, abre canales de K⁺ o cierra los de Ca²⁺. Salen cargas positivas → hiperpolariza y frena el disparo. |

> **Doble efecto simultáneo:** en un mismo circuito (p. ej. los ganglios basales, que
> controlan movimiento y motivación), la misma liberación de dopamina puede **activar la
> vía directa** (vía D1) e **inhibir la vía indirecta** (vía D2) a la vez.

---

## 11. Error de Predicción de Recompensa (RPE)

Este es el punto donde converge todo lo anterior: circuito excitador/inhibidor, dopamina
por volumen y plasticidad LTP/LTD. En neurociencia se llama **Reward Prediction Error
(RPE)** y es la base del aprendizaje por refuerzo en el cerebro.

### 11.1 Cómo calcula el cerebro la predicción

El cerebro es una **máquina de comparar** constante:

1. **La expectativa (top-down):** áreas superiores (como la corteza prefrontal) envían
   señales **excitadoras (+)** hacia las redes del sistema de recompensa (**área tegmental
   ventral, VTA**) con el resultado que esperan obtener.
2. **La resta con la realidad:** cuando llega la experiencia real, interneuronas
   **inhibidoras GABAérgicas (−)** "restan" la expectativa de la realidad recibida:

```
Error = Resultado Real − Resultado Esperado
```

### 11.2 El papel de la dopamina según el resultado

El resultado de esa resta altera la tasa de disparo de las neuronas dopaminérgicas, que
habitualmente emiten una señal base de unos **5 Hz**:

| Caso | Situación | Respuesta dopaminérgica |
|---|---|---|
| **Error positivo** | Mejor de lo esperado: la señal real supera la inhibición de la expectativa | **Disparo fásico masivo** (liberación rápida de dopamina por volumen) |
| **Error cero** | Exactamente lo esperado: el estímulo real coincide con la inhibición de la expectativa | Sin cambio; nivel base. **No hay aprendizaje nuevo**, el cerebro ya sabía lo que pasaría |
| **Error negativo** | Peor de lo esperado: la expectativa (−) supera a la señal real | **Pausa o caída brusca** por debajo del nivel base |

### 11.3 Traducción a cambio físico en la conexión

La dopamina **no transmite el contenido del recuerdo**: actúa como un **interruptor de
actualización de pesos sinápticos** para las neuronas que estuvieron activas justo antes
del resultado.

- **Pico de dopamina (+RPE) → LTP.** La dopamina se une a receptores **tipo D1** de la
  dendrita postsináptica, activando cascadas enzimáticas intracelulares que instan a la
  célula a insertar más receptores de glutamato y agrandar la espina dendrítica.
  **Resultado:** ese camino neuronal se consolida de forma prioritaria porque "dio un
  resultado con superávit".
- **Caída de dopamina (−RPE) → LTD.** La ausencia transitoria de dopamina impide la
  activación de los receptores D1 e incrementa los mecanismos de depresión a largo plazo.
  Las espinas retiran receptores y reducen volumen. **Resultado:** el circuito que llevó a
  la predicción fallida pierde fuerza, para que la conducta no se repita.

> **Resumen biológico:** el cerebro usa la **inhibición (−)** para restar la expectativa de
> la realidad, la **dopamina** para comunicar el margen de error mediante transmisión por
> volumen, y la **plasticidad (LTP/LTD)** para ensanchar o estrechar físicamente las
> conexiones que causaron dicho error.

---

# Parte II — Traducción a inteligencia artificial

## 12. TD-Learning: la ecuación equivalente al RPE

El modelo RPE es la base matemática sobre la que se construyeron los algoritmos de
**Aprendizaje por Refuerzo (RL)**, concretamente el **Aprendizaje por Diferencia Temporal
(TD-Learning)** y las arquitecturas **Actor-Critic**.

La señal equivalente al pico o caída de dopamina se llama **Error de Diferencia Temporal
(δₜ)**:

```
δₜ = rₜ + γ · V(sₜ₊₁) − V(sₜ)
```

| Término | Significado |
|---|---|
| **rₜ** | **Recompensa inmediata real** que el entorno entrega al agente. |
| **γ · V(sₜ₊₁)** | **Expectativa futura:** estimación del valor de los estados futuros, descontada por un factor γ ∈ [0, 1]. |
| **V(sₜ)** | **Expectativa previa:** la predicción que el agente tenía sobre el valor del estado actual antes de actuar. |

---

## 13. Mapeo directo biología ↔ IA

| Fenómeno biológico | Algoritmo de IA (RL) | Impacto en el modelo |
|---|---|---|
| **Error positivo (+RPE)** — pico de disparo fásico de dopamina | **δₜ > 0** — el resultado superó la predicción | Se **incrementan los pesos (W)** asociados a la acción tomada. La probabilidad de repetir ese comportamiento aumenta. |
| **Sin error (RPE = 0)** — liberación basal de dopamina | **δₜ = 0** — el resultado coincidió con lo previsto | **No se modifican los pesos.** La predicción era precisa y el modelo permanece estable. |
| **Error negativo (−RPE)** — pausa/caída del disparo | **δₜ < 0** — el resultado fue peor de lo esperado | Se **penaliza la acción** reduciendo los pesos. La probabilidad de repetir el comportamiento disminuye. |

---

## 14. Arquitectura Actor-Critic

Este paralelismo biológico dio lugar a una de las estructuras más potentes del RL moderno.

```
                  [ ENTORNO ]
                   /        \
   (Recompensa / Estado)   (Acción tomada)
                 /            \
         [ CRÍTICO ]  ----->  [ ACTOR ]
      Calcula el TD Error   Actualiza la política
             (δₜ)                  (π)
```

### El Crítico — equivalente al VTA / estriado ventral

- Mantiene una **función de valor V(s)**.
- Mide la diferencia entre lo esperado y lo obtenido para generar el escalar **δₜ** (la
  "dopamina artificial").

### El Actor — equivalente al córtex motor / ganglios basales

- Mantiene la **política de acción π(a|s)**.
- Usa el **signo y la magnitud de δₜ** enviado por el Crítico para actualizar la matriz de
  pesos de su red mediante descenso de gradiente:

```
θₜ₊₁ = θₜ + α · δₜ · ∇ log π_θ(aₜ | sₜ)
```

### Por qué esta transferencia es crucial

- **Eficiencia muestral:** igual que el cerebro no empieza de cero en cada intento, el uso
  de la predicción de estados futuros (γ·V(sₜ₊₁)) permite aprender **de la diferencia
  temporal** sin esperar al final definitivo del proceso.
- **Ajuste continuo (online learning):** permite reajustar pesos en tiempo real tras cada
  interacción con el entorno, optimizando tanto agentes virtuales (AlphaZero, OpenAI Five)
  como el control de robótica física.

---

## 15. Aclaración: arquitectura (atención) ≠ algoritmo de aprendizaje (RL)

Es fácil mezclar dos conceptos que operan en planos distintos.

| Concepto | Qué es | Pregunta que responde |
|---|---|---|
| **Ajuste por RPE (dopamina / RL)** | Método de aprendizaje por ensayo y error: el modelo actúa, recibe recompensa del entorno y ajusta pesos según δₜ | **Cómo se aprende** |
| **Mecanismo de atención (Transformers)** | Arquitectura de red que decide **qué partes de la información entrante son más relevantes** respecto a otras, mediante matrices Query, Key y Value | **Cómo se procesa la información** |

**Un modelo con atención no aprende por sí solo mediante "dopamina" o recompensas.**
Aprende ajustando sus parámetros minimizando una **función de pérdida (loss function)**.

### Cómo aprende un modelo con atención estándar

La inmensa mayoría de modelos basados en atención (GPT, BERT) **no se entrenan inicialmente
con RL**, sino con **aprendizaje supervisado / autosupervisado**:

1. **Predicción del siguiente token:** el modelo lee una secuencia y predice la siguiente
   palabra.
2. **Cálculo del error:** compara la predicción con la palabra real mediante una función
   de pérdida (**Cross-Entropy Loss**).
3. **Retropropagación (backpropagation):** calcula el gradiente exacto del error y
   actualiza los pesos de las matrices de atención con algoritmos como **AdamW**. Aquí
   **no hay entorno, ni recompensas, ni modelo Crítico**.

> **En una frase:** la **atención** define *cómo la red conecta internamente los datos que
> procesa*; el **RL (RPE)** es el método que decide *cómo cambiar la fuerza de esas
> conexiones en función de recompensas externas*.

---

## 16. Dónde se cruzan Atención y Refuerzo: RLHF y Decision Transformers

### 16.1 RLHF — Reinforcement Learning from Human Feedback

Proceso usado en los LLM modernos:

- **Fase 1 (atención pura):** se entrena el modelo de atención con millones de textos
  (aprendizaje supervisado / autosupervisado).
- **Fase 2 (RL con ajuste de pesos):** se congela parte de la red y se aplica un algoritmo
  de RL (como **PPO**). Un **Modelo de Recompensa** (el Crítico) evalúa la respuesta del
  Transformer (el Actor) y envía una **señal escalar equivalente a la dopamina (δₜ)** para
  ajustar los pesos del mecanismo de atención según si la respuesta fue útil, segura y
  precisa.

### 16.2 Decision Transformers

Arquitecturas donde la red con atención procesa directamente **trayectorias de RL**
—secuencias (Estadoₜ, Acciónₜ, Recompensaₜ)— tratando el problema de toma de decisiones
**como un problema de predicción de secuencias**.

---

## 17. Plasticidad de tres factores: refuerzo sin cascada

> **La pregunta de partida:** ¿se podría hacer un sistema de aprendizaje no supervisado en
> el que, cuando el valor no sea el esperado, en lugar de propagar el error en cascada por
> backpropagation, se refuercen o debiliten los pesos de las neuronas que participaron,
> como hace la dopamina?

**Sí.** Es un campo activo de investigación en neurociencia computacional y computación
neuromórfica. Se conoce como **Plasticidad de Tres Factores** o **Dopamine-Modulated
Hebbian Learning**, y fue diseñado precisamente para resolver las limitaciones físicas y
biológicas del backpropagation tradicional.

### 17.1 Por qué el backpropagation clásico funciona "en cascada"

El backprop convencional aplica la regla de la cadena desde la última capa hasta la primera:

```
∂L/∂w⁽ˡ⁾ = (∂L/∂a⁽ᴸ⁾) · (∂a⁽ᴸ⁾/∂a⁽ᴸ⁻¹⁾) · … · (∂a⁽ˡ⁾/∂w⁽ˡ⁾)
```

**El problema:** requiere que cada capa conozca de forma simétrica los pesos y gradientes
de **todas** las capas posteriores (**Weight Transport Problem**). Esto genera una
dependencia secuencial estricta que obliga a almacenar todas las activaciones en memoria
hasta completar el paso hacia atrás — algo que no ocurre en un cerebro biológico ni resulta
eficiente en arquitecturas paralelas masivas.

### 17.2 Cómo funciona la plasticidad de tres factores

Elimina la cascada dividiendo la actualización en **dos niveles desacoplados**:

**Factor 1 y 2 — ajuste local no supervisado**

Cuando dos neuronas conectadas se activan juntas, la sinapsis **no cambia su peso
inmediatamente**: genera una **traza de elegibilidad** (eᵢⱼ), una marca bioquímica local y
no supervisada que dice *"esta conexión participó en el procesamiento reciente"*:

```
eᵢⱼ(t) = actividad(i) × actividad(j)
```

**Factor 3 — modulación por señal global / dopamina**

En lugar de retropropagar un vector de gradientes complejo, el sistema emite una **única
señal escalar global δ** (el error de predicción o "dopamina"). Esa señal **impregna la
red** sin pasar capa por capa:

```
Δwᵢⱼ = η · δ · eᵢⱼ
```

| Señal | Efecto |
|---|---|
| **δ > 0** (recompensa / error positivo) | La traza de elegibilidad local se **valida** y el peso wᵢⱼ se refuerza |
| **δ < 0** (penalización / error negativo) | La traza local se **debilita** |
| **δ = 0** | La traza **se borra progresivamente** sin alterar los pesos |

---

## 18. Arquitecturas que implementan esta idea

### 18.1 Codificación Predictiva (Predictive Coding)

Cada capa de la red intenta predecir de forma **no supervisada** la actividad de la capa
adyacente. Los errores de predicción se calculan y corrigen de forma **local e
independiente** en cada capa, eliminando la necesidad de una cascada global.

### 18.2 Redes Neuronales de Impulsos (SNN) con R-STDP

Usan la regla **Reward-modulated Spike-Timing-Dependent Plasticity**: las neuronas se
ajustan localmente por la diferencia de tiempo entre sus disparos, y una señal global tipo
dopamina consolida o descarta los cambios.

### 18.3 Direct Feedback Alignment (DFA)

En lugar de pasar el error capa por capa hacia atrás, la señal de error de la capa final se
proyecta **directamente a cada capa oculta** a través de matrices fijas aleatorias,
saltándose la cascada.

### 18.4 Por qué no ha sustituido al backprop en el DL comercial

- **Eficiencia en GPUs:** el backpropagation, aunque biológicamente inviable, se adapta a
  la perfección al **cálculo matricial denso** de las GPUs modernas.
- **Hardware específico:** las reglas locales moduladas por señales globales son
  ineficientes en GPU, pero ofrecen rendimiento y eficiencia energética superiores en
  **chips neuromórficos** (hardware analógico o de impulsos como **Intel Loihi**,
  **SpiNNaker** o **BrainScaleS**).

---

## 19. Glosario de términos para investigar

Términos técnicos para buscar literatura sobre este paradigma de entrenamiento sin
backpropagation tradicional en cascada.

### 1. Plasticidad de Tres Factores (Three-Factor Plasticity / Neo-Hebbian Learning)
- **Qué es:** modelo matemático donde la actualización de un peso depende de tres
  elementos: actividad presináptica, actividad postsináptica y un tercer factor modulador
  global (dopamina o escalar de error).
- **Por qué leerlo:** es la formulación exacta que sustituye el gradiente en cascada por un
  refuerzo/castigo global aplicado a conexiones locales.

### 2. Trazas de Elegibilidad (Eligibility Traces / e-prop)
- **Qué es:** mecanismo temporal que mantiene "marcada" una sinapsis durante un breve
  periodo tras activarse, a la espera de si llega o no una señal de recompensa. Algoritmos
  como **e-prop** (eligibility propagation) lo aplican a redes recurrentes.
- **Por qué leerlo:** resuelve el problema de saber **qué neuronas exactas participaron**
  en una decisión antes de conocerse el resultado final.

### 3. Codificación Predictiva (Predictive Coding)
- **Qué es:** arquitectura donde cada capa intenta predecir el estado de la siguiente de
  forma autosupervisada. Solo los **errores de predicción local** se transmiten entre
  capas.
- **Por qué leerlo:** demuestra matemáticamente cómo una red puede aprender a predecir el
  entorno y ajustar pesos de forma **paralela y local**, sin retropropagar un único vector
  de error global.

### 4. R-STDP (Reward-modulated Spike-Timing-Dependent Plasticity)
- **Qué es:** regla de aprendizaje para **Spiking Neural Networks** donde el ajuste de
  pesos por coincidencia de disparo temporal es modulado por una señal de recompensa/error
  retrasada.
- **Por qué leerlo:** es la implementación más extendida en simulación biológica para
  combinar coincidencia local de eventos con refuerzo tipo dopamina.

### 5. Asignación de Crédito (Credit Assignment Problem)
- **Qué es:** el problema teórico fundamental sobre cómo determinar la **contribución
  individual** de cada nodo o peso dentro de un sistema complejo ante un resultado global.
- **Por qué leerlo:** es el marco teórico general que compara cómo lo resuelve el
  backpropagation (cálculo diferencial) frente a los métodos biológicos (trazas locales +
  señales moduladoras).

### 6. Direct Feedback Alignment (DFA)
- **Qué es:** técnica donde el error de la capa final se proyecta directamente a cada capa
  oculta mediante matrices de pesos aleatorios e invariables.
- **Por qué leerlo:** rompe la dependencia secuencial en cascada del backpropagation,
  permitiendo que las capas adapten sus pesos de forma casi independiente.

### 7. Computación Neuromórfica (Neuromorphic Computing)
- **Qué es:** rama de la ingeniería de hardware que diseña chips analógicos o digitales
  (Intel Loihi, BrainScaleS) optimizados para ejecutar **reglas de aprendizaje local** en
  lugar de multiplicación matricial densa.
- **Por qué leerlo:** es donde estos modelos (no supervisado + refuerzo) muestran su
  ventaja práctica real: reducción drástica del consumo energético y procesamiento en
  tiempo real.

---

## Anexo — Mapa de correspondencias en una tabla

| Biología | IA |
|---|---|
| Peso sináptico (nº de receptores, tamaño de espina) | Peso wᵢⱼ de la red |
| PEPS / PIPS | Contribución positiva / negativa a la preactivación |
| Suma en el soma | Suma ponderada Σ wᵢxᵢ + b |
| Umbral del cono axónico | Función de activación / umbral |
| Potencial de acción ("todo o nada") | Spike (en SNN) / activación |
| Regla de Hebb | Aprendizaje hebbiano, correlación pre×post |
| LTP / LTD | Incremento / decremento de pesos |
| Poda sináptica (microglía) | Pruning de red, regularización, sparsity |
| Dopamina (RPE) | Escalar δₜ de TD-Learning |
| VTA / estriado ventral | Crítico (función de valor V(s)) |
| Córtex motor / ganglios basales | Actor (política π(a\|s)) |
| Traza bioquímica postsináptica | Traza de elegibilidad eᵢⱼ |
| Transmisión por volumen | Señal global broadcast a toda la red |
| Conectoma individual | Configuración concreta de parámetros entrenados |

---

## Preguntas abiertas para profundizar

Temas que quedaron apuntados y merecen seguimiento:

- Cómo la **mielina** acelera la propagación del impulso en el axón.
- Cómo ejecuta la **microglía** la poda sináptica durante el desarrollo.
- Cómo los **astrocitos** limpian el exceso de neurotransmisores para que la señal no se
  disperse.
- Cómo se estudia el **conectoma humano** para mapear todas las conexiones.
- Cómo la **repetición espaciada** aprovecha el mecanismo de consolidación sináptica.
- Comparativa de rendimiento entre **Backpropagation, Predictive Coding y R-STDP**.
- Funcionamiento del doble efecto D1/D2 de la dopamina en el **sistema de recompensa**.
