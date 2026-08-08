---
title: "Aprendizaje federado: pros, contras y por qué es complejo"
tags: [federado, distribuido, data-spaces]
status: borrador
updated: 2026-08-08
---

# Aprendizaje federado: pros, contras y por qué es complejo

> **Resumen ejecutivo.** El aprendizaje federado (FL) entrena un modelo compartido entre múltiples participantes sin que los datos salgan nunca de su origen: viaja el modelo (pesos/gradientes), no el dato. Esa única decisión de diseño explica a la vez todas sus ventajas (privacidad, soberanía, acceso a datos antes inaccesibles) y todas sus complejidades (heterogeneidad, comunicación, seguridad, coordinación). Lo que en aprendizaje centralizado es un supuesto gratuito —datos IID, accesibles, en una sola máquina confiable— en FL se convierte, punto por punto, en un problema de investigación.

---

## 1. Pros del aprendizaje federado

| # | Ventaja | Qué aporta |
|---|---|---|
| P1 | **Privacidad por diseño** | Los datos crudos nunca abandonan al participante; se minimiza la superficie de exposición y se facilita el cumplimiento normativo (RGPD, Data Act, regulación sectorial). |
| P2 | **Soberanía del dato** | Cada participante mantiene el control y puede definir condiciones de uso; encaja de forma natural con los data spaces (Gaia-X / IDS). |
| P3 | **Acceso a datos antes inutilizables** | Permite entrenar con datos que la competencia, la regulación o la sensibilidad hacían imposibles de centralizar (hospitales, bancos, fábricas, móviles). El modelo resultante ve más diversidad de la que ningún silo posee por sí solo. |
| P4 | **Mejor generalización potencial** | Al aprender de distribuciones de muchos participantes, el modelo global puede generalizar mejor que cualquier modelo entrenado en un solo silo. |
| P5 | **Eficiencia de transferencia** | Mover pesos suele ser más barato que mover datasets masivos; los datos en bruto (vídeo, sensores) pueden ser órdenes de magnitud mayores que el modelo. |
| P6 | **Cómputo distribuido** | El coste de entrenamiento se reparte entre los participantes en lugar de concentrarse en un único centro de datos. |
| P7 | **Reducción de responsabilidad legal** | Quien orquesta no custodia datos de terceros: menos riesgo ante brechas y menos carga de cumplimiento como responsable del tratamiento. |

## 2. Contras del aprendizaje federado

| # | Desventaja | Coste que implica |
|---|---|---|
| C1 | **Rendimiento generalmente inferior al centralizado** | Con los mismos datos, el modelo federado suele rendir algo peor que si todos los datos estuvieran juntos; la brecha crece con la heterogeneidad. |
| C2 | **La privacidad no es perfecta por sí sola** | Los pesos y gradientes filtran información (ataques de inversión y de pertenencia). Para garantías fuertes hay que añadir privacidad diferencial, agregación segura o cifrado homomórfico — cada una con coste en rendimiento o en cómputo. |
| C3 | **Complejidad de ingeniería muy superior** | Orquestación de rondas, versionado de modelos, tolerancia a fallos, clientes que se caen, sincronización: un pipeline centralizado de unas líneas se convierte en un sistema distribuido completo. |
| C4 | **Coste de comunicación** | Decenas o cientos de rondas de subida/bajada de modelos; con redes profundas grandes, la comunicación puede dominar el coste total. |
| C5 | **Depuración y observabilidad limitadas** | No puedes inspeccionar los datos que causan un mal comportamiento del modelo: diagnosticar problemas de calidad de datos sin ver los datos es intrínsecamente difícil. |
| C6 | **Nueva superficie de ataque** | Participantes maliciosos pueden envenenar el modelo o insertar puertas traseras; el servidor agregador es un punto único de fallo y de confianza. |
| C7 | **Gobernanza y coordinación humanas** | Acordar el problema, el esquema de datos, las métricas y los incentivos entre organizaciones independientes es a menudo más difícil que la parte técnica. |
| C8 | **Evaluación y reproducibilidad complicadas** | Sin un conjunto de test centralizado y representativo, validar el modelo global y auditar qué contribuyó cada quien es un problema abierto. |

## 3. Lista de retos y por qué cada uno es complejo

La raíz común: **el FL rompe los supuestos sobre los que se construyó todo el aprendizaje automático clásico**. SGD y sus garantías de convergencia asumen muestras IID de una sola distribución; la validación asume acceso a los datos; la confianza asume una sola organización. El FL viola los tres a la vez.

### R1 — Heterogeneidad estadística (datos no-IID)
**El reto:** cada cliente tiene una distribución de datos distinta (un hospital pediátrico vs. uno geriátrico; teclados en idiomas distintos).
**Por qué es complejo:** el promediado de pesos (FedAvg) asume implícitamente que los gradientes locales apuntan "hacia el mismo sitio". Con datos no-IID, cada cliente arrastra el modelo hacia su óptimo local (*client drift*): los entrenamientos locales divergen y su media puede no ser buena para nadie. Las correcciones (FedProx, SCAFFOLD, personalización) mitigan pero introducen nuevos hiperparámetros y trade-offs entre modelo global y modelos personalizados. En el fondo es un problema sin solución universal: si las distribuciones difieren bastante, "un único modelo global óptimo" puede ni siquiera existir.

### R2 — Heterogeneidad de sistemas
**El reto:** los participantes tienen hardware, conectividad y disponibilidad dispares (de un móvil con batería baja a un clúster hospitalario).
**Por qué es complejo:** la ronda avanza al ritmo del cliente más lento (*stragglers*); esperar a todos es lento, descartar a los lentos sesga el modelo hacia los clientes potentes (que pueden tener datos sistemáticamente distintos). Los esquemas asíncronos resuelven la espera pero rompen las garantías de convergencia: se agregan actualizaciones calculadas sobre versiones obsoletas del modelo (*staleness*).

### R3 — Coste de comunicación
**El reto:** cada ronda mueve el modelo completo en ambas direcciones, por muchos clientes y muchas rondas.
**Por qué es complejo:** existe un triángulo de tensión entre comunicación, cómputo y precisión: menos rondas exigen más épocas locales, pero más entrenamiento local agrava el client drift (R1); comprimir o cuantizar los pesos reduce tráfico pero añade ruido al gradiente. Optimizar una esquina del triángulo empeora otra, y el punto óptimo depende de cada despliegue.

### R4 — Privacidad residual
**El reto:** demostrar que de los pesos compartidos no se puede reconstruir información individual.
**Por qué es complejo:** los gradientes son funciones de los datos: con suficientes rondas, un atacante (incluido el propio servidor) puede invertirlos parcialmente o inferir pertenencia. La defensa rigurosa, la privacidad diferencial, añade ruido calibrado que degrada la precisión, y el presupuesto de privacidad se *consume* con cada ronda — justo en un paradigma que necesita muchas rondas. La agregación segura y el cifrado homomórfico protegen sin ruido, pero multiplican el coste computacional y de protocolo. Hay que elegir qué sacrificar: precisión, coste o garantías.

### R5 — Seguridad y robustez (poisoning, puertas traseras)
**El reto:** clientes maliciosos que envían actualizaciones diseñadas para degradar o manipular el modelo global.
**Por qué es complejo:** por construcción, el servidor no puede inspeccionar los datos para detectar el ataque (esa opacidad es la *feature* de privacidad); solo ve vectores de pesos. Distinguir una actualización maliciosa de una simplemente "rara" por datos no-IID legítimos (R1) es ambiguo: las defensas robustas (Krum, mediana, recorte) descartan outliers, pero con heterogeneidad real los clientes minoritarios legítimos *son* outliers — la defensa puede silenciar justo a quien aporta diversidad. Privacidad y robustez tiran en direcciones opuestas.

### R6 — Evaluación, validación y depuración
**El reto:** saber si el modelo global es bueno, para quién, y por qué falla cuando falla.
**Por qué es complejo:** no existe un conjunto de test centralizado y representativo de la federación; la evaluación debe hacerse también federada (cada cliente evalúa en local y se agregan métricas), lo que mezcla la incertidumbre del modelo con la heterogeneidad de los tests. Y al depurar, la herramienta básica del ML —mirar los ejemplos que el modelo falla— está prohibida por diseño.

### R7 — Equidad entre participantes
**El reto:** que el modelo global no funcione sistemáticamente peor para los clientes con menos datos o distribuciones minoritarias.
**Por qué es complejo:** FedAvg pondera por volumen de datos: los clientes grandes dominan la media. Reponderar a favor de los pequeños mejora la equidad pero puede empeorar la media global y abre la puerta a manipulación (declarar menos datos para pesar más). Definir siquiera qué métrica de equidad usar es una decisión normativa, no técnica.

### R8 — Incentivos y free-riding
**El reto:** que participar y contribuir datos de calidad compense.
**Por qué es complejo:** todos los participantes reciben el mismo modelo global, aporten mucho o nada: el incentivo racional es aprovecharse del esfuerzo ajeno (*free-riding*). Pagar por contribución exige valorar cada contribución (¿Shapley? — computacionalmente explosivo y dependiente de R6: hay que poder evaluar) sin ver los datos. Es un problema de teoría de juegos incrustado dentro de un problema de ML.

### R9 — Gobernanza, legal y coordinación
**El reto:** múltiples organizaciones independientes acordando esquema de datos, responsabilidades, propiedad del modelo y cumplimiento.
**Por qué es complejo:** ¿de quién es el modelo entrenado colectivamente? ¿Quién responde si discrimina o falla? ¿Cómo se ejerce el "derecho al olvido" sobre un modelo ya entrenado (*machine unlearning* federado: problema abierto)? Aquí la complejidad no es algorítmica sino institucional — y suele ser el cuello de botella real de los despliegues.

### R10 — El caso agravado: federar aprendizaje por refuerzo
**El reto:** todo lo anterior, aplicado a RL (y más aún a RL offline).
**Por qué es complejo:** el no-IID se convierte en *MDPs distintos* (entornos con dinámicas diferentes, no solo datos diferentes); las pérdidas de RL son no estacionarias, así que promediar políticas en fases distintas de aprendizaje es más frágil que en supervisado; los datasets locales fueron generados por políticas de comportamiento de calidad dispar; y evaluar sin entorno (OPE) hace que R6 sea doblemente difícil. Es la composición de los retos del FL con los del RL, multiplicándose en vez de sumarse.

## 4. Síntesis: la regla mental

| Supuesto del ML centralizado | Lo que pasa en FL | Reto que genera |
|---|---|---|
| Datos IID en un solo lugar | Distribuciones distintas en silos distintos | R1, R7 |
| Una máquina, un reloj | Muchos nodos dispares y poco fiables | R2, R3 |
| El entrenador ve los datos | Nadie ve los datos de otro | R4, R5, R6 |
| Un solo actor con un solo interés | Múltiples actores con intereses propios | R8, R9 |
| Dataset estático y etiquetado | Trayectorias generadas por políticas dispares (caso RL) | R10 |

**Conclusión:** el aprendizaje federado merece la pena exactamente cuando el valor de los datos inaccesibles (P3) supera el sobrecoste de complejidad (C1–C8). Si los datos se pueden centralizar legal y prácticamente, casi siempre conviene centralizar; el FL no es una optimización, es el precio a pagar por aprender de datos que no pueden moverse.

## 5. Retos del FL sobre diferentes paradigmas de aprendizaje

Esta pregunta toca el insight más profundo del tema: los retos del FL no son los mismos en cada paradigma porque dependen de la naturaleza de la señal de aprendizaje que se federa. Cuanto más débil y más entrelazada con los datos locales está esa señal, más se complica federar. Te lo muestro primero como panorama y luego desgrano cada rama.

**Supervisado federado** — el caso base. Aquí la señal de aprendizaje es la más fuerte posible: cada muestra trae su etiqueta exacta y la pérdida es estacionaria (el objetivo no cambia durante el entrenamiento). Por eso es el paradigma donde FedAvg funciona mejor y donde se concentra la madurez del campo. Sus retos son los "clásicos" del FL: el client drift por datos no-IID (cada cliente arrastra el modelo hacia su óptimo local), el sesgo de etiquetas por silo (un hospital pediátrico y uno geriátrico etiquetan distribuciones de clases distintas), y uno más sutil que suele olvidarse: la heterogeneidad de anotación — dos silos pueden etiquetar el mismo caso de forma distinta porque sus criterios humanos difieren, y eso introduce ruido de concepto que el servidor no puede detectar porque no ve los datos. Es el paradigma difícil "estándar": todo lo que viene después lo hereda y le suma algo.


No supervisado federado — el problema del alineamiento. Al quitar las etiquetas pierdes dos cosas a la vez. Primero, la verdad-terreno para validar: ¿cómo sabe el orquestador que el modelo global mejora si no hay accuracy que medir? Segundo, y más profundo, aparece el problema de alineamiento: las estructuras que cada cliente descubre localmente no tienen por qué corresponderse entre sí. En clustering federado, el "cluster 3" del cliente A y el "cluster 1" del cliente B pueden ser el mismo grupo con etiquetas permutadas — promediar sin resolver esa correspondencia destruye la estructura. Con autoencoders pasa lo análogo en continuo: dos clientes pueden aprender espacios latentes equivalentes pero rotados o permutados entre sí (las redes neuronales son invariantes a permutaciones de neuronas), y la media de dos representaciones desalineadas no es una representación, es ruido. Y los modelos generativos adversarios (GANs) añaden su propia inestabilidad de entrenamiento, que la agregación federada amplifica. Caso ilustrativo: en detección de anomalías federada, lo que es anómalo para un cliente puede ser lo normal de otro — ¿la anomalía se define localmente o globalmente? Esa pregunta no tiene respuesta técnica, es de diseño.


Auto-supervisado y semi-supervisado federados — la señal fabricada se empobrece. Aquí la señal se construye desde los propios datos, y eso la encadena a la calidad de los datos locales. El caso contrastivo (estilo SimCLR) lo muestra con claridad: estos métodos necesitan lotes grandes con negativos diversos (ejemplos distintos contra los que contrastar), pero cada cliente solo tiene su porción pequeña y sesgada del mundo — sus negativos son pobres, y compartir embeddings entre clientes para enriquecerlos rompería justo la privacidad que el FL protege. En semi-supervisado el problema es el sesgo de confirmación compuesto: el cliente genera pseudo-etiquetas con el modelo actual, entrena sobre ellas, y sus errores se autoamplifican; en centralizado esto ya es delicado, pero en federado el client drift hace que cada cliente amplifique sesgos distintos, que luego se promedian. Además aparece una asimetría estructural nueva: ¿quién tiene las pocas etiquetas? Los escenarios "etiquetas solo en el servidor" y "etiquetas solo en algunos clientes" requieren algoritmos diferentes. La paradoja del paradigma: es el más prometedor para FL cross-device (nadie etiqueta los datos de su móvil), y a la vez su mecánica interna es la que peor encaja con datos fragmentados.


Refuerzo federado — todos los anteriores, multiplicados. El RL hereda el no-IID pero lo lleva al extremo: los clientes no difieren solo en datos sino en MDPs completos — dinámicas, distribuciones de estados y hasta funciones de recompensa distintas — con lo que "la política global óptima" puede ni siquiera estar bien definida. La pérdida es no estacionaria (los objetivos TD cambian con la propia red), así que promediar políticas en fases distintas de aprendizaje es mucho más frágil que promediar clasificadores. Y luego están las subdivisiones que tú mismo señalaste, que cambian el perfil del reto. En online, el problema añadido es la exploración descoordinada (cada agente explora por su cuenta y la agregación mezcla descubrimientos incompatibles) y la seguridad de explorar en entornos reales. En la distinción on-policy vs off-policy hay una incompatibilidad estructural elegante: los métodos on-policy (PPO) exigen datos frescos de la política actual, pero la agregación federada cambia la política bajo los pies del cliente en cada ronda, invalidando ese supuesto — por eso los métodos off-policy, que toleran datos de otras políticas, son el encaje natural del FL. Y en offline se suma el reto distintivo que ya conoces de tu nicho: los datasets locales fueron generados por políticas de comportamiento de calidad dispar, y el pesimismo de cada cliente debe agregarse sin volverse ni excesivamente conservador ni peligrosamente optimista.


La regla que unifica todo el recorrido: a medida que bajas por el árbol, la señal de aprendizaje pasa de ser externa y exacta (etiqueta) a interna (estructura), luego fabricada (pretexto/pseudo-etiqueta) y finalmente generada por interacción (recompensa) — y cuanto más depende la señal de los datos locales del cliente, más daño hace la fragmentación federada. Por eso tu nicho está en la fila de cuatro cuadrados.
