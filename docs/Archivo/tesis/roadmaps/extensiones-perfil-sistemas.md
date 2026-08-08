---
title: "Roadmap — Extensiones para perfil de tesis de sistemas"
tags: [tesis, roadmap]
status: borrador
updated: 2026-08-08
---

# Roadmap — Extensiones para perfil de tesis de sistemas

> **Documento complementario al `roadmap.md` principal.** Recoge dos bloques que no estaban en la versión 2 del roadmap maestro y que cobran sentido cuando la tesis se orienta hacia **sistemas** (Pontus-X, Compute-to-Data, ingeniería de despliegue real) más que hacia teoría pura o aportación algorítmica genérica.
>
> **Cómo integrarlo**: la Fase 5.5 va entre la Fase 5 (FL clásico) y la Fase 6 (FRL como investigación). La sección 6.1 reemplaza/extiende el bloque *"Vías de contribución doctoral (orientativas)"* de la Fase 6 cuando el perfil de tesis es de sistemas.
>
> **Estado del lector al escribir esto**: Fase 0–1 (fundamentos). Este documento es brújula para tu yo-futuro de dentro de 9–12 meses, no acción inmediata. **No tocar Sutton & Barto por leer esto.**

---

## Fase 5.5 — Systems & Infraestructura para Federated ML sobre Dataspaces

### Por qué este bloque existe

Si tu tesis es de sistemas, no basta con entender FL y FRL como algoritmos. Tienes que dominar la pila de software y protocolos sobre la que se ejecutan en un dataspace real. Pontus-X / Gaia-X tiene piezas que **ningún curso de FL te va a enseñar**, y esa es precisamente tu ventaja competitiva frente a doctorandos de FRL puros: ellos saben más matemáticas que tú durante años; tú vas a saber operar la infraestructura que nadie del mundo académico opera realmente.

Sin este bloque, te ocurre lo siguiente: llegas a Fase 6 sabiendo FRL en teoría, montas un experimento en Flower simulado, escribes un paper, y descubres que tu contribución podría haberla hecho cualquier doctorando con un cluster universitario. El ángulo "dataspace real" se pierde si no lo cultivas explícitamente aquí.

### Objetivo

Ser **la persona** que entiende cómo se orquesta un job de FRL real sobre un dataspace, no solo sobre un cluster experimental con Flower simulado. Salir de esta fase con la capacidad de articular qué pieza de la pila Gaia-X/Pontus-X falta para hacer FRL viable a escala — esa pieza es candidata directa a contribución doctoral.

### Temario

**Compute-to-Data en profundidad**. Cómo se publica un dataset/asset en Ocean Protocol, cómo se publica un algoritmo, cómo se firma el resultado, qué controles tiene el proveedor de datos sobre lo que sale del entorno (logging, salidas, side-channels). Esto es importante porque en FRL **las propias actualizaciones del modelo son un canal por el que puede filtrarse información**: aquí es donde C2D se cruza con privacy ML. Hay que entender el modelo de amenaza completo: qué puede ver el proveedor de cómputo, qué puede ver el proveedor de datos, qué puede ver el consumidor del algoritmo, qué puede filtrarse en los outputs autorizados.

**Identidad y gobernanza en Gaia-X**. DIDs (Decentralized Identifiers), Verifiable Credentials, Self-Sovereign Identity. No para hacerte experto en SSI, sino para entender que cada cliente del FRL es una entidad identificable con políticas asociadas. Esto tiene implicaciones en *quién puede participar en una ronda de federación* y *cómo se audita post-hoc*. La especificación del Gaia-X Trust Framework es lectura obligada aunque sea densa.

**Orquestación de jobs federados sobre dataspace**. Flower es bueno para simular FL, pero en Pontus-X los clientes son entornos C2D independientes que no se hablan directamente. Hay un gap entre "Flower simulado en mi laptop" y "rondas de FL ejecutándose como jobs C2D coordinados". Cerrar ese gap es ingeniería real y probablemente **no tiene solución pública lista** hoy. Aquí entran patrones de coordinación: ¿quién es el servidor de agregación, dentro o fuera del dataspace? ¿Cómo se sincronizan rondas? ¿Qué pasa si un cliente cae a mitad de una ronda?

**Economía del dataspace**. Tokens, precios, incentivos. En Pontus-X cada job cuesta. Esto introduce una dimensión que el FL académico ignora: **comunicación-eficiente no es un nice-to-have, es presupuesto literal**. Y abre la puerta a mecanismos de incentivos (¿cómo se recompensa al cliente que aporta datos más útiles al modelo global? ¿cómo se evita que un cliente saboteador desperdicie tokens del consumidor?). Esto cruza con teoría de juegos aplicada — sub-campo con literatura abundante en FL pero apenas explorada en el contexto específico de dataspaces europeos.

**Versionado y reproducibilidad federada**. Cómo versionar modelos cuando los datos no se mueven y los clientes pueden entrar/salir entre rondas. MLflow + DVC clásicos no encajan directamente: asumen que tienes acceso al dataset. Hay diseño abierto aquí: ¿se versiona la política de selección de clientes? ¿la composición de la federación en cada ronda? ¿el estado del agregador?

**Monitorización de drift entre clientes**. Cuando no puedes mirar los datos, ¿cómo detectas que un cliente está enviando actualizaciones raras? Puede ser drift legítimo (su entorno ha cambiado), un ataque (poisoning, backdoor), o simplemente un bug en su pipeline local. Distinguirlos sin violar privacidad es problema abierto.

### Recursos

Esta fase es la que peor cubierta está en cursos. Vas a vivir más en documentación oficial y papers sueltos que en clases estructuradas.

- **Documentación oficial de Pontus-X** ([docs.pontus-x.eu](https://docs.pontus-x.eu/)) y **deltaDAO tutorials**. Especialmente las partes de operaciones, no solo las de "hello world". Lee la documentación de Ocean Protocol también, porque Pontus-X se apoya en ella.
- **Especificaciones de Gaia-X Trust Framework** ([gaia-x.eu](https://gaia-x.eu/)). Densas, pero hay que pasar por ahí al menos una vez. Concéntrate en las secciones de compliance, identity y policy reasoning.
- **Literatura sobre MLOps for Federated Learning**. Sub-campo todavía pequeño y por tanto con hueco. Búsquedas relevantes en arXiv: "FedOps", "federated MLOps", "production federated learning". Empresas como Owkin y NVIDIA Clara han publicado experiencia operacional aplicable.
- **Papers sobre incentive mechanisms en FL**. Búsquedas: "incentive mechanism federated learning", "Shapley value federated", "game-theoretic federated learning". Hay una rama entera de teoría de juegos aplicada, con surveys recientes (2023–2025).
- **Flower advanced tutorials**, especialmente los de deployment real (no simulación) y los de integración con plataformas cloud. Te dan los patrones que luego adaptarás a C2D.
- **MLSys conference proceedings** ([mlsys.org](https://mlsys.org/)). El venue de sistemas para ML. Hojea las últimas 2–3 ediciones buscando papers sobre FL deployment, model serving distribuido, privacy-preserving infrastructure.

### Ejecución

El proyecto-puente que está en Fase 5 ("desplegar pipeline federado en Pontus-X testnet") se convierte aquí en algo más serio. Despliegas un FedAvg simple sobre **tres** nodos C2D reales, no simulados, con identidades verificables, y mides extremo a extremo: tiempo de ronda, coste en tokens, throughput de modelo, fiabilidad ante caída de un nodo, comportamiento ante un cliente que se une o sale a mitad del entrenamiento.

Este experimento es probablemente tu **primer paper publicable**, antes incluso de tocar RL federado. El género es "experience paper" o "measurement study": narras qué encontraste al desplegar algo conocido (FedAvg) en infraestructura no-académica (Pontus-X). Venues posibles: workshops de MLSys, workshops sobre FL en NeurIPS/ICML, conferencias europeas sobre dataspaces (las hay, busca IDSA y Gaia-X Hackathon proceedings).

**Mini-proyecto adicional opcional**: una librería pequeña que envuelva Flower para que sus clientes puedan ejecutarse como algoritmos C2D en Pontus-X. Aunque sea Hello World funcional, publicarla en GitHub con un README decente es **señalización académica fuerte** — los doctorados de sistemas se ganan tanto por contribuciones de software como por papers.

### Checkpoint Fase 5.5

- Has desplegado FL real (no simulado) sobre Pontus-X con ≥3 nodos.
- Sabes explicar el modelo de amenaza completo de C2D + FL: qué actor puede ver qué, y por dónde puede filtrarse información a través de actualizaciones de modelo.
- Tienes números reales de coste/latencia/fiabilidad que **ningún paper académico de FL tiene**. Estos números son tu activo.
- Puedes articular en una página qué pieza de la pila Gaia-X/Pontus-X falta para hacer FRL viable a escala. Esa página es el embrión de tu propuesta de tesis.
- Tienes una librería o pipeline en GitHub que cualquiera podría usar para reproducir tu setup.

---

## Sección 6.1 — Vías de contribución doctoral para perfil de sistemas

> Reemplaza/extiende la subsección *"Vías de contribución doctoral (orientativas)"* de la Fase 6 del roadmap maestro cuando la tesis se orienta hacia sistemas.

La lista original en Fase 6 tiene cinco direcciones (Offline FRL non-IID, FRL con privacy, Personalized FRL, FRL comunicación-eficiente, Sim2Real federado). De ellas, **cuatro son algorítmicas o teóricas**, y solo "comunicación-eficiente FRL" tira ligeramente hacia sistemas. Para un perfil de tesis de sistemas, las direcciones que de verdad encajan son otras.

### Las cinco direcciones de sistemas

**1. Una arquitectura/framework de referencia para FRL sobre dataspaces.**

Hoy no existe un equivalente de Flower para FRL específicamente sobre Pontus-X o sobre dataspaces Gaia-X en general. Construirlo, evaluarlo y publicarlo como software + paper de sistemas es una tesis perfectamente válida. La contribución es la **integración**: tomar piezas conocidas (algoritmos de FRL, C2D, identidades verificables, agregación segura) y articularlas en un sistema operable. Venues naturales: MLSys, EuroSys, SoCC, NSDI (si hay componente de red), o venues europeos sobre dataspaces.

El argumento defensivo: "X y Y existen por separado, pero nadie ha demostrado que se puedan combinar a esta escala, con estas garantías, y a este coste". Si tu sistema funciona y los números son buenos, el paper se escribe solo.

**2. Estudio empírico a gran escala de FRL en condiciones realistas de dataspace.**

Comunicación cara, heterogeneidad de hardware entre nodos, identidades verificables, fallos parciales, clientes que entran y salen entre rondas. Comparar cómo se degradan QAvg/PAvg/CQL-federado en estas condiciones frente a las del laboratorio simulado. Esto es paper de "measurement study" y se publica muy bien porque **no requiere proponer nada nuevo**: tu contribución es mostrar que lo existente no funciona como la gente cree que funciona, y por qué.

Este tipo de papers son particularmente influyentes a largo plazo porque cambian cómo el resto del campo evalúa sus contribuciones futuras. Henderson et al. 2017 ("Deep RL that Matters") es el ejemplo canónico — un measurement paper que sacudió la disciplina.

**3. Mecanismos de incentivos y gobernanza para FRL en dataspaces.**

Cómo se recompensa a clientes que aportan datos valiosos sin revelarlos. Cómo se detecta y excluye clientes maliciosos manteniendo la privacidad. Cómo se diseña la economía de tokens para que la federación sea sostenible a largo plazo. Esta dirección cruza sistemas con teoría de juegos aplicada y es **muy poco explorada** en el contexto específico de dataspaces europeos — la mayoría de la literatura FL+game theory asume settings genéricos sin la capa de gobernanza Gaia-X.

Ángulo defensivo: si tu sistema introduce un mecanismo de incentivos novedoso y demuestras (empíricamente o con argumento teórico) que tiene propiedades deseables (truthfulness, fairness, robustez ante colusión), tienes paper en un venue de sistemas o de econ-CS.

**4. Auditabilidad y compliance bajo AI Act / Data Governance Act para FRL.**

Cómo se construye un sistema de FRL que pueda demostrar a un regulador europeo que cumple. Esto suena administrativo y no lo es: hay diseño técnico real. Logging selectivo que preserva privacidad. Pruebas criptográficas de ejecución correcta (zk-SNARKs aplicados a FL, área emergente). Trazabilidad de decisiones del modelo agregado de vuelta a los clientes que contribuyeron. Explicabilidad bajo restricciones de federación.

Esto es **único de Europa** y por tanto un ángulo defendible que ningún doctorando americano va a explorar. El AI Act entró en vigor en 2024 y su aplicación se despliega entre 2025 y 2027 — tu tesis cae justo en la ventana donde el sector industrial va a necesitar respuestas técnicas a estas preguntas y no las tiene.

**5. Offline FRL con privacidad verificable, implementada y evaluada.**

Lo que en el roadmap original era "1+2 combinado" (Offline FRL + FRL con privacy), pero ahora con sesgo de sistemas: tu contribución no sería el algoritmo nuevo sino **una implementación de referencia auditada, evaluada en dataspace real, con medidas de privacidad verificables**. Tomas CQL o IQL, lo federas con FedAvg, le metes DP-SGD por encima, lo despliegas en Pontus-X, y mides todo end-to-end. La novedad es la integración funcional, no el algoritmo.

Esta opción es la más "segura" en el sentido de que combina cosas que ya sabes que funcionan y la contribución es ingeniería honesta. Buena candidata para el primer paper de tesis.

### Combinación recomendada

De estas cinco, mi recomendación para tu perfil declarado: **dirección 1 + dirección 4 como combinación principal**, con la dirección 2 como subproducto natural (cuando construyes un framework de sistemas, casi inevitablemente generas el measurement study que justifica por qué hacía falta).

El relato de tesis quedaría: *"Un framework de Federated Reinforcement Learning para dataspaces europeos con compliance integrada bajo AI Act / DGA: diseño, implementación, evaluación empírica sobre Pontus-X, y mecanismos de auditabilidad verificable"*. Tres capítulos = tres papers. Defensa sólida. Y vendible más allá del mundo académico: deltaDAO, Orange, T-Systems, y el ecosistema de proveedores Gaia-X son contratantes potenciales directos.

### Combinaciones a evitar

Mezclar dirección 3 (incentivos) con dirección 4 (compliance) se vuelve administrativo rápidamente y diluye la contribución técnica. Mezclar dirección 1 (framework) con dirección 5 (Offline FRL con privacy) es factible pero te puede llevar a abarcar demasiado: ya construyes el framework, ya implementas Offline FRL, ya integras DP — son tres compromisos de ingeniería y solo tienes 3–4 años. Mejor priorizar.

### Conexión con el contexto regulatorio europeo

Independientemente de qué dirección elijas, hay tres normativas que conviene tener leídas (los resúmenes ejecutivos, no el texto legal completo) antes de cerrar la propuesta de tesis:

- **AI Act** (Reglamento UE 2024/1689). Especialmente las definiciones de sistemas de alto riesgo y los requisitos de documentación técnica.
- **Data Governance Act** (Reglamento UE 2022/868). El marco general de dataspaces como infraestructura de mercado.
- **Data Act** (Reglamento UE 2023/2854). Derechos de acceso y portabilidad de datos generados por dispositivos conectados.

No tienes que volverte abogado, pero saber qué obliga cada uno te permite articular **por qué tu sistema técnico responde a una necesidad regulatoria real**, que es uno de los argumentos más fuertes que puede tener una tesis aplicada en Europa hoy.

---

## Notas para el yo-futuro

Cuando llegues al final de la Fase 5 y empieces a leer este documento en serio:

- **Releer el roadmap maestro completo** antes de tocar nada aquí. Es posible que entre que se escribió esto (Fase 0–1) y que llegues a aplicarlo (Fase 5+) hayas cambiado de opinión sobre el perfil de tesis. Si has virado a teoría o a empirismo algorítmico, este documento sigue siendo útil pero deja de ser la columna vertebral.
- **Hablar con tu director** antes de cerrar la combinación de direcciones. Lo que aquí pongo como recomendación (1+4 con 2 como subproducto) puede no encajar con sus líneas activas de financiación, sus colaboradores industriales, o el perfil del tribunal previsible.
- **Verificar el estado del arte de cada dirección** en el momento que las elijas. La sección de incentivos y la de auditabilidad sobre todo: son áreas que se mueven rápido y donde un paper aparecido en 2027 puede cerrar parte del hueco que ahora veo abierto.
- **No abandonar la Fase 5.5** aunque parezca menos brillante que ponerte ya con FRL puro. La gente que se salta sistemas y va directa a algoritmos es la mayoría; tu ventaja competitiva está precisamente en no ser esa mayoría.
