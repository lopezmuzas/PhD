---
title: "Base de Conocimiento: P2P, Espacios de Datos Federados y Seguridad C2D"
tags: [referencia, data-spaces]
status: borrador
updated: 2026-08-08
---

# Base de Conocimiento: P2P, Espacios de Datos Federados y Seguridad C2D

> Generado a partir de análisis técnico profundo. Apto para recuperar contexto en conversaciones LLM posteriores.
> Fecha de generación: Abril 2026

---

## Índice

1. [Redes P2P: arquitectura técnica](#1-redes-p2p-arquitectura-técnica)
2. [Pontus-X: espacio de datos federado](#2-pontus-x-espacio-de-datos-federado)
3. [Analogía estructural P2P ↔ Pontus-X](#3-analogía-estructural-p2p--pontus-x)
4. [Escenario concreto: consumir 10 datasets de competidores](#4-escenario-concreto-consumir-10-datasets-de-competidores)
5. [Aprendizajes de P2P aplicables a Pontus-X](#5-aprendizajes-de-p2p-aplicables-a-pontus-x)
6. [Protocolo Tit-for-Tat aplicado a datos](#6-protocolo-tit-for-tat-aplicado-a-datos)
7. [Seguridad en Compute-to-Data: amenazas y defensas](#7-seguridad-en-compute-to-data-amenazas-y-defensas)
8. [Analogía Guardrail LLM ↔ Seguridad C2D](#8-analogía-guardrail-llm--seguridad-c2d)
9. [Conclusiones y razonamientos clave](#9-conclusiones-y-razonamientos-clave)
10. [Glosario técnico](#10-glosario-técnico)

---

## 1. Redes P2P: Arquitectura Técnica

### 1.1 Mecanismos fundamentales

Una red P2P pura (referencia: BitTorrent) opera sobre cinco mecanismos ortogonales que pueden analizarse por separado:

#### Descubrimiento de peers
- **DHT (Distributed Hash Table)**: tabla hash distribuida donde cada nodo conoce a sus "vecinos" en el espacio de claves. No existe servidor central. La ubicación de los datos se distribuye entre todos los nodos.
- **Tracker**: servidor de arranque opcional que conoce qué peers tienen qué contenido. En BitTorrent moderno es opcional gracias a DHT.
- **Infohash**: identificador único de un contenido (SHA-1/SHA-256 del torrent). Quien tiene el infohash puede localizar peers.

#### Transferencia de datos
- Los archivos se fragmentan en **chunks** (piezas) de tamaño fijo.
- Un cliente puede descargar **piezas distintas de peers distintos en paralelo** — esto es el núcleo del escalado P2P.
- El mismo cliente puede **servir mientras descarga** (seeding simultáneo).
- Resultado: más demanda → más seeders → más ancho de banda disponible. El sistema escala de forma contraintuitiva.

#### Verificación de integridad
- Cada chunk tiene un **hash criptográfico** (SHA-256, Merkle trees en BitTorrent v2).
- El cliente verifica cada pieza recibida **independientemente del origen**.
- Principio clave: **no confías en el peer, confías en el hash**. Un peer malicioso que envía datos corruptos es detectado automáticamente.
- **Merkle trees**: permiten verificar cualquier subconjunto de piezas sin descargar el árbol completo, lo que habilita verificación eficiente en sistemas grandes.

#### Incentivos (Tit-for-Tat)
- **Tit-for-tat clásico**: compartes ancho de banda con quien te comparte. Los *leechers* (consumidores puros) quedan penalizados mediante **choking** (corte de conexión).
- **Optimistic unchoking**: cada ~30 segundos, cada peer abre una conexión a un peer desconocido al azar. Sirve para explorar nuevos buenos colaboradores sin información previa. Es el mecanismo de **exploración** dentro del sistema de incentivos.
- Sin árbitro central: el protocolo alinea incentivos sin intervención de terceros.

#### Gobernanza del protocolo
- Especificaciones abiertas publicadas como **BEP (BitTorrent Enhancement Proposals)**.
- Sin cumplimiento normativo integrado (privacy, compliance, monetización).
- Cualquiera puede participar si tiene el infohash.

### 1.2 Por qué P2P escala y los sistemas centralizados no

El insight central de P2P es que el **cuello de botella en sistemas centralizados es el servidor origen**. En P2P, cada nuevo consumidor que descarga también contribuye capacidad de upload. La red se auto-refuerza: más popularidad = más recursos disponibles. En sistemas centralizados: más popularidad = más carga en el origen.

---

## 2. Pontus-X: Espacio de Datos Federado

### 2.1 Identidad y posición en el ecosistema

- **Pontus-X** es el primer y mayor ecosistema X público europeo disponible. Es una **implementación de referencia open-source** de los principios Gaia-X.
- Es el segundo Lighthouse Data Space oficial de **Gaia-X**.
- Actúa como "ecosystem of ecosystems": proporciona el framework e interoperabilidad para múltiples proyectos Gaia-X (ACCURATE, COOPERANTS, EuProGigant, Flex4Res, Gaia-X4FutureMobility, etc.).
- Alcance: empresas e instituciones públicas pueden **consumir, ofrecer y monetizar** datos, software e infraestructura como servicio.

### 2.2 Stack tecnológico

```
┌─────────────────────────────────────────────────┐
│              Gaia-X Trust Framework             │  ← Governance / compliance
├─────────────────────────────────────────────────┤
│         Ocean Protocol (Enterprise)             │  ← Marketplace de datos
├─────────────────────────────────────────────────┤
│    DLT (Distributed Ledger / Blockchain EVM)    │  ← Registro inmutable
├─────────────────────────────────────────────────┤
│    OASIS / Smart Contracts (ERC-20 datatokens)  │  ← Control de acceso
├─────────────────────────────────────────────────┤
│         Compute-to-Data (C2D)                   │  ← Privacidad del dato
├─────────────────────────────────────────────────┤
│    Gaia-X Digital Clearing Houses (GXDCH)       │  ← Verificación credenciales
└─────────────────────────────────────────────────┘
```

### 2.3 Principios fundamentales

**Soberanía del dato (Data Sovereignty)**
- El dato **nunca abandona la infraestructura de su propietario**.
- El propietario siempre controla quién accede y cómo se usa su dato.
- Diferencia radical con plataformas centralizadas donde el dato se cede al operador.

**Compute-to-Data (C2D)**
- En lugar de que el dato viaje al consumidor, **el algoritmo viaja al dato**.
- El consumidor recibe solo el resultado (output del cómputo), no el dato raw.
- Habilita: privacidad, federated learning, computación sobre datos sensibles sin exponerlos.
- Casos de uso: entrenamiento de modelos sobre datos distribuidos, análisis agregado cross-empresa, cálculo de Product Carbon Footprint sobre datos IP-sensibles.

**Data Tokens (ERC-20)**
- Cada dataset tiene su propio **token ERC-20 único** que actúa como clave de acceso.
- Comprar el token = comprar el derecho de acceso o cómputo sobre ese dataset.
- Liquidación instantánea on-chain. Modelo pay-per-use.

**DDO (Decentralized Data Object)**
- Metadatos de cada asset publicados on-chain (descripción, condiciones de uso, precio, endpoint de acceso).
- El dato está en la infraestructura del proveedor. Los metadatos (DDO) están en la blockchain.
- Permite descubrimiento descentralizado sin mover el dato.

### 2.4 Cumplimiento regulatorio integrado

Pontus-X alinea con:
- **GDPR**: protección de datos personales
- **Data Act (EU 2023)**: derecho de acceso y portabilidad
- **Data Governance Act**: reutilización de datos del sector público
- **AI Act**: trazabilidad de datos para sistemas de IA
- **Gaia-X Trust Framework**: identidad verificable, self-descriptions, GXDCH

### 2.5 Diferencia Gaia-X vs Pontus-X

| | Gaia-X | Pontus-X |
|---|---|---|
| Rol | Framework / estándar | Implementación de referencia |
| Proporciona | Reglas, arquitectura, gobernanza | Software, herramientas, stack técnico |
| Analogía | Plano arquitectónico | Edificio construido según el plano |

---

## 3. Analogía Estructural P2P ↔ Pontus-X

Mapa capa a capa de equivalencias funcionales:

| Capa | P2P (BitTorrent) | Pontus-X |
|---|---|---|
| **Descubrimiento** | DHT / Tracker | Catálogo on-chain (DDO) |
| **Identificador de recurso** | Infohash (SHA-256) | DDO address + data token address |
| **Autenticación / acceso** | Ninguna (quien tiene el infohash accede) | Data token ERC-20 + credencial Gaia-X (SSI) |
| **Transferencia** | Chunks del dato → consumidor | Algoritmo → dato (C2D). Solo sale el resultado |
| **Verificación integridad** | Hash por chunk (Merkle tree) | DLT + smart contract + remote attestation (TEE) |
| **Incentivos** | Tit-for-tat / choking | Data token + liquidación ERC-20 |
| **Gobernanza** | Protocolo abierto (BEP), sin compliance | GDPR + Data Act + Gaia-X Trust Framework |
| **Árbitro** | Ninguno (matemáticas + protocolo) | TEE + smart contract (sin tercero centralizado) |

### 3.1 La diferencia más importante

> **En P2P el dato viaja al consumidor.**
> **En Pontus-X el consumidor (su algoritmo) viaja al dato.**

Esta inversión es el núcleo de la soberanía del dato. En P2P cualquier participante con el infohash puede descargar el contenido completo. En Pontus-X el dato nunca abandona la infraestructura de quien lo posee.

### 3.2 Qué gana Pontus-X vs P2P

- Control de acceso granular (quién, cuándo, para qué uso)
- Monetización integrada
- Cumplimiento regulatorio
- Soberanía real del dato

### 3.3 Qué pierde vs P2P

- Velocidad de acceso (latencia adicional por negociación de tokens y C2D)
- Simplicidad
- Escalado automático (en P2P más usuarios = más recursos; en Pontus-X el nodo del proveedor sigue siendo el cuello de botella)

---

## 4. Escenario Concreto: Consumir 10 Datasets de Competidores

### 4.1 Caso de uso motivador

10 fabricantes del mismo sector (ej: fabricantes de componentes electrónicos) quieren entrenar un modelo de predicción de fallos usando datos de todos, sin que ninguno exponga sus datos al resto. En P2P esto sería trivial pero sin privacidad. En Pontus-X es posible con privacidad pero más complejo.

### 4.2 Flujo técnico paso a paso

```
1. DESCUBRIMIENTO
   └─ Consulta el catálogo on-chain
   └─ Localiza los 10 DDOs (uno por empresa)
   └─ Verifica condiciones de uso de cada uno

2. NEGOCIACIÓN DE ACCESO (×10 en paralelo)
   └─ Compra data token ERC-20 por cada dataset
   └─ Smart contract verifica credenciales Gaia-X del consumidor
   └─ Acceso autorizado por tiempo/uso limitado

3. EJECUCIÓN C2D (×10 en paralelo)
   └─ Consumidor envía su algoritmo a cada nodo
   └─ Cada nodo ejecuta en TEE aislado
   └─ Cada nodo devuelve gradientes / resultado parcial

4. AGREGACIÓN
   └─ Consumidor agrega los 10 resultados parciales
   └─ Obtiene modelo final entrenado sobre los 10 datasets
   └─ Ninguna empresa vio datos de las otras
   └─ Ninguna empresa vio el algoritmo del consumidor en claro
```

### 4.3 Problema actual: fricción vs paralelismo

Hoy (2026) el flujo es mayormente **secuencial**. La adquisición de 10 tokens y la ejecución C2D en 10 nodos implica 10 transacciones blockchain separadas + 10 jobs C2D independientes. El aprendizaje de P2P es que esto debería ser paralelo y orquestado automáticamente.

---

## 5. Aprendizajes de P2P Aplicables a Pontus-X

### 5.1 Paralelismo y chunking de computación

**Patrón P2P**: BitTorrent descarga piezas distintas de peers distintos simultáneamente.

**Aplicación a Pontus-X**: Descomponer la tarea de ML en sub-tareas que puedan ejecutarse en paralelo sobre los 10 nodos, recoger gradientes parciales como si fueran piezas del torrent. Esto es exactamente **Federated Learning** — ya soportado como caso de uso en Pontus-X pero no orquestado de forma nativa como lo haría un cliente BitTorrent.

**Brecha actual**: No existe un "cliente de datos" que gestione automáticamente la adquisición paralela de N tokens y la orquestación de N jobs C2D como BitTorrent gestiona N chunks.

### 5.2 Verificación por hash de resultados (Result Attestation)

**Patrón P2P**: Confiías en el hash, no en el peer. Verificación automática de cada chunk.

**Aplicación a Pontus-X**: Falta un mecanismo estándar de *result attestation*: ¿cómo verifica el consumidor que el C2D de la empresa A realmente ejecutó su algoritmo sin modificarlo y sobre los datos correctos?

**Solución propuesta**: Aplicar Merkle proofs sobre los resultados intermedios. El TEE firma criptográficamente tanto el hash del algoritmo ejecutado como el hash del dataset usado, y el resultado incluye esta prueba verificable.

### 5.3 Índices de disponibilidad dinámica

**Patrón P2P**: El cliente BitTorrent sabe en tiempo real qué chunks están disponibles en qué peers (availability bitfield). Si un peer cae, lo sabe inmediatamente y busca el chunk en otro.

**Aplicación a Pontus-X**: Los DDOs son estáticos. No hay SLA dinámico en el catálogo. Si el nodo C2D de la empresa 3 está caído, el consumidor lo descubre tarde (tras intentar ejecutar el job). 

**Mejora propuesta**: Publicar disponibilidad como **metadata viva** en el catálogo — uptime, latencia media, capacidad de cómputo disponible. Reduciría drásticamente la fricción operativa de consumir múltiples datasets.

### 5.4 Supernodos para federación semántica de metadatos

**Patrón P2P**: En redes grandes existen nodos con más capacidad que actúan como índices mejorados (superpeer architecture en Gnutella/KaZaA).

**Aplicación a Pontus-X**: Un **supernodo de catálogo sectorial** (uno por industria) que agregue y pre-indexe semánticamente los datasets disponibles aceleraría enormemente el descubrimiento, sin centralizar los datos.

**Estado actual**: Pontus-X actúa como "ecosystem of ecosystems" pero la interoperabilidad semántica entre sub-espacios es una brecha activa en 2026.

### 5.5 Alineación de incentivos multi-parte

**Patrón P2P**: Optimistic unchoking descubre nuevos buenos colaboradores de forma controlada y periódica.

**Aplicación a Pontus-X**: Protocolo de contribución gradual entre competidores del mismo sector (ver sección 6).

---

## 6. Protocolo Tit-for-Tat Aplicado a Datos

### 6.1 Problema que resuelve

En un pool de 10 empresas competidoras que quieren beneficiarse de un modelo agregado, el incentivo natural de cada empresa es **consumir sin contribuir** (free-rider). P2P resolvió este problema en transferencia de archivos; se puede trasladar a contribución de datos.

### 6.2 Diseño del protocolo

#### Fase 1: Contribution Scoring

Cada empresa contribuye su dataset al pool C2D. Un **motor de scoring on-chain** (smart contract) calcula un `contribution_score` para cada participante basado en:

```
contribution_score = f(
  volumen,      // número de registros
  diversidad,   // cobertura de casos edge, distribución de features
  frescura,     // qué tan reciente es el dato
  consistencia  // ausencia de duplicados, coherencia de schema
)
```

**Importante**: el scoring se calcula **dentro del entorno C2D**, nunca en abierto. El orquestador sabe que "empresa A tiene score 0.87" pero no *por qué* exactamente — sin acceso a los datos raw.

#### Fase 2: Tiers de acceso al modelo agregado

| Tier | Condición | Acceso |
|---|---|---|
| **Gold** | Score alto | Modelo completo + fine-tune sobre datos propios |
| **Silver** | Score medio | Modelo base sin fine-tune |
| **Choked** | Score bajo | Solo métricas globales agregadas |

#### Fase 3: Optimistic Unchoking (exploración)

Cada N épocas de entrenamiento (ventana periódica):
- Una empresa en estado *Choked* recibe acceso temporal a tier Silver.
- Si en la siguiente ventana su score mejora → asciende de tier.
- Si no mejora → vuelve a Choked.

Este mecanismo evita que empresas que empiezan con poco dato queden excluidas permanentemente. Es la aplicación directa del *optimistic unchoking* de BitTorrent.

#### Fase 4: Re-evaluación continua on-chain

```
Nueva contribución → Scoring update (smart contract) → Tier update → 
Modelo re-entrenado → [ciclo] → Nueva contribución
```

El ciclo es continuo. Los tiers no son estáticos: una empresa puede ascender o descender en cada ventana de evaluación según su comportamiento.

### 6.3 Por qué funciona entre competidores

El truco es que **ningún competidor ve los datos de los demás**. El scoring se calcula en C2D. El modelo se entrena en federated learning. Los competidores solo ven sus propias métricas de score y el modelo agregado resultante — nunca los datos del rival.

### 6.4 Analogía completa con BitTorrent

| BitTorrent | Protocolo datos |
|---|---|
| Ancho de banda compartido | Calidad de datos aportada |
| Choking (cortar conexión) | Degradar tier de acceso |
| Unchoked (conexión activa) | Tier Gold/Silver |
| Optimistic unchoking | Ventana de prueba periódica |
| Peer evaluation cada 30s | Re-evaluación cada N épocas |
| Infohash (identifica recurso) | DDO + data token |

---

## 7. Seguridad en Compute-to-Data: Amenazas y Defensas

### 7.1 Las dos amenazas simétricas

El sistema C2D tiene dos atacantes potenciales con objetivos opuestos:

```
┌─────────────────────┐        ┌─────────────────────┐
│    CONSUMIDOR       │        │    PROVEEDOR        │
│                     │        │                     │
│  Envía algoritmo    │◄──────►│  Ejecuta algoritmo  │
│  malicioso para     │        │  en su infraestr.   │
│  exfiltrar datos    │        │  y copia la lógica  │
│  raw del proveedor  │        │  del algoritmo      │
└─────────────────────┘        └─────────────────────┘
     Amenaza A: robo datos       Amenaza B: robo algoritmo
```

Ambas amenazas coexisten y el sistema debe defenderse simultáneamente de las dos.

### 7.2 Defensas lado proveedor (proteger datos)

#### 1. Trusted Execution Environment (TEE)
- El algoritmo del consumidor corre en un **enclave seguro** (Intel SGX, AMD SEV).
- La memoria del enclave está **cifrada y aislada**: ni el sistema operativo del host puede leerla.
- El proveedor no puede copiar el algoritmo porque **nunca está en claro en su sistema**.
- Tecnologías: Intel SGX (Software Guard Extensions), AMD SEV (Secure Encrypted Virtualization), ARM TrustZone.

#### 2. Aislamiento de red total
- El contenedor de cómputo **no tiene salida a internet** durante la ejecución.
- Solo puede recibir inputs (el algoritmo cifrado y los datos autorizados) y devolver el output.
- Un algoritmo malicioso que intenta enviar datos a un servidor externo encontrará la red bloqueada.

#### 3. Result Vetting (filtrado de output)
- El resultado **se analiza antes de salir** del entorno C2D.
- Verificaciones: ¿contiene datos raw? ¿la entropía del output es consistente con un resultado legítimo? ¿hay patrones que sugieren datos raw codificados en decimales u otras covert channels?
- Costoso computacionalmente pero crítico.
- Técnica de ataque que defiende: **covert channel en el output** (codificar datos en los decimales de los pesos del modelo, en patrones de bits del resultado, etc.).

#### 4. Privacidad Diferencial (ε-Differential Privacy)
- Incluso con TEE + aislamiento, los resultados de ML pueden filtrar información mediante **ataques de inferencia de membresía** (membership inference attacks).
- Si se entrena un modelo sobre un dataset y se hacen suficientes queries al modelo resultante, se puede inferir si cierto registro estaba en el training set.
- La privacidad diferencial añade **ruido matemáticamente calibrado** (parámetro ε) que hace matemáticamente imposible ese tipo de inferencia.
- Trade-off: más privacidad (ε pequeño) = más ruido = menor precisión del modelo.
- Se aplica a los gradientes durante el entrenamiento (DP-SGD) o al resultado final.

#### 5. Auditoría del algoritmo (pre-ejecución)
- El proveedor inspecciona el código antes de ejecutar.
- **Allowlist de operaciones**: solo se permiten operaciones de ML/estadística reconocidas. Las llamadas a sistema, operaciones de red, o escritura a disco fuera del directorio de output están bloqueadas.
- En sistemas avanzados, el algoritmo se compila a un IR (Intermediate Representation) verificable antes de entrar al TEE.

### 7.3 Defensas lado consumidor (proteger algoritmo)

#### 1. Ofuscación + cifrado del algoritmo
- El modelo o algoritmo viaja **cifrado** hasta el TEE del proveedor.
- Se descifra **solo dentro del enclave**, con una clave que solo el TEE puede obtener.
- El proveedor nunca ve el código fuente o los pesos en claro.

#### 2. Remote Attestation
- El TEE **firma criptográficamente** una prueba que incluye:
  - El hash exacto del código ejecutado
  - El hash de los datos sobre los que operó
  - El hash del resultado producido
- El consumidor verifica esta firma antes de aceptar el resultado.
- Garantía: el proveedor ejecutó **exactamente** el código enviado, sin modificaciones.
- Estándar: Intel DCAP (Data Center Attestation Primitives), IETF RATS (Remote Attestation Procedures).

#### 3. Computación Homomórfica (FHE - Fully Homomorphic Encryption)
- El consumidor envía su algoritmo cifrado de forma que el proveedor puede **ejecutarlo sobre datos cifrados sin ver nunca el algoritmo en claro ni los datos en claro**.
- Solo el consumidor puede descifrar el resultado.
- Estado en 2026: FHE es viable para operaciones simples pero demasiado lento para ML complejo (~100-1000x overhead). Activamente investigado.
- Alternativa parcial: **Secure Multi-Party Computation (SMPC)** — más eficiente para casos específicos.

#### 4. Model Watermarking
- Durante el entrenamiento/diseño, se inserta un **patrón oculto (backdoor)** en los pesos del modelo.
- El patrón: un subconjunto específico de inputs produce una salida característica conocida solo por el propietario.
- Si el proveedor copia el modelo y lo despliega, el propietario puede detectarlo consultando esos inputs específicos.
- Analogía clásica: poner una calle falsa en un mapa para detectar copias cartográficas.
- Técnicas: backdoor watermarking, radioactive data, exponential learning rate scheduling para watermark.

#### 5. Audit Log on-chain
- Cada ejecución queda registrada en la blockchain: hash del algoritmo, timestamp, hash del resultado, identidad del proveedor.
- Inmutable y verificable por cualquier parte.
- Si hay disputa sobre si se ejecutó el código correcto, el log on-chain es evidencia.

### 7.4 El TEE como punto de encuentro

El insight arquitectónico central:

> Ambas amenazas (robo de datos, robo de algoritmo) se resuelven **en el mismo lugar**: el TEE.
>
> - El proveedor no puede copiar el algoritmo porque **el código nunca está en claro en su sistema**.
> - El consumidor confía en esto porque recibe una **remote attestation firmada criptográficamente**.
> - **Ni el proveedor ve el algoritmo real, ni el consumidor ve los datos raw.**

### 7.5 La debilidad que queda: side-channel attacks en TEE

- Vulnerabilidades como Spectre, Meltdown aplicadas a SGX han demostrado que los enclaves no son perfectos.
- Un adversario con acceso físico o privilegios de kernel puede intentar **side-channel attacks** (timing attacks, cache timing, power analysis).
- En un escenario de 10 empresas competidoras de alto valor, un atacante motivado podría intentarlo.

**Defensa arquitectónica**: diseñar el C2D para que incluso con un TEE comprometido, el atacante solo obtenga una partición pequeña de datos de un solo proveedor — no el dataset completo. Mediante:
- Particionado de datos entre múltiples enclaves
- Rotación periódica de enclaves
- Exposición mínima de datos por sesión de cómputo

### 7.6 Defensa en profundidad: ninguna capa es perfecta

```
TEE solo           → vulnerable a side-channel attacks
+ Aislamiento red  → elimina exfiltración directa
+ Result vetting   → elimina covert channels en output  
+ Privacidad dif.  → elimina inference attacks
+ Watermarking     → detecta copia post-facto
+ Audit log        → evidencia para disputas legales
```

La combinación hace que el **coste de atacar supere el beneficio esperado** para cualquier competidor racional. No hay seguridad perfecta; hay seguridad económicamente disuasoria.

---

## 8. Analogía Guardrail LLM ↔ Seguridad C2D

### 8.1 El problema abstracto compartido

Ambos sistemas resuelven el mismo problema:

> Un sistema procesa inputs y produce outputs. Necesitas garantías en **ambas direcciones** — lo que entra y lo que sale — sin poder confiar completamente ni en quien envía ni en quien recibe.

### 8.2 Mapa de equivalencias

| Capa | Guardrail LLM | Seguridad C2D |
|---|---|---|
| **Validación input** | Detecta prompt injection, jailbreaks, contenido dañino | Auditoría del algoritmo: detecta código malicioso, ops prohibidas |
| **Sandbox de ejecución** | System prompt + contexto: delimita qué puede hacer el modelo | TEE + aislamiento de red: delimita qué puede acceder/enviar el algoritmo |
| **Filtrado output** | Output guardrail: filtra PII, temas prohibidos, alucinaciones | Result vetting: filtra datos raw codificados, patrones de exfiltración |
| **Ruido estadístico** | Temperatura + sampling: variabilidad que dificulta extracción de training data | Privacidad diferencial (ε-DP): ruido que dificulta inference attacks |
| **Trazabilidad** | Log de conversaciones para auditoría y fine-tuning | Audit log on-chain: hash de algoritmo + resultado inmutable en blockchain |

### 8.3 La diferencia fundamental: el árbitro

| | Guardrail LLM | Seguridad C2D |
|---|---|---|
| **Árbitro** | Proveedor del modelo (Anthropic, OpenAI, etc.) | TEE + smart contract |
| **Tipo** | Tercero de confianza **centralizado** | Sin tercero central (trustless) |
| **Confianza** | Confías en el proveedor | Confías en las matemáticas y el hardware |
| **Actualización** | El proveedor puede parchear overnight | Requiere actualización del protocolo on-chain |
| **Aplicabilidad** | Adecuado cuando hay un operador responsable | Necesario entre partes que no confían entre sí |

### 8.4 Por qué C2D necesita ser trustless

En el escenario de 10 empresas competidoras, **ninguna aceptaría que otra empresa fuera el árbitro**. No puedes pedirle a Volkswagen que sea el árbitro de un pool de datos donde también está BMW. Por eso C2D necesita un árbitro matemático (TEE + blockchain), no un árbitro humano o corporativo.

### 8.5 Ventaja del guardrail LLM sobre C2D en práctica

El guardrail LLM gestionado por un proveedor puede:
- Actualizarse ante nuevas amenazas en horas
- Adaptarse a nuevos tipos de jailbreak
- Incorporar feedback de incidentes en tiempo real

El sistema C2D en blockchain es más rígido: cambiar las reglas requiere actualizar smart contracts y puede necesitar consenso de los participantes. Más seguro en teoría de modelo, más frágil ante vulnerabilidades imprevistas en práctica.

---

## 9. Conclusiones y Razonamientos Clave

### 9.1 Insight central: P2P lleva 25 años optimizando lo que Pontus-X necesita

P2P resolvió hace décadas problemas de:
- Descubrimiento descentralizado sin servidor central
- Verificación de integridad sin árbitro
- Incentivos alineados sin coordinador
- Paralelismo automático de transferencia

Pontus-X resuelve un problema más difícil (soberanía + compliance + monetización) pero puede **importar directamente los patrones de paralelismo, verificación y alineación de incentivos** que hacen robusto a BitTorrent.

### 9.2 La inversión del flujo de datos es el salto conceptual clave

La transición de P2P a C2D es conceptualmente: **de mover datos hacia el algoritmo, a mover el algoritmo hacia los datos**. Esta inversión es lo que hace posible la soberanía del dato, pero introduce la complejidad de seguridad en ambos lados descrita en la sección 7.

### 9.3 Tit-for-tat es el mecanismo más transferible

El sistema de incentivos de BitTorrent (tit-for-tat + optimistic unchoking) es el patrón más directamente aplicable a pools de datos entre competidores. La clave es que el scoring ocurre **dentro del C2D** — ningún competidor expone su posición de datos al sistema de incentivos.

### 9.4 El TEE es el punto de Schelling de la confianza

En cualquier sistema donde múltiples partes desconfían entre sí pero necesitan colaborar, necesitas un **punto neutral de ejecución**. En P2P este punto son las matemáticas del hash. En C2D es el TEE + blockchain. Sin este punto de Schelling, la colaboración entre competidores es imposible.

### 9.5 La brecha más importante en Pontus-X en 2026

Falta el equivalente de un **cliente BitTorrent para datos**: un agente que gestione automáticamente la adquisición paralela de N accesos, la orquestación de N jobs C2D, la verificación de N resultados y la agregación final. Hoy esto requiere integración manual o scripting ad-hoc.

### 9.6 Privacidad diferencial: el trade-off inevitable

En cualquier sistema que comparte resultados de cómputo sobre datos privados, hay un trade-off fundamental entre:
- **Precisión del resultado** (útil para el consumidor)
- **Privacidad del dato** (garantía para el proveedor)

La privacidad diferencial (ε-DP) formaliza matemáticamente este trade-off. No hay forma de eliminarlo; solo de gestionarlo transparentemente. Este es un aprendizaje que se traslada directamente del mundo LLM (donde la temperatura controla un trade-off análogo) al mundo C2D.

---

## 10. Glosario Técnico

| Término | Definición |
|---|---|
| **BitTorrent** | Protocolo P2P de transferencia de archivos. Referencia técnica central de esta conversación. |
| **BEP** | BitTorrent Enhancement Proposal. Sistema de especificaciones del protocolo. |
| **C2D** | Compute-to-Data. Paradigma donde el algoritmo viaja al dato en lugar del dato al algoritmo. |
| **Choking** | En BitTorrent, cortar la conexión de upload a un peer que no contribuye. |
| **DDO** | Decentralized Data Object. Metadatos de un asset publicados on-chain en Ocean Protocol / Pontus-X. |
| **DP-SGD** | Differentially Private Stochastic Gradient Descent. Variante del entrenamiento de ML con privacidad diferencial integrada. |
| **DHT** | Distributed Hash Table. Estructura de datos distribuida para localizar peers sin servidor central. |
| **ε-DP** | Epsilon Differential Privacy. Parámetro que cuantifica el grado de privacidad diferencial (ε pequeño = más privado). |
| **ERC-20** | Estándar de tokens fungibles en blockchain Ethereum / EVM compatible. Usado para data tokens en Pontus-X. |
| **Federated Learning** | Entrenamiento de modelos ML sobre datos distribuidos sin centralizar los datos. Solo se comparten gradientes o modelos parciales. |
| **FHE** | Fully Homomorphic Encryption. Cifrado que permite operar sobre datos sin descifrarlos. |
| **Free-rider** | Participante que consume recursos de un sistema sin contribuir. Problema que tit-for-tat resuelve. |
| **Gaia-X** | Iniciativa europea de framework para espacios de datos soberanos e interoperables. |
| **GXDCH** | Gaia-X Digital Clearing Houses. Servicios de verificación de credenciales en el ecosistema Gaia-X. |
| **Infohash** | Identificador único de un torrent (hash SHA-256 del archivo .torrent). |
| **Intel SGX** | Software Guard Extensions. Tecnología de Intel para crear enclaves de ejecución seguros (TEE). |
| **Membership Inference Attack** | Ataque que infiere si un dato específico estaba en el dataset de entrenamiento de un modelo. |
| **Merkle Tree** | Árbol de hashes donde cada nodo padre es el hash de sus hijos. Permite verificación eficiente de subconjuntos. |
| **Model Watermarking** | Técnica para insertar un patrón oculto en un modelo ML para detectar copias no autorizadas. |
| **Ocean Protocol** | Protocolo de marketplace descentralizado de datos. Base técnica de Pontus-X. |
| **Optimistic Unchoking** | En BitTorrent, abrir periódicamente conexiones a peers desconocidos para descubrir buenos colaboradores. |
| **Pontus-X** | Primer y mayor ecosistema X europeo. Implementación de referencia open-source de Gaia-X. Lighthouse Data Space. |
| **Remote Attestation** | Mecanismo por el que un TEE firma criptográficamente que ejecutó exactamente el código especificado. |
| **Result Vetting** | Análisis del output de un cómputo C2D antes de entregarlo al consumidor, buscando datos raw exfiltrados. |
| **Seeding** | En BitTorrent, servir chunks de un archivo que ya se ha descargado completamente. |
| **Side-channel Attack** | Ataque que extrae información analizando características físicas de la ejecución (timing, consumo de energía, caché) en lugar de atacar el algoritmo directamente. |
| **SMPC** | Secure Multi-Party Computation. Técnica criptográfica para computación conjunta sin revelar inputs individuales. |
| **SSI** | Self-Sovereign Identity. Modelo de identidad donde el usuario controla sus propias credenciales verificables. |
| **TEE** | Trusted Execution Environment. Entorno de ejecución aislado y cifrado a nivel hardware (Intel SGX, AMD SEV, ARM TrustZone). |
| **Tit-for-tat** | Estrategia de teoría de juegos: coopera con quien coopera contigo, defecta con quien defecta. Base del sistema de incentivos de BitTorrent. |

---

*Fin de la base de conocimiento. Esta sesión cubre: arquitectura P2P, Pontus-X/Gaia-X como espacio de datos federado, analogía estructural entre ambos, protocolo tit-for-tat para datos entre competidores, seguridad bilateral en C2D (amenazas de robo de datos y robo de algoritmo), y analogía entre guardrails LLM y seguridad C2D.*
