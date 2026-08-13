---
title: "3. Data Spaces y Gobernanza"
tags: [data-spaces, gaia-x, IDSA, ocean, pontus-x, gobernanza]
status: esbozo
---

# Data Spaces y Gobernanza

> Un Data Space es el **marco interoperable, basado en principios de gobernanza, estándares, prácticas y servicios comunes, que permite transacciones de datos fiables entre participantes** — preservando la soberanía del dato.
>
> *Definición CEN-CENELEC / DSSC*

---

## La capa de gobernanza que FL necesita

FL resuelve el *"no mover los datos"* a nivel técnico. El Data Space resuelve el *"¿bajo qué reglas, identidad y contrato compartimos?"* a nivel de gobernanza. Sin ambas capas, el aprendizaje federado cross-organización no es viable legalmente.

```
┌──────────────────────────────────────────┐
│           CAPA DE GOBERNANZA             │
│   Data Space = reglas + identidad +      │
│   contratos + auditoría                  │
├──────────────────────────────────────────┤
│           CAPA TÉCNICA                   │
│   Federated Learning = modelo viaja,     │
│   datos no                               │
└──────────────────────────────────────────┘
```

---

## Componentes clave de un Data Space

| Componente | Qué hace |
|---|---|
| **Conector (Connector)** | El endpoint técnico por el que dos participantes intercambian datos de forma soberana |
| **Identidad y confianza (Trust Framework)** | Cada participante tiene una identidad verificable; se establece confianza antes de compartir |
| **Catálogo** | Hace los datos *findable* — describe qué hay disponible y bajo qué condiciones |
| **Rulebook / Governance Framework** | El conjunto de reglas legales, de negocio y técnicas del Data Space |
| **Usage Policies** | Condiciones de uso que viajan pegadas al dato |

---

## El ecosistema europeo

### Gaia-X

Iniciativa europea de infraestructura federada y *trust framework*. Define **las reglas**, no la implementación.

- Proporciona especificaciones, arquitectura y gobernanza
- No es una plataforma técnica sino un estándar

### Pontus-X

El primer y mayor ecosistema X público europeo. **La implementación de referencia open-source de Gaia-X**. Segundo Lighthouse Data Space oficial de Gaia-X.

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

| | Gaia-X | Pontus-X |
|---|---|---|
| Rol | Framework / estándar | Implementación de referencia |
| Proporciona | Reglas, arquitectura, gobernanza | Software, herramientas, stack técnico |
| Analogía | Plano arquitectónico | Edificio construido según el plano |

### IDSA — International Data Spaces Association

Proporciona el **IDS-RAM** (Reference Architecture Model) y el **Dataspace Protocol (DSP)**. La arquitectura se organiza en cinco capas: negocio, funcional, proceso, información y sistema.

### DSSC — Data Spaces Support Centre

Centro de la UE para el ecosistema de Data Spaces. Publica el *blueprint*, el *starter kit* y el glosario de referencia: [dssc.eu](https://dssc.eu).

### Eclipse Dataspace Components (EDC)

La implementación open-source del conector de referencia. Implementa el IDS Dataspace Protocol y es compatible con Gaia-X. Punto de partida técnico: [Minimal Viable Dataspace (MVD)](https://github.com/eclipse-edc).

---

## Compute-to-Data (C2D): la inversión del flujo

La diferencia más importante respecto a P2P y sistemas tradicionales:

> **En sistemas tradicionales:** el dato viaja hacia el algoritmo.
> **En C2D:** el algoritmo (el consumidor) viaja hacia el dato.

Esta inversión es la que hace posible la **soberanía del dato**: el dato nunca abandona la infraestructura de su propietario.

```
SISTEMA TRADICIONAL:
[Proveedor datos] ──dato──▶ [Servidor central] ──resultado──▶ [Consumidor]

COMPUTE-TO-DATA:
[Consumidor] ──algoritmo──▶ [Proveedor datos] ──resultado──▶ [Consumidor]
                                      ↑
                                datos nunca
                                salen de aquí
```

### Flujo técnico en Pontus-X

```
1. DESCUBRIMIENTO
   └─ Consulta el catálogo on-chain
   └─ Localiza los DDOs (uno por dataset)
   └─ Verifica condiciones de uso

2. NEGOCIACIÓN DE ACCESO
   └─ Compra data token ERC-20 por cada dataset
   └─ Smart contract verifica credenciales Gaia-X del consumidor
   └─ Acceso autorizado por tiempo/uso limitado

3. EJECUCIÓN C2D
   └─ Consumidor envía su algoritmo a cada nodo
   └─ Cada nodo ejecuta en TEE aislado
   └─ Cada nodo devuelve gradientes / resultado parcial

4. AGREGACIÓN
   └─ Consumidor agrega los resultados parciales
   └─ Ningún proveedor vio datos de los demás
   └─ Ningún proveedor vio el algoritmo del consumidor en claro
```

---

## Analogía P2P ↔ Pontus-X

Pontus-X y BitTorrent resuelven problemas análogos con arquitecturas similares. La analogía ayuda a entender el sistema:

| Capa | P2P (BitTorrent) | Pontus-X |
|---|---|---|
| **Descubrimiento** | DHT / Tracker | Catálogo on-chain (DDO) |
| **Identificador de recurso** | Infohash (SHA-256) | DDO address + data token address |
| **Autenticación / acceso** | Ninguna | Data token ERC-20 + credencial Gaia-X (SSI) |
| **Transferencia** | Chunks del dato → consumidor | Algoritmo → dato (C2D) |
| **Verificación integridad** | Hash por chunk (Merkle tree) | DLT + smart contract + remote attestation |
| **Incentivos** | Tit-for-tat / choking | Data token + liquidación ERC-20 |
| **Gobernanza** | Protocolo abierto (BEP), sin compliance | GDPR + Data Act + Gaia-X Trust Framework |

### Lo que P2P resolvió hace 25 años y Pontus-X necesita hoy

P2P lleva décadas optimizado para:
- Descubrimiento descentralizado sin servidor central
- Verificación de integridad sin árbitro
- Incentivos alineados sin coordinador
- Paralelismo automático de transferencia

**La brecha más importante en 2026**: no existe el equivalente de un cliente BitTorrent para datos — un agente que gestione automáticamente la adquisición paralela de N accesos, la orquestación de N jobs C2D, la verificación de N resultados y la agregación final. Hoy esto requiere scripting manual.

---

## Seguridad en C2D: las dos amenazas simétricas

C2D tiene dos atacantes potenciales con objetivos opuestos:

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

### Defensa en profundidad

```
TEE solo           → vulnerable a side-channel attacks
+ Aislamiento red  → elimina exfiltración directa
+ Result vetting   → elimina covert channels en output
+ Privacidad dif.  → elimina inference attacks
+ Watermarking     → detecta copia post-facto
+ Audit log        → evidencia para disputas legales
```

| Mecanismo | Protege a | Cómo |
|---|---|---|
| **TEE (Intel SGX, AMD SEV)** | Ambos | El código y los datos del enclave están cifrados en memoria — ni el proveedor ve el algoritmo, ni el consumidor ve los datos |
| **Remote Attestation** | Consumidor | El TEE firma que ejecutó exactamente el código enviado, con los datos especificados |
| **Result Vetting** | Proveedor | El output se analiza antes de salir — ¿contiene datos raw codificados? |
| **DP-SGD** | Proveedor | Ruido matemático que bloquea ataques de inferencia de membresía |
| **Model Watermarking** | Consumidor | Patrón oculto en el modelo que detecta copias no autorizadas |
| **Audit Log on-chain** | Ambos | Registro inmutable de qué algoritmo, sobre qué datos, produjo qué resultado |

### El TEE como punto de Schelling

En un Data Space donde múltiples organizaciones desconfían entre sí, el TEE actúa como árbitro matemático neutral:

- El proveedor no puede copiar el algoritmo porque **el código nunca está en claro en su sistema**
- El consumidor confía en esto porque recibe una **remote attestation firmada criptográficamente**
- **Ni el proveedor ve el algoritmo real, ni el consumidor ve los datos raw**

---

## Regulación de referencia

| Norma | Qué establece | Relevancia para Data Spaces |
|---|---|---|
| **GDPR** | Protección de datos personales | Base legal para procesamiento de datos sensibles |
| **Data Governance Act** | Intermediarios de datos, reutilización de datos públicos | Marco para Data Spaces como intermediarios |
| **Data Act (2023)** | Portabilidad, acceso a datos IoT | Derechos sobre datos generados por máquinas |
| **AI Act (2024)** | Trazabilidad, transparencia de IA | Requisitos de auditoría para sistemas entrenados en Data Spaces |
| **Gaia-X Trust Framework** | Identidad verificable, self-descriptions | Capa de confianza técnica del ecosistema |

---

## Recursos de referencia

- [DSSC Starter Kit](https://dssc.eu/) — el punto de entrada más claro de la UE
- [DSSC Blueprint](https://blueprint.dssc.eu/) — arquitectura de referencia
- [IDS-RAM 4](https://docs.internationaldataspaces.org/) — modelo de referencia IDSA
- [Gaia-X](https://gaia-x.eu/) — trust framework europeo
- [Eclipse EDC](https://projects.eclipse.org/projects/technology.edc) — conector de referencia open-source
- [Pontus-X](https://pontus-x.eu/) — implementación de referencia Gaia-X
- Libro open-access: [Designing Data Spaces](https://link.springer.com/book/10.1007/978-3-030-93975-5) — Otto, ten Hompel, Wrobel (eds.)

<!-- nav-start -->

---

← Anterior: [2. Aprendizaje Federado](../02-aprendizaje-federado/index.md)  
Siguiente: [4. Paradigmas FL en Data Spaces](../04-paradigmas-fl/index.md) →

<!-- nav-end -->
