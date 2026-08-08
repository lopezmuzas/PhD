---
title: "Módulo 4 — Data Spaces (espacios de datos): la capa de gobernanza"
tags: [recursos, ruta, frl]
status: borrador
updated: 2026-08-08
---

# Módulo 4 — Data Spaces (espacios de datos): la capa de gobernanza

> 🧭 **Ruta FRL:** [Índice](index.md) · [0](modulo-0-cimientos-matematicos.md) · [1](modulo-1-ciclo-entrenamiento.md) · [2](modulo-2-paradigmas.md) · [3](modulo-3-aprendizaje-federado.md) · **4**
> 🗺️ **Mapa mental:** Sección 6.
> **Leyenda:** `[ tipo · nivel · tiempo ]` — 📘 PDF/libro · 🎥 vídeo · 🔗 web/curso · 🧑‍💻 práctica · ⭐ intro / ⭐⭐ intermedio / ⭐⭐⭐ avanzado.

**Objetivo:** entender el data space como la **capa de gobernanza** que hace viable el intercambio de datos *entre organizaciones distintas*: reglas comunes (rulebook), identidad y confianza, conectores y **soberanía del dato**. Es lo que convierte el cross-silo del Módulo 3 de "técnicamente posible" en "organizativa y legalmente viable".
**Prerrequisito conceptual:** [Módulo 3](modulo-3-aprendizaje-federado.md) — el FL resuelve el *"no mover los datos"* a nivel técnico; el data space resuelve el *"¿bajo qué reglas, identidad y contrato compartimos?"* a nivel de gobernanza.

> ⚠️ **Cambio de registro:** este módulo es más de **lectura y arquitectura** que de código. No esperes un "entrena tu primer modelo"; espera entender roles, estándares y marco regulatorio europeo. Es el módulo más "europeo" e industrial de la ruta.

## Conceptos que debes dominar al salir

1. **Por qué un data space:** entre organizaciones no basta la tecnología de FL; hace falta acordar *quién accede, bajo qué condiciones de uso, con qué identidad y con qué garantías legales*. Un data space aporta esa capa común.
2. **Definición (CEN-CENELEC / DSSC):** *marco interoperable, basado en principios de gobernanza, estándares, prácticas y servicios comunes, que permite **transacciones de datos fiables** entre participantes* — preservando la soberanía del dato.
3. **Soberanía del dato (data sovereignty):** el proveedor mantiene control sobre **quién** usa sus datos y **cómo**, mediante *usage policies* y *usage control* (las condiciones viajan pegadas al dato).
4. **Componentes y roles clave:**
   - **Conector (connector):** el endpoint técnico por el que dos participantes intercambian datos de forma soberana.
   - **Identidad y confianza (trust framework):** cada participante tiene una identidad verificable; se establece confianza antes de compartir.
   - **Catálogo:** hace los datos *findable* (encontrables) y describe sus condiciones.
   - **Rulebook / governance framework:** el conjunto de reglas legales, de negocio y técnicas del data space.
5. **Building blocks (DSSC):** se agrupan en **business & organizational** (modelo de negocio, gobernanza, legal) y **technical** (interoperabilidad, identidad, soberanía...).
6. **Iniciativas y estándares de referencia:**
   - **IDSA** (International Data Spaces Association) → el **IDS-RAM** (modelo de arquitectura de referencia) y el **Dataspace Protocol (DSP)**.
   - **Gaia-X** → infraestructura federada europea y *trust framework*.
   - **DSSC** (Data Spaces Support Centre) → el centro de la UE con el *blueprint*, *starter kit* y glosario.
   - **Conectores** → **Eclipse Dataspace Components (EDC)**, la implementación open source de referencia.
7. **Marco regulatorio europeo:** GDPR, **Data Governance Act (DGA)**, **Data Act**, y la estrategia de los *Common European Data Spaces* (salud, movilidad, energía, industria...).
8. **Relación con FL/FRL:** el data space es la **gobernanza**; el FL/FRL es un **servicio de creación de valor** (un servicio de IA) que opera *dentro* de ese marco. El cross-silo FL del Módulo 3 vive, en el mundo real, sobre un data space.

## Ruta mínima (en este orden)

- 🔗⭐ **DSSC — Starter Kit & Blueprint (introducción conceptual de la UE)** · https://dssc.eu/ · blueprint: https://blueprint.dssc.eu/
  El punto de entrada más claro y *neutral*: qué es un data space, sus conceptos clave, roles y building blocks. Empieza por el *Starter Kit* y la sección *Key Concepts*.
- 📘⭐⭐ **Designing Data Spaces** — Otto, ten Hompel, Wrobel (eds.), libro **open access** · https://link.springer.com/book/10.1007/978-3-030-93975-5
  La referencia de fondo, gratuita. Lee la **Parte I "Foundations and Contexts"**, que da la visión general de construir/gobernar data spaces e introduce IDS y Gaia-X. Las partes II–IV (tecnología y casos por sector) son para profundizar.
- 🔗⭐⭐ **IDSA — *Understanding the Reference Architecture Model (RAM)*** · https://internationaldataspaces.org/understanding-the-idsa-reference-architecture-model/ · base de conocimiento IDS-RAM 4: https://docs.internationaldataspaces.org/
  Para entender la arquitectura por capas (negocio, funcional, proceso, información, sistema) y dónde encaja el conector. La base de conocimiento permite consultar por temas.

## Profundización opcional

- 🔗⭐⭐ **Gaia-X — European Association for Data and Cloud** · https://gaia-x.eu/
  La iniciativa de infraestructura federada europea y su *trust framework*. Útil para entender la dimensión "soberanía cloud" complementaria a IDS.
- 🧑‍💻⭐⭐⭐ **Eclipse Dataspace Components (EDC) — conector de referencia (open source)** · proyecto: https://projects.eclipse.org/projects/technology.edc · código: https://github.com/eclipse-edc
  Si quieres ver/ejecutar un conector real: implementa el IDS Dataspace Protocol y es compatible con Gaia-X. Empieza por el *Minimal Viable Dataspace (MVD)* del repositorio.
- 🔗⭐⭐ **DSSC — Glosario** · https://dssc.eu/space/bv15e/766061530
  Terminología precisa (data space, conector, rulebook, soberanía...). Útil como diccionario de consulta.
- 🔗⭐ **Estrategia europea del dato / Common European Data Spaces** · https://digital-strategy.ec.europa.eu/en/policies/data-spaces
  El contexto político-regulatorio (DGA, Data Act, espacios sectoriales) que explica *por qué* Europa impulsa este modelo.

## Checkpoint del módulo

Sabes explicar qué es la soberanía del dato, distingues los roles y building blocks de un data space, y entiendes el papel de IDSA, Gaia-X, DSSC y los conectores (EDC). Puedes argumentar por qué un FL cross-silo entre empresas necesita un data space por debajo.

## Puente al siguiente módulo

Ya tienes los **tres ejes** del mapa mental cubiertos: el **CÓMO** (Módulo 2, incluido el RL), el **CON QUÉ** (redes profundas, Módulos 0–1) y el **DÓNDE** (Módulo 3 federado + este Módulo 4 de gobernanza). El **Módulo 5: FRL** los une: aplicar **aprendizaje por refuerzo** (Módulo 2.3) en una **arquitectura federada** (Módulo 3), gobernada por un **data space** (Módulo 4). Es decir: **FRL = RL + federado, operando dentro de un data space.**
