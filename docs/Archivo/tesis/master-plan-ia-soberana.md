---
title: "Master Plan: Arquitectura de IA Soberana y Federada (Abril 2026)"
tags: [tesis]
status: borrador
updated: 2026-08-08
---

# Master Plan: Arquitectura de IA Soberana y Federada (Abril 2026)

## 1. Fundamentos Técnicos y Clasificación
La Inteligencia Artificial en 2026 se divide por su capacidad funcional y método de aprendizaje. 

* **IA Generativa (GenAI):** Modelos probabilísticos (Transformers) diseñados para la creación y síntesis de contenido (texto, imagen, código). Actúa como la interfaz de razonamiento semántico.
* **Reinforcement Learning (RL):** Rama del Machine Learning centrada en la toma de decisiones secuenciales. Un agente interactúa con un entorno para maximizar una recompensa. 
* **Encaje Estratégico:** La GenAI aporta la *comprensión del contexto*, mientras que el RL aporta la *estrategia de decisión*.

---

## 2. Arquitectura del Sistema: The Stack

### A. Nodo de Inteligencia (VPS Propio)
* **Infraestructura:** VPS bajo control total (Ubuntu 24.04/26.04), acceso vía VPN/Wireguard.
* **Motor de Inferencia:** **vLLM** o **Ollama** ejecutando **Phi-4-mini (3.8B)**.
* **Privacidad:** Inferencia 100% local. El modelo es un archivo de pesos; no hay envío de datos a nubes externas (Zero-Data Leakage).
* **Orquestación:** Agente en Python usando `ocean.py`.

### B. Espacio de Datos (Pontus-X / Ocean Protocol)
* **Mecanismo:** **Compute-to-Data (CtD)**. El algoritmo viaja al dato, no al revés.
* **Soberanía:** El proveedor del dato mantiene el control físico; el consumidor (tú) obtiene el resultado del cómputo.
* **Blockchain:** Red **GEN-X** para registro de Smart Contracts, pagos y auditoría de ejecución.

---

## 3. Implementación de RL Offline (Diferido)
En lugar de aprendizaje en tiempo real (arriesgado e ineficiente), se emplea **Offline RL**:
1.  **Recolección:** Captura de logs históricos en formato JSON.
2.  **Entrenamiento:** El modelo estudia trayectorias pasadas (`Estado -> Acción -> Recompensa`).
3.  **Algoritmos:** Uso de enfoques conservadores (CQL o IQL) para evitar que la IA tome acciones no probadas en los datos originales.

---

## 4. Caso de Uso: Sector ITV (Inspección Técnica de Vehículos)

### El Dataset (JSON)
Registros que incluyen: fecha, marca, modelo, año, km/horas y códigos de defectos (Nivel 1: Leves).

### Flujo de Ejecución Federada
1.  **Embarque:** Se empaqueta Phi-4-mini + Script Python en un **Dockerfile** ligero.
2.  **Ejecución:** El contenedor corre en el nodo de la ITV. Procesa miles de registros JSON.
3.  **Insight:** La IA identifica qué defectos leves en tractores de >15 años preceden a fallos críticos (ej. dirección/frenos).
4.  **Resultado:** Se genera un fichero `itv_policy_learned.json` con las políticas de riesgo aprendidas.

---

## 5. Salidas y Entregables

### En el VPS (Consolidación)
* **Fine-tuning (QLoRA):** Se ajusta el Phi-4-mini local con los resultados de múltiples ITVs. El modelo se convierte en un "Experto en Seguridad Vial Industrial".
* **Inferencia Especializada:** Capacidad de responder consultas complejas como: *"¿Por qué la Marca X en la Provincia Y falla un 15% más que la media?"*.

### Reporte Ejecutivo (PDF)
* **Resumen Operativo:** Nodos procesados y cumplimiento normativo.
* **Análisis de Riesgo:** Correlaciones entre antigüedad y fallos específicos.
* **Recomendaciones:** Políticas de mantenimiento preventivo para stakeholders.

---

## 6. Seguridad y Gobernanza
* **Principio de Menor Privilegio (PoLP):** El agente corre en contenedores aislados.
* **Hardening:** Bloqueo de tráfico WAN saliente en el nodo de cómputo para garantizar que el LLM no filtre datos.
* **Trazabilidad:** Cada decisión de la IA queda registrada en el Job ID de Pontus-X para auditoría legal (EU AI Act).

---
**Objetivo Final:** Transformar datos privados inaccesibles en inteligencia comercial propietaria mediante el uso de modelos pequeños pero potentes (SLM) y protocolos de computación descentralizada.