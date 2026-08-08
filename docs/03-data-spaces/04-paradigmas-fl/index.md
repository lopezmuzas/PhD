---
title: "4. Paradigmas de Aprendizaje Federado en Data Spaces"
tags: [data-spaces, federated-learning, paradigmas, lora, offline-rl, autosemisupervised]
status: revisado
---

# Paradigmas de Aprendizaje Automático en Data Spaces

> Los Espacios de Datos soberanos e industriales albergan escenarios extremadamente variados: desde organizaciones con datos estructurados etiquetados hasta participantes con petabytes de logs IoT no etiquetados o trayectorias de sensores. Este documento analiza los **6 paradigmas de Deep Learning distribuido** aplicables a Data Spaces.

---

## 1. Aprendizaje Federado Supervisado (FedSupervised)

El paradigma clásico donde cada nodo posee datos locales etiquetados estructurados uniformemente (Horizontal FL) o con alineación de entidades (Vertical FL).

### Técnicas y Algoritmos Clave
- **FedAvg (Federated Averaging)**: Algoritmo base. Los nodos entrenan localmente durante $E$ épocas y el servidor promedia los pesos: $w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k$.
- **FedProx**: Regularización proximal para limitar la deriva del modelo en entornos no-IID.
- **SCAFFOLD**: Control variates para corregir el *client drift*.

### Pros y Contras
- ✅ **Pros**: Garantía de convergencia robusta, alta precisión sobre métricas directas de negocio.
- ❌ **Contras**: Dependencia crítica del etiquetado local (costoso), sobrecarga de ancho de banda al transferir modelos completos.

---

## 2. Aprendizaje Federado No Supervisado (FedUnsupervised)

Diseñado para escenarios comunes en Data Spaces industriales donde los participantes acumulan telemetría, logs o datos en bruto sin etiquetar.

### Técnicas y Algoritmos Clave
- **Federated Clustering (FedK-Means / FedDBSCAN)**: Compartición de centroides locales anonimizados para segmentación global.
- **Federated Autoencoders (FedAE)**: Cada nodo entrena un Autoencoder local para aprender la representación comprimida de sus datos. El servidor promedia el cuello de botella (*bottleneck*). Crucial para **detección de anomalías sin conocer patrones previos de fallo**.

### Pros y Contras
- ✅ **Pros**: Ingesta directa de datos IoT masivos sin coste de etiquetado; privacidad intrínseca al compartir solo representaciones latentes.
- ❌ **Contras**: Desalineación de espacios latentes si los nodos son muy heterogéneos (non-IID); validación compleja sin conjunto etiquetado centralizado.

---

## 3. Aprendizaje Federado Auto-Supervisado (FedSelfSupervised / FedSSL)

Evolución del no supervisado para Deep Learning. El sistema genera de forma autónoma sus propias "etiquetas" a partir de transformaciones del propio dato (rotación, enmascaramiento, contraste).

### Técnicas y Algoritmos Clave
- **FedSimCLR / FedBYOL**: Adaptaciones federadas de aprendizaje contrastivo. Los nodos maximizan la similitud entre diferentes vistas aumentadas de un mismo dato.

### Pros y Contras
- ✅ **Pros**: Aprovechamiento de petabytes de datos no etiquetados; genera *embeddings* de alta calidad aplicables a múltiples tareas aguas abajo (*downstream tasks*).
- ❌ **Contras**: Cómputo local masivo (*batch sizes* muy grandes); riesgo de colapso de representación si la agregación no está ajustada.

---

## 4. Aprendizaje por Transferencia Federado (Federated Transfer Learning - FTL)

Fundamental cuando los participantes difieren en el espacio de características o cuando un nodo pequeño quiere heredar capacidades de un Foundation Model pre-entrenado.

### Técnicas y Algoritmos Clave
- **FedLoRA (Federated Low-Rank Adaptation)**: En lugar de transmitir gigabytes de pesos de un Foundation Model (LLM o VLM), los nodos solo entrenan e intercambian matrices adaptativas de bajo rango ($A \times B$).
- **Vertical FTL**: Algoritmos basados en *Homomorphic Encryption* para alinear características complementarias de un mismo usuario entre empresas distintas (ej. Banco A y Aseguradora B).

### Pros y Contras
- ✅ **Pros**: Eficiencia de comunicación radical (reduce los *payloads* en hasta un $99\%$); permite participar a nodos pequeños con infraestructura limitada.
- ❌ **Contras**: Complejidad criptográfica alta en Vertical FTL (*Private Set Intersection* + Cifrado Homomórfico).

---

## 5. Aprendizaje por Refuerzo Federado Tradicional (Federated Online RL)

Aplicado a la toma de decisiones secuenciales y control de entornos dinámicos distribuidos mediante prueba y error en tiempo real.

### Técnicas y Algoritmos Clave
- **Federated DDPG / FedPPO**: Agentes locales interactúan con entornos físicos (robots, almacenes) y comparten los parámetros de sus redes Actor-Crítico.

### Pros y Contras
- ✅ **Pros**: Optimización de procesos dinámicos en tiempo real; alta adaptabilidad a la Industria 4.0.
- ❌ **Contras**: **Riesgo catastrófico local**: la exploración en vivo puede dañar equipos críticos; no-estacionariedad extrema.

---

## 6. Aprendizaje por Refuerzo Offline Federado (Federated Offline RL / Batch RL)

La alternativa pragmática para Data Spaces corporativos: **elimina la exploración en vivo**. El algoritmo aprende decisiones óptimas leyendo exclusivamente conjuntos de datos estáticos de experiencias pasadas (logs históricos).

### Técnicas y Algoritmos Clave
- **Federated Conservative Q-Learning (FedCQL)** y **Federated IQL**: Introducen penalizaciones matemáticas para evitar que el modelo asigne valores altos a acciones o estados no presentes en los datos históricos.

### Pros y Contras
- ✅ **Pros**: **Riesgo cero en producción** (exploración segura sobre datos históricos de salud, energía o industria); capitaliza terabytes de logs inactivos.
- ❌ **Contras**: *Distributional shift* (falta de extrapolación si el sistema encuentra estados no representados en el histórico de ningún participante).

---

## 📊 Matriz de Decisión Arquitectónica para Data Spaces

| Dimensión | FedSupervised | FedUnsupervised | FedSelfSupervised | FedTransfer (LoRA) | FedOnlineRL | FedOfflineRL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Requisito de Datos** | Labeled / Homogéneo | Unlabeled | Unlabeled Masivo | Pre-trained + Local | Entorno Interactivo | Logs Históricos |
| **Carga de Red** | Alta | Media | Muy Alta | **Muy Baja** | Alta | Media |
| **Carga de Cómputo** | Media | Baja | **Muy Alta** | Baja | Media-Alta | Media |
| **Riesgo Operacional** | Medio | Bajo | Bajo | Bajo | **Extremo** | **Bajo (Seguro)** |
| **Caso de Uso Primario** | Clasificación Estándar | Detección de Anomalías | Extracción Embeddings | Domain Adaptation | Control IoT Dinámico | **Optimización Segura** |
