---
title: "Aprendizaje Federado: Estado del Arte y Roadmap de Tecnologías"
tags: [tesis, estado-del-arte]
status: borrador
updated: 2026-08-08
---

# Aprendizaje Federado: Estado del Arte y Roadmap de Tecnologías

> Documento de preparación para tesis sobre **Aprendizaje Federado descentralizado y soberano** (Federated Learning + Compute-to-Data + Gaia-X / Pontus-X).
> Última actualización: junio 2026.

---

## 1. Estado del arte (resumen)

El aprendizaje federado (Federated Learning, FL) ha dejado de ser una novedad teórica. El debate de investigación se ha desplazado desde "¿es posible entrenar sin centralizar datos?" hacia "¿cómo lo desplegamos de forma robusta, privada y eficiente en el mundo real?".

Las revisiones sistemáticas recientes (2023–2025) estructuran el campo en torno a **tres ejes**:

1. **Distribución de los datos** → FL Horizontal, FL Vertical, Federated Transfer Learning y FL Personalizado.
2. **Sincronización de modelos** → síncrono vs. asíncrono.
3. **Seguridad** → defensa frente a fuga de datos y envenenamiento (poisoning).

### Retos abiertos dominantes
- **Coste de comunicación** (rondas cliente↔servidor caras).
- **Heterogeneidad estadística** (datos no-IID entre clientes).
- **Heterogeneidad de sistema** (clientes con distinta capacidad de cómputo / red).
- **Vulnerabilidades de privacidad** (reconstrucción a partir de gradientes, ataques de inferencia).
- **Client drop-out** y redes poco fiables.

### Tendencias clave (y dónde hay hueco para una tesis)
- **De centralizado (CFL) a descentralizado (DFL):** el enfoque centralizado introduce latencia por cuellos de botella, vulnerabilidad a fallos y problemas de confianza sobre la entidad que crea el modelo global. El DFL minimiza la dependencia de una autoridad central.
- **FL basado en blockchain / data spaces:** enfoques auto-soberanos. Las revisiones señalan que el FL basado en blockchain, el meta-aprendizaje federado y el **FL por refuerzo** siguen en etapas tempranas y poco estudiados → **buen terreno de tesis**.
- **FL + dataspaces europeos (Gaia-X):** integración de FL con Compute-to-Data sobre infraestructura soberana (Ocean Protocol, Pontus-X). Proyectos faro: **EuProGigant**, moveID.

### Posicionamiento sugerido para la tesis
El cruce natural de tus intereses está en:

> **FL descentralizado y auto-soberano implementado sobre Compute-to-Data (Ocean Protocol / Pontus-X), con FELT Labs como librería de referencia.**

El **offline RL** (aprendizaje por refuerzo con datos fijos) es atractivo como **línea secundaria** —encaja con la privacidad— pero es terreno poco maduro; conviene no convertirlo en el núcleo.

---

## 2. Esquema ejecutivo: tecnologías a dominar (de lo genérico a lo específico)

Pila por capas. Cada capa se apoya en la anterior.

```
CAPA 6  Librerías y plataformas concretas   →  FELT Labs, Flower, NVFLARE, Nautilus
CAPA 5  Infraestructura / data spaces        →  Ocean Protocol, Compute-to-Data, Gaia-X, Pontus-X
CAPA 4  Privacidad y seguridad               →  Differential Privacy, Secure Aggregation
CAPA 3  Aprendizaje Federado (núcleo)        →  FedAvg, no-IID, CFL vs DFL
CAPA 2  Deep Learning / Redes neuronales     →  CNN, RNN, Transformers, backprop
CAPA 1  Machine Learning (paradigmas)        →  Supervisado / No supervisado / Refuerzo
CAPA 0  Fundamentos matemáticos              →  Álgebra lineal, cálculo, probabilidad
```

---

### CAPA 0 — Fundamentos matemáticos
Lo mínimo para leer cualquier paper de FL sin perderte.

| Tema | Conceptos clave | Por qué importa en FL |
|------|-----------------|------------------------|
| Álgebra lineal | Vectores, matrices, producto matricial, normas, autovalores, SVD | Lenguaje de pesos y representaciones |
| Cálculo y optimización | Derivadas, gradientes, regla de la cadena, SGD, Adam | Base del backprop; cada cliente optimiza localmente |
| Probabilidad y estadística | Distribuciones, esperanza, varianza, inferencia bayesiana básica | Entender datos no-IID y heterogeneidad |

---

### CAPA 1 — Machine Learning (los tres paradigmas)
ML = algoritmos que aprenden patrones sin programación explícita.

- **Aprendizaje supervisado** — datos etiquetados (entrada → salida). Regresión y clasificación. *El caso más común en FL.*
- **Aprendizaje no supervisado** — sin etiquetas; se busca estructura. Clustering, PCA, autoencoders.
- **Aprendizaje por refuerzo (RL)** — agente que aprende por prueba-error maximizando recompensa.
  - **Offline / Batch RL:** aprende de un dataset fijo ya recolectado, sin interactuar en vivo. Compatible con escenarios de privacidad. Conceptos: política, función de valor, Q-learning, *distribution shift*.

> **Diferencia ML vs DL:** el ML clásico usa *features* diseñadas a mano; el Deep Learning las aprende solo.

---

### CAPA 2 — Deep Learning y redes neuronales
Subconjunto de ML basado en redes neuronales profundas.

- Perceptrón, redes feedforward, funciones de activación.
- **Backpropagation** (aplicación directa de la regla de la cadena).
- Arquitecturas: **CNN** (visión), **RNN / LSTM** (secuencias), **Transformers** (estado del arte actual).
- Regularización, overfitting, batch normalization, learning rate.
- Frameworks base: **PyTorch**, **TensorFlow**.

---

### CAPA 3 — Aprendizaje Federado (núcleo de la tesis)
- **FedAvg (Federated Averaging)** — algoritmo fundacional (McMahan et al., 2017): entrenamiento local + promediado de pesos. **Punto de partida obligatorio.**
- **Taxonomía por datos:**
  - FL Horizontal (mismas features, distintos sujetos).
  - FL Vertical (mismos sujetos, distintas features).
  - Federated Transfer Learning.
  - FL Personalizado.
- **Arquitectura:** Centralizado (CFL) vs **Descentralizado (DFL)** ← vía hacia Pontus-X.
- **Problemas centrales:** datos no-IID, heterogeneidad de sistema, coste de comunicación (compresión, pruning, cuantización), client drop-out.
- **Variantes de agregación:** FedProx, FedMeta, agregación robusta.

---

### CAPA 4 — Privacidad y seguridad
Es la razón de ser del FL; se evalúa a fondo.

- **Privacidad diferencial** (Differential Privacy).
- **Agregación segura** (Secure Aggregation).
- **Cifrado homomórfico.**
- **Modelos de amenaza:** envenenamiento (poisoning), inferencia de pertenencia, reconstrucción de datos desde gradientes.

---

### CAPA 5 — Infraestructura: data spaces y Compute-to-Data
Tu diferenciador frente a un FL "de libro".

- **Ocean Protocol** + **Compute-to-Data (CtD):** los datos nunca salen de la infraestructura del propietario; se llevan los algoritmos a los datos. Flujo técnico: el usuario busca dato + algoritmo en el catálogo federado → se cargan en un *pod* aislado en Kubernetes → solo se devuelven resultados y logs → el pod se elimina.
- **Gaia-X** y su **Trust Framework:** marco europeo de soberanía de datos.
- **Pontus-X:** primer ecosistema paneuropeo totalmente descentralizado en Gaia-X, construido sobre el stack open-source de Ocean Protocol. Segundo *Gaia-X Lighthouse Data Space*. Cumplimiento con Data Act, Data Governance Act, AI Act y GDPR.
- **Conceptos web3 asociados:** blockchain, smart contracts, DLT, identidad descentralizada, **Nautilus** (librería para interactuar con Pontus-X programáticamente).

---

### CAPA 6 — Librerías y plataformas concretas
Para la parte experimental.

| Herramienta | Rol | Notas |
|-------------|-----|-------|
| **FELT Labs** (`feltlabs`) | FL sobre Ocean Protocol / Compute-to-Data | Tu librería de referencia. Ver detalle abajo. |
| **Flower** | Framework de investigación, agnóstico de framework | El más usado en academia |
| **NVIDIA FLARE** | De simulación a producción | Orientado a despliegue real |
| **TensorFlow Federated** | FL en ecosistema TF | |
| **PySyft / OpenMined** | Privacidad + FL | |

#### FELT Labs en detalle (librería específica)
- **Qué es:** suite de herramientas para FL sobre datos privados y distribuidos, construida sobre Ocean Protocol y Compute-to-Data. Objetivo: FL seguro y anónimo para científicos de datos.
- **Arquitectura:** los algoritmos son **contenedores Docker** publicados como *assets* en Ocean. Usa dos algoritmos principales:
  1. **Local training** — entrena el modelo en cada dataset por separado (en su propio Ocean provider).
  2. **Aggregation** — combina las salidas locales en un único modelo global.
- **Privacidad:** cada modelo local se **cifra añadiendo ruido aleatorio** antes de salir; los datos nunca abandonan la máquina del proveedor, y el acceso se protege vía blockchain.
- **Librería Python (`feltlabs`):** entry points `felt-train`, `felt-aggregate`, `felt-export`. Trabaja con modelos **scikit-learn** serializados a JSON.
  ```python
  from feltlabs.model import load_model
  model = load_model("final-model.json")
  result = model.predict(data)  # data: (n_samples, n_features)
  ```
- **Limitación a tener en cuenta:** implementa actualmente **un solo enfoque** de FL (no toda la taxonomía), y los modelos base son de scikit-learn (no deep learning out-of-the-box). Esto es relevante para acotar el alcance de la tesis.

---

## 3. Ruta de aprendizaje sugerida (orden temporal)

1. **Cimientos (Capas 0–1):** repasar álgebra lineal + optimización + paradigmas de ML.
2. **Deep Learning (Capa 2):** PyTorch, backprop, una CNN y un Transformer mínimos.
3. **FL clásico (Capa 3):** implementar FedAvg desde cero y luego en **Flower** con datos no-IID.
4. **Privacidad (Capa 4):** añadir Differential Privacy / Secure Aggregation a tu prototipo.
5. **Infraestructura (Capa 5):** entender Ocean + Compute-to-Data; desplegar un ejemplo en Pontus-X.
6. **Integración (Capa 6):** reproducir un experimento con **FELT Labs**; comparar con tu FedAvg/Flower.
7. **Aportación de tesis:** identificar la limitación concreta (p. ej. FELT solo soporta un esquema, o falta soporte DL, o falta robustez frente a clientes no-IID) y proponer/medir una mejora.

---

## 4. Bibliografía base recomendada

- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data* — paper fundacional de FedAvg.
- Li, T. et al. (2020), *Federated Learning: Challenges, Methods, and Future Directions*, IEEE Signal Processing Magazine.
- Martínez Beltrán et al. (2023), *Decentralized Federated Learning: Fundamentals, State of the Art, Frameworks, Trends, and Challenges*, IEEE Communications Surveys & Tutorials — referencia clave para DFL.
- Frontiers in Computer Science (2025), *Deep federated learning: a systematic review* — revisión 2018–2025.
- Gehrer et al. (2024), *A decentralized Federated Learning Approach based on Compute-to-Data and Gaia-X* (EuProGigant), Procedia CIRP — caso directo de tu intersección.
- Documentación oficial: Pontus-X (`docs.pontus-x.eu`), Ocean Protocol, FELT Labs (`docs.feltlabs.ai`).

---

## 5. Glosario rápido

| Término | Definición breve |
|---------|------------------|
| **FL** | Federated Learning: entrenar un modelo sin centralizar los datos. |
| **FedAvg** | Algoritmo base: entrenar local + promediar pesos. |
| **no-IID** | Datos no idénticamente distribuidos entre clientes (el reto central). |
| **CFL / DFL** | FL centralizado / descentralizado. |
| **CtD** | Compute-to-Data: llevar el algoritmo al dato, no al revés. |
| **Gaia-X** | Marco europeo de infraestructura de datos soberana. |
| **Pontus-X** | Ecosistema descentralizado Gaia-X sobre Ocean Protocol. |
| **DLT** | Distributed Ledger Technology (blockchain y similares). |
| **DP** | Differential Privacy. |
