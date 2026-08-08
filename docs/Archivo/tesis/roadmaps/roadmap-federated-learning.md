---
title: "Roadmap para aprender Federated Learning (preparación para PhD)"
tags: [tesis, roadmap]
status: borrador
updated: 2026-08-08
---

# Roadmap para aprender Federated Learning (preparación para PhD)

> Documento de trabajo. Todos los enlaces han sido verificados en junio de 2026.
> Objetivo: pasar de cero a estar en condiciones de **leer papers de frontera, reproducir resultados y proponer investigación original** en federated learning (FL).

---

## Nota previa sobre tus referencias

- **CMU 11-868**: ese código corresponde en realidad al curso *"Large Language Model Systems"* de CMU, **no** a un curso de federated learning. Virginia Smith (CMU) es una de las mejores investigadoras de FL, pero **no tiene un curso abierto completo grabado solo de FL**. Lo que sí existe (y es excelente) son sus charlas en seminarios, que incluyo abajo.
- **Charlas de Kairouz / McMahan**: confirmadas y son oro puro. McMahan y Kairouz son coautores de los papers que vas a leer (FedAvg, el survey grande, secure aggregation, DP-FL). Aprender la intuición directamente de los autores acelera mucho la lectura de los papers.

Sustituto recomendado al "curso único completo": el **tutorial de McMahan + Bonawitz + Kairouz** (PPAI'21) más los **Google Workshops on Federated Learning and Analytics**. Juntos cubren el equivalente a un curso académico, dado por los autores.

---

## Cómo usar este roadmap

- Está dividido en **fases**. No saltes la Fase 0 si tu base de optimización/ML no es sólida: en FL todo es optimización distribuida bajo restricciones.
- Para cada fase: **ver el video → leer el paper canónico → implementar algo pequeño**. La implementación es lo que distingue a un candidato a PhD de alguien que solo "ha leído sobre FL".
- Marca casillas `[ ]` a medida que avanzas.
- Tiempo orientativo total: **8–14 semanas** a ritmo de doctorando dedicado.

---

## Prerrequisitos (Fase 0)

Si ya los dominas, sáltalos. Si no, no avances sin esto.

- [ ] **Optimización / SGD**: gradiente, SGD, momentum, convergencia convexa vs no convexa.
- [ ] **Deep learning básico** en PyTorch (entrenar una CNN/MLP de cero).
- [ ] **Probabilidad y estadística**: esperanza, varianza, concentración, IID vs non-IID.
- [ ] **Cálculo de gradientes distribuido / SGD paralelo** (data-parallel).

Recursos puente (opcionales, si necesitas refrescar):
- Optimización para ML: *Convex Optimization* (Boyd & Vandenberghe), libre en https://web.stanford.edu/~boyd/cvxbook/
- SGD distribuido: capítulo introductorio de cualquier curso de "ML with large datasets" (CMU 10-605/10-805): https://10605.github.io/

---

## Fase 1 — Núcleo conceptual de FL

**Meta**: entender qué es FL, por qué existe, el setting cross-device vs cross-silo, y el algoritmo base (FedAvg).

### Videos
- [ ] **Brendan McMahan — "Federated Learning, from Research to Practice"** (mejor charla introductoria de un autor original):
  https://www.youtube.com/watch?v=2KYQlX6tw_M
- [ ] **Brendan McMahan — Guarding User Privacy with FL and Differential Privacy** (visión general + privacidad):
  https://www.youtube.com/watch?v=e5othcNmync

### Papers (lectura obligatoria)
- [ ] **FedAvg — McMahan et al., 2017**, *Communication-Efficient Learning of Deep Networks from Decentralized Data* (el paper fundacional):
  https://arxiv.org/abs/1602.05629 · versión PMLR: https://proceedings.mlr.press/v54/mcmahan17a.html
- [ ] **Survey de orientación — Li et al., 2019**, *Federated Learning: Challenges, Methods, and Future Directions* (mapa del campo, muy legible):
  https://arxiv.org/abs/1908.07873

### Práctica
- [ ] Implementa FedAvg "a mano" sobre MNIST particionado entre, p. ej., 10 clientes (IID y luego non-IID). Sin frameworks todavía: bucle local SGD + promediado de pesos en un servidor simulado.

---

## Fase 2 — La biblia del campo + optimización federada

**Meta**: dominar el documento de referencia del campo y entender por qué FedAvg falla con heterogeneidad y qué se hace al respecto.

### Documento de referencia
- [ ] **Kairouz, McMahan et al., 2021 — *Advances and Open Problems in Federated Learning*** (58+ autores; es *el* documento que define el campo y sus problemas abiertos — léelo por secciones, no de una sentada):
  https://arxiv.org/abs/1912.04977

### Videos sobre heterogeneidad
- [ ] **Virginia Smith — *On Heterogeneity in Federated Settings*** (Stanford MLSys Seminar, Ep. 3 — su mejor charla abierta sobre el problema central de FL):
  https://www.youtube.com/watch?v=laCyJICLyWg
- [ ] **Virginia Smith — *Fairness and Robustness in Federated Learning***:
  https://www.youtube.com/watch?v=vv8v0fdWBUE

### Papers (optimización / heterogeneidad)
- [ ] **FedProx — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith, 2020**, *Federated Optimization in Heterogeneous Networks*:
  https://arxiv.org/abs/1812.06127
- [ ] **SCAFFOLD — Karimireddy et al., 2020**, *Stochastic Controlled Averaging for Federated Learning* (corrige el "client drift"):
  https://arxiv.org/abs/1910.06378
- [ ] **Adaptive Federated Optimization — Reddi et al., 2021** (FedAdam/FedYogi; optimizadores adaptativos en el servidor):
  https://arxiv.org/abs/2003.00295

### Práctica
- [ ] Reproduce la degradación de FedAvg con datos non-IID muy desbalanceados y compara contra FedProx en tu implementación.

---

## Fase 3 — Privacidad: differential privacy y secure aggregation

**Meta**: FL **no es privado por sí solo** (los gradientes filtran información). Esta fase es central si tu PhD va por privacidad.

### Videos (de los autores)
- [ ] **Tutorial PPAI'21 — McMahan, Bonawitz, Kairouz**, *Privacy and Federated Learning: Principles, Techniques and Emerging Frontiers* (el mejor recurso único, equivale a un mini-curso):
  https://www.youtube.com/watch?v=prQI5OT_wzk
- [ ] **Brendan McMahan — *Federated Learning with Formal User-level Differential Privacy Guarantees***:
  https://www.youtube.com/watch?v=97Ybvqf36q8

### Papers
- [ ] **Secure Aggregation — Bonawitz et al., 2017**, *Practical Secure Aggregation for Privacy-Preserving Machine Learning*:
  https://eprint.iacr.org/2017/281 (también CCS'17)
- [ ] **DP-FedAvg — McMahan, Ramage, Talwar, Zhang, 2018**, *Learning Differentially Private Recurrent Language Models*:
  https://arxiv.org/abs/1710.06963
- [ ] **Deep Gradient Leakage — Zhu, Liu, Han, 2019** (por qué hace falta privacidad: reconstrucción de datos desde gradientes):
  https://arxiv.org/abs/1906.08935

### Lectura de apoyo sobre DP
- [ ] **Dwork & Roth — *The Algorithmic Foundations of Differential Privacy*** (monografía de referencia, libre):
  https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf

---

## Fase 4 — Robustez, ataques, fairness y personalización

**Meta**: panorama de los subcampos donde hoy se publica más. Aquí es donde probablemente nazca tu tema de tesis.

### Papers por subárea
- **Ataques / robustez (poisoning, backdoors)**
  - [ ] *How To Backdoor Federated Learning* — Bagdasaryan et al., 2020: https://arxiv.org/abs/1807.00459
  - [ ] *Can You Really Backdoor Federated Learning?* — Sun, Kairouz, Suresh, McMahan, 2019: https://arxiv.org/abs/1911.07963
- **Personalización**
  - [ ] *Ditto: Fair and Robust FL Through Personalization* — Li, Hu, Beirami, Smith, 2021: https://arxiv.org/abs/2012.04221
  - [ ] *Per-FedAvg (meta-learning)* — Fallah, Mokhtari, Ozdaglar, 2020: https://arxiv.org/abs/2002.07948
- **Fairness**
  - [ ] *Agnostic Federated Learning* — Mohri, Sivek, Suresh, 2019: https://arxiv.org/abs/1902.00146
  - [ ] *Fair Resource Allocation in FL (q-FFL)* — Li, Sanjabi, Beirami, Smith, 2020: https://arxiv.org/abs/1905.10497

### Video
- [ ] **Google Workshop on Federated Learning and Analytics (Day 2)** — panorama amplio de varios autores:
  https://www.youtube.com/watch?v=YucSFKU_L1I

---

## Fase 5 — Sistemas y frameworks (manos a la obra)

**Meta**: dejar de simular y trabajar con frameworks reales, como en investigación seria.

### Papers de sistemas
- [ ] *Towards Federated Learning at Scale: System Design* — Bonawitz et al., 2019 (cómo Google lo despliega de verdad):
  https://arxiv.org/abs/1902.01046
- [ ] *Flower: A Friendly Federated Learning Research Framework* — Beutel et al., 2020:
  https://arxiv.org/abs/2007.14390

### Frameworks (elige uno principal + uno secundario)
- [ ] **Flower** — el más usado hoy en investigación, agnóstico al backend (PyTorch/TF/JAX). Empieza aquí:
  https://flower.ai/ · docs: https://flower.ai/docs/
- [ ] **TensorFlow Federated (TFF)** — de Google, ideal para reproducir papers de DP-FL:
  https://www.tensorflow.org/federated
- [ ] **FedML** — amplio, soporta cross-device y cross-silo:
  https://github.com/FedML-AI/FedML
- [ ] **FedScale** — benchmarking realista a escala (heterogeneidad de sistemas):
  https://github.com/SymbioticLab/FedScale

### Práctica (proyecto de portfolio para tu aplicación a PhD)
- [ ] Reimplementa **FedAvg + FedProx + un método de DP** en Flower, sobre un dataset non-IID (p. ej. CIFAR-10 particionado por Dirichlet), y escribe un mini-reporte comparando precisión vs. coste de comunicación vs. presupuesto de privacidad (ε). Esto es exactamente el tipo de artefacto que un comité de admisión valora.

---

## Fase 6 — Fronteras de investigación (2024–2026)

**Meta**: saber qué se está investigando *ahora* para identificar huecos y formular preguntas de tesis.

- **FL + Large Language Models / Foundation Models** (el tema más caliente):
  - [ ] *The Future of LLM Pre-training is Federated* — Sani et al., 2024: https://arxiv.org/abs/2405.10853
  - [ ] *Federated LoRA / fine-tuning eficiente en comunicación* — línea de trabajo de Kuo, Smith et al. (FLASC): ver https://www.cs.cmu.edu/~smithv/
- **One-shot FL, FL descentralizado (peer-to-peer), FL asíncrono**: busca papers recientes en NeurIPS/ICML/ICLR.
- [ ] **Simons Institute — Programa *Federated and Collaborative Learning* (primavera 2026)**, coorganizado por Virginia Smith. Producirá charlas grabadas de altísimo nivel; síguelo:
  https://simons.berkeley.edu/programs/federated-collaborative-learning

---

## Base de conocimiento: tabla resumen de papers canónicos

| # | Paper | Autores (año) | Tema | Enlace |
|---|-------|---------------|------|--------|
| 1 | Communication-Efficient Learning (FedAvg) | McMahan et al. (2017) | Algoritmo base | https://arxiv.org/abs/1602.05629 |
| 2 | Advances and Open Problems in FL | Kairouz, McMahan et al. (2021) | Referencia del campo | https://arxiv.org/abs/1912.04977 |
| 3 | FL: Challenges, Methods, Future Directions | Li et al. (2019) | Survey legible | https://arxiv.org/abs/1908.07873 |
| 4 | FedProx | Li et al. (2020) | Heterogeneidad | https://arxiv.org/abs/1812.06127 |
| 5 | SCAFFOLD | Karimireddy et al. (2020) | Client drift | https://arxiv.org/abs/1910.06378 |
| 6 | Adaptive Federated Optimization | Reddi et al. (2021) | Optimizadores | https://arxiv.org/abs/2003.00295 |
| 7 | Practical Secure Aggregation | Bonawitz et al. (2017) | Privacidad | https://eprint.iacr.org/2017/281 |
| 8 | Learning DP Recurrent LMs (DP-FedAvg) | McMahan et al. (2018) | DP + FL | https://arxiv.org/abs/1710.06963 |
| 9 | Deep Leakage from Gradients | Zhu et al. (2019) | Ataque/privacidad | https://arxiv.org/abs/1906.08935 |
| 10 | FL at Scale: System Design | Bonawitz et al. (2019) | Sistemas | https://arxiv.org/abs/1902.01046 |
| 11 | Flower framework | Beutel et al. (2020) | Herramienta | https://arxiv.org/abs/2007.14390 |
| 12 | Ditto (fairness + personalización) | Li et al. (2021) | Personalización | https://arxiv.org/abs/2012.04221 |
| 13 | How to Backdoor FL | Bagdasaryan et al. (2020) | Ataques | https://arxiv.org/abs/1807.00459 |
| 14 | Federated LLM pre-training | Sani et al. (2024) | Frontera | https://arxiv.org/abs/2405.10853 |

---

## Videos clave (lista consolidada)

| Tipo | Título / autor | Enlace |
|------|----------------|--------|
| Intro | McMahan — FL from Research to Practice | https://www.youtube.com/watch?v=2KYQlX6tw_M |
| Intro+privacidad | McMahan — Guarding User Privacy with FL & DP | https://www.youtube.com/watch?v=e5othcNmync |
| Heterogeneidad | V. Smith — On Heterogeneity in Federated Settings (Stanford MLSys) | https://www.youtube.com/watch?v=laCyJICLyWg |
| Fairness/robustez | V. Smith — Fairness and Robustness in FL | https://www.youtube.com/watch?v=vv8v0fdWBUE |
| Tutorial completo | McMahan/Bonawitz/Kairouz — Privacy & FL (PPAI'21) | https://www.youtube.com/watch?v=prQI5OT_wzk |
| DP user-level | McMahan — Formal User-level DP Guarantees | https://www.youtube.com/watch?v=97Ybvqf36q8 |
| Workshop | Google Workshop on FL & Analytics (Day 2) | https://www.youtube.com/watch?v=YucSFKU_L1I |

> Consejo: en YouTube busca también el canal de **Google TechTalks**, que publica los *Workshops on Federated Learning and Analytics* completos cada año — es lo más cercano a un curso anual del campo dado por los protagonistas.

---

## Dónde se publica FL (para orientar tu PhD)

- **Conferencias ML**: NeurIPS, ICML, ICLR, AISTATS.
- **Sistemas/ML systems**: MLSys.
- **Privacidad/seguridad**: IEEE S&P, CCS, USENIX Security, SaTML.
- **Revista**: TMLR (Transactions on Machine Learning Research) — muchos papers de FL recientes salen aquí.

---

## Investigadores / grupos a seguir

- **H. Brendan McMahan** (Google) — FedAvg, DP-FL: https://research.google/people/h-brendan-mcmahan/
- **Peter Kairouz** (Google) — privacidad, DP, secure aggregation.
- **Virginia Smith** (CMU) — heterogeneidad, fairness, personalización: https://www.cs.cmu.edu/~smithv/
- **Tian Li** (FedProx, Ditto), **Gauri Joshi** (CMU), **Aurélien Bellet** (Inria, FL descentralizado).

Pon sus perfiles de Google Scholar en alertas para no perderte papers nuevos.

---

## Checklist de "estoy listo para aplicar a un PhD en FL"

- [ ] Puedo explicar por qué FedAvg falla con datos non-IID y nombrar 3 soluciones.
- [ ] Sé por qué FL no es privado sin DP/secure aggregation y puedo describir ambos.
- [ ] He implementado y comparado ≥3 algoritmos en un framework real (Flower/TFF).
- [ ] He leído el survey de Kairouz/McMahan completo y puedo señalar 2–3 problemas abiertos que me interesan.
- [ ] Tengo un mini-proyecto reproducible en GitHub con un README claro.
- [ ] Puedo identificar el grupo/asesor cuyo trabajo se alinea con mi pregunta de investigación.

---

*Roadmap creado en junio de 2026. Verifica que los enlaces de los seminarios sigan activos; los repositorios y frameworks evolucionan, consulta siempre su documentación oficial más reciente.*
