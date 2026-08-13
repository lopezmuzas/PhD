---
title: "Anexo B — Línea temporal"
tags: [historia]
status: esbozo
---

# Anexo B — Línea temporal

> De la neurona lógica de 1943 al entrenamiento descentralizado de hoy.

```none
┌─ RAÍCES ──────────────────────────────────────────────────────────────┐
1943  McCulloch & Pitts ....... la neurona como interruptor lógico
1949  Hebb .................... "lo que dispara junto, se conecta junto"
1952  Hodgkin & Huxley ........ las ecuaciones del disparo neuronal
1958  Rosenblatt .............. PERCEPTRÓN
1959  Hubel & Wiesel .......... campos receptivos (germen de la CNN)
1969  Minsky & Papert ......... XOR ⛔ PRIMER INVIERNO
└───────────────────────────────────────────────────────────────────────┘
┌─ LA MAQUINARIA ───────────────────────────────────────────────────────┐
1970  Linnainmaa .............. autodiff en modo reverso
1974  Werbos .................. backprop aplicada a redes
1980  Fukushima ............... Neocognitron (CNN sin backprop)
1986  Rumelhart/Hinton/Williams backprop popularizada
1989  LeCun ................... LeNet lee dígitos
1997  Hochreiter/Schmidhuber .. LSTM
1998  ......................... ⛔ SEGUNDO INVIERNO (ganan SVM y boosting)
2006  Hinton .................. el deshielo (pre-entrenamiento por capas)
2006  Dwork ................... privacidad diferencial
2011  Glorot .................. ReLU
└───────────────────────────────────────────────────────────────────────┘
┌─ LA ERA MODERNA ──────────────────────────────────────────────────────┐
2012  AlexNet ★ ............... el punto de inflexión real
  ├── visión:      2014 VGG/Inception · 2015 ResNet ★ + BatchNorm
  │                2019 EfficientNet · 2022 ConvNeXt
  ├── generativo:  2013 VAE · 2014 GAN · 2020 difusión (DDPM)
  │                2021 CLIP · 2022 Stable Diffusion
  ├── secuencias:  2014 seq2seq + atención · 2017 TRANSFORMER ★
  │                2018 BERT y GPT · 2020 ViT
  ├── grafos:      2016 GCN · 2018 GAT · 2021 Geometric Deep Learning
  ├── escala:      2014 Adam · 2020 leyes de escalado · 2022 Chinchilla
  │                2022 InstructGPT / RLHF
  └── federado:    2016 FedAvg ★ · 2016 DP-SGD · 2017 agregación segura
                   2018 FedProx · 2020 SCAFFOLD
                   2019 "Open Problems in Federated Learning" (Kairouz et al.)
└───────────────────────────────────────────────────────────────────────┘
┌─ 2023 EN ADELANTE ────────────────────────────────────────────────────┐
2023  Mamba (vuelve la recurrencia, paralelizable) · DiT (difusión sobre
      Transformer) · DPO · flow matching · LLaMA/Mixtral (pesos abiertos)
      · DiLoCo ★ (FedAvg para pre-entrenar LLM: ~500× menos comunicación)
2024  Nobel de Física (Hopfield, Hinton) · Nobel de Química (AlphaFold)
      · Llama-3 405B · o1 (pensar antes de responder) · Titans (memoria)
      · OpenDiLoCo · INTELLECT-1 ★ (10B entrenado con recursos dispersos)
2025  DeepSeek-R1 (razonamiento por RL, abierto y documentado) · GRPO y
      recompensa verificable · leyes de escalado para DiLoCo
      · INTELLECT-2 (RL descentralizado global) · Nested Learning / HOPE
2026  Modelos de mundo (JEPA, Genie) · aprendizaje continuo: primeros
      prototipos fiables · híbridos atención+recurrencia+memoria
      · razonamiento destilado en dispositivo · DiLoCoX (clústeres lentos)
      · la energía eléctrica como límite de planificación
  ▼
2027  HORIZONTE (líneas activas, sin resultado consolidado)
      · agentes de largo plazo con memoria persistente
      · aprender sin olvidar en producción
      · robótica sobre modelos de mundo
      · entrenamiento descentralizado como alternativa real al datacenter
      · data spaces operativos con gobernanza ejecutable
└───────────────────────────────────────────────────────────────────────┘
```

## Los tres patrones que se repiten

1. **Cada arquitectura nace de un muro.** No de una idea genial en el vacío.
2. **Las ideas llegan antes que el hardware.** Backprop es de 1970 y no fue útil
   hasta 1986; las CNN son de 1980 y no ganaron hasta 2012.
3. **Lo viejo vuelve.** La recurrencia murió en 2017 y volvió en 2023. El
   promediado federado de 2016 es hoy la base para entrenar LLM.

<!-- nav-start -->

---

← Anterior: [A. Lo que nadie sabe todavía](anexo-a-preguntas-abiertas.md)  
Siguiente: [C. Glosario técnico](anexo-c-glosario-tecnico.md) →

<!-- nav-end -->
