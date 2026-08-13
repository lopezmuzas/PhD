---
title: "11. Federado sobre data spaces"
tags: [federado, ocean, c2d, laboratorio, indice]
status: esbozo
---

# 11. Federado sobre data spaces

Donde la teoría se convierte en infraestructura. Un pipeline de **aprendizaje
federado** ejecutándose sobre **Compute-to-Data** de `ocean-node`: el eje ③ de
[§1.5 Las tres preguntas](../../01-fundamentos/01-que-es-y-de-donde-viene/1.5-las-tres-preguntas.md) llevado hasta el final, con el código corriendo.

> **Qué NO cubre esta sección.** La teoría del federado y de los data spaces
> está en [§1.5 Las tres preguntas](../../01-fundamentos/01-que-es-y-de-donde-viene/1.5-las-tres-preguntas.md) (eje ③). Por qué el RL offline se complica al federar, en [§1.6 RL offline: el problema](../../01-fundamentos/01-que-es-y-de-donde-viene/1.6-rl-offline-el-problema.md).
> La maquinaria de entrenamiento común, en el [Módulo 3: Cómo se entrena una red](../../01-fundamentos/03-como-se-entrena-una-red/index.md). Aquí solo está la
> fontanería de ejecutar una ronda dentro de la infraestructura de otro.

---

## Por qué cuatro laboratorios y no un tutorial

Un pipeline federado sobre C2D junta cuatro tecnologías a la vez: aprendizaje
federado, Docker, la API de un nodo Ocean y criptografía de firmas. Si lo montas
todo de golpe y falla, no sabes cuál de las cuatro capas falló.

**Cada lab añade exactamente una variable:**

| Lab | Añade | Quita como sospechoso | Tiempo |
|---|---|---|---|
| **0** | Aprendizaje federado | — | 2 min |
| **1** | El patrón de ficheros | la lógica federada | 5 min |
| **2** | Docker | el contrato de ficheros | 15 min |
| **3** | `ocean-node` | la imagen | tu servidor |

El código de entrenamiento es **el mismo en los cuatro**. Solo cambia el
adaptador de almacenamiento. Cuando el Lab 3 falle —y fallará— ya sabrás que el
problema es de infraestructura.

```none
┌─ ORIENTARSE ─────────────────────────────────────────┐
│  11.0  Por qué existe esta sección   ← EMPIEZA AQUÍ  │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ ENTENDER ───────────────────────────────────────────┐
│  11.1  Visión general y arquitectura                  │
│  11.8  Dónde encaja Flower (y por qué aquí no)        │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ EJECUTAR ───────────────────────────────────────────┐
│  11.2  Lab 0 · Federado puro          (sin red)       │
│  11.3  Lab 1 · El patrón de ficheros  (+ ficheros)    │
│  11.4  Lab 2 · El contenedor          (+ Docker)      │
│  11.5  Lab 3 · ocean-node             (+ el nodo)     │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ CONSULTAR ──────────────────────────────────────────┐
│  11.6  El contrato C2D                                │
│  11.7  Troubleshooting                                │
│  11.9  Recursos y glosario                            │
│  11.10 Cómo está montado (código, doc y CI)           │
└──────────────────────────────────────────────────────┘
```

---

## Índice

### Orientarse

| Sección | Qué responde |
|---|---|
| [11.0 Por qué existe esta sección](11.0-por-que-existe-esta-seccion.md) | ¿Qué problema real hay detrás, qué es el código de `labs/` y para qué sirve? **Empieza aquí** |

### Entender

| Sección | Qué responde |
|---|---|
| [11.1 Visión general](11.1-vision-general.md) | ¿Por qué el patrón asíncrono por ficheros, y qué arquitectura lo sostiene? |
| [11.8 Dónde encaja Flower](11.8-donde-encaja-flower.md) | ¿Por qué este laboratorio **no** usa Flower, y cuándo habría que migrar? |

### Ejecutar

| Sección | Qué responde |
|---|---|
| [11.2 Lab 0 · Federado puro](11.2-lab0-federado-puro.md) | ¿Funciona el federado, sin Docker, sin Ocean y sin red? |
| [11.3 Lab 1 · El patrón de ficheros](11.3-lab1-patron-de-ficheros.md) | ¿Funciona con carpetas que imitan `/data`? |
| [11.4 Lab 2 · El contenedor](11.4-lab2-el-contenedor.md) | ¿Cumple la imagen Docker el contrato? |
| [11.5 Lab 3 · ocean-node](11.5-lab3-ocean-node.md) | ¿Funciona contra un nodo real? |

### Consultar

| Sección | Qué responde |
|---|---|
| [11.6 El contrato C2D](11.6-contrato-c2d.md) | Rutas, códigos de estado y firmas. **Verificado contra el código fuente** |
| [11.7 Troubleshooting](11.7-troubleshooting.md) | Los fallos, por orden de frecuencia |
| [11.9 Recursos y glosario](11.9-recursos-y-glosario.md) | Papers, documentación y vocabulario |
| [11.10 Cómo está montado](11.10-como-esta-montado.md) | Por qué el código no vive en el Markdown |

---

## Empezar

Si es tu primera vez, lee [11.0 Por qué existe esta sección](11.0-por-que-existe-esta-seccion.md) antes de
ejecutar nada: explica el problema, qué es `labs/` y por qué hay cuatro
laboratorios en lugar de un tutorial.

```bash
cd labs/federado-ocean-c2d
pip install -r requirements-dev.txt
make test      # 38 tests, < 2 s
make lab0      # los tres ejemplos, en memoria
```

---

## Los tres resultados que justifican la sección

### ① La divergencia, no la accuracy, es la señal

Girando la perilla `alpha` del reparto Dirichlet:

| alpha | % clase 1 por nodo | accuracy | divergencia |
|---|---|---|---|
| 100.0 | 0.48 0.48 0.51 0.50 | 0.7333 | **0.309** |
| 1.0 | 0.22 0.45 0.78 0.53 | 0.7300 | **1.755** |
| 0.3 | 0.44 1.00 0.94 0.23 | 0.7367 | **2.696** |

**La accuracy no se mueve. La divergencia se multiplica por 9.**

La regresión logística es convexa: promediar soluciones locales sigue cayendo
cerca del único mínimo. Una red profunda no tiene esa garantía. Si tu prueba de
concepto usa un modelo lineal y funciona, **todavía no has probado nada** sobre
la red que quieres desplegar.

Es exactamente el mismo problema que [§1.6 RL offline: el problema](../../01-fundamentos/01-que-es-y-de-donde-viene/1.6-rl-offline-el-problema.md) plantea para el RL offline federado:
promediar cosas entrenadas sobre soportes distintos no significa lo que parece.

### ② Tres hospitales: la tabla que justifica el data space

| Escenario | Accuracy |
|---|---|
| Centralizado (techo, ilegal en la práctica) | 0.9217 |
| **Federado — FedAvg ponderado** | **0.9217** |
| Federado — media simple (bug) | 0.9167 |
| Solo local — Hospital A (n=110) | 0.9113 |
| Solo local — Hospital B (n=60) | 0.8613 |
| Solo local — Hospital C (n=30) | **0.6470** |

El hospital pequeño pasa de **0.647 a 0.922**: +27,5 puntos sin mover un solo
dato. Y el federado **alcanza el techo centralizado**.

Si el federado no supera claramente al "solo local", no hay ningún motivo para
montar toda esta infraestructura.

### ③ El contenedor no tiene red

Verificado en el fuente de `ocean-node`: el contenedor arranca sin pila de red
salvo que el operador active `enableNetwork`. **Eso descarta Flower y obliga al
patrón asíncrono por ficheros** ([§11.8 Dónde encaja Flower](11.8-donde-encaja-flower.md)). No es una preferencia arquitectónica.

---

## Estado de verificación

Sé honesto contigo mismo sobre qué está probado y qué no:

| Componente | Estado |
|---|---|
| Dominio, agregación, serialización | **Verificado** — 38 tests |
| Los tres ejemplos | **Ejecutados** — las cifras de arriba son salida real |
| [Lab 1 (patrón de ficheros)](11.3-lab1-patron-de-ficheros.md) | **Ejecutado** |
| [Rutas y códigos de estado C2D](11.6-contrato-c2d.md) | **Verificados** leyendo el fuente de `ocean-node` |
| Esquema de firma | **Verificado** — reproducido en `test_ocean_signature.py` |
| [Lab 2 (Dockerfile y contenedor)](11.4-lab2-el-contenedor.md) | **Sin ejecutar** — no había motor Docker disponible |
| [Lab 3 (ocean/client.py contra un nodo vivo)](11.5-lab3-ocean-node.md) | **Sin ejecutar** — tu servidor será la primera prueba |

---

## Enlaces con el resto del libro

- **[§1.5 Las tres preguntas](../../01-fundamentos/01-que-es-y-de-donde-viene/1.5-las-tres-preguntas.md)** — Las 3 decisiones. Esta sección es el eje ③ en su forma extrema: datos en silos de organizaciones distintas, con gobernanza de por medio.
- **[§1.6 RL offline: el problema](../../01-fundamentos/01-que-es-y-de-donde-viene/1.6-rl-offline-el-problema.md)** — RL Offline. El siguiente escalón: qué pasa cuando lo que se federa es una política aprendida de logs y no un clasificador.
- **[§3.6 Escalar el entrenamiento](../../01-fundamentos/03-como-se-entrena-una-red/3.6-escalar-el-entrenamiento.md)** — Escalar el entrenamiento. FedAvg es paralelismo de datos con sincronización infrecuente y datos no-IID; conviene leerlo antes que esto.
- **[§3.7 Transferencia y fine-tuning](../../01-fundamentos/03-como-se-entrena-una-red/3.7-transferencia-y-finetuning.md)** — Transferencia y fine-tuning. Por qué en modelos grandes se intercambian adaptadores LoRA y no pesos completos.
- **[§3.9 Cómo medir si funciona](../../01-fundamentos/03-como-se-entrena-una-red/3.9-como-medir-si-funciona.md)** — Cómo medir si funciona. La tabla de los tres hospitales es precisamente el tipo de comparación que hay que blindar.

---

## Para completar

- [ ] Ejecutar `make test` y los cuatro labs en orden, anotando dónde falla cada uno la primera vez.
- [ ] Construir la imagen Docker para `amd64` y pasar el test de contrato.
- [ ] Levantar un `ocean-node` propio y ejecutar el Lab 3 de principio a fin.
- [ ] Sustituir el learner NumPy por uno de PyTorch y volver a medir la divergencia del ejemplo 02.
- [ ] Implementar FedProx en `learners/` y comparar con FedAvg bajo `alpha` bajo.
- [ ] Decidir qué parte de este laboratorio entra en la tesis y qué parte se queda como material de aprendizaje.

<!-- nav-start -->

---

← Anterior: [2026-08-08 Puesta a punto](../10-bitacora-experimentos/2026-08-08-inicio-experimentos.md)  
Siguiente: [11.0 Por qué existe esta sección](11.0-por-que-existe-esta-seccion.md) →

<!-- nav-end -->
