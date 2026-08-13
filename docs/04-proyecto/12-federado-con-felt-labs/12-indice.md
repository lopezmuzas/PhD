---
title: "12. Federado con FELT Labs"
tags: [federado, felt, ocean, c2d, laboratorio, indice]
status: esbozo
---

# 12. Federado con FELT Labs

Un segundo diseño para el mismo problema que la sección 11: aprendizaje
federado sobre Compute-to-Data. Aquí la pieza nueva es un **doble ciego**: ni tú
ni el agregador llegáis a ver un modelo entrenado sobre un único dataset.

> **Requisito previo.** Esta sección asume → [11.0](../11-federado-sobre-data-spaces/11.0-por-que-existe-esta-seccion.md):
> qué es C2D y por qué el contenedor no tiene red. Si no lo has leído, empieza
> por ahí.

```none
┌─ ORIENTARSE ─────────────────────────────────────────┐
│  12.0  Por qué existe esta sección   ← EMPIEZA AQUÍ  │
│  12.1  Qué es FELT Labs                               │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ ENTENDER ───────────────────────────────────────────┐
│  12.2  El protocolo de doble ciego   ★ la pieza clave │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ EJECUTAR ───────────────────────────────────────────┐
│  12.3  Lab 0 · El protocolo en memoria                │
│  12.4  Lab 1 · Dos algoritmos, dos jobs               │
│  12.5  Lab 2 · El contenedor          (sin ejecutar)  │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ DECIDIR ────────────────────────────────────────────┐
│  12.6  FELT frente a la sección 11                    │
│  12.7  Estado del proyecto y riesgos                  │
│  12.8  Recursos y glosario                            │
└──────────────────────────────────────────────────────┘
```

---

## Índice

| Sección | Qué responde |
|---|---|
| [12.0 Por qué existe esta sección](12.0-por-que-existe-esta-seccion.md) | ¿Qué añade frente a la sección 11? **Empieza aquí** |
| [12.1 Qué es FELT Labs](12.1-que-es-felt-labs.md) | Arquitectura: dos algoritmos, un contenedor, K+1 jobs |
| [12.2 El protocolo de doble ciego](12.2-el-protocolo-de-doble-ciego.md) | Cómo el ruido lineal sobrevive al promedio. **La página central** |
| [12.3 Lab 0 · El protocolo en memoria](12.3-lab0-el-protocolo-en-memoria.md) | Los tres ejemplos, sin Docker ni Ocean |
| [12.4 Lab 1 · Dos algoritmos, dos jobs](12.4-lab1-dos-algoritmos.md) | El mismo protocolo, comunicado por ficheros |
| [12.5 Lab 2 · El contenedor](12.5-lab2-el-contenedor.md) | La imagen Docker y su contrato (sin ejecutar) |
| [12.6 FELT frente a la sección 11](12.6-felt-frente-a-la-seccion-11.md) | Tabla comparativa y cuándo usar cada diseño |
| [12.7 Estado del proyecto y riesgos](12.7-estado-del-proyecto.md) | ¿Es una dependencia viable para una tesis? |
| [12.8 Recursos y glosario](12.8-recursos-y-glosario.md) | Papers, documentación, vocabulario |

---

## Empezar

```bash
cd labs/felt-c2d
pip install -r requirements-dev.txt
make test      # 24 tests, < 2 s
make lab0      # los tres ejemplos
```

---

## El resultado que justifica la sección

`ex02`, midiendo qué ve realmente cada actor tras descifrar:

```none
   nodo 0: |w_visto - w_real| = [353.4, 564.0, 240.4, 403.2, ...]
```

Pesos reales de orden 1, error de orden 100–1000. El agregador descifra y
promedia sin la menor idea de qué representaban esos números. Y en la fase 3,
tu máquina recupera el global con un error de $2\times10^{-13}$: la cancelación
del ruido es exacta, no aproximada.

---

## Estado de verificación

| Componente | Estado |
|---|---|
| Protocolo (cegado, cifrado, agregación) | **Verificado** — 24 tests |
| Los tres ejemplos del Lab 0 | **Ejecutados** — las cifras de esta sección son salida real |
| Lab 1 (dos algoritmos por ficheros) | **Ejecutado** |
| Contrato de rutas C2D | **Verificado** contra `feltc2d/ocean.py` y sus tests |
| `Dockerfile` del Lab 2 | **Sin ejecutar** — sin motor Docker disponible |
| Compatibilidad con `feltlabs` original | **No verificada** — ver → [12.7](12.7-estado-del-proyecto.md) |
| Contra un `ocean-node` real | **No probado** |

---

## Enlaces con el resto del libro

- **→ 11** — El otro diseño para el mismo problema. Léela primero.
- **→ 1.5** — Eje ③. Esta sección añade la variante con privacidad de los
  modelos intermedios.
- **→ 1.6** — RL Offline. La pregunta de cómo promediar cosas cegadas conecta
  con la de cómo promediar funciones $Q$ sobre soportes distintos.
- **→ 3.6** — Escalar el entrenamiento. El coste en jobs de → 12.6 es la versión
  federada del coste de comunicación de un `all-reduce`.
- **→ 3.9** — Cómo medir si funciona. Publicar por digest (→ 12.5) es la misma
  disciplina de trazabilidad que exige esa página.

---

## Para completar

- [ ] Ejecutar `make test` y `make lab0`, y comparar las cifras con esta página.
- [ ] Decidir si incorporas el cegado a la sección 11 (→ 12.6).
- [ ] Completar las verificaciones pendientes de → 12.7 antes de citar FELT en
      la tesis como algo más que trabajo relacionado.

<!-- nav-start -->

---

← Anterior: [11.10 Cómo está montado](../11-federado-sobre-data-spaces/11.10-como-esta-montado.md)  
Siguiente: [12.0 Por qué existe esta sección](12.0-por-que-existe-esta-seccion.md) →

<!-- nav-end -->
