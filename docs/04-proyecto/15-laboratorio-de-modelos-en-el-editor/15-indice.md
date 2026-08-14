---
title: Laboratorio de modelos en el editor
tags:
  - llm
  - instrumentacion
  - herramientas
status: esbozo
---

# 15. Laboratorio de modelos en el editor (LLM lab)

> Estado: montaje iniciado el 13 de agosto de 2026.
> Infraestructura de exploración, no compromiso de tesis.
> Los precios e identificadores de modelo caducan rápido: ver [mantenimiento](15.7-como-esta-montado.md#mantenimiento-lo-que-caduca).

Esta sección es el reverso del resto del libro. En las anteriores el objeto de
estudio es el modelo; aquí lo es **el sistema que lo usa**. La pregunta de fondo
es la misma que abre la Fase 0: no basta con que funcione, hay que poder mirar
dentro.

## Las cuatro fases

```
  Fase 0          Fase 1           Fase 2            Fase 3           Fase 4
  ┌──────┐       ┌──────┐        ┌──────┐          ┌──────┐        ┌──────┐
  │ Usar │  ──▶  │ Ver  │  ──▶   │Medir │   ──▶    │Decidir│  ──▶  │Construir│
  └──────┘       └──────┘        └──────┘          └──────┘        └──────┘
  modelos        proxy           banco de          router          modelo
  en el          espía           pruebas           propio          propio
  editor
   15.1           15.3            15.5              15.4            15.6
```

El orden de montaje **no** es el orden en que surgieron las ideas. El proxy va
en segundo lugar porque es lo que más enseña por hora invertida, y porque el
router no se puede evaluar sin sus registros.

## Bloque A — Poner modelos a trabajar

| Página | Qué contiene |
|---|---|
| [15.1](15.1-el-editor-y-su-configuracion.md) | Dónde vive la configuración, convención de nombres, 31 modelos por nivel de potencia, las tres cuotas gratuitas |
| [15.2](15.2-conectar-un-modelo-propio.md) | Servir cualquier modelo propio por el protocolo de OpenAI: shim, vLLM, Ollama |

## Bloque B — Mirar dentro

| Página | Qué contiene |
|---|---|
| [15.3](15.3-lab1-el-proxy-espia.md) | **Lab 1.** Interceptar el tráfico y descubrir qué contexto inyecta el editor |
| [15.5](15.5-lab3-el-banco-de-pruebas.md) | **Lab 3.** El mismo prompt contra N modelos: latencia, coste y calidad juzgada a mano |

## Bloque C — Tomar decisiones

| Página | Qué contiene |
|---|---|
| [15.4](15.4-lab2-el-router-propio.md) | **Lab 2.** Política explícita de enrutado barato→caro, y cuánto ahorra |
| [15.6](15.6-lab4-un-modelo-de-lenguaje-propio.md) | **Lab 4.** Un transformer por caracteres entrenado con el Quijote |

## Bloque D — Cómo está montado

| Página | Qué contiene |
|---|---|
| [15.7](15.7-como-esta-montado.md) | Infraestructura, árbol de directorios, cuaderno de laboratorio, mantenimiento |
| [15.8](15.8-recursos-y-referencias.md) | Recursos, referencias comentadas, papers, vídeos y fuentes |


## El hallazgo que justifica la sección

Al ordenar los 31 modelos por potencia en vez de por precio, el escalón 4/5
queda ocupado por **dos gratuitos y tres de pago**. Agrupados por coste no se
ve; agrupados por capacidad, salta a la vista que se puede trabajar en ese nivel
sin pagar nada mientras se respeten las cuotas.

Ese es el patrón que esta sección persigue en todo: la ordenación de los datos
cambia lo que se puede concluir de ellos.

## Enlaces cruzados

- **→ 1.6** — aprender la política de enrutado en vez de escribirla a mano es un
  bandido contextual. Ahí está el puente con RL offline, y es la pregunta abierta
  más prometedora de esta sección.
- **→ 11** — el proxy espía es la misma idea de instrumentación que mirar el
  tráfico de un contrato C2D: interponerse en el cable y registrar lo que pasa.
- **→ 3** — el modelo por caracteres de 15.6 reutiliza atención escalada por
  √dk, conexiones residuales y normalización por capas. Es el primer sitio del
  libro donde se entrenan de verdad.

## Para completar

- [ ] Ejecutar el Lab 1 y anotar el tamaño real del contexto que inyecta el editor
- [ ] Rellenar a mano la columna de calidad del Lab 3 sobre tareas del `dl-lab`
- [ ] Medir el ahorro real del router contra el contrafactual «todo a 5/5»
- [ ] Entrenar el modelo del Lab 4 y guardar la película de la pérdida
- [ ] Decidir si la política de enrutado se aprende (anexo de preguntas abiertas)

<!-- nav-start -->

---

← Anterior: [14.7-recursos.md](../14-validacion-por-silo-retenido/14.7-recursos.md)  
Siguiente: [15.1-el-editor-y-su-configuracion.md](15.1-el-editor-y-su-configuracion.md) →

<!-- nav-end -->
