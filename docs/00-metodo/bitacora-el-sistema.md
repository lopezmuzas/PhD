---
title: "Bitácora de experimentos · El sistema"
tags: [metodo, bitacora, doctorado]
status: vivo
---

# Bitácora de experimentos · El sistema

> **Qué es esto.** No es un diario personal ni documentación. Es el mecanismo que hace
> viable un doctorado a tiempo parcial, y el borrador involuntario de la tesis.

---

## Por qué esto importa más de lo que parece

### Razón 1 · Contra el olvido

Un doctorado intermitente no se pierde por falta de horas: se pierde **reconstruyendo
contexto**. Vuelves después de dos semanas ocupado, no recuerdas dónde estabas, y gastas
la primera sesión re-orientándote. Repite eso veinte veces y has perdido un año.

**Cinco minutos al final de cada sesión te devuelven media hora la siguiente.** Es la
inversión con mejor retorno de todo el proceso.

### Razón 2 · Contra racionalizar a posteriori

Escribir la hipótesis **antes** de ver el resultado es la única defensa real contra el
modo de fallo silencioso de la investigación exploratoria: mirar los números y construir
después la explicación que los justifica.

Si la hipótesis está escrita y fechada, no puedes engañarte. Si no lo está, lo harás sin
darte cuenta.

### Razón 3 · Los negativos son la mitad de la tesis

"Probé X y no funcionó, y aquí está por qué" es la mitad de una buena discusión. Y es
exactamente lo que **nadie escribe**, porque en el momento no parece un resultado.

Se pierden si no se anotan el día que ocurren. No se reconstruyen después.

### Razón 4 · Con dos años de bitácora, la tesis casi se escribe sola

Sin ella tendrás cincuenta notebooks y ningún hilo.

---

## La plantilla

Un fichero por experimento, en `docs/04-proyecto/10-bitacora-experimentos/`.
Nombre: `AAAA-MM-DD-descripcion-corta.md`.

```markdown
---
fecha: 2026-08-15
experimento: EXP-001
tema: [ciclo-entrenamiento, mlp, mnist]
estado: cerrado          # abierto | cerrado | abandonado
tiempo: 2h
---

# EXP-001 · Título corto y descriptivo

## Pregunta
Una frase. Qué quiero saber.

## Hipótesis          ← ESCRIBIR ANTES DE EJECUTAR
Qué creo que va a pasar, y por qué. Si me equivoco, eso es lo interesante.

## Diseño
- Datos:
- Modelo:
- Qué varío:
- Qué mantengo fijo:
- Semillas:
- Qué mido:

## Resultado
Los números. Sin interpretar todavía.

## Qué aprendí
Interpretación. Incluye "no aprendí nada" si es el caso.

## Qué me sorprendió
El campo más valioso de toda la plantilla. Si algo no cuadró con la hipótesis,
va aquí. Aunque no sepas explicarlo.

## Qué haría distinto
Errores de diseño detectados a posteriori.

## Siguiente paso
Qué hago mañana. Concreto.

## Reproducir
- commit:
- comando:
- fichero de configuración:
```

---

## Las tres reglas

### 1. La hipótesis se escribe antes de ejecutar

Sin excepciones. Si no sabes qué esperas, el experimento no está diseñado: estás
mirando a ver qué sale, que es legítimo pero se llama exploración y se anota como tal.

### 2. Los negativos se escriben igual que los positivos

Con el mismo detalle. Un experimento que falló y está bien documentado vale más que
tres que funcionaron y no se anotaron.

Marca el estado como `abandonado` si dejas algo a medias, y escribe por qué. Dentro de
un año querrás saberlo.

### 3. Siempre se cierra con el siguiente paso

Aunque sea "releer esto con la cabeza fresca". Es lo que te permite retomar en frío.

---

## La rutina

| Cuándo | Qué | Tiempo |
|---|---|---|
| **Al empezar una sesión** | Leer el "siguiente paso" de la última entrada | 2 min |
| **Antes de ejecutar** | Escribir pregunta, hipótesis y diseño | 10 min |
| **Al cerrar la sesión** | Resultado, qué aprendí, qué me sorprendió, siguiente paso | 5 min |
| **Cada domingo** | Releer la semana. ¿Hay un hilo? ¿Algo que subir al Anexo A? | 15 min |
| **Cada mes** | Índice: una línea por experimento con su conclusión | 20 min |

La revisión semanal es la que convierte entradas sueltas en una línea de investigación.
Sin ella tienes un archivo; con ella, una narrativa.

---

## El índice mensual

Al final de cada mes, una tabla en `INDICE.md`:

| ID | Fecha | Pregunta | Conclusión en una línea | Estado |
|---|---|---|---|---|
| EXP-001 | 15-08 | ¿Sobreajusta 10 muestras? | Sí, en 40 épocas. El bucle funciona | ✅ |
| EXP-002 | 17-08 | ¿Cambia algo con semilla distinta? | Sí, ±3 % de accuracy. **Ojo con esto** | ⚠️ |
| EXP-003 | 20-08 | ¿Aparece doble descenso? | No lo vi. Quizá pocas épocas | ❓ |

Esa tabla, después de un año, es el mapa de todo lo que has aprendido. Y es lo que
enseñas en una reunión con tutores cuando te preguntan por dónde vas.

---

## Qué NO es la bitácora

- **No es documentación.** La documentación explica cómo funciona algo; la bitácora
  registra qué probaste y qué salió.
- **No es un cuaderno de código.** Los notebooks van aparte; la bitácora los enlaza.
- **No tiene que estar bien escrita.** Tiene que estar escrita. Frases sueltas valen.
- **No es para nadie más.** Es para tu yo de dentro de seis meses, que no recordará nada.

---

## La señal de que funciona

Cuando puedas retomar el proyecto después de tres semanas fuera y **estar produciendo en
diez minutos**, la bitácora está haciendo su trabajo.

Cuando tengas que dedicar una sesión entera a recordar dónde estabas, no.
