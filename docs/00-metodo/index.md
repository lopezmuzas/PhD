---
title: "0. Método"
tags: [metodo, indice]
status: vivo
---

# 0. Método

> **Qué es esta parte.** Cómo se trabaja, no qué se estudia. Es lo único de este libro
> que sigue siendo válido aunque cambie el tema de la tesis.

Va primera porque es lo que se lee antes de nada, y porque es lo que hace que el resto
sirva de algo. Un doctorado a tiempo parcial no se pierde por falta de conocimiento: se
pierde por perder el hilo.

---

## El ciclo

```none
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   NOTEBOOKS N00→N30 ───▶ qué construyo ahora              │
        │                                              │
        ▼                                              │
   BITÁCORA ────────────▶ qué salió, y qué me sorprendió
        │                                              │
        ▼                                              │
   ANEXO A ─────────────▶ qué queda por saber          │
        │                                              │
        └──────────────────────────────────────────────┘
```

Tres documentos, un bucle. Todo lo demás del libro es material de consulta que alimenta
este ciclo.

| Documento | Responde a | Cuándo se toca |
|---|---|---|
| [Itinerario](itinerario.md) | ¿Por qué esto y en qué orden? | Una vez, al empezar |
| [Lista de notebooks](../04-proyecto/09-indice-notebooks/notebooks.md) | ¿Qué construyo ahora? | Cada sesión |
| [La bitácora](bitacora-el-sistema.md) | ¿Qué probé y qué salió? | Cada sesión |
| [Anexo A](../05-anexos/anexo-a-preguntas-abiertas.md) | ¿Qué no sabemos? | Cada vez que aparece una pregunta, y al revisar el mes |

---

## Dónde vive cada cosa

```none
docs/
├── 00-metodo/                          ← se LEE
│   ├── itinerario.md                   principios, dataset, mapa
│   └── bitacora-el-sistema.md          el sistema de registro
│
├── 01-fundamentos/ … 03-data-spaces/   ← lo que se estudia
│
├── 04-proyecto/
│   ├── 09-indice-notebooks/
│   │   └── notebooks.md                ← se RECORRE (N00 → N30)
│   └── 10-bitacora-experimentos/       ← se ESCRIBE
│       ├── INDICE.md
│       ├── _plantilla.md
│       └── 2026-08/ …
│
└── 05-anexos/
    └── anexo-a-preguntas-abiertas.md
```

**Tres modos de uso, no tres temas.** El itinerario se **lee** una vez; la lista de
notebooks se **recorre** del N00 al N30; la bitácora se **escribe** cada sesión.

**La distinción que importa:** aquí está el *sistema* de bitácora; las *entradas* viven
en el proyecto, con los experimentos que documentan.

---

## Las cinco reglas

Todo lo demás de esta parte se deriva de estas cinco.

### 1. La hipótesis se escribe antes de ejecutar

Es la única defensa real contra construir la explicación después de ver el resultado.
Si no sabes qué esperas, no estás experimentando: estás explorando, que es legítimo pero
se anota como tal.

### 2. Los negativos se escriben igual que los positivos

"Probé X y no funcionó, y aquí está por qué" es la mitad de una buena discusión de tesis,
y la parte que nadie escribe. Se pierden si no se anotan el día que ocurren.

### 3. Cada sesión cierra con el siguiente paso

Aunque sea "releer esto con la cabeza fresca". Es lo que te permite retomar en frío
después de tres semanas.

### 4. Romper enseña más que construir

Un modelo que funciona no te dice cuál de tus veinte decisiones fue la buena. Uno que
rompes a propósito te enseña el mecanismo, porque has aislado la causa.

### 5. Primero instrumentos, después modelos

Subir a arquitecturas más grandes sin poder ver qué pasa dentro no acelera nada: solo
hace los fallos más caros de depurar.

---

## Cómo empezar hoy

Si es tu primer día con este sistema, dos horas:

1. Leer el [Itinerario](itinerario.md), partes I y II.
2. Copiar `_plantilla.md` a `2026-XX/` y escribir la primera entrada.
3. Abrir la [lista de notebooks](../04-proyecto/09-indice-notebooks/notebooks.md) por
   **N00**, copiar preámbulo + bloque, y montar el arnés.
4. Cerrar la entrada con el siguiente paso.

**Lo que cuesta arrancar no son los experimentos: es el hábito.** Se coge con lo
trivial, no esperando a "tener algo serio".

---

## La señal de que funciona

Cuando puedas volver después de tres semanas fuera y **estar produciendo en diez
minutos**, el sistema está haciendo su trabajo.

Cuando tengas que dedicar una sesión entera a recordar dónde estabas, no.
