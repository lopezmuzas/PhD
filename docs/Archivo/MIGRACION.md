# Reorganización de `docs/` — plan y mapeo

## El principio

Tenías **dos esquemas de organización compitiendo** en el mismo árbol: uno por
situación del lector (Diátaxis: aprendizaje / guías / referencia / recursos) y
otro por tema (`algebra-lineal/`, `federated-rl/`, `dataspaces/`). El resultado
era que `aprendizaje/federated-rl/` acumulaba 25 ficheros de cinco naturalezas
distintas: teoría, rutas de estudio, roadmaps de tesis, estado del arte y
metodología bibliográfica.

La solución **no** es sustituir un esquema por el otro, sino aplicarlos en
niveles distintos:

```
NIVEL 1 (carpetas raíz)   → por SITUACIÓN del lector   ← tu esquema Diátaxis, se conserva
NIVEL 2 (dentro de cada)  → por TEMA                   ← el índice maestro
```

Y separar una situación que estaba mezclada con las demás: **`tesis/`**.
Explicar qué es FedAvg es aprendizaje; decidir qué hueco de FedAvg vas a
atacar es tesis. Estaban en la misma carpeta.

---

## Estructura resultante

```none
docs/
├── index.md                        portada con tarjetas
├── aprendizaje/                    TEORÍA — ordenada según el índice maestro
│   ├── 00-mapa/                      Parte 0
│   ├── 01-fundamentos/               Partes 1–2
│   ├── 02-redes-neuronales/          Partes 3–9
│   ├── 03-refuerzo/                  Parte 10
│   ├── 04-federado-y-distribuido/    Parte 11
│   └── _archivo/                     versiones superadas (fuera del nav)
├── tesis/                          NUEVO — planificación e investigación
│   ├── roadmaps/
│   ├── estado-del-arte/
│   ├── metodologia/
│   └── master-plan-ia-soberana.md
├── guias/                          hacer algo concreto
├── referencia/                     consultar un dato
│   └── dataspaces/
├── recursos/                       material externo
│   └── ruta-frl/                     los 6 módulos de recursos
├── bitacora/
├── meta/
└── assets/
```

---

## Mapeo completo

### → `aprendizaje/00-mapa/` (Parte 0 del índice)

| Antes | Ahora |
|---|---|
| *(nuevo)* | `indice-maestro.md` |
| `aprendizaje/federated-rl/modelo_mental_aprendizaje_federado_RL.md` | `modelo-mental-dl-a-frl.md` |
| `aprendizaje/federated-rl/paradigmas_aprendizaje_federado_data_spaces-v2.md` | `paradigmas-fl-sobre-data-spaces.md` |

### → `aprendizaje/01-fundamentos/` (Partes 1–2)

| Antes | Ahora |
|---|---|
| `recursos/neurona-biologica-a-redes-neuronales.md` | `neurona-biologica.md` |
| `aprendizaje/sistemas_emergentes_y_redes_neuronales.md` | `sistemas-emergentes.md` |
| `aprendizaje/federated-rl/0-prerrequisitos_RL.md` | `prerrequisitos-matematicos.md` |

### → `aprendizaje/02-redes-neuronales/` (Partes 3–9)

| Antes | Ahora |
|---|---|
| `aprendizaje/biblia-redes-neuronales-mlp-a-transformers.md` | `biblia-mlp-a-transformers.md` |
| `aprendizaje/algebra-lineal/backpropagation_de_menos_a_mas_algebra_lineal.md` | `backpropagation-de-menos-a-mas.md` |
| `aprendizaje/batch-normalization.md` | `batch-normalization.md` |
| `aprendizaje/algebra-lineal/deep_learning_antes_de_attention_algebra_lineal.md` | `dl-antes-de-attention.md` |
| `aprendizaje/algebra-lineal/attention_is_all_you_need_algebra_lineal.md` | `attention-is-all-you-need.md` |

### → `aprendizaje/03-refuerzo/` (Parte 10)

| Antes | Ahora |
|---|---|
| `aprendizaje/federated-rl/1-RL_tabular.md` | `1-rl-tabular.md` |
| `aprendizaje/federated-rl/2-deep_RL.md` | `2-deep-rl.md` |
| `aprendizaje/federated-rl/3-sota_online_RL.md` | `3-sota-online-rl.md` |
| `aprendizaje/federated-rl/4-offline_RL.md` | `4-offline-rl.md` |

### → `aprendizaje/04-federado-y-distribuido/` (Parte 11)

| Antes | Ahora |
|---|---|
| `aprendizaje/federated-rl/5-federated_learning.md` | `5-federated-learning.md` |
| `aprendizaje/federated-rl/6-federated_RL.md` | `6-federated-rl.md` |
| `aprendizaje/federated-rl/resumen_pros_contras_retos_FL.md` | `pros-contras-y-retos-fl.md` |
| `aprendizaje/federated-rl/federated_offline_rl.md` | `libro-blanco-offline-rl-federado.md` |

### → `tesis/`

| Antes | Ahora |
|---|---|
| `aprendizaje/federated-rl/roadmap.md` | `roadmaps/roadmap-maestro.md` |
| `aprendizaje/federated-rl/roadmap-federated-learning.md` | `roadmaps/roadmap-federated-learning.md` |
| `aprendizaje/federated-rl/roadmap-federated-reinforcement-learning.md` | `roadmaps/roadmap-federated-rl.md` |
| `aprendizaje/federated-rl/roadmap_extensiones.md` | `roadmaps/extensiones-perfil-sistemas.md` |
| `aprendizaje/federated-rl/estado-del-arte-federated-learning.md` | `estado-del-arte/federated-learning.md` |
| `aprendizaje/federated-rl/mythos-estado_del_arte_y_roadmap_FRL_dataspaces.md` | `estado-del-arte/frl-y-data-spaces.md` |
| `aprendizaje/federated-rl/retos_DL_RL_offline_federado_ocean.md` | `estado-del-arte/retos-dl-rl-offline-federado.md` |
| `aprendizaje/federated-rl/mythos-guia_busqueda_WOS_FL_dataspaces.md` | `metodologia/busqueda-bibliografica-wos.md` |
| `referencia/dataspaces/RL-en-espacio-datos-federado.md` | `master-plan-ia-soberana.md` |

### → `recursos/`

Los seis `modulo_N_*.md` no son teoría: son **listas curadas de recursos**
(PDFs, vídeos, cursos) con un orden de estudio. Su sitio es `recursos/`.

| Antes | Ahora |
|---|---|
| `aprendizaje/federated-rl/README_ruta_de_aprendizaje.md` | `ruta-frl/index.md` |
| `aprendizaje/federated-rl/modulo_0_cimientos_matematicos.md` | `ruta-frl/modulo-0-cimientos-matematicos.md` |
| `aprendizaje/federated-rl/modulo_1_ciclo_entrenamiento.md` | `ruta-frl/modulo-1-ciclo-entrenamiento.md` |
| `aprendizaje/federated-rl/modulo_2_paradigmas_aprendizaje.md` | `ruta-frl/modulo-2-paradigmas.md` |
| `aprendizaje/federated-rl/modulo_3_aprendizaje_federado.md` | `ruta-frl/modulo-3-aprendizaje-federado.md` |
| `aprendizaje/federated-rl/modulo_4_data_spaces.md` | `ruta-frl/modulo-4-data-spaces.md` |
| `aprendizaje/federated-rl/modulo_5_frl.md` | `ruta-frl/modulo-5-frl.md` |
| `aprendizaje/algebra-lineal/recursos_llm_algebra_lineal.md` | `de-cero-a-transformer-algebra-lineal.md` |

### → `guias/` y `referencia/`

| Antes | Ahora | Motivo |
|---|---|---|
| `referencia/dataspaces/ocean-enterprise-instalacion.md` | `guias/ocean-enterprise-instalacion.md` | Es un procedimiento, no un dato a consultar |
| `referencia/dataspaces/base_conocimiento_p2p_pontus_x.md` | `referencia/dataspaces/base-conocimiento-p2p-pontus-x.md` | Solo normalización de nombre |

### → `aprendizaje/_archivo/` (fuera de la navegación)

Cuatro versiones anteriores de documentos que ya tienen sucesor vigente.

| Antes | Ahora | Sustituido por |
|---|---|---|
| `aprendizaje/algebra-lineal/biblia-redes-neuronales-mlp-a-transformers.md` (566 L) | `biblia-v1.md` | `02-redes-neuronales/biblia-mlp-a-transformers.md` |
| `aprendizaje/algebra-lineal/redes-neuronales-guia-aprendizaje.md` (782 L) | `biblia-v2.md` | idem |
| `aprendizaje/federated-rl/ruta_de_aprendizaje_FRL-actualizada.md` | `ruta-frl-monolitica.md` | La serie `recursos/ruta-frl/` |
| `aprendizaje/federated-rl/aprendizaje_refuerzo_offline_federado.md` (18 L) | `offline-rl-federado-esbozo.md` | `libro-blanco-offline-rl-federado.md` |

### Eliminado

`recursos/biblia-redes-neuronales-mlp-a-transformers.md` — byte a byte
idéntico a `aprendizaje/biblia-...md`. Se conserva una sola copia.

---

## Las tres versiones de "La Biblia"

Había **cuatro** ficheros con ese contenido. Comparados por índice de secciones:

| Fichero | Líneas | Contenido |
|---|---|---|
| `algebra-lineal/biblia-...` | 566 | Secciones 0–12. La más antigua |
| `algebra-lineal/redes-neuronales-guia-aprendizaje` | 782 | + metodología "del grafo al código", + anexo cronológico |
| `aprendizaje/biblia-...` | 926 | **Superconjunto**: + bloque 0 de neurona biológica |
| `recursos/biblia-...` | 926 | Copia exacta de la anterior |

La de 926 líneas contiene todo lo de las otras. Es la vigente; las demás van a
`_archivo/`. Antes de borrarlas, si quieres asegurarte:

```bash
diff <(grep '^#' docs/aprendizaje/_archivo/biblia-v2.md) \
     <(grep '^#' docs/aprendizaje/02-redes-neuronales/biblia-mlp-a-transformers.md)
```

---

## Cambios automáticos aplicados

1. **Nombres normalizados** a minúsculas con guiones, según tu propia regla en
   `meta/como-organizo-esto.md` ("el nombre del fichero es la URL"). Los
   `modulo_0_cimientos_matematicos.md` y `mythos-guia_busqueda_WOS_...` la
   incumplían.
2. **Enlaces internos reescritos** — los 21 enlaces `.md` entre notas apuntan a
   las rutas nuevas. Verificado: **0 enlaces rotos**.
3. **Frontmatter añadido** a los 41 ficheros que no lo tenían, con `title`
   tomado del H1, `tags` según la carpeta y `status: borrador`. Revisa los tags
   y sube a `revisado` lo que corresponda.
4. **`.DS_Store` eliminados** (19 ficheros).
5. **`.pages` creados** para cada carpeta nueva, con orden explícito donde el
   alfabético no servía.

---

## Qué falta por tu parte

- **Redirecciones.** Tu propia nota lo pide. En `mkdocs.yml`:

  ```yaml
  plugins:
    - redirects:
        redirect_maps:
          'aprendizaje/batch-normalization.md': 'aprendizaje/02-redes-neuronales/batch-normalization.md'
          'aprendizaje/federated-rl/roadmap.md': 'tesis/roadmaps/roadmap-maestro.md'
          # ...una por cada fila de las tablas de arriba que quieras preservar
  ```

  Solo hace falta si el sitio ya está publicado y hay enlaces externos.

- **Excluir `_archivo/` del buscador** para que no ensucie resultados:

  ```yaml
  plugins:
    - search:
        exclude:
          - aprendizaje/_archivo/*
  ```

- **Revisar los cuatro roadmaps.** Están en `tesis/roadmaps/` pero se solapan
  mucho entre sí. Probablemente uno solo debería ser el vigente y los otros
  tres, `_archivo/`. No lo he decidido yo porque requiere saber cuál sigues
  usando.

- **Los `tags`** que he puesto son genéricos por carpeta. `meta/tags.md` es tu
  navegación transversal: ahí es donde ganas al afinarlos.
