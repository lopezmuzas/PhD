# doctorado

```
doctorado/
├── compose.yaml            # orquestador raíz: incluye los dos stacks
├── mkdocs.yml              # config del sitio de documentación
├── Makefile
├── docs/                   # tus .md (fuente de verdad, ficheros planos)
│   ├── aprendizaje/  guias/  referencia/  recursos/  bitacora/  meta/
├── docs-stack/             # Dockerfile + compose del visor de docs
└── labs/
    └── dl-lab/             # el laboratorio de deep learning (su propio compose)
```

## Uso

```bash
cp .env.example .env
make docs      # http://localhost:8000   documentación con recarga en vivo
make lab       # http://localhost:8888   JupyterLab
make wiki      # http://localhost:3000   edición de notas en el navegador (opcional)
make lint      # markdownlint + enlaces rotos
make ps        # qué hay levantado
make down      # para todo
```

Los dos stacks son independientes: puedes levantar solo las docs sin arrancar
PyTorch. La imagen de documentación pesa ~200 MB y arranca en segundos.

## Puertos

| Puerto | Servicio |
|---|---|
| 8000 | MkDocs |
| 8888 | JupyterLab |
| 6006 | TensorBoard |
| 5000 | MLflow |
| 3000 | SilverBullet (wiki) |

## Publicar

`make docs-build` genera `.site/`. Para GitHub Pages basta un workflow que
ejecute `mkdocs gh-deploy --force`.
