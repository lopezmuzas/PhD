# Makefile raíz del doctorado. Orquesta los dos stacks incluidos en compose.yaml:
#   docs-stack/  -> perfiles: docs, build, wiki, lint
#   labs/dl-lab/ -> perfiles: cpu, gpu, tracking

UID := $(shell id -u)
GID := $(shell id -g)
export UID
export GID

C       := docker compose
TOKEN    = $(shell grep -E '^JUPYTER_TOKEN=' .env 2>/dev/null | cut -d= -f2)
LAB      = $(shell $(C) ps --services --status running 2>/dev/null | grep -q '^lab-gpu$$' && echo lab-gpu || echo lab)

.DEFAULT_GOAL := help

.PHONY: help init build build-gpu up up-gpu docs lab lab-gpu wiki tb mlflow bibliometrix \
        urls ps logs shell sync test lint lint-md docs-build down restart clean

help:  ## Muestra esta ayuda
	@echo "Doctorado — comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

# ---------------------------------------------------------------- preparación
init:  ## Crea .env y las carpetas locales necesarias
	@[ -f .env ] || cp .env.example .env
	@sed -i.bak "s/^UID=.*/UID=$(UID)/; s/^GID=.*/GID=$(GID)/" .env && rm -f .env.bak
	@mkdir -p labs/dl-lab/data/raw labs/dl-lab/data/interim \
	          labs/dl-lab/data/processed labs/dl-lab/outputs/runs \
	          labs/dl-lab/outputs/checkpoints
	@echo "Entorno preparado."

build: init  ## Construye las dos imágenes (docs + laboratorio CPU)
	$(C) --profile docs --profile cpu build

build-gpu: init  ## Construye docs + laboratorio con CUDA
	$(C) --profile docs --profile gpu build

# ------------------------------------------------------------------- arranque
up: init  ## Levanta docs + JupyterLab (CPU)   <-- el habitual
	$(C) --profile docs --profile cpu up -d
	@$(MAKE) --no-print-directory urls

up-gpu: init  ## Levanta docs + JupyterLab con GPU
	$(C) --profile docs --profile gpu up -d
	@$(MAKE) --no-print-directory urls

docs: init  ## Solo la documentación -> http://localhost:8000
	$(C) --profile docs up -d docs
	@echo "  Docs        http://localhost:8000"

lab: init  ## Solo JupyterLab (CPU) -> http://localhost:8888
	$(C) --profile cpu up -d lab
	@echo "  JupyterLab  http://localhost:8888/lab?token=$(TOKEN)"

lab-gpu: init  ## Solo JupyterLab (GPU)
	$(C) --profile gpu up -d lab-gpu
	@echo "  JupyterLab  http://localhost:8888/lab?token=$(TOKEN)"

wiki: init  ## Wiki editable en el navegador -> http://localhost:3000
	$(C) --profile wiki up -d wiki
	@echo "  Wiki        http://localhost:3000"

tb:  ## TensorBoard sobre outputs/runs -> http://localhost:6006
	$(C) exec -d $(LAB) tensorboard --logdir outputs/runs --host 0.0.0.0 --port 6006
	@echo "  TensorBoard http://localhost:6006"

mlflow:  ## Servidor MLflow -> http://localhost:5000
	$(C) --profile tracking up -d mlflow
	@echo "  MLflow      http://localhost:5000"

bibliometrix:  ## Lanza Biblioshiny localmente con R en tu navegador
	Rscript bibliometrix/run_biblioshiny.R

urls:  ## Recuerda dónde está cada cosa
	@echo ""
	@echo "  Docs        http://localhost:8000"
	@echo "  JupyterLab  http://localhost:8888/lab?token=$(TOKEN)"
	@echo "  TensorBoard http://localhost:6006   (make tb)"
	@echo "  MLflow      http://localhost:5000   (make mlflow)"
	@echo "  Wiki        http://localhost:3000   (make wiki)"
	@echo "  Bibliometrix (R local)              (make bibliometrix)"
	@echo ""

# ------------------------------------------------------------------ día a día
shell:  ## Abre una bash en el contenedor del laboratorio
	$(C) exec $(LAB) bash

sync:  ## Sincroniza .ipynb <-> .py con jupytext
	$(C) exec $(LAB) jupytext --sync notebooks/*.ipynb

test:  ## pytest del laboratorio
	$(C) exec $(LAB) pytest -q

lint:  ## ruff sobre el código + markdownlint y enlaces sobre los .md
	-$(C) exec $(LAB) ruff check --fix src tests scripts
	-$(C) --profile lint run --rm lint-md
	-$(C) --profile lint run --rm link-check

lint-md:  ## Solo la higiene de los .md
	-$(C) --profile lint run --rm lint-md
	-$(C) --profile lint run --rm link-check

docs-build:  ## Genera el sitio estático en .site/
	$(C) --profile build run --rm docs-build

ps:  ## Qué hay levantado
	$(C) ps

logs:  ## Sigue los logs de todo lo levantado
	$(C) logs -f

# --------------------------------------------------------------------- parada
down:  ## Para todos los servicios de los dos stacks
	$(C) --profile docs --profile build --profile wiki --profile lint \
	     --profile cpu --profile gpu --profile tracking down

restart: down up  ## Reinicio limpio

clean:  ## Borra cachés de Python y el sitio generado
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .site .pytest_cache .ruff_cache