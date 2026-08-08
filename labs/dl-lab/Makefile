UID := $(shell id -u)
GID := $(shell id -g)
export UID
export GID

COMPOSE := docker compose

.PHONY: help init build up gpu down shell logs lab-url tb mlflow sync test lint clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

init:  ## Crea .env y carpetas locales
	@[ -f .env ] || cp .env.example .env
	@sed -i.bak "s/^UID=.*/UID=$(UID)/;s/^GID=.*/GID=$(GID)/" .env && rm -f .env.bak
	@mkdir -p data/raw data/interim data/processed outputs/runs outputs/checkpoints
	@echo "Listo. Edita .env si quieres cambiar el token de Jupyter."

build: init  ## Construye la imagen CPU
	$(COMPOSE) --profile cpu build

build-gpu: init  ## Construye la imagen GPU (CUDA 12.4)
	$(COMPOSE) --profile gpu build

up: init  ## Arranca JupyterLab en CPU -> http://localhost:8888
	$(COMPOSE) --profile cpu up -d
	@echo "JupyterLab: http://localhost:8888/lab?token=$$(grep JUPYTER_TOKEN .env | cut -d= -f2)"

gpu: init  ## Arranca JupyterLab con GPU -> http://localhost:8888
	$(COMPOSE) --profile gpu up -d
	@echo "JupyterLab: http://localhost:8888/lab?token=$$(grep JUPYTER_TOKEN .env | cut -d= -f2)"

down:  ## Para todo
	$(COMPOSE) --profile cpu --profile gpu --profile tracking down

shell:  ## Abre una bash dentro del contenedor
	$(COMPOSE) --profile cpu exec lab bash || $(COMPOSE) --profile gpu exec lab-gpu bash

logs:  ## Sigue los logs
	$(COMPOSE) --profile cpu --profile gpu logs -f

tb:  ## TensorBoard -> http://localhost:6006
	$(COMPOSE) --profile cpu exec -d lab tensorboard --logdir outputs/runs --host 0.0.0.0 --port 6006

mlflow:  ## Servidor MLflow -> http://localhost:5000
	$(COMPOSE) --profile tracking up -d mlflow

sync:  ## Sincroniza .ipynb <-> .py con jupytext
	$(COMPOSE) --profile cpu exec lab find notebooks -name "*.ipynb" -exec jupytext --sync {} +

test:  ## Tests
	$(COMPOSE) --profile cpu exec lab pytest -q

lint:  ## Ruff
	$(COMPOSE) --profile cpu exec lab ruff check --fix src tests scripts

clean:  ## Borra cachés
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
