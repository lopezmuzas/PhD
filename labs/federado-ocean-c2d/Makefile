# Atajos del laboratorio. `make help` para verlos todos.
.PHONY: help install data test test-all lint lab0 lab1 lab2 docker-build clean

IMAGE ?= fedlab
TAG   ?= test
PY    := PYTHONPATH=src python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

install:  ## Instala dependencias de desarrollo
	pip install -r requirements-dev.txt

data:  ## Genera los CSV sinteticos en ./data
	$(PY) scripts/make_datasets.py

test:  ## Tests rapidos (sin Docker). Deberia tardar < 2s
	$(PY) -m pytest tests/ -q -m "not docker"

test-all: docker-build  ## Todos los tests, incluido el contrato de la imagen
	$(PY) -m pytest tests/ -q

lint:  ## Estilo
	ruff check src tests examples scripts

lab0:  ## Lab 0 - los tres ejemplos en simulacion pura
	$(PY) examples/ex01_linear.py
	@echo
	$(PY) examples/ex02_noniid.py
	@echo
	$(PY) examples/ex03_hospitals.py

lab1: data  ## Lab 1 - patron de ficheros, sin Docker
	$(PY) scripts/run_lab1.py --rounds 15

lab2: docker-build  ## Lab 2 - contrato contra la imagen Docker real
	$(PY) -m pytest tests/test_c2d_contract.py -q -m docker

docker-build:  ## Construye la imagen SIEMPRE para amd64
	docker buildx build --platform linux/amd64 \
	  -t $(IMAGE):$(TAG) -f docker/Dockerfile .

docker-digest:  ## Muestra el digest sha256 para fijarlo en el job C2D
	@docker inspect --format='{{index .RepoDigests 0}}' $(IMAGE):$(TAG) 2>/dev/null \
	  || echo "Sin digest: haz 'docker push' primero. El nodo necesita bajarla de un registro."

clean:
	rm -rf work results .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
