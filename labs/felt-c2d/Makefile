.PHONY: help test lab0 lab1 lab2 ex01 ex02 ex03 docker-build clean

PY ?= python3

help:
	@echo "Comandos:"
	@echo "  make test          tests rapidos (< 2 s)"
	@echo "  make lab0          los tres ejemplos, todo en memoria"
	@echo "  make lab1          pipeline de dos algoritmos con carpetas locales"
	@echo "  make lab2          contrato contra la imagen Docker"
	@echo "  make docker-build  construye la imagen (amd64 obligatorio)"

test:
	$(PY) -m pytest tests -q

lab0: ex01 ex02 ex03

ex01:
	$(PY) examples/ex01_el_protocolo.py

ex02:
	$(PY) examples/ex02_quien_ve_que.py

ex03:
	$(PY) examples/ex03_una_ronda_o_muchas.py

lab1:
	$(PY) scripts/run_lab1.py

lab2:
	$(PY) -m pytest tests -q -m docker

docker-build:
	docker buildx build --platform linux/amd64 -t feltc2d:local -f docker/Dockerfile .

clean:
	rm -rf trabajo .pytest_cache **/__pycache__
