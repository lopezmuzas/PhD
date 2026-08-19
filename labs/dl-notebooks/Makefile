.PHONY: help test lab clean

help:
	@echo "  make test    harness tests (< 10 s)"
	@echo "  make lab     open jupyter"
	@echo "  make clean   remove runs/ and caches"

test:
	python -m pytest tests -q

lab:
	jupyter lab

clean:
	rm -rf runs __pycache__ */__pycache__ .pytest_cache **/.ipynb_checkpoints
