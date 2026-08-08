# syntax=docker/dockerfile:1
# Visor/organizador de los .md del doctorado. Nada que ver con el laboratorio:
# esta imagen no lleva PyTorch ni CUDA, arranca en segundos.
FROM squidfunk/mkdocs-material:9.5.44

# git -> el plugin de fechas lee el historial del repo
RUN apk add --no-cache git

RUN pip install --no-cache-dir \
        mkdocs-awesome-pages-plugin \
        mkdocs-git-revision-date-localized-plugin \
        mkdocs-glightbox \
        mkdocs-jupyter \
        mkdocs-minify-plugin \
        mkdocs-redirects \
        pymdown-extensions

WORKDIR /docs
EXPOSE 8000
