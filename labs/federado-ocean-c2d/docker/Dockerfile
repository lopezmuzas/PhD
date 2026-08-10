# Imagen del algoritmo C2D.
#
# CONSTRUIR SIEMPRE PARA amd64, aunque estes en un Mac con Apple Silicon:
#
#     docker buildx build --platform linux/amd64 \
#         -t tu-usuario/fedlab:v1 -f docker/Dockerfile .
#
# ocean-node valida la arquitectura del manifiesto (checkManifestPlatform en
# compute_engine_docker.ts). Una imagen arm64 en un nodo amd64 muere en estado
# 11 (Pull image FAILED) o 22 sin un mensaje util. Es el fallo numero uno.

FROM python:3.12-slim

# Dependencias primero: capa cacheada, no se reconstruye al tocar el codigo.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# El codigo va a /app, NO a /data. /data lo monta el nodo y machaca lo que haya.
WORKDIR /app
COPY src/ /app/src/
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ocean-node arranca el contenedor con un UID/GID no-root propio
# (C2D_CONTAINER_UID/GID), CapDrop ALL y no-new-privileges. No declaramos USER:
# el nodo impone el suyo. Solo garantizamos que el codigo sea legible por todos.
RUN chmod -R a+rX /app

# Valores por defecto; sobreescribibles con `algorithm.envs` en cada job.
ENV FL_ENV=c2d \
    FL_MODEL=logistic \
    FL_FEATURES=6

# El nodo puede sobreescribir esto con meta.container.entrypoint.
ENTRYPOINT ["python", "-m", "fedlab.entrypoints.client_main"]
