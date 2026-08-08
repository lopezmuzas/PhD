# syntax=docker/dockerfile:1
FROM python:3.11-slim

# cpu | cu124 | cu121 ...  (índice de wheels de PyTorch)
ARG TORCH_INDEX=cpu
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates tini \
        libgl1 libglib2.0-0 graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# 1) PyTorch en su propia capa: es lo más pesado y lo que menos cambia.
#    Las wheels cuXXX incluyen las librerías CUDA, así que NO hace falta partir
#    de nvidia/cuda: basta con el driver del host + nvidia-container-toolkit.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
        torch torchvision torchaudio

# 2) Resto de dependencias del proyecto.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv pip install --system -e ".[dev]"

# 3) Usuario no-root con el UID del host -> lo que escriba Jupyter en el
#    bind-mount no aparece como root en tu máquina.
RUN groupadd -g ${GID} lab 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash lab \
    && mkdir -p /home/lab/.cache \
    && chown -R ${UID}:${GID} /home/lab /workspace

USER lab
ENV HOME=/home/lab \
    JUPYTER_PLATFORM_DIRS=1 \
    PYTHONPATH=/workspace/src

EXPOSE 8888 6006

ENTRYPOINT ["/usr/bin/tini", "--"]
# JUPYTER_TOKEN se lee automáticamente de la variable de entorno.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace"]
