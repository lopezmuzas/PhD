# La imagen que se publica como asset de algoritmo en Ocean.
# OBLIGATORIO construir para linux/amd64: los nodos C2D no ejecutan arm64.
#   docker buildx build --platform linux/amd64 -t usuario/feltc2d:0.1 .
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY algorithms/ /app/algorithms/
ENV PYTHONPATH=/app/src

# Sin ENTRYPOINT: el mismo contenedor sirve para los DOS algoritmos.
# El comando decide cual, igual que hace FELT con feltlabs-py.
#   entrenamiento: python /app/algorithms/local_algorithm.py --aggregation_key ...
#   agregacion   : python /app/algorithms/aggregation_algorithm.py --private_key ...
CMD ["python", "/app/algorithms/local_algorithm.py", "--help"]
