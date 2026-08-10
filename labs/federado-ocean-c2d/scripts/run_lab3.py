"""LAB 3 -- El bucle federado contra un ocean-node REAL.

Requisitos previos:
  1. `make lab2` en verde (la imagen cumple el contrato).
  2. La imagen subida a un registro accesible por el nodo (Docker Hub, GHCR...).
     El nodo hace `docker pull`: una imagen que solo existe en tu portatil no vale.
  3. Los CSV servidos por HTTP en una URL que el nodo pueda alcanzar.
  4. Una clave privada con fondos (o nada, si usas compute gratuito).

    export FL_PRIVATE_KEY=0x...
    python scripts/run_lab3.py \
        --node http://tu-servidor:8000 \
        --image tu-usuario/fedlab --tag v1 \
        --dataset https://tu-host/ex03_hospital_a.csv \
        --dataset https://tu-host/ex03_hospital_b.csv \
        --rounds 5

Con varios nodos proveedores, pasa `--node` varias veces (uno por dataset).
"""

from __future__ import annotations

import argparse
import os

from fedlab.config import TrainConfig
from fedlab.ocean.orchestrator import AlgorithmSpec, Provider, run_federated
from fedlab.serialization import save_params_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", action="append", required=True,
                    help="URL del ocean-node (repetible; se empareja con --dataset)")
    ap.add_argument("--dataset", action="append", required=True,
                    help="URL publica del CSV (repetible)")
    ap.add_argument("--image", required=True, help="ej. tu-usuario/fedlab")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--checksum", default=None,
                    help="digest sha256:... Recomendado: fija la imagen exacta")
    ap.add_argument("--environment", default=None, help="id del compute environment")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--features", type=int, default=6)
    ap.add_argument("--model", default="logistic", choices=["logistic", "linear"])
    ap.add_argument("--out", default="modelo_federado.npz")
    args = ap.parse_args()

    key = os.getenv("FL_PRIVATE_KEY")
    if not key:
        raise SystemExit("Define FL_PRIVATE_KEY. Nunca la pongas en la linea de comandos: "
                         "queda en el historial del shell.")

    nodes = args.node if len(args.node) > 1 else args.node * len(args.dataset)
    if len(nodes) != len(args.dataset):
        raise SystemExit("Pasa un --node por cada --dataset, o un solo --node para todos.")

    providers = [
        Provider(name=f"nodo{i + 1}", node_url=n, dataset_url=d, environment=args.environment)
        for i, (n, d) in enumerate(zip(nodes, args.dataset))
    ]
    algorithm = AlgorithmSpec(
        image=args.image, tag=args.tag, checksum=args.checksum,
        envs={"FL_MODEL": args.model, "FL_FEATURES": str(args.features)},
    )

    print(f"{len(providers)} proveedores | imagen {args.image}:{args.tag} | {args.rounds} rondas")
    params = run_federated(
        providers=providers, algorithm=algorithm, private_key=key,
        cfg=TrainConfig(epochs=args.epochs, lr=args.lr), rounds=args.rounds,
    )
    save_params_npz(args.out, params)
    print(f"\nModelo federado guardado en {args.out}")


if __name__ == "__main__":
    main()
