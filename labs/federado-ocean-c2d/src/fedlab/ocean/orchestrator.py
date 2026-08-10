"""Orquestador federado sobre ocean-node.

Responsabilidad unica: coordinar rondas. NO entrena, NO agrega (delega en
`domain.aggregation`), NO habla HTTP a mano (delega en OceanNodeClient).

Bucle por ronda:
    1. serializar los pesos globales en algoCustomData
    2. lanzar un job C2D por proveedor, en paralelo
    3. esperar, descargar y desempaquetar /data/outputs
    4. FedAvg -> nuevos pesos globales
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ..config import TrainConfig
from ..domain.aggregation import fedavg, params_delta
from ..ports import Params
from ..serialization import check_json_size, load_params_npz, params_to_json, read_json
from .client import OceanNodeClient


@dataclass(frozen=True)
class Provider:
    """Un proveedor de datos: su nodo y el dataset que expone."""
    name: str
    node_url: str
    dataset_url: str
    environment: str | None = None


@dataclass(frozen=True)
class AlgorithmSpec:
    image: str
    tag: str = "latest"
    entrypoint: str = "python -m fedlab.entrypoints.client_main"
    checksum: str | None = None
    envs: dict | None = None


def _extract_outputs(blob: bytes, dest: Path) -> tuple[Params, dict]:
    """Desempaqueta el tar de /data/outputs y lee update.npz + metrics.json."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        # filter="data" bloquea rutas absolutas y ../ (Python 3.12+).
        # Sin esto, un proveedor hostil te escribe donde quiera.
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)

    npz = next(dest.rglob("update.npz"), None)
    meta = next(dest.rglob("metrics.json"), None)
    if npz is None or meta is None:
        found = sorted(p.name for p in dest.rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"El job no produjo update.npz/metrics.json. Ficheros: {found}"
        )
    return load_params_npz(npz), read_json(meta)


def _run_job(provider: Provider, clients: dict[str, OceanNodeClient],
             envs: dict[str, str], algorithm: AlgorithmSpec,
             custom: dict, dest_root: Path) -> tuple[Params, dict]:
    """Lanza, espera y recoge UN job. Funcion de nivel de modulo a proposito:
    una closure sobre las variables del bucle se enlaza tarde y es un bug
    esperando a que alguien reutilice el pool entre rondas."""
    client = clients[provider.name]
    job_id = client.start_free_job(
        environment=envs[provider.name], image=algorithm.image, tag=algorithm.tag,
        entrypoint=algorithm.entrypoint, dataset_url=provider.dataset_url,
        custom_data=custom, envs=algorithm.envs, checksum=algorithm.checksum,
    )
    print(f"  -> {provider.name}: job {job_id}")
    client.wait(job_id)
    return _extract_outputs(client.result(job_id, index=0), dest_root / provider.name)


def run_federated(
    providers: list[Provider],
    algorithm: AlgorithmSpec,
    private_key: str,
    cfg: TrainConfig,
    rounds: int = 5,
    initial_params: Params | None = None,
    workdir: str | None = None,
) -> Params:
    clients = {p.name: OceanNodeClient(p.node_url, private_key) for p in providers}
    envs = {
        p.name: p.environment or clients[p.name].pick_free_environment()
        for p in providers
    }
    work = Path(workdir or tempfile.mkdtemp(prefix="fedlab-"))
    global_params = initial_params

    for r in range(1, rounds + 1):
        print(f"\n=== RONDA {r}/{rounds} ===")
        custom: dict = {"round": r, "config": cfg.to_dict()}
        if global_params is not None:
            check_json_size(global_params)
            custom["params"] = params_to_json(global_params)

        run = partial(_run_job, clients=clients, envs=envs, algorithm=algorithm,
                      custom=custom, dest_root=work / f"r{r}")
        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            results = list(pool.map(run, providers))

        updates = [(params, int(meta["num_samples"])) for params, meta in results]
        previous = global_params
        global_params = fedavg(updates)

        total = sum(n for _, n in updates)
        for p, (_, meta) in zip(providers, results):
            print(f"  {p.name}: n={meta['num_samples']} "
                  + " ".join(f"{k}={v:.4f}" for k, v in meta.items()
                             if k != "num_samples" and isinstance(v, (int, float))))
        drift = f" delta={params_delta(global_params, previous):.5f}" if previous else ""
        print(f"  agregado sobre {total} muestras{drift}")

    return global_params
