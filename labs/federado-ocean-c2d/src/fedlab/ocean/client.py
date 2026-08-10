"""Cliente HTTP minimo para ocean-node. Solo lo que necesita el bucle federado.

Endpoints verificados contra el codigo fuente de ocean-node (agosto 2026):

    GET  /api/services/nonce?userAddress=0x...
    GET  /api/services/computeEnvironments
    POST /api/services/freeCompute
    GET  /api/services/compute?jobId=...&consumerAddress=...
    GET  /api/services/computeResult?...&index=0

Se usa `freeCompute`: sin datatokens, sin gas, sin escrow. Para APRENDER es el
camino correcto; el compute de pago anade una capa de contratos que no aporta
nada hasta que el pipeline funciona.

FIRMA (lo verifique en src/components/core/utils/nonceHandler.ts, porque la
tabla de docs/API.md esta desactualizada para varios comandos):

    mensaje = consumerAddress + str(nonce) + nombre_del_comando
    digest  = keccak256(utf8(mensaje))
    firma   = personal_sign(digest)        # EIP-191 sobre los 32 bytes crudos

El nonce debe ser ESTRICTAMENTE MAYOR que el almacenado en el nodo, y se
actualiza en cada peticion aceptada: por eso se relee antes de cada llamada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import keccak

# Codigos de estado de C2DStatusNumber (src/@types/C2D/C2D.ts)
STATUS = {
    0: "Job started", 1: "Job queued", 2: "Queue expired",
    10: "Pulling image", 11: "Pull image FAILED", 12: "Building image",
    13: "Build image FAILED", 14: "Vulnerable image",
    20: "Configuring volumes", 21: "Volume creation FAILED",
    22: "Container creation FAILED",
    30: "Provisioning", 31: "Data provisioning FAILED",
    32: "Algorithm provisioning FAILED", 33: "Data upload FAILED",
    40: "Running algorithm", 41: "Algorithm FAILED", 42: "Disk quota exceeded",
    50: "Filtering results", 60: "Publishing results",
    61: "Results fetch FAILED", 62: "Results upload FAILED",
    70: "Job finished", 71: "Job settle",
}
FINISHED = 70
FAILED = {2, 11, 13, 14, 21, 22, 31, 32, 33, 41, 42, 61, 62}


@dataclass
class Job:
    node_url: str
    job_id: str
    environment: str


class OceanNodeClient:
    def __init__(self, node_url: str, private_key: str, timeout: int = 60):
        self.node_url = node_url.rstrip("/")
        self.account = Account.from_key(private_key)
        self.address = self.account.address
        self.timeout = timeout

    # -- firma -------------------------------------------------------------
    def _nonce(self) -> int:
        r = requests.get(f"{self.node_url}/api/services/nonce",
                         params={"userAddress": self.address}, timeout=self.timeout)
        r.raise_for_status()
        try:
            current = int(float(r.text.strip().strip('"') or 0))
        except ValueError:
            current = 0
        # ms desde epoch: monotono y siempre por delante del almacenado
        return max(current + 1, int(time.time() * 1000))

    def _sign(self, nonce: int, command: str) -> str:
        digest = keccak(text=f"{self.address}{nonce}{command}")
        return self.account.sign_message(encode_defunct(digest)).signature.hex()

    def _auth(self, command: str) -> dict:
        nonce = self._nonce()
        return {"consumerAddress": self.address, "nonce": str(nonce),
                "signature": self._sign(nonce, command)}

    # -- API ---------------------------------------------------------------
    def environments(self) -> list[dict]:
        r = requests.get(f"{self.node_url}/api/services/computeEnvironments",
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def pick_free_environment(self) -> str:
        """Primer entorno con plazas de compute gratuito."""
        envs = self.environments()
        for env in envs:
            if env.get("free") or env.get("freeComputeOptions") or env.get("maxJobs"):
                return env["id"]
        if not envs:
            raise RuntimeError("El nodo no expone ningun compute environment.")
        return envs[0]["id"]

    def start_free_job(self, environment: str, image: str, tag: str,
                       entrypoint: str, dataset_url: str,
                       custom_data: dict | None = None,
                       envs: dict | None = None,
                       checksum: str | None = None,
                       resources: list[dict] | None = None) -> str:
        container: dict = {"image": image, "tag": tag, "entrypoint": entrypoint}
        if checksum:  # digest sha256:... -> reproducibilidad exacta
            container["checksum"] = checksum

        payload = {
            "command": "freeStartCompute",
            "environment": environment,
            "datasets": [{"fileObject": {"type": "url", "url": dataset_url, "method": "GET"}}],
            "algorithm": {
                "meta": {"container": container},
                "algocustomdata": custom_data or {},
                **({"envs": envs} if envs else {}),
            },
            "resources": resources or [{"id": "cpu", "amount": 1}],
            **self._auth("freeStartCompute"),
        }
        r = requests.post(f"{self.node_url}/api/services/freeCompute",
                          json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"freeCompute {r.status_code}: {r.text[:500]}")
        data = r.json()
        return (data[0] if isinstance(data, list) else data)["jobId"]

    def status(self, job_id: str) -> dict:
        r = requests.get(f"{self.node_url}/api/services/compute",
                         params={"jobId": job_id, "consumerAddress": self.address},
                         timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and data else data

    def wait(self, job_id: str, poll: int = 5, timeout: int = 900,
             verbose: bool = True) -> dict:
        """Espera con timeout duro. Nunca hagas `while True` contra una red."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            job = self.status(job_id)
            code = int(job.get("status", -1))
            if verbose and code != last:
                print(f"    [{job_id[-12:]}] {code} {STATUS.get(code, '?')}")
                last = code
            if code == FINISHED:
                return job
            if code in FAILED:
                raise RuntimeError(
                    f"Job {job_id} fallo: {code} {STATUS.get(code)}. "
                    "Descarga el log (index=1) para ver el motivo."
                )
            time.sleep(poll)
        raise TimeoutError(f"Job {job_id} no termino en {timeout}s.")

    def result(self, job_id: str, index: int = 0) -> bytes:
        """index=0 -> resultado principal (tar de /data/outputs); index=1 -> logs."""
        params = {"jobId": job_id, "index": index, **self._auth("getComputeResult")}
        r = requests.get(f"{self.node_url}/api/services/computeResult",
                         params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"computeResult {r.status_code}: {r.text[:500]}")
        return r.content
