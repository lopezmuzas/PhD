"""EL contrato con Ocean C2D, verificado SIN Docker y SIN nodo.

Montamos un /data falso identico al que crea ocean-node y ejecutamos el
entrypoint real como subproceso. Si estos tests pasan, lo unico que puede
fallar en el nodo de verdad es infraestructura (arquitectura de la imagen,
red, permisos), no tu codigo.

Rutas y nombres verificados contra ocean-node/src/components/c2d/
compute_engine_docker.ts (agosto 2026).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedlab.adapters.stores import C2DStore, find_dataset_file
from fedlab.serialization import load_params_npz, params_to_json

SRC = str(Path(__file__).resolve().parents[1] / "src")


def build_fake_data_dir(tmp_path: Path, csv_name: str = "dataset.csv",
                        custom: dict | None = None) -> Path:
    """Replica la estructura que ocean-node crea antes de arrancar el contenedor."""
    data = tmp_path / "data"
    for sub in ("inputs", "outputs", "logs", "ddos", "transformations"):
        (data / sub).mkdir(parents=True, exist_ok=True)

    # Dataset descargado con su nombre original
    X = np.random.default_rng(0).normal(size=(50, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    header = ",".join([f"f{i}" for i in range(6)] + ["label"])
    np.savetxt(data / "inputs" / csv_name, np.column_stack([X, y]),
               delimiter=",", header=header, comments="", fmt="%.6f")

    # El nodo escribe algoCustomData.json SIEMPRE, aunque este vacio
    (data / "inputs" / C2DStore.CUSTOM_DATA).write_text(json.dumps(custom or {}))
    return data


def run_entrypoint(data: Path, env_extra: dict | None = None):
    env = {
        "PYTHONPATH": SRC,
        "PATH": "/usr/bin:/bin",
        "FL_ENV": "c2d",
        "FL_MODEL": "logistic",
        "FL_FEATURES": "6",
        # Redirigimos /data a nuestro tmpdir. El contenedor real no define
        # estas variables y cae en /data/inputs y /data/outputs.
        "FL_INPUT_DIR": str(data / "inputs"),
        "FL_OUTPUT_DIR": str(data / "outputs"),
        **(env_extra or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "fedlab.entrypoints.client_main"],
        env=env, capture_output=True, text=True, timeout=120,
    )


# ---------------------------------------------------------------------------
# El contrato de entrada
# ---------------------------------------------------------------------------


def test_encuentra_el_dataset_ignorando_algocustomdata(tmp_path):
    """El nodo guarda el CSV con su nombre ORIGINAL, que no conocemos de antemano.
    Hay que localizarlo por extension y descartar algoCustomData.json."""
    data = build_fake_data_dir(tmp_path, csv_name="pacientes_2024_v3.csv")
    found = find_dataset_file(str(data / "inputs"))
    assert found is not None
    assert Path(found).name == "pacientes_2024_v3.csv"


def test_algocustomdata_vacio_no_rompe_nada(tmp_path):
    """Ronda 0: el nodo escribe {}. Debe entrenar desde cero, no fallar."""
    data = build_fake_data_dir(tmp_path, custom={})
    store = C2DStore(str(data / "inputs"), str(data / "outputs"))
    assert store.load_global() is None


def test_algocustomdata_corrupto_no_tumba_el_job(tmp_path):
    data = build_fake_data_dir(tmp_path)
    (data / "inputs" / C2DStore.CUSTOM_DATA).write_text("{esto no es json")
    store = C2DStore(str(data / "inputs"), str(data / "outputs"))
    assert store.read_custom_data() == {}


def test_lee_los_pesos_globales_de_algocustomdata(tmp_path):
    """El canal por el que viajan los pesos de ronda a ronda."""
    params = [np.arange(6, dtype=float), np.array([0.5])]
    data = build_fake_data_dir(tmp_path, custom={"params": params_to_json(params)})
    store = C2DStore(str(data / "inputs"), str(data / "outputs"))
    loaded = store.load_global()
    assert loaded is not None
    assert np.allclose(loaded[0], params[0])
    assert np.allclose(loaded[1], params[1])


# ---------------------------------------------------------------------------
# El contrato de salida (ejecutando el entrypoint de verdad)
# ---------------------------------------------------------------------------


def test_entrypoint_produce_los_ficheros_esperados(tmp_path):
    """Ronda 0 completa: entra CSV, sale update.npz + metrics.json."""
    data = build_fake_data_dir(tmp_path)
    proc = run_entrypoint(data)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    update = data / "outputs" / C2DStore.UPDATE_FILE
    meta = data / "outputs" / C2DStore.META_FILE
    assert update.exists(), f"Falta update.npz. Salida:\n{proc.stdout}"
    assert meta.exists()

    params = load_params_npz(update)
    assert params[0].shape == (6,)
    metrics = json.loads(meta.read_text())
    assert metrics["num_samples"] == 50
    assert "accuracy" in metrics


def test_entrypoint_continua_desde_los_pesos_recibidos(tmp_path):
    """Ronda N: los pesos que entran deben influir en los que salen.

    Si esto falla, cada ronda entrena desde cero y NO hay federacion:
    solo N entrenamientos aislados repetidos. Es el bug silencioso mas caro.
    """
    fresh = build_tmp_run(tmp_path / "a", custom={})
    warm = build_tmp_run(tmp_path / "b",
                         custom={"params": params_to_json([np.full(6, 5.0), np.array([1.0])])})
    assert not np.allclose(fresh[0], warm[0]), (
        "Los pesos de entrada no afectaron al resultado: la ronda ignora el estado global."
    )


def build_tmp_run(tmp_path: Path, custom: dict):
    data = build_fake_data_dir(tmp_path, custom=custom)
    proc = run_entrypoint(data)
    assert proc.returncode == 0, proc.stderr
    return load_params_npz(data / "outputs" / C2DStore.UPDATE_FILE)


def test_falla_con_codigo_2_si_no_hay_dataset(tmp_path):
    """Sin shell dentro del contenedor, el codigo de salida y el log SON tu diagnostico."""
    data = tmp_path / "data"
    for sub in ("inputs", "outputs"):
        (data / sub).mkdir(parents=True)
    (data / "inputs" / C2DStore.CUSTOM_DATA).write_text("{}")
    proc = run_entrypoint(data)
    assert proc.returncode == 2
    assert "No hay dataset" in proc.stderr


def test_modelo_desconocido_falla_ruidosamente(tmp_path):
    data = build_fake_data_dir(tmp_path)
    proc = run_entrypoint(data, {"FL_MODEL": "transformer"})
    assert proc.returncode == 1
    assert "desconocido" in proc.stderr


@pytest.mark.docker
def test_imagen_docker_cumple_el_contrato(tmp_path):
    """El mismo contrato, contra la imagen REAL. Requiere `make docker-build`.

        pytest -m docker

    Detecta casi todo lo que falla en el nodo sin gastar un job.
    """
    data = build_fake_data_dir(tmp_path)
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none",
         "-v", f"{data}:/data",
         "-e", "FL_MODEL=logistic", "-e", "FL_FEATURES=6",
         "fedlab:test"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert (data / "outputs" / C2DStore.UPDATE_FILE).exists()
