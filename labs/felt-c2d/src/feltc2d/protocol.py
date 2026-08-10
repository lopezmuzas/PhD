"""El protocolo de FELT en tres fases, sin Ocean de por medio.

    FASE 1 (job C2D por dataset)   entrenar en local, cegar, cifrar
    FASE 2 (job C2D de agregacion) descifrar, promediar
    FASE 3 (en tu maquina)         quitar el ruido

La propiedad que se obtiene es un DOBLE CIEGO:

  - el usuario NO puede leer los modelos locales (van cifrados al agregador)
  - el agregador NO puede leer los modelos locales (llevan ruido que no conoce)

Nadie ve un modelo entrenado sobre un solo dataset. Solo existe el global.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from feltc2d import crypto, noise
from feltc2d.models import construir


def _serializar(pesos, n, tipo, alfa=None) -> bytes:
    d = {
        "tipo": tipo,
        "n_muestras": int(n),
        "pesos": {k: np.asarray(v).tolist() for k, v in pesos.items()},
    }
    if alfa is not None:
        d["alfa"] = [float(a) for a in alfa]
    return json.dumps(d).encode()


def _deserializar(datos: bytes) -> dict:
    d = json.loads(datos.decode())
    d["pesos"] = {k: np.asarray(v, dtype=float) for k, v in d["pesos"].items()}
    return d


# --8<-- [start:fase1]
@dataclass
class ResultadoLocal:
    """Lo unico que sale del contenedor del proveedor de datos."""

    cifrado: bytes           # ilegible sin la privada del agregador
    n_muestras: int          # metadato en claro, para poder ponderar


def fase1_entrenar(
    X: np.ndarray,
    y: np.ndarray,
    tipo: str,
    clave_publica_agregacion: bytes,
    semilla_ruido: int,
) -> ResultadoLocal:
    """Se ejecuta DENTRO del contenedor, junto a los datos del proveedor."""
    modelo = construir(tipo, X.shape[1])
    modelo.entrenar(X, y)

    cegados = noise.anadir_ruido(modelo.pesos, semilla_ruido)      # ① cegar
    payload = _serializar(cegados, len(X), tipo)
    return ResultadoLocal(
        cifrado=crypto.cifrar(clave_publica_agregacion, payload),  # ② cifrar
        n_muestras=len(X),
    )
# --8<-- [end:fase1]


# --8<-- [start:fase2]
def fase2_agregar(
    resultados: list[bytes],
    clave_privada_agregacion: bytes,
    ponderado: bool = True,
) -> bytes:
    """Se ejecuta DENTRO de otro contenedor. Ve numeros con ruido, nunca pesos."""
    modelos = [_deserializar(crypto.descifrar(clave_privada_agregacion, r)) for r in resultados]

    pesos_n = np.array([m["n_muestras"] for m in modelos], dtype=float)
    alfa = pesos_n / pesos_n.sum() if ponderado else np.full(len(modelos), 1 / len(modelos))

    global_ = {
        k: sum(a * m["pesos"][k] for a, m in zip(alfa, modelos))
        for k in modelos[0]["pesos"]
    }
    # `alfa` viaja en claro: sin el, quien quita el ruido no sabe con que pesos
    # se combino, y el descegado da basura (ver noise.quitar_ruido).
    return _serializar(global_, int(pesos_n.sum()), modelos[0]["tipo"], alfa)
# --8<-- [end:fase2]


# --8<-- [start:fase3]
def fase3_descegar(modelo_global: bytes, semillas: list[int]) -> dict:
    """Se ejecuta EN TU MAQUINA. Solo tu conoces las semillas."""
    d = _deserializar(modelo_global)
    d["pesos"] = noise.quitar_ruido(d["pesos"], semillas, d.get("alfa"))
    return d
# --8<-- [end:fase3]


def ejecutar_protocolo(particiones, tipo: str, ponderado: bool = True, semilla_base: int = 1000):
    """El flujo completo, tal y como lo orquesta la aplicacion de FELT."""
    sk_agg, pk_agg = crypto.generar_par_de_claves()
    semillas = [semilla_base + i for i in range(len(particiones))]

    locales = [
        fase1_entrenar(X, y, tipo, pk_agg, s) for (X, y), s in zip(particiones, semillas)
    ]
    global_cegado = fase2_agregar([r.cifrado for r in locales], sk_agg, ponderado)
    return fase3_descegar(global_cegado, semillas), locales, (sk_agg, pk_agg), semillas
