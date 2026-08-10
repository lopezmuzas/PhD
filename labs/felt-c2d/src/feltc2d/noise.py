"""Ruido pseudoaleatorio determinista, la pieza clave del doble ciego de FELT.

Idea: el entrenador local suma al modelo un ruido generado a partir de una
semilla. Quien conoce la semilla puede regenerar exactamente el mismo ruido y
restarlo; quien no la conoce ve numeros sin sentido.

Como la suma es lineal, el ruido sobrevive al promedio:

    media(theta_k + r_k) = media(theta_k) + media(r_k)

asi que basta con restar la media de los ruidos al final.
"""
from __future__ import annotations

import hashlib

import numpy as np

RANGO = 1000.0  # amplitud del ruido; grande respecto a los pesos tipicos


# --8<-- [start:ruido]
def _semilla_derivada(semilla: int, nombre: str) -> int:
    """Deriva una semilla por parametro, para que dos vectores del mismo modelo
    no reciban el mismo ruido."""
    h = hashlib.sha256(f"{semilla};{nombre}".encode()).hexdigest()
    return int(h, 16) % (2**32 - 1)


def ruido_como(array: np.ndarray, semilla: int, nombre: str) -> np.ndarray:
    """Genera ruido con la misma forma que `array`, reproducible desde la semilla."""
    rng = np.random.default_rng(_semilla_derivada(semilla, nombre))
    return rng.uniform(-RANGO, RANGO, size=np.shape(array))


def anadir_ruido(pesos: dict[str, np.ndarray], semilla: int) -> dict[str, np.ndarray]:
    """Devuelve los pesos cegados. Irreversible sin la semilla."""
    return {k: v + ruido_como(v, semilla, k) for k, v in pesos.items()}


def quitar_ruido(
    pesos: dict[str, np.ndarray],
    semillas: list[int],
    alfa: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Quita la COMBINACION de ruidos con los MISMOS pesos que uso el agregador.

    Aqui esta la trampa que rompe una implementacion ingenua. Si el agregador
    hizo una media ponderada:

        global_cegado = SUM_k alfa_k * (theta_k + r_k)
                      = SUM_k alfa_k * theta_k  +  SUM_k alfa_k * r_k

    entonces hay que restar `SUM_k alfa_k * r_k`, no la media simple de los
    ruidos. Con `alfa` uniforme ambas coinciden, asi que el bug SOLO aparece
    cuando los nodos tienen tamanos distintos: justo el caso real.
    """
    if alfa is None:
        alfa = [1 / len(semillas)] * len(semillas)
    if len(alfa) != len(semillas):
        raise ValueError("alfa y semillas deben tener la misma longitud")

    fuera = {}
    for k, v in pesos.items():
        combinado = sum(a * ruido_como(v, s, k) for a, s in zip(alfa, semillas))
        fuera[k] = v - combinado
    return fuera
# --8<-- [end:ruido]
