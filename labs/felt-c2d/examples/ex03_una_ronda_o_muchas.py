"""Ejemplo 03 - El limite real de FELT: una ronda, no muchas.

Pregunta: ¿cuanto cuesta que la agregacion sea tambien un job C2D?

FELT ejecuta UNA ronda: entrenar en local + agregar. La seccion 11 ejecuta N
rondas iterativas. Este ejemplo mide que se pierde con una sola.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d import crypto
from feltc2d.datos import hospitales
from feltc2d.models import construir
from feltc2d.protocol import fase1_entrenar, fase2_agregar, fase3_descegar


def federado_una_ronda(particiones, tipo, pasos):
    sk, pk = crypto.generar_par_de_claves()
    semillas = [3000 + i for i in range(len(particiones))]
    locales = []
    for (X, y), s in zip(particiones, semillas):
        m = construir(tipo, X.shape[1])
        m.pasos = pasos
        r = fase1_entrenar(X, y, tipo, pk, s)
        locales.append(r)
    g = fase2_agregar([r.cifrado for r in locales], sk)
    return fase3_descegar(g, semillas)["pesos"]


def federado_n_rondas(particiones, tipo, rondas, pasos_por_ronda):
    """Estilo seccion 11: el orquestador reenvia los pesos en cada ronda."""
    n_features = particiones[0][0].shape[1]
    global_ = construir(tipo, n_features).pesos
    n_total = sum(len(X) for X, _ in particiones)
    for _ in range(rondas):
        acum = {k: np.zeros_like(v) for k, v in global_.items()}
        for X, y in particiones:
            m = construir(tipo, n_features, {k: v.copy() for k, v in global_.items()})
            m.pasos = pasos_por_ronda
            m.entrenar(X, y)
            for k in acum:
                acum[k] += (len(X) / n_total) * m.pesos[k]
        global_ = acum
    return global_


def main() -> None:
    particiones, (X_test, y_test) = hospitales()
    n_features = X_test.shape[1]
    X_all = np.vstack([X for X, _ in particiones])
    y_all = np.concatenate([y for _, y in particiones])

    def acc(pesos):
        return construir("logistica", n_features, pesos).accuracy(X_test, y_test)

    central = construir("logistica", n_features)
    central.pasos = 3000
    central.entrenar(X_all, y_all)

    print("=" * 70)
    print("EJEMPLO 03 - Una ronda (FELT) frente a N rondas (seccion 11)")
    print("=" * 70)

    print("\n-- solo local: lo que consigue cada centro por su cuenta --")
    for i, (X, y) in enumerate(particiones):
        m = construir("logistica", n_features)
        m.pasos = 3000
        m.entrenar(X, y)
        print(f"   centro {i}  n={len(X):3d}   accuracy = {m.accuracy(X_test, y_test):.4f}")

    print("\n-- comparativa --")
    print(f"   {'escenario':<44}{'accuracy':>10}{'jobs C2D':>10}")
    print("   " + "-" * 64)
    print(f"   {'Centralizado (techo, ilegal en la practica)':<44}"
          f"{central.accuracy(X_test, y_test):>10.4f}{'-':>10}")

    p1 = federado_una_ronda(particiones, "logistica", 3000)
    print(f"   {'FELT - UNA ronda (3 locales + 1 agregacion)':<44}{acc(p1):>10.4f}{4:>10}")

    for r in (1, 3, 10, 30):
        p = federado_n_rondas(particiones, "logistica", r, 100)
        etiqueta = f"Iterativo - {r} ronda{'s' if r > 1 else ' '} x 100 pasos"
        print(f"   {etiqueta:<44}{acc(p):>10.4f}{r * 3:>10}")

    print("\n-- lectura --")
    print("   Con un modelo CONVEXO, una ronda casi alcanza el techo: promediar")
    print("   optimos locales cae cerca del unico optimo global.")
    print("   Con una red profunda esa garantia desaparece, y ahi es donde el")
    print("   coste de FELT (agregacion como job C2D) se vuelve un problema:")
    print("   cada ronda extra son K+1 jobs, no K.")


if __name__ == "__main__":
    main()
