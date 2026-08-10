"""Ejemplo 01 - El protocolo de FELT, fase a fase.

Pregunta: ¿funciona el doble ciego, y sale un modelo util al final?

Tres nodos ven tramos DISJUNTOS del eje X, asi que ninguno puede estimar bien
la recta y = 3x + 2 en solitario.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d import crypto, noise
from feltc2d.datos import recta_por_tramos
from feltc2d.models import construir
from feltc2d.protocol import (
    _deserializar,
    fase1_entrenar,
    fase2_agregar,
    fase3_descegar,
)


def main() -> None:
    particiones = recta_por_tramos()
    sk_agg, pk_agg = crypto.generar_par_de_claves()
    semillas = [1001, 1002, 1003]

    print("=" * 68)
    print("EJEMPLO 01 - El protocolo de FELT, fase a fase   (verdad: w=3, b=2)")
    print("=" * 68)

    print("\n-- modelos LOCALES (lo que nadie llega a ver) --")
    for i, (X, y) in enumerate(particiones):
        m = construir("lineal", X.shape[1])
        m.entrenar(X, y)
        print(
            f"   nodo {i}  tramo x=[{X.min():.1f},{X.max():.1f}]  "
            f"n={len(X):3d}   w={m.pesos['w'][0]:8.4f}  b={m.pesos['b'][0]:8.4f}"
        )

    print("\n-- FASE 1: cada contenedor cega y cifra --")
    locales = [
        fase1_entrenar(X, y, "lineal", pk_agg, s)
        for (X, y), s in zip(particiones, semillas)
    ]
    for i, r in enumerate(locales):
        print(f"   nodo {i}  salida: {len(r.cifrado):4d} bytes cifrados, n={r.n_muestras}")
    print(f"   primeros bytes del nodo 0: {locales[0].cifrado[:16].hex()}...")

    print("\n-- lo que ve el AGREGADOR tras descifrar (con ruido) --")
    for i, r in enumerate(locales):
        d = _deserializar(crypto.descifrar(sk_agg, r.cifrado))
        print(f"   nodo {i}  w={d['pesos']['w'][0]:12.2f}  b={d['pesos']['b'][0]:12.2f}")

    print("\n-- FASE 2: agrega sin entender lo que agrega --")
    global_cegado = fase2_agregar([r.cifrado for r in locales], sk_agg)
    d = _deserializar(global_cegado)
    print(f"   global cegado:  w={d['pesos']['w'][0]:12.2f}  b={d['pesos']['b'][0]:12.2f}")

    print("\n-- FASE 3: en tu maquina, con tus semillas --")
    final = fase3_descegar(global_cegado, semillas)
    w, b = final["pesos"]["w"][0], final["pesos"]["b"][0]
    print(f"   modelo FINAL:   w={w:12.4f}  b={b:12.4f}")

    X_all = np.vstack([X for X, _ in particiones])
    y_all = np.concatenate([y for _, y in particiones])
    modelo = construir("lineal", 1, final["pesos"])
    central = construir("lineal", 1)
    central.entrenar(X_all, y_all)

    print("\n-- comprobacion --")
    print(f"   MSE del modelo federado      : {modelo.mse(X_all, y_all):.4f}")
    print(f"   MSE del centralizado (techo) : {central.mse(X_all, y_all):.4f}")

    print("\n-- que pasa sin las semillas correctas --")
    falso = fase3_descegar(global_cegado, [1, 2, 3])
    print(f"   con semillas equivocadas: w={falso['pesos']['w'][0]:.2f}  "
          f"b={falso['pesos']['b'][0]:.2f}   <- basura")


if __name__ == "__main__":
    main()
