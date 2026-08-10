"""Ejemplo 02 - ¿Quien ve que? El doble ciego, medido.

Pregunta: ¿que informacion tiene realmente cada actor del sistema?

Este ejemplo NO entrena nada nuevo: audita el protocolo del ejemplo 01 y pone
numeros a lo que cada parte puede y no puede reconstruir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d import crypto, noise
from feltc2d.datos import hospitales
from feltc2d.models import construir
from feltc2d.protocol import _deserializar, fase1_entrenar, fase2_agregar, fase3_descegar


def main() -> None:
    particiones, _ = hospitales()
    sk_agg, pk_agg = crypto.generar_par_de_claves()
    semillas = [2001, 2002, 2003]

    reales = []
    for X, y in particiones:
        m = construir("logistica", X.shape[1])
        m.entrenar(X, y)
        reales.append(m.pesos["w"].copy())

    locales = [
        fase1_entrenar(X, y, "logistica", pk_agg, s)
        for (X, y), s in zip(particiones, semillas)
    ]

    print("=" * 72)
    print("EJEMPLO 02 - Quien ve que")
    print("=" * 72)

    print("\n{:<34}{:>10}{:>12}{:>14}".format(
        "ACTOR", "¿descifra?", "¿ve pesos?", "¿modelo real?"))
    print("-" * 72)
    print("{:<34}{:>10}{:>12}{:>14}".format(
        "Proveedor de datos (nodo k)", "-", "los suyos", "solo el suyo"))
    print("{:<34}{:>10}{:>12}{:>14}".format(
        "Quien pasa por la red", "NO", "no", "no"))
    print("{:<34}{:>10}{:>12}{:>14}".format(
        "Algoritmo de agregacion", "SI", "con ruido", "NO"))
    print("{:<34}{:>10}{:>12}{:>14}".format(
        "Cientifico de datos (tu)", "NO", "solo global", "solo global"))

    print("\n-- 1. Un observador de red intercepta el fichero del nodo 0 --")
    c = locales[0].cifrado
    print(f"   {len(c)} bytes. Sin la privada de agregacion no hay nada que hacer.")
    otra_sk, _ = crypto.generar_par_de_claves()
    try:
        crypto.descifrar(otra_sk, c)
        print("   ERROR: se descifro (no deberia)")
    except Exception as e:
        print(f"   intento con otra clave -> {type(e).__name__}: falla, como debe")

    print("\n-- 2. El agregador descifra. ¿Que aprende? --")
    for i, (r, real) in enumerate(zip(locales, reales)):
        visto = _deserializar(crypto.descifrar(sk_agg, r.cifrado))["pesos"]["w"]
        err = np.abs(visto - real)
        print(f"   nodo {i}: |w_visto - w_real| = "
              f"[{', '.join(f'{e:7.1f}' for e in err)}]")
    print("   Los pesos reales son de orden 1. El error es de orden 100.")
    print("   La relacion senal/ruido para el agregador es ~1/1000.")

    print("\n-- 3. Metadatos que SI se filtran --")
    for i, r in enumerate(locales):
        print(f"   nodo {i}: n_muestras = {r.n_muestras}  <- en claro, hace falta para ponderar")
    print("   El tamano del dataset NO esta protegido. Puede ser sensible.")

    print("\n-- 4. Tu, al final, ¿que tienes? --")
    global_cegado = fase2_agregar([r.cifrado for r in locales], sk_agg)
    final = fase3_descegar(global_cegado, semillas)
    media_real = sum(
        n * w for n, w in zip([len(X) for X, _ in particiones], reales)
    ) / sum(len(X) for X, _ in particiones)
    print(f"   w global recuperado : [{', '.join(f'{v:6.3f}' for v in final['pesos']['w'])}]")
    print(f"   w media ponderada   : [{', '.join(f'{v:6.3f}' for v in media_real)}]")
    print(f"   error maximo        : {np.abs(final['pesos']['w'] - media_real).max():.2e}")
    print("   Solo el global. Los locales son irrecuperables para ti.")

    print("\n-- 5. El limite honesto --")
    print("   Esto es OCULTACION, no privacidad diferencial. El modelo global")
    print("   sigue siendo una funcion de los datos: ataques de inferencia de")
    print("   pertenencia o reconstruccion sobre el global siguen sobre la mesa.")


if __name__ == "__main__":
    main()
