"""Genera los CSV de los tres ejemplos, ya particionados por nodo.

Cada fichero es lo que un proveedor publicaria en Ocean. Formato: cabecera,
features, ULTIMA columna = etiqueta. Es lo que espera CsvSource.

    python scripts/make_datasets.py            # -> ./data/
    python scripts/make_datasets.py --out /tmp/x
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fedlab.adapters.sources import write_csv
from fedlab.domain.datasets import make_blobs, make_linear, make_patients, split_dirichlet

HOSPITALS = [("hospital_a", 110), ("hospital_b", 60), ("hospital_c", 30)]


def build(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # Ejemplo 01: tres tramos disjuntos del eje X
    for i, rng in enumerate([(-6, -2), (-2, 2), (2, 6)], start=1):
        X, y = make_linear(n=200, w=3.0, b=2.0, x_range=rng, seed=i)
        p = out / f"ex01_linear_node{i}.csv"
        write_csv(p, X, y)
        written.append(p)

    # Ejemplo 02: reparto no-IID con Dirichlet
    X, y = make_blobs(n=1200, n_features=2, separation=1.2, seed=42)
    for i, (Xp, yp) in enumerate(split_dirichlet(X, y, 4, alpha=0.3, seed=7), start=1):
        p = out / f"ex02_blobs_node{i}.csv"
        write_csv(p, Xp, yp)
        written.append(p)

    # Ejemplo 03: tres hospitales, repartidos por edad
    X, y = make_patients(n=sum(s for _, s in HOSPITALS), seed=11)
    order = np.argsort(X[:, 0])
    X, y = X[order], y[order]
    start = 0
    for name, size in HOSPITALS:
        p = out / f"ex03_{name}.csv"
        write_csv(p, X[start:start + size], y[start:start + size])
        written.append(p)
        start += size

    # Test independiente, solo para el orquestador. NUNCA se reparte.
    Xte, yte = make_patients(n=3000, seed=999)
    p = out / "ex03_holdout_orquestador.csv"
    write_csv(p, Xte, yte)
    written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for p in build(Path(args.out)):
        rows = sum(1 for _ in p.open()) - 1
        print(f"  {p}  ({rows} filas)")


if __name__ == "__main__":
    main()
