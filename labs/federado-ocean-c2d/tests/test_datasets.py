"""Generadores y particionado. Reproducibilidad y ausencia de fugas."""

import numpy as np
import pytest

from fedlab.domain.datasets import (
    make_blobs,
    make_linear,
    make_patients,
    split_dirichlet,
    split_iid,
    train_test_split,
)


def test_generadores_son_reproducibles():
    for gen in (make_linear, make_blobs, make_patients):
        a, b = gen(seed=3), gen(seed=3)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_semillas_distintas_dan_datos_distintos():
    assert not np.array_equal(make_linear(seed=1)[0], make_linear(seed=2)[0])


def test_split_iid_no_pierde_ni_duplica_muestras():
    X, y = make_blobs(n=100, seed=0)
    parts = split_iid(X, y, 3, seed=0)
    assert sum(len(p[0]) for p in parts) == 100


def test_split_dirichlet_alpha_alto_es_casi_iid():
    X, y = make_blobs(n=1000, seed=0)
    parts = split_dirichlet(X, y, 4, alpha=1000.0, seed=0)
    fracs = [float((yy == 1).mean()) for _, yy in parts]
    assert max(fracs) - min(fracs) < 0.15


def test_split_dirichlet_alpha_bajo_genera_sesgo_fuerte():
    X, y = make_blobs(n=1000, seed=0)
    parts = split_dirichlet(X, y, 4, alpha=0.2, seed=1)
    fracs = [float((yy == 1).mean()) for _, yy in parts]
    assert max(fracs) - min(fracs) > 0.5


def test_split_dirichlet_falla_ruidosamente_con_nodo_vacio():
    """Mejor un error claro ahora que un FedAvg silenciosamente roto luego."""
    X, y = make_blobs(n=40, seed=0)
    with pytest.raises(ValueError, match="sin datos"):
        split_dirichlet(X, y, 30, alpha=0.01, seed=0)


def test_train_test_split_no_filtra_muestras():
    X, y = make_patients(n=200, seed=0)
    Xtr, _ytr, Xte, _yte = train_test_split(X, y, test_frac=0.25, seed=0)
    assert len(Xtr) + len(Xte) == 200
    # ninguna fila de test aparece en train
    train_rows = {tuple(r) for r in Xtr}
    assert not any(tuple(r) in train_rows for r in Xte)
