import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from feltc2d import crypto
from feltc2d.datos import hospitales, recta_por_tramos, reparto_dirichlet
from feltc2d.models import construir
from feltc2d.protocol import (
    _deserializar,
    ejecutar_protocolo,
    fase1_entrenar,
    fase2_agregar,
    fase3_descegar,
)


def test_recupera_la_recta():
    final, *_ = ejecutar_protocolo(recta_por_tramos(), "lineal")
    assert abs(final["pesos"]["w"][0] - 3.0) < 0.2
    assert abs(final["pesos"]["b"][0] - 2.0) < 0.4


def test_el_agregador_no_ve_los_pesos_reales():
    particiones = recta_por_tramos()
    sk, pk = crypto.generar_par_de_claves()
    r = fase1_entrenar(*particiones[0], "lineal", pk, 99)
    visto = _deserializar(crypto.descifrar(sk, r.cifrado))["pesos"]["w"][0]
    assert abs(visto - 3.0) > 10  # lo que ve no se parece al peso real


def test_semillas_equivocadas_dan_basura():
    particiones = recta_por_tramos()
    sk, pk = crypto.generar_par_de_claves()
    semillas = [1, 2, 3]
    locales = [fase1_entrenar(X, y, "lineal", pk, s) for (X, y), s in zip(particiones, semillas)]
    g = fase2_agregar([l.cifrado for l in locales], sk)
    assert abs(fase3_descegar(g, [9, 9, 9])["pesos"]["w"][0] - 3.0) > 10


def test_ponderado_equivale_a_la_media_ponderada_real():
    particiones, _ = hospitales()
    final, *_ = ejecutar_protocolo(particiones, "logistica")
    n = np.array([len(X) for X, _ in particiones], dtype=float)
    esperado = np.zeros(particiones[0][0].shape[1])
    for (X, y), ni in zip(particiones, n):
        m = construir("logistica", X.shape[1])
        m.entrenar(X, y)
        esperado += (ni / n.sum()) * m.pesos["w"]
    assert np.allclose(final["pesos"]["w"], esperado, atol=1e-8)


def test_federado_mejora_al_centro_pequeno():
    particiones, (Xt, yt) = hospitales()
    final, *_ = ejecutar_protocolo(particiones, "logistica")
    n_features = Xt.shape[1]
    global_acc = construir("logistica", n_features, final["pesos"]).accuracy(Xt, yt)

    X, y = particiones[-1]  # el centro mas pequeno
    solo = construir("logistica", n_features)
    solo.entrenar(X, y)
    assert global_acc > solo.accuracy(Xt, yt) + 0.10


def test_media_simple_es_peor_que_ponderada():
    particiones, (Xt, yt) = hospitales()
    n_features = Xt.shape[1]
    pond, *_ = ejecutar_protocolo(particiones, "logistica", ponderado=True)
    simple, *_ = ejecutar_protocolo(particiones, "logistica", ponderado=False)
    a = construir("logistica", n_features, pond["pesos"]).accuracy(Xt, yt)
    b = construir("logistica", n_features, simple["pesos"]).accuracy(Xt, yt)
    assert a >= b


def test_dirichlet_reparte_todo():
    particiones, _ = hospitales()
    X = np.vstack([x for x, _ in particiones])
    y = np.concatenate([yy for _, yy in particiones])
    trozos = reparto_dirichlet(X, y, n_nodos=4, alpha=0.3)
    assert sum(len(x) for x, _ in trozos) == len(X)
