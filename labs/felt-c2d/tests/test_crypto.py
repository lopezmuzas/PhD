import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from feltc2d import crypto


def test_ida_y_vuelta():
    sk, pk = crypto.generar_par_de_claves()
    assert crypto.descifrar(sk, crypto.cifrar(pk, b"pesos")) == b"pesos"


def test_dos_cifrados_del_mismo_mensaje_difieren():
    _sk, pk = crypto.generar_par_de_claves()
    assert crypto.cifrar(pk, b"x") != crypto.cifrar(pk, b"x")


def test_otra_clave_no_descifra():
    _sk, pk = crypto.generar_par_de_claves()
    otra, _ = crypto.generar_par_de_claves()
    with pytest.raises(Exception):
        crypto.descifrar(otra, crypto.cifrar(pk, b"secreto"))


def test_prefijo_de_32_bytes():
    _sk, pk = crypto.generar_par_de_claves()
    assert len(crypto.cifrar(pk, b"")) >= 32
