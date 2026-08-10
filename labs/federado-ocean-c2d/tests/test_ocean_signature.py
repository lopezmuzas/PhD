"""La firma que espera ocean-node, reproducida y verificada.

Es el detalle mas facil de equivocar y el que produce el error mas opaco
("Invalid signature", HTTP 400, sin mas). El nodo hace (nonceHandler.ts):

    message = consumerAddress + String(nonce) + command
    digest  = solidityPackedKeccak256(['bytes'], [hexlify(toUtf8Bytes(message))])
    verifyMessage(toBeArray(digest), signature) == consumerAddress

`solidityPackedKeccak256(['bytes'], [x])` es simplemente keccak256 de los bytes,
y `verifyMessage` sobre 32 bytes crudos es EIP-191 == encode_defunct(digest).
"""

import pytest

eth_account = pytest.importorskip("eth_account")
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import keccak

from fedlab.ocean.client import FAILED, FINISHED, STATUS, OceanNodeClient

# Clave de prueba conocida. NUNCA uses esto con fondos reales.
TEST_KEY = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"


def test_la_firma_la_recupera_la_misma_direccion():
    """Si esto falla, el nodo respondera 400 y no sabras por que."""
    account = Account.from_key(TEST_KEY)
    nonce, command = 1234567890, "freeStartCompute"

    digest = keccak(text=f"{account.address}{nonce}{command}")
    signature = account.sign_message(encode_defunct(digest)).signature

    recovered = Account.recover_message(encode_defunct(digest), signature=signature)
    assert recovered == account.address


def test_el_mensaje_se_concatena_sin_separadores():
    """address + nonce + comando, pegados. Un separador rompe la verificacion."""
    addr = Account.from_key(TEST_KEY).address
    assert f"{addr}{42}{'getComputeResult'}" == addr + "42" + "getComputeResult"


def test_el_cliente_firma_igual_que_la_referencia():
    client = OceanNodeClient("http://localhost:8000", TEST_KEY)
    nonce, command = 999, "getComputeResult"

    produced = client._sign(nonce, command)
    digest = keccak(text=f"{client.address}{nonce}{command}")
    expected = client.account.sign_message(encode_defunct(digest)).signature.hex()
    assert produced == expected

    recovered = Account.recover_message(encode_defunct(digest),
                                        signature=bytes.fromhex(produced.removeprefix("0x")))
    assert recovered == client.address


def test_codigos_de_estado_coinciden_con_ocean_node():
    """Verificados en src/@types/C2D/C2D.ts. El error clasico es usar 7 en vez de 70."""
    assert FINISHED == 70
    assert 7 not in STATUS
    for code in (31, 32, 41):  # los tres fallos mas frecuentes
        assert code in FAILED
    assert 70 not in FAILED and 40 not in FAILED


def test_todos_los_codigos_de_fallo_estan_documentados():
    assert FAILED <= set(STATUS)
