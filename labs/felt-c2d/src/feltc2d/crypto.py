"""Cifrado NaCl box, compatible con el esquema de FELT Labs.

FELT cifra el modelo local con la clave publica del algoritmo de agregacion.
El esquema es el mismo que usa MetaMask (`eth-sig-util`): caja NaCl con clave
efimera, y el texto en claro codificado en ascii85 antes de cifrar para que el
resultado sea utf-8 valido.

Formato del mensaje cifrado:

    [32 bytes: clave publica efimera][ciphertext NaCl]
"""
from __future__ import annotations

from base64 import a85decode, a85encode

from nacl.public import Box, PrivateKey, PublicKey

# --8<-- [start:cifrado]
def generar_par_de_claves() -> tuple[bytes, bytes]:
    """Devuelve (privada, publica) como bytes de 32."""
    sk = PrivateKey.generate()
    return bytes(sk), bytes(sk.public_key)


def cifrar(clave_publica: bytes, datos: bytes) -> bytes:
    """Cifra `datos` de forma que solo el dueno de la privada pueda leerlos.

    Se genera una clave efimera nueva en cada llamada: el emisor no necesita
    identidad previa, y dos cifrados del mismo mensaje son distintos.
    """
    efimera = PrivateKey.generate()
    caja = Box(efimera, PublicKey(clave_publica))
    ciphertext = caja.encrypt(a85encode(datos))
    return bytes(efimera.public_key) + ciphertext


def descifrar(clave_privada: bytes, datos: bytes) -> bytes:
    """Descifra lo producido por `cifrar`."""
    efimera_pub, ciphertext = datos[:32], datos[32:]
    caja = Box(PrivateKey(clave_privada), PublicKey(efimera_pub))
    return a85decode(caja.decrypt(ciphertext))
# --8<-- [end:cifrado]
