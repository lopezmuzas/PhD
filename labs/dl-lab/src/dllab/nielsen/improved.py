"""Red mejorada: entropía cruzada, regularización L2 y mejor inicialización.

Corresponde al capítulo 3 del libro. Misma arquitectura que `network.py`, pero
con los tres cambios que llevan MNIST del ~95% al ~98% sin tocar el número de
neuronas ni de épocas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dllab.nielsen.network import Historial, sigmoide, sigmoide_prima  # noqa: F401


# --------------------------------------------------------------------------
# Funciones de coste
# --------------------------------------------------------------------------
class CosteCuadratico:
    """C = ½‖a − y‖².  El del capítulo 1; aquí solo para comparar."""

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * float(np.sum((a - y) ** 2))

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        # El σ'(z) es justamente el causante del frenazo en el aprendizaje.
        return (a - y) * sigmoide_prima(z)


class CosteEntropiaCruzada:
    """C = −Σ [ y·ln(a) + (1−y)·ln(1−a) ].

    La gracia está en `delta`: al derivar, el σ'(z) se cancela con el
    denominador de la derivada del logaritmo y queda simplemente (a − y).
    Cuanto más equivocada está la neurona, más rápido aprende.
    """

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        # np.nan_to_num evita el inf cuando a llega exactamente a 0 o a 1.
        return float(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))))

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a - y


@dataclass
class HistorialCompleto:
    coste_entrenamiento: list[float] = field(default_factory=list)
    acierto_entrenamiento: list[float] = field(default_factory=list)
    coste_eval: list[float] = field(default_factory=list)
    acierto_eval: list[float] = field(default_factory=list)
    segundos: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------
class RedMejorada:
    def __init__(
        self,
        tamaños: list[int],
        coste=CosteEntropiaCruzada,
        init: str = "escalada",
        seed: int | None = None,
    ) -> None:
        self.tamaños = tamaños
        self.num_capas = len(tamaños)
        self.coste = coste
        self._rng = np.random.default_rng(seed)
        self.inicializar(init)

    def inicializar(self, init: str = "escalada") -> None:
        """`escalada`: pesos ~ N(0, 1/√n_entradas). `grande`: N(0,1), como en el cap. 1.

        Con N(0,1) y 784 entradas, la z de una neurona oculta tiene desviación
        típica ~√784 ≈ 28: casi todas arrancan saturadas, y una neurona saturada
        aprende despacio porque σ'(z) ≈ 0.
        """
        self.sesgos = [self._rng.standard_normal((y, 1)) for y in self.tamaños[1:]]
        if init == "escalada":
            self.pesos = [
                self._rng.standard_normal((y, x)) / np.sqrt(x)
                for x, y in zip(self.tamaños[:-1], self.tamaños[1:])
            ]
        elif init == "grande":
            self.pesos = [
                self._rng.standard_normal((y, x))
                for x, y in zip(self.tamaños[:-1], self.tamaños[1:])
            ]
        else:
            raise ValueError("init debe ser 'escalada' o 'grande'")

    # ---------------------------------------------------------------- forward
    def propagar(self, a: np.ndarray) -> np.ndarray:
        for b, w in zip(self.sesgos, self.pesos):
            a = sigmoide(w @ a + b)
        return a

    # --------------------------------------------------------------- backprop
    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
        grad_b = [np.zeros(b.shape) for b in self.sesgos]
        grad_w = [np.zeros(w.shape) for w in self.pesos]

        activacion, activaciones, zs = x, [x], []
        for b, w in zip(self.sesgos, self.pesos):
            z = w @ activacion + b
            zs.append(z)
            activacion = sigmoide(z)
            activaciones.append(activacion)

        # Única diferencia real con network.py: el delta lo decide la clase de coste.
        delta = self.coste.delta(zs[-1], activaciones[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = delta @ activaciones[-2].T

        for capa in range(2, self.num_capas):
            delta = (self.pesos[-capa + 1].T @ delta) * sigmoide_prima(zs[-capa])
            grad_b[-capa] = delta
            grad_w[-capa] = delta @ activaciones[-capa - 1].T
        return grad_b, grad_w

    # ------------------------------------------------------------ un minilote
    def actualizar_minilote(self, minilote, eta: float, lmbda: float, n: int) -> None:
        """n es el tamaño TOTAL del conjunto de entrenamiento (para la L2)."""
        suma_b = [np.zeros(b.shape) for b in self.sesgos]
        suma_w = [np.zeros(w.shape) for w in self.pesos]
        for x, y in minilote:
            db, dw = self.backprop(x, y)
            suma_b = [sb + d for sb, d in zip(suma_b, db)]
            suma_w = [sw + d for sw, d in zip(suma_w, dw)]

        m = len(minilote)
        # El factor (1 − ηλ/n) es el "decaimiento de pesos": en cada paso los
        # encoge un poco hacia cero, salvo que el gradiente lo compense.
        self.pesos = [
            (1 - eta * lmbda / n) * w - (eta / m) * sw
            for w, sw in zip(self.pesos, suma_w)
        ]
        self.sesgos = [b - (eta / m) * sb for b, sb in zip(self.sesgos, suma_b)]

    # ------------------------------------------------------------------- SGD
    def sgd(
        self,
        entrenamiento,
        epocas: int,
        tam_minilote: int,
        eta: float,
        lmbda: float = 0.0,
        datos_eval=None,
        monitorizar_entrenamiento: bool = False,
        paciencia: int = 0,
        verbose: bool = True,
        seed: int | None = None,
    ) -> HistorialCompleto:
        rng = np.random.default_rng(seed)
        entrenamiento = list(entrenamiento)
        n = len(entrenamiento)
        hist = HistorialCompleto()
        mejor, malas = -1.0, 0

        for epoca in range(1, epocas + 1):
            t0 = time.perf_counter()
            rng.shuffle(entrenamiento)
            for k in range(0, n, tam_minilote):
                self.actualizar_minilote(entrenamiento[k : k + tam_minilote], eta, lmbda, n)
            hist.segundos.append(time.perf_counter() - t0)

            msg = [f"Época {epoca:2d}"]
            if monitorizar_entrenamiento:
                c = self.coste_total(entrenamiento, lmbda, one_hot=True)
                a = self.acierto(entrenamiento, one_hot=True)
                hist.coste_entrenamiento.append(c)
                hist.acierto_entrenamiento.append(a)
                msg.append(f"train: coste {c:.4f} acierto {a:.2%}")
            if datos_eval is not None:
                c = self.coste_total(datos_eval, lmbda, one_hot=False)
                a = self.acierto(datos_eval, one_hot=False)
                hist.coste_eval.append(c)
                hist.acierto_eval.append(a)
                msg.append(f"eval: coste {c:.4f} acierto {a:.2%}")

                if a > mejor + 1e-6:
                    mejor, malas = a, 0
                else:
                    malas += 1
                    if paciencia and malas >= paciencia:
                        if verbose:
                            print(f"Parada temprana en la época {epoca} (mejor {mejor:.2%})")
                        break
            if verbose:
                print("  ".join(msg))
        return hist

    # ------------------------------------------------------------ evaluación
    def acierto(self, datos, one_hot: bool = False) -> float:
        if one_hot:
            ok = sum(
                int(np.argmax(self.propagar(x)) == np.argmax(y)) for x, y in datos
            )
        else:
            ok = sum(int(np.argmax(self.propagar(x)) == y) for x, y in datos)
        return ok / len(datos)

    def coste_total(self, datos, lmbda: float = 0.0, one_hot: bool = False) -> float:
        from dllab.nielsen.data import one_hot as a_one_hot

        total = 0.0
        for x, y in datos:
            a = self.propagar(x)
            objetivo = y if one_hot else a_one_hot(int(y), self.tamaños[-1])
            total += self.coste.fn(a, objetivo) / len(datos)
        total += 0.5 * (lmbda / len(datos)) * sum(
            float(np.linalg.norm(w) ** 2) for w in self.pesos
        )
        return total
