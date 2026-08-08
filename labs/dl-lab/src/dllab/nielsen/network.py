"""Red neuronal desde cero, solo con numpy.

Corresponde a los capítulos 1 y 2 del libro de Nielsen: red totalmente conectada,
neuronas sigmoides, coste cuadrático y descenso de gradiente estocástico con
backpropagation. Implementación propia, escrita para leerse de arriba abajo.

Todo lo que hay aquí cabe en unas 150 líneas. Esa es justamente la idea: una red
que reconoce dígitos manuscritos con ~95% de acierto no es un artefacto
inabarcable, y conviene haberla escrito una vez antes de usar un framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Función de activación
# --------------------------------------------------------------------------
def sigmoide(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^{-z}), aplicada elemento a elemento."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoide_prima(z: np.ndarray) -> np.ndarray:
    """Derivada de la sigmoide: σ'(z) = σ(z)·(1 − σ(z))."""
    s = sigmoide(z)
    return s * (1.0 - s)


# --------------------------------------------------------------------------
# Registro del entrenamiento
# --------------------------------------------------------------------------
@dataclass
class Historial:
    acierto_eval: list[float] = field(default_factory=list)
    coste_entrenamiento: list[float] = field(default_factory=list)
    segundos: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# La red
# --------------------------------------------------------------------------
class Red:
    """Red totalmente conectada con neuronas sigmoides.

    `tamaños` describe la arquitectura: [784, 30, 10] son 784 entradas
    (los píxeles de la imagen), una capa oculta de 30 neuronas y 10 salidas
    (una por dígito).
    """

    def __init__(self, tamaños: list[int], seed: int | None = None) -> None:
        self.tamaños = tamaños
        self.num_capas = len(tamaños)
        rng = np.random.default_rng(seed)

        # Un vector de sesgos por capa, salvo la de entrada (que no tiene).
        self.sesgos = [rng.standard_normal((y, 1)) for y in tamaños[1:]]
        # Una matriz de pesos por cada par de capas consecutivas.
        # w[l][j, k] conecta la neurona k de la capa l con la neurona j de la l+1.
        self.pesos = [
            rng.standard_normal((y, x)) for x, y in zip(tamaños[:-1], tamaños[1:])
        ]

    # ---------------------------------------------------------------- forward
    def propagar(self, a: np.ndarray) -> np.ndarray:
        """Devuelve la salida de la red para una entrada `a` de 784×1."""
        for b, w in zip(self.sesgos, self.pesos):
            a = sigmoide(w @ a + b)
        return a

    # --------------------------------------------------------------- backprop
    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
        """Gradiente del coste de UN ejemplo, respecto a sesgos y pesos.

        Devuelve (∂C/∂b, ∂C/∂w) con la misma estructura que self.sesgos y
        self.pesos. Es el corazón del algoritmo: ver el notebook 12.
        """
        grad_b = [np.zeros(b.shape) for b in self.sesgos]
        grad_w = [np.zeros(w.shape) for w in self.pesos]

        # --- Paso adelante: guardamos las z y las activaciones de cada capa
        activacion = x
        activaciones = [x]
        zs = []
        for b, w in zip(self.sesgos, self.pesos):
            z = w @ activacion + b
            zs.append(z)
            activacion = sigmoide(z)
            activaciones.append(activacion)

        # --- Error en la capa de salida  (BP1)
        #     δ^L = ∇_a C ⊙ σ'(z^L),  con ∇_a C = (a^L − y) para el coste cuadrático
        delta = (activaciones[-1] - y) * sigmoide_prima(zs[-1])
        grad_b[-1] = delta                              # (BP3)
        grad_w[-1] = delta @ activaciones[-2].T         # (BP4)

        # --- Propagación del error hacia atrás  (BP2)
        #     δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ'(z^l)
        for capa in range(2, self.num_capas):
            z = zs[-capa]
            delta = (self.pesos[-capa + 1].T @ delta) * sigmoide_prima(z)
            grad_b[-capa] = delta
            grad_w[-capa] = delta @ activaciones[-capa - 1].T

        return grad_b, grad_w

    # ------------------------------------------------------------ un minilote
    def actualizar_minilote(self, minilote: list, eta: float) -> None:
        """Un paso de descenso de gradiente usando la media del minilote."""
        suma_b = [np.zeros(b.shape) for b in self.sesgos]
        suma_w = [np.zeros(w.shape) for w in self.pesos]

        for x, y in minilote:
            db, dw = self.backprop(x, y)
            suma_b = [sb + d for sb, d in zip(suma_b, db)]
            suma_w = [sw + d for sw, d in zip(suma_w, dw)]

        m = len(minilote)
        self.pesos = [w - (eta / m) * sw for w, sw in zip(self.pesos, suma_w)]
        self.sesgos = [b - (eta / m) * sb for b, sb in zip(self.sesgos, suma_b)]

    # ------------------------------------------------------------------- SGD
    def sgd(
        self,
        entrenamiento: list,
        epocas: int,
        tam_minilote: int,
        eta: float,
        datos_eval: list | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ) -> Historial:
        """Descenso de gradiente estocástico.

        En cada época barajamos los datos, los partimos en minilotes y damos
        un paso de gradiente por minilote.
        """
        rng = np.random.default_rng(seed)
        entrenamiento = list(entrenamiento)
        n = len(entrenamiento)
        hist = Historial()

        for epoca in range(1, epocas + 1):
            t0 = time.perf_counter()
            rng.shuffle(entrenamiento)
            minilotes = [
                entrenamiento[k : k + tam_minilote] for k in range(0, n, tam_minilote)
            ]
            for minilote in minilotes:
                self.actualizar_minilote(minilote, eta)

            dt = time.perf_counter() - t0
            hist.segundos.append(dt)
            hist.coste_entrenamiento.append(self.coste_total(entrenamiento))

            if datos_eval is not None:
                aciertos = self.evaluar(datos_eval)
                acierto = aciertos / len(datos_eval)
                hist.acierto_eval.append(acierto)
                if verbose:
                    print(
                        f"Época {epoca:2d}: {aciertos} / {len(datos_eval)} "
                        f"({acierto:.2%})   [{dt:.1f}s]"
                    )
            elif verbose:
                print(f"Época {epoca:2d} completada [{dt:.1f}s]")

        return hist

    # ------------------------------------------------------------ evaluación
    def predecir(self, x: np.ndarray) -> int:
        """Dígito predicho: el índice de la neurona de salida más activa."""
        return int(np.argmax(self.propagar(x)))

    def evaluar(self, datos: list) -> int:
        """Número de ejemplos clasificados correctamente."""
        return sum(int(self.predecir(x) == y) for x, y in datos)

    def coste_total(self, datos: list) -> float:
        """Coste cuadrático medio sobre un conjunto con etiquetas one-hot."""
        total = 0.0
        for x, y in datos:
            a = self.propagar(x)
            total += 0.5 * float(np.sum((a - y) ** 2))
        return total / len(datos)


# --------------------------------------------------------------------------
# Comprobación numérica del gradiente
# --------------------------------------------------------------------------
def comprobar_gradiente(red: Red, x: np.ndarray, y: np.ndarray, eps: float = 1e-5):
    """Compara el gradiente de backprop con una aproximación por diferencias.

    Si backprop está bien implementado, la diferencia relativa debe rondar
    1e-9 o menos. Es la prueba que convierte "creo que está bien" en "está bien".
    """

    def coste(red_: Red) -> float:
        a = red_.propagar(x)
        return 0.5 * float(np.sum((a - y) ** 2))

    grad_b, grad_w = red.backprop(x, y)

    diffs = []
    for capa, w in enumerate(red.pesos):
        num = np.zeros(w.shape)
        it = np.ndindex(w.shape)
        for idx in it:
            original = w[idx]
            w[idx] = original + eps
            c_mas = coste(red)
            w[idx] = original - eps
            c_menos = coste(red)
            w[idx] = original
            num[idx] = (c_mas - c_menos) / (2 * eps)
        analitico = grad_w[capa]
        denom = np.linalg.norm(num) + np.linalg.norm(analitico)
        diffs.append(np.linalg.norm(num - analitico) / max(denom, 1e-12))
    return diffs
