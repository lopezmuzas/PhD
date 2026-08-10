"""EJEMPLO 01 -- Regresion lineal federada. y = 3x + 2

El modelo tiene DOS numeros. Puedes ver la convergencia a ojo, sin graficas.

Lo interesante: cada nodo ve un tramo distinto del eje X. Ninguno por separado
puede estimar bien la recta; juntos, si. Ese es el argumento completo del
aprendizaje federado en 40 lineas.

    python examples/ex01_linear.py
"""

from fedlab.config import TrainConfig
from fedlab.domain.datasets import make_linear
from fedlab.entrypoints.simulate import simulate
from fedlab.learners.linear import LinearRegressor

W_TRUE, B_TRUE = 3.0, 2.0


def main():
    # Tres nodos, tres tramos disjuntos de X: NO-IID por construccion
    partitions = [
        make_linear(n=200, w=W_TRUE, b=B_TRUE, x_range=(-6, -2), seed=1),
        make_linear(n=200, w=W_TRUE, b=B_TRUE, x_range=(-2, 2), seed=2),
        make_linear(n=200, w=W_TRUE, b=B_TRUE, x_range=(2, 6), seed=3),
    ]

    print(f"Verdad oculta: w={W_TRUE}  b={B_TRUE}")
    print(f"{'ronda':>6} {'w':>9} {'b':>9} {'mse medio':>11}")

    def report(r, params, metrics):
        if r % 5 == 0 or r == 1:
            mse = sum(m["mse"] for m in metrics) / len(metrics)
            print(f"{r:>6} {params[0][0]:>9.4f} {params[1][0]:>9.4f} {mse:>11.4f}")

    final = simulate(
        partitions,
        learner_factory=lambda: LinearRegressor(n_features=1),
        cfg=TrainConfig(epochs=30, lr=0.02),
        rounds=40,
        on_round=report,
    )
    print(f"\nFinal: w={final[0][0]:.4f} (real {W_TRUE})  b={final[1][0]:.4f} (real {B_TRUE})")


if __name__ == "__main__":
    main()
