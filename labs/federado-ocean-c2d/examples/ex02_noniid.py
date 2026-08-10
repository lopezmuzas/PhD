"""EJEMPLO 02 -- El efecto NO-IID y la deriva de clientes (client drift).

Giramos UNA perilla: `alpha`, que controla como de desigual es el reparto de
clases entre nodos. Y medimos DOS cosas:

    accuracy      -- lo que todo el mundo mira
    divergencia   -- distancia L2 media entre los modelos que devuelven los nodos

El resultado sorprende, y por eso es la mejor leccion del laboratorio:
la accuracy apenas se mueve, pero la divergencia se multiplica por ~9.

Por que: la regresion logistica es CONVEXA. Tiene un unico minimo, asi que
promediar soluciones locales sigue cayendo cerca de el aunque los nodos vean
datos muy distintos. Una red neuronal profunda no tiene esa garantia: es
no-convexa y ademas dos redes distintas pueden representar la misma funcion con
pesos completamente diferentes, asi que promediarlas destroza el modelo.

Moraleja: si tu prueba de concepto federada usa un modelo lineal y funciona,
todavia no has probado nada sobre el modelo profundo que quieres desplegar.
La divergencia es la senal temprana que SI se traslada.

    python examples/ex02_noniid.py
"""

import numpy as np

from fedlab.adapters.sources import ArraySource
from fedlab.adapters.stores import InMemoryStore
from fedlab.config import TrainConfig
from fedlab.domain.aggregation import fedavg, params_delta
from fedlab.domain.datasets import make_blobs, split_dirichlet, train_test_split
from fedlab.domain.round import run_local_round
from fedlab.learners.logistic import LogisticRegressor

N_NODES, ROUNDS = 4, 20
CFG = TrainConfig(epochs=50, lr=0.5)


def mean_divergence(updates) -> float:
    """Distancia L2 media entre cada par de modelos locales."""
    pairs = [params_delta(a[0], b[0])
             for i, a in enumerate(updates) for b in updates[i + 1:]]
    return float(np.mean(pairs)) if pairs else 0.0


def run_alpha(partitions):
    global_params, divergences = None, []
    for _ in range(ROUNDS):
        updates = []
        for X, y in partitions:
            store = InMemoryStore(initial=global_params)
            res = run_local_round(store, ArraySource(X, y), LogisticRegressor(2), CFG)
            updates.append((res.params, res.num_samples))
        divergences.append(mean_divergence(updates))
        global_params = fedavg(updates)
    return global_params, float(np.mean(divergences))


def main():
    X, y = make_blobs(n=1200, n_features=2, separation=1.2, seed=42)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_frac=0.25, seed=42)

    central = LogisticRegressor(n_features=2)
    central.fit(X_tr, y_tr, TrainConfig(epochs=1000, lr=0.5))
    print(f"Centralizado (techo teorico): accuracy = "
          f"{central.evaluate(X_te, y_te)['accuracy']:.4f}\n")

    print(f"{'alpha':>7} {'% clase 1 por nodo':<26} {'accuracy':>9} {'divergencia':>12}")
    print("-" * 58)

    for alpha in (100.0, 1.0, 0.3, 0.1):
        try:
            partitions = split_dirichlet(X_tr, y_tr, N_NODES, alpha=alpha, seed=7)
        except ValueError as e:
            # Con alpha muy bajo es normal que un nodo se quede vacio.
            # Que falle aqui y no en la ronda 3 es parte del diseno.
            print(f"{alpha:>7.1f} {'(particion degenerada)':<26} {'--':>9} {'--':>12}")
            print(f"         {e}")
            continue

        shape = " ".join(f"{(yy == 1).mean():.2f}" for _, yy in partitions)
        params, divergence = run_alpha(partitions)

        model = LogisticRegressor(n_features=2)
        model.set_params(params)
        acc = model.evaluate(X_te, y_te)["accuracy"]
        print(f"{alpha:>7.1f} {shape:<26} {acc:>9.4f} {divergence:>12.3f}")

    print("\nalpha alto -> nodos parecidos  -> modelos locales casi identicos")
    print("alpha bajo -> nodos sesgados    -> cada nodo tira hacia un sitio distinto")
    print("La accuracy aguanta porque el modelo es convexo. La divergencia, no.")


if __name__ == "__main__":
    main()
