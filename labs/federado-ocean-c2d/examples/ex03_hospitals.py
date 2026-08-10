"""EJEMPLO 03 -- Tres hospitales. La tabla que justifica el data space entero.

Tres hospitales con historiales que NO pueden compartir por normativa.
Comparamos cuatro mundos sobre el MISMO conjunto de test independiente:

    CENTRALIZADO   todos los datos juntos. Ilegal en la practica. Es el TECHO.
    FEDERADO       FedAvg ponderado. Los datos nunca salen de cada hospital.
    MEDIA SIMPLE   el bug clasico: promediar sin ponderar por num. de muestras.
    SOLO LOCAL     cada hospital por su cuenta. Es el SUELO.

Si el federado no queda claramente por encima del suelo, no hay motivo para
montar Ocean, Docker ni nada de esto. Esta tabla es el entregable real del
laboratorio -- la que va en la presentacion.

Detalle importante: los hospitales tienen tamanos MUY distintos (110/60/30) y
poblaciones distintas (repartidas por edad). El pequeno, en solitario, no tiene
datos suficientes para aprender nada util. Ahi es donde el federado brilla.

    python examples/ex03_hospitals.py
"""

import numpy as np

from fedlab.adapters.sources import ArraySource
from fedlab.adapters.stores import InMemoryStore
from fedlab.config import TrainConfig
from fedlab.domain.aggregation import fedavg, simple_mean
from fedlab.domain.datasets import make_patients
from fedlab.domain.round import run_local_round
from fedlab.learners.logistic import LogisticRegressor

N_FEATURES = 6
HOSPITALS = [("Hospital A", 110), ("Hospital B", 60), ("Hospital C", 30)]
CFG = TrainConfig(epochs=40, lr=0.4)
ROUNDS = 25
LONG = TrainConfig(epochs=ROUNDS * 40, lr=0.4)  # mismo presupuesto de gradientes


def build_partitions():
    """Reparte por edad: cada hospital atiende a una franja distinta.

    Sesgo sistematico, no aleatorio. Es lo que pasa de verdad -- un hospital
    pediatrico y uno geriatrico no ven la misma poblacion.
    """
    total = sum(size for _, size in HOSPITALS)
    X, y = make_patients(n=total, seed=11)
    order = np.argsort(X[:, 0])  # columna 0 = edad
    X, y = X[order], y[order]

    parts, start = [], 0
    for _, size in HOSPITALS:
        parts.append((X[start:start + size], y[start:start + size]))
        start += size
    return X, y, parts


def accuracy(params, X, y) -> float:
    model = LogisticRegressor(n_features=N_FEATURES)
    model.set_params(params)
    return model.evaluate(X, y)["accuracy"]


def federated(partitions, aggregator):
    """Bucle federado con el agregador inyectado. Cambiar de estrategia = un argumento."""
    global_params = None
    for _ in range(ROUNDS):
        updates = []
        for X, y in partitions:
            res = run_local_round(InMemoryStore(initial=global_params), ArraySource(X, y),
                                  LogisticRegressor(n_features=N_FEATURES), CFG)
            updates.append((res.params, res.num_samples))
        global_params = aggregator(updates)
    return global_params


def main():
    # Test grande e INDEPENDIENTE (otra semilla): nadie lo ha visto jamas.
    X_te, y_te = make_patients(n=3000, seed=999)
    X_all, y_all, partitions = build_partitions()

    central = LogisticRegressor(n_features=N_FEATURES)
    central.fit(X_all, y_all, LONG)
    acc_central = central.evaluate(X_te, y_te)["accuracy"]

    acc_fed = accuracy(federated(partitions, fedavg), X_te, y_te)
    acc_naive = accuracy(federated(partitions, simple_mean), X_te, y_te)

    local = []
    for (name, _), (Xp, yp) in zip(HOSPITALS, partitions):
        m = LogisticRegressor(n_features=N_FEATURES)
        m.fit(Xp, yp, LONG)
        local.append((name, len(Xp), m.evaluate(X_te, y_te)["accuracy"]))

    print("=" * 64)
    print(f"{'escenario':<44}{'accuracy':>12}")
    print("=" * 64)
    print(f"{'CENTRALIZADO (techo, ilegal en la practica)':<44}{acc_central:>12.4f}")
    print(f"{'FEDERADO - FedAvg ponderado':<44}{acc_fed:>12.4f}")
    print(f"{'FEDERADO - media simple (BUG)':<44}{acc_naive:>12.4f}")
    print("-" * 64)
    for name, n, acc in local:
        print(f"{f'SOLO LOCAL - {name} (n={n})':<44}{acc:>12.4f}")
    print("=" * 64)

    worst = min(a for _, _, a in local)
    best = max(a for _, _, a in local)
    print(f"\nEl hospital pequeno pasa de {worst:.4f} en solitario a {acc_fed:.4f} "
          f"federado: {100 * (acc_fed - worst):+.1f} puntos.")
    print(f"El federado incluso supera al mejor hospital en solitario: "
          f"{100 * (acc_fed - best):+.1f} puntos.")
    print(f"Y alcanza el techo centralizado sin mover un solo dato "
          f"({100 * (acc_fed - acc_central):+.1f} puntos).")
    print(f"\nUsar media simple en vez de ponderada cuesta "
          f"{100 * (acc_fed - acc_naive):+.1f} puntos: por eso el test "
          f"test_fedavg_pondera_por_muestras existe.")


if __name__ == "__main__":
    main()
