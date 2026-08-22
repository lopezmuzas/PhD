"""Minimal reinforcement-learning harness: sibling of harness.py, not a replacement.

ES: Arnés de refuerzo. El hermano pequeño de `harness.py`. Deliberadamente
mínimo: implementa REINFORCE (gradiente de política puro) y nada más.

Design rule / Regla de diseño:
    harness.py    → los datos EXISTEN antes de entrenar (dataset en disco).
    harness_rl.py → los datos LOS GENERA la política actual (rollouts).

    Todo lo demás se reutiliza: Registry, Callback, TrainingState, save_run,
    load_run, compare_runs, plot_runs. No se duplica una sola línea de eso.

Comment convention / Convenio de comentarios:
    Docstrings in English describe WHAT. Spanish notes explain WHY.

    Marcador `# ALTERNATIVA:` → aquí había una bifurcación de diseño. Se eligió
    la opción más simple de entender, NO la mejor. Al final del archivo hay un
    mapa completo de lo que este arnés deliberadamente no hace.

═══════════════════════════════════════════════════════════════════════════════
⚠️  ESTE ARCHIVO NO ES UNA REFERENCIA DE CÓMO SE HACE RL EN PRODUCCIÓN
═══════════════════════════════════════════════════════════════════════════════
Es la versión más pequeña que sigue siendo honesta: se puede leer entera de una
sentada y se ve de dónde sale cada gradiente. Todo lo que hace falta para que
RL funcione de verdad (PPO, ventajas con GAE, un crítico, replay buffers,
entornos vectorizados) está deliberadamente FUERA, y anotado como tal.

Cuando el laboratorio se quede corto, la salida NO es hacer crecer este archivo:
es cambiar a `CleanRL` (un archivo por algoritmo, pensado para leerse),
`Stable-Baselines3` (API estable, para usar) o `TorchRL` (componible, de PyTorch).
Para alineación de LLMs, `TRL`. Ver §🔀 al final.

La idea central, en una frase:
    El gradiente de política es aprendizaje supervisado sobre tus propias
    muestras, ponderado por lo bien que salieron.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

# ES: Importamos el arnés entero en vez de copiar piezas. Si `save_run` mejora
# allí, mejora aquí gratis. Es la razón de ser de "módulo hermano".
from lab import harness as H

# ES: Reutilizamos los registros de modelos y optimizadores del arnés: una
# política ES un modelo, y Adam es Adam. Solo hace falta un registro nuevo.
envs = H.Registry("env")


# ES: `harness.py` deja el registro de optimizadores vacío y espera que el
# notebook lo rellene. Para que este módulo funcione también en un script suelto
# lo rellenamos aquí, pero SIN sobrescribir: si el notebook ya registró su
# versión, manda la suya.
if "adam" not in H.optimizers:
    @H.optimizers.register("adam")
    def build_adam(params, lr=1e-3, **kwargs):
        return torch.optim.Adam(params, lr=lr, **kwargs)

if "sgd" not in H.optimizers:
    @H.optimizers.register("sgd")
    def build_sgd(params, lr=1e-2, momentum=0.9, **kwargs):
        return torch.optim.SGD(params, lr=lr, momentum=momentum, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# The environment contract
# ─────────────────────────────────────────────────────────────────────────────
class Env(Protocol):
    """What this harness needs from an environment. Three methods, two fields.

    ES: Es un Protocol (tipado estructural), no una clase base. Cualquier objeto
    con estos métodos vale, incluido un `gymnasium.Env` envuelto en 10 líneas.
    Se ha copiado la firma de Gym a propósito: es el estándar de facto, así que
    lo que aprendas aquí se transfiere a Stable-Baselines3 sin traducir nada.
    """

    obs_dim: int      # tamaño del vector de estado que ve la política
    n_actions: int    # número de acciones discretas disponibles

    def reset(self) -> torch.Tensor:
        """Start a new episode. Returns the initial state as a float tensor."""
        ...

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """Apply one action. Returns (next_state, reward, done)."""
        ...


# ALTERNATIVA: acciones continuas (mover un brazo robótico, girar un volante).
# Requieren que la política devuelva media y desviación de una Normal en vez de
# logits de una Categorical. Es un cambio de 5 líneas en `collect_episode`, pero
# se ha dejado fuera para que solo haya UN camino que leer.


# ─────────────────────────────────────────────────────────────────────────────
# Two toy environments, chosen to isolate one idea each
# ─────────────────────────────────────────────────────────────────────────────
@envs.register("two_armed_bandit")
def build_bandit(p_left: float = 0.2, p_right: float = 0.8, **kwargs):
    """One state, two actions, one episode = one step. The RL 'hello world'.

    ES: No hay estado ni crédito temporal que repartir: la política solo tiene
    que descubrir cuál de las dos palancas paga más. Sirve como prueba de humo:
    si esto no converge a ~p_right, el bug está en el gradiente, no en la tarea.
    """

    class Bandit:
        obs_dim, n_actions = 1, 2

        def reset(self):
            return torch.zeros(1)  # ES: estado constante: no hay nada que observar

        def step(self, action):
            prob = p_right if action == 1 else p_left
            reward = float(torch.rand(1).item() < prob)
            return torch.zeros(1), reward, True  # done=True siempre: 1 paso = 1 episodio

    return Bandit()


@envs.register("corridor")
def build_corridor(length: int = 5, step_cost: float = 0.05,
                   quit_reward: float = 0.3, **kwargs):
    """Walk right for a big delayed prize, or quit left for a small instant one.

    ES: El dilema de la paciencia, que es el problema central del refuerzo.
    Ir a la izquierda termina el episodio YA con `quit_reward` seguro. Ir a la
    derecha cuesta `step_cost` en cada paso y solo paga (+1) al llegar al final.

    Lo bonito de este entorno: cambiando SOLO `gamma` se invierte la política
    óptima, sin tocar nada más. Con gamma alto conviene aguantar hasta el premio
    grande; con gamma bajo el futuro se descuenta tanto que conviene cobrar ya.
    Es la mejor forma de ver que gamma no es un detalle de implementación, es
    parte de la DEFINICIÓN del problema que estás resolviendo.
    """

    class Corridor:
        obs_dim, n_actions = 1, 2  # acciones: 0 = cobrar y salir, 1 = seguir

        def reset(self):
            self.position = 0
            return self._observe()

        def _observe(self):
            # ES: Normalizado a [0, 1]. Las redes odian entradas sin escalar.
            return torch.tensor([self.position / length], dtype=torch.float32)

        def step(self, action):
            if action == 0:
                return self._observe(), quit_reward, True  # cobra poco, pero seguro
            self.position += 1
            reached_goal = self.position >= length
            reward = (1.0 if reached_goal else 0.0) - step_cost
            return self._observe(), reward, reached_goal

    return Corridor()


# ES: Aviso honesto sobre los DOS entornos de arriba: en ninguno hace falta que
# la política MIRE el estado (la acción óptima es la misma en todas las
# posiciones). Falta por tanto un tercer escalón —una tarea donde lo correcto
# depende de dónde estés— y se ha dejado fuera a propósito para no crecer. Si al
# construir el tuyo la política ignora la observación y aun así gana, la tarea
# no está midiendo lo que crees.


# ES: El entorno interesante —el LM que genera texto y un clasificador que lo
# puntúa— NO vive aquí, y es una decisión, no un olvido. Depende del vocabulario
# y del checkpoint concretos de un notebook, así que se registra desde allí:
#
#     @rl.envs.register("texto_con_reward_model")
#     def build_text_env(reward_run_id, **kwargs):
#         judge = cargar_clasificador(f"runs/{reward_run_id}/weights.pt").eval()
#         ...  # acción = emitir un token; reward = judge(texto_generado)
#
# Es el mismo patrón que `harness.py` usa para los datasets: la librería trae el
# contrato, el notebook trae el problema.


# ─────────────────────────────────────────────────────────────────────────────
# A default policy
# ─────────────────────────────────────────────────────────────────────────────
@H.models.register("mlp_policy")
def build_mlp_policy(obs_dim: int, n_actions: int, hidden: int = 32, **kwargs):
    """State → logits over actions. Emits logits, never probabilities.

    ES: Sin Softmax final, igual que en clasificación: `Categorical(logits=...)`
    lo aplica internamente y de forma numéricamente estable. Es exactamente la
    misma regla de la tabla "activación final ↔ pérdida" de PYTORCH_EN_EL_ARNES.
    """
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, n_actions),
    )
    # ALTERNATIVA: en RL se usa Tanh más que ReLU. No es superstición: las
    # entradas no están normalizadas por un DataLoader y Tanh acota la señal.
    # Es el tipo de detalle que las librerías serias traen ya decidido.


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    """One episode: what the policy did and what it got for it.

    ES: Esto es el equivalente de un lote del DataLoader, con la diferencia que
    lo cambia todo: no se lee de disco, lo acaba de generar la política. En
    cuanto damos un paso de optimizador, esta trayectoria queda OBSOLETA —
    la generó una política que ya no existe. De ahí el nombre "on-policy".
    """

    log_probs: list[torch.Tensor] = field(default_factory=list)  # log π(a|s), con grafo
    rewards: list[float] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    def __len__(self) -> int:
        return len(self.rewards)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout: where the data comes from
# ─────────────────────────────────────────────────────────────────────────────
def collect_episode(policy: nn.Module, env, device: torch.device,
                    max_steps: int = 100, greedy: bool = False) -> Trajectory:
    """Run one episode. This is the function harness.py does not have.

    ES: Aquí está la diferencia estructural entera. En `harness.py` esta función
    no existe porque su equivalente es `datasets.build()`, que corre UNA vez
    antes del bucle. Esta corre en cada iteración, y el modelo es un ingrediente.
    """
    trajectory = Trajectory()
    state = env.reset().to(device)

    for _ in range(max_steps):
        distribution = Categorical(logits=policy(state))

        if greedy:
            # ES: Evaluación determinista: la mejor acción, sin explorar. Es el
            # análogo de `model.eval()`, pero fíjate en que aquí NO basta con
            # `.eval()`: la aleatoriedad no está en el Dropout, está en cómo
            # elegimos la acción. Es una decisión nuestra, no una capa de PyTorch.
            action = distribution.probs.argmax()
        else:
            action = distribution.sample()

        trajectory.log_probs.append(distribution.log_prob(action))
        trajectory.entropies.append(distribution.entropy())

        state, reward, done = env.step(int(action.item()))
        state = state.to(device)
        trajectory.rewards.append(reward)
        if done:
            break

    return trajectory
    # ALTERNATIVA: entornos vectorizados. Esto ejecuta UN episodio con UN
    # forward por paso: es lo más lento posible. Lo real es correr 64 entornos
    # en paralelo y hacer un forward por lote. Es el 90% de la ingeniería de
    # Stable-Baselines3 y no aporta nada conceptual, así que aquí no está.


# ─────────────────────────────────────────────────────────────────────────────
# Credit assignment: turning rewards into per-step weights
# ─────────────────────────────────────────────────────────────────────────────
def returns_to_go(rewards: list[float], gamma: float = 1.0) -> list[float]:
    """G_t = sum of future rewards from t onward, discounted by gamma.

    ES: Se recorre hacia ATRÁS porque el mérito de un paso depende de lo que
    vino después, no de lo que vino antes. Un paso se juzga por su futuro.

    gamma < 1 dice "prefiero recompensa pronto"; gamma = 1 dice "me da igual
    cuándo". En episodios cortos apenas cambia nada; en episodios largos gamma
    es el hiperparámetro que más duele equivocar.
    """
    returns, running = [], 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return list(reversed(returns))


def compute_weights(returns: torch.Tensor, mode: str = "normalized") -> torch.Tensor:
    """Turn returns into the scalar that multiplies each log-prob.

    ES: ESTA es la función donde vive toda la literatura de RL. Las tres opciones
    de abajo son los tres primeros escalones de una escalera muy larga, y todas
    responden a la misma pregunta: "¿bueno comparado con QUÉ?".
    """
    if mode == "return":
        # ES: El REINFORCE de 1992, literal. Funciona y tiene un problema grave:
        # si todas las recompensas son positivas, TODAS las acciones se refuerzan
        # (unas más que otras). Aprende, pero despacio y con varianza enorme.
        return returns

    if mode == "baseline":
        # ES: Restar la media convierte "bueno" en "mejor que la media". Ahora
        # las acciones peores que el promedio reciben peso NEGATIVO y se
        # desalientan de verdad. Matemáticamente el gradiente sigue siendo
        # correcto (restar una constante no lo sesga) y la varianza cae mucho.
        return returns - returns.mean()

    if mode == "normalized":
        # ES: Además dividir por la desviación. Deja los pesos en torno a ±1, lo
        # que desacopla la tasa de aprendizaje de la escala de las recompensas.
        # Es un truco práctico, no un teorema: técnicamente SÍ sesga el gradiente.
        # Todo el mundo lo usa igualmente porque funciona.
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    raise ValueError(f"unknown advantage mode '{mode}'. "
                     f"Use 'return', 'baseline' or 'normalized'.")

    # ALTERNATIVA — los escalones siguientes de la escalera, en orden:
    #
    #  4. Baseline aprendida  V(s): una segunda red predice "cuánto esperaba
    #     sacar desde este estado". Ventaja = G_t - V(s_t). Esto convierte
    #     REINFORCE en Actor-Crítico. Es el salto grande, y el que este arnés
    #     no puede dar sin dejar de ser mínimo: aparece un segundo modelo, un
    #     segundo optimizador y una segunda pérdida.
    #  5. GAE (Generalized Advantage Estimation): interpola entre "usa solo el
    #     paso siguiente" (sesgado, poca varianza) y "usa el episodio entero"
    #     (sin sesgo, mucha varianza) con un parámetro lambda.
    #  6. PPO: además impide que la política se mueva demasiado en un solo paso,
    #     recortando la razón entre la política nueva y la vieja. Es lo que hace
    #     que RL sea estable en la práctica, y es el estándar actual.
    #
    # Los escalones 4-6 NO son mejoras cosméticas: son la diferencia entre
    # "converge en un juguete" y "converge en un problema real".


# ─────────────────────────────────────────────────────────────────────────────
# The loss: supervised learning, weighted
# ─────────────────────────────────────────────────────────────────────────────
def policy_loss(log_probs: torch.Tensor, weights: torch.Tensor,
                entropies: torch.Tensor | None = None,
                entropy_coef: float = 0.0) -> torch.Tensor:
    """-(log π(a|s) · weight).mean(), optionally minus an entropy bonus.

    ES: Compara con la entropía cruzada de `harness.py`:

        supervisado:  loss = -log π(etiqueta_humana)          # peso implícito 1
        refuerzo:     loss = -log π(accion_propia) * peso     # peso = qué tal salió

    Es la MISMA fórmula. Cambian dos cosas: la diana la eligió el propio modelo,
    y cada muestra lleva un peso escalar. Ese `* weights` es, literalmente, todo
    lo que separa el aprendizaje supervisado del gradiente de política.

    Y explica por qué esto no cabe en `harness.py` tal cual: no existe ninguna
    función en `torch.nn.functional` con esta firma, y el arnés construye su
    pérdida con `getattr(torch.nn.functional, config["loss"])`.
    """
    loss = -(log_probs * weights).mean()

    if entropy_coef and entropies is not None:
        # ES: Premia mantener las probabilidades repartidas, o sea, seguir
        # explorando. Sin esto la política puede volverse determinista muy
        # pronto, quedarse con lo primero que funcionó y no encontrar nada mejor.
        # Está aquí con valor por defecto 0.0 para que el algoritmo base se lea
        # puro, y para que se vea que es un PARCHE añadido, no parte de la teoría.
        loss = loss - entropy_coef * entropies.mean()

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RLComponents:
    """Everything the RL loop needs. Note what is NOT here: a DataLoader."""

    env: object
    policy: nn.Module
    optimizer: torch.optim.Optimizer


def build_rl_components(config: dict, seed: int) -> RLComponents:
    """Instantiate env, policy and optimizer from the config."""
    H.set_seed(seed)
    env = envs.build(config["env"], **config.get("env_args", {}))

    # ES: Desviación consciente respecto a `harness.py`, donde el dataset y el
    # modelo se construyen de forma independiente. Aquí la política NO se puede
    # construir sin conocer el entorno: sus dimensiones de entrada y salida las
    # dicta él. Por eso se inyectan `obs_dim` y `n_actions`.
    # ALTERNATIVA: escribirlos a mano en `model_args`. Más explícito y más
    # frágil: si cambias el entorno y olvidas actualizarlos, el error que sale
    # es un fallo de forma de matriz a 3 capas de profundidad.
    policy = H.models.build(config["model"], obs_dim=env.obs_dim,
                            n_actions=env.n_actions, **config.get("model_args", {}))

    optimizer = H.optimizers.build(
        config.get("optimizer", "adam"),
        params=policy.parameters(),
        **config.get("optimizer_args", {"lr": 1e-2}),
    )
    return RLComponents(env, policy, optimizer)


def train_one_iteration(components: RLComponents, state: H.TrainingState,
                        callbacks: Iterable[H.Callback]) -> dict:
    """Collect episodes with the current policy, then take ONE gradient step.

    ES: Compáralo con `train_one_epoch()`. Allí el bucle es sobre datos que ya
    existían; aquí la primera mitad de la función FABRICA los datos. Y son de un
    solo uso: tras `optimizer.step()` la política ha cambiado, así que estas
    trayectorias ya no describen a nadie y se tiran. Esa es la ineficiencia
    fundamental del gradiente de política on-policy.
    """
    config = state.config
    components.policy.train()

    # ── 1. Generar los datos con la política actual ──
    trajectories = [
        collect_episode(components.policy, components.env, state.device,
                        max_steps=config.get("max_steps", 100))
        for _ in range(config.get("episodes_per_iteration", 16))
    ]

    # ── 2. Repartir el mérito: recompensas → retornos → pesos ──
    gamma = config.get("gamma", 1.0)
    flat_returns, flat_log_probs, flat_entropies = [], [], []
    for trajectory in trajectories:
        flat_returns += returns_to_go(trajectory.rewards, gamma)
        flat_log_probs += trajectory.log_probs
        flat_entropies += trajectory.entropies

    returns = torch.tensor(flat_returns, dtype=torch.float32, device=state.device)
    weights = compute_weights(returns, config.get("advantage", "normalized"))

    # ES: `weights` NO debe llevar gradiente. Es un juicio sobre lo que pasó, no
    # una cantidad a optimizar. Aquí sale de un tensor sin grafo, pero en cuanto
    # la recompensa venga de un modelo (un reward model) hará falta `no_grad()`
    # explícito: si no, el gradiente intentaría "mejorar" al juez en vez de al
    # alumno. Es uno de los errores más fáciles de cometer en RLHF.
    weights = weights.detach()

    # ── 3. Un solo paso de optimizador. Los 4 pasos sagrados, idénticos ──
    loss = policy_loss(
        torch.stack(flat_log_probs), weights,
        entropies=torch.stack(flat_entropies),
        entropy_coef=config.get("entropy_coef", 0.0),
    )
    components.optimizer.zero_grad()
    loss.backward()

    if config.get("grad_clip"):
        # ES: En RL la norma del gradiente pega picos brutales cuando aparece
        # una recompensa inesperada. Recortar evita que un episodio afortunado
        # destruya la política. En supervisado es opcional; aquí, casi siempre no.
        nn.utils.clip_grad_norm_(components.policy.parameters(), config["grad_clip"])

    components.optimizer.step()
    state.step += 1
    for callback in callbacks:
        callback.on_batch_end(state, loss.item())

    episode_rewards = np.array([t.total_reward for t in trajectories])
    return {
        "policy_loss": loss.item(),
        "reward_mean": float(episode_rewards.mean()),
        "reward_std": float(episode_rewards.std()),
        "episode_len": float(np.mean([len(t) for t in trajectories])),
        "entropy": float(torch.stack(flat_entropies).mean().item()),
    }


@torch.no_grad()
def evaluate_greedy(components: RLComponents, device: torch.device,
                    n_episodes: int = 20, max_steps: int = 100) -> dict:
    """Mean return of the deterministic (argmax) policy.

    ES: El análogo de `evaluate()`, y ojo con la diferencia: la política de
    entrenamiento EXPLORA (muestrea) y la de evaluación EXPLOTA (argmax). Son
    dos políticas distintas sacadas de los mismos pesos. En supervisado no hay
    nada parecido: `model.eval()` no cambia lo que el modelo cree, solo apaga
    Dropout y BatchNorm.
    """
    components.policy.eval()
    rewards = [
        collect_episode(components.policy, components.env, device,
                        max_steps=max_steps, greedy=True).total_reward
        for _ in range(n_episodes)
    ]
    return {"reward": float(np.mean(rewards))}


def run_rl_experiment(config: dict, seed: int | None = None,
                      callbacks: list[H.Callback] | None = None,
                      save: bool = True, verbose: bool = True) -> H.ExperimentResult:
    """Train a policy end to end. Same signature and return type as run_experiment.

    ES: Devuelve un `H.ExperimentResult`, no un tipo nuevo, y eso es a propósito:
    así `save_run`, `load_run`, `compare_runs` y `plot_runs` funcionan sin
    tocarlos. La infraestructura del arnés se reutiliza al 100%; lo único que
    cambia es el bucle.
    """
    started_at = time.time()
    seed = config.get("seed", 0) if seed is None else seed
    callbacks = callbacks or []
    device = H.resolve_device(config)

    components = build_rl_components(config, seed)
    components.policy.to(device)

    # ES: `H.TrainingState` sirve tal cual, sin cambiar un campo. Es la prueba de
    # que el sistema de callbacks del arnés ya era genérico: mira los pesos y el
    # historial, y eso no depende de dónde vengan los datos.
    state = H.TrainingState(config=config, model=components.policy,
                            optimizer=components.optimizer, device=device)
    for callback in callbacks:
        callback.on_train_start(state)

    # ES: El config dice "iterations", no "epochs". No es un capricho: una época
    # es "una pasada por el dataset", y aquí no hay dataset por el que pasar.
    total_iterations = config["iterations"]
    log_every = max(1, total_iterations // 5)

    for iteration in range(total_iterations):
        state.epoch = iteration
        metrics = train_one_iteration(components, state, callbacks)
        eval_metrics = evaluate_greedy(components, device,
                                       n_episodes=config.get("eval_episodes", 20),
                                       max_steps=config.get("max_steps", 100))

        # ES: La columna se llama "epoch" aunque sea una iteración, solo para que
        # `H.plot_runs` la encuentre. Es una fricción real de reutilizar el
        # arnés supervisado, y se deja a la vista en vez de esconderla.
        state.history.append({"epoch": iteration, **metrics,
                              **{f"eval_{k}": v for k, v in eval_metrics.items()}})
        for callback in callbacks:
            callback.on_epoch_end(state)

        if verbose and (iteration % log_every == 0 or iteration == total_iterations - 1):
            print(f"  iter {iteration:3d}  reward {metrics['reward_mean']:+.3f}"
                  f" ± {metrics['reward_std']:.3f}   greedy {eval_metrics['reward']:+.3f}"
                  f"   loss {metrics['policy_loss']:+.4f}   H {metrics['entropy']:.3f}")

    for callback in callbacks:
        callback.on_train_end(state)

    result = H.ExperimentResult(
        run_id=H.make_run_id(config, seed),
        config=config, seed=seed, history=state.history,
        model=components.policy, scratch=state.scratch,
        elapsed_seconds=round(time.time() - started_at, 2),
    )
    if save:
        H.save_run(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Comparing: the one helper that could NOT be reused as-is
# ─────────────────────────────────────────────────────────────────────────────
def compare_rl_runs(run_ids: list[str], metric: str = "reward_mean"):
    """Like H.compare_runs, but 'best' is the MAXIMUM, not the minimum.

    ES: `H.compare_runs` calcula `best` con `.min()`, porque en supervisado
    menos pérdida es mejor. En refuerzo es al revés: más recompensa es mejor.
    Llamar a `H.compare_runs(ids, "reward_mean")` no da error — da la peor
    iteración etiquetada como "best", que es mucho peor que un error.

    Es el problema nº4 de HARNESS.md §4.1 hecho código: la métrica de progreso
    en RL no es una pérdida, y la pérdida de política puede SUBIR mientras el
    agente mejora (al reducirse la varianza de los pesos). No mires la loss aquí.
    """
    table = H.compare_runs(run_ids, metric)
    table["best"] = [H.load_run(rid)["history"][metric].max() for rid in run_ids]

    # ES: Corrección de un fallo heredado de `H.compare_runs`: allí la columna
    # "seed" se rellena con `meta["seed"]` (la semilla REAL usada) pero acto
    # seguido se hace `**flat_config`, y si el config trae una clave "seed" la
    # pisa. Resultado: al comparar 5 semillas, la tabla dice que todas son la 0.
    # El `run_id` sí lleva la buena, así que el dato no se pierde, solo se
    # muestra mal. Afecta igual a `H.repeat_with_seeds`.
    table["seed"] = [H.load_run(rid)["meta"]["seed"] for rid in run_ids]
    return table


def plot_rl_runs(run_ids: list[str], metrics=("reward_mean", "eval_reward"), ax=None):
    """H.plot_runs with the two defaults that break on rewards."""
    # ES: `log_scale=False` obligatorio: las recompensas pueden ser negativas o
    # cero, y log(≤0) no existe. El default del arnés supervisado revienta aquí.
    ax = H.plot_runs(run_ids, metrics=metrics, log_scale=False, ax=ax)
    ax.set_xlabel("iteration")   # ES: no son épocas, aunque la columna se llame así
    ax.set_ylabel("reward")      # ES: H.plot_runs rotula "loss" en duro
    return ax


def repeat_rl_with_seeds(config: dict, n_seeds: int = 5,
                         metric: str = "eval_reward"):
    """Same experiment, several seeds. In RL this is not optional.

    ES: `H.repeat_with_seeds` no sirve aquí: llama a `run_experiment` en duro y
    calcula el "mejor" con el mínimo. Esta es la misma idea con las dos cosas
    corregidas.

    Y el motivo por el que existe: en supervisado repetir con semillas es buena
    práctica; en refuerzo es la diferencia entre un resultado y una anécdota. El
    estimador de REINFORCE tiene varianza tan alta que dos semillas del MISMO
    config pueden dar curvas opuestas. Si solo lanzas una, no sabes si has
    aprendido algo o si has tenido suerte.
    """
    run_ids = []
    for seed in range(n_seeds):
        seeded = dict(config, name=f"{config.get('name', 'rl')}-rep")
        print(f"▶ seed {seed}")
        run_ids.append(run_rl_experiment(seeded, seed=seed, verbose=False).run_id)

    table = compare_rl_runs(run_ids, metric)
    mean, std = table["final"].mean(), table["final"].std()
    print(f"\n{metric}: mean {mean:+.4f} · std {std:.4f} · "
          f"range [{table['final'].min():+.4f}, {table['final'].max():+.4f}]")
    print(f"→ A difference smaller than ~{2 * std:.4f} is NOT a result.")
    if std > abs(mean) * 0.5:
        # ES: Si la dispersión es del orden de la propia señal, la política no ha
        # aprendido de forma fiable: ha aprendido en algunas semillas. Eso es un
        # resultado negativo, y conviene que el arnés lo diga en voz alta.
        print("⚠️  std comparable a la media: el resultado depende de la semilla.")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 🔀 Lo que este arnés NO hace, y qué usarías en su lugar
# ─────────────────────────────────────────────────────────────────────────────
"""
ES: Mapa de salida. Cada fila es algo que este archivo deliberadamente no tiene,
por qué se ha dejado fuera, y a dónde ir cuando haga falta de verdad.

┌─────────────────────────┬──────────────────────────────┬─────────────────────┐
│ Lo que falta            │ Por qué importa              │ Dónde mirar         │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ Crítico V(s)            │ Reduce la varianza mucho más │ A2C / PPO           │
│ (Actor-Crítico)         │ que restar la media. Es el   │ CleanRL: ppo.py     │
│                         │ salto de juguete a real.     │                     │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ GAE (lambda)            │ Controla el compromiso       │ Schulman et al.     │
│                         │ sesgo/varianza del crédito.  │ arXiv:1506.02438    │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ PPO (clipping)          │ Evita que un paso destruya   │ arXiv:1707.06347    │
│                         │ la política. Estándar hoy.   │ SB3 / CleanRL       │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ Replay buffer           │ Reutilizar experiencia vieja │ DQN, SAC            │
│ (off-policy)            │ en vez de tirarla. Cambia el │ arXiv:1312.5602     │
│                         │ bucle entero, no es un añadido│                    │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ Entornos vectorizados   │ 100× throughput. Pura        │ gymnasium.vector    │
│                         │ ingeniería, cero teoría.      │ SB3: VecEnv        │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ Acciones continuas      │ Robótica, control.           │ SAC, TD3            │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ Penalización KL contra  │ Sin esto, RLHF degenera en   │ TRL: PPOTrainer     │
│ el modelo de referencia │ reward hacking. Ver §4.5 de  │ arXiv:2203.02155    │
│                         │ HARNESS.md.                  │                     │
├─────────────────────────┼──────────────────────────────┼─────────────────────┤
│ DPO                     │ Colapsa reward model + RL en │ arXiv:2305.18290    │
│                         │ un bucle supervisado. Más    │ TRL: DPOTrainer     │
│                         │ estable y más barato.        │                     │
└─────────────────────────┴──────────────────────────────┴─────────────────────┘

Qué librería según lo que quieras:
  · ENTENDER un algoritmo    → CleanRL (un archivo por algoritmo, sin capas)
  · USARLO y que funcione    → Stable-Baselines3 (API estable, bien probada)
  · COMPONER piezas propias  → TorchRL (oficial de PyTorch, modular)
  · ALINEAR un LLM           → TRL (PPO, DPO, GRPO sobre HuggingFace)

Y la advertencia que más vale que las cuatro anteriores:
    Un experimento de RL con UNA semilla no dice nada. El estimador tiene
    varianza altísima y es trivial "demostrar" ruido. Usa
    `repeat_rl_with_seeds` (no `H.repeat_with_seeds`, que está atada a
    `run_experiment`) y dibuja el abanico antes de creerte cualquier curva.
    Ver HARNESS.md §4.4.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Sane starting configs
# ─────────────────────────────────────────────────────────────────────────────
BANDIT_CONFIG = {
    "name": "rl_bandit",
    "env": "two_armed_bandit",
    "env_args": {"p_left": 0.2, "p_right": 0.8},
    "model": "mlp_policy",
    "model_args": {"hidden": 16},
    "optimizer": "adam",
    "optimizer_args": {"lr": 0.05},
    "iterations": 40,
    "episodes_per_iteration": 32,
    "gamma": 1.0,
    "advantage": "normalized",
    "max_steps": 1,
    "eval_episodes": 20,
    "seed": 0,
}

CORRIDOR_CONFIG = {
    "name": "rl_corridor",
    "env": "corridor",
    "env_args": {"length": 5, "step_cost": 0.05, "quit_reward": 0.3},
    "model": "mlp_policy",
    "model_args": {"hidden": 32},
    "optimizer": "adam",
    "optimizer_args": {"lr": 0.02},
    "iterations": 80,
    "episodes_per_iteration": 16,
    # ES: Con gamma=0.99 el premio lejano (+1) vale más que cobrar ya (+0.3), y
    # la política óptima es aguantar. Baja gamma a 0.6 y se invierte: el mismo
    # entorno, el mismo código, la decisión contraria. Pruébalo.
    "gamma": 0.99,
    "advantage": "normalized",
    "entropy_coef": 0.01,
    "grad_clip": 1.0,
    "max_steps": 50,
    "eval_episodes": 20,
    "seed": 0,
}

# ES: Dato medido, no teórico. Lanzando CORRIDOR_CONFIG con 5 semillas
# (`repeat_rl_with_seeds`), 3 aprenden a aguantar (+0.75) y 2 se atascan
# cobrando ya (+0.30). El mismo config, el mismo código: la semilla decide.
#
# Es el fallo clásico del gradiente de política: si las primeras trayectorias
# aleatorias cobran pronto, esa acción se refuerza, la entropía se hunde y la
# política deja de explorar antes de haber visto nunca el premio grande. Un
# óptimo local del que ya no sale.
#
# Y es la mejor justificación posible de todo lo que hay en el mapa 🔀:
# `entropy_coef` sube el suelo pero no lo arregla; lo que lo arregla es una
# baseline aprendida V(s) y PPO. Por eso "converge en un juguete" y "converge
# de verdad" son cosas distintas — y por eso una sola semilla no vale nada.
