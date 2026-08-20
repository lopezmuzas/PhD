"""The arithmetic language: the thread that runs from N18 to N30.

ES: El lenguaje aritmético. Se eligió porque cumple cuatro cosas a la vez:
verificador de tres líneas, preferencias objetivas sin anotar, dificultad
graduable, y todo cabe en CPU.

Comment convention / Convenio: docstrings in English say WHAT, Spanish notes say WHY.
"""
from __future__ import annotations

import ast
import operator
import random
from dataclasses import dataclass
from typing import Literal

Level = Literal[0, 1, 2, 3]
FormatStyle = Literal["raw", "chat", "reversed"]
SplitStrategy = Literal["random", "by_result", "by_range"]
PreferenceKind = Literal["correctness", "brevity", "honesty"]

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = [PAD_TOKEN, END_TOKEN, USER_TOKEN, ASSISTANT_TOKEN]

DIGITS = list("0123456789")
SYMBOLS = list("+-*()=")
VOCABULARY = SPECIAL_TOKENS + DIGITS + SYMBOLS

TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}

_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

QUESTION_TEMPLATES = [
    "¿cuánto es {expression}?",
    "calcula {expression}",
    "{expression} = ?",
    "resuelve {expression}",
]


# ─────────────────────────────────────────────────────────────────────────────
# Solving and verifying
# ─────────────────────────────────────────────────────────────────────────────
def solve(expression: str) -> str:
    """Evaluate an expression safely. This is also the reward signal for N28.

    ES: Se usa `ast` en vez de `eval` porque un verificador que ejecuta código
    arbitrario deja de ser un verificador.
    """
    return str(_evaluate(ast.parse(expression, mode="eval").body))


def _evaluate(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPERATIONS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


def verify(expression: str, answer: str) -> bool:
    """Is this answer correct? The whole reward function for RLVR fits here."""
    try:
        return answer.strip() == solve(expression)
    except Exception:
        return False


def reasoning_steps(expression: str) -> str:
    """Intermediate steps, as a reference trace for N29.

    ES: En N28 el modelo debería DESCUBRIR algo parecido por su cuenta. Esto
    es la respuesta que le habríamos enseñado, para poder comparar.
    """
    steps: list[str] = []
    _collect_steps(ast.parse(expression, mode="eval").body, steps)
    return ", ".join(steps)


def _collect_steps(node: ast.AST, steps: list[str]) -> int:
    if isinstance(node, ast.Constant):
        return node.value
    left = _collect_steps(node.left, steps)
    right = _collect_steps(node.right, steps)
    symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(node.op)]
    result = _OPERATIONS[type(node.op)](left, right)
    steps.append(f"{left}{symbol}{right}={result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Problem:
    """An expression and its answer. Everything else is derived."""

    expression: str
    answer: str
    level: int

    @property
    def raw(self) -> str:
        return f"{self.expression}={self.answer}"

    def __str__(self) -> str:
        return self.raw


def generate(level: Level = 1, n_problems: int = 1000, seed: int = 0,
             max_digits: int | None = None,
             allow_negative: bool = False) -> list[Problem]:
    """Build n problems of the requested difficulty level.

    ES: El nivel es el único mando de dificultad. Subes de nivel cuando el
    anterior se resuelve, no antes.

    `allow_negative=False` mantiene el problema pequeño: sin resultados
    negativos, el modelo no tiene que aprender el signo además de la
    aritmética. Actívalo cuando quieras subir la dificultad sin cambiar de
    nivel.
    """
    rng = random.Random(seed)
    builders = {0: _level_0, 1: _level_1, 2: _level_2, 3: _level_3}
    if level not in builders:
        raise ValueError(f"level must be one of {sorted(builders)}")

    digits = max_digits or {0: 1, 1: 3, 2: 2, 3: 2}[level]
    problems = []
    while len(problems) < n_problems:
        expression = builders[level](rng, digits)
        answer = solve(expression)
        if not allow_negative and answer.startswith("-"):
            continue
        problems.append(Problem(expression, answer, level))
    return problems


def _number(rng: random.Random, max_digits: int) -> int:
    return rng.randint(0, 10 ** rng.randint(1, max_digits) - 1)


def _level_0(rng: random.Random, _max_digits: int) -> str:
    return f"{rng.randint(0, 9)}+{rng.randint(0, 9)}"


def _level_1(rng: random.Random, max_digits: int) -> str:
    return f"{_number(rng, max_digits)}+{_number(rng, max_digits)}"


def _level_2(rng: random.Random, max_digits: int) -> str:
    """Three operands, no parentheses: the model must learn precedence."""
    a, b, c = (_number(rng, max_digits) for _ in range(3))
    return f"{a}{rng.choice('+-*')}{b}{rng.choice('+-*')}{c}"


def _level_3(rng: random.Random, max_digits: int) -> str:
    """Parentheses: hard in one pass, easy step by step. That gap is what makes
    the reasoning emerge in N28."""
    a, b, c = (_number(rng, max_digits) for _ in range(3))
    inner, outer = rng.choice("+-"), rng.choice("+-*")
    if rng.random() < 0.5:
        return f"({a}{inner}{b}){outer}{c}"
    return f"{a}{outer}({b}{inner}{c})"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_example(problem: Problem, style: FormatStyle = "raw",
                   seed: int | None = None) -> str:
    """Render a problem in one of the three training formats.

    ES: 'reversed' existe para un experimento concreto de N18: con la respuesta
    al revés el acarreo fluye en el sentido de la generación, y el modelo lo
    aprende mucho antes. La representación del dato importa tanto como la
    arquitectura.
    """
    if style == "raw":
        return problem.raw
    if style == "reversed":
        return f"{problem.expression}={problem.answer[::-1]}"
    if style == "chat":
        rng = random.Random(seed if seed is not None else hash(problem.expression))
        question = rng.choice(QUESTION_TEMPLATES).format(expression=problem.expression)
        return f"{USER_TOKEN}{question}{ASSISTANT_TOKEN}{problem.answer}{END_TOKEN}"
    raise ValueError(f"unknown style '{style}'")


def build_corpus(problems: list[Problem], style: FormatStyle = "raw",
                 separator: str = "\n") -> str:
    """One long string, ready for next-token pretraining (N21)."""
    return separator.join(format_example(p, style) for p in problems)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizing
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str, group_digits: bool = False) -> list[str]:
    """Split text into tokens. Character level by default.

    ES: `group_digits=True` existe solo para demostrar que empeora la
    aritmética. Es la razón por la que los modelos reales fallan al sumar.
    """
    tokens, position = [], 0
    while position < len(text):
        special = next((s for s in SPECIAL_TOKENS if text.startswith(s, position)), None)
        if special:
            tokens.append(special)
            position += len(special)
            continue

        character = text[position]
        if group_digits and character.isdigit():
            end = position
            while end < len(text) and text[end].isdigit():
                end += 1
            tokens.append(text[position:end])
            position = end
            continue

        tokens.append(character)
        position += 1
    return tokens


def encode(text: str) -> list[int]:
    """Token ids, character level. Unknown characters are skipped."""
    return [TOKEN_TO_ID[t] for t in tokenize(text) if t in TOKEN_TO_ID]


def decode(token_ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in token_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Preference pairs — objective, no human annotation needed
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PreferencePair:
    """A prompt and two answers, one preferred over the other."""

    prompt: str
    preferred: str
    rejected: str
    kind: str


def generate_preferences(kind: PreferenceKind = "correctness", n_pairs: int = 500,
                         level: Level = 2, seed: int = 0,
                         out_of_range_digits: int = 6) -> list[PreferencePair]:
    """Build preference pairs without a single human annotation.

    ES: Esta función es la razón principal de haber elegido aritmética. En un
    dominio real, estos pares costarían semanas de anotadores.
    """
    rng = random.Random(seed)
    builders = {"correctness": _pairs_correctness,
                "brevity": _pairs_brevity,
                "honesty": _pairs_honesty}
    if kind not in builders:
        raise ValueError(f"kind must be one of {sorted(builders)}")
    return builders[kind](rng, n_pairs, level, out_of_range_digits)


def _pairs_correctness(rng, n_pairs, level, _digits):
    pairs = []
    for problem in generate(level, n_pairs, seed=rng.randint(0, 10 ** 6)):
        wrong = str(int(problem.answer) + rng.choice([-2, -1, 1, 2]))
        pairs.append(PreferencePair(problem.expression, problem.answer, wrong, "correctness"))
    return pairs


def _pairs_brevity(rng, n_pairs, level, _digits):
    padding = ["El resultado de la operación {e} es, efectivamente, {a}.",
               "Vamos a calcularlo con calma. Tenemos {e}, y el resultado final es {a}.",
               "Para responder a {e}, hay que operar paso a paso; el resultado es {a}."]
    pairs = []
    for problem in generate(level, n_pairs, seed=rng.randint(0, 10 ** 6)):
        inflated = rng.choice(padding).format(e=problem.expression, a=problem.answer)
        pairs.append(PreferencePair(problem.expression, problem.answer, inflated, "brevity"))
    return pairs


def _pairs_honesty(rng, n_pairs, level, out_of_range_digits):
    """Numbers far outside the training range: the model CANNOT know the answer.

    ES: Y por eso se puede medir si aprendió a decir "no lo sé", cosa que con
    datos reales es casi imposible de comprobar.
    """
    pairs = []
    lower = 10 ** (out_of_range_digits - 1)
    upper = 10 ** out_of_range_digits - 1
    for _ in range(n_pairs):
        expression = f"{rng.randint(lower, upper)}+{rng.randint(lower, upper)}"
        invented = str(rng.randint(lower, upper * 2))
        pairs.append(PreferencePair(expression, "no lo sé", invented, "honesty"))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Splitting — three strategies that measure three different things
# ─────────────────────────────────────────────────────────────────────────────
def split_problems(problems: list[Problem], strategy: SplitStrategy = "random",
                   val_fraction: float = 0.2,
                   seed: int = 0) -> tuple[list[Problem], list[Problem]]:
    """Split problems into train and validation.

    ES: Los tres miden cosas distintas, y ahí está la trampa de este dataset:
      random     → 47+38 en train y 38+47 en validación. ¿Generalizó o
                   memorizó la conmutatividad?
      by_result  → todos los que dan 85 caen del mismo lado. Mucho más duro.
      by_range   → entrena con pocos dígitos, evalúa con más. Extrapolación.
    """
    rng = random.Random(seed)
    shuffled = problems[:]
    rng.shuffle(shuffled)

    if strategy == "random":
        cut = int(len(shuffled) * (1 - val_fraction))
        return shuffled[:cut], shuffled[cut:]

    if strategy == "by_result":
        results = sorted({p.answer for p in shuffled})
        rng.shuffle(results)
        n_val = max(1, int(len(results) * val_fraction))
        val_results = set(results[:n_val])
        train = [p for p in shuffled if p.answer not in val_results]
        val = [p for p in shuffled if p.answer in val_results]
        return train, val

    if strategy == "by_range":
        def longest_operand(problem: Problem) -> int:
            numbers = [n for n in _split_numbers(problem.expression)]
            return max(len(n) for n in numbers)

        lengths = sorted({longest_operand(p) for p in shuffled})
        threshold = lengths[-1] if len(lengths) > 1 else lengths[0]
        train = [p for p in shuffled if longest_operand(p) < threshold]
        val = [p for p in shuffled if longest_operand(p) >= threshold]
        return train, val

    raise ValueError(f"unknown strategy '{strategy}'")


def _split_numbers(expression: str) -> list[str]:
    numbers, current = [], ""
    for character in expression:
        if character.isdigit():
            current += character
        elif current:
            numbers.append(current)
            current = ""
    if current:
        numbers.append(current)
    return numbers
