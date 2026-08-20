"""Tests for the arithmetic language.

ES: El verificador es la pieza más crítica del itinerario: es la recompensa de
N28. Si se equivoca, el modelo aprende lo contrario de lo que queremos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from lab import arithmetic as A


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_generated_problems_are_correct(level):
    """Every generated answer must survive its own verifier."""
    for problem in A.generate(level=level, n_problems=200, seed=level):
        assert A.verify(problem.expression, problem.answer)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_generation_is_deterministic(level):
    first = [p.raw for p in A.generate(level, 20, seed=0)]
    second = [p.raw for p in A.generate(level, 20, seed=0)]
    assert first == second


def test_generation_responds_to_seed():
    assert ([p.raw for p in A.generate(1, 20, seed=0)] !=
            [p.raw for p in A.generate(1, 20, seed=1)])


def test_verifier_rejects_wrong_answers():
    assert A.verify("(3+4)*2", "14")
    assert not A.verify("(3+4)*2", "13")
    assert not A.verify("(3+4)*2", "")
    assert not A.verify("(3+4)*2", "not a number")


def test_verifier_does_not_execute_arbitrary_code():
    """ES: Un verificador que ejecuta código deja de ser un verificador."""
    with pytest.raises(Exception):
        A.solve("__import__('os').system('echo unsafe')")


def test_default_generation_avoids_negative_answers():
    for level in [2, 3]:
        for problem in A.generate(level, 100, seed=0):
            assert not problem.answer.startswith("-")


def test_reasoning_steps_end_in_the_answer():
    for problem in A.generate(level=3, n_problems=50, seed=0):
        steps = A.reasoning_steps(problem.expression)
        assert steps.split("=")[-1] == problem.answer


def test_level_3_has_more_steps_than_level_1():
    """ES: Esa brecha es lo que hace que el razonamiento emerja en N28."""
    def mean_steps(level):
        problems = A.generate(level, 50, seed=0)
        return sum(A.reasoning_steps(p.expression).count("=") for p in problems) / 50

    assert mean_steps(3) > mean_steps(1)


def test_reversed_format_reverses_only_the_answer():
    problem = A.Problem("47+38", "85", level=1)
    assert A.format_example(problem, "reversed") == "47+38=58"


def test_chat_format_carries_the_special_tokens():
    text = A.format_example(A.generate(1, 1, seed=0)[0], "chat")
    for token in [A.USER_TOKEN, A.ASSISTANT_TOKEN, A.END_TOKEN]:
        assert token in text


def test_tokenizer_round_trips():
    for text in ["47+38=85", "(3+4)*2=14", "0+0=0"]:
        assert A.decode(A.encode(text)) == text


def test_character_tokenizer_splits_digits():
    assert A.tokenize("47+38") == ["4", "7", "+", "3", "8"]
    assert A.tokenize("47+38", group_digits=True) == ["47", "+", "38"]


def test_special_tokens_survive_tokenization():
    tokens = A.tokenize(f"{A.USER_TOKEN}3+4{A.ASSISTANT_TOKEN}7{A.END_TOKEN}")
    assert A.USER_TOKEN in tokens and A.END_TOKEN in tokens


@pytest.mark.parametrize("kind", ["correctness", "brevity", "honesty"])
def test_preference_pairs_differ(kind):
    for pair in A.generate_preferences(kind, 50, seed=0):
        assert pair.preferred != pair.rejected


def test_correctness_pairs_prefer_the_right_answer():
    for pair in A.generate_preferences("correctness", 100, seed=0):
        assert A.verify(pair.prompt, pair.preferred)
        assert not A.verify(pair.prompt, pair.rejected)


def test_brevity_pairs_prefer_the_shorter_answer():
    for pair in A.generate_preferences("brevity", 50, seed=0):
        assert len(pair.preferred) < len(pair.rejected)


def test_honesty_pairs_use_out_of_range_numbers():
    """ES: El modelo NO puede saber la respuesta. Por eso se puede medir si
    aprendió a decir 'no lo sé'."""
    for pair in A.generate_preferences("honesty", 20, seed=0, out_of_range_digits=6):
        assert pair.preferred == "no lo sé"
        assert all(len(n) >= 6 for n in A._split_numbers(pair.prompt))


def test_by_result_split_shares_no_answers():
    """ES: Cero solapamiento. Es el split duro."""
    train, val = A.split_problems(A.generate(1, 1000, seed=0), "by_result", seed=0)
    assert {p.answer for p in train}.isdisjoint({p.answer for p in val})


def test_random_split_leaks_answers():
    """ES: 47+38 en train y 38+47 en val. Acertar no prueba que generalice."""
    train, val = A.split_problems(A.generate(1, 1000, seed=0), "random", seed=0)
    train_answers = {p.answer for p in train}
    leaked = sum(1 for p in val if p.answer in train_answers) / len(val)
    assert leaked > 0.5


def test_by_range_split_separates_by_operand_length():
    train, val = A.split_problems(A.generate(1, 1000, seed=0), "by_range", seed=0)

    def longest(problems):
        return max(max(len(n) for n in A._split_numbers(p.expression)) for p in problems)

    assert longest(train) < longest(val)


def test_splits_do_not_lose_problems():
    problems = A.generate(1, 500, seed=0)
    for strategy in ["random", "by_result", "by_range"]:
        train, val = A.split_problems(problems, strategy, seed=0)
        assert len(train) + len(val) == len(problems)
