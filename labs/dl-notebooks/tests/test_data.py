"""Tests for the synthetic dataset factory.

ES: Lo que se comprueba aquí es sobre todo que el TECHO tiene sentido. Si un
techo está mal, todos los experimentos que lo usen se interpretan mal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from scipy.stats import norm

from lab import data
from lab import harness


@pytest.mark.parametrize("name,generator", sorted(data.GENERATORS.items()))
def test_every_generator_reports_a_ceiling(name, generator):
    dataset = generator(n_samples=100)
    assert dataset.ceiling is not None
    assert dataset.ceiling_metric in {"mse", "accuracy"}


@pytest.mark.parametrize("name,generator", sorted(data.GENERATORS.items()))
def test_generators_are_deterministic(name, generator):
    assert np.array_equal(generator(n_samples=50, seed=0).inputs,
                          generator(n_samples=50, seed=0).inputs)


@pytest.mark.parametrize("name,generator", sorted(data.GENERATORS.items()))
def test_generators_respond_to_seed(name, generator):
    assert not np.array_equal(generator(n_samples=50, seed=0).inputs,
                              generator(n_samples=50, seed=1).inputs)


def test_bayes_ceiling_matches_the_closed_form():
    """The only generator whose ceiling is exact, not estimated."""
    for separation in [0.5, 2.0, 4.0]:
        dataset = data.make_two_gaussians(separation=separation, spread=1.0)
        assert dataset.ceiling == pytest.approx(1 - norm.cdf(-separation / 2))


def test_regression_ceiling_is_the_noise_variance():
    for noise_std in [0.1, 0.5, 1.0]:
        assert data.make_line(noise_std=noise_std).ceiling == pytest.approx(noise_std ** 2)


def test_pure_noise_ceiling_is_the_majority_class():
    dataset = data.make_pure_noise(n_samples=1000, seed=0)
    counts = np.bincount(dataset.targets)
    assert dataset.ceiling == pytest.approx(counts.max() / len(dataset))


def test_label_noise_lowers_the_ceiling():
    clean = data.make_two_gaussians(n_samples=500, separation=4.0)
    noisy = data.add_label_noise(clean, fraction=0.3, seed=0)
    assert noisy.ceiling < clean.ceiling
    assert (noisy.targets != clean.targets).mean() == pytest.approx(0.3, abs=0.02)


def test_group_split_keeps_groups_apart():
    """ES: Si un grupo aparece en ambos lados, el split no mide lo que crees (N15)."""
    grouped = data.add_groups(data.make_two_gaussians(n_samples=400), n_groups=4, seed=0)
    train, val = grouped.split(strategy="by_group", seed=0)
    assert set(train.groups).isdisjoint(set(val.groups))


def test_random_split_leaks_groups():
    grouped = data.add_groups(data.make_two_gaussians(n_samples=400), n_groups=4, seed=0)
    train, val = grouped.split(strategy="random", seed=0)
    assert not set(train.groups).isdisjoint(set(val.groups))


def test_all_generators_are_registered_in_the_harness():
    assert set(data.GENERATORS) <= set(harness.datasets)


def test_loaders_have_the_expected_shapes():
    train_loader, val_loader = data.make_line(n_samples=100).to_loaders(batch_size=10)
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (10, 1)
    assert targets.shape == (10,)
    assert len(val_loader.dataset) == 20
