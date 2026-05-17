import numpy as np
import pytest
from hopfield.network import HopfieldNetwork


def test_weight_matrix_shape():
    net = HopfieldNetwork(100)
    patterns = np.array([np.random.choice([-1, 1], 100)])
    net.train(patterns)
    assert net.weights.shape == (100, 100)


def test_weight_matrix_is_symmetric():
    net = HopfieldNetwork(100)
    rng = np.random.default_rng(42)
    patterns = rng.choice([-1, 1], size=(5, 100))
    net.train(patterns)
    np.testing.assert_array_equal(net.weights, net.weights.T)


def test_weight_diagonal_is_zero():
    net = HopfieldNetwork(100)
    rng = np.random.default_rng(42)
    patterns = rng.choice([-1, 1], size=(5, 100))
    net.train(patterns)
    assert np.all(np.diag(net.weights) == 0)


def test_energy_returns_float():
    net = HopfieldNetwork(10)
    rng = np.random.default_rng(42)
    pattern = rng.choice([-1, 1], size=(1, 10))
    net.train(pattern)
    state = rng.choice([-1, 1], size=10)
    assert isinstance(net.energy(state), float)


def test_energy_monotonically_non_increasing():
    rng = np.random.default_rng(42)
    net = HopfieldNetwork(50)
    patterns = rng.choice([-1, 1], size=(3, 50))
    net.train(patterns)
    for _ in range(1000):
        state = rng.choice([-1, 1], size=50)
        _, history = net.recall(state, max_steps=100, rng=rng, return_history=True)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-10


def test_single_pattern_recall():
    rng = np.random.default_rng(42)
    net = HopfieldNetwork(100)
    pattern = rng.choice([-1, 1], size=(1, 100))
    net.train(pattern)
    probe = pattern[0].copy()
    probe[:10] = -probe[:10]
    recalled = net.recall(probe, max_steps=200, rng=rng)
    overlap = float(np.dot(recalled, pattern[0]) / 100)
    assert overlap >= 0.95