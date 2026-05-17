import numpy as np
from hopfield.patterns import random_pattern, add_noise, pattern_overlap, is_stored_pattern


def test_random_pattern_values():
    rng = np.random.default_rng(42)
    p = random_pattern(200, rng)
    assert np.all(np.abs(p) == 1)


def test_random_pattern_shape():
    rng = np.random.default_rng(42)
    p = random_pattern(100, rng)
    assert p.shape == (100,)


def test_add_noise_exact_flip_count():
    rng = np.random.default_rng(42)
    p = random_pattern(100, rng)
    noisy = add_noise(p, 0.20, rng)
    hamming = int(np.sum(p != noisy))
    assert hamming == 20


def test_pattern_overlap_identical():
    rng = np.random.default_rng(42)
    a = random_pattern(100, rng)
    assert pattern_overlap(a, a) == 1.0


def test_pattern_overlap_inverted():
    rng = np.random.default_rng(42)
    a = random_pattern(100, rng)
    assert pattern_overlap(a, -a) == -1.0


def test_is_stored_pattern_match():
    rng = np.random.default_rng(42)
    patterns = np.array([random_pattern(100, rng) for _ in range(5)])
    assert is_stored_pattern(patterns[0], patterns) is True


def test_is_stored_pattern_no_match():
    rng = np.random.default_rng(42)
    patterns = np.array([random_pattern(100, rng) for _ in range(5)])
    spurious = random_pattern(100, rng)
    assert is_stored_pattern(spurious, patterns) is False