import numpy as np
from hopfield.network import HopfieldNetwork

def test_five_pattern_integration():
    rng = np.random.default_rng(42)
    from hopfield.patterns import random_pattern, add_noise, pattern_overlap

    net = HopfieldNetwork(100)
    patterns = np.array([random_pattern(100, rng) for _ in range(5)])
    net.train(patterns)

    for i, pattern in enumerate(patterns):
        probe = add_noise(pattern, 0.20, rng)
        recalled = net.recall(probe, max_steps=500, rng=rng)
        overlap = pattern_overlap(recalled, pattern)
        assert overlap >= 0.95, f"Pattern {i} failed: overlap was {overlap:.3f}"


def test_high_accuracy_below_capacity():
    from hopfield.analysis import recall_accuracy
    from hopfield.patterns import random_pattern
    rng = np.random.default_rng(42)
    net = HopfieldNetwork(100)
    patterns = np.array([random_pattern(100, rng) for _ in range(5)])
    net.train(patterns)
    accuracy = recall_accuracy(net, patterns, noise_ratio=0.20, n_trials=20, rng=rng)
    assert accuracy >= 0.90, f"Expected >= 0.90 accuracy below capacity, got {accuracy:.3f}"


def test_accuracy_drops_above_capacity():
    from hopfield.analysis import recall_accuracy
    from hopfield.patterns import random_pattern
    rng = np.random.default_rng(42)
    net = HopfieldNetwork(100)
    patterns = np.array([random_pattern(100, rng) for _ in range(18)])
    net.train(patterns)
    accuracy = recall_accuracy(net, patterns, noise_ratio=0.20, n_trials=20, rng=rng)
    assert accuracy <= 0.60, f"Expected accuracy to drop above capacity, got {accuracy:.3f}"