import numpy as np
from hopfield.network import HopfieldNetwork
from hopfield.patterns import random_pattern, add_noise, pattern_overlap


def recall_accuracy(
    net: HopfieldNetwork,
    patterns: np.ndarray,
    noise_ratio: float,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Measure fraction of retrievals with overlap >= 0.95 with correct pattern."""
    correct = 0
    total = len(patterns) * n_trials
    for pattern in patterns:
        for _ in range(n_trials):
            probe = add_noise(pattern, noise_ratio, rng)
            recalled = net.recall(probe, max_steps=500, rng=rng)
            if pattern_overlap(recalled, pattern) >= 0.95:
                correct += 1
    return correct / total


def capacity_sweep(
    p_values: list,
    n_neurons: int,
    noise_ratio: float,
    n_trials: int,
    rng: np.random.Generator,
) -> list:
    """Sweep over pattern counts and measure recall accuracy for each."""
    results = []
    for p in p_values:
        net = HopfieldNetwork(n_neurons)
        patterns = np.array([random_pattern(n_neurons, rng) for _ in range(p)])
        net.train(patterns)
        accuracy = recall_accuracy(net, patterns, noise_ratio, n_trials, rng)
        results.append((p, accuracy))
    return results