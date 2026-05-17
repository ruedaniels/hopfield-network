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
    """
    Measure the fraction of retrievals with overlap >= 0.95 with the correct pattern.

    For each pattern, runs n_trials noisy retrievals and counts how many
    return a state with overlap >= 0.95 with the original pattern.

    Parameters
    ----------
    net : HopfieldNetwork
        A trained Hopfield network.
    patterns : np.ndarray
        Array of shape (P, N) containing the P stored patterns.
    noise_ratio : float
        Fraction of bits to flip when generating noisy probes.
    n_trials : int
        Number of retrieval trials per pattern.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    float
        Fraction of successful retrievals in [0.0, 1.0].
    """
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
    """
    Sweep over pattern counts and measure recall accuracy for each.

    Trains a fresh network for each value of P and measures recall
    accuracy at the given noise level. Expect accuracy to drop sharply
    near P = 0.138 * n_neurons (McEliece capacity limit).

    Parameters
    ----------
    p_values : list of int
        List of pattern counts to test.
    n_neurons : int
        Number of neurons in each network.
    noise_ratio : float
        Fraction of bits to flip in each probe.
    n_trials : int
        Number of retrieval trials per pattern per P value.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    list of tuple
        List of (p, accuracy) pairs, one per value in p_values.
    """
    results = []
    for p in p_values:
        net = HopfieldNetwork(n_neurons)
        patterns = np.array([random_pattern(n_neurons, rng) for _ in range(p)])
        net.train(patterns)
        accuracy = recall_accuracy(net, patterns, noise_ratio, n_trials, rng)
        results.append((p, accuracy))
    return results