import numpy as np


def random_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random bipolar pattern of length n.

    Parameters
    ----------
    n : int
        Length of the pattern.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n,) with all values in {-1, +1}.
    """
    return rng.choice(np.array([-1, 1]), size=n)


def add_noise(
    pattern: np.ndarray,
    noise_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Flip exactly round(noise_ratio * N) bits in a pattern.

    Parameters
    ----------
    pattern : np.ndarray
        Bipolar (+-1) pattern of shape (N,).
    noise_ratio : float
        Fraction of bits to flip. Must be in [0, 1].
    rng : np.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy copy of the pattern with exactly round(noise_ratio * N)
        bits flipped. Hamming distance equals the number of flipped bits.
    """
    n = len(pattern)
    n_flips = round(noise_ratio * n)
    indices = rng.choice(n, size=n_flips, replace=False)
    noisy = pattern.copy()
    noisy[indices] *= -1
    return noisy


def pattern_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the overlap between two bipolar patterns.

    Overlap m = (1/N) * dot(a, b). Returns 1.0 for identical patterns,
    -1.0 for inverted patterns, and approximately 0 for orthogonal patterns.

    Parameters
    ----------
    a : np.ndarray
        First bipolar (+-1) pattern of shape (N,).
    b : np.ndarray
        Second bipolar (+-1) pattern of shape (N,).

    Returns
    -------
    float
        Overlap value in [-1.0, 1.0].
    """
    return float(np.dot(a, b) / len(a))


def is_stored_pattern(
    state: np.ndarray,
    patterns: np.ndarray,
    threshold: float = 0.95,
) -> bool:
    """
    Check whether a state matches any stored pattern above a threshold.

    Parameters
    ----------
    state : np.ndarray
        Bipolar (+-1) state vector of shape (N,).
    patterns : np.ndarray
        Array of shape (P, N) containing P stored patterns.
    threshold : float, optional
        Minimum overlap to consider a match. Default is 0.95.

    Returns
    -------
    bool
        True if overlap with any stored pattern >= threshold, else False.
    """
    for pattern in patterns:
        if pattern_overlap(state, pattern) >= threshold:
            return True
    return False