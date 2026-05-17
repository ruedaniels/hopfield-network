import numpy as np


def random_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random bipolar pattern of length n."""
    return rng.choice(np.array([-1, 1]), size=n)


def add_noise(pattern: np.ndarray, noise_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly round(noise_ratio * N) bits in a pattern."""
    n = len(pattern)
    n_flips = round(noise_ratio * n)
    indices = rng.choice(n, size=n_flips, replace=False)
    noisy = pattern.copy()
    noisy[indices] *= -1
    return noisy


def pattern_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Compute overlap m = (1/N) * dot(a, b)."""
    return float(np.dot(a, b) / len(a))


def is_stored_pattern(state: np.ndarray, patterns: np.ndarray, threshold: float = 0.95) -> bool:
    """Return True if state matches any stored pattern above threshold."""
    for pattern in patterns:
        if pattern_overlap(state, pattern) >= threshold:
            return True
    return False