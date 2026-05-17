import numpy as np


class HopfieldNetwork:
    """A Hopfield Network implementing associative memory via energy minimisation."""

    def __init__(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns: np.ndarray) -> None:
        """Learn patterns using the Hebbian learning rule."""
        n = self.n_neurons
        self.weights = np.zeros((n, n))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= n
        np.fill_diagonal(self.weights, 0)

    def energy(self, state: np.ndarray) -> float:
        """Compute network energy E = -0.5 * s^T * W * s."""
        return float(-0.5 * state @ self.weights @ state)

    def recall(
        self,
        state: np.ndarray,
        max_steps: int = 100,
        rng: np.random.Generator = None,
        return_history: bool = False,
    ):
        """Retrieve a stored pattern from a probe state via async updates."""
        if rng is None:
            rng = np.random.default_rng()
        state = state.copy()
        history = [self.energy(state)]
        for _ in range(max_steps):
            i = rng.integers(0, self.n_neurons)
            h = self.weights[i] @ state
            if h != 0:
                state[i] = np.sign(h)
            history.append(self.energy(state))
        if return_history:
            return state, history
        return state