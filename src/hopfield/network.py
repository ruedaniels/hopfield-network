import numpy as np


class HopfieldNetwork:
    """
    A Hopfield Network implementing associative memory via energy minimisation.

    Patterns are stored as stable energy minima using Hebbian learning.
    Retrieval is performed via asynchronous state updates that guarantee
    monotonically non-increasing energy at every step.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the network. Patterns must have length n_neurons.

    Examples
    --------
    >>> net = HopfieldNetwork(100)
    >>> patterns = np.array([np.random.choice([-1, 1], 100)])
    >>> net.train(patterns)
    >>> recalled = net.recall(patterns[0])
    """

    def __init__(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns: np.ndarray) -> None:
        """
        Learn patterns using the Hebbian learning rule.

        Computes W = (1/N) * sum_p( outer(xi_p, xi_p) ) with zero diagonal.
        Calling train() overwrites any previously learned weights.

        Parameters
        ----------
        patterns : np.ndarray
            Array of shape (P, N) containing P bipolar (+-1) patterns of length N.

        Raises
        ------
        ValueError
            If pattern length does not match n_neurons.
        """
        if patterns.shape[1] != self.n_neurons:
            raise ValueError(
                f"Pattern length {patterns.shape[1]} does not match "
                f"n_neurons {self.n_neurons}"
            )
        n = self.n_neurons
        self.weights = np.zeros((n, n))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= n
        np.fill_diagonal(self.weights, 0)

    def energy(self, state: np.ndarray) -> float:
        """
        Compute network energy E = -0.5 * s^T * W * s.

        Lower energy indicates a more stable state. Stored patterns
        are local energy minima.

        Parameters
        ----------
        state : np.ndarray
            Bipolar (+-1) state vector of shape (N,).

        Returns
        -------
        float
            Scalar energy value for the given state.
        """
        return float(-0.5 * state @ self.weights @ state)

    def recall(
        self,
        state: np.ndarray,
        max_steps: int = 100,
        rng: np.random.Generator | None = None,
        return_history: bool = False,
    ) -> np.ndarray | tuple:
        """
        Retrieve a stored pattern from a probe state via asynchronous updates.

        At each step, one neuron is selected at random and its state updated
        as sign(W[i] . s). Energy is guaranteed non-increasing at every step.

        Parameters
        ----------
        state : np.ndarray
            Initial probe state of shape (N,) with bipolar (+-1) values.
        max_steps : int, optional
            Maximum number of update steps. Default is 100.
        rng : np.random.Generator or None, optional
            Seeded random number generator for reproducibility.
            If None, a default generator is created.
        return_history : bool, optional
            If True, return energy at each step alongside the final state.

        Returns
        -------
        np.ndarray
            Final recalled state of shape (N,).
        tuple of (np.ndarray, list of float)
            If return_history=True, returns (final_state, energy_history).
        """
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