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

def test_capacity_sweep_returns_correct_length():
    from hopfield.analysis import capacity_sweep
    rng = np.random.default_rng(42)
    results = capacity_sweep(
        p_values=[1, 3, 5],
        n_neurons=50,
        noise_ratio=0.20,
        n_trials=5,
        rng=rng,
    )
    assert len(results) == 3
    for p, acc in results:
        assert 0.0 <= acc <= 1.0


def test_visualize_retrieval_grid_creates_file():
    import tempfile, os
    from hopfield.visualize import plot_retrieval_grid
    rng = np.random.default_rng(42)
    patterns = np.array([np.ones(100) * rng.choice([-1, 1]) for _ in range(3)])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name
    plot_retrieval_grid(patterns, patterns, patterns, filename=fname)
    assert os.path.exists(fname)
    assert os.path.getsize(fname) > 1000
    os.unlink(fname)


def test_visualize_convergence_creates_file():
    import tempfile, os
    from hopfield.visualize import plot_convergence
    histories = [[0.0, -1.0, -2.0, -2.5], [0.0, -0.5, -1.0, -1.0]]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name
    plot_convergence(histories, filename=fname)
    assert os.path.exists(fname)
    assert os.path.getsize(fname) > 1000
    os.unlink(fname)

def test_visualize_capacity_heatmap_creates_file():
    import tempfile, os
    from hopfield.visualize import plot_capacity_heatmap
    p_values = [1, 3, 5]
    noise_levels = [0.10, 0.20, 0.30]
    accuracies = np.array([
        [1.0, 0.9, 0.8],
        [0.8, 0.6, 0.4],
        [0.4, 0.2, 0.1],
    ])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fname = f.name
    plot_capacity_heatmap(p_values, noise_levels, accuracies, n_neurons=100, filename=fname)
    assert os.path.exists(fname)
    assert os.path.getsize(fname) > 1000
    os.unlink(fname)