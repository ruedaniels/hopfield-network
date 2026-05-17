import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt


def plot_retrieval_grid(
    patterns: np.ndarray,
    probes: np.ndarray,
    recalled: np.ndarray,
    filename: str = "figures/retrieval_grid.png",
) -> None:
    """Save a grid showing original, corrupted, and recalled patterns."""
    p = len(patterns)
    fig, axes = plt.subplots(p, 3, figsize=(6, 2 * p))
    cols = ["Original", "Corrupted", "Recalled"]

    for col_idx, title in enumerate(cols):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")

    for row, (orig, probe, rec) in enumerate(zip(patterns, probes, recalled)):
        size = int(np.sqrt(len(orig)))
        for col, pat in enumerate([orig, probe, rec]):
            ax = axes[row, col]
            ax.imshow(((pat + 1) / 2).reshape(size, size), cmap="binary", vmin=0, vmax=1)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_convergence(
    energy_histories: list,
    filename: str = "figures/convergence.png",
) -> None:
    """Save a plot of energy vs update step for each retrieval."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(energy_histories):
        ax.plot(history, label=f"Pattern {i + 1}")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Convergence During Recall")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_capacity_heatmap(
    sweep_results: list,
    noise_levels: list,
    accuracies: np.ndarray,
    n_neurons: int,
    filename: str = "figures/capacity_heatmap.png",
) -> None:
    """Save a heatmap of recall accuracy across pattern count and noise level."""
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(accuracies, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
    ax.set_yticks(range(len(sweep_results)))
    ax.set_yticklabels([str(p) for p in sweep_results])
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Number of Patterns (P)")
    ax.set_title("Recall Accuracy: Pattern Count vs Noise Level")
    capacity = round(0.138 * n_neurons)
    if capacity <= max(sweep_results):
        cap_idx = sweep_results.index(capacity) if capacity in sweep_results else None
        if cap_idx is not None:
            ax.axhline(y=cap_idx, color="blue", linestyle="--", linewidth=2,
                      label=f"Theoretical capacity (P={capacity})")
            ax.legend()
    plt.colorbar(im, ax=ax, label="Recall Accuracy")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()