import math

import matplotlib.pyplot as plt


def plot_ecg(signal):
    plt.plot(signal)
    plt.title("ECG Signal")
    plt.show()


def plot_ecg_grid(signals, title="Generated ECG Signals", save_path=None):
    """Plot multiple ECG signals in a grid.

    Args:
        signals:   List or array of 1-D signals, or a 2-D array (N, L).
        title:     Figure title.
        save_path: If given, save to this path instead of showing interactively.
    """
    n    = len(signals)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2))
    axes = [axes] if n == 1 else axes.flatten().tolist()

    for i, (ax, sig) in enumerate(zip(axes, signals)):
        ax.plot(sig, linewidth=0.8)
        ax.set_title(f"Sample {i + 1}", fontsize=9)
        ax.set_xlabel("Time step", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
