"""
Overlap of univariate Laplace densities: same absolute shift in μ (resp. σ)
costs more under a small dispersion than under a large one. Absolute
difference |p - q| is filled (same layout as the Gaussian overlap figures).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR_A = "#4C78A8"
COLOR_B = "#C44E52"
COLOR_DIFF = "#E8A838"


def laplace(x, mu, sigma):
    return (1.0 / (4.0 * sigma)) * np.exp(-np.abs(x - mu) / (2.0 * sigma))


def panel(ax, mu_a, sig_a, mu_b, sig_b, title, x_pad=6.0):
    lo = min(mu_a - x_pad * sig_a, mu_b - x_pad * sig_b)
    hi = max(mu_a + x_pad * sig_a, mu_b + x_pad * sig_b)
    x = np.linspace(lo, hi, 1000)
    ya = laplace(x, mu_a, sig_a)
    yb = laplace(x, mu_b, sig_b)
    diff = np.abs(ya - yb)

    ax.fill_between(x, 0.0, diff, color=COLOR_DIFF, alpha=0.55, label=r"$|p-q|$")
    ax.plot(
        x, ya, color=COLOR_A, lw=2.0, label=rf"$L({mu_a:g},\ {sig_a:g})$"
    )
    ax.plot(
        x, yb, color=COLOR_B, lw=2.0, label=rf"$L({mu_b:g},\ {sig_b:g})$"
    )

    l1 = float(np.trapz(diff, x))
    ax.set_title(title + "\n" + rf"$\int |p-q|\,dx = {l1:.3f}$", fontsize=11)
    ax.set_xlabel(r"$x$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def make_figure(pairs, suptitle, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, (mu_a, sig_a, mu_b, sig_b, title) in zip(axes, pairs):
        panel(ax, mu_a, sig_a, mu_b, sig_b, title)
    axes[0].set_ylabel(r"$p(x)$")
    fig.suptitle(suptitle, y=1.04, fontsize=12)
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


make_figure(
    pairs=[
        (0.0, 0.1, 0.05, 0.1, r"Small dispersion ($\sigma = 0.1$)"),
        (0.0, 1.0, 0.05, 1.0, r"Large dispersion ($\sigma = 1$)"),
    ],
    suptitle=r"Laplace: same $\Delta\mu = 0.05$, loss of overlap",
    filename="14_laplace_overlap_mean.png",
)

make_figure(
    pairs=[
        (0.0, 0.1, 0.0, 0.15, r"Small scale ($\sigma : 0.1 \rightarrow 0.15$)"),
        (0.0, 1.0, 0.0, 1.05, r"Large scale ($\sigma : 1 \rightarrow 1.05$)"),
    ],
    suptitle=r"Laplace: same $\Delta\sigma = 0.05$, loss of overlap",
    filename="14_laplace_overlap_sigma.png",
)
# plt.show()
