"""Three univariate Laplace densities with different dispersions σ, side by side."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

MU = 0.0
SIGMAS = (0.5, 1.0, 2.0)
X = np.linspace(-8.0, 8.0, 800)


def laplace(x, mu, sigma):
    """Normalized univariate density from the paper: (1/(4σ)) exp(-|x-μ|/(2σ))."""
    return (1.0 / (4.0 * sigma)) * np.exp(-np.abs(x - mu) / (2.0 * sigma))


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)

ymax = 1.05 * laplace(0.0, MU, min(SIGMAS))
for ax, sigma in zip(axes, SIGMAS):
    y = laplace(X, MU, sigma)
    ax.fill_between(X, y, color="#4C78A8", alpha=0.35)
    ax.plot(X, y, color="#4C78A8", lw=2.0)
    ax.set_title(rf"$\sigma = {sigma:g}$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0.0, ymax)
    ax.axvline(MU, color="0.4", ls="--", lw=0.8)

axes[0].set_ylabel(r"$p_{\mu,\sigma}(x)$")
fig.suptitle(rf"Centered Laplace densities ($\mu = {MU:g}$) for three dispersions", y=1.02)
fig.tight_layout()

path = OUT / "12_univariate_laplace.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()
