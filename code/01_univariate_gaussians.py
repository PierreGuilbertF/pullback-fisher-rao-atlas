"""Three univariate Gaussians with different standard deviations, side by side."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

MU = 0.0
SIGMAS = (0.5, 1.0, 2.0)
X = np.linspace(-6.0, 6.0, 600)


def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)

for ax, sigma in zip(axes, SIGMAS):
    y = gaussian(X, MU, sigma)
    ax.fill_between(X, y, color="#4C78A8", alpha=0.35)
    ax.plot(X, y, color="#4C78A8", lw=2.0)
    ax.set_title(rf"$\sigma = {sigma:g}$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0.0, 1.05 * (1.0 / (np.sqrt(2.0 * np.pi) * min(SIGMAS))))
    ax.axvline(MU, color="0.4", ls="--", lw=0.8)

axes[0].set_ylabel(r"$p_{\mu,\sigma}(x)$")
fig.suptitle(rf"Centered Gaussians ($\mu = {MU:g}$) for three standard deviations", y=1.02)
fig.tight_layout()

path = OUT / "01_univariate_gaussians.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()  # uncomment for interactive display
