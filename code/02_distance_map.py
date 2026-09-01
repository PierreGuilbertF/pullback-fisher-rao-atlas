"""
Fisher--Rao distance map on the (μ, σ) half-plane, from the standard normal (0, 1).

Metric:  ds² = (dμ² + 2 dσ²) / σ²
Distance: d((μ,σ), (0,1)) = √2 arcosh( (μ² + 2(σ² + 1)) / (4σ) )
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

MU0, SIG0 = 0.0, 1.0


def fisher_rao_distance(mu, sigma, mu0=MU0, sigma0=SIG0):
    """Closed-form distance for ds² = (dμ² + 2 dσ²)/σ²."""
    arg = (mu - mu0) ** 2 + 2.0 * (sigma**2 + sigma0**2)
    arg = arg / (4.0 * sigma * sigma0)
    # numerical floor: arg >= 1
    arg = np.maximum(arg, 1.0)
    return np.sqrt(2.0) * np.arccosh(arg)


mu = np.linspace(-4.0, 4.0, 400)
sigma = np.linspace(0.15, 3.5, 350)
MU, SIG = np.meshgrid(mu, sigma)
DIST = fisher_rao_distance(MU, SIG)

fig, ax = plt.subplots(figsize=(7.2, 5.2))

levels_fill = np.linspace(0.0, DIST.max(), 40)
cmap = plt.get_cmap("coolwarm")
cf = ax.contourf(MU, SIG, DIST, levels=levels_fill, cmap=cmap)

iso = np.arange(0.5, DIST.max(), 0.5)
cs = ax.contour(MU, SIG, DIST, levels=iso, colors="k", linewidths=0.7, alpha=0.75)
ax.clabel(cs, iso[::2], inline=True, fmt=r"$d=%.1f$", fontsize=8)

ax.plot(MU0, SIG0, "o", color="white", ms=9, mew=1.4, mec="black", zorder=5)
ax.annotate(
    r"$(\mu,\sigma)=(0,1)$",
    xy=(MU0, SIG0),
    xytext=(0.6, 0.45),
    textcoords="data",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
)

ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\sigma$")
ax.set_title(
    r"Fisher--Rao distance from $\mathcal{N}(0,1)$"
    "\n"
    r"($ds^{2}=(d\mu^{2}+2\,d\sigma^{2})/\sigma^{2}$)"
)
cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label(r"$d((\mu,\sigma),(0,1))$")

fig.tight_layout()
path = OUT / "02_distance_map.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()  # uncomment for interactive display
