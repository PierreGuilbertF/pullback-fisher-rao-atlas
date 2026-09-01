"""
Mahalanobis view of g_{μ,Σ} at fixed Σ: the metric on mean displacements dμ
is v ↦ vᵀ Σ⁻¹ v. Σ is rotated by π/4 with principal variances (1, 4).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

THETA = np.pi / 4
LAM1, LAM2 = 1.0, 4.0  # principal variances (stretch 1 and 4)
LIM = 3.0
N = 400


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def covariance(lam1, lam2, theta):
    R = rotation(theta)
    return R @ np.diag([lam1, lam2]) @ R.T


Sigma = covariance(LAM1, LAM2, THETA)
Sigma_inv = np.linalg.inv(Sigma)

u = np.linspace(-LIM, LIM, N)
v = np.linspace(-LIM, LIM, N)
U, V = np.meshgrid(u, v)
# Mahalanobis norm of the tangent vector (u, v) = dμ
NORM = np.sqrt(
    Sigma_inv[0, 0] * U**2
    + 2.0 * Sigma_inv[0, 1] * U * V
    + Sigma_inv[1, 1] * V**2
)

fig, ax = plt.subplots(figsize=(6.2, 5.6))
levels = np.linspace(0.0, NORM.max(), 24)
cf = ax.contourf(U, V, NORM, levels=levels, cmap="coolwarm")
cs = ax.contour(
    U, V, NORM, levels=np.arange(0.5, NORM.max(), 0.5), colors="k", linewidths=0.6
)
ax.clabel(cs, inline=True, fontsize=8, fmt=r"$%.1f$")

# Principal axes of Σ (directions of stretch), length ∝ σ_i
eigvals, eigvecs = np.linalg.eigh(Sigma)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
colors = ("#C44E52", "#2A6F97")
for i in range(2):
    direction = eigvecs[:, i] * np.sqrt(eigvals[i])
    ax.add_patch(
        FancyArrowPatch(
            -direction,
            direction,
            arrowstyle="<->",
            mutation_scale=12,
            color=colors[i],
            lw=1.6,
            zorder=5,
        )
    )

ax.plot(0, 0, "ko", ms=6, zorder=6)
ax.set_aspect("equal")
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_xlabel(r"$d\mu_{1}$")
ax.set_ylabel(r"$d\mu_{2}$")
ax.set_title(
    r"Mahalanobis metric $g_{\mu,\Sigma}(d\mu)=d\mu^{\top}\Sigma^{-1}d\mu$"
    "\n"
    r"$\Sigma$: $\theta=\pi/4$, principal variances $(1,\ 4)$"
)
cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label(r"$\sqrt{d\mu^{\top}\Sigma^{-1}d\mu}$")

fig.tight_layout()
path = OUT / "07_mahalanobis_metric.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"Σ =\n{Sigma}")
# plt.show()
