"""
Bivariate elliptical Laplace densities, centered at (0, 0), with different
dispersion matrices: isotropic, axis-aligned stretch, then orientations π/4
and 3π/4. Drawn as translucent 2.5D surfaces with floor contours.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

MU = np.zeros(2)
LIM = 4.0
N_GRID = 161
LEVELS = 8
ELEV, AZIM = 32.0, -55.0
# n = 2 normalization constant from the paper
C2 = 1.0 / (8.0 * np.pi)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def dispersion_from_axes(lam1, lam2, theta):
    """Σ = R diag(λ1, λ2) Rᵀ (dispersion parameters, not covariances)."""
    R = rotation(theta)
    return R @ np.diag([lam1, lam2]) @ R.T


def laplace_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    quad = np.einsum("...i,ij,...j->...", pos, inv, pos)
    r = np.sqrt(np.maximum(quad, 0.0))
    return C2 / np.sqrt(det) * np.exp(-0.5 * r)


CASES = [
    (r"$\Sigma = I$ (isotropic)", 1.0, 1.0, 0.0),
    (r"aligned stretch", 4.0, 1.0, 0.0),
    (r"orientation $\theta = \pi/4$", 4.0, 1.0, np.pi / 4),
    (r"orientation $\theta = 3\pi/4$", 4.0, 1.0, 3.0 * np.pi / 4),
]

x = np.linspace(-LIM, LIM, N_GRID)
y = np.linspace(-LIM, LIM, N_GRID)
X, Y = np.meshgrid(x, y)

Sigmas = [dispersion_from_axes(l1, l2, th) for _, l1, l2, th in CASES]
zmax = max(laplace_pdf(X, Y, MU, S).max() for S in Sigmas)

fig = plt.figure(figsize=(13.5, 3.8))

for k, ((title, _, _, _), Sigma) in enumerate(zip(CASES, Sigmas)):
    Z = laplace_pdf(X, Y, MU, Sigma)
    ax = fig.add_subplot(1, 4, k + 1, projection="3d")
    ax.contourf(X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, cmap="Blues", alpha=0.9)
    ax.contour(
        X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, colors="#1b3a5c", linewidths=0.8
    )
    ax.plot_surface(
        X,
        Y,
        Z,
        color="#4C78A8",
        alpha=0.22,
        rstride=8,
        cstride=8,
        edgecolor="#2b4a6f",
        linewidth=0.3,
        shade=False,
    )
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_zlim(0.0, 1.05 * zmax)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$x_{1}$", labelpad=-6)
    ax.set_ylabel(r"$x_{2}$", labelpad=-6)
    ax.set_title(title, fontsize=11, pad=0)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 0.62))

fig.suptitle(
    r"Elliptical Laplace family in $\mathbb{R}^{2}$"
    r" ($\mu = 0$, various dispersions)",
    y=1.02,
)
fig.subplots_adjust(left=0.01, right=0.99, wspace=0.02, top=0.86, bottom=0.02)

path = OUT / "13_multivariate_laplace.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
for (title, _, _, _), Sigma in zip(CASES, Sigmas):
    print(f"  {title}: Σ = {np.array2string(Sigma, precision=3)}")
# plt.show()
