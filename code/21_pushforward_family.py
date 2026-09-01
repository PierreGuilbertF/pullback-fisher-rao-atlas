"""
Push-forward families p_theta = (phi_theta)_# q, base law q = N(0, I).

Top row:    the transport map phi_theta itself (1D curves, 2D deformed grid).
Bottom row: the base density q and its push-forwards p_theta.

Examples:
  1. E subset R,   phi_theta(y) = theta_1 y + theta_2 y^3
  2. E subset R,   phi_theta(y) = y + theta_1 sin(theta_2 y)
  3. E subset R^2, phi_theta(y) = (y_1 + theta_1 tanh y_2, theta_2 y_2)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

BASE = "#9AA0A6"
SHADES = ["#A7C4DE", "#4C78A8", "#1F4E79"]

y = np.linspace(-4.0, 4.0, 1200)
q_1d = np.exp(-0.5 * y**2) / np.sqrt(2.0 * np.pi)


def cubic(theta):
    t1, t2 = theta
    return t1 * y + t2 * y**3, t1 + 3.0 * t2 * y**2


def sine(theta):
    t1, t2 = theta
    return y + t1 * np.sin(t2 * y), 1.0 + t1 * t2 * np.cos(t2 * y)


CASES_1D = [
    {
        "map": cubic,
        "title": r"$\phi_{\theta}(x)=\theta_{1}x+\theta_{2}x^{3}$",
        "thetas": [(1.0, 0.02), (0.7, 0.25), (0.35, 0.6)],
        "xlim": 5.0,
    },
    {
        "map": sine,
        "title": r"$\phi_{\theta}(x)=x+\theta_{1}\sin(\theta_{2}x)$",
        "thetas": [(0.15, 1.0), (0.4, 1.1), (0.6, 1.15)],
        "xlim": 5.0,
    },
]

THETA_2D = (1.2, 0.6)


def transport_2d(Y1, Y2, theta):
    t1, t2 = theta
    return Y1 + t1 * np.tanh(Y2), t2 * Y2


def density_2d(X1, X2, theta):
    """p_theta(x) = q(psi(x)) / |det J| with det J = theta_2."""
    t1, t2 = theta
    y2 = X2 / t2
    y1 = X1 - t1 * np.tanh(y2)
    return np.exp(-0.5 * (y1**2 + y2**2)) / (2.0 * np.pi * t2)


fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.4))

for column, case in enumerate(CASES_1D):
    ax_map, ax_den = axes[0, column], axes[1, column]

    ax_map.plot(y, y, "--", color=BASE, lw=1.2, label=r"$\mathrm{id}$")
    ax_den.plot(y, q_1d, "--", color=BASE, lw=1.4, label=r"$q$")

    for shade, theta in zip(SHADES, case["thetas"]):
        x, derivative = case["map"](theta)
        label = rf"$\theta=({theta[0]:g},{theta[1]:g})$"
        ax_map.plot(y, x, color=shade, lw=1.8, label=label)
        ax_den.plot(x, q_1d / derivative, color=shade, lw=1.8, label=label)

    ax_map.set_title(case["title"], fontsize=10)
    ax_map.set_xlim(-3.0, 3.0)
    ax_map.set_ylim(-case["xlim"], case["xlim"])
    ax_map.set_xlabel(r"$x$")
    ax_map.grid(alpha=0.25)

    ax_den.set_xlim(-case["xlim"], case["xlim"])
    ax_den.set_ylim(bottom=0.0)
    ax_den.set_xlabel(r"$y$")
    ax_den.grid(alpha=0.25)

axes[0, 0].set_ylabel(r"$\phi_{\theta}(x)$")
axes[1, 0].set_ylabel(r"$p_{\theta}(y)$")
axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
axes[1, 1].legend(fontsize=7, frameon=False, loc="upper left")

# --- 2D example: deformed grid, then density contours ---
ax_map, ax_den = axes[0, 2], axes[1, 2]
lines = np.linspace(-2.4, 2.4, 13)
fine = np.linspace(-2.4, 2.4, 200)
for value in lines:
    X1, X2 = transport_2d(fine, np.full_like(fine, value), THETA_2D)
    ax_map.plot(X1, X2, color=SHADES[1], lw=0.8)
    X1, X2 = transport_2d(np.full_like(fine, value), fine, THETA_2D)
    ax_map.plot(X1, X2, color=SHADES[1], lw=0.8)
ax_map.set_title(
    r"$\phi_{\theta}(x)=(x_{1}+\theta_{1}\tanh x_{2},\ \theta_{2}x_{2})$"
    + "\n"
    + rf"$\theta=({THETA_2D[0]:g},{THETA_2D[1]:g})$",
    fontsize=10,
)
ax_map.set_xlabel(r"$y_{1}$")
ax_map.set_ylabel(r"$y_{2}$")
ax_map.set_aspect("equal")
ax_map.set_xlim(-4.0, 4.0)
ax_map.set_ylim(-2.6, 2.6)

grid = np.linspace(-4.0, 4.0, 400)
X1, X2 = np.meshgrid(grid, grid)
Z = density_2d(X1, X2, THETA_2D)
Q = np.exp(-0.5 * (X1**2 + X2**2)) / (2.0 * np.pi)
ax_den.contourf(X1, X2, Z, levels=8, cmap="Blues")
ax_den.contour(X1, X2, Z, levels=8, colors="#1b3a5c", linewidths=0.5)
ax_den.contour(X1, X2, Q, levels=5, colors=BASE, linewidths=0.7,
               linestyles="dashed")
ax_den.set_aspect("equal")
ax_den.set_xlabel(r"$y_{1}$")
ax_den.set_ylabel(r"$y_{2}$")
ax_den.set_xlim(-4.0, 4.0)
ax_den.set_ylim(-2.6, 2.6)

fig.suptitle(
    r"Push-forward families from a base $q=\mathcal{N}(0,I)$: "
    r"the transport $\phi_{\theta}$ (top), the densities $p_{\theta}$ (bottom)",
    y=1.0,
    fontsize=12,
)
fig.tight_layout()

output = OUT / "21_pushforward_family.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Wrote {output}")
# plt.show()
