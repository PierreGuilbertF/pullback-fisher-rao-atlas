"""
Fisher--Rao geodesic for the two-dimensional push-forward family

    phi_theta(y) = (y_1 + theta_1 tanh y_2,  theta_2 y_2),   q = N(0, I_2).

The base-coordinate scores reduce to
    xi_1(phi(y)) = -y_1 tanh y_2,
    xi_2(phi(y)) = (1 + theta_1 (1 - tanh^2 y_2) y_1 y_2 - y_2^2) / theta_2,
so the metric has the closed form
    g_11 = a,
    g_12 = -theta_1 d / theta_2,
    g_22 = (2 + theta_1^2 c) / theta_2^2,
with a = E[tanh^2 Y], d = E[Y tanh Y (1 - tanh^2 Y)],
c = E[(1 - tanh^2 Y)^2 Y^2] and Y ~ N(0,1). A quadrature check against the
general formula xi_i = tr(J^-1 J_{d_i phi}) + <J^-T grad(log q - log|det J|),
d_i phi> is printed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

N_SEG = 28
N_SNAP = 5
GEOD = "#C44E52"
CHORD = "#9AA0A6"
START = "#4C78A8"
END = "#54A24B"

THETA0 = np.array([0.0, 1.5])
THETA1 = np.array([1.8, 0.5])

# --- the three universal constants, by 1D quadrature ---
y = np.linspace(-8.0, 8.0, 8001)
dy = y[1] - y[0]
weight = np.exp(-0.5 * y**2) / np.sqrt(2.0 * np.pi) * dy
t = np.tanh(y)
s = 1.0 - t**2
A = float(np.sum(weight * t**2))
D = float(np.sum(weight * y * t * s))
C = float(np.sum(weight * s**2 * y**2))
print(f"a={A:.6f}, d={D:.6f}, c={C:.6f}")


def metric(theta):
    t1, t2 = float(theta[0]), max(float(theta[1]), 1e-6)
    return np.array(
        [
            [A, -t1 * D / t2],
            [-t1 * D / t2, (2.0 + t1**2 * C) / t2**2],
        ]
    )


def metric_quadrature(theta):
    """Same metric, from the general base-coordinate formula for xi."""
    t1, t2 = float(theta[0]), float(theta[1])
    grid = np.linspace(-7.0, 7.0, 701)
    step = grid[1] - grid[0]
    Y1, Y2 = np.meshgrid(grid, grid, indexing="ij")
    w = (
        np.exp(-0.5 * (Y1**2 + Y2**2))
        / (2.0 * np.pi)
        * step**2
    )
    sech2 = 1.0 - np.tanh(Y2) ** 2

    # J = [[1, t1 sech2], [0, t2]], det J = t2 (constant in y).
    # d_1 phi = (tanh y2, 0),  d_2 phi = (0, y2).
    # J^{-1} d_1 phi = (tanh y2, 0), J^{-1} d_2 phi = (-t1 sech2 y2 / t2, y2 / t2).
    # grad(log q - log|det J|) = -y.
    xi_1 = 0.0 - (Y1 * np.tanh(Y2))
    xi_2 = (
        1.0 / t2
        + (t1 * sech2 * Y1 * Y2) / t2
        - Y2**2 / t2
    )
    G = np.empty((2, 2))
    G[0, 0] = float(np.sum(w * xi_1**2))
    G[0, 1] = G[1, 0] = float(np.sum(w * xi_1 * xi_2))
    G[1, 1] = float(np.sum(w * xi_2**2))
    return G


for probe in (THETA0, THETA1, np.array([0.9, 0.9])):
    error = np.abs(metric(probe) - metric_quadrature(probe)).max()
    print(f"  theta={probe}: max |G_closed - G_quad| = {error:.2e}")


def energy(interior, theta0, theta1):
    path = np.vstack([theta0, interior.reshape(-1, 2), theta1])
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        delta = b - a
        total += float(delta @ metric(0.5 * (a + b)) @ delta)
    return total


def geodesic(theta0, theta1):
    t = np.linspace(0.0, 1.0, N_SEG + 2)[1:-1]
    initial = np.outer(1.0 - t, theta0) + np.outer(t, theta1)
    direction = theta1 - theta0
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)
    initial += (
        0.05 * np.linalg.norm(direction) * np.sin(np.pi * t)[:, None] * normal
    )

    result = minimize(
        energy,
        initial.ravel(),
        args=(theta0, theta1),
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0), (0.1, 4.0)] * N_SEG,
        options={"maxiter": 800, "maxfun": 200000, "ftol": 1e-13},
    )
    print(
        f"success={result.success}, nit={result.nit}, "
        f"energy={result.fun:.6g}, message={result.message}"
    )
    return np.vstack([theta0, result.x.reshape(-1, 2), theta1])


path = geodesic(THETA0, THETA1)
indices = np.linspace(0, len(path) - 1, N_SNAP, dtype=int)


def transport(Y1, Y2, theta):
    t1, t2 = theta
    return Y1 + t1 * np.tanh(Y2), t2 * Y2


def density(X1, X2, theta):
    t1, t2 = theta
    y2 = X2 / t2
    y1 = X1 - t1 * np.tanh(y2)
    return np.exp(-0.5 * (y1**2 + y2**2)) / (2.0 * np.pi * t2)


fig = plt.figure(figsize=(11.0, 6.8))
outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.38)
top = outer[0].subgridspec(1, 2, wspace=0.24)

ax = fig.add_subplot(top[0, 0])
ax.plot(
    [path[0, 0], path[-1, 0]],
    [path[0, 1], path[-1, 1]],
    "--",
    color=CHORD,
    lw=1.4,
    label="Euclidean chord",
)
ax.plot(path[:, 0], path[:, 1], color=GEOD, lw=2.3, label="geodesic")
ax.plot(*path[0], "o", color=START, ms=8, zorder=5)
ax.plot(*path[-1], "o", color=END, ms=8, zorder=5)
ax.set_xlabel(r"$\theta_{1}$")
ax.set_ylabel(r"$\theta_{2}$")
ax.set_title(r"Path in $\Theta\subset\mathbb{R}^{2}$", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=7, frameon=False, loc="best")

ax_grid = fig.add_subplot(top[0, 1])
lines = np.linspace(-2.4, 2.4, 11)
fine = np.linspace(-2.4, 2.4, 200)
for color, theta, label in (
    (START, path[0], "start"),
    (END, path[-1], "end"),
):
    for k, value in enumerate(lines):
        X1, X2 = transport(fine, np.full_like(fine, value), theta)
        ax_grid.plot(X1, X2, color=color, lw=0.7, label=label if k == 0 else None)
        X1, X2 = transport(np.full_like(fine, value), fine, theta)
        ax_grid.plot(X1, X2, color=color, lw=0.7)
ax_grid.set_title(
    r"Image of the grid under $\phi_{\theta}$ at the endpoints", fontsize=10
)
ax_grid.set_xlabel(r"$y_{1}$")
ax_grid.set_ylabel(r"$y_{2}$")
ax_grid.set_aspect("equal")
ax_grid.legend(fontsize=7, frameon=False, loc="upper left")

cells = outer[1].subgridspec(1, N_SNAP, wspace=0.12)
grid1 = np.linspace(-4.6, 4.6, 340)
grid2 = np.linspace(-3.6, 3.6, 300)
X1, X2 = np.meshgrid(grid1, grid2)
for position, index in enumerate(indices):
    theta = path[index]
    Z = density(X1, X2, theta)
    ax_snap = fig.add_subplot(cells[0, position])
    ax_snap.contourf(X1, X2, Z, levels=10, cmap="Blues")
    ax_snap.contour(X1, X2, Z, levels=10, colors="#1b3a5c", linewidths=0.45)
    ax_snap.set_aspect("equal")
    ax_snap.set_xticks([])
    ax_snap.set_yticks([])
    ax_snap.set_title(rf"$({theta[0]:.2f},{theta[1]:.2f})$", fontsize=7)
    if position == 0:
        ax_snap.set_ylabel(r"$y_{2}$", fontsize=8)

fig.suptitle(
    r"Fisher--Rao geodesic for "
    r"$\phi_{\theta}(x)=(x_{1}+\theta_{1}\tanh x_{2},\ \theta_{2}x_{2})$",
    fontsize=12,
    y=0.98,
)
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.05)

output = OUT / "23_pushforward_geodesic_2d.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Wrote {output}")
for position, index in enumerate(indices):
    print(f"  {position+1}: theta=({path[index,0]:.3f},{path[index,1]:.3f})")

# How far is the path from log-linear in theta_2 at given theta_1?
fraction = (path[:, 0] - THETA0[0]) / (THETA1[0] - THETA0[0])
reference = np.exp(
    (1.0 - fraction) * np.log(THETA0[1]) + fraction * np.log(THETA1[1])
)
print(
    "max relative gap to log-linear theta_2: "
    f"{np.abs(path[:, 1] / reference - 1.0).max():.3f}"
)
# plt.show()
