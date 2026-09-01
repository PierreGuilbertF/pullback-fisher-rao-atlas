"""
Fisher--Rao geodesics for three two-parameter Gibbs families:
  1. E subset R,  phi(x)=(x,x^2)
  2. E subset R,  phi(x)=(x^2,cos x)
  3. E subset R^2, phi(x)=(cos(x1^2+x2^2),
                           sin(4 x1 x2)+x1+x2)

The metric is g_theta = Cov_theta(phi(X)). Geodesics are approximated by
minimizing a discrete Riemannian energy with fixed endpoints.
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

# One-dimensional domains
x_short = np.linspace(-2.0, 2.0, 500)
dx_short = x_short[1] - x_short[0]
phi_poly = np.column_stack([x_short, x_short**2])
phi_cos = np.column_stack([x_short**2, np.cos(x_short)])

# Two-dimensional domain
lim = 2.5
grid = np.linspace(-lim, lim, 70)
X, Y = np.meshgrid(grid, grid)
darea = (grid[1] - grid[0]) ** 2
phi_trig = np.stack(
    [np.cos(X**2 + Y**2), np.sin(4.0 * X * Y) + X + Y],
    axis=-1,
)


def metric_and_density(theta, phi, measure):
    """Return Cov_theta(phi) and the normalized density on the grid."""
    theta = np.asarray(theta, dtype=float)
    logu = -np.einsum("...i,i->...", phi, theta)
    logu -= float(logu.max())
    u = np.exp(logu)
    density = u / float(u.sum() * measure)

    values = phi.reshape(-1, 2)
    weights = density.ravel() * measure
    mean = weights @ values
    centered = values - mean
    metric = (centered * weights[:, None]).T @ centered
    metric = 0.5 * (metric + metric.T) + 1e-10 * np.eye(2)
    return metric, density


def energy(interior, theta0, theta1, phi, measure):
    path = np.vstack([theta0, interior.reshape(-1, 2), theta1])
    value = 0.0
    for a, b in zip(path[:-1], path[1:]):
        delta = b - a
        metric, _ = metric_and_density(0.5 * (a + b), phi, measure)
        value += float(delta @ metric @ delta)
    return value


def geodesic(theta0, theta1, phi, measure):
    theta0 = np.asarray(theta0, dtype=float)
    theta1 = np.asarray(theta1, dtype=float)
    t = np.linspace(0.0, 1.0, N_SEG + 2)[1:-1]
    initial = np.outer(1.0 - t, theta0) + np.outer(t, theta1)

    # Give the optimizer a small transverse perturbation.
    direction = theta1 - theta0
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)
    initial += (
        0.05
        * np.linalg.norm(direction)
        * np.sin(np.pi * t)[:, None]
        * normal
    )

    result = minimize(
        energy,
        initial.ravel(),
        args=(theta0, theta1, phi, measure),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-11},
    )
    print(
        f"success={result.success}, nit={result.nit}, "
        f"energy={result.fun:.6g}, message={result.message}"
    )
    return np.vstack([theta0, result.x.reshape(-1, 2), theta1])


def draw_parameter_path(ax, path, title):
    theta0, theta1 = path[0], path[-1]
    ax.plot(
        [theta0[0], theta1[0]],
        [theta0[1], theta1[1]],
        "--",
        color=CHORD,
        lw=1.4,
        label="Euclidean chord",
    )
    ax.plot(path[:, 0], path[:, 1], color=GEOD, lw=2.3, label="geodesic")
    ax.plot(*theta0, "o", color=START, ms=8, zorder=5)
    ax.plot(*theta1, "o", color=END, ms=8, zorder=5)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")


def snapshot_indices(path):
    return np.linspace(0, len(path) - 1, N_SNAP, dtype=int)


def draw_1d_snapshots(cell, path, phi, measure):
    axes = [
        fig.add_subplot(cell.subgridspec(1, N_SNAP, wspace=0.10)[0, i])
        for i in range(N_SNAP)
    ]
    densities = [
        metric_and_density(path[i], phi, measure)[1]
        for i in snapshot_indices(path)
    ]
    ymax = 1.08 * max(p.max() for p in densities)
    for ax, index, density in zip(axes, snapshot_indices(path), densities):
        ax.fill_between(x_short, density, color="#4C78A8", alpha=0.30)
        ax.plot(x_short, density, color="#4C78A8", lw=1.3)
        ax.set_xlim(x_short[0], x_short[-1])
        ax.set_ylim(0.0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        theta = path[index]
        ax.set_title(rf"$({theta[0]:.2f},{theta[1]:.2f})$", fontsize=7)
    axes[0].set_ylabel(r"$p_\theta(x)$", fontsize=8)
    return axes


def draw_2d_snapshots(cell, path, phi, measure):
    axes = [
        fig.add_subplot(cell.subgridspec(1, N_SNAP, wspace=0.08)[0, i])
        for i in range(N_SNAP)
    ]
    for ax, index in zip(axes, snapshot_indices(path)):
        _, density = metric_and_density(path[index], phi, measure)
        ax.contourf(X, Y, density, levels=7, cmap="Blues")
        ax.contour(
            X, Y, density, levels=7, colors="#1b3a5c", linewidths=0.45
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        theta = path[index]
        ax.set_title(rf"$({theta[0]:.2f},{theta[1]:.2f})$", fontsize=7)
    axes[0].set_ylabel(r"$x_2$", fontsize=8)
    return axes


cases = [
    {
        "title": r"$E\subset\mathbb{R},\quad\phi(x)=(x,x^2)$",
        "phi": phi_poly,
        "measure": dx_short,
        "theta0": np.array([-0.9, 0.35]),
        "theta1": np.array([0.9, 1.25]),
        "kind": "1d",
    },
    {
        "title": r"$E\subset\mathbb{R},\quad\phi(x)=(x^2,\cos x)$",
        "phi": phi_cos,
        "measure": dx_short,
        "theta0": np.array([0.45, 1.2]),
        "theta1": np.array([1.15, 1.1]),
        "kind": "1d",
    },
    {
        "title": (
            r"$E\subset\mathbb{R}^2,\quad"
            r"\phi=(\cos(x_1^2+x_2^2),\,\sin(4x_1x_2)+x_1+x_2)$"
        ),
        "phi": phi_trig,
        "measure": darea,
        "theta0": np.array([0.25, 0.15]),
        "theta1": np.array([1.35, 0.85]),
        "kind": "2d",
    },
]

paths = []
for case in cases:
    print(case["title"])
    paths.append(
        geodesic(
            case["theta0"],
            case["theta1"],
            case["phi"],
            case["measure"],
        )
    )

fig = plt.figure(figsize=(12.2, 7.0))
outer = fig.add_gridspec(
    2, 3, height_ratios=[1.15, 0.9], hspace=0.42, wspace=0.28
)

for column, (case, path) in enumerate(zip(cases, paths)):
    ax = fig.add_subplot(outer[0, column])
    draw_parameter_path(ax, path, case["title"])
    if column == 0:
        ax.legend(fontsize=7, frameon=False, loc="best")

    if case["kind"] == "1d":
        draw_1d_snapshots(outer[1, column], path, case["phi"], case["measure"])
    else:
        draw_2d_snapshots(outer[1, column], path, case["phi"], case["measure"])

fig.suptitle(
    r"Fisher--Rao geodesics in three parameter spaces "
    r"$\Theta\subset\mathbb{R}^2$",
    fontsize=12,
    y=0.98,
)
fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.06)

output = OUT / "17_gibbs_geodesics.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Wrote {output}")
# plt.show()
