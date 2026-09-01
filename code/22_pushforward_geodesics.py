"""
Fisher--Rao geodesics for two push-forward families on E subset R, with base
law q = N(0, 1):

  1. phi_theta(y) = theta_1 y + theta_2 y^3
  2. phi_theta(y) = y + theta_1 sin(theta_2 y)

The metric is g_ij = E_q[xi_i(phi(Y)) xi_j(phi(Y))] with
    xi_i(phi(y)) = d_i phi'(y)/phi'(y)
                   + (d_i phi(y)/phi'(y)) (log q)'(y)
                   - (d_i phi(y)/phi'(y)) phi''(y)/phi'(y),
i.e. the base-coordinate form of xi = div(p v)/p. A finite-difference check of
d_i log p_theta is printed for one parameter value of each family.

Geodesics are approximated by minimizing a discrete Riemannian energy.
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

Y = np.linspace(-7.0, 7.0, 4001)
DY = Y[1] - Y[0]
Q = np.exp(-0.5 * Y**2) / np.sqrt(2.0 * np.pi)
DLOG_Q = -Y


def cubic(theta):
    """phi, phi', phi'', d_i phi, d_i phi' for phi = t1 y + t2 y^3."""
    t1, t2 = theta
    phi = t1 * Y + t2 * Y**3
    d1 = t1 + 3.0 * t2 * Y**2
    d2 = 6.0 * t2 * Y
    grad = np.stack([Y, Y**3])
    grad_prime = np.stack([np.ones_like(Y), 3.0 * Y**2])
    return phi, d1, d2, grad, grad_prime


def sine(theta):
    """phi = y + t1 sin(t2 y)."""
    t1, t2 = theta
    c, s = np.cos(t2 * Y), np.sin(t2 * Y)
    phi = Y + t1 * s
    d1 = 1.0 + t1 * t2 * c
    d2 = -t1 * t2**2 * s
    grad = np.stack([s, t1 * Y * c])
    grad_prime = np.stack([t2 * c, t1 * c - t1 * t2 * Y * s])
    return phi, d1, d2, grad, grad_prime


def scores(family, theta):
    """xi_i evaluated along x = phi_theta(y), plus phi and the density."""
    phi, d1, d2, grad, grad_prime = family(theta)
    if d1.min() <= 1e-3:
        return None
    ratio = grad / d1
    xi = grad_prime / d1 + ratio * (DLOG_Q - d2 / d1)
    return phi, d1, xi


def metric(family, theta):
    result = scores(family, theta)
    if result is None:
        return None
    _, _, xi = result
    weights = Q * DY
    G = np.einsum("iy,jy,y->ij", xi, xi, weights)
    return 0.5 * (G + G.T) + 1e-12 * np.eye(2)


def energy(interior, family, theta0, theta1):
    path = np.vstack([theta0, interior.reshape(-1, 2), theta1])
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        G = metric(family, 0.5 * (a + b))
        if G is None:
            return 1e6
        delta = b - a
        total += float(delta @ G @ delta)
    return total


def geodesic(family, theta0, theta1, bounds):
    theta0 = np.asarray(theta0, dtype=float)
    theta1 = np.asarray(theta1, dtype=float)
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
        args=(family, theta0, theta1),
        method="L-BFGS-B",
        bounds=bounds * N_SEG,
        options={"maxiter": 600, "maxfun": 200000, "ftol": 1e-12},
    )
    print(
        f"success={result.success}, nit={result.nit}, "
        f"energy={result.fun:.6g}, message={result.message}"
    )
    return np.vstack([theta0, result.x.reshape(-1, 2), theta1])


def log_density(family, theta, x):
    """log p_theta(x), obtained by inverting the monotone map phi_theta."""
    phi, d1, _, _, _ = family(theta)
    y = np.interp(x, phi, Y)
    log_q = -0.5 * y**2 - 0.5 * np.log(2.0 * np.pi)
    return log_q - np.log(np.interp(y, Y, d1))


def check_scores(family, theta, name):
    """Compare -xi_i with a finite difference of log p_theta at fixed x."""
    phi, _, xi = scores(family, theta)
    x = phi[np.abs(Y) < 2.5]
    reference = xi[:, np.abs(Y) < 2.5]
    step = 1e-5
    for i in range(2):
        shift = np.zeros(2)
        shift[i] = step
        numeric = (
            log_density(family, np.array(theta) + shift, x)
            - log_density(family, np.array(theta) - shift, x)
        ) / (2.0 * step)
        error = np.abs(numeric + reference[i]).max()
        print(f"  {name}: max |d_{i+1} log p + xi_{i+1}| = {error:.2e}")


CASES = [
    {
        "family": cubic,
        "title": (
            r"$q=\mathcal{N}(0,1),\quad"
            r"\phi_{\theta}(x)=\theta_{1}x+\theta_{2}x^{3}$"
        ),
        "theta0": (1.0, 0.03),
        "theta1": (0.35, 0.55),
        "bounds": [(0.05, 2.0), (0.005, 1.2)],
        "xlim": 4.5,
    },
    {
        "family": sine,
        "title": (
            r"$q=\mathcal{N}(0,1),\quad"
            r"\phi_{\theta}(x)=x+\theta_{1}\sin(\theta_{2}x)$"
        ),
        "theta0": (0.12, 0.9),
        "theta1": (0.6, 1.15),
        "bounds": [(0.01, 0.8), (0.2, 1.6)],
        "xlim": 4.5,
    },
]

paths = []
for case in CASES:
    print(case["title"])
    check_scores(case["family"], case["theta0"], "score")
    paths.append(
        geodesic(
            case["family"], case["theta0"], case["theta1"], case["bounds"]
        )
    )

fig = plt.figure(figsize=(10.6, 6.6))
outer = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.9], hspace=0.4, wspace=0.26)

for column, (case, path) in enumerate(zip(CASES, paths)):
    ax = fig.add_subplot(outer[0, column])
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
    ax.set_title(case["title"], fontsize=10)
    ax.grid(alpha=0.25)
    if column == 0:
        ax.legend(fontsize=7, frameon=False, loc="best")

    indices = np.linspace(0, len(path) - 1, N_SNAP, dtype=int)
    cells = outer[1, column].subgridspec(1, N_SNAP, wspace=0.12)
    curves = []
    for index in indices:
        phi, d1, _ = scores(case["family"], path[index])
        curves.append((phi, Q / d1))
    ymax = 1.08 * max(density.max() for _, density in curves)

    for position, (index, (x, density)) in enumerate(zip(indices, curves)):
        ax_snap = fig.add_subplot(cells[0, position])
        ax_snap.fill_between(x, density, color=START, alpha=0.30)
        ax_snap.plot(x, density, color=START, lw=1.3)
        ax_snap.set_xlim(-case["xlim"], case["xlim"])
        ax_snap.set_ylim(0.0, ymax)
        ax_snap.set_xticks([])
        ax_snap.set_yticks([])
        theta = path[index]
        ax_snap.set_title(
            rf"$({theta[0]:.2f},{theta[1]:.2f})$", fontsize=7
        )
        if position == 0:
            ax_snap.set_ylabel(r"$p_{\theta}(y)$", fontsize=8)

fig.suptitle(
    r"Fisher--Rao geodesics for two push-forward families",
    fontsize=12,
    y=0.98,
)
fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.06)

output = OUT / "22_pushforward_geodesics.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Wrote {output}")
# plt.show()
