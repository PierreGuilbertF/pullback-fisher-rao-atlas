"""
Hessian metric vs a generic Riemannian metric, in a 2D parameter plane.

A Hessian metric is g = Hess ψ for a convex scalar ψ. Then ∂_k g_ij is
totally symmetric (Schwarz). A generic metric can twist: the unit balls
rotate as one moves, in a way no potential can produce.

Left:  g = Hess ψ, ψ = cosh(θ1)+cosh(θ2)+0.3 θ1 θ2, with level sets of ψ.
Right: the same ellipses for a metric whose principal axes rotate with θ2.
The background is the obstruction |∂_2 g_11 - ∂_1 g_12|, zero iff Hessian.

Not included in the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

LIM = 1.35
ELLIPSE_SCALE = 0.16
GRID = np.linspace(-1.05, 1.05, 6)
COUPLING = 0.30
TWIST = 0.70
COLOR_ELL = "#1b3a5c"


def psi(x, y):
    return np.cosh(x) + np.cosh(y) + COUPLING * x * y


def g_hess(x, y):
    return np.array([[np.cosh(x), COUPLING], [COUPLING, np.cosh(y)]])


def g_generic(x, y):
    """Axis-aligned stretch, then a rotation that depends only on y."""
    angle = TWIST * y
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    diag = np.diag([1.7, 0.50])
    return rot @ diag @ rot.T


def obstruction(g_fn, x, y, h=1e-4):
    """|∂_y g_xx - ∂_x g_xy|: vanishes identically for a Hessian metric."""
    g = g_fn(x, y)
    g_dy = g_fn(x, y + h)
    g_dx = g_fn(x + h, y)
    d_y_gxx = (g_dy[0, 0] - g[0, 0]) / h
    d_x_gxy = (g_dx[0, 1] - g[0, 1]) / h
    return d_y_gxx - d_x_gxy


def ellipse_points(origin, metric, scale=ELLIPSE_SCALE, n=90):
    evals, evecs = np.linalg.eigh(metric)
    evals = np.maximum(evals, 1e-8)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    shape = evecs @ np.diag(scale / np.sqrt(evals)) @ evecs.T
    return origin + (shape @ circle).T


def draw_ellipses(ax, g_fn):
    for x0 in GRID:
        for y0 in GRID:
            pts = ellipse_points(np.array([x0, y0]), g_fn(x0, y0))
            ax.fill(pts[:, 0], pts[:, 1], color=COLOR_ELL, alpha=0.18, zorder=3)
            ax.plot(pts[:, 0], pts[:, 1], color=COLOR_ELL, lw=1.05, zorder=4)


def draw_frame(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$\theta_{1}$")
    ax.set_ylabel(r"$\theta_{2}$")
    ax.axhline(0.0, color="0.75", lw=0.5, zorder=1)
    ax.axvline(0.0, color="0.75", lw=0.5, zorder=1)


xs = np.linspace(-LIM, LIM, 220)
ys = np.linspace(-LIM, LIM, 220)
X, Y = np.meshgrid(xs, ys)

obs_h = np.vectorize(lambda x, y: obstruction(g_hess, x, y))(X, Y)
obs_g = np.vectorize(lambda x, y: obstruction(g_generic, x, y))(X, Y)
print(
    f"obstruction max |Hess|={np.max(np.abs(obs_h)):.2e}  "
    f"|generic|={np.max(np.abs(obs_g)):.3f}"
)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)

# --- Hessian ---
ax = axes[0]
psi_grid = psi(X, Y)
levels = np.linspace(float(psi_grid.min()), float(psi_grid.max()), 12)
cf = ax.contourf(X, Y, psi_grid, levels=levels, cmap="YlGnBu", alpha=0.85, zorder=0)
ax.contour(X, Y, psi_grid, levels=levels, colors="0.35", linewidths=0.45, zorder=1)
draw_ellipses(ax, g_hess)
draw_frame(ax)
ax.set_title(
    r"Hessian metric $g=\mathrm{Hess}\,\psi$"
    "\n"
    r"$\psi=\cosh\theta_{1}+\cosh\theta_{2}+0{,}3\,\theta_{1}\theta_{2}$",
    fontsize=10,
)
cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label(r"$\psi(\theta)$", fontsize=9)

# --- generic ---
ax = axes[1]
vmax = float(np.max(np.abs(obs_g)))
cf2 = ax.contourf(
    X, Y, obs_g, levels=24, cmap="coolwarm", vmin=-vmax, vmax=vmax, zorder=0
)
draw_ellipses(ax, g_generic)
draw_frame(ax)
ax.set_title(
    r"Generic metric (axes rotating with $\theta_{2}$)"
    "\n"
    r"background: $\partial_{2}g_{11}-\partial_{1}g_{12}$  (vanishes iff Hessian)",
    fontsize=10,
)
cbar2 = fig.colorbar(cf2, ax=ax, pad=0.02, fraction=0.046)
cbar2.set_label(r"$\partial_{2}g_{11}-\partial_{1}g_{12}$", fontsize=9)

path = OUT / "27_hessian_vs_generic.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
