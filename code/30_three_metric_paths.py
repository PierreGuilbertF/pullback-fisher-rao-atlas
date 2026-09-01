"""
Three paths between two densities in square-root coordinates.

The positive orthant of the sphere of radius 2 represents probability
densities under p -> 2 sqrt(p). A black curve represents a one-dimensional
parametric subfamily. The three panels compare:

1. a path constrained to the parametric family;
2. the Fisher--Rao great-circle path on the sphere;
3. the straight chord in the ambient L2 space of square-root coordinates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

RADIUS = 2.0
N_SURFACE = 90
N_CURVE = 500
T0, T1 = 0.02, 0.98
ELEV, AZIM = 24.0, 42.0

GREY = "#B8B8B8"
BLACK = "#1A1A1A"
RED = "#C44E52"
BLUE = "#4C72B0"
GREEN = "#4C956C"


def sphere_point(alpha, beta):
    """Point in the positive orthant, using colatitude and azimuth."""
    return RADIUS * np.stack(
        [
            np.sin(alpha) * np.cos(beta),
            np.sin(alpha) * np.sin(beta),
            np.cos(alpha),
        ],
        axis=-1,
    )


def family(t):
    """A curved one-dimensional subfamily on the sphere."""
    t = np.asarray(t)
    alpha = 0.38 + 0.50 * t + 0.17 * np.sin(2.0 * np.pi * t + 0.25)
    beta = 0.18 + 1.15 * t + 0.08 * np.sin(2.0 * np.pi * t)
    return sphere_point(alpha, beta)


def spherical_geodesic(p0, p1, s):
    """Great-circle interpolation between two points of the same sphere."""
    u0 = p0 / RADIUS
    u1 = p1 / RADIUS
    omega = np.arccos(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if omega < 1e-12:
        return np.repeat(p0[None, :], len(s), axis=0)
    return RADIUS * (
        np.sin((1.0 - s) * omega)[:, None] * u0
        + np.sin(s * omega)[:, None] * u1
    ) / np.sin(omega)


alpha = np.linspace(0.0, np.pi / 2.0, N_SURFACE)
beta = np.linspace(0.0, np.pi / 2.0, N_SURFACE)
ALPHA, BETA = np.meshgrid(alpha, beta)
SURFACE = sphere_point(ALPHA, BETA)

t = np.linspace(0.0, 1.0, N_CURVE)
curve = family(t)
mask = (t >= T0) & (t <= T1)
constrained_path = curve[mask]

p0 = family(np.array(T0))
p1 = family(np.array(T1))
s = np.linspace(0.0, 1.0, 240)
fisher_rao_path = spherical_geodesic(p0, p1, s)
straight_path = (1.0 - s)[:, None] * p0 + s[:, None] * p1


def set_axes(ax, title):
    ax.plot_surface(
        SURFACE[..., 0],
        SURFACE[..., 1],
        SURFACE[..., 2],
        color=GREY,
        alpha=0.22,
        linewidth=0,
        antialiased=True,
        shade=True,
        zorder=0,
    )

    # Boundary arcs of the positive orthant.
    q = np.linspace(0.0, np.pi / 2.0, 180)
    boundaries = [
        np.column_stack(
            [RADIUS * np.sin(q), np.zeros_like(q), RADIUS * np.cos(q)]
        ),
        np.column_stack(
            [np.zeros_like(q), RADIUS * np.sin(q), RADIUS * np.cos(q)]
        ),
        np.column_stack(
            [RADIUS * np.cos(q), RADIUS * np.sin(q), np.zeros_like(q)]
        ),
    ]
    for boundary in boundaries:
        ax.plot(*boundary.T, color="#777777", lw=0.8, alpha=0.8, zorder=1)

    # Complete parametric family.
    ax.plot(*curve.T, color=BLACK, lw=2.0, zorder=4)
    ax.scatter(*p0, s=38, color="white", edgecolor=BLACK, linewidth=1.2, zorder=8)
    ax.scatter(*p1, s=38, color="white", edgecolor=BLACK, linewidth=1.2, zorder=8)
    ax.text(*(p0 + np.array([-0.08, -0.07, 0.10])), r"$p_{0}$", fontsize=9)
    ax.text(*(p1 + np.array([0.02, 0.03, 0.10])), r"$p_{1}$", fontsize=9)

    ax.set_title(title, fontsize=10.5, pad=5)
    ax.set_xlim(0.0, 2.05)
    ax.set_ylim(0.0, 2.05)
    ax.set_zlim(0.0, 2.05)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r"$2\sqrt{p_{1}}$", fontsize=8, labelpad=-5)
    ax.set_ylabel(r"$2\sqrt{p_{2}}$", fontsize=8, labelpad=-5)
    ax.set_zlabel(r"$2\sqrt{p_{3}}$", fontsize=8, labelpad=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


fig = plt.figure(figsize=(12.3, 4.4))

ax0 = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
set_axes(ax0, "Path within the parametric family")
ax0.plot(*constrained_path.T, color=RED, lw=4.0, zorder=6)
ax0.legend(
    handles=[Line2D([0], [0], color=RED, lw=3, label="constrained path")],
    loc="lower center",
    frameon=False,
    fontsize=8,
)

ax1 = fig.add_subplot(1, 3, 2, projection="3d", computed_zorder=False)
set_axes(ax1, "Fisher--Rao path on the sphere")
ax1.plot(*fisher_rao_path.T, color=BLUE, lw=4.0, zorder=6)
ax1.legend(
    handles=[Line2D([0], [0], color=BLUE, lw=3, label="great-circle arc")],
    loc="lower center",
    frameon=False,
    fontsize=8,
)

ax2 = fig.add_subplot(1, 3, 3, projection="3d", computed_zorder=False)
set_axes(ax2, r"Straight path in ambient $L^{2}$")
ax2.plot(*straight_path.T, color=GREEN, lw=4.0, zorder=7)
ax2.legend(
    handles=[Line2D([0], [0], color=GREEN, lw=3, label="straight chord")],
    loc="lower center",
    frameon=False,
    fontsize=8,
)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.03)
path = OUT / "30_three_metric_paths.png"
fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote {path}")
