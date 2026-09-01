"""
Conservation of probability forces density to change opposite to divergence.

Illustration of the differentiated transport identity:

    partial_theta log p (theta, y) dtheta
        + <grad log p_theta(y), v_{dtheta}(y)>
        = - div v_{dtheta}(y),

hence
    dp_theta(dtheta) = - div(p_theta v_{dtheta}).

Three panels, same particle count (= same probability mass):
  left    current cell at theta, with local shading of p_theta and field arrows
  middle  local expansion  (div v > 0): larger cell, lighter density
  right   local compression (div v < 0): smaller cell, darker density

No notation beyond p_theta, v_{dtheta}, theta(s) = theta + s dtheta, and div.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# --- Parameters -------------------------------------------------------------
HALF0 = 0.55          # half-side of the reference cell at theta
EPS = 0.32            # small deformation amplitude
A_EXP = 0.55          # expansion / compression strength
N_PART = 64           # particles = conserved probability mass
RNG = np.random.default_rng(11)

VIEW = (-1.55, 1.55)
SHADE_LIM = 1.35

# Palette shared with the other scripts.
COLOR_CELL = "#4C78A8"
COLOR_EXPAND = "#2A9D8F"
COLOR_COMPRESS = "#C44E52"
COLOR_ARROW = "#6B7280"
COLOR_PARTICLE = "#1F4E79"
BOX_EDGE = "#1F4E79"

# Density colormap: light -> dark blue (higher p_theta darker).
CMAP = LinearSegmentedColormap.from_list(
    "p_theta",
    ["#F7FAFC", "#D6E4F0", "#8FB3D1", "#4C78A8", "#1F4E79"],
)


def square(half, center=(0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    return np.array(
        [
            [c[0] - half, c[1] - half],
            [c[0] + half, c[1] - half],
            [c[0] + half, c[1] + half],
            [c[0] - half, c[1] + half],
            [c[0] - half, c[1] - half],
        ]
    )


def sample_in_square(n, half, rng):
    return rng.uniform(-half, half, size=(n, 2))


def expand_map(points, half0=HALF0, eps=EPS, a=A_EXP):
    """Radial expansion y |-> y + eps * a * y, relative to the origin."""
    return points * (1.0 + eps * a)


def compress_map(points, half0=HALF0, eps=EPS, a=A_EXP):
    return points * (1.0 - eps * a)


def v_expand(y):
    return A_EXP * y


def v_compress(y):
    return -A_EXP * y


def density_level(half, half0=HALF0):
    """
    Relative density under conserved mass: mass ~ area * density constant,
    so density scales as (half0 / half)^2.
    """
    return (half0 / half) ** 2


def draw_density_background(ax, level, vmax=2.2):
    """Uniform local shading representing p_theta inside the viewing window."""
    g = np.linspace(-SHADE_LIM, SHADE_LIM, 80)
    G1, G2 = np.meshgrid(g, g)
    # Soft radial falloff so the shading reads as a local density bump,
    # scaled by the conserved-mass level.
    bump = np.exp(-0.55 * (G1**2 + G2**2))
    Z = level * bump
    ax.imshow(
        Z,
        extent=(-SHADE_LIM, SHADE_LIM, -SHADE_LIM, SHADE_LIM),
        origin="lower",
        cmap=CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        alpha=0.95,
        zorder=0,
        aspect="equal",
    )


def draw_arrows(ax, field, color, scale=2.3):
    g = np.linspace(-1.05, 1.05, 5)
    G1, G2 = np.meshgrid(g, g)
    pts = np.stack([G1.ravel(), G2.ravel()], axis=-1)
    vel = field(pts)
    mask = np.hypot(pts[:, 0], pts[:, 1]) > 0.15
    ax.quiver(
        pts[mask, 0],
        pts[mask, 1],
        vel[mask, 0],
        vel[mask, 1],
        color=color,
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.006,
        headwidth=3.0,
        headlength=3.6,
        headaxislength=3.2,
        alpha=0.85,
        zorder=2,
    )


def draw_cell(ax, corners, edge, face_alpha=0.10):
    ax.add_patch(
        Polygon(
            corners[:-1],
            closed=True,
            facecolor=edge,
            edgecolor=edge,
            alpha=face_alpha,
            lw=1.8,
            zorder=3,
        )
    )
    ax.plot(corners[:, 0], corners[:, 1], color=edge, lw=1.9, zorder=4)


def draw_particles(ax, pts):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=18,
        c=COLOR_PARTICLE,
        alpha=0.9,
        linewidths=0,
        zorder=5,
    )


def style(ax, title):
    ax.set_xlim(*VIEW)
    ax.set_ylim(*VIEW)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y_{1}$")
    ax.set_ylabel(r"$y_{2}$")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


# --- Shared particle set at theta ------------------------------------------
particles0 = sample_in_square(N_PART, HALF0 * 0.92, RNG)
cell0 = square(HALF0)
half_exp = HALF0 * (1.0 + EPS * A_EXP)
half_cmp = HALF0 * (1.0 - EPS * A_EXP)
particles_exp = expand_map(particles0)
particles_cmp = compress_map(particles0)
cell_exp = square(half_exp)
cell_cmp = square(half_cmp)

level0 = density_level(HALF0)
level_exp = density_level(half_exp)
level_cmp = density_level(half_cmp)

# --- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(12.8, 5.8))
gs = fig.add_gridspec(
    3,
    3,
    height_ratios=[1.0, 0.18, 0.22],
    left=0.05,
    right=0.98,
    top=0.92,
    bottom=0.06,
    wspace=0.28,
    hspace=0.35,
)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax_id = fig.add_subplot(gs[1, :])
ax_box = fig.add_subplot(gs[2, :])

# Panel 1: current configuration at theta -----------------------------------
draw_density_background(ax0, level0)
draw_arrows(ax0, v_expand, COLOR_ARROW, scale=2.6)
draw_cell(ax0, cell0, COLOR_CELL)
draw_particles(ax0, particles0)
ax0.text(
    0.03,
    0.97,
    r"at $\theta$",
    transform=ax0.transAxes,
    fontsize=10,
    color="#333333",
    va="top",
)
ax0.text(
    0.03,
    0.88,
    r"$v_{d\theta}$",
    transform=ax0.transAxes,
    fontsize=10,
    color=COLOR_ARROW,
    va="top",
)
style(ax0, r"Current: $p_{\theta}$, region and $v_{d\theta}$")

# Panel 2: local expansion --------------------------------------------------
draw_density_background(ax1, level_exp)
draw_arrows(ax1, v_expand, COLOR_EXPAND, scale=2.3)
draw_cell(ax1, cell_exp, COLOR_EXPAND)
draw_particles(ax1, particles_exp)
ax1.text(
    0.03,
    0.97,
    r"$\mathrm{div}\,v_{d\theta}>0$",
    transform=ax1.transAxes,
    fontsize=11,
    color=COLOR_EXPAND,
    va="top",
    fontweight="bold",
)
ax1.text(
    0.03,
    0.88,
    "density decreases",
    transform=ax1.transAxes,
    fontsize=9,
    color="#444444",
    va="top",
    style="italic",
)
style(ax1, r"Local expansion")

# Panel 3: local compression ------------------------------------------------
draw_density_background(ax2, level_cmp)
draw_arrows(ax2, v_compress, COLOR_COMPRESS, scale=2.3)
draw_cell(ax2, cell_cmp, COLOR_COMPRESS)
draw_particles(ax2, particles_cmp)
ax2.text(
    0.03,
    0.97,
    r"$\mathrm{div}\,v_{d\theta}<0$",
    transform=ax2.transAxes,
    fontsize=11,
    color=COLOR_COMPRESS,
    va="top",
    fontweight="bold",
)
ax2.text(
    0.03,
    0.88,
    "density increases",
    transform=ax2.transAxes,
    fontsize=9,
    color="#444444",
    va="top",
    style="italic",
)
style(ax2, r"Local compression")

# Identity under the panels -------------------------------------------------
ax_id.axis("off")
ax_id.text(
    0.50,
    0.55,
    r"$\frac{\partial \log p}{\partial \theta}(\theta,y)\,d\theta"
    r"\;+\;"
    r"\langle\nabla\log p_{\theta}(y),\, v_{d\theta}(y)\rangle"
    r"\;=\;"
    r"-\,\mathrm{div}\,v_{d\theta}(y)$",
    ha="center",
    va="center",
    fontsize=12,
    color="#1F4E79",
    transform=ax_id.transAxes,
)
ax_id.text(
    0.50,
    0.05,
    r"same probability mass $+$ changing local volume $=$ changing density",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
    style="italic",
    transform=ax_id.transAxes,
)

# Boxed continuity equation -------------------------------------------------
ax_box.axis("off")
ax_box.add_patch(
    FancyBboxPatch(
        (0.18, 0.15),
        0.64,
        0.70,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=BOX_EDGE,
        facecolor="#F4F7FB",
        transform=ax_box.transAxes,
        clip_on=False,
    )
)
ax_box.text(
    0.50,
    0.50,
    r"$dp_{\theta}(d\theta)"
    r"\;=\;"
    r"-\,\mathrm{div}(p_{\theta}\, v_{d\theta})$",
    ha="center",
    va="center",
    fontsize=14,
    color="#1F4E79",
    transform=ax_box.transAxes,
    zorder=5,
)

assert len(particles0) == len(particles_exp) == len(particles_cmp) == N_PART

png = OUT / "41_density_from_divergence.png"
pdf = OUT / "41_density_from_divergence.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
print(f"Wrote {png}")
print(f"Wrote {pdf}")
print(f"N_PART={N_PART}, half0={HALF0}, half_exp={half_exp:.3f}, half_cmp={half_cmp:.3f}")
print(f"levels: {level0:.3f}, {level_exp:.3f}, {level_cmp:.3f}")
