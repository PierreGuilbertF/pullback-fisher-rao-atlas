"""
Divergence as infinitesimal local rate of volume change.

Three panels, same initial square cell, same visual conventions:
  left    v(y) = (a y_1, a y_2)       =>  div v = 2a > 0   (expansion)
  middle  v(y) = (a y_2, 0)           =>  div v = 0        (shear, area preserved)
  right   v(y) = (-a y_1, -a y_2)     =>  div v = -2a < 0  (compression)

In each panel the solid square is the material cell A, the dashed
parallelogram is its first-order image A_eps under y |-> y + eps v(y),
and the arrows are the velocity field sampled at the cell.

The message: div v measures volume change, not shape change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# --- Parameters -------------------------------------------------------------
A = 0.55          # strength of the velocity field
EPS = 0.28        # infinitesimal step (small enough to look first-order)
HALF = 0.55       # half-side of the initial square cell
CENTER = np.array([0.0, 0.0])

# Quiver sampling around / on the cell.
ARROW_HALF = 0.95
ARROW_N = 5

VIEW = (-1.55, 1.55)

# Palette shared with the other scripts.
COLOR_CELL = "#4C78A8"
COLOR_DEFORMED = "#C44E52"
COLOR_ARROW = "#9AA0A6"
COLOR_POS = "#2A9D8F"
COLOR_ZERO = "#4C78A8"
COLOR_NEG = "#C44E52"
BOX_EDGE = "#1F4E79"


def v_expand(y):
    return A * y


def v_shear(y):
    out = np.zeros_like(y)
    out[..., 0] = A * y[..., 1]
    return out


def v_contract(y):
    return -A * y


def square_corners(center, half):
    """Closed polygon of the axis-aligned square (last = first)."""
    c = np.asarray(center, dtype=float)
    pts = np.array(
        [
            [c[0] - half, c[1] - half],
            [c[0] + half, c[1] - half],
            [c[0] + half, c[1] + half],
            [c[0] - half, c[1] + half],
            [c[0] - half, c[1] - half],
        ]
    )
    return pts


def deform(points, field, eps=EPS):
    return points + eps * field(points)


def draw_panel(ax, field, div_value, title, subtitle, accent):
    cell = square_corners(CENTER, HALF)
    cell_eps = deform(cell, field)

    # Background velocity arrows on a small local grid.
    g = np.linspace(-ARROW_HALF, ARROW_HALF, ARROW_N)
    G1, G2 = np.meshgrid(g, g)
    pts = np.stack([G1.ravel(), G2.ravel()], axis=-1)
    vel = field(pts)
    # Drop the exact centre arrow (zero for expand/contract).
    mask = np.hypot(pts[:, 0], pts[:, 1]) > 1e-8
    ax.quiver(
        pts[mask, 0],
        pts[mask, 1],
        vel[mask, 0],
        vel[mask, 1],
        color=COLOR_ARROW,
        angles="xy",
        scale_units="xy",
        scale=2.4,
        width=0.006,
        headwidth=3.0,
        headlength=3.6,
        headaxislength=3.2,
        alpha=0.75,
        zorder=1,
    )

    # Original cell.
    ax.add_patch(
        Polygon(
            cell[:-1],
            closed=True,
            facecolor=COLOR_CELL,
            edgecolor=COLOR_CELL,
            alpha=0.18,
            lw=1.8,
            zorder=2,
        )
    )
    ax.plot(cell[:, 0], cell[:, 1], color=COLOR_CELL, lw=1.9, zorder=3)

    # Deformed cell.
    ax.add_patch(
        Polygon(
            cell_eps[:-1],
            closed=True,
            facecolor=COLOR_DEFORMED,
            edgecolor=COLOR_DEFORMED,
            alpha=0.10,
            lw=1.6,
            linestyle="--",
            zorder=2,
        )
    )
    ax.plot(
        cell_eps[:, 0],
        cell_eps[:, 1],
        color=COLOR_DEFORMED,
        lw=1.7,
        ls="--",
        zorder=3,
    )

    # Corner displacement connectors (material points).
    for i in range(4):
        ax.annotate(
            "",
            xy=cell_eps[i],
            xytext=cell[i],
            arrowprops=dict(
                arrowstyle="-|>",
                color=accent,
                lw=1.05,
                mutation_scale=8,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )
        ax.plot(*cell[i], "o", color=COLOR_CELL, ms=3.5, zorder=5)

    # Analytical area check (first order).
    area0 = (2.0 * HALF) ** 2
    # Shoelace on the open polygon.
    pe = cell_eps[:-1]
    area_eps = 0.5 * np.abs(
        np.dot(pe[:, 0], np.roll(pe[:, 1], -1))
        - np.dot(pe[:, 1], np.roll(pe[:, 0], -1))
    )
    predicted = area0 * (1.0 + EPS * div_value)

    ax.text(
        0.03,
        0.97,
        rf"$\mathrm{{div}}\,v={div_value:g}$",
        transform=ax.transAxes,
        fontsize=11,
        color=accent,
        va="top",
        ha="left",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.88,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="top",
        ha="left",
        style="italic",
    )
    ax.text(
        0.97,
        0.04,
        rf"$\mathrm{{Vol}}(A_{{\varepsilon}})/\mathrm{{Vol}}(A)"
        rf"\approx{area_eps / area0:.3f}$"
        "\n"
        rf"$1+\varepsilon\,\mathrm{{div}}\,v={predicted / area0:.3f}$",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="right",
    )

    ax.set_xlim(*VIEW)
    ax.set_ylim(*VIEW)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y_{1}$")
    ax.set_ylabel(r"$y_{2}$")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.20)
    return area0, area_eps, predicted


# --- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(12.8, 5.0))
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[0.20, 1.0],
    left=0.05,
    right=0.98,
    top=0.94,
    bottom=0.10,
    wspace=0.28,
    hspace=0.18,
)
ax_banner = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])

# Banner --------------------------------------------------------------------
ax_banner.set_xlim(0, 1)
ax_banner.set_ylim(0, 1)
ax_banner.axis("off")
ax_banner.add_patch(
    FancyBboxPatch(
        (0.12, 0.08),
        0.76,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.15,
        edgecolor=BOX_EDGE,
        facecolor="#F4F7FB",
        transform=ax_banner.transAxes,
        clip_on=False,
    )
)
ax_banner.text(
    0.50,
    0.64,
    r"$\frac{d}{ds}\log(\mathrm{Vol})"
    r"\;=\;"
    r"\mathrm{div}\,v$",
    ha="center",
    va="center",
    fontsize=14,
    color="#1F4E79",
    transform=ax_banner.transAxes,
    zorder=5,
)
ax_banner.text(
    0.50,
    0.22,
    r"divergence $=$ infinitesimal local rate of volume change",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
    style="italic",
    transform=ax_banner.transAxes,
    zorder=5,
)

draw_panel(
    ax0,
    v_expand,
    div_value=2.0 * A,
    title=rf"$v(y)=({A:g}\,y_{{1}},\,{A:g}\,y_{{2}})$",
    subtitle="local expansion",
    accent=COLOR_POS,
)
draw_panel(
    ax1,
    v_shear,
    div_value=0.0,
    title=rf"$v(y)=({A:g}\,y_{{2}},\,0)$",
    subtitle="shape changes, volume preserved",
    accent=COLOR_ZERO,
)
draw_panel(
    ax2,
    v_contract,
    div_value=-2.0 * A,
    title=rf"$v(y)=(-{A:g}\,y_{{1}},\,-{A:g}\,y_{{2}})$",
    subtitle="local compression",
    accent=COLOR_NEG,
)

# Shared small legend under the middle panel.
ax1.text(
    0.50,
    -0.20,
    r"solid: material cell $A$"
    r"$\qquad$"
    r"dashed: $A_{\varepsilon}=\{y+\varepsilon v(y)\}$"
    rf"$\qquad\varepsilon={EPS:g}$",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=9,
    color="#444444",
)

png = OUT / "40_divergence_volume_change.png"
pdf = OUT / "40_divergence_volume_change.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
print(f"Wrote {png}")
print(f"Wrote {pdf}")
print(f"a={A}, eps={EPS}, half={HALF}")
