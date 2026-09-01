"""
Eulerian velocity field induced by a parameter direction dtheta.

    phi_theta(x) = (x_1 + theta_1 tanh(x_2),  theta_2 x_2),
    dtheta = e_1 = (1, 0),
    partial_theta phi (theta, x) dtheta = (tanh(x_2), 0)
                                       = v_{dtheta}(phi_theta(x)).

One panel in the current transported space y = phi_theta(x):
  - light deformed grid at theta,
  - lighter dashed grid at theta + eps dtheta,
  - velocity arrows v_{dtheta}(y) computed from the corresponding initial x
    (no inverse map),
  - a few short connectors phi_theta(x) -> phi_{theta+eps dtheta}(x).

Notation matches the paper: x is the initial point, y = phi(theta, x) the
current transported position. The field lives on E, not on T_theta Theta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# --- Parameters (easy to retune) --------------------------------------------
THETA = np.array([0.8, 1.2])
DTHETA = np.array([1.0, 0.0])  # e_1
EPS = 0.15

# Initial Cartesian grid.
X_LIM = 2.4
N_LINES = 13
N_FINE = 400
ARROW_STEP = 2  # subsample of grid nodes for velocity arrows
CONNECT_STEP = 4  # even sparser subset for finite-eps connectors

VIEW = (-3.6, 3.6)
ARROW_SCALE = 4.8  # larger = shorter quiver shafts
ARROW_WIDTH = 0.0045

# Palette shared with the other scripts of the paper.
COLOR_GRID = "#4C78A8"
COLOR_PERT = "#9AA0A6"
COLOR_FIELD = "#C44E52"
COLOR_CONNECT = "#E07A35"
BOX_EDGE = "#1F4E79"


def phi(x1, x2, theta):
    """phi_theta(x) = (x_1 + theta_1 tanh(x_2), theta_2 x_2)."""
    t1, t2 = theta
    return x1 + t1 * np.tanh(x2), t2 * x2


def partial_theta_dtheta(x1, x2, dtheta):
    """
    (partial phi / partial theta)(theta, x) dtheta, computed from x.

    For this family:
        d/d theta_1  -> (tanh(x_2), 0)
        d/d theta_2  -> (0, x_2)
    """
    return dtheta[0] * np.tanh(x2), dtheta[1] * x2


# --- Initial grid -----------------------------------------------------------
nodes = np.linspace(-X_LIM, X_LIM, N_LINES)
fine = np.linspace(-X_LIM, X_LIM, N_FINE)
X1_nodes, X2_nodes = np.meshgrid(nodes, nodes)

# Arrow nodes (subsampled).
arrow_mask_i = np.arange(0, N_LINES, ARROW_STEP)
arrow_mask_j = np.arange(0, N_LINES, ARROW_STEP)
ax1 = X1_nodes[np.ix_(arrow_mask_i, arrow_mask_j)].ravel()
ax2 = X2_nodes[np.ix_(arrow_mask_i, arrow_mask_j)].ravel()

# Connector nodes (even sparser).
conn_mask_i = np.arange(0, N_LINES, CONNECT_STEP)
conn_mask_j = np.arange(0, N_LINES, CONNECT_STEP)
cx1 = X1_nodes[np.ix_(conn_mask_i, conn_mask_j)].ravel()
cx2 = X2_nodes[np.ix_(conn_mask_i, conn_mask_j)].ravel()

theta_eps = THETA + EPS * DTHETA

# --- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(7.6, 7.2))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[0.16, 1.0],
    left=0.10,
    right=0.96,
    top=0.96,
    bottom=0.08,
    hspace=0.12,
)
ax_banner = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[1, 0])

# Banner with the defining identity -----------------------------------------
ax_banner.set_xlim(0, 1)
ax_banner.set_ylim(0, 1)
ax_banner.axis("off")
ax_banner.add_patch(
    FancyBboxPatch(
        (0.06, 0.08),
        0.88,
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
    r"$\frac{\partial\phi}{\partial\theta}(\theta,x)\,d\theta"
    r"\;=\;"
    r"v_{d\theta}(\phi_{\theta}(x))$",
    ha="center",
    va="center",
    fontsize=13,
    color="#1F4E79",
    transform=ax_banner.transAxes,
    zorder=5,
)
ax_banner.text(
    0.50,
    0.22,
    r"a parameter direction induces a velocity field on the transported space",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
    style="italic",
    transform=ax_banner.transAxes,
    zorder=5,
)

# Current transported grid at theta -----------------------------------------
for value in nodes:
    y1, y2 = phi(fine, np.full_like(fine, value), THETA)
    ax.plot(y1, y2, color=COLOR_GRID, lw=0.85, alpha=0.55)
    y1, y2 = phi(np.full_like(fine, value), fine, THETA)
    ax.plot(y1, y2, color=COLOR_GRID, lw=0.85, alpha=0.55)

# Perturbed grid at theta + eps dtheta (dashed, lighter) --------------------
for value in nodes:
    y1, y2 = phi(fine, np.full_like(fine, value), theta_eps)
    ax.plot(y1, y2, color=COLOR_PERT, lw=0.7, alpha=0.45, ls="--")
    y1, y2 = phi(np.full_like(fine, value), fine, theta_eps)
    ax.plot(y1, y2, color=COLOR_PERT, lw=0.7, alpha=0.45, ls="--")

# Velocity field arrows, from initial x, plotted at y = phi_theta(x) --------
y1_arr, y2_arr = phi(ax1, ax2, THETA)
v1, v2 = partial_theta_dtheta(ax1, ax2, DTHETA)
ax.quiver(
    y1_arr,
    y2_arr,
    v1,
    v2,
    color=COLOR_FIELD,
    angles="xy",
    scale_units="xy",
    scale=ARROW_SCALE,
    width=ARROW_WIDTH,
    headwidth=3.2,
    headlength=4.0,
    headaxislength=3.4,
    zorder=4,
)

# Short connectors for a few material points --------------------------------
y1_c, y2_c = phi(cx1, cx2, THETA)
y1_e, y2_e = phi(cx1, cx2, theta_eps)
for i in range(len(cx1)):
    ax.annotate(
        "",
        xy=(y1_e[i], y2_e[i]),
        xytext=(y1_c[i], y2_c[i]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLOR_CONNECT,
            lw=1.0,
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
    )
    ax.plot(y1_c[i], y2_c[i], "o", color=COLOR_CONNECT, ms=3.2, zorder=6)

# Annotations ---------------------------------------------------------------
# Label one representative transported point near the positive x2 half.
ix = np.argmin((ax1 - 0.8) ** 2 + (ax2 - 1.2) ** 2)
yx, yy = float(y1_arr[ix]), float(y2_arr[ix])
ax.annotate(
    r"$y=\phi_{\theta}(x)$",
    xy=(yx, yy),
    xytext=(yx + 0.55, yy + 0.85),
    fontsize=10,
    color="#333333",
    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8),
)
ax.text(
    0.03,
    0.97,
    r"$d\theta=e_{1}=(1,0)$"
    "\n"
    rf"$\theta=({THETA[0]:g},{THETA[1]:g})$"
    "\n"
    rf"$\varepsilon={EPS:g}$",
    transform=ax.transAxes,
    fontsize=9,
    color="#333333",
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
)
ax.text(
    0.97,
    0.04,
    r"$v_{d\theta}$",
    transform=ax.transAxes,
    fontsize=12,
    color=COLOR_FIELD,
    ha="right",
    va="bottom",
    fontweight="bold",
)
ax.text(
    0.97,
    0.12,
    r"$\phi_{\theta}$",
    transform=ax.transAxes,
    fontsize=10,
    color=COLOR_GRID,
    ha="right",
    va="bottom",
)
ax.text(
    0.97,
    0.18,
    r"$\phi_{\theta+\varepsilon d\theta}$",
    transform=ax.transAxes,
    fontsize=10,
    color=COLOR_PERT,
    ha="right",
    va="bottom",
)

ax.set_xlim(*VIEW)
ax.set_ylim(*VIEW)
ax.set_aspect("equal")
ax.set_xlabel(r"$y_{1}$")
ax.set_ylabel(r"$y_{2}$")
ax.set_title(
    r"Eulerian velocity field $v_{d\theta}$ on the transported space",
    fontsize=11,
)
ax.grid(alpha=0.18)

png = OUT / "39_eulerian_velocity_field.png"
pdf = OUT / "39_eulerian_velocity_field.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
print(f"Wrote {png}")
print(f"Wrote {pdf}")
print(f"theta={THETA}, dtheta={DTHETA}, eps={EPS}")
print(
    "max |v| on arrow nodes =",
    float(np.max(np.hypot(v1, v2))),
)
