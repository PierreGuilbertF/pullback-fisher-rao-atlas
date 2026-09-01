"""
Conservation of probability under a push-forward by a diffeomorphism.

    q = N(0, I_2) on E = R^2,
    phi_theta(x) = (x_1 + a tanh(x_2), b x_2),
    p_theta = (phi_theta)_# q,
    A_theta = phi_theta(A).

The identity illustrated is
    int_A q(x) dx = int_{A_theta} p_theta(y) dy.

Three panels:
  left   initial configuration: density q, region A, particles (in/out of A)
  middle transport: selected particles from A mapped x |-> y = phi_theta(x)
  right  transported configuration: density p_theta, region A_theta,
         the same particles after transport

Notation matches the paper: x is read by q, y = phi(theta, x) is read by
p_theta. The Jacobian determinant is constantly equal to b.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# --- Transport parameters ---------------------------------------------------
A_SHEAR = 0.8
B_SCALE = 1.3

# --- Region A: ellipse away from the origin ---------------------------------
A_CENTER = np.array([1.35, 0.85])
A_RADII = np.array([0.55, 0.38])  # (semi-axis along e1, along e2)
N_BOUNDARY = 360

# --- Particles --------------------------------------------------------------
RNG = np.random.default_rng(7)
N_BACKGROUND = 280
N_INSIDE = 55
VIEW = (-3.2, 3.2)

# --- Density rendering ------------------------------------------------------
GRID = np.linspace(-3.4, 3.4, 320)
LEVELS_Q = 6
LEVELS_P = 6

# --- Palette (shared with the other scripts) --------------------------------
BASE = "#9AA0A6"
DENSITY = "#4C78A8"
REGION = "#C44E52"
PARTICLE_IN = "#C44E52"
PARTICLE_OUT = "#9AA0A6"
ARROW = "#4C78A8"
BOX_EDGE = "#1F4E79"


def phi(x1, x2):
    """phi_theta(x) = (x_1 + a tanh(x_2), b x_2)."""
    return x1 + A_SHEAR * np.tanh(x2), B_SCALE * x2


def phi_inv(y1, y2):
    """Analytical inverse: x_2 = y_2/b, x_1 = y_1 - a tanh(y_2/b)."""
    x2 = y2 / B_SCALE
    x1 = y1 - A_SHEAR * np.tanh(x2)
    return x1, x2


def q_density(x1, x2):
    """Standard Gaussian density on R^2."""
    return np.exp(-0.5 * (x1**2 + x2**2)) / (2.0 * np.pi)


def p_density(y1, y2):
    """Exact push-forward density: q(phi^{-1}(y)) / |det D phi| = q(psi(y)) / b."""
    x1, x2 = phi_inv(y1, y2)
    return q_density(x1, x2) / B_SCALE


def ellipse_boundary(center, radii, n=N_BOUNDARY):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack(
        [center[0] + radii[0] * np.cos(t), center[1] + radii[1] * np.sin(t)]
    )


def inside_ellipse(points, center, radii):
    d = (points - center) / radii
    return np.sum(d**2, axis=1) <= 1.0


def sample_in_ellipse(n, center, radii, rng):
    """Rejection sampling of a uniform cloud inside the ellipse (for display)."""
    out = []
    while len(out) < n:
        cand = center + radii * rng.uniform(-1.0, 1.0, size=(4 * n, 2))
        mask = inside_ellipse(cand, center, radii)
        out.extend(cand[mask])
    return np.asarray(out[:n])


def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlim(*VIEW)
    ax.set_ylim(*VIEW)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22)


# --- Geometry ---------------------------------------------------------------
boundary_x = ellipse_boundary(A_CENTER, A_RADII)
boundary_y = np.column_stack(phi(boundary_x[:, 0], boundary_x[:, 1]))

background = RNG.normal(size=(N_BACKGROUND, 2))
outside_mask = ~inside_ellipse(background, A_CENTER, A_RADII)
particles_out = background[outside_mask]
particles_in = sample_in_ellipse(N_INSIDE, A_CENTER, A_RADII, RNG)
particles_in_y = np.column_stack(phi(particles_in[:, 0], particles_in[:, 1]))

# Approximate probability mass of A under q (for the annotation).
t_mc = RNG.uniform(0.0, 2.0 * np.pi, size=20000)
r_mc = np.sqrt(RNG.uniform(0.0, 1.0, size=20000))
pts_mc = A_CENTER + A_RADII * np.column_stack(
    [r_mc * np.cos(t_mc), r_mc * np.sin(t_mc)]
)
area_A = np.pi * A_RADII[0] * A_RADII[1]
mass_A = area_A * np.mean(q_density(pts_mc[:, 0], pts_mc[:, 1]))
mass_str = f"{mass_A:.3f}"

X1, X2 = np.meshgrid(GRID, GRID)
Q = q_density(X1, X2)
Y1, Y2 = np.meshgrid(GRID, GRID)
P = p_density(Y1, Y2)

arrow_idx = np.linspace(0, N_INSIDE - 1, 14, dtype=int)
label_y = np.mean(boundary_y, axis=0)

# --- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 5.4))
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[0.22, 1.0],
    left=0.05,
    right=0.98,
    top=0.96,
    bottom=0.10,
    wspace=0.28,
    hspace=0.18,
)
ax_banner = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])

# Banner with the conservation identity -------------------------------------
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
    0.62,
    r"$\int_{A} q(x)\,dx"
    r"\;=\;"
    r"\int_{\phi_{\theta}(A)} p_{\theta}(y)\,dy$",
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
    r"same probability mass, deformed region",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
    style="italic",
    transform=ax_banner.transAxes,
    zorder=5,
)

# Panel 1: initial configuration --------------------------------------------
ax0.contour(X1, X2, Q, levels=LEVELS_Q, colors=DENSITY, linewidths=0.7, alpha=0.55)
ax0.scatter(
    particles_out[:, 0],
    particles_out[:, 1],
    s=6,
    c=PARTICLE_OUT,
    alpha=0.45,
    linewidths=0,
    zorder=2,
)
ax0.scatter(
    particles_in[:, 0],
    particles_in[:, 1],
    s=14,
    c=PARTICLE_IN,
    alpha=0.9,
    linewidths=0,
    zorder=3,
)
ax0.plot(boundary_x[:, 0], boundary_x[:, 1], color=REGION, lw=1.8, zorder=4)
ax0.fill(boundary_x[:, 0], boundary_x[:, 1], color=REGION, alpha=0.12, zorder=1)
ax0.text(
    A_CENTER[0] + 0.05,
    A_CENTER[1] + A_RADII[1] + 0.22,
    r"$A$",
    color=REGION,
    fontsize=12,
    ha="center",
    va="bottom",
)
ax0.text(
    0.04,
    0.04,
    rf"$\mathbb{{P}}_{{q}}(A)\approx {mass_str}$",
    transform=ax0.transAxes,
    fontsize=9,
    color="#333333",
    va="bottom",
)
style_axes(ax0, r"$x_{1}$", r"$x_{2}$", r"Initial: density $q$, region $A$")

# Panel 2: transport --------------------------------------------------------
ax1.plot(boundary_x[:, 0], boundary_x[:, 1], color=REGION, lw=1.2, alpha=0.45)
ax1.plot(boundary_y[:, 0], boundary_y[:, 1], color=REGION, lw=1.2, alpha=0.85)
ax1.scatter(
    particles_in[:, 0],
    particles_in[:, 1],
    s=12,
    c=PARTICLE_IN,
    alpha=0.55,
    linewidths=0,
    zorder=3,
)
ax1.scatter(
    particles_in_y[:, 0],
    particles_in_y[:, 1],
    s=12,
    c=PARTICLE_IN,
    alpha=0.95,
    linewidths=0,
    zorder=3,
)
for i in arrow_idx:
    ax1.annotate(
        "",
        xy=particles_in_y[i],
        xytext=particles_in[i],
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW,
            lw=0.9,
            mutation_scale=8,
            shrinkA=1,
            shrinkB=1,
        ),
        zorder=2,
    )
ax1.text(
    A_CENTER[0] - 0.15,
    A_CENTER[1] + A_RADII[1] + 0.18,
    r"$x$",
    color=REGION,
    fontsize=11,
    ha="center",
)
ax1.text(
    label_y[0] + 0.15,
    label_y[1] + 0.55,
    r"$y=\phi_{\theta}(x)$",
    color=REGION,
    fontsize=11,
    ha="center",
)
style_axes(ax1, r"$x_{1},\; y_{1}$", r"$x_{2},\; y_{2}$", r"Transport $\phi_{\theta}$")
ax1.text(
    0.50,
    -0.18,
    rf"$\phi_{{\theta}}(x)=(x_{1}+{A_SHEAR:g}\tanh x_{2},\;{B_SCALE:g}\,x_{2})$",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=9,
    color="#444444",
)

# Panel 3: transported configuration ----------------------------------------
ax2.contour(Y1, Y2, P, levels=LEVELS_P, colors=DENSITY, linewidths=0.7, alpha=0.55)
out_y = np.column_stack(phi(particles_out[:, 0], particles_out[:, 1]))
ax2.scatter(
    out_y[:, 0],
    out_y[:, 1],
    s=6,
    c=PARTICLE_OUT,
    alpha=0.45,
    linewidths=0,
    zorder=2,
)
ax2.scatter(
    particles_in_y[:, 0],
    particles_in_y[:, 1],
    s=14,
    c=PARTICLE_IN,
    alpha=0.9,
    linewidths=0,
    zorder=3,
)
ax2.plot(boundary_y[:, 0], boundary_y[:, 1], color=REGION, lw=1.8, zorder=4)
ax2.fill(boundary_y[:, 0], boundary_y[:, 1], color=REGION, alpha=0.12, zorder=1)
ax2.text(
    label_y[0],
    np.max(boundary_y[:, 1]) + 0.22,
    r"$A_{\theta}=\phi_{\theta}(A)$",
    color=REGION,
    fontsize=11,
    ha="center",
    va="bottom",
)
ax2.text(
    0.04,
    0.04,
    rf"$\mathbb{{P}}_{{p_{{\theta}}}}(A_{{\theta}})\approx {mass_str}$",
    transform=ax2.transAxes,
    fontsize=9,
    color="#333333",
    va="bottom",
)
style_axes(
    ax2,
    r"$y_{1}$",
    r"$y_{2}$",
    r"Transported: density $p_{\theta}$, region $A_{\theta}$",
)

png = OUT / "38_conservation_of_probability.png"
pdf = OUT / "38_conservation_of_probability.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
print(f"Wrote {png}")
print(f"Wrote {pdf}")
print(f"P_q(A) ≈ {mass_A:.6f}  (det Dphi = {B_SCALE:g})")
