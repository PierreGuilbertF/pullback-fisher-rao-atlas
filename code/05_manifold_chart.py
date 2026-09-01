"""
Two overlapping charts on the unit sphere.

φ is the stereographic chart from the north pole: its origin is the south pole
p₀, and the chart plane is drawn underneath the sphere. ψ is the stereographic
chart from the west pole: its origin is the east pole p₁ = (1, 0, 0), and the
chart plane is drawn to the right of the sphere. A regular grid on each chart
is pushed onto the sphere by the corresponding inverse map. On the overlap, the
red φ-grid is also drawn on the blue ψ-plane, as its image under the transition
map ψ ∘ φ^{-1}: the curves stay smooth, which is the geometric content of a
C¹ (here C^∞) change of charts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# Three-quarter view, slightly from above and to the left, so both planes show.
ELEV, AZIM = 18.0, -48.0

PLANE_Z = -1.95  # φ-plane, under the south pole
PLANE_X = 1.95  # ψ-plane, to the right of the east pole
# mplot3d needs a small lift so things drawn on a plane win against the plane.
LIFT = 0.02
GRID_HALF = 1.35
N_LINES = 9
N_SAMPLES = 220

P0 = np.array([0.0, 0.0, -1.0])  # south pole
P1 = np.array([1.0, 0.0, 0.0])  # east pole

SPHERE_COLOR = "#A9C6E8"
PLANE_COLOR = "#EDE7DC"
GRID_PHI = "#C44E52"  # red
GRID_PSI = "#3A7CA5"  # blue


def inverse_phi(u, v):
    """Inverse stereographic projection from the north pole.

    Sends (0, 0) to the south pole p₀.
    """
    t = 4.0 / (u**2 + v**2 + 4.0)
    return t * u, t * v, 1.0 - 2.0 * t


def inverse_psi(s, t):
    """Inverse stereographic projection from the west pole.

    Sends (0, 0) to the east pole p₁ = (1, 0, 0). Chart coordinates (s, t)
    live in the (y, z) directions of the tangent plane at p₁.
    """
    r2 = s**2 + t**2
    den = 4.0 + r2
    return (4.0 - r2) / den, 4.0 * s / den, 4.0 * t / den


def psi(x, y, z):
    """Forward chart ψ: sphere → R², stereographic from the west pole.

    Defined away from (-1, 0, 0). Returns chart coordinates (s, t).
    """
    den = x + 1.0
    return 2.0 * y / den, 2.0 * z / den


def sphere_surface(n_theta=180, n_phi=90):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    T, P = np.meshgrid(theta, phi)
    return np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)


def draw_grid_on_plane_and_sphere(ax, coords_on_plane, push, color, z_plane, z_sphere):
    """Draw a regular square grid on a chart plane and its image on the sphere."""
    ticks = np.linspace(-GRID_HALF, GRID_HALF, N_LINES)
    line = np.linspace(-GRID_HALF, GRID_HALF, N_SAMPLES)
    for c in ticks:
        for a, b in ((np.full_like(line, c), line), (line, np.full_like(line, c))):
            px, py, pz = coords_on_plane(a, b)
            ax.plot(px, py, pz, color=color, lw=0.75, alpha=0.8, zorder=z_plane)
            sx, sy, sz = push(a, b)
            ax.plot(sx, sy, sz, color=color, lw=1.05, alpha=0.95, zorder=z_sphere)


def draw_transition_on_psi_plane(ax):
    """Image of the φ-grid under ψ ∘ φ^{-1}, restricted to the chart overlap.

    On the blue plane, this is the red grid written in ψ-coordinates: still a
    smooth family of curves, but no longer a Euclidean square grid — which is
    exactly what C¹ (in fact C^∞) transition maps look like.
    """
    ticks = np.linspace(-GRID_HALF, GRID_HALF, N_LINES)
    line = np.linspace(-GRID_HALF, GRID_HALF, N_SAMPLES)
    margin = GRID_HALF * 1.001

    for c in ticks:
        for a, b in ((np.full_like(line, c), line), (line, np.full_like(line, c))):
            x, y, z = inverse_phi(a, b)
            s, t = psi(x, y, z)
            # keep only the overlap: points that also land inside ψ(V)
            inside = (
                (x > -0.95)
                & (np.abs(s) <= margin)
                & (np.abs(t) <= margin)
            )
            s_plot = np.where(inside, s, np.nan)
            t_plot = np.where(inside, t, np.nan)
            ax.plot(
                np.full_like(s_plot, PLANE_X - LIFT),
                s_plot,
                t_plot,
                color=GRID_PHI,
                lw=1.15,
                alpha=0.95,
                zorder=2,
            )


fig = plt.figure(figsize=(8.2, 6.2))
# computed_zorder=False: automatic depth sort paints planes over their own grids.
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

pad = GRID_HALF * 1.15

# --- φ-plane (horizontal, under the south pole) ---
PX, PY = np.meshgrid(np.linspace(-pad, pad, 2), np.linspace(-pad, pad, 2))
ax.plot_surface(
    PX,
    PY,
    np.full_like(PX, PLANE_Z),
    color=PLANE_COLOR,
    alpha=0.95,
    linewidth=0.0,
    edgecolor="none",
    shade=False,
    zorder=0,
)

# --- ψ-plane (vertical, to the right of the east pole) ---
PY2, PZ2 = np.meshgrid(np.linspace(-pad, pad, 2), np.linspace(-pad, pad, 2))
ax.plot_surface(
    np.full_like(PY2, PLANE_X),
    PY2,
    PZ2,
    color=PLANE_COLOR,
    alpha=0.95,
    linewidth=0.0,
    edgecolor="none",
    shade=False,
    zorder=0,
)

# --- the manifold ---
XS, YS, ZS = sphere_surface()
ax.plot_surface(
    XS,
    YS,
    ZS,
    color=SPHERE_COLOR,
    alpha=0.30,
    rstride=1,
    cstride=1,
    linewidth=0.0,
    edgecolor="none",
    antialiased=True,
    shade=True,
    zorder=3,
)

# --- grids: φ then ψ (ψ on top in the overlap, so the covering is readable) ---
draw_grid_on_plane_and_sphere(
    ax,
    coords_on_plane=lambda u, v: (u, v, np.full_like(u, PLANE_Z + LIFT)),
    push=inverse_phi,
    color=GRID_PHI,
    z_plane=1,
    z_sphere=4,
)
draw_grid_on_plane_and_sphere(
    ax,
    coords_on_plane=lambda s, t: (np.full_like(s, PLANE_X - LIFT), s, t),
    push=inverse_psi,
    color=GRID_PSI,
    z_plane=1,
    z_sphere=5,
)
# red grid seen from the blue chart: image of φ-coordinates under ψ ∘ φ^{-1}
draw_transition_on_psi_plane(ax)

# --- marked points and chart images ---
ax.plot(*P0, "o", color="black", ms=7, zorder=6)
ax.plot(*P1, "o", color="black", ms=7, zorder=6)
ax.plot(0.0, 0.0, PLANE_Z + LIFT, "o", color=GRID_PHI, ms=6, zorder=2)
ax.plot(PLANE_X - LIFT, 0.0, 0.0, "o", color=GRID_PSI, ms=6, zorder=2)

ax.plot(
    [0.0, 0.0],
    [0.0, 0.0],
    [P0[2], PLANE_Z + LIFT],
    ls="--",
    color=GRID_PHI,
    lw=1.0,
    zorder=2,
)
ax.plot(
    [P1[0], PLANE_X - LIFT],
    [0.0, 0.0],
    [0.0, 0.0],
    ls="--",
    color=GRID_PSI,
    lw=1.0,
    zorder=2,
)

# corner correspondences for each chart
for u0, v0 in ((GRID_HALF, GRID_HALF), (-GRID_HALF, GRID_HALF)):
    sx, sy, sz = inverse_phi(u0, v0)
    ax.plot(
        [u0, sx],
        [v0, sy],
        [PLANE_Z + LIFT, sz],
        ls=":",
        color=GRID_PHI,
        lw=0.7,
        alpha=0.7,
        zorder=2,
    )
for s0, t0 in ((GRID_HALF, GRID_HALF), (GRID_HALF, -GRID_HALF)):
    sx, sy, sz = inverse_psi(s0, t0)
    ax.plot(
        [PLANE_X - LIFT, sx],
        [s0, sy],
        [t0, sz],
        ls=":",
        color=GRID_PSI,
        lw=0.7,
        alpha=0.7,
        zorder=2,
    )

# --- labels ---
ax.text(0.55, 0.95, 0.85, r"$\mathcal{M}$", fontsize=15, zorder=7)
ax.text(0.12, 0.14, -1.05, r"$p_{0}$", fontsize=13, zorder=7)
ax.text(1.08, 0.12, 0.12, r"$p_{1}$", fontsize=13, zorder=7)
ax.text(0.16, 0.16, PLANE_Z + 0.05, r"$\varphi(p_{0})$", fontsize=11, color=GRID_PHI, zorder=7)
ax.text(
    PLANE_X + 0.08,
    0.18,
    0.18,
    r"$\psi(p_{1})$",
    fontsize=11,
    color=GRID_PSI,
    zorder=7,
)
ax.text(0.22, 0.0, -1.48, r"$\varphi$", fontsize=13, color=GRID_PHI, zorder=7)
ax.text(1.45, 0.0, 0.22, r"$\psi$", fontsize=13, color=GRID_PSI, zorder=7)
ax.text(
    -pad * 0.35,
    -pad * 1.55,
    PLANE_Z,
    r"$\varphi(U) \subset \mathbb{R}^{2}$",
    fontsize=11,
    color=GRID_PHI,
    zorder=7,
)
ax.text(
    PLANE_X + 0.05,
    -pad * 0.15,
    -pad * 1.15,
    r"$\psi(V) \subset \mathbb{R}^{2}$",
    fontsize=11,
    color=GRID_PSI,
    zorder=7,
)
ax.text(
    PLANE_X + 0.12,
    -0.15,
    -0.85,
    r"$\psi\circ\varphi^{-1}$",
    fontsize=12,
    color=GRID_PHI,
    zorder=7,
)

ax.view_init(elev=ELEV, azim=AZIM)
ax.set_xlim(-1.55, 2.35)
ax.set_ylim(-1.7, 1.7)
ax.set_zlim(PLANE_Z - 0.15, 1.15)
ax.set_box_aspect((3.9, 3.4, 3.3), zoom=1.18)
ax.set_axis_off()

ax.set_title(
    "Two overlapping local charts\n"
    r"on $\psi(V)$: image of the red grid under "
    r"$\psi\circ\varphi^{-1}$ (change of charts)",
    fontsize=12,
    pad=2,
)

path = OUT / "05_manifold_chart.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"  p0 (south) {tuple(P0)} <-> φ-origin")
print(f"  p1 (east)  {tuple(P1)} <-> ψ-origin")
# plt.show()  # uncomment for interactive display
