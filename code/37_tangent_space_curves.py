"""
Intrinsic definition of the tangent space via equivalence classes of curves.

Same visual grammar as 05_manifold_chart.py: unit sphere M in the centre,
φ-chart plane underneath, ψ-chart plane to the right. Here the focus is a
single point p in the overlap of the two charts, and two curves γ₁, γ₂ on M
that pass through p with the same velocity but visibly different curvature.

In each chart the pushed-forward curves (φ ∘ γⱼ) and (ψ ∘ γⱼ) remain distinct,
yet share a common tangent arrow
    v_φ = (φ ∘ γ₁)'(0) = (φ ∘ γ₂)'(0),
    v_ψ = (ψ ∘ γ₁)'(0) = (ψ ∘ γ₂)'(0),
related by the differential of the transition map:
    v_ψ = d(ψ ∘ φ⁻¹)_{φ(p)} v_φ.

The figure thus illustrates
    γ₁ ∼_p γ₂  ⇔  (φ ∘ γ₁)'(0) = (φ ∘ γ₂)'(0),
independently of the chosen chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

ELEV, AZIM = 18.0, -48.0
PLANE_Z = -1.95
PLANE_X = 1.95
LIFT = 0.02
GRID_HALF = 1.55
N_LINES = 9
N_SAMPLES = 220
N_CURVE = 281  # odd, so that s = 0 is sampled exactly

# Point p chosen in the overlap so that both chart images sit near the
# centres of their planes (eastern hemisphere, mildly southern).
P_DIR = np.array([0.70, 0.42, -0.40])
P_DIR = P_DIR / np.linalg.norm(P_DIR)

# Common velocity in φ-coordinates, and two opposite curvatures.
# The base point U0 = φ(p) is computed after the chart maps are defined.
W_PHI = np.array([0.80, 0.30])
NORMAL = np.array([-0.30, 0.80])
NORMAL = NORMAL / np.linalg.norm(NORMAL)
KAPPA1, KAPPA2 = 1.55, -1.75
S_RANGE = (-0.75, 0.75)
ARROW_SCALE = 0.48

SPHERE_COLOR = "#A9C6E8"
PLANE_COLOR = "#EDE7DC"
GRID_PHI = "#C44E52"
GRID_PSI = "#3A7CA5"
CURVE1 = "#1A1A1A"
CURVE2 = "#E07A35"
TANGENT = "#111111"


class Arrow3D(FancyArrowPatch):
    """A 3D arrow drawn with the usual FancyArrowPatch machinery."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):  # noqa: ARG002
        xs, ys, zs = self._verts3d
        x, y, z = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return np.min(z)


def inverse_phi(u, v):
    """Inverse stereographic projection from the north pole → south pole at 0."""
    t = 4.0 / (u**2 + v**2 + 4.0)
    return t * u, t * v, 1.0 - 2.0 * t


def inverse_psi(s, t):
    """Inverse stereographic projection from the west pole → east pole at 0."""
    r2 = s**2 + t**2
    den = 4.0 + r2
    return (4.0 - r2) / den, 4.0 * s / den, 4.0 * t / den


def phi(x, y, z):
    """Forward chart φ: sphere → R², stereographic from the north pole."""
    den = 1.0 - z
    return 2.0 * x / den, 2.0 * y / den


def psi(x, y, z):
    """Forward chart ψ: sphere → R², stereographic from the west pole."""
    den = x + 1.0
    return 2.0 * y / den, 2.0 * z / den


def transition(u, v):
    """τ = ψ ∘ φ⁻¹."""
    return psi(*inverse_phi(u, v))


def d_transition(u0, v0, h=1e-6):
    """Jacobian matrix of τ = ψ ∘ φ⁻¹ at (u0, v0)."""
    s0, t0 = transition(u0, v0)
    su, tu = transition(u0 + h, v0)
    sv, tv = transition(u0, v0 + h)
    return np.array(
        [
            [(su - s0) / h, (sv - s0) / h],
            [(tu - t0) / h, (tv - t0) / h],
        ]
    )


def sphere_surface(n_theta=160, n_phi=80):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi_ang = np.linspace(0.0, np.pi, n_phi)
    T, P = np.meshgrid(theta, phi_ang)
    return np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)


def chart_curve(s, kappa):
    """Quadratic curve in φ-coordinates through U0 with velocity W_PHI."""
    s = np.asarray(s)
    return (
        U0[0] + s * W_PHI[0] + 0.5 * kappa * s**2 * NORMAL[0],
        U0[1] + s * W_PHI[1] + 0.5 * kappa * s**2 * NORMAL[1],
    )


def draw_light_grid(ax, coords_on_plane, push, color, z_plane, z_sphere):
    ticks = np.linspace(-GRID_HALF, GRID_HALF, N_LINES)
    line = np.linspace(-GRID_HALF, GRID_HALF, N_SAMPLES)
    for c in ticks:
        for a, b in ((np.full_like(line, c), line), (line, np.full_like(line, c))):
            px, py, pz = coords_on_plane(a, b)
            ax.plot(px, py, pz, color=color, lw=0.45, alpha=0.45, zorder=z_plane)
            sx, sy, sz = push(a, b)
            ax.plot(sx, sy, sz, color=color, lw=0.55, alpha=0.35, zorder=z_sphere)


def draw_arrow(ax, origin, tip, color, lw=1.8, mutation=12):
    arrow = Arrow3D(
        [origin[0], tip[0]],
        [origin[1], tip[1]],
        [origin[2], tip[2]],
        mutation_scale=mutation,
        lw=lw,
        arrowstyle="-|>",
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(arrow)


# ---------------------------------------------------------------------------
# Curves and point
# ---------------------------------------------------------------------------
U0 = np.array(phi(*P_DIR))
s = np.linspace(*S_RANGE, N_CURVE)
u1, v1 = chart_curve(s, KAPPA1)
u2, v2 = chart_curve(s, KAPPA2)

gamma1 = np.column_stack(inverse_phi(u1, v1))
gamma2 = np.column_stack(inverse_phi(u2, v2))
p = np.array(inverse_phi(*U0))
phi_p = U0.copy()
psi_p = np.array(psi(*p))

# Chart images of the curves
phi_g1 = np.column_stack([u1, v1])
phi_g2 = np.column_stack([u2, v2])
psi_g1 = np.column_stack(psi(*gamma1.T))
psi_g2 = np.column_stack(psi(*gamma2.T))

# Common chart tangents
v_phi = W_PHI
jac = d_transition(*U0)
v_psi = jac @ v_phi

# Ambient tangent on the sphere (for a short black arrow at p), via finite diff
eps = 1e-5
p_plus = np.array(inverse_phi(*(U0 + eps * W_PHI)))
v_ambient = (p_plus - p) / eps
v_ambient /= np.linalg.norm(v_ambient)

assert np.allclose(phi_g1[N_CURVE // 2], U0, atol=1e-8)
assert np.allclose(phi_g2[N_CURVE // 2], U0, atol=1e-8)
# Numerical check that both chart speeds agree
du1 = (phi_g1[N_CURVE // 2 + 1] - phi_g1[N_CURVE // 2 - 1]) / (s[1] - s[0]) / 2
du2 = (phi_g2[N_CURVE // 2 + 1] - phi_g2[N_CURVE // 2 - 1]) / (s[1] - s[0]) / 2
assert np.allclose(du1, du2, atol=5e-3)
assert np.allclose(du1, v_phi, atol=5e-3)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9.0, 6.6))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
pad = GRID_HALF * 1.12

# Chart planes
PX, PY = np.meshgrid(np.linspace(-pad, pad, 2), np.linspace(-pad, pad, 2))
ax.plot_surface(
    PX, PY, np.full_like(PX, PLANE_Z),
    color=PLANE_COLOR, alpha=0.95, linewidth=0.0, edgecolor="none",
    shade=False, zorder=0,
)
PY2, PZ2 = np.meshgrid(np.linspace(-pad, pad, 2), np.linspace(-pad, pad, 2))
ax.plot_surface(
    np.full_like(PY2, PLANE_X), PY2, PZ2,
    color=PLANE_COLOR, alpha=0.95, linewidth=0.0, edgecolor="none",
    shade=False, zorder=0,
)

# Manifold
XS, YS, ZS = sphere_surface()
ax.plot_surface(
    XS, YS, ZS,
    color=SPHERE_COLOR, alpha=0.28, rstride=1, cstride=1,
    linewidth=0.0, edgecolor="none", antialiased=True, shade=True, zorder=3,
)

# Light coordinate grids (context only)
draw_light_grid(
    ax,
    coords_on_plane=lambda u, v: (u, v, np.full_like(u, PLANE_Z + LIFT)),
    push=inverse_phi,
    color=GRID_PHI,
    z_plane=1,
    z_sphere=4,
)
draw_light_grid(
    ax,
    coords_on_plane=lambda s_, t_: (np.full_like(s_, PLANE_X - LIFT), s_, t_),
    push=inverse_psi,
    color=GRID_PSI,
    z_plane=1,
    z_sphere=4,
)

# Curves on the manifold
ax.plot(*gamma1.T, color=CURVE1, lw=2.15, zorder=6)
ax.plot(*gamma2.T, color=CURVE2, lw=2.15, zorder=6)

def mask_plane(coords):
    """Keep only the part of a chart curve that stays inside the drawn plane."""
    inside = (np.abs(coords[:, 0]) <= GRID_HALF) & (np.abs(coords[:, 1]) <= GRID_HALF)
    out = coords.astype(float).copy()
    out[~inside] = np.nan
    return out


# Curves on the φ-plane
phi_g1_draw = mask_plane(phi_g1)
phi_g2_draw = mask_plane(phi_g2)
ax.plot(
    phi_g1_draw[:, 0], phi_g1_draw[:, 1], np.full(N_CURVE, PLANE_Z + LIFT),
    color=CURVE1, lw=1.8, zorder=2,
)
ax.plot(
    phi_g2_draw[:, 0], phi_g2_draw[:, 1], np.full(N_CURVE, PLANE_Z + LIFT),
    color=CURVE2, lw=1.8, zorder=2,
)

# Curves on the ψ-plane
psi_g1_draw = mask_plane(psi_g1)
psi_g2_draw = mask_plane(psi_g2)
ax.plot(
    np.full(N_CURVE, PLANE_X - LIFT), psi_g1_draw[:, 0], psi_g1_draw[:, 1],
    color=CURVE1, lw=1.8, zorder=2,
)
ax.plot(
    np.full(N_CURVE, PLANE_X - LIFT), psi_g2_draw[:, 0], psi_g2_draw[:, 1],
    color=CURVE2, lw=1.8, zorder=2,
)

# Point p and its chart images
ax.plot(*p, "o", color="black", ms=7, zorder=8)
ax.plot(phi_p[0], phi_p[1], PLANE_Z + LIFT, "o", color=GRID_PHI, ms=6, zorder=5)
ax.plot(PLANE_X - LIFT, psi_p[0], psi_p[1], "o", color=GRID_PSI, ms=6, zorder=5)

# Dashed chart projections φ and ψ
ax.plot(
    [p[0], phi_p[0]], [p[1], phi_p[1]], [p[2], PLANE_Z + LIFT],
    ls="--", color=GRID_PHI, lw=1.05, zorder=5,
)
ax.plot(
    [p[0], PLANE_X - LIFT], [p[1], psi_p[0]], [p[2], psi_p[1]],
    ls="--", color=GRID_PSI, lw=1.05, zorder=5,
)

# Ambient tangent at p on M (short black arrow)
draw_arrow(ax, p, p + ARROW_SCALE * 0.85 * v_ambient, TANGENT, lw=1.7, mutation=11)

# Common tangent in φ-plane
phi_origin = np.array([phi_p[0], phi_p[1], PLANE_Z + LIFT])
phi_tip = phi_origin + ARROW_SCALE * np.array([v_phi[0], v_phi[1], 0.0])
draw_arrow(ax, phi_origin, phi_tip, TANGENT, lw=1.8, mutation=12)

# Common tangent in ψ-plane
psi_origin = np.array([PLANE_X - LIFT, psi_p[0], psi_p[1]])
psi_tip = psi_origin + ARROW_SCALE * np.array([0.0, v_psi[0], v_psi[1]])
draw_arrow(ax, psi_origin, psi_tip, TANGENT, lw=1.8, mutation=12)

# Transition map: dashed guide between the two chart images of p
ax.plot(
    [phi_p[0], PLANE_X - LIFT],
    [phi_p[1], psi_p[0]],
    [PLANE_Z + LIFT, psi_p[1]],
    ls=":",
    color="#666666",
    lw=1.0,
    alpha=0.85,
    zorder=2,
)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
ax.text(0.55, 0.95, 0.85, r"$\mathcal{M}$", fontsize=15, zorder=9)
ax.text(p[0] + 0.08, p[1] + 0.18, p[2] + 0.12, r"$p$", fontsize=13, zorder=9)

# Curve labels on the sphere
mid1 = gamma1[int(0.72 * N_CURVE)]
mid2 = gamma2[int(0.78 * N_CURVE)]
ax.text(
    mid1[0] + 0.05, mid1[1] + 0.12, mid1[2] + 0.08,
    r"$\gamma_{1}$", color=CURVE1, fontsize=12, zorder=9,
)
ax.text(
    mid2[0] - 0.05, mid2[1] + 0.18, mid2[2] - 0.05,
    r"$\gamma_{2}$", color=CURVE2, fontsize=12, zorder=9,
)

ax.text(
    phi_p[0] - 0.55, phi_p[1] - 0.05, PLANE_Z + 0.08,
    r"$\varphi(p)$", fontsize=11, color=GRID_PHI, zorder=9,
)
ax.text(
    PLANE_X + 0.08, psi_p[0] + 0.10, psi_p[1] + 0.18,
    r"$\psi(p)$", fontsize=11, color=GRID_PSI, zorder=9,
)

# Chart map names near the dashed projections
ax.text(0.15, 0.55, -1.05, r"$\varphi$", fontsize=13, color=GRID_PHI, zorder=9)
ax.text(1.25, 0.55, 0.35, r"$\psi$", fontsize=13, color=GRID_PSI, zorder=9)

ax.text(
    -pad * 0.20, -pad * 1.45, PLANE_Z,
    r"$\varphi(U)\subset\mathbb{R}^{2}$",
    fontsize=11, color=GRID_PHI, zorder=9,
)
ax.text(
    PLANE_X + 0.05, -pad * 0.10, -pad * 1.10,
    r"$\psi(V)\subset\mathbb{R}^{2}$",
    fontsize=11, color=GRID_PSI, zorder=9,
)

# Tangent labels (short near the arrows; the equalities live in the title)
ax.text(
    phi_tip[0] + 0.08, phi_tip[1] + 0.08, PLANE_Z + 0.08,
    r"$v_{\varphi}$", fontsize=12, color=TANGENT, zorder=9,
)
ax.text(
    PLANE_X + 0.08, psi_tip[1] + 0.05, psi_tip[2] + 0.08,
    r"$v_{\psi}$", fontsize=12, color=TANGENT, zorder=9,
)

# Transition and differential relation
mid_tr = 0.5 * (phi_origin + psi_origin) + np.array([0.05, -0.45, 0.15])
ax.text(
    mid_tr[0], mid_tr[1], mid_tr[2],
    r"$\psi\circ\varphi^{-1}$",
    fontsize=12, color="#555555", zorder=9,
)
ax.text(
    PLANE_X + 0.08, -1.05, 0.75,
    r"$v_{\psi}=d(\psi\circ\varphi^{-1})_{\varphi(p)}\,v_{\varphi}$",
    fontsize=10.5, color=TANGENT, zorder=9,
)

ax.view_init(elev=ELEV, azim=AZIM)
ax.set_xlim(-1.75, 2.55)
ax.set_ylim(-1.9, 1.9)
ax.set_zlim(PLANE_Z - 0.20, 1.20)
ax.set_box_aspect((4.3, 3.8, 3.4), zoom=1.12)
ax.set_axis_off()
ax.set_title(
    r"$\gamma_{1}\sim_{p}\gamma_{2}"
    r"\;\Longleftrightarrow\;"
    r"(\varphi\circ\gamma_{1})'(0)=(\varphi\circ\gamma_{2})'(0)"
    r"\;(=\!v_{\varphi})$"
    "\n"
    r"same tangent class in every chart: "
    r"$v_{\psi}=d(\psi\circ\varphi^{-1})_{\varphi(p)}\,v_{\varphi}$",
    fontsize=11.5,
    pad=2,
)

path = OUT / "37_tangent_space_curves.png"
fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote {path}")
print(f"  p          = {p}")
print(f"  φ(p)       = {phi_p}")
print(f"  ψ(p)       = {psi_p}")
print(f"  v_φ        = {v_phi}")
print(f"  v_ψ        = {v_psi}")
print(f"  J_τ @ v_φ  = {jac @ v_phi}")
# plt.show()
