"""
Natural gradient vs Euclidean gradient for f(t1, t2) = 1/2 (t1^2 + t2^2).

At theta0 = (1, 0):
  - Euclidean gradient  grad_f      = (1, 0)
  - Riemannian gradient grad_g f    = G^{-1} grad_f = (2/3, -1/3),  G = [[2,1],[1,2]].

Panels (2x2):
  Top left   : graph of f with the Euclidean metric I.  Unit circle and the
               Euclidean gradient at the point.
  Top right  : graph of f with the constant metric G.  Unit ellipse (t^T G t = 1)
               and the Riemannian gradient at the point.
  Bottom left: the same function read in straightened coordinates.  We apply
               A^{-1} to the coordinates with A^T A = G, i.e. theta = A^{-1} z.
               In z the metric is Euclidean, so the Riemannian gradient is just
               the ordinary gradient of the stretched graph.
  Bottom right: the two gradients drawn in the theta-plane (canonical basis),
               together with the two unit balls centered at the point.

Comments and labels are in English.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data: metric and gradients
# ---------------------------------------------------------------------------
G = np.array([[2.0, 1.0], [1.0, 2.0]])
L = np.linalg.cholesky(G)
A = L.T  # upper-triangular Cholesky factor, A^T A = G

theta0 = np.array([1.0, 0.0])
f0 = 0.5 * float(theta0 @ theta0)

grad_eucl = np.array([1.0, 0.0])                    # nabla f
grad_g = np.linalg.solve(G, grad_eucl)              # G^{-1} nabla f = (2/3,-1/3)


def unit_wrt(v, metric):
    """Rescale v so it has unit length under `metric` (v^T metric v = 1).

    Drawn this way the gradient arrow touches the unit ball of the metric,
    which is exactly the "steepest direction among unit-length displacements"
    picture: the Euclidean gradient touches the unit circle, the Riemannian
    gradient touches the unit ellipse.
    """
    n = np.sqrt(float(v @ metric @ v))
    return v / n


# Unit gradients: each touches its own metric's unit ball.
u_eucl = unit_wrt(grad_eucl, np.eye(2))
u_g = unit_wrt(grad_g, G)

# Straightened coordinates: z = A theta  <=>  theta = A^{-1} z.
# In z the metric is Euclidean, so the Riemannian gradient becomes ordinary.
z0 = A @ theta0
grad_z = A @ grad_g                                  # = Euclidean grad in z
AA = A @ A.T
assert np.allclose(A.T @ G @ A, np.eye(2)) is False   # (pullback theta->z is not I)
assert np.allclose(np.linalg.solve(A.T, G @ np.linalg.solve(A, np.eye(2))),
                   np.eye(2))                        # metric in z IS the identity
assert np.allclose(grad_z, np.linalg.solve(AA, z0))  # riemannian = euclidean in z

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
LIM = 2.2
COLOR_SURF = "#4C78A8"
COLOR_GRID = "#2b4a6f"
COLOR_EUCL = "#3A7CA5"   # blue: euclidean
COLOR_RIEM = "#C44E52"   # red:  riemannian

ELEV, AZIM = 26.0, -62.0


def f_surface(x, y):
    return 0.5 * (x**2 + y**2)


def f_straightened(x, y):
    """tilde f(z) = f(A^{-1} z) = 1/2 z^T (A A^T)^{-1} z, over the z-plane."""
    M = np.linalg.inv(AA)
    return 0.5 * (M[0, 0] * x**2 + 2.0 * M[0, 1] * x * y + M[1, 1] * y**2)


def unit_ellipse(metric, center, n=200):
    """Points of {v : (v-center)^T metric (v-center) = 1}."""
    evals, evecs = np.linalg.eigh(metric)
    evals = np.maximum(evals, 1e-12)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)  # (n, 2)
    shape = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return center + (shape @ circle.T).T


def draw_surface(ax, fn, zmax, title):
    xs = np.linspace(-LIM, LIM, 70)
    X, Y = np.meshgrid(xs, xs)
    Z = fn(X, Y)
    # faint level lines of the surface projected onto the floor (z = 0)
    ax.contour(
        X, Y, Z, levels=np.linspace(0.0, zmax, 9)[1:],
        zdir="z", offset=0.0, colors="0.80", linewidths=0.5, zorder=1,
    )
    ax.plot_surface(
        X, Y, Z, color=COLOR_SURF, alpha=0.32,
        rstride=5, cstride=5, edgecolor=COLOR_GRID, linewidth=0.2,
        shade=True, zorder=3,
    )
    ax.set_zlim(0.0, zmax)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$\theta_{1}$", labelpad=-7)
    ax.set_ylabel(r"$\theta_{2}$", labelpad=-7)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_zticks([])
    ax.set_title(title, fontsize=10.5, pad=2)


def draw_point_and_gradient(ax, point, height, grad, metric, color, label, center):
    """Unit ball of `metric` on the floor, the point, and the gradient arrow."""
    # unit ball on the floor (centered at `center`)
    pts = unit_ellipse(metric, center)
    ax.plot(pts[:, 0], pts[:, 1], np.zeros(len(pts)), color=color, lw=1.8, zorder=5)
    # point on the surface + drop line to the floor
    x, y = point
    ax.scatter([x], [y], [height], color="white", edgecolor="black", s=42, zorder=8)
    ax.plot([x, x], [y, y], [0.0, height], ls="--", color="0.40", lw=0.8, zorder=6)
    ax.scatter([x], [y], [0.0], color=color, s=20, zorder=7)
    # gradient arrow from the floor point
    g = grad
    ax.quiver(x, y, 0.0, g[0], g[1], 0.0, color=color, lw=2.2,
              arrow_length_ratio=0.22, zorder=9)
    ax.text(x + g[0] * 1.12, y + g[1] * 1.12, 0.0, label, color=color,
            fontsize=11, zorder=10)


fig = plt.figure(figsize=(11.6, 9.6))

# --- Top left: Euclidean metric ------------------------------------------
ax00 = fig.add_subplot(2, 2, 1, projection="3d", computed_zorder=False)
zmax = f_surface(LIM, LIM)
draw_surface(ax00, f_surface, zmax, r"Euclidean metric $I$")
draw_point_and_gradient(
    ax00, theta0, f0, u_eucl, np.eye(2), COLOR_EUCL,
    r"$\nabla f$", center=theta0,
)

# --- Top right: Riemannian metric G ---------------------------------------
ax01 = fig.add_subplot(2, 2, 2, projection="3d", computed_zorder=False)
draw_surface(ax01, f_surface, zmax,
             r"Riemannian metric $G=\left(\substack{2\;\;1\\1\;\;2}\right)$")
draw_point_and_gradient(
    ax01, theta0, f0, u_g, G, COLOR_RIEM,
    r"$\nabla_{g} f$", center=theta0,
)

# --- Bottom left: straightened coordinates ---------------------------------
ax10 = fig.add_subplot(2, 2, 3, projection="3d", computed_zorder=False)
zmax_z = f_straightened(LIM, LIM)
draw_surface(ax10, f_straightened, zmax_z,
             r"Straightened coordinates, $\theta=A^{-1}z$" "\n"
             r"metric $=$ Euclidean: $\nabla_g f$ is the ordinary gradient")
# in z the metric is Euclidean (identity), unit ball is a circle
draw_point_and_gradient(
    ax10, z0, f_straightened(z0[0], z0[1]), grad_z, np.eye(2), COLOR_RIEM,
    r"$\nabla_{g} f$", center=z0,
)
# (grad_z is already the ordinary Euclidean gradient in the straightened plane)
ax10.set_xlabel(r"$z_{1}$", labelpad=-7)
ax10.set_ylabel(r"$z_{2}$", labelpad=-7)

# --- Bottom right: the two gradients in the theta-plane --------------------
ax11 = fig.add_subplot(2, 2, 4)
ax11.set_aspect("equal")
ax11.axhline(0.0, color="0.75", lw=0.6)
ax11.axvline(0.0, color="0.75", lw=0.6)
# unit balls centered at the point (1,0)
pe = unit_ellipse(np.eye(2), theta0)
pg = unit_ellipse(G, theta0)
ax11.plot(pe[:, 0], pe[:, 1], color=COLOR_EUCL, lw=1.6, label=r"unit ball, metric $I$")
ax11.plot(pg[:, 0], pg[:, 1], color=COLOR_RIEM, lw=1.6, label=r"unit ball, metric $G$")
# point and unit gradients (each touches its own unit ball)
ax11.plot(theta0[0], theta0[1], "o", color="white", ms=8, mec="black", mew=1.1, zorder=5)
ax11.annotate(
    "", xy=theta0 + u_eucl, xytext=theta0,
    arrowprops=dict(arrowstyle="-|>", color=COLOR_EUCL, lw=2.2),
)
ax11.annotate(
    "", xy=theta0 + u_g, xytext=theta0,
    arrowprops=dict(arrowstyle="-|>", color=COLOR_RIEM, lw=2.2),
)
ax11.text(theta0[0] + 1.12 * u_eucl[0], theta0[1] + 1.12 * u_eucl[1],
          r"$\nabla f$", color=COLOR_EUCL, fontsize=12)
ax11.text(theta0[0] + 1.15 * u_g[0], theta0[1] + 1.15 * u_g[1],
          r"$\nabla_{g} f$", color=COLOR_RIEM, fontsize=12)
ax11.set_xlabel(r"$\theta_{1}$")
ax11.set_ylabel(r"$\theta_{2}$")
ax11.set_xlim(-0.7, 2.5)
ax11.set_ylim(-1.25, 1.25)
ax11.set_title(r"Euclidean vs Riemannian gradient", fontsize=10.5)
ax11.legend(loc="upper left", frameon=False, fontsize=8)

fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.02,
                    wspace=0.02, hspace=0.06)

path = OUT / "31_natural_gradient.png"
fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Wrote {path}")
print(f"grad_eucl = {grad_eucl}")
print(f"grad_g    = {grad_g}")
print(f"grad in z = {grad_z}  (riemannian = ordinary gradient in straightened coords)")
