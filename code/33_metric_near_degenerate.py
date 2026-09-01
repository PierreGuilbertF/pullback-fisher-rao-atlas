"""
Fisher--Rao distance map for the linear Gibbs family
phi_alpha(x) = (x, h_alpha(x)), compared with the linear immersion (x, x),
with
    h_alpha(x) = (2-alpha) x + (alpha-1) x^2.

Hence alpha = 2 gives phi(x) = (x, x^2), and alpha -> 1 gives phi(x) -> (x, x).
As alpha -> 1 the feature map becomes almost a linear immersion, so the
pulled-back Fisher--Rao metric
    g_theta = Cov_theta(phi(X))
loses rank: one eigenvalue lambda_min -> 0 while lambda_max stays bounded.
The Riemannian distance to the reference parameter therefore flattens along the
near-zero-cost parameter direction: iso-distance contours become more and more
elongated as alpha decreases, and the distance itself collapses in that
direction.

Layout (2x2, decreasing alpha):
  top-left     alpha = 3.00
  top-right    alpha = 2.00
  bottom-left  alpha = 1.50
  bottom-right alpha = 1.10

Each panel shows the geodesic Fisher--Rao distance to theta* = (-0.4, 0.55),
evaluated numerically with Dijkstra on a grid (same method as script 32), with
iso-distance contours; a small inset shows the feature curve
phi_alpha(x) = (x, h_alpha(x)) approaching the line (x, x).

Comments and labels are in English.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Density data on the bounded domain E = [-2, 2]
# ---------------------------------------------------------------------------
X = np.linspace(-2.0, 2.0, 600)
DX = X[1] - X[0]
THETA0 = np.array([-0.4, 0.55])

COLOR_FEATURE = "#4C78A8"


def feature(alpha):
    """
    phi_alpha(x) = (x, h_alpha(x))

    with
        h_alpha(x) = (2-alpha) x + (alpha-1) x^2.

    Hence:
        alpha = 2  -> phi(x) = (x, x^2)
        alpha -> 1 -> phi(x) -> (x, x)
    """
    h = (2.0 - alpha) * X + (alpha - 1.0) * X**2
    return np.column_stack([X, h])


def metric_at(theta, phi):
    """Cov_theta(phi(X)) for p_theta(x) propto exp(-phi(x)^T theta)."""
    logu = -(phi @ theta)
    logu -= float(logu.max())
    u = np.exp(logu)
    density = u / (u.sum() * DX)
    weights = density * DX
    mean = weights @ phi
    centered = phi - mean
    cov = (centered * weights[:, None]).T @ centered
    return 0.5 * (cov + cov.T)


def distance_map(phi, theta0, t1, t2):
    """Geodesic Fisher--Rao distance from theta0 on a grid over Theta.

    Returns DIST[i, j] = d((t1[j], t2[i]), theta0).
    """
    n1, n2 = len(t1), len(t2)

    # metric on every grid node
    G = np.empty((n2, n1, 2, 2))
    for i in range(n2):
        for j in range(n1):
            G[i, j] = metric_at(np.array([t1[j], t2[i]]), phi)

    # Grid graph with a richer directional stencil.
    #
    # A plain 4-neighbor stencil (horizontal/vertical moves only) approximates
    # the metric by a grid-induced L^1-type metric: each edge costs g11*|dtheta1|
    # or g22*|dtheta2|, so the iso-distance contours collapse onto the diamond
    # |dtheta1| + |dtheta2| = cst, aligned with the coordinate grid, and the
    # off-diagonal term g12 is never used. The richer stencil below lets paths
    # propagate in many more directions, which removes that four-corner
    # anisotropy, and every edge now carries the full quadratic form
    # delta.T @ G @ delta. The result is still a graph-based numerical
    # approximation of the true Riemannian geodesic distance.

    h1 = t1[1] - t1[0]
    h2 = t2[1] - t2[0]

    def node(i, j):
        return i * n1 + j

    # Primitive integer directions (di, dj) with |di| <= 8, |dj| <= 8 and
    # gcd(|di|, |dj|) = 1. We keep only one representative of each opposite
    # pair (di > 0, or di == 0 and dj > 0); edges are inserted symmetrically in
    # both directions below, so the undirected graph covers both orientations.
    # The wider stencil gives the graph's "unit ball" a higher-order polygonal
    # shape, rounding the wavefront facets and smoothing the iso-contours.
    dirs = [
        (di, dj)
        for di in range(-8, 9)
        for dj in range(-8, 9)
        if (di, dj) != (0, 0)
        and math.gcd(abs(di), abs(dj)) == 1
        and (di > 0 or (di == 0 and dj > 0))
    ]

    # Preallocate the graph arrays as numpy arrays (not Python lists) so that
    # the fine grid keeps memory bounded and the build stays fast.
    total = 0
    for i in range(n2):
        for j in range(n1):
            for di, dj in dirs:
                ii, jj = i + di, j + dj
                if 0 <= ii < n2 and 0 <= jj < n1:
                    total += 2

    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    data = np.empty(total, dtype=np.float64)

    k = 0
    for i in range(n2):
        for j in range(n1):
            a = node(i, j)
            for di, dj in dirs:
                ii, jj = i + di, j + dj
                if 0 <= ii < n2 and 0 <= jj < n1:
                    b = node(ii, jj)
                    # coordinate displacement, in physical units
                    delta = np.array([dj * h1, di * h2])
                    # metric evaluated at the edge midpoint: this is what
                    # brings the off-diagonal term g12 into the edge length
                    Gmid = 0.5 * (G[i, j] + G[ii, jj])
                    w = np.sqrt(delta @ Gmid @ delta)
                    rows[k], cols[k], data[k] = a, b, w
                    k += 1
                    rows[k], cols[k], data[k] = b, a, w
                    k += 1

    n_nodes = n2 * n1
    graph = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    i0 = int(np.argmin(np.abs(t2 - theta0[1])))
    j0 = int(np.argmin(np.abs(t1 - theta0[0])))
    dist = dijkstra(graph, directed=False, indices=node(i0, j0))
    return dist.reshape(n2, n1)


cases = [3.0, 2.0, 1.5, 1.1]

# parameter grid over Theta, shared by all panels for a comparable color scale
# (finer grid -> smoother iso-distance contours and color map)
N_GRID = 480
# zoomed window: centered on theta*, with equal span in theta_1 and theta_2 so
# the panels can be drawn square (aspect equal)
# (grid slightly coarser than before: the 88-direction stencil now provides the
#  smoothness, so a very fine grid is no longer needed to round the contours)
span = 0.45   # common span (full width) in both theta_1 and theta_2
T1 = np.linspace(THETA0[0] - span / 2, THETA0[0] + span / 2, N_GRID)
T2 = np.linspace(THETA0[1] - span / 2, THETA0[1] + span / 2, N_GRID)

maps = []
for a in cases:
    print(f"alpha = {a}")
    maps.append(distance_map(feature(a), THETA0, T1, T2))

# shared color scale so the shrinking distance as alpha -> 1 is visible
vmax = max(d.max() for d in maps)
# iso-contour levels: the gap (dist.max()-dist.min())/(N_ISO-1) is recomputed
# per panel, so it automatically shrinks with the zoomed window
N_ISO = 24

fig, axes = plt.subplots(2, 2, figsize=(18.5, 15.2))
axes = axes.ravel()
cmap = plt.get_cmap("coolwarm")

for ax, a, dist in zip(axes, cases, maps):
    # --- distance color map + iso-distance contours -------------------------
    levels_fill = np.linspace(0.0, vmax, 64)
    cf = ax.contourf(T1, T2, dist, levels=levels_fill, cmap=cmap)

    iso = np.linspace(dist.min(), dist.max(), N_ISO)
    cs = ax.contour(T1, T2, dist, levels=iso, colors="k", linewidths=0.7,
                    alpha=0.75)
    ax.clabel(cs, iso[1::3], inline=True, fmt=r"$d=%.1f$", fontsize=8)

    # reference point theta*
    ax.plot(*THETA0, "o", color="white", ms=8, mew=1.4, mec="black", zorder=5)
    ax.annotate(
        r"$\theta_{\star}=(-0.4,\ 0.55)$",
        xy=THETA0,
        xytext=(THETA0[0] + 0.5, THETA0[1] - 0.12),
        textcoords="data",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )

    # --- annotations ---------------------------------------------------------
    G = metric_at(THETA0, feature(a))
    evals = np.linalg.eigvalsh(G)
    ax.text(0.03, 0.96,
            r"$\alpha = %.2f$" % a + "\n"
            r"$\lambda_{\min}=%.3g$" % evals[0] + "\n"
            r"$\lambda_{\max}=%.3g$" % evals[1] + "\n"
            r"$\lambda_{\min}/\lambda_{\max}=%.2g$" % (evals[0] / evals[1]),
            transform=ax.transAxes, va="top", fontsize=10)

    # small inset: feature curve phi_alpha(x) = (x, h_alpha(x)), approaching a
    # line (x, x) as alpha -> 1
    axin = ax.inset_axes([0.60, 0.05, 0.34, 0.30])
    h = (2.0 - a) * X + (a - 1.0) * X**2
    axin.plot(X, h, color=COLOR_FEATURE, lw=2.0)
    axin.plot(X, X, ls="--", color="0.55", lw=1.0)   # alpha = 1 limit
    axin.set_xlim(-2.2, 2.2)
    axin.set_ylim(-2.2, 2.2)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_title(r"$\phi(x)=(x,h_{\alpha}(x))$ vs $(x,x)$", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_{1}$")
    ax.set_ylabel(r"$\theta_{2}$")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"Fisher--Rao distance $d(\theta,\theta_{\star})$")

fig.suptitle(
    r"Fisher--Rao distance to $\theta_{\star}=(-0.4,0.55)$ for "
    r"$g_{\theta}=\mathrm{Cov}_{\theta}(\phi(X))$, "
    r"$\phi(x)=(x,h_{\alpha}(x))$"
    "\n"
    r"distance flattens along the near-zero-cost direction as "
    r"$\alpha\to1$ (feature becomes a linear immersion)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
path = OUT / "33_metric_near_degenerate.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
for a in cases:
    ev = np.linalg.eigvalsh(metric_at(THETA0, feature(a)))
    print(f"alpha={a:.2f}  lambda=({ev[0]:.4f},{ev[1]:.4f})  "
          f"cond={ev[1]/max(ev[0],1e-12):.1f}")
# plt.show()  # uncomment for interactive display
