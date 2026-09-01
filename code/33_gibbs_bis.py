"""
Fisher--Rao distance map on the parameter space Theta of the linear Gibbs
(exponential) family phi_alpha(x) = (x, h_alpha(x)), normalized on E = [-2, 2],
with
    h_alpha(x) = (2-alpha) x + (alpha-1) x^2.

Hence alpha = 2 gives phi(x) = (x, x^2), and alpha -> 1 gives phi(x) -> (x, x).

This is the same numerical setup as 32_gibbs_fisher_rao_distance.py: the metric
    g_theta = Cov_theta(phi(X))
is evaluated on a 160 x 160 grid over Theta, each edge is weighted by the local
Riemannian length ds^2 = dtheta^T g dtheta, and Dijkstra's shortest-path
algorithm gives the geodesic distance to theta* = (-0.4, 0.55). The distance
color map and the iso-distance contours use exactly the same code as script 32.

Layout (2x2, decreasing alpha):
  top-left     alpha = 2.00
  top-right    alpha = 1.50
  bottom-left  alpha = 1.25
  bottom-right alpha = 1.10

Each panel shows the distance color map with iso-distance contours, and a small
inset comparing the feature curve phi_alpha(x) = (x, h_alpha(x)) with the
linear immersion (x, x).

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
X = np.linspace(-2.0, 2.0, 400)
DX = X[1] - X[0]

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

    Returns DIST[i, j] = d((t1[j], t2[i]), theta0), plus the reference node
    coordinates (i0, j0) used for Dijkstra.
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

    # Primitive integer directions (di, dj) with |di| <= 3, |dj| <= 3 and
    # gcd(|di|, |dj|) = 1. We keep only one representative of each opposite
    # pair (di > 0, or di == 0 and dj > 0); edges are inserted symmetrically in
    # both directions below, so the undirected graph covers both orientations.
    dirs = [
        (di, dj)
        for di in range(-3, 4)
        for dj in range(-3, 4)
        if (di, dj) != (0, 0)
        and math.gcd(abs(di), abs(dj)) == 1
        and (di > 0 or (di == 0 and dj > 0))
    ]

    rows, cols, data = [], [], []
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
                    rows += [a, b]
                    cols += [b, a]
                    data += [w, w]

    n_nodes = n2 * n1
    graph = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    i0 = int(np.argmin(np.abs(t2 - theta0[1])))
    j0 = int(np.argmin(np.abs(t1 - theta0[0])))
    dist = dijkstra(graph, directed=False, indices=node(i0, j0))
    return dist.reshape(n2, n1)


# ---------------------------------------------------------------------------
# The four cases: phi_alpha(x) = (x, h_alpha(x)), decreasing alpha
# ---------------------------------------------------------------------------
cases = [2.0, 1.5, 1.25, 1.1]

THETA0 = np.array([-0.4, 0.55])
T1 = np.linspace(-1.2, 0.6, 160)     # same grid as script 32
T2 = np.linspace(0.05, 1.5, 160)

maps = []
for a in cases:
    print(f"alpha = {a}")
    maps.append(distance_map(feature(a), THETA0, T1, T2))

# ---------------------------------------------------------------------------
# Figure: distance color map + iso-distance contours, 2x2 panels
# (same contour code as script 32)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.6))
axes = axes.ravel()

cmap = plt.get_cmap("coolwarm")
step = 0.1

for ax, a, dist in zip(axes, cases, maps):
    levels_fill = np.linspace(0.0, dist.max(), 40)
    cf = ax.contourf(T1, T2, dist, levels=levels_fill, cmap=cmap)

    iso = np.arange(step, dist.max(), step)
    cs = ax.contour(T1, T2, dist, levels=iso, colors="k", linewidths=0.7,
                    alpha=0.75)
    ax.clabel(cs, iso[::3], inline=True, fmt=r"$d=%.1f$", fontsize=8)

    # reference point theta*
    ax.plot(*THETA0, "o", color="white", ms=9, mew=1.4, mec="black", zorder=5)
    ax.annotate(
        r"$\theta_{\star}=(-0.4,\ 0.55)$",
        xy=THETA0,
        xytext=(0.12, 1.05),
        textcoords="data",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )

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

    ax.set_xlabel(r"$\theta_{1}$")
    ax.set_ylabel(r"$\theta_{2}$")
    ax.set_title(
        r"Fisher--Rao distance from $\theta_{\star}$, "
        + r"$\phi(x)=(x,h_{\alpha}(x))$"
        + "\n"
        + r"$g_{\theta}=\mathrm{Cov}_{\theta}(\phi(X))$ on $E=[-2,2]$",
        fontsize=11,
    )

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"Fisher--Rao distance $d(\theta,\theta_{\star})$")

fig.tight_layout()
path = OUT / "33_gibbs_bis.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
for a, dist in zip(cases, maps):
    print(f"alpha={a:.2f}  dmax = {dist.max():.3f}")
# plt.show()  # uncomment for interactive display
