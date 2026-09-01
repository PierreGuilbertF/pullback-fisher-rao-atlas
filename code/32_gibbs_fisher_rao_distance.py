"""
Fisher--Rao distance map on the parameter space Theta of two linear Gibbs
(exponential) families, each normalized on a bounded interval E = [-2, 2]:

  left : phi(x) = (x, x^2),   reference theta* = (-0.4, 0.55)
  right: phi(x) = (x^2, cos x), reference theta* = (0.45, 1.2)

Metric:  g_theta = Cov_theta(phi(X)).
Distance: geodesic distance on (Theta, g) from the reference parameter.  The
          metric has no closed form here, so the distance is evaluated
          numerically: we build a fine grid over Theta, weight each grid edge
          by the local Riemannian length ds^2 = dtheta^T g dtheta, and run
          Dijkstra's shortest-path algorithm from theta*.

This mirrors the Gaussian distance map (02_distance_map.py): a color map of
the Fisher--Rao distance to the reference point, with iso-distance contours
at a regular interval.

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
# The two cases
# ---------------------------------------------------------------------------
phi_poly = np.column_stack([X, X**2])
phi_cos = np.column_stack([X**2, np.cos(X)])

cases = [
    {
        "phi": phi_poly,
        "theta0": np.array([-0.4, 0.55]),
        "t1": np.linspace(-1.2, 0.6, 160),
        "t2": np.linspace(0.05, 1.5, 160),
        "title": r"Gibbs family $\phi(x)=(x,x^{2})$",
        "ref": r"$\theta_{\star}=(-0.4,\ 0.55)$",
        "arrow": (0.12, 1.05),
    },
    {
        "phi": phi_cos,
        "theta0": np.array([0.45, 1.2]),
        "t1": np.linspace(0.0, 1.2, 160),
        "t2": np.linspace(0.6, 1.8, 160),
        "title": r"Gibbs family $\phi(x)=(x^{2},\cos x)$",
        "ref": r"$\theta_{\star}=(0.45,\ 1.2)$",
        "arrow": (0.15, 1.02),
    },
]

maps = []
for case in cases:
    print(case["title"])
    maps.append(distance_map(case["phi"], case["theta0"], case["t1"], case["t2"]))

# ---------------------------------------------------------------------------
# Figure: distance color map + iso-distance contours, two panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))

cmap = plt.get_cmap("coolwarm")
step = 0.1

for ax, case, dist in zip(axes, cases, maps):
    t1, t2 = case["t1"], case["t2"]
    theta0 = case["theta0"]

    levels_fill = np.linspace(0.0, dist.max(), 40)
    cf = ax.contourf(t1, t2, dist, levels=levels_fill, cmap=cmap)

    iso = np.arange(step, dist.max(), step)
    cs = ax.contour(t1, t2, dist, levels=iso, colors="k", linewidths=0.7,
                    alpha=0.75)
    ax.clabel(cs, iso[::3], inline=True, fmt=r"$d=%.1f$", fontsize=8)

    # reference point theta*
    ax.plot(theta0[0], theta0[1], "o", color="white", ms=9, mew=1.4,
            mec="black", zorder=5)
    ax.annotate(
        case["ref"],
        xy=(theta0[0], theta0[1]),
        xytext=case["arrow"],
        textcoords="data",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )

    ax.set_xlabel(r"$\theta_{1}$")
    ax.set_ylabel(r"$\theta_{2}$")
    ax.set_title(
        r"Fisher--Rao distance from $\theta_{\star}$, "
        + case["title"]
        + "\n"
        + r"$g_{\theta}=\mathrm{Cov}_{\theta}(\phi(X))$ on $E=[-2,2]$"
    )

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"Fisher--Rao distance $d(\theta,\theta_{\star})$")

fig.tight_layout()
path = OUT / "32_gibbs_fisher_rao_distance.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
for case, dist in zip(cases, maps):
    print(f"theta* = {case['theta0']}  dmax = {dist.max():.3f}")
# plt.show()  # uncomment for interactive display
