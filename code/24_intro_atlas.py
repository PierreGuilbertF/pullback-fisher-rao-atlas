"""
Introduction figure: three univariate families and their Fisher--Rao
iso-distance maps in parameter space.

  1. Univariate Gaussian N(0, 1/2), closed-form hyperbolic distance.
  2. Linear Gibbs, phi(x)=(x, x^2), around theta=(-0.4, 0.55).
  3. Nonlinear Gibbs, U=cos(x theta1 + x^2 theta2), around theta=(0.7, 0.45).

Row 1: densities. Row 2: parameter plane, colormap + iso-distance contours.
For Gibbs families the plotted distance is the Riemannian length of the
Euclidean chord in parameter space (an upper bound on the geodesic
distance), using g_theta = Cov(phi) or Cov(dU).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR = "#4C78A8"


def gaussian_density(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def fisher_rao_gaussian(mu, sigma, mu0, sigma0):
    arg = (mu - mu0) ** 2 + 2.0 * (sigma**2 + sigma0**2)
    arg = arg / (4.0 * sigma * sigma0)
    arg = np.maximum(arg, 1.0)
    return np.sqrt(2.0) * np.arccosh(arg)


def normalize(x, unnorm):
    return unnorm / float(np.trapz(unnorm, x))


def cov_features(thetas, features):
    """Fisher metric g = Cov_theta(features) on a (n1, n2) parameter grid."""
    energy = np.einsum("xi,abi->abx", features, thetas)
    logw = -energy
    logw -= logw.max(axis=-1, keepdims=True)
    weights = np.exp(logw)
    weights /= weights.sum(axis=-1, keepdims=True)
    mean = np.einsum("abx,xi->abi", weights, features)
    centered = features[None, None, :, :] - mean[:, :, None, :]
    metric = np.einsum("abx,abxi,abxj->abij", weights, centered, centered)
    metric = 0.5 * (metric + np.swapaxes(metric, -1, -2))
    return metric + 1e-10 * np.eye(2)


def nonlinear_metric_grid(thetas, x):
    """g_ij = Cov_theta(dU/d theta_i, dU/d theta_j) for U=cos(x t1 + x^2 t2)."""
    phase = thetas[:, :, 0, None] * x + thetas[:, :, 1, None] * (x**2)
    energy = np.cos(phase)
    sine = np.sin(phase)
    xi = np.stack((-sine * x, -sine * (x**2)), axis=-1)
    logw = -energy
    logw -= logw.max(axis=-1, keepdims=True)
    weights = np.exp(logw)
    weights /= weights.sum(axis=-1, keepdims=True)
    mean = np.einsum("abx,abxi->abi", weights, xi)
    centered = xi - mean[:, :, None, :]
    metric = np.einsum("abx,abxi,abxj->abij", weights, centered, centered)
    metric = 0.5 * (metric + np.swapaxes(metric, -1, -2))
    return metric + 1e-10 * np.eye(2)


def chord_distance_map(t1, t2, metric, theta0, n_quad=18):
    """Riemannian length of the Euclidean segment from theta0 to each grid point.

    This is an upper bound on the geodesic distance. It is smooth (no grid
    Dijkstra octagons) and already shows how g_theta warps the plane.
    """
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    dtheta = np.stack([T1 - theta0[0], T2 - theta0[1]], axis=-1)
    g00 = RegularGridInterpolator(
        (t1, t2), metric[:, :, 0, 0], bounds_error=False, fill_value=None
    )
    g01 = RegularGridInterpolator(
        (t1, t2), metric[:, :, 0, 1], bounds_error=False, fill_value=None
    )
    g11 = RegularGridInterpolator(
        (t1, t2), metric[:, :, 1, 1], bounds_error=False, fill_value=None
    )
    ts = np.linspace(0.0, 1.0, n_quad)
    speeds = []
    pts_shape = dtheta.shape[:2]
    for t in ts:
        pts = np.stack(
            [theta0[0] + t * dtheta[:, :, 0], theta0[1] + t * dtheta[:, :, 1]],
            axis=-1,
        ).reshape(-1, 2)
        a = g00(pts).reshape(pts_shape)
        b = g01(pts).reshape(pts_shape)
        c = g11(pts).reshape(pts_shape)
        quad = (
            a * dtheta[:, :, 0] ** 2
            + 2.0 * b * dtheta[:, :, 0] * dtheta[:, :, 1]
            + c * dtheta[:, :, 1] ** 2
        )
        speeds.append(np.sqrt(np.maximum(quad, 0.0)))
    return np.trapz(np.stack(speeds, axis=0), ts, axis=0)


def draw_density(ax, x, p, title, ylabel=None):
    ax.fill_between(x, p, color=COLOR, alpha=0.35)
    ax.plot(x, p, color=COLOR, lw=2.0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$x$")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def draw_distance(fig, ax, t1, t2, dist, theta0, xlabel, ylabel, cbar_label):
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    finite = dist[np.isfinite(dist)]
    dmax = float(np.nanmax(finite))
    fill_levels = np.linspace(0.0, dmax, 36)
    cf = ax.contourf(T1, T2, dist, levels=fill_levels, cmap="coolwarm")
    if dmax > 3.0:
        step = 0.5
        fmt = r"$d=%.1f$"
    elif dmax > 1.2:
        step = 0.25
        fmt = r"$d=%.2f$"
    else:
        step = 0.1
        fmt = r"$d=%.1f$"
    iso = np.arange(step, dmax, step)
    if len(iso) == 0:
        iso = np.array([0.5 * dmax])
    cs = ax.contour(T1, T2, dist, levels=iso, colors="k", linewidths=0.65, alpha=0.75)
    ax.clabel(cs, iso[::2], inline=True, fmt=fmt, fontsize=7)
    ax.plot(theta0[0], theta0[1], "o", color="white", ms=8, mew=1.2, mec="black", zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# 1. Gaussian N(0, 1/2)
# ---------------------------------------------------------------------------
mu0, sig0 = 0.0, 0.5
x_g = np.linspace(-2.5, 2.5, 600)
p_g = gaussian_density(x_g, mu0, sig0)

mu = np.linspace(-2.0, 2.0, 280)
sigma = np.linspace(0.14, 1.7, 240)
MU, SIG = np.meshgrid(mu, sigma, indexing="ij")
dist_g = fisher_rao_gaussian(MU, SIG, mu0, sig0)

# ---------------------------------------------------------------------------
# 2. Linear Gibbs, phi = (x, x^2)
# phi=(x^2, cos x) on a short interval is nearly affinely dependent
# (cos x ≈ 1 - x^2/2), so Cov_theta(phi) is almost rank-1.
# ---------------------------------------------------------------------------
theta_lin = np.array([-0.4, 0.55])
x_lin = np.linspace(-3.0, 3.0, 600)
phi_lin = np.column_stack([x_lin, x_lin**2])
p_lin = normalize(
    x_lin, np.exp(-(theta_lin[0] * x_lin + theta_lin[1] * x_lin**2))
)

t1_lin = np.linspace(-1.30, 0.50, 90)
t2_lin = np.linspace(0.18, 1.25, 90)
T1l, T2l = np.meshgrid(t1_lin, t2_lin, indexing="ij")
thetas_lin = np.stack([T1l, T2l], axis=-1)
G_lin = cov_features(thetas_lin, phi_lin)
dist_lin = chord_distance_map(t1_lin, t2_lin, G_lin, theta_lin)
i0 = int(np.argmin(np.abs(t1_lin - theta_lin[0])))
j0 = int(np.argmin(np.abs(t2_lin - theta_lin[1])))
print("linear Gibbs cond(G(theta0)) =", float(np.linalg.cond(G_lin[i0, j0])))

# ---------------------------------------------------------------------------
# 3. Nonlinear Gibbs, U = cos(x theta1 + x^2 theta2)
# ---------------------------------------------------------------------------
theta_nl = np.array([0.7, 0.45])
x_nl = np.linspace(-3.0, 3.0, 700)
p_nl = normalize(x_nl, np.exp(-np.cos(x_nl * theta_nl[0] + x_nl**2 * theta_nl[1])))

t1_nl = np.linspace(0.20, 1.20, 90)
t2_nl = np.linspace(0.05, 0.90, 90)
T1n, T2n = np.meshgrid(t1_nl, t2_nl, indexing="ij")
thetas_nl = np.stack([T1n, T2n], axis=-1)
G_nl = nonlinear_metric_grid(thetas_nl, x_nl)
dist_nl = chord_distance_map(t1_nl, t2_nl, G_nl, theta_nl)

print(
    "distance maxima:",
    f"gauss={dist_g.max():.3f}",
    f"lin={np.nanmax(dist_lin):.3f}",
    f"nl={np.nanmax(dist_nl):.3f}",
)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    3,
    figsize=(12.6, 7.1),
    gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.42, "wspace": 0.34},
)

draw_density(
    axes[0, 0],
    x_g,
    p_g,
    rf"Gaussian $\mathcal{{N}}(0,\ {sig0:g})$",
    ylabel=r"$p(x)$",
)
draw_density(
    axes[0, 1],
    x_lin,
    p_lin,
    r"Linear Gibbs, $\phi=(x,x^{2})$"
    + "\n"
    + rf"$\theta=({theta_lin[0]:g},\ {theta_lin[1]:g})$",
)
draw_density(
    axes[0, 2],
    x_nl,
    p_nl,
    r"Nonlinear Gibbs, $U=\cos(x\theta_{1}+x^{2}\theta_{2})$"
    + "\n"
    + rf"$\theta=({theta_nl[0]:g},\ {theta_nl[1]:g})$",
)

draw_distance(
    fig,
    axes[1, 0],
    mu,
    sigma,
    dist_g,
    (mu0, sig0),
    r"$\mu$",
    r"$\sigma$",
    r"$d((\mu,\sigma),(0,1/2))$",
)
draw_distance(
    fig,
    axes[1, 1],
    t1_lin,
    t2_lin,
    dist_lin,
    theta_lin,
    r"$\theta_{1}$",
    r"$\theta_{2}$",
    r"$d(\theta,\theta_{0})$",
)
draw_distance(
    fig,
    axes[1, 2],
    t1_nl,
    t2_nl,
    dist_nl,
    theta_nl,
    r"$\theta_{1}$",
    r"$\theta_{2}$",
    r"$d(\theta,\theta_{0})$",
)

path = OUT / "24_intro_atlas.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
