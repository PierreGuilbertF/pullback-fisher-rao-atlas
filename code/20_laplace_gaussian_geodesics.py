"""
Compare full Fisher--Rao geodesics for bivariate Laplace and Gaussian laws.

Both paths join the same endpoints:
    mu_0=(0,0), (sigma_1,sigma_2,theta)_0=(1,2,pi/4),
    mu_1=(2,2), (sigma_1,sigma_2,theta)_1=(2,1,3pi/4).

Coordinates: q=(mu_1,mu_2,sigma_1,sigma_2,theta).
The geodesics are approximated by discrete Riemannian energy minimization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

Q0 = np.array([0.0, 0.0, 1.0, 2.0, np.pi / 4])
Q1 = np.array([2.0, 2.0, 2.0, 1.0, 3.0 * np.pi / 4])
N_SEG = 36
N_MARK = 7


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sigma_from_params(sigma_1, sigma_2, theta):
    R = rotation(theta)
    return R @ np.diag([sigma_1**2, sigma_2**2]) @ R.T


def gaussian_metric(q):
    """Gaussian Fisher--Rao metric in (mu_1,mu_2,sigma_1,sigma_2,theta)."""
    _, _, sigma_1, sigma_2, theta = q
    sigma_1 = max(float(sigma_1), 1e-6)
    sigma_2 = max(float(sigma_2), 1e-6)

    G = np.zeros((5, 5))
    G[:2, :2] = np.linalg.inv(
        sigma_from_params(sigma_1, sigma_2, theta)
    )
    G[2, 2] = 2.0 / sigma_1**2
    G[3, 3] = 2.0 / sigma_2**2
    G[4, 4] = max(
        (sigma_1**2 - sigma_2**2) ** 2
        / (sigma_1**2 * sigma_2**2),
        1e-8,
    )
    return G


def laplace_metric(q):
    """Elliptic Laplace Fisher--Rao metric for n=2."""
    _, _, sigma_1, sigma_2, theta = q
    sigma_1 = max(float(sigma_1), 1e-6)
    sigma_2 = max(float(sigma_2), 1e-6)

    G = np.zeros((5, 5))

    # a_2=1/8 for the mean block.
    G[:2, :2] = (
        1.0
        / 8.0
        * np.linalg.inv(sigma_from_params(sigma_1, sigma_2, theta))
    )

    # Dispersion block:
    # (3/2) sum_i (d sigma_i / sigma_i)^2
    # - (1/4) (sum_i d sigma_i / sigma_i)^2.
    G[2, 2] = 5.0 / (4.0 * sigma_1**2)
    G[3, 3] = 5.0 / (4.0 * sigma_2**2)
    G[2, 3] = G[3, 2] = -1.0 / (4.0 * sigma_1 * sigma_2)

    # Orientation coefficient (n+1)/(n+2)=3/4.
    G[4, 4] = max(
        3.0
        / 4.0
        * (sigma_1**2 - sigma_2**2) ** 2
        / (sigma_1**2 * sigma_2**2),
        1e-8,
    )
    return G


def path_energy(interior_flat, q0, q1, metric):
    path = np.vstack([q0, interior_flat.reshape(-1, 5), q1])
    energy = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dq = b - a
        G = metric(0.5 * (a + b))
        energy += float(dq @ G @ dq)
    return energy


def compute_geodesic(metric, name, options):
    t = np.linspace(0.0, 1.0, N_SEG + 2)[1:-1]
    initial = np.outer(1.0 - t, Q0) + np.outer(t, Q1)
    initial[:, 2] = np.exp(
        (1.0 - t) * np.log(Q0[2]) + t * np.log(Q1[2])
    )
    initial[:, 3] = np.exp(
        (1.0 - t) * np.log(Q0[3]) + t * np.log(Q1[3])
    )

    bounds = []
    for _ in range(N_SEG):
        bounds.extend(
            [
                (None, None),
                (None, None),
                (0.2, 5.0),
                (0.2, 5.0),
                (0.0, np.pi),
            ]
        )

    result = minimize(
        path_energy,
        initial.ravel(),
        args=(Q0, Q1, metric),
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )
    print(
        f"{name}: energy={result.fun:.6f}, "
        f"success={result.success}, nit={result.nit}, message={result.message}"
    )
    return np.vstack([Q0, result.x.reshape(-1, 5), Q1])


laplace_path = compute_geodesic(
    laplace_metric,
    "Laplace",
    {"maxiter": 800, "maxfun": 100000, "ftol": 1e-10},
)
# Keep exactly the optimization parameters used by 11_geodesic_full.py, so
# the blue trajectory is the same one as in the Gaussian section.
gaussian_path = compute_geodesic(
    gaussian_metric,
    "Gauss",
    {"maxiter": 600, "ftol": 1e-11},
)
mark_indices = np.linspace(0, N_SEG + 1, N_MARK, dtype=int)


def draw_path(ax, path, color, label, annotate=False):
    sigma_1, sigma_2, theta = path[:, 2], path[:, 3], path[:, 4]
    ax.plot(sigma_1, sigma_2, theta, color=color, lw=2.5, label=label)
    for number, index in enumerate(mark_indices, start=1):
        ax.scatter(
            [sigma_1[index]],
            [sigma_2[index]],
            [theta[index]],
            color=color,
            edgecolor="white",
            linewidth=0.4,
            s=22,
            zorder=4,
        )
        if annotate:
            ax.text(
                sigma_1[index],
                sigma_2[index],
                theta[index],
                f"  {number}",
                fontsize=8,
            )


fig = plt.figure(figsize=(6.8, 5.5))
ax = fig.add_subplot(111, projection="3d")
draw_path(
    ax,
    gaussian_path,
    color="#3A7CA5",
    label="Gaussian",
    annotate=True,
)
draw_path(
    ax,
    laplace_path,
    color="#C44E52",
    label="elliptical Laplace",
)

ax.plot(
    [Q0[2]], [Q0[3]], [Q0[4]], "o",
    color="#4C78A8", ms=8, label="start", zorder=5,
)
ax.plot(
    [Q1[2]], [Q1[3]], [Q1[4]], "s",
    color="#55A868", ms=7, label="end", zorder=5,
)
ax.set_xlabel(r"$\sigma_{1}$")
ax.set_ylabel(r"$\sigma_{2}$")
ax.set_zlabel(r"$\theta$")
ax.set_title(
    r"Geodesics in $(\sigma_{1},\sigma_{2},\theta)$",
    pad=10,
)
ax.view_init(elev=18, azim=-55)
ax.set_zticks([np.pi / 4, np.pi / 2, 3.0 * np.pi / 4])
ax.set_zticklabels([r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"])
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.subplots_adjust(left=0.02, right=0.94, bottom=0.03, top=0.90)

output = OUT / "20_laplace_gaussian_geodesics.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"Wrote {output}")


# ----- density snapshots along both geodesics -----
# n = 2 normalization constant of the elliptic Laplace law.
C2 = 1.0 / (8.0 * np.pi)


def gaussian_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    quad = np.einsum(
        "...i,ij,...j->...", pos, np.linalg.inv(Sigma), pos
    )
    det = np.linalg.det(Sigma)
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


def laplace_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    quad = np.einsum(
        "...i,ij,...j->...", pos, np.linalg.inv(Sigma), pos
    )
    det = np.linalg.det(Sigma)
    r = np.sqrt(np.maximum(quad, 0.0))
    return C2 / np.sqrt(det) * np.exp(-0.5 * r)


xg = np.linspace(-3.5, 5.5, 220)
yg = np.linspace(-3.5, 5.5, 220)
X, Y = np.meshgrid(xg, yg)

ROWS = [
    ("Gaussian", gaussian_path, gaussian_pdf, "Blues", "#1b3a5c"),
    ("elliptical Laplace", laplace_path, laplace_pdf, "Reds", "#7b1f22"),
]

fig3, axes = plt.subplots(
    2, N_MARK, figsize=(14.0, 6.2), sharex=True, sharey=True
)
for row, (name, path, pdf, cmap, line_color) in enumerate(ROWS):
    for column, index in enumerate(mark_indices):
        mu_1, mu_2, sigma_1, sigma_2, theta = path[index]
        Sigma = sigma_from_params(sigma_1, sigma_2, theta)
        Z = pdf(X, Y, np.array([mu_1, mu_2]), Sigma)
        ax = axes[row, column]
        ax.contourf(X, Y, Z, levels=8, cmap=cmap)
        ax.contour(X, Y, Z, levels=8, colors=line_color, linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"{column + 1}\n"
            + rf"$\mu=({mu_1:.1f},{mu_2:.1f})$"
            + "\n"
            + rf"$\sigma=({sigma_1:.2f},{sigma_2:.2f})$"
            + "\n"
            + rf"$\theta={theta:.2f}$",
            fontsize=7,
        )
        ax.set_xlim(xg[0], xg[-1])
        ax.set_ylim(yg[0], yg[-1])
        if column == 0:
            ax.set_ylabel(f"{name}\n" + r"$x_{2}$", fontsize=9)
        if row == 1:
            ax.set_xlabel(r"$x_{1}$")

fig3.suptitle(
    r"Densities along the two geodesics: "
    r"Gaussian (top) and elliptical Laplace (bottom)",
    y=1.02,
    fontsize=12,
)
fig3.tight_layout()
output3 = OUT / "20_laplace_gaussian_densities.png"
fig3.savefig(output3, dpi=180, bbox_inches="tight")
print(f"Wrote {output3}")

for name, path, _, _, _ in ROWS:
    print(name)
    for number, index in enumerate(mark_indices, start=1):
        print(
            f"  {number}: mu=({path[index, 0]:.2f},{path[index, 1]:.2f}), "
            f"sigma=({path[index, 2]:.2f},{path[index, 3]:.2f}), "
            f"theta={path[index, 4]:.2f}"
        )
# plt.show()
