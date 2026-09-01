"""
Fisher--Rao geodesic between two bivariate Gaussians with the same covariance
but different means:
    N((0,0), Σ)  →  N((2,2), Σ),
with Σ rotated by π/4 and principal variances (1, 4).

The geodesic is computed by minimizing the discrete Riemannian energy in
coordinates (μ1, μ2, σ1, σ2); θ is held at π/4 (the endpoints share this
orientation, and the mean displacement follows a principal axis).

Figures:
  08_geodesic_mu.png         — path of μ in R²
  08_geodesic_sigma.png      — path of (σ1, σ2, θ) in 3D
  08_geodesic_densities.png  — density snapshots along the geodesic
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

THETA0 = np.pi / 4
SIG1_0, SIG2_0 = 1.0, 2.0  # stds; variances 1 and 4
MU_A = np.array([0.0, 0.0])
MU_B = np.array([2.0, 2.0])
N_SEG = 40
N_SNAP = 7


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sigma_from_params(sig1, sig2, theta):
    R = rotation(theta)
    return R @ np.diag([sig1**2, sig2**2]) @ R.T


def metric_matrix(mu1, mu2, s1, s2, th):
    """Fisher--Rao metric in coordinates (μ1, μ2, σ1, σ2, θ)."""
    s1 = max(float(s1), 1e-6)
    s2 = max(float(s2), 1e-6)
    G = np.zeros((5, 5))
    G[:2, :2] = np.linalg.inv(sigma_from_params(s1, s2, th))
    G[2, 2] = 2.0 / (s1**2)
    G[3, 3] = 2.0 / (s2**2)
    G[4, 4] = max((s1**2 - s2**2) ** 2 / (s1**2 * s2**2), 1e-8)
    return G


def path_energy(interior_flat, q0, q1):
    """Discrete energy; interior holds (μ1, μ2, σ1, σ2), θ frozen at THETA0."""
    mid = interior_flat.reshape(-1, 4)
    path4 = np.vstack([q0[:4], mid, q1[:4]])
    e = 0.0
    for i in range(len(path4) - 1):
        a = path4[i]
        b = path4[i + 1]
        dq = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3], 0.0])
        mid_pt = 0.5 * (a + b)
        g = metric_matrix(mid_pt[0], mid_pt[1], mid_pt[2], mid_pt[3], THETA0)
        e += float(dq @ g @ dq)
    return e


def compute_geodesic():
    q0 = np.array([MU_A[0], MU_A[1], SIG1_0, SIG2_0, THETA0])
    q1 = np.array([MU_B[0], MU_B[1], SIG1_0, SIG2_0, THETA0])
    t = np.linspace(0.0, 1.0, N_SEG + 2)[1:-1]
    interior0 = np.outer(1.0 - t, q0[:4]) + np.outer(t, q1[:4])
    # seed a smooth bump in both standard deviations
    bump = np.sin(np.pi * t)
    interior0[:, 2] += 0.5 * bump
    interior0[:, 3] += 0.15 * bump

    bounds = []
    for _ in range(N_SEG):
        bounds.extend([(None, None), (None, None), (0.2, 5.0), (0.2, 5.0)])

    res = minimize(
        path_energy,
        interior0.ravel(),
        args=(q0, q1),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=800, ftol=1e-12),
    )
    mid = res.x.reshape(-1, 4)
    path = np.column_stack(
        [
            np.concatenate([[q0[0]], mid[:, 0], [q1[0]]]),
            np.concatenate([[q0[1]], mid[:, 1], [q1[1]]]),
            np.concatenate([[q0[2]], mid[:, 2], [q1[2]]]),
            np.concatenate([[q0[3]], mid[:, 3], [q1[3]]]),
            np.full(N_SEG + 2, THETA0),
        ]
    )
    print(f"geodesic energy={res.fun:.6f}, success={res.success}, nit={res.nit}")
    return path


def gaussian_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    quad = np.einsum("...i,ij,...j->...", pos, inv, pos)
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


path = compute_geodesic()
mu = path[:, :2]
sig1, sig2, theta = path[:, 2], path[:, 3], path[:, 4]
snap_idx = np.linspace(0, len(path) - 1, N_SNAP, dtype=int)

# ----- figure 1: μ-path -----
fig1, ax1 = plt.subplots(figsize=(5.6, 5.2))
ax1.plot(mu[:, 0], mu[:, 1], color="#C44E52", lw=2.2, label="geodesic")
ax1.plot(
    [MU_A[0], MU_B[0]],
    [MU_A[1], MU_B[1]],
    ls="--",
    color="0.6",
    lw=1.0,
    label=r"Euclidean segment ($\Sigma$ fixed)",
)
ax1.plot(*MU_A, "o", color="#4C78A8", ms=9, zorder=5, label=r"start $(0,0)$")
ax1.plot(*MU_B, "s", color="#55A868", ms=8, zorder=5, label=r"end $(2,2)$")
for k, i in enumerate(snap_idx):
    ax1.plot(mu[i, 0], mu[i, 1], "o", color="0.2", ms=4, zorder=4)
    ax1.annotate(str(k + 1), mu[i], textcoords="offset points", xytext=(4, 4), fontsize=8)
ax1.set_aspect("equal")
ax1.set_xlabel(r"$\mu_{1}$")
ax1.set_ylabel(r"$\mu_{2}$")
ax1.set_title(r"Geodesic in the plane of means $(\mu_{1},\mu_{2})$")
ax1.legend(loc="best", fontsize=8, frameon=False)
fig1.tight_layout()
p1 = OUT / "08_geodesic_mu.png"
fig1.savefig(p1, dpi=180, bbox_inches="tight")
print(f"Wrote {p1}")

# ----- figure 2: (σ1, σ2, θ) -----
fig2 = plt.figure(figsize=(6.6, 5.4))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot(sig1, sig2, theta, color="#3A7CA5", lw=2.4)
ax2.plot([sig1[0]], [sig2[0]], [theta[0]], "o", color="#4C78A8", ms=8, label="start")
ax2.plot([sig1[-1]], [sig2[-1]], [theta[-1]], "s", color="#55A868", ms=7, label="end")
for k, i in enumerate(snap_idx):
    ax2.scatter([sig1[i]], [sig2[i]], [theta[i]], color="0.2", s=18)
    ax2.text(sig1[i], sig2[i], theta[i], f"  {k+1}", fontsize=8)
ax2.set_xlabel(r"$\sigma_{1}$")
ax2.set_ylabel(r"$\sigma_{2}$")
ax2.set_zlabel(r"$\theta$")
ax2.set_title(r"Parameters of $\Sigma$ along the geodesic")
ax2.view_init(elev=18, azim=-60)
# zoom on the relevant σ-range so the bump is readable
ax2.set_xlim(0.9, 1.05 * sig1.max())
ax2.set_ylim(1.9, 1.05 * max(sig2.max(), 2.05))
ax2.set_zlim(THETA0 - 0.15, THETA0 + 0.15)
fig2.tight_layout()
p2 = OUT / "08_geodesic_sigma.png"
fig2.savefig(p2, dpi=180, bbox_inches="tight")
print(f"Wrote {p2}")
print(
    f"  σ1: {sig1[0]:.3f} → max {sig1.max():.3f} → {sig1[-1]:.3f}\n"
    f"  σ2: {sig2[0]:.3f} → max {sig2.max():.3f} → {sig2[-1]:.3f}"
)

# ----- figure 3: density snapshots -----
xg = np.linspace(-3.5, 5.5, 220)
yg = np.linspace(-3.5, 5.5, 220)
X, Y = np.meshgrid(xg, yg)

fig3, axes = plt.subplots(1, N_SNAP, figsize=(14.0, 2.7), sharex=True, sharey=True)
for k, i in enumerate(snap_idx):
    Sig = sigma_from_params(sig1[i], sig2[i], theta[i])
    Z = gaussian_pdf(X, Y, mu[i], Sig)
    ax = axes[k]
    ax.contourf(X, Y, Z, levels=8, cmap="Blues")
    ax.contour(X, Y, Z, levels=8, colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{k+1}\n"
        + rf"$\sigma=({sig1[i]:.2f},{sig2[i]:.2f})$",
        fontsize=8,
    )
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(yg[0], yg[-1])
    if k == 0:
        ax.set_ylabel(r"$x_{2}$")
    ax.set_xlabel(r"$x_{1}$")

fig3.suptitle(
    r"Densities along the geodesic "
    r"$\mathcal{N}((0,0),\Sigma)\to\mathcal{N}((2,2),\Sigma)$",
    y=1.06,
    fontsize=12,
)
fig3.tight_layout()
p3 = OUT / "08_geodesic_densities.png"
fig3.savefig(p3, dpi=180, bbox_inches="tight")
print(f"Wrote {p3}")
# plt.show()
