"""
Fisher--Rao geodesic with everything moving at once:
    N((0,0), Σ1) → N((2,2), Σ2),
Σ1: principal variances (1, 4), θ = π/4
Σ2: principal variances (4, 1), θ = 3π/4

Coordinates q = (μ1, μ2, σ1, σ2, θ). Discrete energy minimization.

Figures:
  11_geodesic_full_densities.png — density snapshots along the path
  11_geodesic_full_sigma.png     — path in (σ1, σ2, θ)
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
N_SNAP = 7


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sigma_from_params(sig1, sig2, theta):
    R = rotation(theta)
    return R @ np.diag([sig1**2, sig2**2]) @ R.T


def metric_matrix(q):
    _, _, s1, s2, th = q
    s1 = max(float(s1), 1e-6)
    s2 = max(float(s2), 1e-6)
    G = np.zeros((5, 5))
    G[:2, :2] = np.linalg.inv(sigma_from_params(s1, s2, th))
    G[2, 2] = 2.0 / (s1**2)
    G[3, 3] = 2.0 / (s2**2)
    G[4, 4] = max((s1**2 - s2**2) ** 2 / (s1**2 * s2**2), 1e-8)
    return G


def path_energy(interior_flat, q0, q1):
    mid = interior_flat.reshape(-1, 5)
    path = np.vstack([q0, mid, q1])
    e = 0.0
    for i in range(len(path) - 1):
        dq = path[i + 1] - path[i]
        g = metric_matrix(0.5 * (path[i] + path[i + 1]))
        e += float(dq @ g @ dq)
    return e


def compute_geodesic():
    t = np.linspace(0.0, 1.0, N_SEG + 2)[1:-1]
    # linear in μ and θ; log-linear in σ
    interior0 = np.outer(1.0 - t, Q0) + np.outer(t, Q1)
    interior0[:, 2] = np.exp((1.0 - t) * np.log(Q0[2]) + t * np.log(Q1[2]))
    interior0[:, 3] = np.exp((1.0 - t) * np.log(Q0[3]) + t * np.log(Q1[3]))

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

    res = minimize(
        path_energy,
        interior0.ravel(),
        args=(Q0, Q1),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=600, ftol=1e-11),
    )
    path = np.vstack([Q0, res.x.reshape(-1, 5), Q1])
    print(f"geodesic energy={res.fun:.6f}, success={res.success}, nit={res.nit}")
    return path


def gaussian_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    quad = np.einsum("...i,ij,...j->...", pos, inv, pos)
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


path = compute_geodesic()
snap_idx = np.linspace(0, len(path) - 1, N_SNAP, dtype=int)

xg = np.linspace(-3.5, 5.5, 220)
yg = np.linspace(-3.5, 5.5, 220)
X, Y = np.meshgrid(xg, yg)

fig, axes = plt.subplots(1, N_SNAP, figsize=(14.0, 2.9), sharex=True, sharey=True)
for k, i in enumerate(snap_idx):
    mu1, mu2, s1, s2, th = path[i]
    Sig = sigma_from_params(s1, s2, th)
    Z = gaussian_pdf(X, Y, np.array([mu1, mu2]), Sig)
    ax = axes[k]
    ax.contourf(X, Y, Z, levels=8, cmap="Blues")
    ax.contour(X, Y, Z, levels=8, colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{k+1}\n"
        + rf"$\mu=({mu1:.1f},{mu2:.1f})$"
        + "\n"
        + rf"$\sigma=({s1:.2f},{s2:.2f})$"
        + "\n"
        + rf"$\theta={th:.2f}$",
        fontsize=7,
    )
    ax.set_xlim(xg[0], xg[-1])
    ax.set_ylim(yg[0], yg[-1])
    if k == 0:
        ax.set_ylabel(r"$x_{2}$")
    ax.set_xlabel(r"$x_{1}$")

fig.suptitle(
    r"Everything moves at once: "
    r"$\mathcal{N}((0,0),\Sigma_{1})\to\mathcal{N}((2,2),\Sigma_{2})$",
    y=1.08,
    fontsize=12,
)
fig.tight_layout()
out = OUT / "11_geodesic_full_densities.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Wrote {out}")

# ----- (σ1, σ2, θ) path -----
sig1, sig2, theta = path[:, 2], path[:, 3], path[:, 4]
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
ax2.set_title(r"Path of $\Sigma$ in $(\sigma_{1},\sigma_{2},\theta)$")
ax2.view_init(elev=18, azim=-55)
ax2.set_zticks([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
ax2.set_zticklabels([r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"])
ax2.legend(loc="upper left", fontsize=8, frameon=False)
fig2.tight_layout()
out2 = OUT / "11_geodesic_full_sigma.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Wrote {out2}")

for k, i in enumerate(snap_idx):
    print(f"  {k+1}: μ=({path[i,0]:.2f},{path[i,1]:.2f}), "
          f"σ=({path[i,2]:.2f},{path[i,3]:.2f}), θ={path[i,4]:.2f}")
# plt.show()
