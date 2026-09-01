"""
Fisher--Rao geodesic between two centered bivariate Gaussians that share the
same orientation but swap their principal stretches:
    N(0, Σ1) → N(0, Σ2),
with θ = π/4 fixed, principal variances (1, 4) → (4, 1), i.e. (σ1, σ2): (1, 2) → (2, 1).

With μ and θ held fixed, the metric collapses to
    ds² = 2 (dσ1²/σ1² + dσ2²/σ2²),
so geodesics are straight lines in (log σ1, log σ2): σ_i(t) = σ_i(0)^{1-t} σ_i(1)^t.

Figures:
  09_geodesic_sigma_plane.png  — path in the (σ1, σ2) plane
  09_geodesic_sigma_densities.png — density snapshots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

THETA = np.pi / 4
# stds: variances (1, 4) → (4, 1)
SIG_A = np.array([1.0, 2.0])
SIG_B = np.array([2.0, 1.0])
MU = np.zeros(2)
N_PATH = 200
N_SNAP = 7


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sigma_from_params(sig1, sig2, theta):
    R = rotation(theta)
    return R @ np.diag([sig1**2, sig2**2]) @ R.T


def gaussian_pdf(X, Y, mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    quad = np.einsum("...i,ij,...j->...", pos, inv, pos)
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


# Closed-form geodesic in scale coordinates
t = np.linspace(0.0, 1.0, N_PATH)
log_sig = (1.0 - t)[:, None] * np.log(SIG_A) + t[:, None] * np.log(SIG_B)
sig = np.exp(log_sig)  # (N_PATH, 2)
snap_idx = np.linspace(0, N_PATH - 1, N_SNAP, dtype=int)

# ----- figure 1: (σ1, σ2) plane -----
fig1, ax1 = plt.subplots(figsize=(5.6, 5.2))
ax1.plot(sig[:, 0], sig[:, 1], color="#C44E52", lw=2.2, label="geodesic")
# Euclidean segment in (σ1, σ2) for comparison
ax1.plot(
    [SIG_A[0], SIG_B[0]],
    [SIG_A[1], SIG_B[1]],
    ls="--",
    color="0.6",
    lw=1.0,
    label="Euclidean segment",
)
ax1.plot(*SIG_A, "o", color="#4C78A8", ms=9, zorder=5, label=r"start $(1,\ 2)$")
ax1.plot(*SIG_B, "s", color="#55A868", ms=8, zorder=5, label=r"end $(2,\ 1)$")
for k, i in enumerate(snap_idx):
    ax1.plot(sig[i, 0], sig[i, 1], "o", color="0.2", ms=4, zorder=4)
    ax1.annotate(
        str(k + 1),
        (sig[i, 0], sig[i, 1]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )
ax1.set_aspect("equal")
ax1.set_xlabel(r"$\sigma_{1}$")
ax1.set_ylabel(r"$\sigma_{2}$")
ax1.set_title(
    r"Geodesic in the $(\sigma_{1},\sigma_{2})$ plane"
    "\n"
    r"($\mu=0$, $\theta=\pi/4$ fixed)"
)
ax1.legend(loc="best", fontsize=8, frameon=False)
ax1.set_xlim(0.85, 2.15)
ax1.set_ylim(0.85, 2.15)
fig1.tight_layout()
p1 = OUT / "09_geodesic_sigma_plane.png"
fig1.savefig(p1, dpi=180, bbox_inches="tight")
print(f"Wrote {p1}")
print(f"  product σ1·σ2 along path: min={np.min(sig[:,0]*sig[:,1]):.4f}, "
      f"max={np.max(sig[:,0]*sig[:,1]):.4f} (exact geodesic: constant = 2)")

# ----- figure 2: density snapshots -----
lim = 4.5
xg = np.linspace(-lim, lim, 220)
yg = np.linspace(-lim, lim, 220)
X, Y = np.meshgrid(xg, yg)

fig2, axes = plt.subplots(1, N_SNAP, figsize=(14.0, 2.7), sharex=True, sharey=True)
for k, i in enumerate(snap_idx):
    Sig = sigma_from_params(sig[i, 0], sig[i, 1], THETA)
    Z = gaussian_pdf(X, Y, MU, Sig)
    ax = axes[k]
    ax.contourf(X, Y, Z, levels=8, cmap="Blues")
    ax.contour(X, Y, Z, levels=8, colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{k+1}\n" + rf"$\sigma=({sig[i,0]:.2f},{sig[i,1]:.2f})$",
        fontsize=8,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if k == 0:
        ax.set_ylabel(r"$x_{2}$")
    ax.set_xlabel(r"$x_{1}$")

fig2.suptitle(
    r"Densities along the geodesic "
    r"(principal variances $(1,4)\to(4,1)$, $\theta=\pi/4$)",
    y=1.06,
    fontsize=12,
)
fig2.tight_layout()
p2 = OUT / "09_geodesic_sigma_densities.png"
fig2.savefig(p2, dpi=180, bbox_inches="tight")
print(f"Wrote {p2}")
# plt.show()
