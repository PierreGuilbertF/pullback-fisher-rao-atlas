"""
Fisher--Rao geodesic for a pure orientation change:
    N(0, Σ_{π/4}) → N(0, Σ_{3π/4}),
with fixed principal variances (1, 4), i.e. (σ1, σ2) = (1, 2), and μ = 0.

With only θ free, the metric collapses to
    ds² = ((σ1² - σ2²)² / (σ1² σ2²)) dθ²,
a constant multiple of dθ². Geodesics are therefore affine in θ: constant
coordinate speed, and constant Riemannian speed.

Figures:
  10_geodesic_orientation_densities.png — density snapshots
  10_geodesic_orientation_angle.png     — θ(t) and Riemannian speed
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

SIG1, SIG2 = 1.0, 2.0  # variances 1 and 4
THETA_A = np.pi / 4
THETA_B = 3.0 * np.pi / 4
MU = np.zeros(2)
N_PATH = 200
N_SNAP = 7

# G_θθ is constant along this slice
G_THETA = (SIG1**2 - SIG2**2) ** 2 / (SIG1**2 * SIG2**2)


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


t = np.linspace(0.0, 1.0, N_PATH)
theta = THETA_A + t * (THETA_B - THETA_A)
dtheta_dt = np.full_like(t, THETA_B - THETA_A)
speed = np.sqrt(G_THETA) * np.abs(dtheta_dt)
snap_idx = np.linspace(0, N_PATH - 1, N_SNAP, dtype=int)

print(f"G_θθ = {G_THETA:.4f}, Riemannian speed = {speed[0]:.4f} (constant)")

# ----- densities -----
lim = 4.5
xg = np.linspace(-lim, lim, 220)
yg = np.linspace(-lim, lim, 220)
X, Y = np.meshgrid(xg, yg)

fig1, axes = plt.subplots(1, N_SNAP, figsize=(14.0, 2.7), sharex=True, sharey=True)
for k, i in enumerate(snap_idx):
    Sig = sigma_from_params(SIG1, SIG2, theta[i])
    Z = gaussian_pdf(X, Y, MU, Sig)
    ax = axes[k]
    ax.contourf(X, Y, Z, levels=8, cmap="Blues")
    ax.contour(X, Y, Z, levels=8, colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{k+1}\n" + rf"$\theta={theta[i]:.2f}$",
        fontsize=8,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if k == 0:
        ax.set_ylabel(r"$x_{2}$")
    ax.set_xlabel(r"$x_{1}$")

fig1.suptitle(
    r"Densities along the orientation geodesic "
    r"($\theta:\pi/4\to 3\pi/4$, variances $(1,4)$)",
    y=1.06,
    fontsize=12,
)
fig1.tight_layout()
p1 = OUT / "10_geodesic_orientation_densities.png"
fig1.savefig(p1, dpi=180, bbox_inches="tight")
print(f"Wrote {p1}")

# ----- angle path and speed -----
fig2, (ax_th, ax_sp) = plt.subplots(1, 2, figsize=(10.0, 3.6))

ax_th.plot(t, theta, color="#C44E52", lw=2.2)
for k, i in enumerate(snap_idx):
    ax_th.plot(t[i], theta[i], "o", color="0.2", ms=5)
    ax_th.annotate(str(k + 1), (t[i], theta[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax_th.axhline(THETA_A, color="0.7", ls=":", lw=0.8)
ax_th.axhline(THETA_B, color="0.7", ls=":", lw=0.8)
ax_th.set_xlabel(r"$t$")
ax_th.set_ylabel(r"$\theta(t)$")
ax_th.set_title(r"Angle along the geodesic")
ax_th.set_yticks([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
ax_th.set_yticklabels([r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"])

ax_sp.plot(t, speed, color="#3A7CA5", lw=2.2)
ax_sp.set_xlabel(r"$t$")
ax_sp.set_ylabel(r"$\sqrt{g_{\theta\theta}}\,|\dot\theta|$")
ax_sp.set_title(r"Riemannian speed (constant)")
ax_sp.set_ylim(0.0, 1.2 * speed[0])

fig2.suptitle(
    r"With $\sigma_{1},\sigma_{2}$ fixed, $g_{\theta\theta}$ is constant: "
    r"the geodesic advances at uniform angular speed",
    y=1.02,
    fontsize=11,
)
fig2.tight_layout()
p2 = OUT / "10_geodesic_orientation_angle.png"
fig2.savefig(p2, dpi=180, bbox_inches="tight")
print(f"Wrote {p2}")
# plt.show()
