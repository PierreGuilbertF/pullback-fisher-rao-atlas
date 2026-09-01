"""
Geodesic of the univariate Fisher--Rao metric between two Gaussians.

Metric: ds² = (dμ² + 2 dσ²)/σ², isometric (up to √2) to the Poincaré half-plane
in coordinates (ξ, η) = (μ/√2, σ). Geodesics are semicircles orthogonal to σ = 0
(or vertical lines). Connecting N(μ_a, σ) to N(μ_b, σ) with μ_a ≠ μ_b, the path
climbs in σ first, then translates in μ, then comes back down — the classic
"enlarge then shift" picture.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# Endpoints at equal σ: the geodesic must bulge upward in σ.
MU_A, SIG_A = -2.0, 1.0
MU_B, SIG_B = 2.0, 1.0
N_PATH = 400
N_SNAPSHOTS = 7


def to_poincare(mu, sigma):
    return mu / np.sqrt(2.0), sigma


def from_poincare(xi, eta):
    return np.sqrt(2.0) * xi, eta


def geodesic_semicircle(mu_a, sig_a, mu_b, sig_b, n=N_PATH):
    """Unit-speed-ish semicircle geodesic in the Poincaré chart (ξ, η)."""
    xi_a, eta_a = to_poincare(mu_a, sig_a)
    xi_b, eta_b = to_poincare(mu_b, sig_b)

    if np.isclose(xi_a, xi_b):
        eta = np.linspace(eta_a, eta_b, n)
        xi = np.full_like(eta, xi_a)
        return from_poincare(xi, eta)

    # Circle centered on the ξ-axis, through both endpoints.
    center = (xi_a**2 + eta_a**2 - xi_b**2 - eta_b**2) / (2.0 * (xi_a - xi_b))
    radius = np.sqrt((xi_a - center) ** 2 + eta_a**2)

    theta_a = np.arctan2(eta_a, xi_a - center)
    theta_b = np.arctan2(eta_b, xi_b - center)
    # Stay in the upper half-plane: take the arc with θ ∈ (0, π).
    if theta_a < 0:
        theta_a += 2.0 * np.pi
    if theta_b < 0:
        theta_b += 2.0 * np.pi
    # Prefer the short upper arc between the two angles in (0, π).
    theta_a = np.arctan2(eta_a, xi_a - center)
    theta_b = np.arctan2(eta_b, xi_b - center)
    theta = np.linspace(theta_a, theta_b, n)

    xi = center + radius * np.cos(theta)
    eta = radius * np.sin(theta)
    return from_poincare(xi, eta)


def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


mu_path, sig_path = geodesic_semicircle(MU_A, SIG_A, MU_B, SIG_B)
snap_idx = np.linspace(0, len(mu_path) - 1, N_SNAPSHOTS, dtype=int)

fig = plt.figure(figsize=(11.0, 4.6), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

# --- left: geodesic in parameter space ---
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(mu_path, sig_path, color="#C44E52", lw=2.2,     label="geodesic")
ax0.plot(MU_A, SIG_A, "o", color="#4C78A8", ms=9, zorder=5, label=r"start $\mathcal{N}(-2,1)$")
ax0.plot(MU_B, SIG_B, "s", color="#55A868", ms=8, zorder=5, label=r"end $\mathcal{N}(2,1)$")
for k, i in enumerate(snap_idx):
    ax0.plot(mu_path[i], sig_path[i], "o", color="0.25", ms=4.5, zorder=4)
    ax0.annotate(
        str(k + 1),
        (mu_path[i], sig_path[i]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )
ax0.axhline(SIG_A, color="0.7", ls=":", lw=0.9)
ax0.set_xlabel(r"$\mu$")
ax0.set_ylabel(r"$\sigma$")
ax0.set_title(r"Geodesic in $(\mu,\sigma)$")
ax0.legend(loc="upper right", fontsize=8, frameon=False)
ax0.set_xlim(-3.2, 3.2)
ax0.set_ylim(0.0, 1.15 * sig_path.max())

# --- right: densities along the geodesic ---
ax1 = fig.add_subplot(gs[0, 1])
x = np.linspace(-6.0, 6.0, 500)
cmap = plt.get_cmap("viridis")
for k, i in enumerate(snap_idx):
    color = cmap(k / (N_SNAPSHOTS - 1))
    ax1.plot(
        x,
        gaussian(x, mu_path[i], sig_path[i]),
        color=color,
        lw=2.0,
        label=rf"{k+1}: $\mu={mu_path[i]:.2f},\ \sigma={sig_path[i]:.2f}$",
    )
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$p_{\mu(t),\sigma(t)}(x)$")
ax1.set_title("Densities along the geodesic")
ax1.legend(loc="upper right", fontsize=7, frameon=False)

fig.suptitle(
    r"Fisher--Rao: widen $\sigma$ before translating $\mu$"
    "\n"
    r"$ds^{2}=(d\mu^{2}+2\,d\sigma^{2})/\sigma^{2}$",
)
path = OUT / "03_geodesic.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(
    f"σ along path: min={sig_path.min():.3f}, max={sig_path.max():.3f} "
    f"(endpoints at σ={SIG_A})"
)
# plt.show()  # uncomment for interactive display
