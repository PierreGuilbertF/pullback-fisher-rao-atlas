"""
Displacement of the density along the eigen-directions of the Fisher metric.

For the linear Gibbs (exponential) family
    phi_alpha(x) = (x, h_alpha(x)),  h_alpha(x) = (2-alpha) x + (alpha-1) x^2,
with alpha = 1.25 at theta_star = (-0.4, 0.55), the pulled-back Fisher metric
    g_theta = Cov_theta(phi(X))
has eigenvalues lambda_min << lambda_max. The two eigenvectors are the
parameter directions in which a displacement of the density costs almost
nothing (degenerate) or costs the most (strongest curvature).

Three panels:
  left    displacement function along the degenerate direction (lambda_min),
          d_theta p(x)(eps v_min) ~ p_{theta_star+eps v_min} - p_{theta_star}
  middle  the reference density p_{theta_star}(x), in the palette used for
          1D densities in the other scripts
  right   displacement function along the strongest direction (lambda_max),
          d_theta p(x)(eps v_max) ~ p_{theta_star+eps v_max} - p_{theta_star}

Both displacements have the same Euclidean length eps, so the much larger
amplitude on the right shows how anisotropic the metric is near a degenerate
point. To first order the displacement function reads
    d_theta p(theta, x)(dtheta) = -p(x) (phi(x) - E[phi])^T dtheta.

Comments and labels are in English.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Density data on the bounded domain E = [-2, 2]
# ---------------------------------------------------------------------------
X = np.linspace(-2.0, 2.0, 600)
DX = X[1] - X[0]

ALPHA = 1.25
THETA0 = np.array([-0.4, 0.55])

COLOR_DENSITY = "#4C78A8"   # palette used for 1D densities in other scripts
COLOR_DISPL = "#E45756"     # displacement functions
EPS = 0.2                   # common displacement length for the two directions


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


def density_at(theta, phi):
    """Normalized density p_theta(x) on E, propto exp(-phi(x)^T theta)."""
    logu = -(phi @ theta)
    logu -= float(logu.max())
    u = np.exp(logu)
    return u / (u.sum() * DX)


# ---------------------------------------------------------------------------
# Setup: metric at theta*, eigen-directions
# ---------------------------------------------------------------------------
phi = feature(ALPHA)
G = metric_at(THETA0, phi)

evals, evecs = np.linalg.eigh(G)
v_min = evecs[:, 0]   # degenerate direction (smallest lambda)
v_max = evecs[:, 1]   # strongest direction (largest lambda)
lam_min, lam_max = evals[0], evals[1]

p0 = density_at(THETA0, phi)

# displaced densities and the displacement functions along each eigen-direction
p_plus_min = density_at(THETA0 + EPS * v_min, phi)
p_plus_max = density_at(THETA0 + EPS * v_max, phi)
d_min = p_plus_min - p0
d_max = p_plus_max - p0

# ---------------------------------------------------------------------------
# Figure:
#   left   = displaced density along lambda_min (blue) + displacement (red)
#   middle = reference density
#   right  = displaced density along lambda_max (blue) + displacement (red)
# ---------------------------------------------------------------------------
fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(18.0, 5.2))

# ---- left: displaced density along the degenerate direction ---------------
axl.axhline(0.0, color="0.6", lw=0.8, zorder=1)
axl.fill_between(X, p_plus_min, color=COLOR_DENSITY, alpha=0.35)
axl.plot(X, p_plus_min, color=COLOR_DENSITY, lw=2.0,
         label=r"$p+\mathrm{d}p(\varepsilon v_{\min})$")
axl.fill_between(X, d_min, color=COLOR_DISPL, alpha=0.35)
axl.plot(X, d_min, color=COLOR_DISPL, lw=2.0,
         label=r"$\mathrm{d}p(\varepsilon v_{\min})$")
axl.set_xlim(-2.0, 2.0)
axl.set_xlabel(r"$x$")
axl.set_ylabel(r"$p_{\theta_{\star}+\varepsilon v_{\min}}(x)$")
axl.set_title(
    r"Displaced density along the degenerate direction" + "\n"
    r"$\varepsilon=%.2f,\ \lambda_{\min}=%.3g$" % (EPS, lam_min),
    fontsize=11,
)
axl.legend(fontsize=8, loc="upper left")

# ---- middle: reference density (1D-density palette) ----------------------
axm.fill_between(X, p0, color=COLOR_DENSITY, alpha=0.35)
axm.plot(X, p0, color=COLOR_DENSITY, lw=2.0)
axm.set_xlim(-2.0, 2.0)
axm.set_xlabel(r"$x$")
axm.set_ylabel(r"$p_{\theta_{\star}}(x)$")
axm.set_title(
    r"Reference density, $\alpha=%.2f$" % ALPHA + "\n"
    r"$\theta_{\star}=(-0.4,\ 0.55)$",
    fontsize=11,
)

# ---- right: displaced density along the strongest direction ---------------
axr.axhline(0.0, color="0.6", lw=0.8, zorder=1)
axr.fill_between(X, p_plus_max, color=COLOR_DENSITY, alpha=0.35)
axr.plot(X, p_plus_max, color=COLOR_DENSITY, lw=2.0,
         label=r"$p+\mathrm{d}p(\varepsilon v_{\max})$")
axr.fill_between(X, d_max, color=COLOR_DISPL, alpha=0.35)
axr.plot(X, d_max, color=COLOR_DISPL, lw=2.0,
         label=r"$\mathrm{d}p(\varepsilon v_{\max})$")
axr.set_xlim(-2.0, 2.0)
axr.set_xlabel(r"$x$")
axr.set_ylabel(r"$p_{\theta_{\star}+\varepsilon v_{\max}}(x)$")
axr.set_title(
    r"Displaced density along the strongest direction" + "\n"
    r"$\varepsilon=%.2f,\ \lambda_{\max}=%.3g$" % (EPS, lam_max),
    fontsize=11,
)
axr.legend(fontsize=8, loc="upper left")

# common vertical scale across all three panels: show the displaced density
# hump and the (smaller, possibly negative) displacement together
ylow = min(0.0, d_min.min(), d_max.min())
yhigh = max(p0.max(), p_plus_min.max(), p_plus_max.max())
for ax in (axl, axm, axr):
    ax.set_ylim(ylow, yhigh)

fig.suptitle(
    r"Displacement of the density along the eigen-directions of "
    r"$g_{\theta}=\mathrm{Cov}_{\theta}(\phi(X))$"
    "\n"
    r"$\phi(x)=(x,\,h_{1.25}(x))$, "
    r"$d_{\theta}p(x)(\mathrm{d}\theta)\simeq "
    r"-p(x)(\phi(x)-\mathbb{E}\phi)^T\mathrm{d}\theta$",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.91])
path = OUT / "34_gibbs_degenerate_displacement.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"alpha={ALPHA}  lambda=({lam_min:.4f},{lam_max:.4f})")
print(f"eps={EPS}  |d_min|max={abs(d_min).max():.3f}  "
      f"|d_max|max={abs(d_max).max():.3f}")
# plt.show()  # uncomment for interactive display
