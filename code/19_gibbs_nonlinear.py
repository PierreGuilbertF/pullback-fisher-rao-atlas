"""
Nonlinear Gibbs illustrations, side by side.
  Left:  E ⊂ R,   U(x,θ) = cos(x θ1 + x² θ2)
  Right: E ⊂ R²,  U(x,θ) = (x1 - θ1²) θ2 + x2² θ2
Z_θ is computed by numerical quadrature on a bounded domain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR = "#4C78A8"
LEVELS = 8
ELEV, AZIM = 32.0, -55.0

# ---- 1D: U = cos(x θ1 + x² θ2) ----
# Energy is bounded, so the density does not decay at infinity: integrate on a
# finite interval large enough to show a few oscillations.
x1d = np.linspace(-3.0, 3.0, 800)
theta_1d = np.array([0.7, 0.45])
U_1d = np.cos(x1d * theta_1d[0] + x1d**2 * theta_1d[1])
unnorm_1d = np.exp(-U_1d)
Z_1d = float(np.trapz(unnorm_1d, x1d))
p_1d = unnorm_1d / Z_1d

# ---- 2D: U = (x1 - θ1²) θ2 + x2² θ2 ----
# With θ2 > 0 the energy grows in |x2| and in +x1; on a square window the
# partition function is finite by construction.
lim = 2.5
n = 140
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]
theta_2d = np.array([0.9, 0.85])
U_2d = (X - theta_2d[0] ** 2) * theta_2d[1] + (Y**2) * theta_2d[1]
unnorm_2d = np.exp(-U_2d)
Z_2d = float(unnorm_2d.sum() * dx * dy)
p_2d = unnorm_2d / Z_2d

fig = plt.figure(figsize=(11.0, 4.0))

# Left: 1D density
ax0 = fig.add_subplot(1, 2, 1)
ax0.fill_between(x1d, p_1d, color=COLOR, alpha=0.35)
ax0.plot(x1d, p_1d, color=COLOR, lw=2.0)
ax0.set_xlim(x1d[0], x1d[-1])
ax0.set_ylim(bottom=0.0)
ax0.set_xlabel(r"$x$")
ax0.set_ylabel(r"$p_{\theta}(x)$")
ax0.set_title(
    r"$U(x,\theta)=\cos(x\theta_{1}+x^{2}\theta_{2})$"
    + "\n"
    + rf"$\theta=({theta_1d[0]:g},\ {theta_1d[1]:g})$, "
    + rf"$Z_{{\theta}}\approx{Z_1d:.3f}$"
)

# Right: 2.5D density surface
ax1 = fig.add_subplot(1, 2, 2, projection="3d")
ax1.contourf(X, Y, p_2d, levels=LEVELS, zdir="z", offset=0.0, cmap="Blues", alpha=0.9)
ax1.contour(
    X, Y, p_2d, levels=LEVELS, zdir="z", offset=0.0, colors="#1b3a5c", linewidths=0.8
)
ax1.plot_surface(
    X,
    Y,
    p_2d,
    color=COLOR,
    alpha=0.22,
    rstride=4,
    cstride=4,
    edgecolor="#2b4a6f",
    linewidth=0.3,
    shade=False,
)
ax1.view_init(elev=ELEV, azim=AZIM)
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_zlim(0.0, 1.05 * float(p_2d.max()))
ax1.set_xlabel(r"$x_{1}$", labelpad=-6)
ax1.set_ylabel(r"$x_{2}$", labelpad=-6)
ax1.set_zticks([])
ax1.tick_params(labelsize=7, pad=-2)
ax1.set_box_aspect((1.0, 1.0, 0.62))
ax1.set_title(
    r"$U(x,\theta)=(x_{1}-\theta_{1}^{2})\theta_{2}+x_{2}^{2}\theta_{2}$"
    + "\n"
    + rf"$\theta=({theta_2d[0]:g},\ {theta_2d[1]:g})$, "
    + rf"$Z_{{\theta}}\approx{Z_2d:.3f}$",
    fontsize=11,
    pad=2,
)

fig.suptitle(r"Nonlinear Gibbs family: two examples", y=1.02, fontsize=12)
fig.subplots_adjust(left=0.07, right=0.98, wspace=0.18, top=0.82, bottom=0.12)

path = OUT / "19_gibbs_nonlinear.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"Z_1d ≈ {Z_1d:.6f}, Z_2d ≈ {Z_2d:.6f}")
# plt.show()
