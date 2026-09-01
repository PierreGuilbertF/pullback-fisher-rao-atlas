"""
Linear Gibbs illustrations in R²: translucent 2.5D graphs.
  1. φ(x)=(x1, x2)
  2. φ(x)=x1² + x1 x2 + x2²  (scalar feature)
  3. φ(x)=(cos(x1²+x2²), sin(4 x1 x2)+x1+x2)
Normalized on a square window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

LIM = 2.5
N = 120
LEVELS = 8
ELEV, AZIM = 32.0, -55.0

x = np.linspace(-LIM, LIM, N)
y = np.linspace(-LIM, LIM, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]


def normalize2d(U):
    z = float(U.sum() * dx * dy)
    return U / z


# Case 1: φ = (x1, x2)
theta_a = np.array([0.55, -0.35])
U_a = np.exp(-(theta_a[0] * X + theta_a[1] * Y))
Z_a = normalize2d(U_a)

# Case 2: φ = x1² + x1 x2 + x2²
theta_b = 0.7
U_b = np.exp(-theta_b * (X**2 + X * Y + Y**2))
Z_b = normalize2d(U_b)

# Case 3: φ = (cos(x1²+x2²), sin(4 x1 x2)+x1+x2)
theta_c = np.array([1.1, 0.55])
U_c = np.exp(
    -(
        theta_c[0] * np.cos(X**2 + Y**2)
        + theta_c[1] * (np.sin(4.0 * X * Y) + X + Y)
    )
)
Z_c = normalize2d(U_c)

cases = [
    (r"$\phi(x)=(x_{1},x_{2})$", Z_a),
    (r"$\phi(x)=x_{1}^{2}+x_{1}x_{2}+x_{2}^{2}$", Z_b),
    (
        r"$\phi(x)=(\cos(x_{1}^{2}+x_{2}^{2}),\,\sin(4x_{1}x_{2})+x_{1}+x_{2})$",
        Z_c,
    ),
]
zmax = max(Z.max() for _, Z in cases)

fig = plt.figure(figsize=(12.0, 3.8))
for k, (title, Z) in enumerate(cases):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.contourf(X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, cmap="Blues", alpha=0.9)
    ax.contour(
        X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, colors="#1b3a5c", linewidths=0.8
    )
    ax.plot_surface(
        X,
        Y,
        Z,
        color="#4C78A8",
        alpha=0.22,
        rstride=4,
        cstride=4,
        edgecolor="#2b4a6f",
        linewidth=0.3,
        shade=False,
    )
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_zlim(0.0, 1.05 * zmax)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$x_{1}$", labelpad=-6)
    ax.set_ylabel(r"$x_{2}$", labelpad=-6)
    ax.set_title(title, fontsize=11, pad=2)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 0.62))

fig.suptitle(
    r"Linear Gibbs family in $\mathbb{R}^{2}$",
    y=1.02,
)
fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06, top=0.84, bottom=0.02)

path = OUT / "16_gibbs_2d.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()
