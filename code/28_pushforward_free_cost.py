"""
Free vs costly push-forwards of a Gaussian, as a reading of div(p v).

1D  (28_pushforward_free_cost_1d.png)
  Row 1: two maps that give the same density (sigma(y)=-y preserves q),
          so Fisher--Rao sees no difference: g=0.
  Row 2: dilation phi(y)=s y, which compresses or spreads mass.

2D  (28_pushforward_free_cost_2d.png)
  Row 1: rotation of N(0,I).  div(p v)=0, the graph is unchanged,
          only the floor grid turns.
  Row 2: isotropic compression / dilatation. The graph peaks or flattens,
          and the floor grid shows the squeeze.

Not included in the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR_P = "#4C78A8"
COLOR_A = "#C44E52"
COLOR_B = "#54A24B"
COLOR_GRID = "#2b4a6f"
BASE = "#9AA0A6"

# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------
y = np.linspace(-4.5, 4.5, 900)
q = np.exp(-0.5 * y**2) / np.sqrt(2.0 * np.pi)
MU = 1.15
quant = norm.ppf(np.linspace(0.08, 0.92, 9))


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


fig, axes = plt.subplots(
    2, 2, figsize=(11.4, 6.4), gridspec_kw={"hspace": 0.42, "wspace": 0.28}
)

# maps that preserve q: id and reflection, then translate
ax = axes[0, 0]
ax.plot(y, y + MU, color=COLOR_A, lw=2.0, label=r"$\phi(y)=y+\mu$")
ax.plot(y, -y + MU, color=COLOR_B, lw=2.0, label=r"$\phi\circ\sigma(y)=-y+\mu$")
ax.plot(y, y, "--", color=BASE, lw=1.0, label=r"$\mathrm{id}$")
for yi in quant:
    ax.annotate(
        "",
        xy=(yi, yi + MU),
        xytext=(yi, yi),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_A, lw=0.9, mutation_scale=8),
    )
    ax.annotate(
        "",
        xy=(yi, -yi + MU),
        xytext=(yi, yi),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_B, lw=0.9, mutation_scale=8),
    )
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.8)
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$\phi(y)$")
ax.set_title(r"Two transports, $\sigma_{\sharp}q=q$" "\n" r"$\sigma(y)=-y$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[0, 1]
p = gaussian(y, MU, 1.0)
ax.fill_between(y, p, color=COLOR_P, alpha=0.35)
ax.plot(y, p, color=COLOR_P, lw=2.0, label=r"$p=(\phi)_{\sharp}q=(\phi\circ\sigma)_{\sharp}q$")
ax.plot(y, q, "--", color=BASE, lw=1.3, label=r"$q=\mathcal{N}(0,1)$")
ax.set_xlim(-3.2, 4.2)
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$")
ax.set_title(r"Same density" "\n" r"two maps, a single $p$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper right")

# dilation
ax = axes[1, 0]
s_c, s_d = 0.48, 1.85
ax.plot(y, s_c * y, color=COLOR_A, lw=2.0, label=rf"$s={s_c:g}$ (compression)")
ax.plot(y, s_d * y, color=COLOR_B, lw=2.0, label=rf"$s={s_d:g}$ (dilation)")
ax.plot(y, y, "--", color=BASE, lw=1.0, label=r"$s=1$")
for yi in quant:
    ax.annotate(
        "",
        xy=(yi, s_c * yi),
        xytext=(yi, yi),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_A, lw=0.9, mutation_scale=8),
    )
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.8, 3.8)
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$\phi(y)=sy$")
ax.set_title(r"Compression / dilation" "\n" r"$\phi(y)=sy$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.fill_between(y, gaussian(y, 0.0, s_c), color=COLOR_A, alpha=0.30)
ax.plot(y, gaussian(y, 0.0, s_c), color=COLOR_A, lw=2.0, label=rf"$\sigma={s_c:g}$")
ax.plot(y, gaussian(y, 0.0, s_d), color=COLOR_B, lw=2.0, label=rf"$\sigma={s_d:g}$")
ax.plot(y, q, "--", color=BASE, lw=1.3, label=r"$q$")
ax.set_xlim(-4.2, 4.2)
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$")
ax.set_title(
    r"The mass concentrates or spreads"
    "\n"
    r"$\mathrm{div}(pv)\neq 0$, $g=2\,(ds/s)^{2}$",
    fontsize=10,
)
ax.legend(fontsize=8, frameon=False, loc="upper right")

fig.suptitle(
    r"Push-forward in dimension one, $q=\mathcal{N}(0,1)$: "
    r"redundancy (top) vs. compression (bottom)",
    y=1.02,
    fontsize=12,
)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.09)
path = OUT / "28_pushforward_free_cost_1d.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------
LIM = 3.2
N = 90
LEVELS = 7
ELEV, AZIM = 28.0, -58.0
FLOOR_LIM = 2.6
N_GRID_LINES = 9


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def apply_A(Y1, Y2, A):
    return A[0, 0] * Y1 + A[0, 1] * Y2, A[1, 0] * Y1 + A[1, 1] * Y2


def gaussian_iso(X, Y, sigma):
    return np.exp(-0.5 * (X**2 + Y**2) / sigma**2) / (2.0 * np.pi * sigma**2)


def draw_floor_grid(ax, A, color=COLOR_GRID):
    t = np.linspace(-FLOOR_LIM, FLOOR_LIM, 80)
    ticks = np.linspace(-FLOOR_LIM, FLOOR_LIM, N_GRID_LINES)
    z = np.zeros_like(t)
    for c in ticks:
        x1, x2 = apply_A(t, np.full_like(t, c), A)
        ax.plot(x1, x2, z, color=color, lw=0.7, alpha=0.9, zorder=2)
        x1, x2 = apply_A(np.full_like(t, c), t, A)
        ax.plot(x1, x2, z, color=color, lw=0.7, alpha=0.9, zorder=2)


def draw_scene(ax, A, sigma, zmax, title):
    xs = np.linspace(-LIM, LIM, N)
    X, Y = np.meshgrid(xs, xs)
    Z = gaussian_iso(X, Y, sigma)
    ax.contourf(X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, cmap="Blues", alpha=0.55)
    ax.contour(
        X, Y, Z, levels=LEVELS, zdir="z", offset=0.0, colors="#1b3a5c", linewidths=0.55
    )
    draw_floor_grid(ax, A)
    ax.plot_surface(
        X,
        Y,
        Z,
        color=COLOR_P,
        alpha=0.22,
        rstride=6,
        cstride=6,
        edgecolor="#2b4a6f",
        linewidth=0.25,
        shade=False,
        zorder=3,
    )
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(0.0, 1.05 * zmax)
    ax.set_xlabel(r"$x_{1}$", labelpad=-6)
    ax.set_ylabel(r"$x_{2}$", labelpad=-6)
    ax.set_title(title, fontsize=10, pad=2)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 0.58))


A_id = np.eye(2)
A_rot = rotation(np.pi / 5.0)
s_c, s_d = 0.50, 1.70
A_c = s_c * np.eye(2)
A_d = s_d * np.eye(2)

z_iso = 1.0 / (2.0 * np.pi)  # sigma=1
z_c = 1.0 / (2.0 * np.pi * s_c**2)
z_d = 1.0 / (2.0 * np.pi * s_d**2)

fig = plt.figure(figsize=(10.6, 8.2))
ax00 = fig.add_subplot(2, 2, 1, projection="3d")
ax01 = fig.add_subplot(2, 2, 2, projection="3d")
ax10 = fig.add_subplot(2, 2, 3, projection="3d")
ax11 = fig.add_subplot(2, 2, 4, projection="3d")

draw_scene(ax00, A_id, 1.0, z_iso, r"Identity" "\n" r"grid of $q$, $p=q$")
draw_scene(
    ax01,
    A_rot,
    1.0,
    z_iso,
    r"Rotation $\pi/5$"
    "\n"
    r"$\mathrm{div}(pv)=0$: same graph",
)
draw_scene(
    ax10,
    A_c,
    s_c,
    z_c,
    r"Compression $s=1/2$"
    "\n"
    r"concentrated mass, tighter grid",
)
draw_scene(
    ax11,
    A_d,
    s_d,
    max(z_d, 0.35 * z_c),
    r"Dilation $s=1{,}7$"
    "\n"
    r"spread mass, expanded grid",
)
# keep a readable height on the dilated panel: use a shared modest zmax
ax11.set_zlim(0.0, 1.05 * z_iso)

fig.suptitle(
    r"Isotropic bivariate Gaussian: free rearrangement (top) "
    r"vs. compression / dilation (bottom)",
    y=0.98,
    fontsize=12,
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, hspace=0.12, wspace=0.02)
path = OUT / "28_pushforward_free_cost_2d.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"peaks: iso={z_iso:.3f}, compress={z_c:.3f}, dilate={z_d:.3f}")
