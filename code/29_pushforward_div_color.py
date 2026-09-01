"""
Push-forward of q = N(0, I_2) by a map that compresses along x1
in two places: the bulk (x1=0) and a tail (x1=c).

The graph of p_theta is the same in both panels. The deformed grid
sits on the floor, coloured by
  left:  div v          (geometric compression of volume)
  right: div(p v)       (compression of mass)
Fisher--Rao sees the second. A strong div v in the tail, where p is
small, barely contributes.

Not included in the paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR_P = "#4C78A8"
COLOR_Q = "#9AA0A6"
COLOR_GRID = "#1b3a5c"
COLOR_MARK = "#C44E52"

ALPHA = -0.42
C_TAIL = 2.25
LIM = 3.4
N = 140
LEVELS = 7
ELEV, AZIM = 26.0, -58.0
FLOOR = 2.8
N_LINES = 11


def sech2(z):
    return 1.0 / np.cosh(z) ** 2


def f(y):
    return y + ALPHA * (np.tanh(y) + np.tanh(y - C_TAIL))


def fp(y):
    return 1.0 + ALPHA * (sech2(y) + sech2(y - C_TAIL))


def q_pdf(y1, y2):
    return np.exp(-0.5 * (y1**2 + y2**2)) / (2.0 * np.pi)


# Inverse of f by interpolation (f is strictly increasing for this ALPHA).
Y_SAMP = np.linspace(-8.0, 8.0, 4000)
X_SAMP = f(Y_SAMP)
assert np.all(np.diff(X_SAMP) > 0.0), "f is not increasing"


def y1_of(x1):
    return np.interp(x1, X_SAMP, Y_SAMP)


def p_pdf(x1, x2):
    y1 = y1_of(x1)
    return q_pdf(y1, x2) / fp(y1)


def div_v(x1, x2):
    """Eulerian divergence of the displacement v = (phi - id) o psi."""
    y1 = y1_of(x1)
    return ALPHA * (sech2(y1) + sech2(y1 - C_TAIL)) / fp(y1)


def div_pv(x1, x2):
    """div(p v) = <grad p, v> + p div v."""
    h = 0.02
    p = p_pdf(x1, x2)
    y1 = y1_of(x1)
    v1 = ALPHA * (np.tanh(y1) + np.tanh(y1 - C_TAIL))
    p_plus = p_pdf(x1 + h, x2)
    p_minus = p_pdf(x1 - h, x2)
    dp_dx1 = (p_plus - p_minus) / (2.0 * h)
    return dp_dx1 * v1 + p * div_v(x1, x2)


xs = np.linspace(-LIM, LIM, N)
X, Y = np.meshgrid(xs, xs)
P = p_pdf(X, Y)
Q = q_pdf(X, Y)
DV = div_v(X, Y)
DPV = div_pv(X, Y)
print(
    f"fp min={fp(Y_SAMP).min():.3f}  "
    f"div v in [{DV.min():.3f},{DV.max():.3f}]  "
    f"div(pv) in [{DPV.min():.3f},{DPV.max():.3f}]"
)
print(f"peak q={Q.max():.3f}  peak p={P.max():.3f}")


def floor_grid():
    t = np.linspace(-FLOOR, FLOOR, 90)
    ticks = np.linspace(-FLOOR, FLOOR, N_LINES)
    segs = []
    for c in ticks:
        segs.append((f(t), np.full_like(t, c)))
        segs.append((f(np.full_like(t, c)), t))
    return segs


def draw_panel(ax, floor_field, vmax, cmap_label, title):
    cf = ax.contourf(
        X,
        Y,
        floor_field,
        levels=24,
        zdir="z",
        offset=0.0,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.95,
    )
    ax.contour(
        X,
        Y,
        Q,
        levels=5,
        zdir="z",
        offset=0.002,
        colors=COLOR_Q,
        linewidths=0.8,
        linestyles="--",
    )
    for gx, gy in floor_grid():
        ax.plot(gx, gy, np.zeros_like(gx), color=COLOR_GRID, lw=0.55, alpha=0.75)
    ax.plot_surface(
        X,
        Y,
        P,
        color=COLOR_P,
        alpha=0.28,
        rstride=5,
        cstride=5,
        edgecolor="#2b4a6f",
        linewidth=0.22,
        shade=False,
    )
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(0.0, 1.08 * P.max())
    ax.set_xlabel(r"$y_{1}$", labelpad=-5)
    ax.set_ylabel(r"$y_{2}$", labelpad=-5)
    ax.set_title(title, fontsize=10, pad=4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 0.55))
    return cf


def draw_grid_2d(ax, deformed, title, xlabel, ylabel, marks, jac=None):
    t = np.linspace(-FLOOR, FLOOR, 90)
    ticks = np.linspace(-FLOOR, FLOOR, N_LINES)
    if jac is not None:
        ax.contourf(X, Y, jac, levels=16, cmap="Blues_r", alpha=0.85)
    for c in ticks:
        if deformed:
            ax.plot(f(t), np.full_like(t, c), color=COLOR_GRID, lw=0.85)
            ax.plot(f(np.full_like(t, c)), t, color=COLOR_GRID, lw=0.85)
        else:
            ax.plot(t, np.full_like(t, c), color=COLOR_GRID, lw=0.85)
            ax.plot(np.full_like(t, c), t, color=COLOR_GRID, lw=0.85)
    for m in marks:
        ax.axvline(m, color=COLOR_MARK, ls="--", lw=1.0, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


fig = plt.figure(figsize=(11.8, 9.4))
ax0 = fig.add_subplot(2, 2, 1, projection="3d")
ax1 = fig.add_subplot(2, 2, 2, projection="3d")
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 4)

vmax_v = float(np.max(np.abs(DV)))
vmax_pv = float(np.max(np.abs(DPV)))

cf0 = draw_panel(
    ax0,
    DV,
    vmax_v,
    r"$\mathrm{div}\,v$",
    r"Background: $\mathrm{div}\,v$"
    "\n"
    r"(geometric compression)",
)
cf1 = draw_panel(
    ax1,
    DPV,
    vmax_pv,
    r"$\mathrm{div}(pv)$",
    r"Background: $\mathrm{div}(pv)$"
    "\n"
    r"(mass compression)",
)
fig.colorbar(cf0, ax=ax0, shrink=0.55, pad=0.02, label=r"$\mathrm{div}\,v$")
fig.colorbar(cf1, ax=ax1, shrink=0.55, pad=0.02, label=r"$\mathrm{div}(pv)$")

marks_y = (0.0, C_TAIL)
marks_x = (float(f(0.0)), float(f(C_TAIL)))
draw_grid_2d(
    ax2,
    deformed=False,
    title=r"Uniform grid (coordinates $x$ of $q$)",
    xlabel=r"$x_{1}$",
    ylabel=r"$x_{2}$",
    marks=marks_y,
)
ax2.text(0.08, LIM - 0.45, r"$x_{1}=0$", color=COLOR_MARK, fontsize=8)
ax2.text(C_TAIL + 0.08, LIM - 0.45, r"$x_{1}=c$", color=COLOR_MARK, fontsize=8)
jac = fp(y1_of(X))
draw_grid_2d(
    ax3,
    deformed=True,
    title=r"Image under $\phi$  "
    r"(background: $\det\,\partial\phi/\partial x=f'(x_{1})$)",
    xlabel=r"$y_{1}$",
    ylabel=r"$y_{2}$",
    marks=marks_x,
    jac=jac,
)
ax3.text(marks_x[0] + 0.08, LIM - 0.45, r"$f(0)$", color=COLOR_MARK, fontsize=8)
ax3.text(marks_x[1] + 0.08, LIM - 0.45, r"$f(c)$", color=COLOR_MARK, fontsize=8)

fig.suptitle(
    r"$\phi(x)=(x_1+\alpha(\tanh x_1+\tanh(x_1-c)),\, x_2)$, "
    rf"$\alpha={ALPHA:g}$, $c={C_TAIL:g}$"
    "\n"
    r"$q=\mathcal{N}(0,I)$ (dashed level sets), $p_{\theta}=\phi_{\sharp}q$",
    y=0.98,
    fontsize=11,
)
fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.06, hspace=0.28, wspace=0.22)
path = OUT / "29_pushforward_div_color.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
