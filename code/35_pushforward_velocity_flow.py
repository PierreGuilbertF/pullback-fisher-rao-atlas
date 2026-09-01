"""
Velocity field and flow of the two-dimensional push-forward transport.

    phi_theta(y) = (y_1 + theta_1 tanh y_2,  theta_2 y_2),   q = N(0, I_2).

For a displacement dtheta in T_theta Theta the field
    v_{dtheta}(x) = (d phi_theta(.)[dtheta]) o psi_theta
is the velocity with which the transported point is read at the arrival
point x. With psi_theta(x) = (x_1 - theta_1 tanh(x_2/theta_2), x_2/theta_2)
and d phi_theta(y)[dtheta] = (dtheta_1 tanh y_2, dtheta_2 y_2) it is explicit:
    v_{dtheta}(x) = (dtheta_1 tanh(x_2/theta_2),  dtheta_2 x_2/theta_2).
Note that the field depends on theta_2, so it changes as theta moves along
dtheta.

Three panels:
  left   the transported grid at theta +/- dtheta (green/red) around the
         reference grid at theta (blue)
  middle the velocity vector field v_{dtheta} itself, as arrows on the
         reference grid at theta
  right  the flow on theta: the trajectory theta(s) = theta_0 + s dtheta in
         the parameter space Theta. For a fixed theta and dtheta the field
         v_{dtheta} is defined on x, but the flow parameter is theta: moving
         theta continuously along dtheta drags the transported grid.

Colors follow the other scripts: blue = reference, red = plus / field,
green = minus. Comments and labels are in English.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

THETA = np.array([1.2, 0.6])
DTHETA = np.array([0.25, 0.15])

COLOR_REF = "#4C78A8"   # reference grid at theta
COLOR_PLUS = "#E45756"  # grid at theta + dtheta, field and flow
COLOR_MINUS = "#54A24B" # grid at theta - dtheta

# base grid, before transport (more nodes than before)
LINES = np.linspace(-3.0, 3.0, 21)
FINE = np.linspace(-3.0, 3.0, 300)

def transport(y1, y2, theta):
    """phi_theta(y) = (y_1 + theta_1 tanh y_2, theta_2 y_2)."""
    t1, t2 = theta
    return y1 + t1 * np.tanh(y2), t2 * y2


def velocity(x1, x2, theta, dtheta):
    """v_{dtheta}(x), the field read at the arrival point."""
    t2 = theta[1]
    return dtheta[0] * np.tanh(x2 / t2), dtheta[1] * x2 / t2


fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(19.0, 6.4))

# ---------------------------------------------------------------------------
# Left: transported grid at theta - dtheta, theta, theta + dtheta
# ---------------------------------------------------------------------------
for color, theta, label in (
    (COLOR_MINUS, THETA - DTHETA, r"$\theta-\mathrm{d}\theta$"),
    (COLOR_REF, THETA, r"$\theta$"),
    (COLOR_PLUS, THETA + DTHETA, r"$\theta+\mathrm{d}\theta$"),
):
    for k, value in enumerate(LINES):
        X1, X2 = transport(FINE, np.full_like(FINE, value), theta)
        ax_left.plot(X1, X2, color=color, lw=0.8, label=label if k == 0 else None)
        X1, X2 = transport(np.full_like(FINE, value), FINE, theta)
        ax_left.plot(X1, X2, color=color, lw=0.8)
ax_left.set_title(
    r"Transported grid at $\theta\pm\mathrm{d}\theta$",
    fontsize=12,
)
ax_left.set_xlabel(r"$x_{1}$")
ax_left.set_ylabel(r"$x_{2}$")
ax_left.set_aspect("equal")
ax_left.legend(fontsize=9, frameon=False, loc="upper left")

# ---------------------------------------------------------------------------
# Middle: the velocity vector field v_{dtheta}, arrows on the reference grid
# ---------------------------------------------------------------------------
for k, value in enumerate(LINES):
    X1, X2 = transport(FINE, np.full_like(FINE, value), THETA)
    ax_mid.plot(X1, X2, color=COLOR_REF, lw=0.8,
                label=r"$\phi_{\theta}$" if k == 0 else None)
    X1, X2 = transport(np.full_like(FINE, value), FINE, THETA)
    ax_mid.plot(X1, X2, color=COLOR_REF, lw=0.8)

Y1, Y2 = np.meshgrid(LINES, LINES)
X1, X2 = transport(Y1, Y2, THETA)
V1, V2 = velocity(X1, X2, THETA, DTHETA)
ax_mid.quiver(
    X1, X2, V1, V2,
    color=COLOR_PLUS, scale=8.0, width=0.004, headwidth=3.5,
)
ax_mid.set_title(r"Velocity vector field $v_{\mathrm{d}\theta}$", fontsize=12)
ax_mid.set_xlabel(r"$x_{1}$")
ax_mid.set_ylabel(r"$x_{2}$")
ax_mid.set_aspect("equal")
ax_mid.legend(fontsize=9, frameon=False, loc="upper left")

# ---------------------------------------------------------------------------
# Right: the flow on theta, i.e. the trajectory theta(s) = theta_0 + s dtheta
# in the parameter space Theta (the flow parameter is theta)
# ---------------------------------------------------------------------------
S_SEG = np.linspace(-1.0, 1.0, 200)
flow_theta = THETA[None, :] + S_SEG[:, None] * DTHETA[None, :]

ax_right.plot(
    flow_theta[:, 0], flow_theta[:, 1],
    color=COLOR_PLUS, lw=2.0,
    label=r"$\theta(s)=\theta_{0}+s\,\mathrm{d}\theta$",
)
# arrowheads along the flow to show its direction
for s in (-0.6, 0.0, 0.6):
    t = THETA + s * DTHETA
    ax_right.annotate(
        "", xy=t + 0.5 * DTHETA, xytext=t - 0.5 * DTHETA,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_PLUS, lw=1.8),
    )
ax_right.plot(*THETA, "o", color=COLOR_REF, ms=9, zorder=5,
              label=r"$\theta_{0}$")
ax_right.plot(*(THETA - DTHETA), "o", color=COLOR_MINUS, ms=8, zorder=5,
              label=r"$\theta_{0}-\mathrm{d}\theta$")
ax_right.plot(*(THETA + DTHETA), "o", color=COLOR_PLUS, ms=8, zorder=5,
              label=r"$\theta_{0}+\mathrm{d}\theta$")
ax_right.set_title(
    r"Flow on $\theta$: $\theta'=\mathrm{d}\theta$", fontsize=12
)
ax_right.set_xlabel(r"$\theta_{1}$")
ax_right.set_ylabel(r"$\theta_{2}$")
ax_right.set_xlim(0.8, 1.6)
ax_right.set_ylim(0.35, 0.85)
ax_right.set_aspect("equal")
ax_right.grid(alpha=0.25)
ax_right.legend(fontsize=8, frameon=False, loc="upper left")

# common window for the two x-space panels: theta_1 in [0.95, 1.45] and
# theta_2 in [0.45, 0.75] over y in [-3, 3] give roughly x_1 in [-4.5, 4.5],
# x_2 in [-2.4, 2.4]
for ax in (ax_left, ax_mid):
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-2.6, 2.6)

fig.suptitle(
    r"$v_{\mathrm{d}\theta}=\left(\mathrm{d}\phi_{\theta}(\cdot)[\mathrm{d}\theta]\right)"
    r"\circ\psi_{\theta}$,\quad "
    r"$\phi_{\theta}(y)=\left(y_{1}+\theta_{1}\tanh y_{2},\ \theta_{2}y_{2}\right)$",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

path = OUT / "35_pushforward_velocity_flow.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
print(f"theta={THETA}, dtheta={DTHETA}")
print(f"theta-dtheta={THETA-DTHETA}, theta+dtheta={THETA+DTHETA}")
# plt.show()  # uncomment for interactive display
