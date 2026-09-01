"""
Linear Gibbs / exponential family illustrations in 1D.
Densities p_θ(x) ∝ exp(-φ(x)ᵀ θ), normalized on a bounded interval.
Three features side by side: φ(x)=x, φ(x)=(x,x²), φ(x)=(x², cos x).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR = "#4C78A8"


def normalize(x, unnorm):
    z = float(np.trapz(unnorm, x))
    return unnorm / z


fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), sharey=False)

# --- φ(x) = x ---
x0 = np.linspace(-3.0, 3.0, 600)
theta0 = 0.7
u0 = np.exp(-theta0 * x0)
p0 = normalize(x0, u0)
axes[0].fill_between(x0, p0, color=COLOR, alpha=0.35)
axes[0].plot(x0, p0, color=COLOR, lw=2.0)
axes[0].set_title(r"$\phi(x)=x$" + "\n" + rf"$\theta={theta0:g}$")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$p_{\theta}(x)$")

# --- φ(x) = (x, x²) ---
x1 = np.linspace(-3.0, 3.0, 600)
theta1 = np.array([-0.4, 0.55])
u1 = np.exp(-(theta1[0] * x1 + theta1[1] * x1**2))
p1 = normalize(x1, u1)
axes[1].fill_between(x1, p1, color=COLOR, alpha=0.35)
axes[1].plot(x1, p1, color=COLOR, lw=2.0)
axes[1].set_title(
    r"$\phi(x)=(x,x^{2})$"
    + "\n"
    + rf"$\theta=({theta1[0]:g},\ {theta1[1]:g})$"
)
axes[1].set_xlabel(r"$x$")

# --- φ(x) = (x², cos x) on a short interval ---
x2 = np.linspace(-2.0, 2.0, 600)
theta2 = np.array([0.45, 1.2])
u2 = np.exp(-(theta2[0] * x2**2 + theta2[1] * np.cos(x2)))
p2 = normalize(x2, u2)
axes[2].fill_between(x2, p2, color=COLOR, alpha=0.35)
axes[2].plot(x2, p2, color=COLOR, lw=2.0)
axes[2].set_title(
    r"$\phi(x)=(x^{2},\cos x)$"
    + "\n"
    + rf"$\theta=({theta2[0]:g},\ {theta2[1]:g})$"
)
axes[2].set_xlabel(r"$x$")

for ax in axes:
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(ax.get_lines()[0].get_xdata()[0], ax.get_lines()[0].get_xdata()[-1])

fig.suptitle(r"Linear Gibbs family in dimension one", y=1.03, fontsize=12)
fig.tight_layout()
path = OUT / "15_gibbs_1d.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()
