"""
Pullback of the Euclidean metric on the graph of (u, v) ↦ u² + v².

Left: the graph in R³, coloured by Riemannian distance from φ(1,1)=(1,1,2),
with iso-contours, and the image dφ(ξ) of a small displacement.
The (u, v) chart sits underneath, with the same colouring.
Right: the same distance in coordinates, with ξ itself.
By definition (φ* ⟨,⟩)(ξ, ξ) = ⟨dφ(ξ), dφ(ξ)⟩.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR_VEC = "#C44E52"
U0, V0 = 1.0, 1.0
# Visible in the chart, small enough to read as a tangent increment.
XI = np.array([0.32, 0.20])
U_MIN, U_MAX = 0.05, 1.95
V_MIN, V_MAX = 0.05, 1.95
N_GRID = 160
Z_FLOOR = -1.55
ELEV, AZIM = 22.0, -58.0


def graph(u, v):
    return u, v, u**2 + v**2


def metric(u, v):
    return np.array(
        [[1.0 + 4.0 * u**2, 4.0 * u * v], [4.0 * u * v, 1.0 + 4.0 * v**2]]
    )


def dphi(u, v, xi):
    return np.array([xi[0], xi[1], 2.0 * u * xi[0] + 2.0 * v * xi[1]])


def geodesic_distance_map(u, v, u0, v0, t_max=3.2, n_ang=280, n_steps=420):
    """Arc-length distance of the induced metric, by geodesic shooting.

    The geodesic equation on the graph z=u²+v² is
        ẍ = -4 |ẋ|_eucl² x / (1 + 4 |x|²),  x=(u,v).
    """
    g0 = metric(u0, v0)
    alphas = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    w = np.stack([np.cos(alphas), np.sin(alphas)], axis=1)
    lam = 1.0 / np.sqrt(np.einsum("ai,ij,aj->a", w, g0, w))
    pos = np.repeat(np.array([[u0, v0]]), n_ang, axis=0)
    vel = lam[:, None] * w

    dt = t_max / n_steps
    nu, nv = len(u), len(v)
    dist = np.full((nv, nu), np.inf)
    du = u[1] - u[0]
    dv = v[1] - v[0]

    def deposit(p, t):
        iu = np.rint((p[:, 0] - u[0]) / du).astype(int)
        iv = np.rint((p[:, 1] - v[0]) / dv).astype(int)
        ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        np.minimum.at(dist, (iv[ok], iu[ok]), t[ok])

    times = np.zeros(n_ang)
    deposit(pos, times)

    def accel(p, vel):
        speed2 = np.sum(vel**2, axis=1, keepdims=True)
        r2 = np.sum(p**2, axis=1, keepdims=True)
        return -4.0 * speed2 * p / (1.0 + 4.0 * r2)

    for _ in range(n_steps):
        p1, v1 = pos, vel
        a1 = accel(p1, v1)
        p2, v2 = pos + 0.5 * dt * v1, vel + 0.5 * dt * a1
        a2 = accel(p2, v2)
        p3, v3 = pos + 0.5 * dt * v2, vel + 0.5 * dt * a2
        a3 = accel(p3, v3)
        p4, v4 = pos + dt * v3, vel + dt * a3
        a4 = accel(p4, v4)
        pos = pos + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        vel = vel + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        times = times + dt
        deposit(pos, times)

    # nearest-neighbour fill for the few cells the rays miss
    inf = ~np.isfinite(dist)
    if inf.any():
        jj, ii = np.indices(dist.shape)
        good = np.flatnonzero(~inf.ravel())
        gi, gj = np.unravel_index(good, dist.shape)
        for iv, iu in zip(*np.where(inf)):
            k = np.argmin((gi - iv) ** 2 + (gj - iu) ** 2)
            dist[iv, iu] = dist[gi[k], gj[k]]
    dist[np.argmin(np.abs(v - v0)), np.argmin(np.abs(u - u0))] = 0.0
    return dist


def contour_segments(U, V, dist, levels):
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(U, V, dist, levels=levels)
    segs = [np.asarray(s) for level in cs.allsegs for s in level if len(s) > 1]
    plt.close(fig_tmp)
    return segs


def add_iso_3d(ax, segs, z_of, **kwargs):
    plot_kw = dict(kwargs)
    if "colors" in plot_kw:
        plot_kw["color"] = plot_kw.pop("colors")
    if "linewidths" in plot_kw:
        plot_kw["lw"] = plot_kw.pop("linewidths")
    for seg in segs:
        z = z_of(seg[:, 0], seg[:, 1])
        ax.plot(seg[:, 0], seg[:, 1], z, **plot_kw)


u = np.linspace(U_MIN, U_MAX, N_GRID)
v = np.linspace(V_MIN, V_MAX, N_GRID)
U, V = np.meshgrid(u, v)
Z = U**2 + V**2
DIST = geodesic_distance_map(u, v, U0, V0, t_max=4.5, n_ang=320, n_steps=500)
dmax = float(np.nanmax(DIST))
print(f"distance max on chart: {dmax:.3f}")

G0 = metric(U0, V0)
dphi_xi = dphi(U0, V0, XI)
pull = float(XI @ G0 @ XI)
push = float(np.dot(dphi_xi, dphi_xi))
print(f"phi*g(xi,xi)={pull:.6f}  <dphi(xi),dphi(xi)>={push:.6f}")

iso = np.arange(0.25, dmax, 0.25)
segs = contour_segments(U, V, DIST, iso)
norm = Normalize(vmin=0.0, vmax=dmax)
facecolors = cm.coolwarm(norm(DIST))

fig = plt.figure(figsize=(12.8, 6.0))
ax3 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
ax2 = fig.add_subplot(1, 2, 2)

# --- 3D: chart plane underneath ---
floor_colors = facecolors.copy()
floor_colors[..., 3] = 0.92
ax3.plot_surface(
    U,
    V,
    np.full_like(U, Z_FLOOR),
    facecolors=floor_colors,
    rstride=3,
    cstride=3,
    linewidth=0,
    antialiased=False,
    shade=False,
    zorder=0,
)
add_iso_3d(
    ax3,
    segs,
    lambda uu, vv: np.full_like(uu, Z_FLOOR + 0.02),
    colors="k",
    linewidths=0.55,
    alpha=0.8,
    zorder=1,
)

# --- 3D: graph ---
surf_colors = facecolors.copy()
surf_colors[..., 3] = 0.82
ax3.plot_surface(
    U,
    V,
    Z,
    facecolors=surf_colors,
    rstride=2,
    cstride=2,
    linewidth=0,
    antialiased=True,
    shade=False,
    zorder=3,
)
add_iso_3d(
    ax3,
    segs,
    lambda uu, vv: uu**2 + vv**2,
    colors="k",
    linewidths=0.7,
    alpha=0.85,
    zorder=4,
)

# marked point and its chart image
p0 = np.array(graph(U0, V0))
ax3.plot([U0], [V0], [p0[2]], "o", color="white", ms=8, mew=1.2, mec="black", zorder=6)
ax3.plot([U0], [V0], [Z_FLOOR + 0.03], "o", color="white", ms=7, mew=1.1, mec="black", zorder=2)
ax3.plot(
    [U0, U0],
    [V0, V0],
    [Z_FLOOR + 0.03, p0[2]],
    ls="--",
    color="0.25",
    lw=1.0,
    zorder=2,
)

# displacement: ξ on the chart, dφ(ξ) on the graph
ax3.quiver(
    U0,
    V0,
    Z_FLOOR + 0.04,
    XI[0],
    XI[1],
    0.0,
    color=COLOR_VEC,
    lw=2.0,
    arrow_length_ratio=0.18,
    zorder=2,
)
ax3.quiver(
    p0[0],
    p0[1],
    p0[2],
    dphi_xi[0],
    dphi_xi[1],
    dphi_xi[2],
    color=COLOR_VEC,
    lw=2.0,
    arrow_length_ratio=0.12,
    zorder=6,
)

ax3.text(U0 + 0.08, V0 - 0.22, p0[2] + 0.15, r"$\varphi(1,1)=(1,1,2)$", fontsize=9, zorder=7)
ax3.text(U0 + XI[0] + 0.04, V0 + XI[1] - 0.02, Z_FLOOR + 0.12, r"$\xi$", color=COLOR_VEC, fontsize=11, zorder=7)
ax3.text(
    p0[0] + dphi_xi[0] * 0.55,
    p0[1] + dphi_xi[1] * 0.55 + 0.08,
    p0[2] + dphi_xi[2] * 0.55 + 0.12,
    r"$d\varphi(\xi)$",
    color=COLOR_VEC,
    fontsize=11,
    zorder=7,
)
ax3.text(0.15, 1.55, Z_FLOOR - 0.05, r"$(u,v)$", fontsize=11, zorder=7)
ax3.set_xlabel(r"$u$")
ax3.set_ylabel(r"$v$")
ax3.set_zlabel(r"$u^{2}+v^{2}$")
ax3.view_init(elev=ELEV, azim=AZIM)
ax3.set_xlim(U_MIN, U_MAX)
ax3.set_ylim(V_MIN, V_MAX)
ax3.set_zlim(Z_FLOOR - 0.1, float(Z.max()) + 0.2)
ax3.set_box_aspect((1.0, 1.0, 1.15))
ax3.set_title(r"Graph of $\varphi(u,v)=(u,v,u^{2}+v^{2})$", fontsize=11)

# --- 2D chart ---
levels_fill = np.linspace(0.0, dmax, 36)
cf = ax2.contourf(U, V, DIST, levels=levels_fill, cmap="coolwarm")
cs = ax2.contour(U, V, DIST, levels=iso, colors="k", linewidths=0.65, alpha=0.8)
ax2.clabel(cs, iso[::2], inline=True, fmt=r"$d=%.2f$", fontsize=7)
ax2.plot(U0, V0, "o", color="white", ms=8, mew=1.2, mec="black", zorder=5)
ax2.annotate(
    "",
    xy=(U0 + XI[0], V0 + XI[1]),
    xytext=(U0, V0),
    arrowprops=dict(arrowstyle="-|>", color=COLOR_VEC, lw=2.0, mutation_scale=12),
)
ax2.text(U0 + XI[0] + 0.03, V0 + XI[1] + 0.02, r"$\xi$", color=COLOR_VEC, fontsize=12)
ax2.set_aspect("equal")
ax2.set_xlim(U_MIN, U_MAX)
ax2.set_ylim(V_MIN, V_MAX)
ax2.set_xlabel(r"$u$")
ax2.set_ylabel(r"$v$")
ax2.set_title(r"Chart $(u,v)$, pullback distance", fontsize=11)
cbar = fig.colorbar(cf, ax=ax2, pad=0.02, fraction=0.046)
cbar.set_label(r"$d((u,v),(1,1))$", fontsize=9)

ax2.text(
    0.03,
    0.97,
    r"$(\varphi^{*}\langle\cdot,\cdot\rangle)_{(1,1)}(\xi,\xi)"
    r"=\langle d\varphi(\xi),\,d\varphi(\xi)\rangle$"
    "\n"
    + rf"$={pull:.2f}$",
    transform=ax2.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.92),
)

fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=0.18)
path = OUT / "25_pullback_graph.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
