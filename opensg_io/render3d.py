"""opensg_io.render3d -- shaded 3-D renders of the generated segment meshes.

Renders ACTUAL mesh cells (never a parametric sketch): shaded faces with visible
element edges, colored by element set, so hex/quad elements are individually visible.
PyVista (VTK) is used when importable; otherwise a matplotlib Poly3DCollection
fallback shades the BOUNDARY faces of the solid (outer skin + the two end caps,
which expose the through-thickness hex layering and the webs) or the shell quads.
"""
import os
import numpy as np

PAL = np.array([
    [0.42, 0.42, 0.42], [0.12, 0.47, 0.71], [0.17, 0.63, 0.17], [1.00, 0.50, 0.05],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.09, 0.75, 0.81], [0.84, 0.15, 0.16],
    [0.74, 0.74, 0.13], [0.89, 0.47, 0.76],
])

_HEXF = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def boundary_faces(hexes):
    """Faces of the hex mesh shared by exactly ONE hex (outer skin + end caps),
    with the owning hex index."""
    seen = {}
    for k, h in enumerate(hexes):
        for f in _HEXF:
            quad = tuple(int(h[i]) for i in f)
            key = tuple(sorted(quad))
            if key in seen:
                seen[key] = None
            else:
                seen[key] = (quad, k)
    return [v for v in seen.values() if v is not None]


# --------------------------------------------------------------------------- pyvista
def _render_pyvista(nodes, cells, celltype, setmap, png, title):
    import pyvista as pv
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass
    n = len(cells)
    if celltype == "hex":
        vtk_cells = np.hstack([np.full((n, 1), 8, np.int64), cells]).ravel()
        ct = np.full(n, pv.CellType.HEXAHEDRON, np.uint8)
    else:
        vtk_cells = np.hstack([np.full((n, 1), 4, np.int64), cells]).ravel()
        ct = np.full(n, pv.CellType.QUAD, np.uint8)
    # plot in (z, x, y): the beam/span axis runs horizontally in the view
    grid = pv.UnstructuredGrid(vtk_cells, ct, np.asarray(nodes, float)[:, [2, 0, 1]])
    grid.cell_data["set"] = np.asarray(setmap, int)
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
    from matplotlib.colors import ListedColormap
    nset = int(np.max(setmap)) + 1
    pl.add_mesh(grid, scalars="set", cmap=ListedColormap(PAL[np.arange(nset) % len(PAL)]),
                show_edges=True, edge_color="black", line_width=0.4,
                show_scalar_bar=False)
    pl.add_text(title, font_size=11)
    pl.camera_position = "iso"
    pl.camera.azimuth = 12
    pl.camera.elevation = -8
    pl.camera.zoom(1.25)
    pl.screenshot(png)
    pl.close()
    return png


# ------------------------------------------------------------------------ matplotlib
def _render_mpl(nodes, cells, celltype, setmap, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nodes = np.asarray(nodes, float)
    if celltype == "hex":
        faceset = boundary_faces(cells)
        quads = np.array([q for q, _k in faceset], int)
        fset = np.array([setmap[k] for _q, k in faceset], int)
    else:
        quads = np.asarray(cells, int)
        fset = np.asarray(setmap, int)

    # plot in (z, x, y) so the span runs along the horizontal axis
    P = nodes[:, [2, 0, 1]]
    polys = P[quads]                                     # (nf, 4, 3)
    # lambertian shade per face
    nrm = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    nn = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.where(nn > 1e-30, nn, 1.0)[:, None]
    light = np.array([0.35, -0.5, 0.79])
    shade = 0.55 + 0.45 * np.abs(nrm @ light)
    base = PAL[fset % len(PAL)]
    fc = np.clip(base * shade[:, None], 0, 1)

    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(polys, facecolors=fc, edgecolor="k", linewidths=0.12)
    ax.add_collection3d(pc)
    lo, hi = P[quads.ravel()].min(0), P[quads.ravel()].max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((hi - lo))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-72)
    ax.set_xlabel("span z [m]")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png


def render_mesh_png(nodes, cells, celltype, setmap, png, title):
    """Shaded render of a hex solid ('hex') or quad shell ('quad') mesh, colored by
    element set, element edges visible.  Uses PyVista when available, else the
    matplotlib boundary-face fallback."""
    try:
        return _render_pyvista(nodes, cells, celltype, setmap, png, title)
    except Exception as e:
        print("pyvista unavailable (%s) -> matplotlib fallback" % type(e).__name__, flush=True)
        return _render_mpl(nodes, cells, celltype, setmap, png, title)


def render_section_ends(sec, shell_sec2d, r1, r2, png, title=None):
    """Cross-section views at BOTH ends for the solid and the shell, as one 2x2 figure:
    top row = solid section (through-thickness quad mesh, webs crimson) at r1 | r2;
    bottom row = shell section (mid-surface line + web lines) at r1 | r2.  Rendered
    from the ACTUAL station coordinates, so taper (chord shrink) is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = sec["faces2d"]
    ftag = sec["ftag"]
    xy = np.vstack([sec["stations"][0][:, :2], sec["stations"][1][:, :2]])
    xlo, xhi = xy[:, 0].min(), xy[:, 0].max()
    ylo, yhi = xy[:, 1].min(), xy[:, 1].max()
    padx = 0.03 * (xhi - xlo); pady = 0.06 * (yhi - ylo)
    S = shell_sec2d["S"]; skin_loop = shell_sec2d["skin_loop"]; web_lines = shell_sec2d["web_lines"]

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    for col, (P, r) in enumerate([(sec["stations"][0][:, :2], r1),
                                  (sec["stations"][1][:, :2], r2)]):
        ax = axs[0][col]
        for qq, tg in zip(q, ftag):
            lp = list(qq) + [int(qq[0])]
            ax.plot(P[lp, 0], P[lp, 1], color=("crimson" if tg[0] == "web" else "0.35"),
                    lw=(0.6 if tg[0] == "web" else 0.5))
        ax.set_title("SOLID hex section  @ r = %.2f" % r)
        for col2, (Ps, r2v) in enumerate([(S[0], r1), (S[1], r2)]):
            axs[1][col2].plot(Ps[skin_loop + [skin_loop[0]], 0],
                              Ps[skin_loop + [skin_loop[0]], 1], "-", color="0.15", lw=1.0)
            for wl in web_lines:
                axs[1][col2].plot(Ps[wl, 0], Ps[wl, 1], "-", color="crimson", lw=1.2)
            axs[1][col2].set_title("SHELL mid-surface section  @ r = %.2f" % r2v)
    for ax in axs.ravel():
        ax.set_aspect("equal")
        ax.set_xlim(xlo - padx, xhi + padx); ax.set_ylim(ylo - pady, yhi + pady)
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png


def render_section_png(sec, png, title, te_inset=True):
    """The ACTUAL 2-D station mesh (rings + web columns) with a trailing-edge zoom
    inset, from a build_section_mesh result (station 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = sec["stations"][0][:, :2]
    q = sec["faces2d"]
    fig, ax = plt.subplots(figsize=(13, 6))

    def draw(a, lw_skin, lw_web):
        for qq, tg in zip(q, sec["ftag"]):
            lp = list(qq) + [int(qq[0])]
            a.plot(P[lp, 0], P[lp, 1],
                   color=("crimson" if tg[0] == "web" else "0.35"),
                   lw=(lw_web if tg[0] == "web" else lw_skin))

    draw(ax, 0.4, 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    if te_inset:
        xmax = P[:, 0].max(); xmin = P[:, 0].min()
        cut = xmax - 0.18 * (xmax - xmin)
        m = P[:, 0] >= cut
        if m.any():
            y0, y1 = P[m, 1].min(), P[m, 1].max()
            pad = 0.15 * (y1 - y0 + 1e-9)
            axi = ax.inset_axes([0.56, 0.55, 0.42, 0.42])
            draw(axi, 0.7, 0.8)
            axi.set_xlim(cut, xmax + 0.01 * (xmax - xmin))
            axi.set_ylim(y0 - pad, y1 + pad)
            axi.set_aspect("equal")
            axi.set_xticks([]); axi.set_yticks([])
            axi.set_title("TE zoom (fold-free offset)", fontsize=9)
            ax.indicate_inset_zoom(axi, edgecolor="0.3")
    fig.tight_layout()
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png
