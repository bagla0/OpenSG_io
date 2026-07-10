"""render_comparisons.py -- shell-vs-solid comparison figure set for an OpenSG_io taper
segment, rendered straight from the exported YAMLs.  Produces, into an output folder:

  boundary_row_L.png / boundary_row_R.png
      solid 2-D quad cross-section (by material) + shell 1-D contour (by layup),
      side by side in one horizontal row, WITH NODE MARKERS, at each end.
  taper_comparison.png
      the full 3-D shell (by layup) and solid (by material) taper, side by side (PyVista).
  boundary_orient_solid.png / boundary_orient_shell.png
      e1 / e2 / e3 material frame of each boundary cross-section (one row of three).

Usage:  python render_comparisons.py [yaml_dir] [out_dir]
Defaults: yaml_dir = ./output, out_dir = <OneDrive>/IEA_taper_segment/comparisons
Layup/material colours are shared across every figure.
"""
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
HERE = os.path.dirname(os.path.abspath(__file__))
YDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
ODIR = sys.argv[2] if len(sys.argv) > 2 else (
    r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_data\IEA_taper_segment\comparisons")
os.makedirs(ODIR, exist_ok=True)

MAT_COLOR = {"gelcoat": "#bdbdbd", "glass_triax": "#1f77b4", "glass_biax": "#17becf",
             "glass_uniax": "#2ca02c", "carbon_uniax": "#333333",
             "medium_density_foam": "#ff7f0e"}
LAYUP_COLOR = {"spar cap (carbon)": "#333333", "panel (foam core)": "#ff7f0e",
               "reinforcement (glass uni)": "#2ca02c", "skin (triax)": "#1f77b4",
               "shear web": "#d62728"}


def layup_type(lam, is_web):
    mats = [m for m, _t, _a in lam]
    if is_web:
        return "shear web"
    if any("carbon" in m for m in mats):
        return "spar cap (carbon)"
    if any(("foam" in m.lower() or "balsa" in m.lower()) for m in mats):
        return "panel (foam core)"
    if any(("uniax" in m or "_uni" in m) for m in mats):
        return "reinforcement (glass uni)"
    return "skin (triax)"


def load(path):
    d = yaml.load(open(path), Loader=CLoad)
    rows = d["nodes"]
    if isinstance(rows[0][0], str):
        nodes = np.array([[float(v) for v in r[0].split()] for r in rows])
        cells = [[int(v) - 1 for v in r[0].split()] for r in d["elements"]]
        one = True
    else:
        nodes = np.array(rows, float)
        cells = [list(map(int, e)) for e in d["elements"]]
        one = False
    ori = np.array(d["elementOrientations"], float)
    setname = ["?"] * len(cells)
    for st in d["sets"]["element"]:
        for lab in st["labels"]:
            setname[lab - (1 if one else 0)] = st["name"]
    sections = {s["elementSet"]: s["layup"] for s in d.get("sections", [])}
    return dict(nodes=nodes, cells=cells, ori=ori, setname=setname, sections=sections)


def solid_matcolor(m):
    return [MAT_COLOR.get(mm, "#999999") for mm in m["setname"]]


def shell_typecolor(m, n_skin):
    out = []
    for k in range(len(m["cells"])):
        lam = m["sections"][m["setname"][k]]
        out.append(LAYUP_COLOR[layup_type(lam, k >= n_skin)])
    return out


# ---------------------------------------------------------------- boundary rows (2-D, nodes)
def boundary_row(end):
    so = load(os.path.join(YDIR, "iea22_boundary_%s_solid.yaml" % end))
    sh = load(os.path.join(YDIR, "iea22_boundary_%s_shell.yaml" % end))
    n_skin = sum(1 for c in sh["cells"] if len(c) == 2) - \
        sum(1 for c in sh["cells"] if len(c) == 2 and False)  # all lines; webs are the trailing ones
    # identify skin vs web lines: skin loop is a closed ring of the outer nodes; webs are the
    # trailing lines whose section layup is the web laminate (>=45 deg biax or matches web).
    # robust: a line is a web if BOTH its section is not on the outer convex-ish loop.  Simpler:
    # the first Nloop lines form the outer loop -> count until the loop closes back to node 0.
    first = sh["cells"][0][0]
    nloop = 1
    for k in range(1, len(sh["cells"])):
        if sh["cells"][k - 1][1] == sh["cells"][k][0]:
            nloop += 1
            if sh["cells"][k][1] == first:
                break
        else:
            break

    fig, axs = plt.subplots(1, 2, figsize=(17, 6.4))
    # solid: fill quads by material + node dots
    Xs = so["nodes"]
    for c, col in zip(so["cells"], solid_matcolor(so)):
        p = Xs[c][:, :2]
        axs[0].fill(p[:, 0], p[:, 1], color=col, edgecolor="k", linewidth=0.12)
    axs[0].plot(Xs[:, 0], Xs[:, 1], "o", ms=1.1, color="k", alpha=0.5)
    seen = [mm for mm in MAT_COLOR if mm in set(so["setname"])]
    axs[0].legend(handles=[Patch(color=MAT_COLOR[mm], label=mm) for mm in seen],
                  loc="upper right", fontsize=8, framealpha=0.9)
    axs[0].set_title("SOLID boundary r=%s  --  2-D quad section, by material" % end)
    # shell: colour each line by layup type + node dots
    Xh = sh["nodes"]
    used = set()
    for k, c in enumerate(sh["cells"]):
        lam = sh["sections"][sh["setname"][k]]
        t = layup_type(lam, k >= nloop)
        used.add(t)
        p = Xh[c][:, :2]
        axs[1].plot(p[:, 0], p[:, 1], color=LAYUP_COLOR[t], linewidth=2.4, solid_capstyle="round",
                    zorder=2)
    axs[1].plot(Xh[:, 0], Xh[:, 1], "o", ms=3.0, color="k", zorder=3)     # NODES
    stypes = [t for t in LAYUP_COLOR if t in used]
    axs[1].legend(handles=[Patch(color=LAYUP_COLOR[t], label=t) for t in stypes],
                  loc="upper right", fontsize=8, framealpha=0.9)
    axs[1].set_title("SHELL boundary r=%s  --  1-D contour on OML, by layup (nodes shown)" % end)
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(ODIR, "boundary_row_%s.png" % end)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out), flush=True)
    return sh, nloop


# ---------------------------------------------------------------- boundary orientation (e1/e2/e3)
def boundary_orient(kind, end):
    tag = "%s_%s" % (end, kind)
    m = load(os.path.join(YDIR, "iea22_boundary_%s.yaml" % tag))
    nodes, cells, ori = m["nodes"], m["cells"], m["ori"]
    cen = np.array([nodes[c].mean(0) for c in cells])
    diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    Larrow = 0.05 * diag
    # arrows on a subsample; section lies in x-y at z=const (e1 = +z out of plane)
    stride = max(1, len(cells) // 220)
    idx = np.arange(0, len(cells), stride)
    COMP = [(0, "e1 (beam axis +z)", "#d62728"), (1, "e2 (in-plane transverse)", "#1f77b4"),
            (2, "e3 (inward normal)", "0.0")]
    fig = plt.figure(figsize=(18, 5.6))
    for j, (comp, label, col) in enumerate(COMP):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        for c in cells:                                   # faint section outline
            p = nodes[c]
            ax.plot(p[:, 2], p[:, 0], p[:, 1], color="0.8", lw=0.3)
        P = cen[idx]; V = ori[idx, 3 * comp:3 * comp + 3] * Larrow
        ax.quiver(P[:, 2], P[:, 0], P[:, 1], V[:, 2], V[:, 0], V[:, 1],
                  color=col, linewidth=1.0, arrow_length_ratio=0.35, normalize=False)
        lo, hi = nodes[:, [2, 0, 1]].min(0), nodes[:, [2, 0, 1]].max(0)
        rng = (hi - lo); rng[rng < 1e-9] = Larrow * 4
        ax.set_xlim(lo[0] - Larrow, lo[0] + Larrow); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect((Larrow * 4, rng[1], rng[2]))
        except Exception:
            pass
        ax.view_init(elev=22, azim=-62); ax.set_axis_off()
        ax.set_title("%s boundary r=%s  --  %s" % (kind.upper(), end, label), fontsize=11)
    fig.tight_layout()
    out = os.path.join(ODIR, "boundary_orient_%s_%s.png" % (kind, end))
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out), flush=True)


# ---------------------------------------------------------------- taper comparison (PyVista)
def taper_comparison():
    try:
        import pyvista as pv
        from matplotlib.colors import ListedColormap
    except Exception as e:
        print("pyvista unavailable (%s) -> skip taper_comparison" % e, flush=True)
        return
    pv.OFF_SCREEN = True
    sh = load(os.path.join(YDIR, "iea22_seg_shell.yaml"))
    so = load(os.path.join(YDIR, "iea22_seg_solid.yaml"))
    # shell layup type (need skin/web split: a shell quad is a web if its layup is a web laminate;
    # derive from region by section makeup -- webs are the +-45-ish biax OR the trailing sets).
    # robust: a quad is a web if its two "span" edges are near-vertical in the section... instead
    # use: sections whose layup materials include 'biax' as a heuristic, else the taper stores it.
    n = len(sh["cells"])
    # web quads: those whose section is used ONLY by webs -- fall back to material heuristic
    sh_type = []
    for k in range(n):
        lam = sh["sections"][sh["setname"][k]]
        mats = [mm for mm, _t, _a in lam]
        is_web = any("biax" in mm for mm in mats) and not any("foam" in mm.lower() for mm in mats)
        sh_type.append(layup_type(lam, is_web))
    sh_cats = [t for t in LAYUP_COLOR if t in set(sh_type)]
    so_cats = sorted(set(so["setname"]),
                     key=lambda mm: list(MAT_COLOR).index(mm) if mm in MAT_COLOR else 99)

    def grid(m, celltype):
        cells = np.array(m["cells"])
        nc = len(cells)
        if celltype == "hex":
            vc = np.hstack([np.full((nc, 1), 8, np.int64), cells]).ravel()
            ct = np.full(nc, pv.CellType.HEXAHEDRON, np.uint8)
        else:
            vc = np.hstack([np.full((nc, 1), 4, np.int64), cells]).ravel()
            ct = np.full(nc, pv.CellType.QUAD, np.uint8)
        return pv.UnstructuredGrid(vc, ct, m["nodes"][:, [2, 0, 1]])

    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2100, 800))
    pl.subplot(0, 0)
    g = grid(sh, "quad"); idx = {c: i for i, c in enumerate(sh_cats)}
    g.cell_data["c"] = np.array([idx[t] for t in sh_type])
    pl.add_mesh(g, scalars="c", cmap=ListedColormap([LAYUP_COLOR[c] for c in sh_cats]),
                show_edges=True, edge_color="black", line_width=0.3,
                clim=[-0.5, len(sh_cats) - 0.5], show_scalar_bar=False)
    pl.add_legend([[c, LAYUP_COLOR[c]] for c in sh_cats], bcolor="white", size=(0.36, 0.22))
    pl.add_text("SHELL taper (by layup)", font_size=10)
    pl.camera_position = "iso"; pl.camera.azimuth = 12; pl.camera.elevation = -8; pl.camera.zoom(1.3)
    pl.subplot(0, 1)
    g2 = grid(so, "hex"); idx2 = {c: i for i, c in enumerate(so_cats)}
    g2.cell_data["c"] = np.array([idx2[mm] for mm in so["setname"]])
    pl.add_mesh(g2, scalars="c", cmap=ListedColormap([MAT_COLOR.get(c, "#999") for c in so_cats]),
                show_edges=True, edge_color="black", line_width=0.3,
                clim=[-0.5, len(so_cats) - 0.5], show_scalar_bar=False)
    pl.add_legend([[c, MAT_COLOR.get(c, "#999")] for c in so_cats], bcolor="white", size=(0.36, 0.22))
    pl.add_text("SOLID taper (by material)", font_size=10)
    pl.camera_position = "iso"; pl.camera.azimuth = 12; pl.camera.elevation = -8; pl.camera.zoom(1.3)
    pl.link_views()
    out = os.path.join(ODIR, "taper_comparison.png")
    pl.screenshot(out); pl.close()
    print("wrote", os.path.basename(out), flush=True)


if __name__ == "__main__":
    for end in ("L", "R"):
        boundary_row(end)
    for kind in ("solid", "shell"):
        for end in ("L", "R"):
            boundary_orient(kind, end)
    taper_comparison()
    print("all comparison figures ->", ODIR, flush=True)
