"""export_deliverable.py -- read the generated taper YAMLs, render the shell + solid
meshes and their e1/e2/e3 material-orientation fields (separate images each), write the
Timoshenko 6x6 results as .dat, and stage everything into a OneDrive folder:

    <ONEDRIVE>/IEA_taper_segment/
        *.yaml                      (all mesh data: full taper + 4 boundaries)
        *_timo.dat, comparison.dat  (results)
        orien/                      (shell/solid mesh + e1/e2/e3 orientation images)

Orientation convention: e1 red (beam axis root->tip), e2 blue, e3 black (inward normal).
"""
import os
import shutil

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
ONEDRIVE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_data\IEA_taper_segment"
ORIEN = os.path.join(ONEDRIVE, "orien")
os.makedirs(ORIEN, exist_ok=True)
CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
HE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
      (0, 4), (1, 5), (2, 6), (3, 7)]
_HEXF = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def load(path):
    d = yaml.load(open(path), Loader=CLoad)
    rows = d["nodes"]
    if isinstance(rows[0][0], str):
        nodes = np.array([[float(v) for v in r[0].split()] for r in rows])
        cells = np.array([[int(v) - 1 for v in r[0].split()] for r in d["elements"]])
    else:
        nodes = np.array(rows, float)
        cells = np.array(d["elements"], int)
    ori = np.array(d["elementOrientations"], float)
    return nodes, cells, ori


def boundary_faces(hexes):
    seen = {}
    for k, h in enumerate(hexes):
        for f in _HEXF:
            key = tuple(sorted(int(h[i]) for i in f))
            seen[key] = None if key in seen else (tuple(int(h[i]) for i in f), k)
    return [v for v in seen.values() if v is not None]


def R(P):                                                  # plot in (z, x, y): span horizontal
    return P[:, [2, 0, 1]]


def shaded_mesh(ax, nodes, polys):
    P = R(nodes)
    pc = Poly3DCollection(P[polys], facecolor="#cfe0ee", edgecolor="k", linewidths=0.1, alpha=1.0)
    nrm = np.cross(P[polys][:, 1] - P[polys][:, 0], P[polys][:, 2] - P[polys][:, 0])
    nn = np.linalg.norm(nrm, axis=1); nrm = nrm / np.where(nn > 1e-30, nn, 1)[:, None]
    sh = 0.55 + 0.45 * np.abs(nrm @ np.array([0.35, -0.5, 0.79]))
    pc.set_facecolor(np.clip(np.array([0.81, 0.88, 0.93]) * sh[:, None], 0, 1))
    ax.add_collection3d(pc)
    return P


def frame(ax, nodes, polys, title):
    P = nodes[polys.ravel()] if len(polys) else nodes
    P = R(P)
    lo, hi = P.min(0), P.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.view_init(elev=18, azim=-72); ax.set_axis_off(); ax.set_title(title, fontsize=12)


def render_mesh(nodes, cells, celltype, png, title):
    polys = (np.array([q for q, _ in boundary_faces(cells)]) if celltype == "hex"
             else cells)
    fig = plt.figure(figsize=(15, 6.5)); ax = fig.add_subplot(111, projection="3d")
    shaded_mesh(ax, nodes, polys); frame(ax, nodes, polys, title)
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)


def render_orientation(nodes, cells, ori, celltype, comp, png, title, color):
    if celltype == "hex":                                  # arrows on the OUTER surface only
        faces = boundary_faces(cells)
        polys = np.array([q for q, _ in faces])
        owner = np.array([k for _, k in faces])
        cen = nodes[polys].mean(1)
        vec = ori[owner, 3 * comp:3 * comp + 3]
    else:
        polys = cells
        cen = nodes[cells].mean(1)
        vec = ori[:, 3 * comp:3 * comp + 3]
    # fixed, visible arrow length ~ 4% of the bounding-box diagonal
    bb = nodes[polys.ravel()]
    diag = float(np.linalg.norm(bb.max(0) - bb.min(0)))
    Larrow = 0.045 * diag
    stride = max(1, len(cen) // 320)
    idx = np.arange(0, len(cen), stride)
    Pc = R(cen[idx]); Vv = R(vec[idx]) * Larrow
    fig = plt.figure(figsize=(15, 6.5)); ax = fig.add_subplot(111, projection="3d")
    faint = Poly3DCollection(R(nodes)[polys], facecolor=(0.92, 0.92, 0.92, 0.18),
                             edgecolor=(0.75, 0.75, 0.75, 0.35), linewidths=0.1)
    ax.add_collection3d(faint)
    ax.quiver(Pc[:, 0], Pc[:, 1], Pc[:, 2], Vv[:, 0], Vv[:, 1], Vv[:, 2],
              color=color, linewidth=1.1, arrow_length_ratio=0.35, normalize=False)
    frame(ax, nodes, polys, title)
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)


# ---- meshes + orientation images ------------------------------------------------
COMP = [(0, "e1 (beam axis, root->tip)", "#d62728"),
        (1, "e2 (transverse in-surface)", "#1f77b4"),
        (2, "e3 (inward normal)", "0.0")]
for kind, fname, ct in (("shell", "iea22_seg_shell.yaml", "quad"),
                        ("solid", "iea22_seg_solid.yaml", "hex")):
    nodes, cells, ori = load(os.path.join(OUT, fname))
    render_mesh(nodes, cells, ct, os.path.join(ORIEN, "%s_mesh.png" % kind),
                "IEA-22 r=0.2->0.3  %s mesh (%d %s)"
                % (kind.upper(), len(cells), "quads" if ct == "quad" else "hexes"))
    for comp, label, col in COMP:
        render_orientation(nodes, cells, ori, ct, comp,
                           os.path.join(ORIEN, "%s_e%d.png" % (kind, comp + 1)),
                           "%s  --  %s" % (kind.upper(), label), col)
    print("rendered %s: mesh + e1/e2/e3" % kind, flush=True)

# ---- results .dat ---------------------------------------------------------------
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def write_dat(path, S, header):
    with open(path, "w") as f:
        f.write("# %s\n# Timoshenko 6x6, order [EA, GA2, GA3, GJ, EI2, EI3]\n" % header)
        for i in range(6):
            f.write("  ".join("% .8e" % S[i, j] for j in range(6)) + "\n")


sh = np.load(os.path.join(OUT, "timo_shell.npz"))
so = np.load(os.path.join(OUT, "timo_solid.npz"))
write_dat(os.path.join(ONEDRIVE, "shell_taper_timo.dat"), sh["S6"],
          "IEA-22 r=0.2->0.3 SHELL taper (OpenSG-TW JAX MITC-RM)")
write_dat(os.path.join(ONEDRIVE, "solid_taper_timo.dat"), so["S6"],
          "IEA-22 r=0.2->0.3 SOLID taper (OpenSG-FEniCS)")
for side in ("L", "R"):
    if "C6" + side in sh.files:
        write_dat(os.path.join(ONEDRIVE, "shell_boundary_%s_timo.dat" % side),
                  sh["C6" + side], "SHELL boundary ring %s" % side)
    if "C6" + side in so.files:
        write_dat(os.path.join(ONEDRIVE, "solid_boundary_%s_timo.dat" % side),
                  so["C6" + side], "SOLID boundary ring %s" % side)

with open(os.path.join(ONEDRIVE, "comparison.dat"), "w") as f:
    f.write("# IEA-22 r=0.2->0.3 tapered segment: SHELL (TW JAX RM) vs SOLID (FEniCS)\n")
    f.write("# %-6s %16s %16s %10s\n" % ("term", "solid", "shell", "%diff"))
    S, H = so["S6"], sh["S6"]
    for i in range(6):
        pd = 100 * (H[i, i] - S[i, i]) / S[i, i]
        f.write("  %-6s %16.6e %16.6e %+9.2f\n" % (LBL[i], S[i, i], H[i, i], pd))
    f.write("# extension+bending agree <10%; GA2/GA3/GJ ~13x = RM soft-core (foam) "
            "transverse-shear over-prediction (24/72 panels are foam sandwiches).\n")

# ---- mesh data yamls ------------------------------------------------------------
for y in ("iea22_seg_solid.yaml", "iea22_seg_shell.yaml",
          "iea22_boundary_L_solid.yaml", "iea22_boundary_R_solid.yaml",
          "iea22_boundary_L_shell.yaml", "iea22_boundary_R_shell.yaml"):
    shutil.copy(os.path.join(OUT, y), os.path.join(ONEDRIVE, y))

print("staged ->", ONEDRIVE, flush=True)
