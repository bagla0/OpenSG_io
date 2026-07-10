"""export_layup_views.py -- colour the IEA-22 taper meshes by their composite makeup:

  * SHELL mesh coloured by LAYUP TYPE (spar cap / foam-core panel / reinforcement /
    plain skin / shear web)   -- PyVista, with a legend;
  * SOLID mesh coloured by MATERIAL of each hex                       -- PyVista, legend;
  * the BOUNDARY cross-sections (root end) of solid and shell in ONE horizontal row.

The shell sits on the OML (default reference = "OML"), the same outer surface the solid
is built inward from, so the shell contour coincides with the solid's outer edge.
Outputs staged into the OneDrive deliverable folder.
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
import sys
sys.path.insert(0, REPO)
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import (hex_between_sections, shell_between_sections,
                                solid_yaml_payload, solid_boundary_payload,
                                shell_boundary_payload)

ONEDRIVE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_data\IEA_taper_segment"
os.makedirs(ONEDRIVE, exist_ok=True)
CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# ---- consistent colours -----------------------------------------------------------
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


# ---- build meshes in memory -------------------------------------------------------
WINDIO = os.path.join(REPO, "examples", "data", "IEA-22-280-RWT.yaml")
blade = load_blade(WINDIO)
cs1 = build_cross_section(blade, 0.20, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.30, mesh_size=0.02)
try:
    ra = blade.osh["reference_axis"]["z"]
    z1 = float(np.interp(0.20, ra["grid"], ra["values"]))
    z2 = float(np.interp(0.30, ra["grid"], ra["values"]))
except Exception:
    z1, z2 = 0.20 * 137.0, 0.30 * 137.0
res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=2, mesh_size=0.02)
shell = shell_between_sections(res, cs1, cs2, reference="OML")
_oris, hmats = solid_yaml_payload(res, cs1, cs2)
nodes, hexes = res["nodes"], res["hexes"]

# per-quad shell layup type
shell_type = [layup_type(shell["sections"][shell["qsec"][k]], shell["qweb"][k])
              for k in range(len(shell["quads"]))]
shell_types = [t for t in LAYUP_COLOR if t in set(shell_type)]
# per-hex solid material
mats = sorted(set(hmats), key=lambda m: list(MAT_COLOR).index(m) if m in MAT_COLOR else 99)


# ---- PyVista renders --------------------------------------------------------------
def pv_render(nodes, cells, celltype, cat_of_cell, cats, color_of, png, title):
    import pyvista as pv
    from matplotlib.colors import ListedColormap
    pv.OFF_SCREEN = True
    n = len(cells)
    if celltype == "hex":
        vc = np.hstack([np.full((n, 1), 8, np.int64), cells]).ravel()
        ct = np.full(n, pv.CellType.HEXAHEDRON, np.uint8)
    else:
        vc = np.hstack([np.full((n, 1), 4, np.int64), cells]).ravel()
        ct = np.full(n, pv.CellType.QUAD, np.uint8)
    grid = pv.UnstructuredGrid(vc, ct, np.asarray(nodes, float)[:, [2, 0, 1]])
    idx = {c: i for i, c in enumerate(cats)}
    grid.cell_data["cat"] = np.array([idx[c] for c in cat_of_cell])
    pl = pv.Plotter(off_screen=True, window_size=(1500, 750))
    pl.add_mesh(grid, scalars="cat", cmap=ListedColormap([color_of[c] for c in cats]),
                show_edges=True, edge_color="black", line_width=0.35,
                clim=[-0.5, len(cats) - 0.5], show_scalar_bar=False)
    pl.add_legend([[c, color_of[c]] for c in cats], bcolor="white", size=(0.32, 0.24),
                  loc="upper right", face="rectangle")
    pl.add_text(title, font_size=11)
    pl.camera_position = "iso"; pl.camera.azimuth = 12; pl.camera.elevation = -8
    pl.camera.zoom(1.25)
    pl.screenshot(png); pl.close()
    print("wrote", os.path.basename(png), flush=True)


pv_render(shell["nodes"], shell["quads"], "quad", shell_type, shell_types, LAYUP_COLOR,
          os.path.join(ONEDRIVE, "shell_layup.png"),
          "IEA-22 r=0.2->0.3  SHELL mesh coloured by LAYUP (OML reference)")
pv_render(nodes, hexes, "hex", hmats, mats, MAT_COLOR,
          os.path.join(ONEDRIVE, "solid_material.png"),
          "IEA-22 r=0.2->0.3  SOLID mesh coloured by MATERIAL")


# ---- boundary cross-sections in one horizontal row --------------------------------
def parse_nodes(d):
    return np.array([[float(v) for v in r[0].split()] for r in d["nodes"]])


def parse_cells(d):
    return [[int(v) - 1 for v in r[0].split()] for r in d["elements"]]


def set_of_elem(d, n):
    s = ["?"] * n
    for st in d["sets"]["element"]:
        for lab in st["labels"]:
            s[lab - 1] = st["name"]
    return s


bs = solid_boundary_payload(res, cs1, cs2, 0, blade, _mat_block)     # solid boundary L (root)
bh = shell_boundary_payload(res, shell, cs1, cs2, 0, blade, _mat_block)
bsx = parse_nodes(bs); bsq = parse_cells(bs); bsm = set_of_elem(bs, len(bsq))
bhx = parse_nodes(bh); bhl = parse_cells(bh); bhsec = {s["elementSet"]: s["layup"] for s in bh["sections"]}
bhset = set_of_elem(bh, len(bhl))
# shell line layup type: web lines vs skin (by whether the section is a web layup)
web_layups = {("web_%d" % 0)}  # placeholder; identify webs geometrically below

fig, axs = plt.subplots(1, 2, figsize=(17, 6.2))
# solid section: fill quads by material
for q, m in zip(bsq, bsm):
    poly = bsx[q][:, :2]
    axs[0].fill(poly[:, 0], poly[:, 1], color=MAT_COLOR.get(m, "#999999"),
                edgecolor="k", linewidth=0.15)
axs[0].set_title("SOLID boundary (root r=0.2)  --  2-D quad section, by material")
seen = [m for m in MAT_COLOR if m in set(bsm)]
axs[0].legend(handles=[Patch(color=MAT_COLOR[m], label=m) for m in seen],
              loc="upper right", fontsize=8, framealpha=0.9)

# shell section: draw lines; colour a line by the layup type of its section
line_type = []
for k, ln in enumerate(bhl):
    lam = bhsec[bhset[k]]
    # a web line is (near) vertical/internal -> classify by geometry: is it on the outer loop?
    line_type.append(lam)
# classify each line: web if its section materials are the web laminate (biax) OR it is
# one of the last lines (webs appended after the skin loop); simplest: skin loop first
NC_lines = len(bhl)
# reconstruct: skin loop has len(skin_loop) lines, rest are web lines
n_skin = len(shell["sec2d"]["skin_loop"])
for k, ln in enumerate(bhl):
    lam = bhsec[bhset[k]]
    is_web = k >= n_skin
    t = layup_type(lam, is_web)
    p = bhx[ln][:, :2]
    axs[1].plot(p[:, 0], p[:, 1], color=LAYUP_COLOR[t], linewidth=2.2, solid_capstyle="round")
axs[1].set_title("SHELL boundary (root r=0.2)  --  1-D contour on OML, by layup")
stypes = [t for t in LAYUP_COLOR if t in {layup_type(bhsec[bhset[k]], k >= n_skin) for k in range(len(bhl))}]
axs[1].legend(handles=[Patch(color=LAYUP_COLOR[t], label=t) for t in stypes],
              loc="upper right", fontsize=8, framealpha=0.9)
for ax in axs:
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(ONEDRIVE, "boundary_row.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote boundary_row.png", flush=True)
print("staged ->", ONEDRIVE, flush=True)
