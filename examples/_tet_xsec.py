"""_tet_xsec.py -- slice the clean-region tet solid at mid-span and color by MATERIAL, to
verify regions are SHARP (spar cap / panels / skin plies / webs) and not intermixed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opensg_io  # noqa  (sets software-GL env before pyvista)
import pyvista as pv
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.tapered_tet import windio_taper_tets
from opensg_io.render3d import PAL

blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
r1, r2 = 0.1967, 0.2470
z1, z2 = r1 * 137.0, r2 * 137.0
cs1 = build_cross_section(blade, r1, mesh_size=0.02)
cs2 = build_cross_section(blade, r2, mesh_size=0.02)
nodes, tets, oris, hmats = windio_taper_tets(cs1, cs2, z1, z2, nr=4, nw=3, mesh_size=0.02)
mat_names = sorted(set(hmats)); hset = {m: i for i, m in enumerate(mat_names)}
setmap = np.array([hset[m] for m in hmats])
print("materials:", mat_names, flush=True)

cells = np.hstack([np.full((len(tets), 1), 4, np.int64), tets]).ravel()
ct = np.full(len(tets), pv.CellType.TETRA, np.uint8)
grid = pv.UnstructuredGrid(cells, ct, np.asarray(nodes, float))
grid.cell_data["mat"] = setmap
sl = grid.slice(normal="z", origin=(0, 0, 0.5 * (z1 + z2)))
print("slice cells:", sl.n_cells, flush=True)

from matplotlib.colors import ListedColormap
pl = pv.Plotter(off_screen=True, window_size=(1600, 560))
pl.add_mesh(sl, scalars="mat", cmap=ListedColormap(PAL[np.arange(len(mat_names)) % len(PAL)]),
            show_edges=True, edge_color="black", line_width=0.4, show_scalar_bar=False,
            clim=[0, len(mat_names) - 1])
pl.add_text("mid-span cross-section by material (clean regions)", font_size=10)
pl.view_xy()
pl.camera.zoom(1.5)
out = "examples/mesh_out/r020_025_tet_xsec.png"
pl.screenshot(out)
pl.close()
print("wrote", out, "bytes=%d" % os.path.getsize(out), flush=True)
