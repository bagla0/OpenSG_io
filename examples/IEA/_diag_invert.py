"""Diagnose inverted hexes on a tapering segment: ply-conforming vs uniform layers,
and WHERE the inversions are (skin vs web, which layer)."""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import hex_between_sections
from opensg_io.conformity import min_scaled_jacobian, HEX_CORNERS

r1, r2 = float(sys.argv[1]), float(sys.argv[2])
blade = load_blade(os.path.expanduser("~/OpenSG_io/examples/data/IEA-22-280-RWT.yaml"))
cs1 = build_cross_section(blade, r1, mesh_size=0.02)
cs2 = build_cross_section(blade, r2, mesh_size=0.02)
z1, z2 = r1 * 137.0, r2 * 137.0


def per_hex_msj(nodes, hexes):
    X = np.asarray(nodes)[np.asarray(hexes)]
    mins = np.full(len(X), np.inf)
    for (c, a, b, t) in HEX_CORNERS:
        e1 = X[:, a] - X[:, c]; e2 = X[:, b] - X[:, c]; e3 = X[:, t] - X[:, c]
        det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        sc = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
        mins = np.minimum(mins, det / np.where(sc > 1e-300, sc, 1.0))
    return mins


NW = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for mode in ("plyconform", "uniform"):
    os.environ.pop("OPENSG_UNIFORM_LAYERS", None)
    if mode == "uniform":
        os.environ["OPENSG_UNIFORM_LAYERS"] = "1"
    res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=NW, mesh_size=0.02)
    nodes, hexes, htag = res["nodes"], res["hexes"], res["htag"]
    msj = per_hex_msj(nodes, hexes)
    bad = np.where(msj <= 0)[0]
    kinds = Counter((htag[k][0], htag[k][2] if htag[k][0] == "skin" else "col%d" % htag[k][2]) for k in bad)
    print("r=%.2f-%.2f  %-11s: min SJ %.3f, %d inverted ; by (region,layer): %s"
          % (r1, r2, mode, msj.min(), len(bad), dict(kinds)))
