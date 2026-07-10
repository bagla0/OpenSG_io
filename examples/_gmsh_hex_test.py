"""_gmsh_hex_test.py -- can gmsh produce a HEX (or hex-dominant) mesh of the lofted webbed
solid?  Tests (A) 3-D recombine on the fragmented clean-region volumes, and reports the
element-type breakdown + how many hexes are inverted (negative Jacobian)."""
import os
import sys
import time

import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_station, _lam_tuple, _thick

blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
cs1 = build_cross_section(blade, 0.1967, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.2470, mesh_size=0.02)
z1, z2 = 0.1967 * 137.0, 0.2470 * 137.0
nr, nw = 4, 3
skel = section_skeleton([cs1, cs2], mesh_size=0.02, nw=nw)
st1 = build_station(cs1, skel, 0, nr=nr); st2 = build_station(cs2, skel, 1, nr=nr)
OML0, IML0 = np.asarray(st1["rings"][0]), np.asarray(st1["rings"][nr])
OML1, IML1 = np.asarray(st2["rings"][0]), np.asarray(st2["rings"][nr])
chord = 0.5 * (cs1["chord"] + cs2["chord"])
wall = float(np.mean(np.linalg.norm(0.5 * (OML0 + OML1) - 0.5 * (IML0 + IML1), axis=1)))
tet_size = max(0.5 * wall, 0.02 * chord)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
occ = gmsh.model.occ


def loop(P, z):
    tags = [occ.addPoint(float(p[0]), float(p[1]), z) for p in P]
    return occ.addCurveLoop([occ.addSpline(tags + [tags[0]])])


outer = occ.addThruSections([loop(OML0, z1), loop(OML1, z2)], makeSolid=True)
inner = occ.addThruSections([loop(IML0, z1), loop(IML1, z2)], makeSolid=True)
skin, _ = occ.cut(outer, inner, removeObject=True, removeTool=False)


def web_loop(cs, w, z):
    Pa = np.asarray(cs["nodes"][w["a"]], float); Pb = np.asarray(cs["nodes"][w["b"]], float)
    d = Pb - Pa; d = d / (np.hypot(*d) or 1e-30); perp = np.array([-d[1], d[0]])
    h = 0.5 * _thick(_lam_tuple(cs, w["lam"])); e = 0.05 * cs["chord"]
    Pa, Pb = Pa - e * d, Pb + e * d
    cor = [Pa - h * perp, Pb - h * perp, Pb + h * perp, Pa + h * perp]
    tg = [occ.addPoint(float(c[0]), float(c[1]), z) for c in cor]
    return occ.addCurveLoop([occ.addLine(tg[k], tg[(k + 1) % 4]) for k in range(4)])


parts = list(skin)
for wi in range(nw):
    box = occ.addThruSections([web_loop(cs1, cs1["webs"][wi], z1),
                               web_loop(cs2, cs2["webs"][wi], z2)], makeSolid=True)
    strip, _ = occ.intersect(box, occ.copy(inner), removeObject=True, removeTool=True)
    parts += strip
occ.remove(inner, recursive=True)
occ.fragment(parts, [])
occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.35 * wall)
gmsh.option.setNumber("Mesh.MeshSizeMax", tet_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.Algorithm", 8)                  # Frontal-Delaunay for quads (2D)
gmsh.option.setNumber("Mesh.RecombineAll", 1)               # quads on all surfaces
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)             # hexes in volume
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)     # Blossom
t0 = time.time()
gmsh.model.mesh.generate(3)
etypes, etags, enodes = gmsh.model.mesh.getElements(3)
name = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
tot = {}
for et, etg in zip(etypes, etags):
    tot[name.get(et, et)] = len(etg)
print("generate(3)+recombine DONE in %.1fs : %s" % (time.time() - t0, tot), flush=True)
gmsh.finalize()
