"""_gmsh3_test.py -- can gmsh's OWN 3-D mesher (generate(3)) tet-fill the lofted webbed solid
directly, now that the mesh size is WALL-based (not the old span-based 1 m that hung it)?
Also tests occ.fragment of SEPARATE region volumes so each tet is tagged by physical volume
(= clean material regions, no centroid tagging)."""
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
tet_size = max(0.5 * wall, 0.02 * chord); smin = max(0.35 * wall, 0.004 * chord)
print("wall=%.4f tet_size=%.4f smin=%.4f" % (wall, tet_size, smin), flush=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
occ = gmsh.model.occ


def loop(P, z):
    tags = [occ.addPoint(float(p[0]), float(p[1]), z) for p in P]
    return occ.addCurveLoop([occ.addSpline(tags + [tags[0]])])


# --- per-hoop-SEGMENT skin slabs (OML..IML full thickness), one OCC volume per interval, so
#     every tet is tagged by its segment volume = CLEAN material regions (no centroid mixing) ---
hk = np.asarray(st1["hoop_kind"])
NC = len(hk)
ranges = []
start = 0
for i in range(1, NC + 1):
    if i == NC or hk[i] != hk[i - 1]:
        ranges.append((int(hk[start]), start, i - 1)); start = i
print("hoop intervals: %d" % len(ranges), flush=True)


def slab(Oml, Iml, i0, i1, z):
    pts = [Oml[i] for i in range(i0, i1 + 1)] + [Iml[i] for i in range(i1, i0 - 1, -1)]
    tg = [occ.addPoint(float(p[0]), float(p[1]), z) for p in pts]
    ls = [occ.addLine(tg[k], tg[k + 1]) for k in range(len(tg) - 1)] + [occ.addLine(tg[-1], tg[0])]
    return occ.addCurveLoop(ls)


skin_vols = []
for (k, i0, i1) in ranges:
    v = occ.addThruSections([slab(OML0, IML0, i0, i1, z1), slab(OML1, IML1, i0, i1, z2)], makeSolid=True)
    skin_vols += v
print("skin slab volumes: %d" % len(skin_vols), flush=True)

inner = occ.addThruSections([loop(IML0, z1), loop(IML1, z2)], makeSolid=True)   # IML plug to trim webs


def web_loop(cs, w, z):
    Pa = np.asarray(cs["nodes"][w["a"]], float); Pb = np.asarray(cs["nodes"][w["b"]], float)
    d = Pb - Pa; d = d / (np.hypot(*d) or 1e-30); perp = np.array([-d[1], d[0]])
    h = 0.5 * _thick(_lam_tuple(cs, w["lam"])); e = 0.05 * cs["chord"]
    Pa, Pb = Pa - e * d, Pb + e * d
    cor = [Pa - h * perp, Pb - h * perp, Pb + h * perp, Pa + h * perp]
    tg = [occ.addPoint(float(c[0]), float(c[1]), z) for c in cor]
    return occ.addCurveLoop([occ.addLine(tg[k], tg[(k + 1) % 4]) for k in range(4)])


parts = list(skin_vols)
for wi in range(nw):
    box = occ.addThruSections([web_loop(cs1, cs1["webs"][wi], z1),
                               web_loop(cs2, cs2["webs"][wi], z2)], makeSolid=True)
    strip, _ = occ.intersect(box, occ.copy(inner), removeObject=True, removeTool=True)
    parts += strip
occ.remove(inner, recursive=True)
print("total parts (skin slabs + webs): %d" % len(parts), flush=True)

# FRAGMENT (not fuse): keep the skin + 3 webs as SEPARATE but conformal volumes, so every
# tet belongs to exactly one region -> material by physical volume, no centroid intermixing.
frag, fmap = occ.fragment(parts, [])
occ.removeAllDuplicates()                                     # weld near-coincident web-skin faces
occ.synchronize()
vols = gmsh.model.getEntities(3)
print("fragment -> %d volumes" % len(vols), flush=True)
for i, (d3, tag) in enumerate(vols):
    gmsh.model.addPhysicalGroup(3, [tag], tag)

gmsh.option.setNumber("Mesh.MeshSizeMin", smin)
gmsh.option.setNumber("Mesh.MeshSizeMax", tet_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
t0 = time.time()
gmsh.model.mesh.generate(3)
etypes, etags, enodes = gmsh.model.mesh.getElements(3)
ntet = sum(len(en) // 4 for et, en in zip(etypes, enodes) if et == 4)
print("generate(3) DONE in %.1fs : %d tets across %d physical volumes"
      % (time.time() - t0, ntet, len(vols)), flush=True)
gmsh.finalize()
