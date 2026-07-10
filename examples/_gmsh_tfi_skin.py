"""_gmsh_tfi_skin.py -- STAGE 1: mesh the SKIN as per-hoop-CELL transfinite blocks (one block
between each adjacent ring-node pair, straight edges) -> recombine -> structured HEX.
Params: NSP (span div), NPL (through-thickness div), REFINE (hoop div per cell)."""
import os
import sys
import time

import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_station

NSP, NPL, REFINE = 6, 4, 1
blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
cs1 = build_cross_section(blade, 0.1967, mesh_size=0.02)
cs2 = build_cross_section(blade, 0.2470, mesh_size=0.02)
z1, z2 = 0.1967 * 137.0, 0.2470 * 137.0
nr, nw = 4, 3
skel = section_skeleton([cs1, cs2], mesh_size=0.02, nw=nw)
st1 = build_station(cs1, skel, 0, nr=nr); st2 = build_station(cs2, skel, 1, nr=nr)
O0, I0 = np.asarray(st1["rings"][0]), np.asarray(st1["rings"][nr])
O1, I1 = np.asarray(st2["rings"][0]), np.asarray(st2["rings"][nr])
NC = len(O0)
print("NC=%d hoop cells" % NC, flush=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
geo = gmsh.model.geo
P = {}
for s, (O, I, z) in enumerate([(O0, I0, z1), (O1, I1, z2)]):
    for i in range(NC):
        P[(s, "O", i)] = geo.addPoint(float(O[i][0]), float(O[i][1]), z)
        P[(s, "I", i)] = geo.addPoint(float(I[i][0]), float(I[i][1]), z)

lc = {}


def L(a, b):
    if (a, b) in lc:
        return lc[(a, b)]
    if (b, a) in lc:
        return -lc[(b, a)]
    t = geo.addLine(a, b); lc[(a, b)] = t
    return t


def face(pts):
    return geo.addSurfaceFilling([geo.addCurveLoop([L(pts[k], pts[(k + 1) % 4]) for k in range(4)])])


NH = REFINE
for i in range(NC):
    j = (i + 1) % NC
    a, b = P[(0, "O", i)], P[(0, "O", j)]
    d, c = P[(0, "I", i)], P[(0, "I", j)]
    e, f = P[(1, "O", i)], P[(1, "O", j)]
    h, g = P[(1, "I", i)], P[(1, "I", j)]
    z0 = face([a, b, c, d]); z1s = face([e, f, g, h])
    fO = face([a, b, f, e]); fI = face([d, c, g, h])
    si = face([a, d, h, e]); sj = face([b, c, g, f])
    vol = geo.addVolume([geo.addSurfaceLoop([z0, z1s, fO, fI, si, sj])])
    for (p, q), n in [((a, b), NH), ((d, c), NH), ((e, f), NH), ((h, g), NH),
                      ((a, d), NPL), ((b, c), NPL), ((e, h), NPL), ((f, g), NPL),
                      ((a, e), NSP), ((b, f), NSP), ((c, g), NSP), ((d, h), NSP)]:
        geo.mesh.setTransfiniteCurve(abs(L(p, q)), n + 1)
    for s in [z0, z1s, fO, fI, si, sj]:
        geo.mesh.setTransfiniteSurface(s); geo.mesh.setRecombine(2, s)
    geo.mesh.setTransfiniteVolume(vol)

geo.removeAllDuplicates()
geo.synchronize()
t0 = time.time()
gmsh.model.mesh.generate(3)
etypes, etags, enodes = gmsh.model.mesh.getElements(3)
name = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
tot = {name.get(et, et): len(etg) for et, etg in zip(etypes, etags)}
ntag, ncoord, _ = gmsh.model.mesh.getNodes()
Pn = ncoord.reshape(-1, 3); idx = {int(t): i for i, t in enumerate(ntag)}
ninv = 0
for et, en in zip(etypes, enodes):
    if et == 5:
        H = np.array([[idx[int(x)] for x in row] for row in en.reshape(-1, 8)])
        X = Pn[H]
        e1 = X[:, 1] - X[:, 0]; e2 = X[:, 3] - X[:, 0]; e3 = X[:, 4] - X[:, 0]
        ninv = int((np.einsum("ij,ij->i", np.cross(e1, e2), e3) <= 0).sum())
print("transfinite skin DONE in %.1fs : %s ; inverted hex=%d" % (time.time() - t0, tot, ninv), flush=True)
gmsh.finalize()
