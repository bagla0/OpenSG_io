"""_gmsh_tfi_full.py -- STAGE 2: full section (skin + webs) as gmsh transfinite HEX blocks, one
per 2-D quad cell of the proven build_section_mesh, lofted with NSP span divisions.  Reports the
element breakdown + inverted-hex count -- tells us whether the WEB cells fold under transfinite
(they fold under the hand-rolled linear loft)."""
import os
import sys
import time

import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh

NSP = int(os.environ.get("NSP", "6"))
MS = float(os.environ.get("MS", "0.02"))
R1 = float(os.environ.get("R1", "0.1967")); R2 = float(os.environ.get("R2", "0.2470"))
blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
cs1 = build_cross_section(blade, R1, mesh_size=MS)
cs2 = build_cross_section(blade, R2, mesh_size=MS)
z1, z2 = R1 * 137.0, R2 * 137.0
skel = section_skeleton([cs1, cs2], mesh_size=MS, nw=3)
sec = build_section_mesh([cs1, cs2], skel, nr=4)
P1, P2 = sec["stations"]; faces = np.asarray(sec["faces2d"]); ftag = sec["ftag"]
NP = sec["NP"]
print("section: %d nodes, %d quad cells (%d web)" %
      (NP, len(faces), sum(t[0] == "web" for t in ftag)), flush=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
geo = gmsh.model.geo
p0 = [geo.addPoint(float(P1[n, 0]), float(P1[n, 1]), z1) for n in range(NP)]
p1 = [geo.addPoint(float(P2[n, 0]), float(P2[n, 1]), z2) for n in range(NP)]

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


t0 = time.time()
for fc in faces:
    a, b, c, d = (p0[fc[0]], p0[fc[1]], p0[fc[2]], p0[fc[3]])
    e, f, g, h = (p1[fc[0]], p1[fc[1]], p1[fc[2]], p1[fc[3]])
    z0 = face([a, b, c, d]); zt = face([e, f, g, h])
    s1 = face([a, b, f, e]); s2 = face([b, c, g, f]); s3 = face([c, d, h, g]); s4 = face([d, a, e, h])
    vol = geo.addVolume([geo.addSurfaceLoop([z0, zt, s1, s2, s3, s4])])
    for (pp, qq) in [(a, b), (b, c), (c, d), (d, a), (e, f), (f, g), (g, h), (h, e)]:
        geo.mesh.setTransfiniteCurve(abs(L(pp, qq)), 2)
    for (pp, qq) in [(a, e), (b, f), (c, g), (d, h)]:
        geo.mesh.setTransfiniteCurve(abs(L(pp, qq)), NSP + 1)
    for s in [z0, zt, s1, s2, s3, s4]:
        geo.mesh.setTransfiniteSurface(s); geo.mesh.setRecombine(2, s)
    geo.mesh.setTransfiniteVolume(vol)

geo.removeAllDuplicates()
geo.synchronize()
print("geometry built in %.1fs; meshing" % (time.time() - t0), flush=True)
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
        ninv += int((np.einsum("ij,ij->i", np.cross(e1, e2), e3) <= 0).sum())
print("transfinite FULL r=%.3f->%.3f DONE in %.1fs : %s ; inverted hex=%d"
      % (R1, R2, time.time() - t0, tot, ninv), flush=True)
gmsh.finalize()
