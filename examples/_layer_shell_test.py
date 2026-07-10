"""_layer_shell_test.py -- does partitioning the skin into npl nested guide-surface SHELLS force
~npl elements through the wall at COARSE in-plane cost (no explosion)?  Builds nr nested shells
between rings[l]..rings[l+1] + webs, fragment, generate(3) at coarse in-plane size, and reports
the tet count + a through-wall element count along a radial probe."""
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
R1 = np.asarray(st1["rings"]); R2 = np.asarray(st2["rings"])       # (nr+1, NC, 2)
chord = 0.5 * (cs1["chord"] + cs2["chord"])
wall = float(np.mean(np.linalg.norm(R1[0] - R1[nr], axis=1)))
inplane = 0.5 * wall                                              # COARSE in-plane target
print("wall=%.4f  layer_thk~%.4f  in-plane target=%.4f" % (wall, wall / nr, inplane), flush=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
occ = gmsh.model.occ


def loop(P, z):
    tags = [occ.addPoint(float(p[0]), float(p[1]), z) for p in P]
    return occ.addCurveLoop([occ.addSpline(tags + [tags[0]])])


t0 = time.time()
shells = []
for l in range(nr):                                              # nr nested layer shells
    outer = occ.addThruSections([loop(R1[l], z1), loop(R2[l], z2)], makeSolid=True)
    inner = occ.addThruSections([loop(R1[l + 1], z1), loop(R2[l + 1], z2)], makeSolid=True)
    sh, _ = occ.cut(outer, inner, removeObject=True, removeTool=True)
    shells += sh
plug = occ.addThruSections([loop(R1[nr], z1), loop(R2[nr], z2)], makeSolid=True)   # IML plug for webs


def web_loop(cs, w, z):
    Pa = np.asarray(cs["nodes"][w["a"]], float); Pb = np.asarray(cs["nodes"][w["b"]], float)
    d = Pb - Pa; d = d / (np.hypot(*d) or 1e-30); perp = np.array([-d[1], d[0]])
    h = 0.5 * _thick(_lam_tuple(cs, w["lam"])); e = 0.05 * cs["chord"]
    Pa, Pb = Pa - e * d, Pb + e * d
    cor = [Pa - h * perp, Pb - h * perp, Pb + h * perp, Pa + h * perp]
    tg = [occ.addPoint(float(c[0]), float(c[1]), z) for c in cor]
    return occ.addCurveLoop([occ.addLine(tg[k], tg[(k + 1) % 4]) for k in range(4)])


parts = list(shells)
for wi in range(nw):
    box = occ.addThruSections([web_loop(cs1, cs1["webs"][wi], z1),
                               web_loop(cs2, cs2["webs"][wi], z2)], makeSolid=True)
    strip, _ = occ.intersect(box, occ.copy(plug), removeObject=True, removeTool=True)
    parts += strip
occ.remove(plug, recursive=True)
occ.fragment(parts, [])
occ.synchronize()
nvol = len(gmsh.model.getEntities(3))
print("fragment -> %d volumes (built in %.1fs)" % (nvol, time.time() - t0), flush=True)

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3 * wall)
gmsh.option.setNumber("Mesh.MeshSizeMax", inplane)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5); gmsh.option.setNumber("Mesh.Algorithm3D", 1)
t0 = time.time()
gmsh.model.mesh.generate(3)
etypes, etags, enodes = gmsh.model.mesh.getElements(3)
ntet = sum(len(etg) for et, etg in zip(etypes, etags) if et == 4)
# through-wall count: probe a radial line at a spar-cap hoop node, mid-span; count tets it crosses
ntag, ncoord, _ = gmsh.model.mesh.getNodes()
Pn = ncoord.reshape(-1, 3)
i = NC = R1.shape[1]; imid = NC // 4                              # a spar-cap-ish hoop index
Po = 0.5 * (R1[0, imid] + R2[0, imid]); Pi = 0.5 * (R1[nr, imid] + R2[nr, imid])
zc = 0.5 * (z1 + z2)
# count distinct tet layers along Po->Pi at z=zc by sampling
tets = None
for et, en in zip(etypes, enodes):
    if et == 4:
        tets = en.reshape(-1, 4)
idx = {int(t): k for k, t in enumerate(ntag)}
cent = np.array([Pn[[idx[int(x)] for x in row]].mean(0) for row in tets])
# tets whose centroid is within a thin tube around the radial probe line at mid-span
seg = Pi - Po; L2 = seg[:2] @ seg[:2]
proj = ((cent[:, :2] - Po) @ seg[:2]) / L2
perp = np.linalg.norm(cent[:, :2] - (Po + np.outer(np.clip(proj, 0, 1), seg[:2])), axis=1)
near = (perp < 0.3 * inplane) & (np.abs(cent[:, 2] - zc) < 0.3 * inplane) & (proj > -0.1) & (proj < 1.1)
print("generate(3) DONE in %.1fs : %d tets across %d volumes ; ~%d tets through the wall (probe)"
      % (time.time() - t0, ntet, nvol, int(near.sum())), flush=True)
gmsh.finalize()
