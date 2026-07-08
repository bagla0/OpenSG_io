"""opensg_io.mesh3d -- 3D solid-mesh generation for tapered thin-walled sections, with a
mandatory conformity gate on every export (see opensg_io.conformity).

Two ways to turn a 2D cross-section into a 3D solid:
  * HexGen (structured, PREFERRED for thin walls) -- loft_to_hex(): stack a 2D QUAD
    cross-section at nsp+1 span stations and connect each quad to the same quad at the next
    station -> structured 8-node hexes marching root->tip.  A conforming quad mesh lofts to a
    conforming hex mesh; the through-wall resolution is exact (n regular layers), so it
    captures transverse shear (unlike a coarse tet).
  * TetGen (unstructured, robust for arbitrary boundaries) -- opensg_io.tapered_tet: gmsh
    lofts the boundary SURFACE, TetGen fills the interior with Delaunay tets.

The 2D cross-section itself comes from the core mesher (gmsh) or PreVABS
(opensg_io.prevabs_webbed_ellipse).  Generators return (nodes, cells[, tags]);
export_solid_yaml() runs assert_conforming() BEFORE writing, so a non-conforming mesh is
never written.

  webbed_ellipse_hex(...) -> conforming const-thickness webbed-ellipse HEX = build the 2D quad
      cross-section (skin annulus + web plates; web top/bottom rows ARE inner-skin nodes;
      const web thickness kept by rebuilding the skin parametrization per station) then
      loft_to_hex it along the span.
"""
import math
import numpy as np
from .conformity import assert_conforming, conformity_report


def loft_to_hex(station_fn, faces2d, nsp, L=2.0):
    """Structured 8-node HEX mesh by lofting a 2D QUAD cross-section along the span.

    This is the structured alternative to a TetGen fill: instead of randomly filling the
    interior with tets, we stack the cross-section at nsp+1 span stations and connect each
    quad at station s to the SAME quad at station s+1 -> one 8-node hex per quad per span
    slice, marching linearly from one end to the other.  A conforming 2D quad mesh lofts to a
    conforming hex mesh automatically (shared quad edges -> shared hex faces, no hanging nodes).

      station_fn(z) -> (NP,3) node coords of the cross-section at span position z (SAME node
                       ordering/topology at every station; apply the taper here).
      faces2d        -> (M,4) quad connectivity into that node ordering.
      returns (nodes, hexes): nodes stacked station-major, hexes (nsp*M, 8) in VTK order.
    """
    stations = [station_fn(L * s / nsp) for s in range(nsp + 1)]
    NP = len(stations[0])
    nodes = np.vstack(stations)
    q = np.asarray(faces2d, int)
    hexes = np.empty((nsp * len(q), 8), int)
    for s in range(nsp):
        hexes[s * len(q):(s + 1) * len(q)] = np.hstack([s * NP + q, (s + 1) * NP + q])
    return nodes, hexes


def webbed_ellipse_hex(t, A0=1.0, A1=0.65, B0=0.60, B1=0.42, L=2.0, webs=(0.5, 0.0, -0.5),
                       nr=4, nsp=20, nw=4, nct=100):
    """Conforming const-thickness webbed-ellipse hex.  Returns (nodes, hexes, is_web)."""
    CSW = list(webs)

    def top_boundaries(a_z):
        u = t / (2 * a_z)
        b = [0.0]
        for c in CSW:
            b += [math.acos(min(1, max(-1, c + u))), math.acos(min(1, max(-1, c - u)))]
        b.append(math.pi)
        return b

    b0 = top_boundaries(A0)
    types = ["F", "W", "F", "W", "F", "W", "F"]
    hang = 2 * math.pi / nct
    counts = [nw if types[k] == "W" else max(2, int(round((b0[k + 1] - b0[k]) / hang))) for k in range(7)]
    len_top = sum(counts) + 1
    NC = 2 * len_top - 2

    def top_angles(a_z):
        b = top_boundaries(a_z)
        ang = []
        for k in range(7):
            ang += list(np.linspace(b[k], b[k + 1], counts[k] + 1)[:-1])
        ang.append(b[-1])
        return ang

    def circ_angles(a_z):
        top = top_angles(a_z)
        return np.array(top + [2 * math.pi - a for a in top[-2:0:-1]])

    MIRROR = np.arange(NC)
    for i in range(1, len_top - 1):
        MIRROR[i] = 2 * len_top - 2 - i
        MIRROR[2 * len_top - 2 - i] = i
    starts = np.cumsum([0] + counts)
    WT = [list(range(starts[k], starts[k] + counts[k] + 1)) for k in (1, 3, 5)]
    WB = [[int(MIRROR[i]) for i in top] for top in WT]

    NS = NC * (nr + 1)
    NY = []
    for wk in range(3):
        th = circ_angles(A0)[WT[wk][len(WT[wk]) // 2]]
        NY.append(max(4, int(round(2 * B0 * math.sin(th) / (t / nr)))))
    WBASE = [NS]
    for wk in range(3):
        WBASE.append(WBASE[-1] + len(WT[wk]) * (NY[wk] - 1))
    NP = WBASE[-1]

    def sid(i, l):
        return i * (nr + 1) + l

    def wid(wk, j, m):
        return WBASE[wk] + j * (NY[wk] - 1) + (m - 1)

    def station(z):
        a_z = A0 + (A1 - A0) * z / L; b_z = B0 + (B1 - B0) * z / L
        th = circ_angles(a_z); c, s = np.cos(th), np.sin(th)
        mid = np.column_stack([a_z * c, b_z * s])
        nrm = np.column_stack([b_z * c, a_z * s]); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        P = np.zeros((NP, 3)); P[:, 2] = z
        for i in range(NC):
            for l in range(nr + 1):
                P[sid(i, l), :2] = mid[i] + (l / nr - 0.5) * t * nrm[i]
        for wk in range(3):
            for j in range(len(WT[wk])):
                pt, pb = P[sid(WT[wk][j], 0), :2], P[sid(WB[wk][j], 0), :2]
                for m in range(1, NY[wk]):
                    P[wid(wk, j, m), :2] = pt + (m / NY[wk]) * (pb - pt)
        return P

    # Build the 2D QUAD cross-section ONCE (skin annulus + web plates), then loft it to hexes.
    faces2d, face_web = [], []
    for i in range(NC):                                    # skin annulus quads
        ii = (i + 1) % NC
        for l in range(nr):
            faces2d.append([sid(i, l), sid(ii, l), sid(ii, l + 1), sid(i, l + 1)]); face_web.append(0)
    for wk in range(3):                                    # web-plate quads
        def wn(j, m):
            return sid(WT[wk][j], 0) if m == 0 else sid(WB[wk][j], 0) if m == NY[wk] else wid(wk, j, m)
        for j in range(len(WT[wk]) - 1):
            for m in range(NY[wk]):
                faces2d.append([wn(j, m), wn(j + 1, m), wn(j + 1, m + 1), wn(j, m + 1)]); face_web.append(1)
    nodes, hexes = loft_to_hex(station, faces2d, nsp, L)   # <-- structured span loft, not TetGen
    return nodes, hexes, np.tile(face_web, nsp)


def export_solid_yaml(path, nodes, cells, celltype, orientations, materials, sets=None):
    """Write an OpenSG solid-mesh YAML -- ONLY after the conformity gate passes."""
    import yaml
    rep = assert_conforming(nodes, cells, celltype)     # raises NonConformingMesh on failure
    doc = {"nodes": [["%.9f %.9f %.9f" % tuple(p)] for p in nodes],
           "elements": [[" ".join(str(int(v) + 1) for v in c)] for c in cells],
           "sets": sets or {"element": [{"name": "all", "labels": list(range(1, len(cells) + 1))}]},
           "elementOrientations": [[float(v) for v in o] for o in orientations],
           "materials": materials}
    yaml.safe_dump(doc, open(path, "w"), default_flow_style=None, sort_keys=False)
    return rep
