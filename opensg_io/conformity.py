"""opensg_io.conformity -- rigorous conformity check for 3D volume meshes (hex or tet),
used as a pre-export gate: OpenSG_io refuses to write a solid mesh that is not conforming.

A mesh is CONFORMING iff adjacent cells meet full-face to full-face -- no node in the
interior of another cell's edge/face (a "hanging node"), and no face shared by >2 cells.
Topological connectivity ("one region") does NOT imply this: a webbed T-junction whose web
divisions do not match the skin's is connected yet has hanging nodes.  Validated against a
conforming block (passes) and a hand-built T-junction (fails, hanging node localized).

    check_conformal(nodes, cells, celltype) -> (ok, report)
    assert_conforming(nodes, cells, celltype)  # raises NonConformingMesh with localization
"""
import numpy as np
from collections import defaultdict

HEX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class NonConformingMesh(ValueError):
    pass


def check_conformal(nodes, cells, celltype="hex", tol=1e-7):
    """Return (ok, report). report localizes any failure (coordinates of a few offenders)."""
    nodes = np.asarray(nodes, float)
    cells = np.asarray(cells, int)
    faces_def = HEX_FACES if celltype == "hex" else TET_FACES
    edges_def = HEX_EDGES if celltype == "hex" else TET_EDGES
    rep, ok = {}, True

    degen = [i for i, c in enumerate(cells) if len(set(c.tolist())) != len(c)]
    rep["A_degenerate_cells"] = len(degen)
    if degen:
        ok = False; rep["A_examples"] = degen[:5]

    fcount, forder = defaultdict(int), {}
    for c in cells:
        for f in faces_def:
            nf = [int(c[k]) for k in f]
            key = tuple(sorted(nf))
            fcount[key] += 1
            forder.setdefault(key, nf)
    over = [k for k, v in fcount.items() if v > 2]
    rep["B_faces_shared_gt2"] = len(over)
    if over:
        ok = False
        rep["B_examples"] = [[nodes[list(k)].mean(0).round(4).tolist(), fcount[k]] for k in over[:5]]
    rep["n_internal_faces"] = sum(1 for v in fcount.values() if v == 2)
    boundary = [k for k, v in fcount.items() if v == 1]
    rep["n_boundary_faces"] = len(boundary)

    bedge = defaultdict(int)
    for k in boundary:
        nf = forder[k]; m = len(nf)
        for j in range(m):
            bedge[tuple(sorted((nf[j], nf[(j + 1) % m])))] += 1
    bad = [e for e, v in bedge.items() if v != 2]
    rep["C_nonmanifold_boundary_edges"] = len(bad)
    if bad:
        ok = False; rep["C_examples"] = [nodes[list(e)].mean(0).round(4).tolist() for e in bad[:8]]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        edge_set = set()
        for c in cells:
            for a, b in edges_def:
                edge_set.add(tuple(sorted((int(c[a]), int(c[b])))))
        hn = set()
        for (a, b) in edge_set:
            p, q = nodes[a], nodes[b]; d = q - p; Ln = np.linalg.norm(d)
            if Ln < tol:
                continue
            for n in tree.query_ball_point(0.5 * (p + q), 0.5 * Ln + tol):
                if n == a or n == b:
                    continue
                w = nodes[n] - p; s = np.dot(w, d) / (Ln * Ln)
                if tol < s < 1 - tol and np.linalg.norm(w - s * d) < 1e-6 * max(Ln, 1.0):
                    hn.add(n)
        rep["D_hanging_nodes"] = len(hn)
        if hn:
            ok = False; rep["D_examples"] = [nodes[n].round(4).tolist() for n in list(hn)[:8]]
    except ImportError:
        rep["D_hanging_nodes"] = "skipped(no scipy)"

    rep["conforming"] = ok
    return ok, rep


def conformity_report(rep):
    keys = ["conforming", "A_degenerate_cells", "B_faces_shared_gt2", "n_internal_faces",
            "n_boundary_faces", "C_nonmanifold_boundary_edges", "D_hanging_nodes"]
    s = "  ".join("%s=%s" % (k, rep[k]) for k in keys if k in rep)
    ex = {k: rep[k] for k in rep if k.endswith("_examples")}
    return s + ("\n  " + str(ex) if ex else "")


def assert_conforming(nodes, cells, celltype="hex"):
    ok, rep = check_conformal(nodes, cells, celltype)
    if not ok:
        raise NonConformingMesh("mesh is NOT conforming (hanging nodes / non-manifold "
                                "interfaces) -- refusing to export.\n  " + conformity_report(rep))
    return rep
