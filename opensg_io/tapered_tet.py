"""opensg_io.tapered_tet -- tapered thin-walled webbed solids as CONFORMING TetGen tets,
built from clean section boundaries and gated by opensg_io.conformity.

Pipeline (matches the PreVABS -> TetGen workflow):
  1. PreVABS meshes/validates the 2D cross-section (skin + webs, layups) -- see the
     make_ellipse_prevabs helper / prevabs_xml.to_solid; that 2D mesh is the reference
     "tw solid boundary".
  2. The tapered 3D boundary SURFACE (same geometry) is lofted with a CONSTANT-thickness
     taper via gmsh OCC (webbed_ellipse_surface_stl), meshed fine near the webs.
  3. TetGen fills the watertight surface with conforming Delaunay tets (tetgen_fill), which
     assert_conforming() gates before returning -- a non-conforming result is refused.

Why the surface (not the filled 2D mesh) is lofted: morphing a filled webbed cross-section
into the tapered shape self-intersects at the web-skin T-junctions (skin thickness is radial,
web thickness is horizontal); lofting only the outline and letting TetGen fill avoids that.
"""
import os
import math
import numpy as np


def webbed_ellipse_surface_stl(out_stl, t=0.2, A0=1.0, A1=0.65, B0=0.60, B1=0.42, L=2.0,
                               webs=(0.5, 0.0, -0.5), npt=160, fine=0.018, coarse=0.068, band=0.14):
    """Watertight boundary surface of the tapered webbed-ellipse solid (skin annulus + webs at
    x=c*a(z)), meshed fine near the web planes.  Written as `out_stl` for TetGen.  Needs gmsh."""
    import gmsh
    rate = (A1 - A0) / L

    def off_ellipse(a, b, sgn):
        th = np.linspace(0, 2 * math.pi, npt, endpoint=False)
        px, py = a * np.cos(th), b * np.sin(th)
        nx, ny = b * np.cos(th), a * np.sin(th); nn = np.hypot(nx, ny)
        return np.column_stack([px + sgn * 0.5 * t * nx / nn, py + sgn * 0.5 * t * ny / nn])

    gmsh.initialize()
    gmsh.model.add("ellw_surf")
    occ = gmsh.model.occ

    def wire(pts2, z):
        tags = [occ.addPoint(p[0], p[1], z) for p in pts2]
        return occ.addCurveLoop([occ.addSpline(tags + [tags[0]])])

    o0 = wire(off_ellipse(A0, B0, +1), 0.0); o1 = wire(off_ellipse(A1, B1, +1), L)
    i0 = wire(off_ellipse(A0, B0, -1), 0.0); i1 = wire(off_ellipse(A1, B1, -1), L)
    outer = occ.addThruSections([o0, o1], makeSolid=True)
    inner = occ.addThruSections([i0, i1], makeSolid=True)
    outer_copy = occ.copy(outer)
    skin, _ = occ.cut(outer, inner)
    strips = []
    for c in webs:
        def rect(a, z):
            x0, x1 = c * a - t / 2, c * a + t / 2
            p = [occ.addPoint(x0, -B0 - 0.2, z), occ.addPoint(x1, -B0 - 0.2, z),
                 occ.addPoint(x1, B0 + 0.2, z), occ.addPoint(x0, B0 + 0.2, z)]
            ls = [occ.addLine(p[k], p[(k + 1) % 4]) for k in range(4)]
            return occ.addCurveLoop(ls)
        box = occ.addThruSections([rect(A0, 0.0), rect(A1, L)], makeSolid=True)
        strip, _ = occ.intersect(box, occ.copy(outer_copy))
        strips += strip
    occ.fuse(skin, strips)
    occ.remove(outer_copy, recursive=True)
    occ.synchronize()
    az = "(%g+(%g)*z)" % (A0, rate)
    d = "Min(Min(Fabs(x-0.5*%s),Fabs(x+0.5*%s)),Fabs(x))" % (az, az)
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", "%g + %g*Min(1.0,(%s)/%g)" % (fine, coarse - fine, d, band))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    for opt in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + opt, 0)
    gmsh.model.mesh.generate(2)
    gmsh.write(out_stl)
    gmsh.finalize()
    return out_stl


def tetgen_fill(stl, mindihedral=18, minratio=1.4):
    """Fill a watertight surface STL with conforming TetGen tets.  Returns (nodes, tets).
    Raises opensg_io.conformity.NonConformingMesh if the surface is not watertight or the
    result is not perfectly conforming."""
    import pyvista as pv
    import tetgen
    from .conformity import assert_conforming, NonConformingMesh
    surf = pv.read(stl).clean()                        # weld coincident seam vertices (watertight)
    n_open = int(surf.n_open_edges)
    if n_open > 0:
        raise NonConformingMesh("input surface is not watertight: %d open edges (a web box did "
                                "not fully pierce the inner plug, or the fuse left a hairline gap)" % n_open)
    tg = tetgen.TetGen(surf)
    # steinerleft=-1 (unlimited Steiner points): with quality on, a finite budget makes TetGen HANG.
    tg.tetrahedralize(order=1, mindihedral=mindihedral, minratio=minratio, quality=True, steinerleft=-1)
    grid = tg.grid
    tets = grid.cells_dict[pv.CellType.TETRA]
    nodes = np.asarray(grid.points)
    used = np.unique(tets); nodes = nodes[used]; tets = np.searchsorted(used, tets)
    assert_conforming(nodes, tets, "tet")             # gate: refuse non-conforming
    return nodes, tets


# ===================================================================================================
#  General windIO webbed-airfoil TAPER as conforming TETS  (robust: never inverts on any taper)
# ===================================================================================================
def _seg_dist(p, a, b):
    """Perpendicular distance from 2-D point p to segment [a,b] and the signed side."""
    ab = b - a
    L2 = float(ab @ ab) or 1e-30
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    proj = a + t * ab
    perp = np.array([-ab[1], ab[0]]) / (np.hypot(*ab) or 1e-30)
    return float(np.hypot(*(p - proj))), float((p - a) @ perp)


def windio_taper_tets(cs1, cs2, z1, z2, nr=4, nw=2, mesh_size=0.02, tet_size=None,
                      stl_path=None, mindihedral=18, minratio=1.4):
    """Tapered webbed-airfoil SOLID as CONFORMING TETS -- the robust alternative to the hex
    loft: it fills the volume with an unstructured mesher, so it NEVER inverts on a steep /
    twisting taper (the hex loft's failure mode).  The user accepts refinement for robustness.

    Geometry (gmsh OCC): skin annulus = loft(OML) - loft(IML) using the SAME canonical rings
    the hex loft builds (build_station); web plates = box per web (spanning its two attachments,
    thickened by the web-laminate thickness) intersected with the inner plug; all fused.  Only the
    SURFACE is meshed (generate 2) and TetGen fills the volume -- meshing the fused webbed solid
    directly (generate 3) self-intersects at the web-skin T-junctions and hangs.  Each tet is then
    assigned a material + NuMAD orientation by locating its centroid (skin: nearest hoop node ->
    segment laminate, through-thickness fraction -> ply; web: nearest web -> web laminate).

    Returns (nodes[N,3], tets[M,4] 0-based, oris[M,9], hmats[M] material names).
    """
    import gmsh
    import time as _time
    from .hex_loft import (section_skeleton, build_station, _lam_tuple, _thick, _set_at, _ply_at)
    from .orientation import element_frame
    from .conformity import assert_conforming

    _vb = bool(os.environ.get("OPENSG_TET_VERBOSE"))
    _t0 = [_time.time()]

    def vb(msg):
        if _vb:
            print("  [tet %6.1fs] %s" % (_time.time() - _t0[0], msg), flush=True)

    skel = section_skeleton([cs1, cs2], mesh_size=mesh_size, nw=nw)
    st1 = build_station(cs1, skel, 0, nr=nr)
    st2 = build_station(cs2, skel, 1, nr=nr)
    OML0, IML0 = np.asarray(st1["rings"][0]), np.asarray(st1["rings"][nr])
    OML1, IML1 = np.asarray(st2["rings"][0]), np.asarray(st2["rings"][nr])
    chord = 0.5 * (cs1["chord"] + cs2["chord"])
    dz = z2 - z1
    # surface element size must be WALL-based, not span-based: the skin annulus is thin (cm) on a
    # metre-scale chord/span, so a span-scaled size can't place nodes across the wall and the 2-D
    # mesher loops.  Size from the local wall thickness (mean OML->IML distance).
    wall = float(np.mean(np.linalg.norm(0.5 * (OML0 + OML1) - 0.5 * (IML0 + IML1), axis=1)))
    if tet_size is None:
        tet_size = max(0.5 * wall, 0.02 * chord)             # ~1-2 elements across the wall
    smin = max(0.35 * wall, 0.004 * chord)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if _vb else 0)
    gmsh.model.add("taper_tet")
    occ = gmsh.model.occ

    def loop(P, z):
        tags = [occ.addPoint(float(p[0]), float(p[1]), z) for p in P]
        return occ.addCurveLoop([occ.addSpline(tags + [tags[0]])])

    vb("rings built (NC=%d, nwebs=%d); lofting OML/IML" % (len(OML0), min(len(cs1["webs"]), len(cs2["webs"]))))
    outer = occ.addThruSections([loop(OML0, z1), loop(OML1, z2)], makeSolid=True)
    inner = occ.addThruSections([loop(IML0, z1), loop(IML1, z2)], makeSolid=True)
    vb("OML/IML lofted; cutting skin annulus")
    skin, _ = occ.cut(outer, inner, removeObject=True, removeTool=False)
    vb("skin annulus done")

    def web_loop(cs, w, z):
        Pa = np.asarray(cs["nodes"][w["a"]], float); Pb = np.asarray(cs["nodes"][w["b"]], float)
        d = Pb - Pa; d = d / (np.hypot(*d) or 1e-30); perp = np.array([-d[1], d[0]])
        h = 0.5 * _thick(_lam_tuple(cs, w["lam"])); e = 0.05 * cs["chord"]
        Pa, Pb = Pa - e * d, Pb + e * d
        cor = [Pa - h * perp, Pb - h * perp, Pb + h * perp, Pa + h * perp]
        tg = [occ.addPoint(float(c[0]), float(c[1]), z) for c in cor]
        return occ.addCurveLoop([occ.addLine(tg[k], tg[(k + 1) % 4]) for k in range(4)])

    # Keep the SKIN annulus and each WEB as SEPARATE, conformal OCC volumes (occ.fragment welds
    # their shared faces).  gmsh meshes the assembly directly (generate 3 -- the fragmented
    # geometry is properly imprinted, so it does NOT self-intersect at the web-skin T-junctions
    # the way a single FUSED body did).  Every tet is then tagged by its PHYSICAL VOLUME, so
    # skin vs web is exact by CONTAINMENT -- no centroid vote that intermixes at the junction.
    skin_dt = list(skin)
    nwebs = min(len(cs1["webs"]), len(cs2["webs"]))
    web_dt = []
    for wi in range(nwebs):
        box = occ.addThruSections([web_loop(cs1, cs1["webs"][wi], z1),
                                   web_loop(cs2, cs2["webs"][wi], z2)], makeSolid=True)
        strip, _ = occ.intersect(box, occ.copy(inner), removeObject=True, removeTool=True)
        web_dt.append(strip)
        vb("web %d trimmed" % wi)
    occ.remove(inner, recursive=True)
    parts = skin_dt + [dt for st_ in web_dt for dt in st_]
    frag, fmap = occ.fragment(parts, [])                     # weld shared faces -> conformal
    occ.synchronize()
    # map every OUTPUT volume to its region via the fragment child-map (fmap[j] <- parts[j])
    region_of_vol = {}
    for (_d, t) in fmap[0]:
        region_of_vol[t] = ("skin", -1)
    off = 1
    for wi in range(nwebs):
        for _dt in web_dt[wi]:
            for (_d, t) in fmap[off]:
                region_of_vol[t] = ("web", wi)
            off += 1
    vols = gmsh.model.getEntities(3)
    for (_d, t) in vols:
        gmsh.model.addPhysicalGroup(3, [t], t)
    vb("fragment -> %d volumes (%d skin, %d web)" % (
        len(vols), sum(r[0] == "skin" for r in region_of_vol.values()),
        sum(r[0] == "web" for r in region_of_vol.values())))

    gmsh.option.setNumber("Mesh.MeshSizeMin", smin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", tet_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)             # Delaunay 3-D
    vb("mesh size min=%.4f max=%.4f (wall=%.4f) -> generate(3)" % (smin, tet_size, wall))
    gmsh.model.mesh.generate(3)
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    P = ncoord.reshape(-1, 3)
    id_of = {int(t): i for i, t in enumerate(ntag)}
    tets = []
    region_of_tet = []
    for (_d, t) in vols:                                     # read tets grouped BY VOLUME
        ets, _etg, enod = gmsh.model.mesh.getElements(3, t)
        reg = region_of_vol.get(t, ("skin", -1))
        for et, en in zip(ets, enod):
            if et == 4:
                conn = en.reshape(-1, 4)
                for row in conn:
                    tets.append([id_of[int(x)] for x in row])
                    region_of_tet.append(reg)
    gmsh.finalize()
    tets = np.asarray(tets, int)
    used = np.unique(tets); nodes = P[used]; tets = np.searchsorted(used, tets)
    assert_conforming(nodes, tets, "tet")                    # gate: refuse non-conforming
    vb("generate(3): %d nodes / %d tets; assigning materials by volume" % (len(nodes), len(tets)))

    # ---- per-tet material + NuMAD orientation.  REGION (skin vs which web) is from the tet's
    #      owning physical VOLUME (clean, no intermixing); the ply within the region is from the
    #      tet's through-thickness/across-web fraction; skin segment from the EXACT arc. ----------
    oris = np.zeros((len(tets), 9))
    hmats = []
    for k, tet in enumerate(tets):
        c = nodes[tet].mean(0); cxy = c[:2]
        tau = 0.0 if abs(dz) < 1e-12 else float(np.clip((c[2] - z1) / dz, 0.0, 1.0))
        cs = cs1 if tau < 0.5 else cs2
        st = st1 if tau < 0.5 else st2
        OMLt = (1 - tau) * OML0 + tau * OML1
        IMLt = (1 - tau) * IML0 + tau * IML1
        i = int(np.argmin(np.sum((OMLt - cxy) ** 2, axis=1)))
        Po, Pi = OMLt[i], IMLt[i]
        inward = Pi - Po; tskin = float(np.hypot(*inward)) or 1e-9; inward = inward / tskin
        depth = float((cxy - Po) @ inward)
        reg = region_of_tet[k]
        if reg[0] == "web":
            w = cs["webs"][reg[1]]
            lam = _lam_tuple(cs, w["lam"])
            Pa = np.asarray(cs["nodes"][w["a"]], float); Pb = np.asarray(cs["nodes"][w["b"]], float)
            twidth = _thick(lam)
            _dp, sside = _seg_dist(cxy, Pa, Pb)
            m, ang = _ply_at(lam, float(np.clip(sside / (twidth or 1e-9) + 0.5, 0.0, 1.0)))
            t = Pb - Pa; t = t / (np.hypot(*t) or 1e-9)
            n_surf = np.array([-t[1], t[0], 0.0])            # across-web in-plane normal
        else:
            lam = _lam_tuple(cs, _set_at(cs, float(st["hoop_s"][i])))
            m, ang = _ply_at(lam, float(np.clip(depth / tskin, 0.0, 1.0)))
            n_surf = np.array([-inward[0], -inward[1], 0.0])  # outward normal; e3 = inward
        oris[k] = element_frame(np.array([0.0, 0.0, dz if dz else 1.0]), n_surf, ang)
        hmats.append(m)
    return nodes, tets, oris, hmats
