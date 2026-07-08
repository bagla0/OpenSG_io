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
    Raises opensg_io.conformity.NonConformingMesh if the result is not perfectly conforming."""
    import pyvista as pv
    import tetgen
    from .conformity import assert_conforming
    surf = pv.read(stl).clean()
    tg = tetgen.TetGen(surf)
    tg.tetrahedralize(order=1, mindihedral=mindihedral, minratio=minratio, quality=True)
    grid = tg.grid
    tets = grid.cells_dict[pv.CellType.TETRA]
    nodes = np.asarray(grid.points)
    used = np.unique(tets); nodes = nodes[used]; tets = np.searchsorted(used, tets)
    assert_conforming(nodes, tets, "tet")             # gate: refuse non-conforming
    return nodes, tets
