"""render_msh_gmsh.py -- visualize a .msh with THIRD-PARTY gmsh, colored by SUBDOMAIN.

Each physical group (= the element set / material / layup subdomain written into the
.msh) gets its own gmsh color, so the rendering shows the subdomain numbering by color.

A (near-)planar cross-section is shown in the **y-z plane**: the beam axis becomes X
(out of the screen), the chord is horizontal (Y), the thickness is vertical (Z), so it
reads like a beam cross-section.  A genuine 3-D loft keeps an isometric angle.  Either
way the **axis triad** (small XYZ gnomon) is drawn at the bottom-left.

To get a clean zoom-to-fit under the y-z camera, the planar mesh is permuted + recentred
in a sibling `*_disp.msh` BEFORE gmsh opens it (so gmsh fits the final geometry; a pure
camera rotation about a centred model then preserves the fit).

Runs gmsh's own OpenGL renderer off-screen (needs a display; use `xvfb-run -a python
render_msh_gmsh.py <in.msh> <out.png>` on a headless server, or run it on Windows).
"""
import os
import sys

import gmsh


def _read_msh_nodes(path):
    """Return (header_lines, nodes[list of (id,x,y,z)], rest_lines) for a gmsh 2.2 ASCII .msh."""
    lines = open(path).read().splitlines()
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    nodes = []
    for k in range(n):
        p = lines[i + 2 + k].split()
        nodes.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return lines[:i + 2], nodes, lines[i + 2 + n:]


def _planar_display_msh(src, dst):
    """If `src` is a (near-)planar section, write `dst` with coords permuted to
    (X=beam, Y=chord, Z=thickness) and recentred at the origin.  Return True if planar."""
    head, nodes, rest = _read_msh_nodes(src)
    import numpy as np
    P = np.array([[x, y, z] for _, x, y, z in nodes])
    d = P.max(0) - P.min(0)
    dmax = max(d.max(), 1e-30)
    flat = [i for i in range(3) if d[i] / dmax < 1e-3]
    if len(flat) != 1:
        return False
    beam = flat[0]
    inplane = [i for i in range(3) if i != beam]
    chord_ax, thick_ax = (inplane if d[inplane[0]] >= d[inplane[1]] else inplane[::-1])
    Q = P[:, [beam, chord_ax, thick_ax]]                    # -> (beam, chord, thickness)
    Q = Q - 0.5 * (Q.max(0) + Q.min(0))                     # recentre at origin
    with open(dst, "w") as f:
        f.write("\n".join(head) + "\n")
        for (nid, _, _, _), q in zip(nodes, Q):
            f.write("%s %.9g %.9g %.9g\n" % (nid, q[0], q[1], q[2]))
        f.write("\n".join(rest) + "\n")
    return True


msh = os.path.abspath(sys.argv[1])
png = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else msh.replace(".msh", "_gmsh.png")

disp = msh.replace(".msh", "_disp.msh")
planar = False
try:
    planar = _planar_display_msh(msh, disp)
except Exception:
    planar = False

gmsh.initialize()
gmsh.open(disp if planar else msh)

if planar:
    rx, ry, rz = -90.0, 0.0, -90.0                          # look down -X: Y right (chord), Z up (thickness)
    gw, gh, zoom = 1700, 560, 2.2                           # zoom in so thin walls show their subdomain color
else:
    rx, ry, rz = 300.0, 0.0, 240.0                          # genuine 3-D -> isometric
    gw, gh, zoom = 1600, 800, 1.0

# --- COLOR BY PHYSICAL GROUP (subdomain).  Mesh.ColorCarousel: 0 elem-type, 1 elementary,
#     2 physical group, 3 partition. -----------------------------------------------------
gmsh.option.setNumber("Mesh.ColorCarousel", 2)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
gmsh.option.setNumber("Mesh.VolumeFaces", 1)
gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.VolumeEdges", 1)
gmsh.option.setNumber("Mesh.LineWidth", 1.0)
gmsh.option.setNumber("General.RotationX", rx)
gmsh.option.setNumber("General.RotationY", ry)
gmsh.option.setNumber("General.RotationZ", rz)
gmsh.option.setNumber("General.GraphicsWidth", gw)
gmsh.option.setNumber("General.GraphicsHeight", gh)
gmsh.option.setNumber("General.ScaleX", zoom)               # zoom in on the fitted model
gmsh.option.setNumber("General.ScaleY", zoom)
gmsh.option.setNumber("General.ScaleZ", zoom)
gmsh.option.setNumber("General.Light0", 1)
gmsh.option.setNumber("General.Trackball", 0)

# --- axis triad (small XYZ gnomon) at the bottom-left; hide the big bounding-box axes ------
gmsh.option.setNumber("General.Axes", 0)
gmsh.option.setNumber("General.SmallAxes", 1)
gmsh.option.setNumber("General.SmallAxesPositionX", 90)
gmsh.option.setNumber("General.SmallAxesPositionY", gh - 70)
gmsh.option.setNumber("General.SmallAxesSize", 44)

gmsh.option.setNumber("PostProcessing.HorizontalScales", 0)
try:
    gmsh.fltk.initialize()
    gmsh.graphics.draw()
    gmsh.fltk.update()
    gmsh.write(png)
    print("wrote", png, "(planar=%s -> y-z view)" % planar)
finally:
    gmsh.finalize()
    if planar and os.path.exists(disp):
        try:
            os.remove(disp)
        except OSError:
            pass
