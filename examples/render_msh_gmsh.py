"""render_msh_gmsh.py -- visualize a .msh with THIRD-PARTY gmsh, colored by SUBDOMAIN.

Each physical group (= the element set / material / layup subdomain written into the
.msh) gets its own gmsh color, so the rendering shows the subdomain numbering by color.
Runs gmsh's own OpenGL renderer off-screen (needs a display; use `xvfb-run -a python
render_msh_gmsh.py <in.msh> <out.png>` on a headless server).
"""
import os
import sys

import gmsh

msh = os.path.abspath(sys.argv[1])
png = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else msh.replace(".msh", "_gmsh.png")

gmsh.initialize()
gmsh.open(msh)

# pick the view from the mesh extent: a (near-)planar 2-D cross-section is best shown
# face-on (front/top/side view); a genuine 3-D loft keeps an isometric angle.
try:
    _, coords, _ = gmsh.model.mesh.getNodes()
    P = coords.reshape(-1, 3)
    d = P.max(0) - P.min(0)
    dmax = max(d.max(), 1e-30)
    flat = [i for i in range(3) if d[i] / dmax < 1e-3]
    flat_axis = flat[0] if len(flat) == 1 else None
except Exception:
    flat_axis = None
if flat_axis == 2:                                       # flat in Z -> look down Z (front view)
    rx, ry, rz = 0.0, 0.0, 0.0
elif flat_axis == 1:                                     # flat in Y -> look down Y (top view)
    rx, ry, rz = 90.0, 0.0, 0.0
elif flat_axis == 0:                                     # flat in X -> look down X (side view)
    rx, ry, rz = 0.0, 90.0, 0.0
else:                                                    # genuine 3-D -> isometric
    rx, ry, rz = 300.0, 0.0, 240.0

# COLOR BY PHYSICAL GROUP (subdomain).  Mesh.ColorCarousel: 0 elem-type, 1 elementary,
# 2 physical group, 3 partition.
gmsh.option.setNumber("Mesh.ColorCarousel", 2)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
gmsh.option.setNumber("Mesh.VolumeFaces", 1)
gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.VolumeEdges", 1)
gmsh.option.setNumber("Mesh.LineWidth", 1.0)
gmsh.option.setNumber("General.Axes", 0)
gmsh.option.setNumber("General.SmallAxes", 0)
gmsh.option.setNumber("General.Trackball", 0)
gmsh.option.setNumber("General.RotationX", rx)
gmsh.option.setNumber("General.RotationY", ry)
gmsh.option.setNumber("General.RotationZ", rz)
gmsh.option.setNumber("General.GraphicsWidth", 1600)
gmsh.option.setNumber("General.GraphicsHeight", 750)
gmsh.option.setNumber("General.Light0", 1)

# show the physical-group legend (the subdomain numbers)
gmsh.option.setNumber("PostProcessing.HorizontalScales", 0)
try:
    gmsh.fltk.initialize()
    gmsh.graphics.draw()
    gmsh.fltk.update()
    gmsh.write(png)                                       # writes at the model's fitted scale
    print("wrote", png, "(flat_axis=%s)" % flat_axis)
finally:
    gmsh.finalize()
