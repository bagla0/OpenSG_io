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
gmsh.option.setNumber("General.RotationX", 300)          # isometric-ish view
gmsh.option.setNumber("General.RotationY", 0)
gmsh.option.setNumber("General.RotationZ", 240)
gmsh.option.setNumber("General.GraphicsWidth", 1600)
gmsh.option.setNumber("General.GraphicsHeight", 750)
gmsh.option.setNumber("General.Light0", 1)

# show the physical-group legend (the subdomain numbers)
gmsh.option.setNumber("PostProcessing.HorizontalScales", 0)
try:
    gmsh.fltk.initialize()
    gmsh.graphics.draw()
    gmsh.write(png)
    print("wrote", png)
finally:
    gmsh.finalize()
