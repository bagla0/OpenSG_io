"""run_timo_solid_fenics.py -- OpenSG-FEniCS solid tapered-segment Timoshenko 6x6 of
the generated IEA-22 hex taper (output/iea22_seg_solid.yaml).  Run ON THE SERVER in
the dolfinx env:

    cd ~/OpenSG_io/examples/IEA && python run_timo_solid_fenics.py

Saves S6 to output/timo_solid.npz.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))

from opensg.mesh.segment import SolidSegmentMesh          # noqa: E402
from opensg.core.solid import compute_stiffness           # noqa: E402

t0 = time.time()
os.chdir(OUT)                                             # SG_mesh.msh scratch goes here
sm = SolidSegmentMesh(os.path.join(OUT, "iea22_seg_solid.yaml"))
material_parameters, density = sm.material_database
Sseg, V0, V1s, DL, DR = compute_stiffness(material_parameters, sm.meshdata,
                                          sm.left_submesh, sm.right_submesh, Taper=True)
S = 0.5 * (np.asarray(Sseg, float) + np.asarray(Sseg, float).T)
DL = np.asarray(DL, float); DR = np.asarray(DR, float)
dt = time.time() - t0

np.savez(os.path.join(OUT, "timo_solid.npz"), S6=S,
         C6L=0.5 * (DL + DL.T), C6R=0.5 * (DR + DR.T), wall=dt)
print("=== OpenSG-FEniCS: IEA-22 tapered SOLID segment ===")
print("wall %.1f s ; origin (axial) %.4f" % (dt, sm.origin))
for name, M in (("SEGMENT (taper)", S), ("BOUNDARY L", DL), ("BOUNDARY R", DR)):
    print("\n%s Timoshenko 6x6 [EA GA2 GA3 GJ EI2 EI3]:" % name)
    M = 0.5 * (M + M.T)
    for i in range(6):
        print("  " + "".join("%13.4e" % M[i, j] for j in range(6)))
