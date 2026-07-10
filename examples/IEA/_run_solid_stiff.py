import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "output")
sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
os.chdir(OUT)
sm = SolidSegmentMesh(os.path.join(OUT, "iea22_seg_solid_stiff.yaml"))
mp, den = sm.material_database
S, V0, V1s, DL, DR = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)
S = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
np.savez(os.path.join(OUT, "timo_solid_stiff.npz"), S6=S,
         C6L=0.5*(np.asarray(DL)+np.asarray(DL).T), C6R=0.5*(np.asarray(DR)+np.asarray(DR).T))
LBL = ["EA","GA2","GA3","GJ","EI2","EI3"]
print("SOLID stiff-foam diagonal:", "  ".join("%s=%.3e"%(LBL[i],S[i,i]) for i in range(6)))
