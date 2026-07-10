"""run_timo_shell.py -- OpenSG-TW (JAX) MITC-RM tapered-segment Timoshenko 6x6 of the
generated IEA-22 shell taper (output/iea22_seg_shell.yaml).

Uses the settled production pipeline (all-6-DOF independent-omega3 element, FULL
transverse-shear integration on the segment, gamma_23-tied rings): run_indep.
shell_solve_lagrange_sparse.  Saves S6 + ring 6x6s to output/timo_shell.npz.
"""
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TW = r"Y:\OpenSG-TW-claude"
sys.path.insert(0, TW)
sys.path.insert(0, os.path.join(TW, "mitc_rm_segment"))

import run_indep  # noqa: E402

work = os.path.join(OUT, "_shellrun")
os.makedirs(work, exist_ok=True)
shutil.copy(os.path.join(OUT, "iea22_seg_shell.yaml"), os.path.join(work, "shell_iea22.yaml"))

t0 = time.time()
res = run_indep.shell_solve_lagrange_sparse("iea22", work, work, shear="full", return_full=True)
S6, C6L, C6R = res["S6"], res["C6L"], res["C6R"]
dt = time.time() - t0

np.savez(os.path.join(OUT, "timo_shell.npz"), S6=S6, C6L=C6L, C6R=C6R, wall=dt)
np.set_printoptions(precision=5, suppress=False, linewidth=160)
print("=== OpenSG-TW JAX MITC-RM: IEA-22 tapered SHELL segment ===")
print("wall %.1f s  (extract %.1f + rings %.1f + segment %.1f)"
      % (dt, res["t_extract"], res["t_rings"], res["t_seg"]))
for name, M in (("SEGMENT (taper)", S6), ("BOUNDARY L ring", C6L), ("BOUNDARY R ring", C6R)):
    print("\n%s Timoshenko 6x6 [EA GA2 GA3 GJ EI2 EI3]:" % name)
    for i in range(6):
        print("  " + "".join("%13.4e" % M[i, j] for j in range(6)))
