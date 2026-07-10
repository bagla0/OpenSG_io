"""Run the shell Timo on the two foam-G bisect variants (g12-only, g23-only) and print
the segment + L-ring diagonals next to the original and all-stiff results."""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TW = r"Y:\OpenSG-TW-claude"
sys.path.insert(0, TW)
sys.path.insert(0, os.path.join(TW, "mitc_rm_segment"))
import run_indep  # noqa: E402

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
work = os.path.join(OUT, "_bisect")
os.makedirs(work, exist_ok=True)
res = {}
for mode in ("g12", "g23"):
    shutil.copy(os.path.join(OUT, "iea22_seg_shell_%s.yaml" % mode),
                os.path.join(work, "shell_%s.yaml" % mode))
    r = run_indep.shell_solve_lagrange_sparse(mode, work, work, shear="full", return_full=True)
    res[mode] = r
    np.savez(os.path.join(OUT, "timo_shell_%s.npz" % mode), S6=r["S6"], C6L=r["C6L"], C6R=r["C6R"])

base = np.load(os.path.join(OUT, "timo_shell.npz"))
stiff = np.load(os.path.join(OUT, "timo_shell_stiff.npz"))
solid = np.load(os.path.join(OUT, "timo_solid.npz"))

print("\nSHELL segment diagonal under foam-G variants   (solid reference in last col)")
print("       %-12s %-12s %-12s %-12s | %-12s" % ("orig(soft)", "G12-only", "G23-only", "all-stiff", "solid(soft)"))
for i in range(6):
    print(" %-4s %12.4e %12.4e %12.4e %12.4e | %12.4e"
          % (LBL[i], base["S6"][i, i], res["g12"]["S6"][i, i], res["g23"]["S6"][i, i],
             stiff["S6"][i, i], solid["S6"][i, i]))
print("\nSHELL L-ring diagonal")
for i in range(6):
    print(" %-4s %12.4e %12.4e %12.4e %12.4e | %12.4e"
          % (LBL[i], base["C6L"][i, i], res["g12"]["C6L"][i, i], res["g23"]["C6L"][i, i],
             stiff["C6L"][i, i], solid["C6L"][i, i]))
