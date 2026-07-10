import os, shutil, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "output")
TW = r"Y:\OpenSG-TW-claude"; sys.path.insert(0, TW); sys.path.insert(0, os.path.join(TW, "mitc_rm_segment"))
import run_indep
work = os.path.join(OUT, "_stiffrun"); os.makedirs(work, exist_ok=True)
shutil.copy(os.path.join(OUT, "iea22_seg_shell_stiff.yaml"), os.path.join(work, "shell_stiff.yaml"))
res = run_indep.shell_solve_lagrange_sparse("stiff", work, work, shear="full", return_full=True)
np.savez(os.path.join(OUT, "timo_shell_stiff.npz"), S6=res["S6"], C6L=res["C6L"], C6R=res["C6R"])
LBL = ["EA","GA2","GA3","GJ","EI2","EI3"]; S = res["S6"]
print("SHELL stiff-foam diagonal:", "  ".join("%s=%.3e"%(LBL[i],S[i,i]) for i in range(6)))
