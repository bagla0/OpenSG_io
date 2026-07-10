"""compare_timo.py -- shell-taper (OpenSG-TW JAX MITC-RM) vs solid-taper (OpenSG-FEniCS)
Timoshenko 6x6 of the IEA-22 r=0.2->0.3 segment.  Prints EVERY non-zero term of both
6x6s and the shell-vs-solid % difference.  Order [EA, GA2, GA3, GJ, EI2, EI3]."""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
sh = np.load(os.path.join(OUT, "timo_shell.npz"))
so = np.load(os.path.join(OUT, "timo_solid.npz"))
Ssh, Sso = sh["S6"], so["S6"]

# scale so the largest term ~ O(1) for the "zero" cutoff
scale = max(abs(Sso).max(), abs(Ssh).max())
cut = 1e-4 * scale                                        # below this = structurally zero


def block(name, M):
    print("\n%s Timoshenko 6x6 (order %s):" % (name, LBL))
    for i in range(6):
        print("  " + "".join("%13.4e" % M[i, j] for j in range(6)))


block("SHELL taper  (OpenSG-TW JAX RM)", Ssh)
block("SOLID taper  (OpenSG-FEniCS)", Sso)

print("\n=== ALL non-zero terms: shell vs solid ===")
print("  term        solid          shell         %diff (shell vs solid)")
rows = []
for i in range(6):
    for j in range(i, 6):
        s, h = Sso[i, j], Ssh[i, j]
        if max(abs(s), abs(h)) < cut:
            continue
        pd = 100.0 * (h - s) / s if abs(s) > cut else float("nan")
        rows.append((abs(pd) if pd == pd else -1, i, j, s, h, pd))
        tag = "%s" % LBL[i] if i == j else "%s-%s" % (LBL[i], LBL[j])
        print("  %-9s %14.4e %14.4e   %+8.2f%%" % (tag, s, h, pd))

print("\n=== diagonal summary ===")
for i in range(6):
    s, h = Sso[i, i], Ssh[i, i]
    pd = 100.0 * (h - s) / s
    print("  %-4s solid %13.4e  shell %13.4e  %+7.2f%%" % (LBL[i], s, h, pd))

if "C6L" in so.files and "C6L" in sh.files:
    print("\n=== boundary rings (diagonal, shell vs solid) ===")
    for side in ("L", "R"):
        print("  -- %s ring --" % side)
        Ds, Dh = so["C6" + side], sh["C6" + side]
        for i in range(6):
            pd = 100.0 * (Dh[i, i] - Ds[i, i]) / Ds[i, i]
            print("    %-4s solid %13.4e  shell %13.4e  %+7.2f%%"
                  % (LBL[i], Ds[i, i], Dh[i, i], pd))
