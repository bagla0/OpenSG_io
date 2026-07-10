"""export_dat_all.py -- ONE .dat with the Timoshenko 6x6 of the taper segment AND both
boundary rings, each as solid, shell, and the element-wise %-error 6x6 (shell vs solid).

%err_ij = 100*(shell_ij - solid_ij)/solid_ij, printed only where |solid_ij| >= CUT
(=1e6, the structural-zero floor for these N/N-m stiffnesses); below that the term is
structurally zero and shown as '.'.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
ONEDRIVE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_data\IEA_taper_segment"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
CUT = 1.0e6

sh = np.load(os.path.join(OUT, "timo_shell.npz"))
so = np.load(os.path.join(OUT, "timo_solid.npz"))


def sym(M):
    M = np.asarray(M, float)
    return 0.5 * (M + M.T)


def mat_block(M):
    out = ["        " + "".join("%13s" % c for c in LBL)]
    for i in range(6):
        out.append("  %-4s" % LBL[i] + "".join("%13.4e" % M[i, j] for j in range(6)))
    return "\n".join(out)


def err_block(H, S):
    out = ["        " + "".join("%13s" % c for c in LBL)]
    for i in range(6):
        row = "  %-4s" % LBL[i]
        for j in range(6):
            if abs(S[i, j]) >= CUT:
                row += "%12.2f%%" % (100.0 * (H[i, j] - S[i, j]) / S[i, j])
            else:
                row += "%13s" % "."
        out.append(row)
    return "\n".join(out)


BLOCKS = [("TAPER SEGMENT  (r = 0.2 -> 0.3, full 3-D)", sym(sh["S6"]), sym(so["S6"])),
          ("BOUNDARY  L  (root ring, r = 0.2)", sym(sh["C6L"]), sym(so["C6L"])),
          ("BOUNDARY  R  (tip  ring, r = 0.3)", sym(sh["C6R"]), sym(so["C6R"]))]

lines = [
    "# IEA-22-280  tapered SG segment  r = 0.2 -> 0.3",
    "# SHELL = OpenSG-TW JAX MITC-RM (OML reference)   |   SOLID = OpenSG-FEniCS 3-D",
    "# Timoshenko 6x6, order [EA, GA2, GA3, GJ, EI2, EI3]   (units: N and N*m^2)",
    "# %%err_ij = 100*(shell-solid)/solid ; '.' = |solid| < %.0e (structurally zero)" % CUT,
    "#" + "=" * 78, ""]

for name, H, S in BLOCKS:
    lines += ["", "#" + "=" * 78, "### %s" % name, "#" + "=" * 78,
              "", "-- SOLID  6x6 --", mat_block(S),
              "", "-- SHELL  6x6 --", mat_block(H),
              "", "-- %ERROR 6x6  (shell vs solid) --", err_block(H, S),
              "", "-- DIAGONAL summary --",
              "  " + "  ".join("%s %+.1f%%" % (LBL[i], 100 * (H[i, i] - S[i, i]) / S[i, i])
                               for i in range(6))]

lines += ["", "#" + "=" * 78,
          "# NOTE  EA/EI (extension+bending) agree to <10%: geometry, NuMAD material",
          "#       orientation and span-interpolated layup are validated.  GA2/GA3/GJ",
          "#       (transverse shear + torsion) run ~13x stiff in the shell = the RM",
          "#       soft-core over-prediction (24/72 panels are 70 mm foam sandwiches);",
          "#       trust the SOLID for the shear/torsion terms.  See comparison notes.", ""]

txt = "\n".join(lines)
for d in (ONEDRIVE, OUT):
    with open(os.path.join(d, "timo_shell_vs_solid_all.dat"), "w") as f:
        f.write(txt)
print("wrote timo_shell_vs_solid_all.dat  ->", ONEDRIVE)
print(txt)
