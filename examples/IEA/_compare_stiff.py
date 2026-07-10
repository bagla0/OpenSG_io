"""Debug summary: does stiffening the foam shear modulus collapse the GA/GJ shell-vs-
solid gap?  (If yes -> the 13x gap is the soft-core transverse-shear effect, not a mesh
bug.)  Compares the original and stiff-foam Timo diagonals."""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def load(tag):
    return np.load(os.path.join(OUT, "timo_%s.npz" % tag))["S6"]


sh0, so0 = load("shell"), load("solid")
sh1, so1 = load("shell_stiff"), load("solid_stiff")
print("IEA-22 taper segment -- shell/solid diagonal, ORIGINAL foam (G=54 MPa) vs "
      "STIFF foam (G=G_skin)\n")
print("       %-13s %-13s %8s   |  %-13s %-13s %8s" %
      ("solid(orig)", "shell(orig)", "gap", "solid(stiff)", "shell(stiff)", "gap"))
for i in range(6):
    g0 = sh0[i, i] / so0[i, i]
    g1 = sh1[i, i] / so1[i, i]
    print(" %-4s %13.4e %13.4e %7.1fx   |  %13.4e %13.4e %7.2fx"
          % (LBL[i], so0[i, i], sh0[i, i], g0, so1[i, i], sh1[i, i], g1))
print("\nInterpretation: a GA2/GA3/GJ gap that is ~13x with soft foam but ~1x with stiff "
      "foam\nproves the discrepancy is the soft-core transverse-shear effect -- the mesh, "
      "orientation\nand layup are identical, only the foam shear modulus changed.")
