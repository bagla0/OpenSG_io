"""_web_reversal.py -- localize the taper hex inversion to specific WEB plates and quantify
why the linear node loft folds them (orientation reversal between the two stations)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh

blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
r1, r2 = 0.1967, 0.3993
cs1 = build_cross_section(blade, r1, mesh_size=0.02)
cs2 = build_cross_section(blade, r2, mesh_size=0.02)
skel = section_skeleton([cs1, cs2], mesh_size=0.02, nw=3)
sec = build_section_mesh([cs1, cs2], skel, nr=4)
P1, P2 = sec["stations"]
faces, ftag = sec["faces2d"], sec["ftag"]


def area(P, f):
    q = P[f, :2]; a = 0.0
    for k in range(4):
        b = (k + 1) % 4
        a += q[k, 0] * q[b, 1] - q[b, 0] * q[k, 1]
    return 0.5 * a


# per-web count of faces that reverse orientation (A0>0 -> A1<0)
rev = {}
for f, tg in zip(faces, ftag):
    if tg[0] != "web":
        continue
    a0, a1 = area(P1, f), area(P2, f)
    wi = tg[1]
    rev.setdefault(wi, [0, 0])
    rev[wi][1] += 1
    if a0 > 0 and a1 < 0:
        rev[wi][0] += 1
print("per-web face-orientation reversal (station-0 CCW -> station-1 CW):")
for wi in sorted(rev):
    print("  web %d: %d of %d faces reverse" % (wi, rev[wi][0], rev[wi][1]))

# per-web geometry change: inner-skin attachment points + web-line angle at each station
wpair = sec["wpair"]

def sid(i, l, nr=4):
    return i * (nr + 1) + l


print("\nper-web plate geometry (inner-skin attach points, web-line angle, across-thickness sense):")
for wi, (top, bot) in enumerate(wpair):
    for si, (P, r) in enumerate([(P1, r1), (P2, r2)]):
        pt = P[sid(top[len(top) // 2], 4), :2]                  # mid suction attach
        pb = P[sid(bot[len(bot) // 2], 4), :2]                  # mid pressure attach
        line = pb - pt
        ang = np.degrees(np.arctan2(line[1], line[0]))
        # across-thickness direction (column 0 -> last) at the top attach
        acr = P[sid(top[-1], 4), :2] - P[sid(top[0], 4), :2]
        cross_z = line[0] * acr[1] - line[1] * acr[0]           # sign = plate winding sense
        print("  web %d @ r=%.4f: line_angle=%+7.2f deg  |line|=%.3f  winding_sign=%+.4f"
              % (wi, r, ang, np.hypot(*line), np.sign(cross_z) * min(abs(cross_z), 9.99)))
    print()
