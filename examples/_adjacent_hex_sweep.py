"""_adjacent_hex_sweep.py -- does the HEX loft work for EVERY adjacent windIO station pair?
Reports inverted-hex count + min scaled Jacobian per adjacent pair, at the default mesh and
at a fine-hoop mesh (0.01), so we know exactly which adjacent pairs hex can/can't handle."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import hex_between_sections, _hex_min_sj

W = "examples/data/IEA-22-280-RWT.yaml"
blade = load_blade(W)
raw = __import__("yaml").safe_load(open(W))
afs = raw["components"]["blade"]["outer_shape"]["airfoils"]
stations = sorted((float(a["spanwise_position"]), str(a["name"])) for a in afs)


def zof(r):
    return r * 137.0


def meshable(r, name):
    if "circular" in name.lower():
        return False
    try:
        cs = build_cross_section(blade, r, mesh_size=0.05)
        return len(cs["webs"]) >= 1
    except Exception:
        return False


ms = [(r, n) for r, n in stations if meshable(r, n)]
print("meshable windIO stations: %d" % len(ms))
print("%-18s %-16s %-8s %-8s | %-10s %-10s | %-10s %-10s" %
      ("pair r1->r2", "dchord", "gap", "dz[m]", "inv@0.02", "minSJ", "inv@0.01", "minSJ"))
csc = {}
def cs_of(r, m):
    if (r, m) not in csc:
        csc[(r, m)] = build_cross_section(blade, r, mesh_size=m)
    return csc[(r, m)]

bad = []
for (r1, _), (r2, _) in zip(ms[:-1], ms[1:]):
    z1, z2 = zof(r1), zof(r2)
    row = []
    for m in (0.02, 0.01):
        try:
            c1, c2 = cs_of(r1, m), cs_of(r2, m)
            res = hex_between_sections(c1, c2, z1, z2, nr=4, nsp=8, nw=3, mesh_size=m)
            sj = _hex_min_sj(res["nodes"], res["hexes"])
            row.append((int((sj < 0).sum()), float(sj.min())))
        except Exception as e:
            row.append(("ERR:" + type(e).__name__, 0.0))
    ch1, ch2 = blade.scalar("chord", r1), blade.scalar("chord", r2)
    inv02 = row[0][0]
    flag = "" if (isinstance(inv02, int) and inv02 == 0) else "  <-- FAILS"
    print("%-18s %5.2f->%-5.2f    %.3f    %5.1f | %-10s %+10.3f | %-10s %+10.3f%s" %
          ("%.3f->%.3f" % (r1, r2), ch1, ch2, r2 - r1, z2 - z1,
           str(row[0][0]), row[0][1], str(row[1][0]), row[1][1], flag))
    if not (isinstance(inv02, int) and inv02 == 0):
        bad.append((r1, r2, inv02))

print("\nadjacent pairs where HEX fails at mesh=0.02: %d of %d" % (len(bad), len(ms) - 1))
for r1, r2, n in bad:
    print("  %.3f -> %.3f : %s inverted" % (r1, r2, n))
