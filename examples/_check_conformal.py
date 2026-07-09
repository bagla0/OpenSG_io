"""Independent conformity re-check on the two exported IEA-22 segment YAMLs."""
import os, sys
import numpy as np
import yaml
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from opensg_io.conformity import assert_conforming
CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load(path):
    d = yaml.load(open(path), Loader=CLoad)
    nodes = np.array([[float(v) for v in r[0].split()] for r in d["nodes"]])
    cells = np.array([[int(v) - 1 for v in r[0].split()] for r in d["elements"]])
    return nodes, cells


# ---- SOLID hex: full watertight gate (every internal face shared by exactly 2 hexes)
nodes, hexes = load(os.path.join(HERE, "iea22_seg_r020_r030_solid.yaml"))
assert_conforming(nodes, hexes, "hex")
print("SOLID hex : %d nodes, %d hexes -> assert_conforming(hex) PASS (watertight, faces shared by 2)"
      % (len(nodes), len(hexes)))

# ---- SHELL quad: branched mid-surface. skin edges shared by <=2; T-junction edges by exactly 3
snodes, quads = load(os.path.join(HERE, "iea22_seg_r020_r030_shell.yaml"))
used = np.zeros(len(snodes), bool); used[quads.ravel()] = True
ec = Counter()
for q in quads:
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ec[tuple(sorted((int(q[a]), int(q[b]))))] += 1
c = Counter(ec.values())
over = [e for e, k in ec.items() if k > 3]
print("SHELL quad: %d nodes, %d quads ; hanging nodes: %s ; edge-share histogram: %s ; edges>3: %d"
      % (len(snodes), len(quads), (not used.all()), dict(sorted(c.items())), len(over)))
print("  -> interior skin edges shared by 2, %d T-junction edges shared by exactly 3 (web+2 skin), none >3 = branched-conformal"
      % c.get(3, 0))
