"""webbed_ellipse_hex_demo.py -- reproduce the structured 3-web ellipse HEX solid.

The hex solid is built by the LOFT feature (opensg_io.mesh3d.webbed_ellipse_hex ->
loft_to_hex): a parametric structured QUAD cross-section (skin annulus with nr
through-thickness layers + three web plates whose top/bottom node rows ARE inner-skin
nodes, so the T-junctions are watertight) is stacked at nsp+1 span stations and each
quad is connected to itself at the next station -> one conforming 8-node hex per quad
per slice.  Note this is NOT lofted from the PreVABS cross-section: PreVABS meshes the
section with TRIANGLES (see opensg_io.prevabs_webbed_ellipse), which loft to prisms.

    python examples/webbed_ellipse_hex_demo.py [t]

Prints the mesh size, runs the mandatory conformity gate, and writes a PNG of the
z=0 quad cross-section + the lofted hex segment.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opensg_io.mesh3d import webbed_ellipse_hex
from opensg_io.conformity import assert_conforming

t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
nodes, hexes, isweb = webbed_ellipse_hex(t, nr=4, nsp=20, nw=4, nct=100)
print("loft_to_hex: %d nodes, %d hexes (%d skin, %d web)"
      % (len(nodes), len(hexes), int((~isweb.astype(bool)).sum()), int(isweb.sum())))
assert_conforming(nodes, hexes, "hex")
print("conformity gate: PASS")

M = len(hexes) // 20
q0 = hexes[:M, :4]
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121)
for q, w in zip(q0, isweb[:M]):
    lp = list(q) + [int(q[0])]
    ax.plot(nodes[lp, 0], nodes[lp, 1], color=("crimson" if w else "0.35"), lw=0.5)
ax.set_aspect("equal"); ax.set_title("z=0 quad cross-section (loft input)")
ax3 = fig.add_subplot(122, projection="3d")
for k in range(0, len(hexes), max(1, len(hexes) // 400)):
    h = hexes[k]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax3.plot(nodes[[h[a], h[b]], 2], nodes[[h[a], h[b]], 0], nodes[[h[a], h[b]], 1],
                 color=("crimson" if isweb[k] else "0.6"), lw=0.25)
ax3.set_title("structured HEX (lofted along span)"); ax3.view_init(18, -70)
try:
    ax3.set_box_aspect((3, 1, 1))
except Exception:
    pass
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webbed_ellipse_hex_demo.png")
fig.savefig(out, dpi=100)
print("wrote", out)
