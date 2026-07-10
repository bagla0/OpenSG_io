import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.tapered_tet import windio_taper_hex

blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
for r1, r2, tag in [(0.1967, 0.2470, "MILD"), (0.3993, 0.5336, "STEEP")]:
    cs1 = build_cross_section(blade, r1, mesh_size=0.02)
    cs2 = build_cross_section(blade, r2, mesh_size=0.02)
    try:
        nodes, hexes, oris, hmats = windio_taper_hex(cs1, cs2, r1 * 137, r2 * 137, nr=4, nw=3, mesh_size=0.02)
        print("%s OK: %d nodes / %d hexes / %d mats" % (tag, len(nodes), len(hexes), len(set(hmats))), flush=True)
    except Exception as e:
        print("%s FAIL: %s" % (tag, str(e)[:120]), flush=True)
