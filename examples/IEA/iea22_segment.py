"""IEA-22 3-D tapered segment -- standalone example.

Builds the tapered segment of the IEA-22-280 blade between the r = 0.2 and r = 0.3
span stations from the bundled windIO definition (examples/data/IEA-22-280-RWT.yaml),
producing BOTH SG meshes and writing every artifact to examples/IEA/output/:

  * conforming structured 8-node HEX solid   -> output/iea22_seg_solid.yaml
  * equivalent mid-surface QUAD shell        -> output/iea22_seg_shell.yaml
  * mesh renders: 3-D solid, 3-D shell, and cross-sections at BOTH ends (shell+solid).

Run from anywhere:

    python examples/IEA/iea22_segment.py            # bundled IEA-22 windIO
    python examples/IEA/iea22_segment.py <windio.yaml> <r1> <r2>
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import (hex_between_sections, shell_between_sections,
                                solid_yaml_payload, shell_yaml_payload,
                                assert_shell_conforming)
from opensg_io.conformity import assert_conforming, min_scaled_jacobian
from opensg_io.render3d import render_mesh_png, render_section_png, render_section_ends

# ----- inputs (windIO from the repo's examples/data, NOT an absolute path) -----------
WINDIO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "examples", "data",
                                                            "IEA-22-280-RWT.yaml")
R1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
R2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
NR, NSP, NW, MESH = 4, 12, 2, 0.02
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


def main():
    print("windIO:", os.path.relpath(WINDIO, REPO), flush=True)
    blade = load_blade(WINDIO)
    cs1 = build_cross_section(blade, R1, mesh_size=MESH)
    cs2 = build_cross_section(blade, R2, mesh_size=MESH)
    try:                                                    # span z from the reference axis
        ra = blade.osh["reference_axis"]["z"]
        z1 = float(np.interp(R1, ra["grid"], ra["values"]))
        z2 = float(np.interp(R2, ra["grid"], ra["values"]))
    except Exception:
        z1, z2 = R1 * 137.0, R2 * 137.0
    print("stations: r=%.2f chord=%.3f (%d webs) | r=%.2f chord=%.3f  ->  z=[%.2f, %.2f] m"
          % (R1, cs1["chord"], len(cs1["webs"]), R2, cs2["chord"], z1, z2), flush=True)

    # ---- structured hex solid ----------------------------------------------------
    res = hex_between_sections(cs1, cs2, z1, z2, nr=NR, nsp=NSP, nw=NW, mesh_size=MESH)
    nodes, hexes, sec = res["nodes"], res["hexes"], res["sec"]
    print("HEX: %d nodes, %d hexes  (section: %d hoop nodes x %d layers + webs NY=%s)"
          % (len(nodes), len(hexes), sec["NC"], NR, sec["NYs"]), flush=True)
    for si, s in enumerate(sec["st"]):
        print("  station %d: TE opening max %.2f mm (full laminate preserved: min ply scale %.4f)"
              % (si, 1e3 * s["te_moved"].max(), s["fscale"].min()), flush=True)
    assert_conforming(nodes, hexes, "hex")
    msj, ninv = min_scaled_jacobian(nodes, hexes)
    assert ninv == 0, "%d inverted hexes" % ninv
    print("conformity gate (solid): PASS   min scaled Jacobian = %.3f (0 inverted)" % msj, flush=True)

    oris, hmats = solid_yaml_payload(res, cs1)
    mat_names = sorted(set(hmats))
    solid = dict(nodes=nodes, hexes=hexes, oris=oris, hmats=hmats, mat_names=mat_names)
    sets = {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                        for m in mat_names]}
    mats = [{"name": m, **{k: _mat_block(blade, m)["elastic"][k] for k in ("E", "G", "nu")},
             "rho": _mat_block(blade, m)["density"]} for m in mat_names]
    from opensg_io.mesh3d import export_solid_yaml
    export_solid_yaml(os.path.join(OUT, "iea22_seg_solid.yaml"), nodes, hexes, "hex",
                      oris, mats, sets=sets)
    print("wrote output/iea22_seg_solid.yaml", flush=True)

    # ---- equivalent mid-surface shell -------------------------------------------
    shell = shell_between_sections(res, cs1)
    njunc = assert_shell_conforming(shell, len(cs1["webs"]), NSP)
    print("conformity (shell, branched): PASS  (%d nodes, %d quads; %d T-junction edges)"
          % (len(shell["nodes"]), len(shell["quads"]), njunc), flush=True)
    yaml.safe_dump(shell_yaml_payload(shell, blade, _mat_block),
                   open(os.path.join(OUT, "iea22_seg_shell.yaml"), "w"),
                   default_flow_style=None, sort_keys=False)
    print("wrote output/iea22_seg_shell.yaml", flush=True)

    # ---- renders -----------------------------------------------------------------
    render_section_png(sec, os.path.join(OUT, "iea22_loft_input.png"),
                       "IEA-22 r=%.2f loft input (webs crimson, junction bands refined)" % R1)
    render_section_ends(sec, shell["sec2d"], R1, R2,
                        os.path.join(OUT, "iea22_sections.png"),
                        "IEA-22 tapered segment -- cross-sections at both ends")
    hset = {m: i for i, m in enumerate(mat_names)}
    render_mesh_png(nodes, hexes, "hex", np.array([hset[m] for m in hmats], int),
                    os.path.join(OUT, "iea22_solid_3d.png"),
                    "IEA-22 segment r=%.2f->%.2f: structured HEX (%d hexes, by material)"
                    % (R1, R2, len(hexes)))
    render_mesh_png(shell["nodes"], shell["quads"], "quad", np.array(shell["qlam"], int),
                    os.path.join(OUT, "iea22_shell_3d.png"),
                    "IEA-22 segment r=%.2f->%.2f: mid-surface SHELL (%d quads, by layup)"
                    % (R1, R2, len(shell["quads"])))
    for n in ("iea22_loft_input", "iea22_sections", "iea22_solid_3d", "iea22_shell_3d"):
        print("wrote output/%s.png" % n, flush=True)
    return solid, shell


if __name__ == "__main__":
    main()
