"""iea22_hex_segment.py -- GENERAL two-station hex loft applied to the IEA-22-280 blade.

Builds the 3-D tapered segment between r=0.2 and r=0.3 from the windIO definition:
  * conforming structured 8-node HEX solid (skin through-thickness layers + 3 webs with
    refined, node-shared junction bands)  -> iea22_seg_r020_r030_solid.yaml
  * the EQUIVALENT mid-surface QUAD shell segment (same hoop skeleton, same stations)
    -> iea22_seg_r020_r030_shell.yaml
plus cross-section / 3-D PNG renders.  The conformity gate is mandatory on both.

    python examples/iea22_hex_segment.py [windio_yaml]
"""
import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload, _thick, _lam_tuple
from opensg_io.mesh3d import export_solid_yaml
from opensg_io.conformity import assert_conforming
import yaml

WINDIO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "data", "IEA-22-280-RWT.yaml")
R1, R2 = 0.2, 0.3
NR, NSP, NW, MESH = 4, 12, 2, 0.02

print("loading blade:", WINDIO, flush=True)
blade = load_blade(WINDIO)
cs1 = build_cross_section(blade, R1, mesh_size=MESH)
cs2 = build_cross_section(blade, R2, mesh_size=MESH)
# physical span positions from the windIO reference axis (z along the blade)
try:
    ra = blade.osh["reference_axis"]["z"]
    L = float(np.interp(1.0, ra["grid"], ra["values"]))
    z1, z2 = float(np.interp(R1, ra["grid"], ra["values"])), float(np.interp(R2, ra["grid"], ra["values"]))
except Exception:
    L = 137.0; z1, z2 = R1 * L, R2 * L
print("stations: r=%.2f chord=%.3f (%d webs) | r=%.2f chord=%.3f  ->  z=[%.2f, %.2f] m"
      % (R1, cs1["chord"], len(cs1["webs"]), R2, cs2["chord"], z1, z2), flush=True)

res = hex_between_sections(cs1, cs2, z1, z2, nr=NR, nsp=NSP, nw=NW, mesh_size=MESH)
nodes, hexes, sec = res["nodes"], res["hexes"], res["sec"]
print("HEX: %d nodes, %d hexes  (section: %d hoop nodes x %d layers + webs NY=%s)"
      % (len(nodes), len(hexes), sec["NC"], NR, sec["NYs"]), flush=True)
assert_conforming(nodes, hexes, "hex")
from opensg_io.conformity import min_scaled_jacobian
msj, ninv = min_scaled_jacobian(nodes, hexes)
assert ninv == 0, "%d inverted hexes" % ninv
print("conformity gate (solid): PASS   min scaled Jacobian = %.3f (0 inverted)" % msj, flush=True)

# ---- solid yaml (per-hex fiber frames from the ply at that through-thickness depth)
oris, hmats = solid_yaml_payload(res, cs1)
mat_names = sorted(set(hmats))
sets = {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                    for m in mat_names]}
mats = []
for m in mat_names:
    b = _mat_block(blade, m)
    mats.append({"name": m, "E": b["elastic"]["E"], "G": b["elastic"]["G"],
                 "nu": b["elastic"]["nu"], "rho": b["density"]})
solid_path = os.path.join(HERE, "iea22_seg_r020_r030_solid.yaml")
export_solid_yaml(solid_path, nodes, hexes, "hex", oris, mats, sets=sets)
print("wrote", solid_path, flush=True)

# ---- EQUIVALENT quad shell segment: mid-surface hoop + web mid-columns, same stations
P1, P2 = sec["stations"]
NC, nr = sec["NC"], sec["nr"]


def mid_ids():
    ids = {}
    for i in range(NC):
        ids[("s", i)] = len(ids)
    for wi, NY in enumerate(sec["NYs"]):
        for m in range(1, NY):
            ids[("w", wi, m)] = len(ids)
    return ids


IDS = mid_ids()
NPs = len(IDS)


def station_shell(P):
    X = np.zeros((NPs, 3))
    for i in range(NC):
        X[IDS[("s", i)], :2] = 0.5 * (P[i * (nr + 1) + 0, :2] + P[i * (nr + 1) + nr, :2])
    for wi, NY in enumerate(sec["NYs"]):
        top, bot = sec["wpair"][wi]
        jmid = len(top) // 2
        for m in range(1, NY):
            tau = 0.5 * (1 - math.cos(math.pi * m / NY))
            pt = X[IDS[("s", top[jmid])], :2]; pb = X[IDS[("s", bot[jmid])], :2]
            X[IDS[("w", wi, m)], :2] = (1 - tau) * pt + tau * pb
    return X


def wnode(wi, m):
    top, bot = sec["wpair"][wi]; NY = sec["NYs"][wi]; jm = len(top) // 2
    if m == 0:
        return IDS[("s", top[jm])]
    if m == NY:
        return IDS[("s", bot[jm])]
    return IDS[("w", wi, m)]


S1, S2 = station_shell(P1), station_shell(P2)
snodes = np.zeros(((NSP + 1) * NPs, 3))
for s in range(NSP + 1):
    tau = s / NSP
    snodes[s * NPs:(s + 1) * NPs, :2] = (1 - tau) * S1[:, :2] + tau * S2[:, :2]
    snodes[s * NPs:(s + 1) * NPs, 2] = (1 - tau) * z1 + tau * z2

hoop_kind = sec["st"][0]["hoop_kind"]
lam_by_id = {sid: lam for lam, sid in cs1["laminates"].items()}
quads, qlam, qweb = [], [], []
for s in range(NSP):
    for i in range(NC):
        ii = (i + 1) % NC
        kind = res["skel"]["kinds"][hoop_kind[i]]
        sid_lam = kind[1] if kind[0] == "skin" else None
        if sid_lam is None:                                # junction band strip carries the local skin layup
            sid_lam = [sg["set_id"] for sg in cs1["segments"]
                       if sg["s_a"] - 1e-9 <= sec["st"][0]["hoop_s"][i] <= sg["s_b"] + 1e-9][0]
        quads.append([s * NPs + IDS[("s", i)], s * NPs + IDS[("s", ii)],
                      (s + 1) * NPs + IDS[("s", ii)], (s + 1) * NPs + IDS[("s", i)]])
        qlam.append(sid_lam); qweb.append(False)
    for wi, NY in enumerate(sec["NYs"]):
        for m in range(NY):
            quads.append([s * NPs + wnode(wi, m), s * NPs + wnode(wi, m + 1),
                          (s + 1) * NPs + wnode(wi, m + 1), (s + 1) * NPs + wnode(wi, m)])
            qlam.append(cs1["webs"][wi]["lam"]); qweb.append(True)
quads = np.array(quads, int)
# shell (branched mid-surface) conformity: every node referenced, junction edges shared by
# EXACTLY 3 quads (skin-left + skin-right + web = watertight T-junction), nothing >3.
used_nodes = np.zeros(len(snodes), bool); used_nodes[quads.ravel()] = True
assert used_nodes.all(), "hanging shell nodes"
from collections import Counter
ec = Counter()
for qq in quads:
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ec[tuple(sorted((int(qq[a]), int(qq[b]))))] += 1
over = [e for e, c in ec.items() if c > 3]
junc = [e for e, c in ec.items() if c == 3]
expected_junc = 2 * len(cs1["webs"]) * NSP                 # 2 attach lines/web x NSP span edges
assert not over, "shell edges shared by >3 quads: %d" % len(over)
assert len(junc) == expected_junc, "junction edges %d != expected %d" % (len(junc), expected_junc)
print("conformity (shell, branched): PASS  (%d nodes, %d quads; %d T-junction edges as expected)"
      % (len(snodes), len(quads), len(junc)), flush=True)

soris = np.zeros((len(quads), 9))
for k, q in enumerate(quads):
    gen = snodes[q[3]] - snodes[q[0]]; a1 = gen / np.linalg.norm(gen)
    e2r = snodes[q[1]] - snodes[q[0]]; e2 = e2r - (e2r @ a1) * a1; e2 /= np.linalg.norm(e2)
    e3 = np.cross(a1, e2); e3 /= np.linalg.norm(e3)
    soris[k] = np.concatenate([a1, e2, e3])
used = sorted(set(qlam))
shell = {"nodes": [["%.9f %.9f %.9f" % tuple(p)] for p in snodes],
         "elements": [[" ".join(str(v + 1) for v in q)] for q in quads],
         "sets": {"element": [{"name": "layup_%d" % l,
                               "labels": [k + 1 for k, ql in enumerate(qlam) if ql == l]} for l in used]},
         "sections": [{"elementSet": "layup_%d" % l, "type": "shell",
                       "layup": [[m, float(t), float(a)] for (m, t, a) in lam_by_id[l]]} for l in used],
         "elementOrientations": [[float(v) for v in o] for o in soris],
         "materials": [_mat_block(blade, m) for m in sorted({mm for l in used for (mm, _t, _a) in lam_by_id[l]})]}
shell_path = os.path.join(HERE, "iea22_seg_r020_r030_shell.yaml")
yaml.safe_dump(shell, open(shell_path, "w"), default_flow_style=None, sort_keys=False)
print("wrote", shell_path, flush=True)

# ---- renders: the ACTUAL r=0.2 section mesh (with TE zoom) + shaded 3-D hex
from opensg_io.render3d import render_section_png, render_mesh_png
png = os.path.join(HERE, "iea22_hex_segment.png")
render_section_png(sec, png,
                   "IEA-22 r=0.2 quad cross-section (loft input; webs crimson, junction bands refined)")
print("wrote", png, flush=True)
hsets = {m: i for i, m in enumerate(mat_names)}
render_mesh_png(nodes, hexes, "hex", np.array([hsets[m] for m in hmats], int),
                os.path.join(HERE, "iea22_hex_3d.png"),
                "IEA-22 segment r=0.2->0.3: structured HEX (%d hexes, colored by material)" % len(hexes))
print("wrote", os.path.join(HERE, "iea22_hex_3d.png"), flush=True)
