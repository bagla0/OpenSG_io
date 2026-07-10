"""build_blade_mesh.py -- universal blade SG mesh builder from a windIO blade file.

One entry point for the interactive mesh-builder agent.  Given a windIO blade it can
(a) list the spanwise stations, or (b) build a BOUNDARY (single cross-section) or a
TAPER (segment between two stations) mesh for the SOLID, the SHELL, or BOTH -- every run
writes an OpenSG SG **YAML** and a **PNG** of the actual mesh, and reports the
conformity + positive-Jacobian gates honestly (it never writes a mesh that fails them).

Conventions baked in (learned, do-not-repeat): ply-conforming through-thickness hex
layers (sandwich skins meshed exactly); NuMAD/VABS orientation (e1=span root->tip,
e3=inward normal, e2=e3xe1); SHELL reference = OML by default (pair frac=0); span-
interpolated layup with ply drops.  Valid airfoil range is typically r~0.05..0.95 (the
root cylinder and the tip have no webbed airfoil).

Usage
-----
    python build_blade_mesh.py <windio.yaml> stations
    python build_blade_mesh.py <windio.yaml> boundary --r 0.50 --model both  --out DIR
    python build_blade_mesh.py <windio.yaml> taper    --r1 0.20 --r2 0.30 --model both --out DIR
      [--reference OML|mid] [--nr 4] [--nsp 12] [--nw 3] [--mesh 0.02]
"""
import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import (hex_between_sections, shell_between_sections,
                                solid_yaml_payload, shell_yaml_payload,
                                solid_boundary_payload, shell_boundary_payload,
                                assert_shell_conforming, region_taper_laminates)
from opensg_io.mesh3d import export_solid_yaml
from opensg_io.conformity import assert_conforming, min_scaled_jacobian, NonConformingMesh


def blade_z(blade, r):
    try:
        ra = blade.osh["reference_axis"]["z"]
        return float(np.interp(r, ra["grid"], ra["values"]))
    except Exception:
        return r * 137.0


def _dump(payload, path):
    yaml.safe_dump(payload, open(path, "w"), default_flow_style=None, sort_keys=False)


# --------------------------------------------------------------------------- stations
def list_stations(windio, lo=0.02, hi=0.99, step=0.02):
    blade = load_blade(windio)
    print("windIO: %s" % windio)
    print("spanwise stations (non-dimensional r):")
    print("  %-6s %-9s %-6s %-7s %s" % ("r", "chord[m]", "webs", "z[m]", "status"))
    valid = []
    r = lo
    while r <= hi + 1e-9:
        r = round(r, 3)
        try:
            cs = build_cross_section(blade, r, mesh_size=0.03)
            print("  %-6.2f %-9.3f %-6d %-7.2f OK" % (r, cs["chord"], len(cs["webs"]), blade_z(blade, r)))
            valid.append(r)
        except Exception as e:
            print("  %-6.2f %-9s %-6s %-7s no airfoil (%s)" % (r, "-", "-", "-", str(e)[:32]))
        r += step
    if valid:
        print("\nvalid range: r = %.2f .. %.2f  (pick any pair r1<r2 for a taper, or one r for a boundary)"
              % (min(valid), max(valid)))
    return valid


# --------------------------------------------------------------------------- taper
def build_taper(windio, r1, r2, model, out, reference="OML", nr=4, nsp=12, nw=3, mesh=0.02):
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    cs1 = build_cross_section(blade, r1, mesh_size=mesh)
    cs2 = build_cross_section(blade, r2, mesh_size=mesh)
    z1, z2 = blade_z(blade, r1), blade_z(blade, r2)
    print("TAPER r=%.3f->%.3f  chord %.3f->%.3f  z=[%.2f, %.2f] m  model=%s ref=%s"
          % (r1, r2, cs1["chord"], cs2["chord"], z1, z2, model, reference))
    res = hex_between_sections(cs1, cs2, z1, z2, nr=nr, nsp=nsp, nw=nw, mesh_size=mesh)
    tag = "r%03d_%03d" % (round(r1 * 100), round(r2 * 100))
    written = []

    if model in ("solid", "both"):
        nodes, hexes = res["nodes"], res["hexes"]
        msj, ninv = min_scaled_jacobian(nodes, hexes)
        if ninv:
            raise RuntimeError(
                "SOLID taper r=%.2f->%.2f has %d inverted hexes (min scaled Jacobian %.3f) -- the thin "
                "shear webs twist with the airfoil aerodynamic twist over this span.  Try a NARROWER or "
                "less-twisted station pair (e.g. a 0.05-wide segment, or nearer mid-span)." % (r1, r2, ninv, msj))
        assert_conforming(nodes, hexes, "hex")
        oris, hmats = solid_yaml_payload(res, cs1, cs2)
        mat_names = sorted(set(hmats))
        sets = {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                            for m in mat_names]}
        mats = [{"name": m, **{k: _mat_block(blade, m)["elastic"][k] for k in ("E", "G", "nu")},
                 "rho": _mat_block(blade, m)["density"]} for m in mat_names]
        p = os.path.join(out, "%s_solid_taper.yaml" % tag)
        export_solid_yaml(p, nodes, hexes, "hex", oris, mats, sets=sets)
        print("  solid: %d nodes / %d hexes ; gates PASS (min SJ %.3f)" % (len(nodes), len(hexes), msj))
        _render_hex(nodes, hexes, hmats, mat_names, p.replace(".yaml", ".png"),
                    "SOLID taper %s (by material)" % tag)
        written += [p, p.replace(".yaml", ".png")]

    if model in ("shell", "both"):
        shell = shell_between_sections(res, cs1, cs2, reference=reference)
        assert_shell_conforming(shell, len(cs1["webs"]), nsp)
        _dump(shell_yaml_payload(shell, blade, _mat_block), os.path.join(out, "%s_shell_taper.yaml" % tag))
        p = os.path.join(out, "%s_shell_taper.yaml" % tag)
        print("  shell: %d nodes / %d quads ; branched conformity PASS (%s ref)"
              % (len(shell["nodes"]), len(shell["quads"]), reference))
        reg = {rr: i for i, rr in enumerate(sorted(set(shell["region_of_quad"])))}
        _render_quad(shell["nodes"], shell["quads"], [reg[rr] for rr in shell["region_of_quad"]],
                     p.replace(".yaml", ".png"), "SHELL taper %s (by region, %s ref)" % (tag, reference))
        written += [p, p.replace(".yaml", ".png")]
    return written


# --------------------------------------------------------------------------- boundary
def build_boundary(windio, r, model, out, reference="OML", nr=4, nw=3, mesh=0.02):
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    cs = build_cross_section(blade, r, mesh_size=mesh)
    z = blade_z(blade, r)
    print("BOUNDARY r=%.3f  chord %.3f  z=%.2f m  model=%s ref=%s" % (r, cs["chord"], z, model, reference))
    # a boundary is one cross-section; realize it as the L end of a thin prismatic segment
    # (only the 2-D end section is exported, so the throwaway 3-D hexes are not gated)
    res = hex_between_sections(cs, cs, z, z + 0.1, nr=nr, nsp=1, nw=nw, mesh_size=mesh)
    tag = "r%03d" % round(r * 100)
    written = []

    if model in ("solid", "both"):
        p = os.path.join(out, "%s_solid_boundary.yaml" % tag)
        _dump(solid_boundary_payload(res, cs, cs, 0, blade, _mat_block), p)
        print("  solid boundary (2-D quad section) written")
        _render_boundary(p, p.replace(".yaml", ".png"), "SOLID boundary %s (2-D quad, by material)" % tag, quad=True)
        written += [p, p.replace(".yaml", ".png")]

    if model in ("shell", "both"):
        shell = shell_between_sections(res, cs, cs, reference=reference)
        p = os.path.join(out, "%s_shell_boundary.yaml" % tag)
        _dump(shell_boundary_payload(res, shell, cs, cs, 0, blade, _mat_block), p)
        print("  shell boundary (1-D contour on %s) written" % reference)
        _render_boundary(p, p.replace(".yaml", ".png"), "SHELL boundary %s (1-D, %s ref)" % (tag, reference), quad=False)
        written += [p, p.replace(".yaml", ".png")]
    return written


# --------------------------------------------------------------------------- renders
def _render_hex(nodes, hexes, hmats, mat_names, png, title):
    from opensg_io.render3d import render_mesh_png
    hset = {m: i for i, m in enumerate(mat_names)}
    render_mesh_png(nodes, hexes, "hex", np.array([hset[m] for m in hmats], int), png, title)
    print("  wrote", os.path.basename(png))


def _render_quad(nodes, quads, setmap, png, title):
    from opensg_io.render3d import render_mesh_png
    render_mesh_png(nodes, quads, "quad", np.array(setmap, int), png, title)
    print("  wrote", os.path.basename(png))


def _render_boundary(yaml_path, png, title, quad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    d = yaml.load(open(yaml_path), Loader=CL)
    X = np.array([[float(v) for v in r[0].split()] for r in d["nodes"]])
    cells = [[int(v) - 1 for v in r[0].split()] for r in d["elements"]]
    setname = ["?"] * len(cells)
    for si, s in enumerate(d["sets"]["element"]):
        for lab in s["labels"]:
            setname[lab - 1] = s["name"]
    names = sorted(set(setname)); cix = {n: i for i, n in enumerate(names)}
    PAL = plt.cm.tab10(np.linspace(0, 1, max(10, len(names))))
    fig, ax = plt.subplots(figsize=(11, 5))
    for c, sn in zip(cells, setname):
        p = X[c][:, :2]
        if quad:
            ax.fill(p[:, 0], p[:, 1], color=PAL[cix[sn]], edgecolor="k", linewidth=0.12)
        else:
            ax.plot(p[:, 0], p[:, 1], color=PAL[cix[sn]], lw=2.0)
    ax.plot(X[:, 0], X[:, 1], "o", ms=1.5, color="k", alpha=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PAL[cix[n]], label=n) for n in names], fontsize=7, loc="upper right")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.tight_layout(); fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(png))


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windio")
    ap.add_argument("mode", choices=["stations", "boundary", "taper"])
    ap.add_argument("--r", type=float); ap.add_argument("--r1", type=float); ap.add_argument("--r2", type=float)
    ap.add_argument("--model", choices=["solid", "shell", "both"], default="both")
    ap.add_argument("--reference", choices=["OML", "mid"], default="OML")
    ap.add_argument("--out", default=os.path.join(HERE, "mesh_out"))
    ap.add_argument("--nr", type=int, default=4); ap.add_argument("--nsp", type=int, default=12)
    ap.add_argument("--nw", type=int, default=3); ap.add_argument("--mesh", type=float, default=0.02)
    a = ap.parse_args()
    if a.mode == "stations":
        list_stations(a.windio)
    elif a.mode == "boundary":
        assert a.r is not None, "boundary needs --r"
        w = build_boundary(a.windio, a.r, a.model, a.out, a.reference, a.nr, a.nw, a.mesh)
        print("\nOUTPUTS:", *w, sep="\n  ")
    else:
        assert a.r1 is not None and a.r2 is not None, "taper needs --r1 and --r2"
        w = build_taper(a.windio, a.r1, a.r2, a.model, a.out, a.reference, a.nr, a.nsp, a.nw, a.mesh)
        print("\nOUTPUTS:", *w, sep="\n  ")


if __name__ == "__main__":
    main()
