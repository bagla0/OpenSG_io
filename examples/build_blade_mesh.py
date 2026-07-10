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
import glob
import os
import platform
import subprocess
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


def yaml_to_msh(yaml_path, msh_path=None):
    """Write a gmsh 2.2 .msh next to any OpenSG SG YAML (hex/quad/tri/line; numeric-0-based
    or string-1-based node/element formats both handled).  Physical/geometric tag = the
    element's set index, so the material/layup regions survive into the .msh."""
    msh_path = msh_path or yaml_path.replace(".yaml", ".msh")
    CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    d = yaml.load(open(yaml_path), Loader=CL)
    rows = d["nodes"]
    if isinstance(rows[0][0], str):                        # "x y z" / 1-based
        nodes = [[float(v) for v in r[0].split()] for r in rows]
        cells = [[int(v) - 1 for v in r[0].split()] for r in d["elements"]]
    else:                                                  # [x,y,z] / 0-based
        nodes = [list(map(float, r)) for r in rows]
        cells = [list(map(int, e)) for e in d["elements"]]
    n0 = len(cells[0])
    if n0 == 4:                                            # 4 nodes: tet (3-D volume) or quad (planar)
        q = np.array(nodes)[cells[0]]
        vol = abs(float(np.dot(np.cross(q[1] - q[0], q[2] - q[0]), q[3] - q[0])))
        etype = 4 if vol > 1e-12 else 3                    # gmsh 4 = tet, 3 = quad
    else:
        etype = {8: 5, 3: 2, 2: 1}[n0]                     # hex / tri / line -> gmsh type
    tag = [1] * len(cells)
    for si, s in enumerate(d.get("sets", {}).get("element", [])):
        for lab in s["labels"]:
            tag[lab - 1 if isinstance(rows[0][0], str) else lab] = si + 1
    with open(msh_path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n" % len(nodes))
        for i, p in enumerate(nodes):
            f.write("%d %.9g %.9g %.9g\n" % (i + 1, p[0], p[1], p[2]))
        f.write("$EndNodes\n$Elements\n%d\n" % len(cells))
        for j, c in enumerate(cells):
            f.write("%d %d 2 %d %d %s\n" % (j + 1, etype, tag[j], tag[j],
                                            " ".join(str(int(n) + 1) for n in c)))
        f.write("$EndElements\n")
    return msh_path


# --------------------------------------------------------------------------- stations
def windio_stations(windio):
    """The blade's OWN spanwise stations, straight from the windIO
    outer_shape.airfoils list -- (r, airfoil_name) sorted by r.  These are the ONLY
    stations the mesher uses: no interpolated r that the windIO does not define."""
    raw = yaml.safe_load(open(windio))
    try:
        afs = raw["components"]["blade"]["outer_shape"]["airfoils"]
    except KeyError:
        afs = raw["components"]["blade"]["outer_shape_bem"]["airfoils"]
    return sorted((float(a["spanwise_position"]), str(a["name"])) for a in afs)


def prev_station(windio, r):
    """The windIO station immediately toward the root of r (for the default taper span)."""
    rs = [s[0] for s in windio_stations(windio)]
    below = [x for x in rs if x < r - 1e-6]
    return below[-1] if below else rs[0]


def list_stations(windio):
    blade = load_blade(windio)
    stations = windio_stations(windio)
    print("windIO: %s" % windio)
    print("blade-defined spanwise stations (use ONLY these r):")
    print("  %-8s %-22s %-9s %-6s %-7s %s" % ("r", "airfoil", "chord[m]", "webs", "z[m]", "mesh"))
    valid = []
    for r, name in stations:
        try:
            cs = build_cross_section(blade, r, mesh_size=0.03)
            webs = len(cs["webs"])
            ok = "yes" if webs >= 1 and "circular" not in name.lower() else "no (circular/root)"
            print("  %-8.4f %-22s %-9.3f %-6d %-7.2f %s" % (r, name, cs["chord"], webs, blade_z(blade, r), ok))
            if ok == "yes":
                valid.append(round(r, 4))
        except Exception as e:
            print("  %-8.4f %-22s %-9s %-6s %-7s no airfoil (%s)" % (r, name, "-", "-", "-", str(e)[:24]))
    if valid:
        print("\nmeshable airfoil stations: r = %s" % ", ".join("%.4f" % v for v in valid))
        print("boundary: pick ONE r.  taper: pick a station r (spans the previous station->r), "
              "or give an explicit windIO station pair r1,r2.")
    return valid


def write_blade_dat(windio, out):
    """Whenever a windIO blade is read, dump the MAIN BLADE DATA as a .dat table so the user
    can see the blade at a glance: one row per windIO station -- r (nondim span, ROOT 0 -> TIP 1),
    airfoil name, chord, twist, #shear-webs, span z, and whether it is meshable -- plus a header
    with the blade span length and the material list.  Returns the .dat path."""
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    stations = windio_stations(windio)
    stem = os.path.splitext(os.path.basename(windio))[0]
    path = os.path.join(out, "%s_blade.dat" % stem)
    try:
        length = float(blade.osh["reference_axis"]["z"]["values"][-1])
    except Exception:
        length = blade_z(blade, 1.0)
    rows = []
    for r, name in stations:
        z = blade_z(blade, r)
        try:
            cs = build_cross_section(blade, r, mesh_size=0.05)
            nweb = len(cs["webs"])
            tw = float(cs.get("twist", 0.0) or 0.0)          # build_cross_section stores twist in DEGREES
            meshable = "yes" if (nweb >= 1 and "circular" not in name.lower()) else "no"
            rows.append((r, name, cs["chord"], tw, nweb, z, meshable))
        except Exception:
            rows.append((r, name, float("nan"), float("nan"), 0, z, "no"))
    try:
        mats = ", ".join(sorted(blade.mats.keys()))
    except Exception:
        mats = "?"
    with open(path, "w") as f:
        f.write("# OpenSG_io -- blade summary (auto-written whenever a windIO file is read)\n")
        f.write("# windIO source     : %s\n" % os.path.basename(windio))
        f.write("# blade span length : %.4f m\n" % length)
        f.write("# windIO stations   : %d   (r = non-dimensional span, ROOT r=0 -> TIP r=1)\n" % len(rows))
        f.write("# materials         : %s\n" % mats)
        f.write("# meshable          : yes = webbed airfoil section (SG mesh available); "
                "no = circular root / no webs\n#\n")
        f.write("# %-8s %-22s %-10s %-11s %-5s %-10s %s\n"
                % ("r", "airfoil", "chord[m]", "twist[deg]", "webs", "z[m]", "meshable"))
        for (r, name, chord, tw, nweb, z, meshable) in rows:
            cs_ = "%-10.4f" % chord if chord == chord else "%-10s" % "-"     # nan-safe
            tw_ = "%-11.3f" % tw if tw == tw else "%-11s" % "-"
            f.write("%-10.4f %-22s %s %s %-5d %-10.3f %s\n" % (r, name, cs_, tw_, nweb, z, meshable))
    print("blade summary .dat -> %s" % path)
    return path


def _mat_cards(blade, mat_names):
    return [{"name": m, **{k: _mat_block(blade, m)["elastic"][k] for k in ("E", "G", "nu")},
             "rho": _mat_block(blade, m)["density"]} for m in mat_names]


def _mat_sets(hmats, mat_names):
    return {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                        for m in mat_names]}


# --------------------------------------------------------------------------- taper
def build_taper(windio, r1, r2, model, out, reference="OML", nr=4, nsp=12, nw=3, mesh=0.02, element="hex"):
    """Tapered SG segment.  SOLID element type is the user's choice:
      element='hex' -- structured ply-conforming loft (most accurate; can INVERT the thin web
                       hexes on a steep/twisting taper -> honest gate refuses a broken mesh);
      element='tet' -- unstructured gmsh/TetGen fill (ROBUST: never inverts on any taper; user
                       accepts refinement).  The SHELL is always the quad mid-surface loft."""
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    cs1 = build_cross_section(blade, r1, mesh_size=mesh)
    cs2 = build_cross_section(blade, r2, mesh_size=mesh)
    z1, z2 = blade_z(blade, r1), blade_z(blade, r2)
    print("TAPER r=%.3f->%.3f  chord %.3f->%.3f  z=[%.2f, %.2f] m  model=%s ref=%s  solid-element=%s"
          % (r1, r2, cs1["chord"], cs2["chord"], z1, z2, model, reference, element))
    tag = "r%03d_%03d" % (round(r1 * 100), round(r2 * 100))
    written = []
    need_res = (model in ("shell", "both")) or (model in ("solid", "both") and element == "hex")
    res = hex_between_sections(cs1, cs2, z1, z2, nr=nr, nsp=nsp, nw=nw, mesh_size=mesh) if need_res else None

    if model in ("solid", "both") and element == "tet":
        from opensg_io.tapered_tet import windio_taper_tets
        nodes, tets, oris, hmats = windio_taper_tets(cs1, cs2, z1, z2, nr=nr, nw=nw, mesh_size=mesh)
        mat_names = sorted(set(hmats))
        p = os.path.join(out, "%s_solid_taper.yaml" % tag)
        export_solid_yaml(p, nodes, tets, "tet", oris, _mat_cards(blade, mat_names),
                          sets=_mat_sets(hmats, mat_names))
        print("  solid (TET): %d nodes / %d tets ; conforming, robust (no inversion possible)"
              % (len(nodes), len(tets)))
        _render_tet(nodes, tets, hmats, mat_names, p.replace(".yaml", ".png"),
                    "SOLID taper %s (TET, by material)" % tag)
        written += [p, yaml_to_msh(p), p.replace(".yaml", ".png")]

    if model in ("solid", "both") and element == "hex":
        nodes, hexes = res["nodes"], res["hexes"]
        msj, ninv = min_scaled_jacobian(nodes, hexes)
        if ninv:
            raise RuntimeError(
                "SOLID hex taper r=%.2f->%.2f has %d inverted hexes (min scaled Jacobian %.3f) -- the thin "
                "shear webs twist with the airfoil aerodynamic twist over this span.  Use --element tet for a "
                "robust (never-inverting) mesh, or try a NARROWER / less-twisted station pair." % (r1, r2, ninv, msj))
        assert_conforming(nodes, hexes, "hex")
        oris, hmats = solid_yaml_payload(res, cs1, cs2)
        mat_names = sorted(set(hmats))
        p = os.path.join(out, "%s_solid_taper.yaml" % tag)
        export_solid_yaml(p, nodes, hexes, "hex", oris, _mat_cards(blade, mat_names),
                          sets=_mat_sets(hmats, mat_names))
        print("  solid (HEX): %d nodes / %d hexes ; gates PASS (min SJ %.3f)" % (len(nodes), len(hexes), msj))
        _render_hex(nodes, hexes, hmats, mat_names, p.replace(".yaml", ".png"),
                    "SOLID taper %s (HEX, by material)" % tag)
        written += [p, yaml_to_msh(p), p.replace(".yaml", ".png")]

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
        written += [p, yaml_to_msh(p), p.replace(".yaml", ".png")]
    return written


# --------------------------------------------------------------------------- boundary
def build_boundary(windio, r, model, out, reference="OML", nr=4, nw=3, mesh=0.02, mesher="auto"):
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    cs = build_cross_section(blade, r, mesh_size=mesh)
    z = blade_z(blade, r)
    # SOLID boundary mesher: 'prevabs' = the established VABS cross-section mesher (native webs,
    # no loft/twist inversion) -- linux binary only; 'struct' = the portable structured section.
    # 'auto' picks PreVABS when its binary is present on a linux host, else the structured section.
    if mesher == "auto":
        mesher = "prevabs" if (platform.system() == "Linux" and _prevabs_binary()) else "struct"
    print("BOUNDARY r=%.3f  chord %.3f  z=%.2f m  model=%s ref=%s  solid-mesher=%s"
          % (r, cs["chord"], z, model, reference, mesher))
    # a boundary is one cross-section; realize the structured section as the L end of a thin
    # prismatic segment (only the 2-D end section is exported, throwaway 3-D hexes are not gated)
    res = hex_between_sections(cs, cs, z, z + 0.1, nr=nr, nsp=1, nw=nw, mesh_size=mesh)
    tag = "r%03d" % round(r * 100)
    written = []

    if model in ("solid", "both"):
        if mesher == "prevabs":
            p = _build_solid_boundary_prevabs(cs, blade, out, tag, mesh)
        else:
            p = os.path.join(out, "%s_solid_boundary.yaml" % tag)
            _dump(solid_boundary_payload(res, cs, cs, 0, blade, _mat_block), p)
            print("  solid boundary (2-D quad section) written")
            _render_boundary(p, p.replace(".yaml", ".png"),
                             "SOLID boundary %s (2-D quad, by material)" % tag, quad=True)
        written += [p, yaml_to_msh(p), p.replace(".yaml", ".png")]

    if model in ("shell", "both"):
        shell = shell_between_sections(res, cs, cs, reference=reference)
        p = os.path.join(out, "%s_shell_boundary.yaml" % tag)
        _dump(shell_boundary_payload(res, shell, cs, cs, 0, blade, _mat_block), p)
        print("  shell boundary (1-D contour on %s) written" % reference)
        _render_boundary(p, p.replace(".yaml", ".png"), "SHELL boundary %s (1-D, %s ref)" % (tag, reference), quad=False)
        written += [p, yaml_to_msh(p), p.replace(".yaml", ".png")]
    return written


# --------------------------------------------------------------------------- PreVABS boundary mesher
def _prevabs_binary():
    """Locate the PreVABS linux binary shipped under third_party/ (or None)."""
    pats = ["~/OpenSG_io/third_party/prevabs_bin/**/prevabs",
            "~/OpenSG_io/third_party/prevabs/**/prevabs",
            os.path.join(os.path.dirname(HERE), "third_party", "prevabs_bin", "**", "prevabs"),
            os.path.join(os.path.dirname(HERE), "third_party", "prevabs", "**", "prevabs")]
    for pat in pats:
        hits = sorted(glob.glob(os.path.expanduser(pat), recursive=True))
        hits = [h for h in hits if os.path.isfile(h) and not h.endswith((".so", ".dll"))]
        if hits:
            return hits[0]
    return None


def _build_solid_boundary_prevabs(cs, blade, out, tag, mesh):
    """windIO station -> PreVABS XML -> run PreVABS -> .sg -> 2-D solid YAML (FEniCS format).

    PreVABS is the established VABS cross-section mesher: it meshes the webbed airfoil natively,
    so there is no loft/twist hex inversion.  Needs the linux binary (run on the server)."""
    from opensg_io.converter import emit_prevabs
    if platform.system() != "Linux":
        raise RuntimeError("--mesher prevabs needs the linux PreVABS binary; run this on the server "
                           "(msg.ecn.purdue.edu, opensg_2_0 env). On Windows use --mesher struct.")
    pv = _prevabs_binary()
    if pv is None:
        raise RuntimeError("PreVABS binary not found under third_party/prevabs*; run scripts/fetch_prevabs.py.")
    work = os.path.join(out, "pv_%s" % tag)
    name = "%s_pv" % tag
    emit_prevabs(cs, work, name=name, mesh_size=mesh)
    env_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")   # active conda env lib
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(
        [os.path.dirname(pv), env_lib, os.environ.get("LD_LIBRARY_PATH", "")]))
    res = subprocess.run([pv, "-i", name + ".xml", "--vabs", "--hm"], cwd=work, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("PreVABS failed (rc=%d):\n%s" % (res.returncode, (res.stdout + res.stderr)[-1400:]))
    sg = os.path.join(work, name + ".sg")
    conv = os.path.join(os.path.dirname(HERE), "scripts", "convert_sg_to_yaml.py")
    p = os.path.join(out, "%s_solid_boundary.yaml" % tag)
    r2 = subprocess.run([sys.executable, conv, sg, p], capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError("convert_sg_to_yaml failed:\n%s" % (r2.stdout + r2.stderr)[-1200:])
    # gate: consistent winding (all element signed-areas same sign) => no tangled/inverted elements
    d = yaml.safe_load(open(p))
    P = np.array([[float(v) for v in row[0].split()] for row in d["nodes"]])[:, 1:]
    cells = [[int(v) - 1 for v in row[0].split()] for row in d["elements"]]
    area = [sum(P[e[k], 0] * P[e[(k + 1) % len(e)], 1] - P[e[(k + 1) % len(e)], 0] * P[e[k], 1]
                for k in range(len(e))) for e in cells]
    npos = sum(a > 0 for a in area); nneg = sum(a < 0 for a in area)
    if npos and nneg:
        raise RuntimeError("PreVABS mesh has MIXED-winding elements (%d +, %d -) -- tangled." % (npos, nneg))
    tri = sum(len(e) == 3 for e in cells); quad = sum(len(e) == 4 for e in cells)
    print("  solid boundary via PreVABS: %d nodes / %d elems (tri=%d quad=%d) ; winding consistent"
          % (len(P), len(cells), tri, quad))
    _render_boundary(p, p.replace(".yaml", ".png"),
                     "SOLID boundary %s (PreVABS 2-D mesh, by material)" % tag, quad=True)
    return p


# --------------------------------------------------------------------------- renders
def _render_hex(nodes, hexes, hmats, mat_names, png, title):
    from opensg_io.render3d import render_mesh_png
    hset = {m: i for i, m in enumerate(mat_names)}
    render_mesh_png(nodes, hexes, "hex", np.array([hset[m] for m in hmats], int), png, title)
    print("  wrote", os.path.basename(png))


def _render_tet(nodes, tets, hmats, mat_names, png, title):
    from opensg_io.render3d import render_mesh_png
    hset = {m: i for i, m in enumerate(mat_names)}
    render_mesh_png(nodes, tets, "tet", np.array([hset[m] for m in hmats], int), png, title)
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


# --------------------------------------------------------------------------- sweep
def build_sweep(windio, model, out, reference="OML", nr=4, nsp=12, nw=3, mesh=0.02, element="tet"):
    """Root -> tip sweep over ADJACENT windIO station pairs.  Each pair gets its OWN folder with
    the gated SG segment(s).  A pair that fails its gate is RECORDED and the sweep continues (one
    bad pair never aborts the run).  Writes a summary .dat listing every pair PASS/FAIL."""
    os.makedirs(out, exist_ok=True)
    blade = load_blade(windio)
    valid = []
    for r, name in windio_stations(windio):
        try:
            cs = build_cross_section(blade, r, mesh_size=0.05)
            if len(cs["webs"]) >= 1 and "circular" not in name.lower():
                valid.append(round(r, 4))
        except Exception:
            pass
    pairs = list(zip(valid[:-1], valid[1:]))
    print("SWEEP %s: %d adjacent windIO pairs, solid element=%s, model=%s"
          % (os.path.basename(windio), len(pairs), element, model))
    rows = []
    for r1, r2 in pairs:
        tag = "r%03d_%03d" % (round(r1 * 100), round(r2 * 100))
        seg = os.path.join(out, tag)
        try:
            build_taper(windio, r1, r2, model, seg, reference, nr, nsp, nw, mesh, element)
            rows.append((r1, r2, "PASS", ""))
            print("  [PASS] %s -> %s" % (tag, seg))
        except Exception as e:
            rows.append((r1, r2, "FAIL", "%s: %s" % (type(e).__name__, str(e)[:70])))
            print("  [FAIL] %s : %s" % (tag, str(e)[:90]))
    dat = os.path.join(out, "sweep_summary.dat")
    npass = sum(1 for _r1, _r2, s, _n in rows if s == "PASS")
    with open(dat, "w") as f:
        f.write("# OpenSG_io root->tip sweep : %s\n" % os.path.basename(windio))
        f.write("# solid element = %s ; model = %s ; %d/%d pairs PASS\n#\n"
                % (element, model, npass, len(rows)))
        f.write("# %-16s %-8s %s\n" % ("pair r1->r2", "status", "note"))
        for r1, r2, s, note in rows:
            f.write("%-18s %-8s %s\n" % ("%.4f->%.4f" % (r1, r2), s, note))
    print("\nsweep summary: %d/%d PASS -> %s" % (npass, len(rows), dat))
    return rows


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windio")
    ap.add_argument("mode", choices=["stations", "boundary", "taper", "sweep"])
    ap.add_argument("--r", type=float); ap.add_argument("--r1", type=float); ap.add_argument("--r2", type=float)
    ap.add_argument("--to-root", action="store_true",
                    help="taper: with --r R only, build the segment [R-step, R] (adjacent station toward root)")
    ap.add_argument("--model", choices=["solid", "shell", "both"], default="both")
    ap.add_argument("--reference", choices=["OML", "mid"], default="OML")
    ap.add_argument("--out", default=os.path.join(HERE, "mesh_out"))
    ap.add_argument("--nr", type=int, default=4); ap.add_argument("--nsp", type=int, default=12)
    ap.add_argument("--nw", type=int, default=3); ap.add_argument("--mesh", type=float, default=0.02)
    ap.add_argument("--mesher", choices=["prevabs", "auto", "struct"], default="prevabs",
                    help="solid-boundary mesher: prevabs (DEFAULT -- established VABS mesher, guarantees an "
                         "unbroken section; needs the linux binary/server), auto (prevabs if available else "
                         "struct), struct (portable structured fallback for Windows dev)")
    ap.add_argument("--element", choices=["hex", "tet"], default="tet",
                    help="taper SOLID element type: tet (DEFAULT -- unstructured TetGen fill, ROBUST: never "
                         "inverts on any taper; the hex loft inverts on ~12/13 IEA-22 adjacent pairs), hex "
                         "(structured ply-conforming loft, most accurate but only valid where it passes the "
                         "inversion gate -- use for mild/near-prismatic pairs)")
    a = ap.parse_args()
    # whenever a windIO file is read, always emit the main blade-data .dat
    write_blade_dat(a.windio, a.out)
    if a.mode == "stations":
        list_stations(a.windio)
    elif a.mode == "boundary":
        assert a.r is not None, "boundary needs --r"
        w = build_boundary(a.windio, a.r, a.model, a.out, a.reference, a.nr, a.nw, a.mesh, a.mesher)
        print("\nOUTPUTS:", *w, sep="\n  ")
    elif a.mode == "sweep":
        build_sweep(a.windio, a.model, a.out, a.reference, a.nr, a.nsp, a.nw, a.mesh, a.element)
    else:
        r1, r2 = a.r1, a.r2
        if r1 is None or r2 is None:                       # single windIO station: span from the one toward root
            assert a.r is not None, "taper needs --r1 --r2 (windIO stations), or a single --r (uses the adjacent windIO station toward root)"
            r2 = a.r
            r1 = prev_station(a.windio, a.r)
        w = build_taper(a.windio, r1, r2, a.model, a.out, a.reference, a.nr, a.nsp, a.nw, a.mesh, a.element)
        print("\nOUTPUTS:", *w, sep="\n  ")


if __name__ == "__main__":
    main()
