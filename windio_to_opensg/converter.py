"""windIO v2 blade  ->  OpenSG 1D-shell SG YAML  (+ PreVABS XML)  cross-section converter.

For a chosen span station r in [0,1] (non-dimensional blade length) this resolves the windIO v2
blade structure (outer_shape airfoil/chord/twist + structure anchors/layers/webs) into a 2D
cross-section and emits:
  - an OpenSG 1D-shell YAML (nodes, line elements, element sets = distinct laminates, sections, materials)
    consumed by strip_RM.rm_timoshenko_6x6 / gradient_junction_kirchhoff;
  - a PreVABS XML (airfoil .dat baseline + dividing points + webs + layups + materials) for the 2D-solid.

Conventions:
  nd_arc s in [0,1] runs TE(s=0) -> suction/upper -> LE(s~0.5) -> pressure/lower -> TE(s=1), matching the
  windIO airfoil coordinate ordering (coords start at TE upper, x:1->0->1).  Laminate at an arc segment =
  the stack of all layers covering it, ordered outer->inner by LAYER_ORDER.  Ply angle = fiber_orientation
  (deg, relative to the spanwise/beam axis e1, same as the OpenSG/PreVABS theta3).  windIO material
  E=[E1,E2,E3] G=[G12,G13,G23] nu=[nu12,nu13,nu23] maps 1:1 to the OpenSG material block (isotropic ->
  replicate the scalar).
"""
import os
import sys
import numpy as np
import windIO

# outer -> inner lamination order (windIO layer-name stems); region layers slot between the shells.
LAYER_ORDER = ["gelcoat", "shell_triax_outer", "te_reinforcement", "spar_cap", "le_reinforcement",
               "te_filler", "le_filler", "shell_triax_inner"]


def interp(spec, r):
    if isinstance(spec, dict) and "grid" in spec:
        return float(np.interp(r, spec["grid"], spec["values"]))
    if isinstance(spec, (int, float)):
        return float(spec)
    return None


def arc_param(xy):
    """Normalised cumulative arc length s in [0,1] along the (open) airfoil point list."""
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    return d / d[-1]


class WindIOBlade:
    def __init__(self, yaml_path):
        self.d = windIO.load_yaml(yaml_path)
        self.bl = self.d["components"]["blade"]
        self.osh = self.bl["outer_shape"]
        self.st = self.bl["structure"]
        self.mats = {m["name"]: m for m in self.d["materials"]}
        self.afs = {a["name"]: a for a in self.d["airfoils"]}
        self.anch = {}
        for a in self.st["anchors"]:
            self.anch[a["name"]] = a
        for w in self.st["webs"]:
            for a in w.get("anchors", []):
                self.anch[a["name"]] = a

    def scalar(self, name, r):
        return interp(self.osh[name], r)

    def resolve(self, spec, r):
        if spec is None:
            return None
        if "anchor" in spec:
            ref = spec["anchor"]
            return interp(self.anch[ref["name"]].get(ref["handle"]), r)
        return interp(spec, r)

    def airfoil_coords(self, r, n=400):
        """Blend the two bracketing airfoils (by spanwise position) at r; return closed (n,2) contour
        in chord-normalised coords (x:TE=1..LE=0..TE=1, y up/down)."""
        ent = sorted(self.osh["airfoils"], key=lambda a: a["spanwise_position"])
        pos = [a["spanwise_position"] for a in ent]
        i = int(np.clip(np.searchsorted(pos, r) - 1, 0, len(ent) - 2))
        a0, a1 = ent[i], ent[i + 1]
        w = 0.0 if a1["spanwise_position"] == a0["spanwise_position"] else \
            (r - a0["spanwise_position"]) / (a1["spanwise_position"] - a0["spanwise_position"])
        w = float(np.clip(w, 0, 1))

        def resample(afname):
            c = self.afs[afname]["coordinates"]
            xy = np.column_stack([c["x"], c["y"]]).astype(float)
            s = arc_param(xy); sn = np.linspace(0, 1, n)
            return np.column_stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])])
        p0 = resample(a0["name"]); p1 = resample(a1["name"])
        return (1 - w) * p0 + w * p1

    def layers_at(self, r, tol=1e-6):
        out = []
        for L in self.st["layers"]:
            t = interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            out.append(dict(name=L["name"], material=L["material"],
                            s=self.resolve(L.get("start_nd_arc"), r), e=self.resolve(L.get("end_nd_arc"), r),
                            t=t, fiber=interp(L.get("fiber_orientation"), r) or 0.0))
        return out

    def webs_at(self, r, tol=1e-6):
        webs = []
        for w in self.st["webs"]:
            s = self.resolve(w.get("start_nd_arc"), r); e = self.resolve(w.get("end_nd_arc"), r)
            if s is None or e is None:
                continue
            # web laminate = the layers whose name starts with this web's name (skin/filler/skin)
            lam = [L for L in self.layers_at(r, tol) if L["name"].startswith(w["name"])]
            webs.append(dict(name=w["name"], s=s, e=e, layers=lam))
        return webs


def _order_key(layer_name):
    for i, stem in enumerate(LAYER_ORDER):
        if layer_name.startswith(stem):
            return i
    return len(LAYER_ORDER)


def build_cross_section(blade, r, mesh_size=0.01):
    """Resolve the station to nodes(2D), line elements, per-element laminate id, distinct laminates,
    webs (as node chains), and the material set.  mesh_size in chord-normalised arc units."""
    chord = blade.scalar("chord", r)
    xy = blade.airfoil_coords(r) * chord                    # chord-scaled contour
    s_arc = arc_param(xy)

    skin_layers = [L for L in blade.layers_at(r) if not L["name"].startswith("web")]
    webs = blade.webs_at(r)

    # segment breakpoints = all skin-layer start/end + web attachment arc positions
    brks = {0.0, 1.0}
    for L in skin_layers:
        for v in (L["s"], L["e"]):
            if v is not None:
                brks.add(float(np.clip(v, 0, 1)))
    for w in webs:
        brks.add(float(np.clip(w["s"], 0, 1))); brks.add(float(np.clip(w["e"], 0, 1)))
    brks = np.array(sorted(brks))

    def laminate_at(smid):
        cov = [L for L in skin_layers if (L["s"] is not None and L["e"] is not None
                                          and min(L["s"], L["e"]) - 1e-9 <= smid <= max(L["s"], L["e"]) + 1e-9)]
        cov.sort(key=lambda L: _order_key(L["name"]))
        return tuple((L["material"], round(L["t"], 8), round(L["fiber"], 3)) for L in cov)

    def pt(starc):                                          # (x,y) at nd_arc s
        return np.array([np.interp(starc, s_arc, xy[:, 0]), np.interp(starc, s_arc, xy[:, 1])])

    # build nodes per inter-breakpoint segment at ~mesh_size arc spacing; assign laminate per element
    perim = float(np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))][-1])
    nodes = []
    node_arc = []
    elems = []
    elem_lam = []
    laminates = {}                                          # laminate tuple -> set index
    segments = []                                           # ordered skin segments (s_a, s_b, set_id) for the XML

    def add_node(starc):
        nodes.append(pt(starc)); node_arc.append(starc); return len(nodes) - 1

    prev = None
    for bi in range(len(brks) - 1):
        a, b = brks[bi], brks[bi + 1]
        seg_len = (b - a) * perim
        nsub = max(1, int(round(seg_len / (mesh_size * chord))))
        ss = np.linspace(a, b, nsub + 1)
        smid = 0.5 * (a + b)
        lam = laminate_at(smid)
        if lam not in laminates:
            laminates[lam] = len(laminates)
        lid = laminates[lam]
        segments.append(dict(s_a=float(a), s_b=float(b), set_id=lid))
        if prev is None:
            prev = add_node(ss[0])
        for sN in ss[1:]:
            cur = add_node(sN)
            elems.append((prev, cur)); elem_lam.append(lid); prev = cur
    # close the loop (last node back to first); drop the duplicate TE node if coincident
    nodes = np.array(nodes)
    if np.linalg.norm(nodes[-1] - nodes[0]) < 1e-9 * chord:
        elems[-1] = (elems[-1][0], 0); nodes = nodes[:-1]; node_arc = node_arc[:-1]
    else:
        elems.append((len(nodes) - 1, 0)); elem_lam.append(elem_lam[-1])

    # webs: connect the two attachment nodes (nearest existing node to each web arc position)
    node_arc = np.array(node_arc)
    web_chains = []
    for w in webs:
        if not w["layers"]:
            continue
        na = int(np.argmin(np.abs(node_arc - w["s"]))); nb = int(np.argmin(np.abs(node_arc - w["e"])))
        lam = tuple((L["material"], round(L["t"], 8), round(L["fiber"], 3))   # windIO order = skin/foam/skin
                    for L in w["layers"])
        if lam not in laminates:
            laminates[lam] = len(laminates)
        web_chains.append(dict(a=na, b=nb, lam=laminates[lam], name=w["name"], s=float(w["s"]), e=float(w["e"])))

    return dict(r=r, chord=chord, twist=blade.scalar("twist", r), nodes=nodes, elems=elems,
                elem_lam=elem_lam, laminates=laminates, webs=web_chains, blade=blade,
                segments=segments, xy=xy, s_arc=s_arc, perim=perim)


# ---------------------------------------------------------------------------------------------------
#  Emit OpenSG 1D-shell YAML
# ---------------------------------------------------------------------------------------------------
import yaml as _yaml


class _Flow(list):
    pass


_yaml.add_representer(_Flow, lambda d, data: d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


def _mat_block(blade, name):
    m = blade.mats[name]
    E, G, nu = m["E"], m["G"], m["nu"]
    if not isinstance(E, (list, tuple)):                    # isotropic -> replicate
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
    return dict(name=name, density=float(m.get("rho", 1.0)),
                elastic=dict(E=[float(x) for x in E], G=[float(x) for x in G], nu=[float(x) for x in nu]))


def emit_opensg_yaml(cs, out_path, web_mesh=None):
    """Write the OpenSG 1D-shell SG YAML. Adds web node-chains; e1=+z, e2=tangent, e3 inward (skin) /
    e1xe2 (web)."""
    blade = cs["blade"]; chord = cs["chord"]
    nodes = [np.asarray(p, float) for p in cs["nodes"]]
    elems = list(cs["elems"]); elem_lam = list(cs["elem_lam"])
    web_mesh = web_mesh if web_mesh else 0.01 * chord
    set_of_lam = {v: k for k, v in cs["laminates"].items()}   # set index -> laminate tuple

    # build web node chains.  Order each web bottom(-y3) -> top(+y3) so e2 = +y3 matches the PreVABS web
    # baseline (<angle>90</angle> = +y3); then e3 = e1 x e2 = -y2 agrees with the 2D-solid web frame.
    for w in cs["webs"]:
        a, b = w["a"], w["b"]
        if nodes[a][1] > nodes[b][1]:                        # lower-y node first (bottom -> top)
            a, b = b, a
        Pa, Pb = nodes[a].copy(), nodes[b].copy()
        nseg = max(2, int(round(np.linalg.norm(Pb - Pa) / web_mesh)))
        ts = np.linspace(0, 1, nseg + 1)
        chain = [a]
        for t in ts[1:-1]:
            nodes.append(Pa + t * (Pb - Pa)); chain.append(len(nodes) - 1)
        chain.append(b)
        for ia, ib in zip(chain[:-1], chain[1:]):
            elems.append((ia, ib)); elem_lam.append(w["lam"])

    nodes = np.array(nodes)
    nsets = len(cs["laminates"])
    web_sets = {w["lam"] for w in cs["webs"]}
    # consistent skin inward normal (OML->IML) from the CLOSED-LOOP ORIENTATION. Robust at the TE, where the
    # old "point e3 toward the centroid" heuristic was unstable: there the skins are ~horizontal so the local
    # normal is ~vertical, while (C - element) is ~horizontal -> dot ~ 0 -> spurious e3 flips near the TE.
    skin_xy = np.asarray(cs["nodes"])
    area2 = float(np.sum(skin_xy[:, 0] * np.roll(skin_xy[:, 1], -1) - np.roll(skin_xy[:, 0], -1) * skin_xy[:, 1]))
    inward = 1.0 if area2 > 0 else -1.0            # e3=[-e2y,e2x] is inward for a CCW loop; flip for CW

    seg = {"nodes": [], "elements": [], "sets": {"element": []}, "sections": [],
           "elementOrientations": [], "materials": []}
    for (X, Y) in nodes:
        seg["nodes"].append(_Flow(["%.8f %.8f %.8f" % (X, Y, 0.0)]))
    for (n1, n2) in elems:
        seg["elements"].append(_Flow(["%d %d" % (n1 + 1, n2 + 1)]))
    for k in range(nsets):
        labels = [i + 1 for i, s in enumerate(elem_lam) if s == k]
        seg["sets"]["element"].append({"name": "layup_%d" % k, "labels": labels})
    for k in range(nsets):
        layup = [[mat, float(t), float(ang)] for (mat, t, ang) in set_of_lam[k]]
        seg["sections"].append({"type": "shell", "elementSet": "layup_%d" % k, "layup": layup})
    for ei, (n1, n2) in enumerate(elems):
        P1, P2 = nodes[n1], nodes[n2]
        e2 = (P2 - P1) / (np.linalg.norm(P2 - P1) + 1e-30)
        e3 = np.array([-e2[1], e2[0]])
        if elem_lam[ei] not in web_sets:                   # skin: consistent loop-orientation inward normal
            e3 = inward * e3
        seg["elementOrientations"].append(_Flow([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0,
                                                  float(e3[0]), float(e3[1]), 0.0]))
    used = []
    for k in range(nsets):
        for (mat, t, ang) in set_of_lam[k]:
            if mat not in used:
                used.append(mat)
    for name in used:
        seg["materials"].append(_mat_block(blade, name))
    with open(out_path, "w") as f:
        _yaml.dump(seg, f, sort_keys=False, default_flow_style=False)
    return dict(n_nodes=len(nodes), n_elems=len(elems), n_sets=nsets, n_webs=len(cs["webs"]),
                n_mats=len(used), out=out_path)


# ---------------------------------------------------------------------------------------------------
#  Emit PreVABS XML + materials.xml + airfoil .dat (for the 2D-solid mesh via prevabs.exe)
# ---------------------------------------------------------------------------------------------------
def _mat_xml(blade, name):
    m = blade.mats[name]
    E, G, nu, rho = m["E"], m["G"], m["nu"], float(m.get("rho", 1.0))
    if not isinstance(E, (list, tuple)):           # isotropic -> replicate (windIO G == E/2(1+nu), exact);
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]   # uniform orthotropic cards keep the .sg parser happy
    return ('  <material name="%s" type="orthotropic">\n    <density>%g</density>\n    <elastic>\n'
            '      <e1>%g</e1><e2>%g</e2><e3>%g</e3>\n      <g12>%g</g12><g13>%g</g13><g23>%g</g23>\n'
            '      <nu12>%g</nu12><nu13>%g</nu13><nu23>%g</nu23>\n    </elastic>\n  </material>\n'
            % (name, rho, E[0], E[1], E[2], G[0], G[1], G[2], nu[0], nu[1], nu[2]))


def emit_prevabs(cs, outdir, name="xsec", mesh_size=0.005):
    """Write {name}.dat (normalised airfoil), materials.xml (materials + laminae), and {name}.xml
    (PreVABS: general/baselines/dividing points/webs/layups/components) for prevabs.exe -i {name}.xml."""
    blade = cs["blade"]; chord = cs["chord"]
    os.makedirs(outdir, exist_ok=True)
    inv = {v: k for k, v in cs["laminates"].items()}
    web_sets = {w["lam"] for w in cs["webs"]}
    used = []
    for k in range(len(inv)):
        for (mat, t, a) in inv[k]:
            if mat not in used:
                used.append(mat)

    def ply_t(mat):
        return float(blade.mats[mat].get("ply_t", 0.001)) or 0.001

    # 1. airfoil .dat (normalised: PreVABS <scale> = chord applies the size)
    xyn = cs["xy"] / chord
    with open(os.path.join(outdir, name + ".dat"), "w") as f:
        f.write("%s\n" % name)
        for X, Y in xyn:
            f.write("% .8f % .8f\n" % (X, Y))

    # 2. materials.xml
    mx = "<materials>\n" + "".join(_mat_xml(blade, m) for m in used)
    for m in used:
        mx += ('  <lamina name="la_%s">\n    <material>%s</material>\n    <thickness>%g</thickness>\n  </lamina>\n'
               % (m, m, ply_t(m)))
    mx += "</materials>\n"
    open(os.path.join(outdir, "materials.xml"), "w").write(mx)

    # 3. layups (each laminate -> plies as lamina:count, angle from fiber orientation)
    def layup_xml(k):
        s = '    <layup name="layup_%d">\n' % k
        for (mat, t, a) in inv[k]:
            s += '      <layer lamina="la_%s">%g:%d</layer>\n' % (mat, a, max(1, int(round(t / ply_t(mat)))))
        return s + '    </layup>\n'

    # 4. skin dividing points (by normalised x + side) and baselines between consecutive breakpoints
    s_arc = cs["s_arc"]; xyc = cs["xy"]; segs = cs["segments"]

    def xn(s):
        return float(np.interp(s, s_arc, xyc[:, 0])) / chord

    def side(s):
        return "top" if s < 0.5 else "bottom"
    bks = sorted({round(float(v), 6) for seg in segs for v in (seg["s_a"], seg["s_b"]) if 1e-6 < v < 1 - 1e-6})
    pn = {s: "d%d" % i for i, s in enumerate(bks)}
    pts = "".join('    <point name="%s" on="ln_af" by="x2" which="%s">%.6f</point>\n' % (pn[s], side(s), xn(s))
                  for s in bks)
    # baselines+segments: interior segs use d(a):d(b); the two TE-adjacent segs merge into one wrap (sm:s1 thru TE)
    skin = [seg for seg in segs if seg["set_id"] not in web_sets]
    bls = ""; comps = ""; bi = 0
    te_first = skin[0]; te_last = skin[-1]
    for seg in skin[1:-1]:
        a, b = round(seg["s_a"], 6), round(seg["s_b"], 6)
        bls += '    <line name="bl_%d"><points>%s:%s</points></line>\n' % (bi, pn[a], pn[b])
        comps += ('    <segment name="sg_%d">\n      <baseline>bl_%d</baseline>\n      <layup>layup_%d</layup>\n    </segment>\n'
                  % (bi, bi, seg["set_id"]))
        bi += 1
    # TE wrap segment (last interior breakpoint -> first interior breakpoint, through TE)
    sm = round(te_last["s_a"], 6); s1 = round(te_first["s_b"], 6)
    bls += '    <line name="bl_te"><points>%s:%s</points></line>\n' % (pn[sm], pn[s1])
    comps += ('    <segment name="sg_te">\n      <baseline>bl_te</baseline>\n      <layup>layup_%d</layup>\n    </segment>\n'
              % te_first["set_id"])

    # 5. webs: point at normalised x + vertical line (angle 90); component per web
    web_xml = ""; web_comp = ""
    for wi, w in enumerate(cs["webs"]):
        wx = xn(w["s"])
        web_xml += ('    <point name="wp_%d">%.6f  0</point>\n'
                    '    <line name="bl_web_%d"><point>wp_%d</point><angle>90</angle></line>\n' % (wi, wx, wi, wi))
        web_comp += ('  <component name="web_%d" depend="surface">\n    <segment name="sg_web_%d">\n'
                     '      <baseline>bl_web_%d</baseline>\n      <layup>layup_%d</layup>\n    </segment>\n  </component>\n'
                     % (wi, wi, wi, w["lam"]))

    layups = "".join(layup_xml(k) for k in range(len(inv)))
    xml = ('<cross_section name="%s">\n  <include><material>materials</material></include>\n'
           '  <analysis><model>1</model></analysis>\n'
           '  <general>\n    <scale>%.6f</scale>\n    <mesh_size>%g</mesh_size>\n    <element_type>linear</element_type>\n  </general>\n'
           '  <baselines>\n    <line name="ln_af" type="airfoil"><points data="file" format="1" header="1">%s.dat</points></line>\n'
           '%s%s%s  </baselines>\n  <layups>\n%s  </layups>\n'
           '  <component name="surface">\n%s  </component>\n%s'
           '  <global><loads>1 2 3 4 5 6</loads></global>\n</cross_section>\n'
           % (name, chord, mesh_size, name, pts, bls, web_xml, layups, comps, web_comp))
    open(os.path.join(outdir, name + ".xml"), "w").write(xml)
    return dict(dat=name + ".dat", xml=name + ".xml", materials="materials.xml", n_layups=len(inv),
                n_webs=len(cs["webs"]), out=outdir)


if __name__ == "__main__":
    import windIO as _w
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(_w.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "iea22_r%03d.yaml" % round(r * 100))
    blade = WindIOBlade(src)
    cs = build_cross_section(blade, r, mesh_size=ms)
    info = emit_opensg_yaml(cs, out)
    print("station r=%.2f  chord=%.3f m  twist=%.2f deg" % (cs["r"], cs["chord"], cs["twist"]))
    print("laminates (element sets):")
    inv = {v: k for k, v in cs["laminates"].items()}
    for k in range(len(inv)):
        plies = " + ".join("%s@%g(%.4f)" % (m, a, t) for (m, t, a) in inv[k])
        print("  layup_%d: %s" % (k, plies[:140]))
    print("WROTE", info)

