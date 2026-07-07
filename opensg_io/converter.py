"""windIO blade  ->  OpenSG 1D-shell SG YAML  (+ PreVABS XML)  cross-section converter.

GENERAL and version-agnostic: reads any windIO file via load_blade() -- v2 (outer_shape / structure /
anchors, e.g. IEA-22-280-RWT) and v1 (outer_shape_bem / internal_structure_2d_fem, e.g. the NREL BAR
designs) share one downstream pipeline. No blade-specific assumptions: laminate stacking order = the
windIO layer order, materials accept scalar (isotropic) or 3-list (orthotropic) E/G/nu (G filled from
E,nu if absent). Per-blade specifics (source path, stations, mesh sizes, PreVABS exe) live in the
driver convert_blade.py, not here.

For a chosen span station r in [0,1] (non-dimensional blade length) this resolves the windIO
blade structure (airfoil/chord/twist + layers/webs) into a 2D cross-section and emits:
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
# windIO is imported lazily inside WindIOBlade (v2 only) so v1 blades (plain yaml) need no windIO package.


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
        import windIO                                       # v2 only -- v1 (WindIOBladeV1) needs no windIO
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
        for idx, L in enumerate(self.st["layers"]):
            t = interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            out.append(dict(name=L["name"], material=L["material"],
                            s=self.resolve(L.get("start_nd_arc"), r), e=self.resolve(L.get("end_nd_arc"), r),
                            t=t, fiber=interp(L.get("fiber_orientation"), r) or 0.0, order=idx))
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


class WindIOBladeV1(WindIOBlade):
    """windIO v1 reader (outer_shape_bem / internal_structure_2d_fem) exposing the SAME interface as
    WindIOBlade. v1 has no anchors -- layer/web start_nd_arc & end_nd_arc are direct grid/values; web layers
    are tagged by a `web:` key; airfoils are placed by outer_shape_bem.airfoil_position (grid + labels);
    laminate stacking order = the layer's index in the yaml list (already outer->inner)."""

    def __init__(self, yaml_path):
        import yaml as _y
        self.d = _y.safe_load(open(yaml_path))          # windIO.load_yaml validates the v2 schema -> use raw
        self.bl = self.d["components"]["blade"]
        self.osh = self.bl["outer_shape_bem"]
        self.ist = self.bl["internal_structure_2d_fem"]
        self.mats = {m["name"]: m for m in self.d["materials"]}
        self.afs = {a["name"]: a for a in self.d["airfoils"]}
        self._layers = self.ist["layers"]
        self._webs = self.ist["webs"]

    def scalar(self, name, r):
        return interp(self.osh[name], r)                # chord (m), twist (rad -- reporting only, no rotation)

    def resolve(self, spec, r):
        return interp(spec, r)                          # no anchors in v1

    def airfoil_coords(self, r, n=400):
        ap = self.osh["airfoil_position"]; grid = ap["grid"]; labels = ap["labels"]
        i = int(np.clip(np.searchsorted(grid, r) - 1, 0, len(grid) - 2))
        w = 0.0 if grid[i + 1] == grid[i] else (r - grid[i]) / (grid[i + 1] - grid[i])
        w = float(np.clip(w, 0, 1))

        def resample(afname):
            c = self.afs[afname]["coordinates"]
            xy = np.column_stack([c["x"], c["y"]]).astype(float)
            s = arc_param(xy); sn = np.linspace(0, 1, n)
            return np.column_stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])])
        return (1 - w) * resample(labels[i]) + w * resample(labels[i + 1])

    def layers_at(self, r, tol=1e-6):
        out = []
        for idx, L in enumerate(self._layers):
            if L.get("web"):                            # web layers -> webs_at
                continue
            t = interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            out.append(dict(name=L["name"], material=L["material"],
                            s=interp(L.get("start_nd_arc"), r), e=interp(L.get("end_nd_arc"), r),
                            t=t, fiber=interp(L.get("fiber_orientation"), r) or 0.0, order=idx))
        return out

    def webs_at(self, r, tol=1e-6):
        webs = []
        for w in self._webs:
            s = interp(w.get("start_nd_arc"), r); e = interp(w.get("end_nd_arc"), r)
            if s is None or e is None:
                continue
            lam = []
            for idx, L in enumerate(self._layers):      # web laminate = layers tagged web==this web (skin/core/skin)
                if L.get("web") == w["name"]:
                    t = interp(L.get("thickness"), r)
                    if t and t >= tol:
                        lam.append(dict(name=L["name"], material=L["material"], s=None, e=None,
                                        t=t, fiber=interp(L.get("fiber_orientation"), r) or 0.0, order=idx))
            webs.append(dict(name=w["name"], s=s, e=e, layers=lam))
        return webs


def load_blade(path):
    """Auto-detect windIO v1 (outer_shape_bem) vs v2 (outer_shape) and return the matching reader."""
    import yaml as _y
    bl = _y.safe_load(open(path))["components"]["blade"]
    return WindIOBladeV1(path) if "outer_shape_bem" in bl else WindIOBlade(path)


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
        cov.sort(key=lambda L: L["order"])              # outer -> inner = windIO layer order (general)
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
    E, G, nu = m.get("E"), m.get("G"), m.get("nu")
    if not isinstance(E, (list, tuple)):                    # isotropic -> replicate (G from E,nu if absent)
        nu = 0.3 if nu is None else nu
        G = E / (2.0 * (1.0 + nu)) if G is None else G
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
    return dict(name=name, density=float(m.get("rho", 1.0)),
                elastic=dict(E=[float(x) for x in E], G=[float(x) for x in G], nu=[float(x) for x in nu]))


def emit_opensg_yaml(cs, out_path, web_mesh=None, fraction=0.5):
    """Write the OpenSG 1D-shell SG YAML. Adds web node-chains; e1=+z, e2=tangent, e3 inward (skin) /
    e1xe2 (web).

    fraction: through-thickness position of the shell REFERENCE surface, measured from the OML
    (the airfoil curve) inward as a fraction of the local laminate thickness.  0.5 (default) puts
    the contour on the laminate mid-surface = the centre reference the RM/Kirchhoff shell models
    assume; 0.0 leaves it on the OML.  The offset is applied to the SKIN nodes before the webs are
    built (so webs stay attached to the offset skin); the 2-D solid (emit_prevabs) is unaffected --
    it keeps the OML and lays the ply inward, so both models remain centred on the same wall."""
    blade = cs["blade"]; chord = cs["chord"]
    nodes = [np.asarray(p, float) for p in cs["nodes"]]
    elems = list(cs["elems"]); elem_lam = list(cs["elem_lam"])
    web_mesh = web_mesh if web_mesh else 0.01 * chord
    set_of_lam = {v: k for k, v in cs["laminates"].items()}   # set index -> laminate tuple

    # ---- reference-surface offset: move the SKIN contour from the OML to the laminate reference
    # surface at through-thickness `fraction` (0.5 = mid-surface / centre reference).  For a thick
    # wall the OML is t/2 outside the mid-surface, which spuriously stiffens the shell; at 0.5 the
    # contour lands on the wall centre where the shell ABD is referenced.
    nskin = len(cs["nodes"])
    web_sets_pre = {w["lam"] for w in cs["webs"]}
    skin_xy0 = np.asarray(cs["nodes"])
    area2 = float(np.sum(skin_xy0[:, 0] * np.roll(skin_xy0[:, 1], -1)
                         - np.roll(skin_xy0[:, 0], -1) * skin_xy0[:, 1]))
    inward0 = 1.0 if area2 > 0 else -1.0                       # e3=[-e2y,e2x] inward for CCW loop
    if fraction:
        acc_n = np.zeros((nskin, 2)); acc_t = np.zeros(nskin); acc_c = np.zeros(nskin)
        for ei, (n1, n2) in enumerate(elems):
            if elem_lam[ei] in web_sets_pre:
                continue                                      # skin elements only
            e2 = nodes[n2] - nodes[n1]; e2 = e2 / (np.linalg.norm(e2) + 1e-30)
            nin = inward0 * np.array([-e2[1], e2[0]])
            tlam = float(sum(t for (_m, t, _a) in set_of_lam[elem_lam[ei]]))
            for nd in (n1, n2):
                if nd < nskin:
                    acc_n[nd] += nin; acc_t[nd] += tlam; acc_c[nd] += 1.0
        offs = [np.asarray(p, float) for p in nodes]
        for i in range(nskin):
            if acc_c[i] > 0 and np.linalg.norm(acc_n[i]) > 1e-12:
                nrm = acc_n[i] / np.linalg.norm(acc_n[i])
                offs[i] = nodes[i] + fraction * (acc_t[i] / acc_c[i]) * nrm
        nodes = offs

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
    E, G, nu, rho = m.get("E"), m.get("G"), m.get("nu"), float(m.get("rho", 1.0))
    if not isinstance(E, (list, tuple)):           # isotropic -> replicate (G == E/2(1+nu) if absent);
        nu = 0.3 if nu is None else nu             # uniform orthotropic cards keep the .sg parser happy
        G = E / (2.0 * (1.0 + nu)) if G is None else G
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
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
        pt = blade.mats[mat].get("ply_t")
        return float(pt) if pt else 0.001                   # foam/core layers have no manufacturing ply_t

    # per-(material, thickness) lamina with a CAPPED through-thickness ply count. A thick isotropic core
    # (e.g. 70 mm foam) divided into 1 mm "plies" makes 70 high-aspect solid layers that crash the PreVABS
    # mesher; cap at MAX_PLIES and set lamina thickness = t/n so the layer thickness stays exact.
    MAX_PLIES = 8
    lam_reg = {}                                            # (mat, round(t,8)) -> (lamina_name, n_plies, lam_thk)
    for k in range(len(inv)):
        for (mat, t, a) in inv[k]:
            key = (mat, round(t, 8))
            if key not in lam_reg:
                n = min(max(1, int(round(t / ply_t(mat)))), MAX_PLIES)
                lam_reg[key] = ("la_%s_%d" % (mat, len(lam_reg)), n, t / n)

    # 1. airfoil .dat (normalised: PreVABS <scale> = chord applies the size)
    xyn = cs["xy"] / chord
    with open(os.path.join(outdir, name + ".dat"), "w") as f:
        f.write("%s\n" % name)
        for X, Y in xyn:
            f.write("% .8f % .8f\n" % (X, Y))

    # 2. materials.xml (material cards + one lamina per distinct layer thickness)
    mx = "<materials>\n" + "".join(_mat_xml(blade, m) for m in used)
    for (mat, _tr), (lname, _n, lthk) in lam_reg.items():
        mx += ('  <lamina name="%s">\n    <material>%s</material>\n    <thickness>%g</thickness>\n  </lamina>\n'
               % (lname, mat, lthk))
    mx += "</materials>\n"
    open(os.path.join(outdir, "materials.xml"), "w").write(mx)

    # 3. layups (each laminate -> plies as lamina:count, angle from fiber orientation)
    def layup_xml(k):
        s = '    <layup name="layup_%d">\n' % k
        for (mat, t, a) in inv[k]:
            lname, n, _lthk = lam_reg[(mat, round(t, 8))]
            s += '      <layer lamina="%s">%g:%d</layer>\n' % (lname, a, n)
        return s + '    </layup>\n'

    # 4. skin dividing points (by normalised x + side) and baselines between consecutive breakpoints
    s_arc = cs["s_arc"]; xyc = cs["xy"]; segs = cs["segments"]

    LE_XMIN = 0.02      # keep skin dividing points off the degenerate LE nose: PreVABS by-x placement crashes
    # at x~0 (e.g. BAR LE_reinf straddles the nose at x~0.001). Only shifts the solid-mesh LE-reinf boundary a
    # fraction of a percent chord; the 1D shell uses exact nodes and is unaffected. (IEA-22 LE points >0.02.)
    def xn(s):
        x = float(np.interp(s, s_arc, xyc[:, 0])) / chord
        return x if x >= LE_XMIN else LE_XMIN

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

    # 5. webs: a point on the web line (midpoint of the two attachments) + the ACTUAL web angle from those
    #    attachments (NOT hardcoded 90: the windIO webs are tilted up to ~9deg at the root, so a vertical
    #    PreVABS web would not match the 1D shell, which connects the true suction<->pressure points).
    web_xml = ""; web_comp = ""
    for wi, w in enumerate(cs["webs"]):
        Ps = np.asarray(cs["nodes"][w["a"]], float); Pe = np.asarray(cs["nodes"][w["b"]], float)
        if Ps[1] < Pe[1]:
            Ps, Pe = Pe, Ps                                # Ps = upper (suction) attachment
        ang = float(np.degrees(np.arctan2(Ps[1] - Pe[1], Ps[0] - Pe[0])))   # web-line angle (90 = vertical)
        M = 0.5 * (Ps + Pe) / chord                        # midpoint on the web line, normalised
        web_xml += ('    <point name="wp_%d">%.6f %.6f</point>\n'
                    '    <line name="bl_web_%d"><point>wp_%d</point><angle>%.4f</angle></line>\n'
                    % (wi, M[0], M[1], wi, wi, ang))
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
    # single-station demo for ANY windIO file:  python opensg_io.py <blade.yaml> [r] [mesh_size] [out.yaml]
    if len(sys.argv) < 2:
        sys.exit("usage: python opensg_io.py <windio_blade.yaml> [r=0.5] [mesh_size=0.01] [out.yaml]")
    src = sys.argv[1]
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    stem = os.path.splitext(os.path.basename(src))[0]
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "%s_r%03d.yaml" % (stem, round(r * 100)))
    blade = load_blade(src)                              # auto-detects windIO v1 / v2
    cs = build_cross_section(blade, r, mesh_size=ms)
    info = emit_opensg_yaml(cs, out)
    print("%s  station r=%.2f  chord=%.3f m  twist=%.4f (rad)" % (type(blade).__name__, cs["r"], cs["chord"], cs["twist"]))
    print("laminates (element sets):")
    inv = {v: k for k, v in cs["laminates"].items()}
    for k in range(len(inv)):
        plies = " + ".join("%s@%g(%.4f)" % (m, a, t) for (m, t, a) in inv[k])
        print("  layup_%d: %s" % (k, plies[:140]))
    print("WROTE", info)

