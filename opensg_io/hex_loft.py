"""opensg_io.hex_loft -- GENERAL two-station structured HEX loft for thin-walled blade
sections, with the matching mid-surface QUAD shell segment.

Given the cross-sections of a blade at two span stations (build_cross_section outputs, i.e.
"two boundary XMLs"), build ONE canonical hoop skeleton -- the union of layup-segment
breakpoints and web junction BANDS (each web occupies a band of its own thickness on the
skin, subdivided nw times = junction refinement) -- label-matched between the stations so
both get the IDENTICAL topology.  Each station is then realized as a structured quad
cross-section:

  * skin: hoop nodes on the OML, offset inward through nr through-thickness layers using
    the LOCAL laminate thickness (thickness steps between layup segments are honored);
  * webs: plates whose across-thickness columns attach to the nw+1 inner-skin nodes of the
    junction band (top/bottom rows ARE skin nodes -> watertight, conforming T-junction),
    with depth rows cosine-clustered so the mesh is REFINED at both web junctions.

The two stations are linearly lofted (loft_to_hex pattern) into conforming 8-node hexes;
every export runs the mandatory conformity gate.  A matching mid-surface QUAD shell segment
(same hoop skeleton, same span stations) is produced for equivalent shell-vs-solid studies.
"""
import math
import numpy as np

from .conformity import assert_conforming
from .section_offset import offset_rings, open_thin_gaps


# --------------------------------------------------------------------------- skeleton
def _lam_tuple(cs, set_id):
    for lam, sid in cs["laminates"].items():
        if sid == set_id:
            return lam
    raise KeyError(set_id)


def _thick(lam):
    return float(sum(t for (_m, t, _a) in lam))


def _station_breaks(cs):
    """Labeled breakpoints for ONE station: segment boundaries + web junction band edges.
    Web attach arc positions ARE segment breakpoints in build_cross_section, so any segment
    breakpoint strictly inside a band is subsumed by it (returned in `inside` for canonical
    dropping across stations)."""
    bands = []
    for w in cs["webs"]:
        u = 0.5 * _thick(_lam_tuple(cs, w["lam"])) / cs["perim"]
        for tag, sw in (("s", w["s"]), ("e", w["e"])):
            bands.append((sw - u, sw + u, "band:%s:%s" % (w["name"], tag)))
    segb = [(float(seg["s_a"]), "seg:%d" % k) for k, seg in enumerate(cs["segments"])]
    segb.append((float(cs["segments"][-1]["s_b"]), "seg:end"))
    inside = {lab for v, lab in segb
              if any(lo + 1e-12 < v < hi - 1e-12 for lo, hi, _n in bands)}
    brks = segb + [(lo, nm + ":lo") for lo, _hi, nm in bands] + [(hi, nm + ":hi") for _lo, hi, nm in bands]
    return brks, inside


def _set_at(cs, smid):
    for seg in cs["segments"]:
        if seg["s_a"] - 1e-12 <= smid <= seg["s_b"] + 1e-12:
            return seg["set_id"]
    return cs["segments"][-1]["set_id"]


def section_skeleton(cs_list, mesh_size=0.02, nw=2):
    """Canonical shared hoop skeleton across stations.

    Returns dict(breaks_by_station, counts, kinds) where counts[k] is the subdivision of
    interval k (same for every station), and kinds[k] = ('skin', set_id) or
    ('band', web_index, '
    s'|'e').  Raises if the stations are not topologically compatible."""
    raw = [_station_breaks(cs) for cs in cs_list]
    drop = set().union(*[ins for _b, ins in raw])          # canonical: drop at EVERY station
    BLB, BLL = [], None
    for brks, _ins in raw:
        kept = sorted([(v, lab) for v, lab in brks if lab not in drop], key=lambda t: t[0])
        vals = [v for v, _ in kept]; labs = [l for _, l in kept]
        if BLL is None:
            BLL = labs
        elif labs != BLL:
            raise ValueError("stations are not topology-compatible:\n%s\nvs\n%s" % (BLL, labs))
        BLB.append(vals)
    labels = BLL
    nint = len(labels) - 1
    counts, kinds = [], []
    for k in range(nint):
        lab, labn = labels[k], labels[k + 1]
        band = (lab.startswith("band:") and lab.endswith(":lo")
                and labn == lab[:-3] + ":hi")
        if band:
            wname, side = lab.split(":")[1], lab.split(":")[2]
            wi = [i for i, w in enumerate(cs_list[0]["webs"]) if w["name"] == wname][0]
            counts.append(nw)
            kinds.append(("band", wi, side))
        else:
            n = 1
            for cs, brk in zip(cs_list, BLB):
                seg_len = (brk[k + 1] - brk[k]) * cs["perim"]
                n = max(n, int(round(seg_len / (mesh_size * cs["chord"]))))
            counts.append(max(1, n))
            smid = 0.5 * (BLB[0][k] + BLB[0][k + 1])
            kinds.append(("skin", _set_at(cs_list[0], smid)))
    return dict(breaks=BLB, counts=counts, kinds=kinds, labels=labels)


# --------------------------------------------------------------------------- one station
def _contour_pt(cs, s):
    xy, sa = cs["xy"], cs["s_arc"]
    s = float(np.clip(s, 0.0, 1.0))
    return np.array([np.interp(s, sa, xy[:, 0]), np.interp(s, sa, xy[:, 1])])


def build_station(cs, skel, si, nr=4):
    """Realize the canonical skeleton at ONE station.

    Returns dict(hoop_s, oml, rings, fscale, tnode, NC) plus the
    section topology helpers shared by all stations (ids are identical by construction)."""
    breaks = skel["breaks"][si]
    counts, kinds = skel["counts"], skel["kinds"]
    hoop_s, hoop_kind = [], []
    for k in range(len(counts)):
        ss = np.linspace(breaks[k], breaks[k + 1], counts[k] + 1)[:-1]
        hoop_s += list(ss)
        hoop_kind += [k] * counts[k]
    hoop_s = np.array(hoop_s)
    NC = len(hoop_s)

    oml = np.array([_contour_pt(cs, s) for s in hoop_s])

    # local wall thickness per hoop node = mean of the two adjacent intervals' laminates
    # (NuMAD getSolidMesh averages the offset distance over the element sets sharing a
    # node, BladeDef.m:1692-1694 -- same device, prevents layer crossing at panel breaks)
    tint = []
    for k, kind in enumerate(kinds):
        sid = kind[1] if kind[0] == "skin" else _set_at(cs, 0.5 * (breaks[k] + breaks[k + 1]))
        tint.append(_thick(_lam_tuple(cs, sid)))
    tnode = np.array([0.5 * (tint[hoop_kind[i]] + tint[hoop_kind[i - 1]]) for i in range(NC)])

    # FULL-ACCURACY trailing edge (NuMAD expandBladeGeometryTEs device): first OPEN the
    # contour wherever the gap cannot host the full nominal laminate of both walls, so
    # ply thicknesses are preserved exactly at the TE (commercial benchmark behavior);
    # then build the through-thickness rings by the robust PreVABS-style miter offset
    # (signed-area orientation, bisector normals, thin-gap clamp as a mere backstop --
    # after the opening the clamp should not engage, fscale = 1 everywhere).
    oml_open, te_moved = open_thin_gaps(oml, tnode, nr=nr)
    rings, fscale = offset_rings(oml_open, tnode, nr)
    return dict(hoop_s=hoop_s, hoop_kind=np.array(hoop_kind), oml=oml_open, rings=rings,
                fscale=fscale, te_moved=te_moved, tnode=tnode, NC=NC)


def _band_cols(skel, kinds_index):
    """Hoop indices of the nw+1 columns of each web band, per web and side."""
    counts, kinds = skel["counts"], skel["kinds"]
    starts = np.cumsum([0] + counts)
    NC = starts[-1]
    bands = {}
    for k, kind in enumerate(kinds):
        if kind[0] == "band":
            cols = [(starts[k] + j) % NC for j in range(counts[k] + 1)]
            bands[(kind[1], kind[2])] = cols
    return bands


def build_section_mesh(cs_list, skel, nr=4, ny_target=None):
    """Build the shared 2-D quad topology + per-station node coordinates.

    Returns dict(faces2d, ftag (list of ('skin',set_id,layer) / ('web',wi,col)),
    stations=[(NP,3) coords ...] (z=0 plane), NP, shell (mid-surface line topology),
    NYs, bands)."""
    ns = len(cs_list)
    st = [build_station(cs, skel, i, nr) for i, cs in enumerate(cs_list)]
    NC = st[0]["NC"]
    bands = _band_cols(skel, 0)
    webs0 = cs_list[0]["webs"]

    def sid(i, l):
        return i * (nr + 1) + l
    NS = NC * (nr + 1)

    # web depth rows: NY per web, cosine-clustered (refined at BOTH junctions)
    NYs, wpair = [], []
    for wi, w in enumerate(webs0):
        top = bands[(wi, "s")]; bot = bands[(wi, "e")]
        # pair columns by physical proximity (arc directions may oppose across the section)
        t0 = st[0]["oml"][top[0]]; b0 = st[0]["oml"][bot[0]]; bN = st[0]["oml"][bot[-1]]
        bot = bot if np.linalg.norm(t0 - b0) <= np.linalg.norm(t0 - bN) else bot[::-1]
        depth = float(np.linalg.norm(st[0]["oml"][top[len(top) // 2]] - st[0]["oml"][bot[len(bot) // 2]]))
        htarget = ny_target or max(1e-9, (cs_list[0]["perim"] / NC))
        NYs.append(max(4, int(round(depth / htarget))))
        wpair.append((top, bot))

    WBASE = [NS]
    for wi, w in enumerate(webs0):
        WBASE.append(WBASE[-1] + (len(wpair[wi][0])) * (NYs[wi] - 1))
    NP = WBASE[-1]

    def wid(wi, j, m):
        return WBASE[wi] + j * (NYs[wi] - 1) + (m - 1)

    def wn(wi, j, m):
        top, bot = wpair[wi]
        if m == 0:
            return sid(top[j], nr)                     # inner-skin node (suction)
        if m == NYs[wi]:
            return sid(bot[j], nr)                     # inner-skin node (pressure)
        return wid(wi, j, m)

    # ---- station coordinates
    stations = []
    for s in st:
        P = np.zeros((NP, 3))
        for i in range(NC):
            for l in range(nr + 1):
                P[sid(i, l), :2] = s["rings"][l, i]
        for wi in range(len(webs0)):
            top, bot = wpair[wi]; NY = NYs[wi]
            for j in range(len(top)):
                pt = P[sid(top[j], nr), :2]; pb = P[sid(bot[j], nr), :2]
                for m in range(1, NY):
                    tau = 0.5 * (1 - math.cos(math.pi * m / NY))   # cosine: refined at junctions
                    P[wid(wi, j, m), :2] = (1 - tau) * pt + tau * pb
        stations.append(P)

    # ---- shared quad topology (+ per-face INWARD surface normal at station 0, for the
    # NuMAD-convention element frame; computed geometrically so it is independent of the
    # CCW re-winding below)
    from .orientation import skin_inward_normal_2d, web_plate_normal_2d
    r0 = st[0]["rings"]; P0 = stations[0]
    faces2d, ftag, fn2d, fregion = [], [], [], []
    hoop_kind = st[0]["hoop_kind"]
    for i in range(NC):
        ii = (i + 1) % NC
        kind = skel["kinds"][hoop_kind[i]]
        sid_lam = kind[1] if kind[0] == "skin" else _set_at(cs_list[0], float(st[0]["hoop_s"][i]) + 1e-9)
        for l in range(nr):
            faces2d.append([sid(i, l), sid(ii, l), sid(ii, l + 1), sid(i, l + 1)])
            ftag.append(("skin", sid_lam, l))
            fregion.append(("R", int(hoop_kind[i])))       # label-matched skeleton region
            fn2d.append(skin_inward_normal_2d(r0, l, i, ii))
    for wi in range(len(webs0)):
        top, _b = wpair[wi]; NY = NYs[wi]
        for j in range(len(top) - 1):
            for m in range(NY):
                faces2d.append([wn(wi, j, m), wn(wi, j + 1, m), wn(wi, j + 1, m + 1), wn(wi, j, m + 1)])
                ftag.append(("web", wi, j))
                fregion.append(("web", wi))
                fn2d.append(web_plate_normal_2d(P0[wn(wi, j, m)][:2], P0[wn(wi, j, m + 1)][:2]))
    # canonical CCW winding of every 2-D face (signed area > 0), so the +z extrusion
    # always yields right-handed (positive-Jacobian) hexes AND the per-face hoop/depth
    # tangent e2 has one consistent sense for skin and webs (NuMAD repairs the same
    # defect post-hoc with a det check + node swap, NuMesh3D.m:61-98 -- we orient at
    # the source instead)
    faces2d = np.array(faces2d, int)
    Q = stations[0][faces2d, :2]                           # (nf, 4, 2)
    area2 = np.zeros(len(faces2d))
    for a in range(4):
        b = (a + 1) % 4
        area2 += Q[:, a, 0] * Q[:, b, 1] - Q[:, b, 0] * Q[:, a, 1]
    faces2d[area2 < 0] = faces2d[area2 < 0][:, ::-1]
    return dict(faces2d=faces2d, ftag=ftag, fregion=fregion, fn2d=np.array(fn2d),
                stations=stations, NP=NP, NC=NC, nr=nr, NYs=NYs, wpair=wpair, st=st,
                bands=bands)


# --------------------------------------------------------------------------- hex loft
def hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=12, nw=2, mesh_size=0.02):
    """GENERAL two-station hex loft.  Returns dict(nodes, hexes, htag, sec, skel)."""
    skel = section_skeleton([cs1, cs2], mesh_size=mesh_size, nw=nw)
    sec = build_section_mesh([cs1, cs2], skel, nr=nr)
    P1, P2 = sec["stations"]
    NP = sec["NP"]
    nodes = np.zeros(((nsp + 1) * NP, 3))
    for s in range(nsp + 1):
        tau = s / nsp
        nodes[s * NP:(s + 1) * NP, :2] = (1 - tau) * P1[:, :2] + tau * P2[:, :2]
        nodes[s * NP:(s + 1) * NP, 2] = (1 - tau) * z1 + tau * z2
    q = sec["faces2d"]
    hexes = np.empty((nsp * len(q), 8), int)
    for s in range(nsp):
        hexes[s * len(q):(s + 1) * len(q)] = np.hstack([s * NP + q, (s + 1) * NP + q])
    htag = sec["ftag"] * nsp
    return dict(nodes=nodes, hexes=hexes, htag=htag, sec=sec, skel=skel,
                z1=z1, z2=z2, nsp=nsp)


# ------------------------------------------------------------------- equivalent shell
def _lam_round_key(lam):
    return tuple((m, round(float(t), 9), round(float(a), 4)) for (m, t, a) in lam)


def shell_between_sections(res, cs1, cs2=None):
    """EQUIVALENT mid-surface QUAD shell segment for a hex_between_sections result.

    Same canonical hoop skeleton and span stations as the solid, so shell-vs-solid
    comparisons are one-to-one.  Skin quads sit on the wall MID-surface; each web is a
    strip of mid-columns whose top/bottom rows are the inner-skin mid nodes (a branched
    T-junction shared by exactly 3 quads).

    The layup is the SPAN-INTERPOLATED laminate at each span bay (like NuMAD's per-
    (region, bay) stacks): the taper's stiffness change lives entirely in the shell
    layup thickness, so each span slice carries its own interpolated section.  Element
    frames use the NuMAD convention (e1=span root->tip, e3=inward normal, e2=e3 x e1).

    Returns dict(nodes, quads, qsec (per-quad section index), qweb, oris, sections
    (list of laminates), skin_tl, web_tl, region_of_quad, sec2d)."""
    from .orientation import element_frame
    from .section_offset import miter_normals, ensure_ccw
    cs2 = cs2 or cs1
    sec, skel = res["sec"], res["skel"]
    z1, z2, nsp = res["z1"], res["z2"], res["nsp"]
    P1, P2 = sec["stations"]
    NC, nr, NYs, wpair = sec["NC"], sec["nr"], sec["NYs"], sec["wpair"]
    tl_by_region = region_taper_laminates(cs1, cs2, skel)

    ids = {}
    for i in range(NC):
        ids[("s", i)] = len(ids)
    for wi, NY in enumerate(NYs):
        for m in range(1, NY):
            ids[("w", wi, m)] = len(ids)
    NPs = len(ids)

    def station_shell(P):
        X = np.zeros((NPs, 2))
        for i in range(NC):
            X[ids[("s", i)]] = 0.5 * (P[i * (nr + 1) + 0, :2] + P[i * (nr + 1) + nr, :2])
        for wi, NY in enumerate(NYs):
            top, bot = wpair[wi]; jmid = len(top) // 2
            pt = X[ids[("s", top[jmid])]]; pb = X[ids[("s", bot[jmid])]]
            for m in range(1, NY):
                tau = 0.5 * (1 - math.cos(math.pi * m / NY))
                X[ids[("w", wi, m)]] = (1 - tau) * pt + tau * pb
        return X

    def wnode(wi, m):
        top, bot = wpair[wi]; NY = NYs[wi]; jm = len(top) // 2
        if m == 0:
            return ids[("s", top[jm])]
        if m == NY:
            return ids[("s", bot[jm])]
        return ids[("w", wi, m)]

    S1, S2 = station_shell(P1), station_shell(P2)
    snodes = np.zeros(((nsp + 1) * NPs, 3))
    for s in range(nsp + 1):
        tau = s / nsp
        snodes[s * NPs:(s + 1) * NPs, :2] = (1 - tau) * S1 + tau * S2
        snodes[s * NPs:(s + 1) * NPs, 2] = (1 - tau) * z1 + tau * z2

    # inward mid-surface skin normals (fold-free miter, oriented by signed area)
    loop = np.array([S1[ids[("s", i)]] for i in range(NC)])
    loopc, flipped = ensure_ccw(loop)
    m_in, _s = miter_normals(loopc)
    skin_n = m_in[::-1] if flipped else m_in                # (NC,2) inward normal per skin node

    # per-hoop skin region key (label-matched skeleton region -> its TaperLaminate)
    hoop_kind = sec["st"][0]["hoop_kind"]
    skin_region = [("R", int(hoop_kind[i])) for i in range(NC)]

    sections, sec_index = [], {}                            # dedup identical laminates

    def sec_id_for(lam):
        key = _lam_round_key(lam)
        if key not in sec_index:
            sec_index[key] = len(sections)
            sections.append([(m, float(t), float(a)) for (m, t, a) in lam])
        return sec_index[key]

    quads, qsec, qweb, region_of_quad, oris = [], [], [], [], []
    for s in range(nsp):
        tau = (s + 0.5) / nsp
        for i in range(NC):
            ii = (i + 1) % NC
            q = [s * NPs + ids[("s", i)], s * NPs + ids[("s", ii)],
                 (s + 1) * NPs + ids[("s", ii)], (s + 1) * NPs + ids[("s", i)]]
            quads.append(q)
            lam = tl_by_region[skin_region[i]].at(tau)
            qsec.append(sec_id_for(lam)); qweb.append(False)
            region_of_quad.append(skin_region[i])
            n2d = 0.5 * (skin_n[i] + skin_n[ii])
            oris.append(element_frame(snodes[q[3]] - snodes[q[0]],
                                      np.array([n2d[0], n2d[1], 0.0]), 0.0))
        for wi, NY in enumerate(NYs):
            lam = tl_by_region[("web", wi)].at(tau)
            sid = sec_id_for(lam)
            for m in range(NY):
                q = [s * NPs + wnode(wi, m), s * NPs + wnode(wi, m + 1),
                     (s + 1) * NPs + wnode(wi, m + 1), (s + 1) * NPs + wnode(wi, m)]
                quads.append(q)
                qsec.append(sid); qweb.append(True)
                region_of_quad.append(("web", wi))
                d = snodes[s * NPs + wnode(wi, m + 1)][:2] - snodes[s * NPs + wnode(wi, m)][:2]
                n2d = np.array([-d[1], d[0]])
                oris.append(element_frame(snodes[q[3]] - snodes[q[0]],
                                          np.array([n2d[0], n2d[1], 0.0]), 0.0))
    quads = np.array(quads, int)

    skin_loop = [ids[("s", i)] for i in range(NC)]
    web_lines = [[wnode(wi, m) for m in range(NYs[wi] + 1)] for wi in range(len(NYs))]
    sec2d = dict(S=[S1, S2], skin_loop=skin_loop, web_lines=web_lines, NPs=NPs,
                 skin_region=skin_region, skin_n=skin_n, wnode=wnode, NYs=NYs, ids=ids)
    return dict(nodes=snodes, quads=quads, qsec=np.array(qsec), qweb=qweb,
                oris=np.array(oris), sections=sections, tl_by_region=tl_by_region,
                region_of_quad=region_of_quad, sec2d=sec2d)


def assert_shell_conforming(shell, n_webs, nsp):
    """Branched mid-surface conformity: no hanging nodes, junction edges shared by
    EXACTLY 3 quads (skin-left + skin-right + web), none by more; returns the junction
    edge count (== 2*n_webs*nsp)."""
    from collections import Counter
    snodes, quads = shell["nodes"], shell["quads"]
    used = np.zeros(len(snodes), bool); used[quads.ravel()] = True
    if not used.all():
        raise ValueError("hanging shell nodes")
    ec = Counter()
    for q in quads:
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            ec[tuple(sorted((int(q[a]), int(q[b]))))] += 1
    over = [e for e, c in ec.items() if c > 3]
    junc = [e for e, c in ec.items() if c == 3]
    if over:
        raise ValueError("shell edges shared by >3 quads: %d" % len(over))
    if len(junc) != 2 * n_webs * nsp:
        raise ValueError("junction edges %d != expected %d" % (len(junc), 2 * n_webs * nsp))
    return len(junc)


def shell_yaml_payload(shell, blade, mat_block):
    """OpenSG 3-D shell SEGMENT YAML dict (FEniCS ShellSegmentMesh format: numeric
    [x,y,z] nodes, numeric 0-BASED quad connectivity and set labels, `elastic`-nested
    materials).  Span-interpolated per-bay layups, one element set per unique section."""
    snodes, quads = shell["nodes"], shell["quads"]
    qsec, oris, sections = shell["qsec"], shell["oris"], shell["sections"]
    used = sorted(set(int(s) for s in qsec))
    mats = sorted({m for lam in sections for (m, _t, _a) in lam})
    return {"nodes": [[float(p[0]), float(p[1]), float(p[2])] for p in snodes],
            "elements": [[int(v) for v in q] for q in quads],           # 0-based
            "sets": {"element": [{"name": "layup_%d" % l,
                                  "labels": [k for k in range(len(qsec)) if qsec[k] == l]}
                                 for l in used]},
            "sections": [{"type": "shell", "elementSet": "layup_%d" % l,
                          "layup": [[m, float(t), float(a)] for (m, t, a) in sections[l]]}
                         for l in used],
            "elementOrientations": [[float(v) for v in o] for o in oris],
            "materials": [mat_block(blade, m) for m in mats]}


# ------------------------------------------------------------------- boundary meshes
def _end_layer(nlayer_nodes, si, nsp):
    """Node-index offset of the end cross-section: si=0 -> left (root, s=0),
    si=1 -> right (tip, s=nsp)."""
    return (0 if si == 0 else nsp) * nlayer_nodes


def solid_boundary_payload(res, cs1, cs2, si, blade, mat_block):
    """2-D solid cross-section (quad mesh) at end si -- FEniCS SolidBounMesh format
    (string "x y z" nodes, string 1-BASED quad connectivity, flat E/G/nu/rho materials).
    Beam axis is +z (out of plane); each quad carries the ply material at its
    (region, layer) of the span-interpolated laminate and a fiber frame about +z."""
    from .orientation import element_frame
    sec = res["sec"]; NP = sec["NP"]; nr = sec["nr"]; fn2d = sec["fn2d"]
    faces2d, ftag, fregion = sec["faces2d"], sec["ftag"], sec["fregion"]
    z_end = res["z1"] if si == 0 else res["z2"]
    tau = 0.0 if si == 0 else 1.0
    off = _end_layer(NP, si, res["nsp"])
    P = res["nodes"][off:off + NP]                          # (NP,3) end-station coords
    tl_by_region = region_taper_laminates(cs1, cs2, res["skel"])
    zhat = np.array([0.0, 0.0, 1.0])
    fmats, oris = [], []
    for k, (tag, f) in enumerate(zip(ftag, faces2d)):
        tl = tl_by_region[fregion[k]]
        if tag[0] == "skin":
            frac = (tag[2] + 0.5) / nr
        else:
            ncols = len(sec["wpair"][tag[1]][0]) - 1
            frac = (tag[2] + 0.5) / max(1, ncols)
        m, ang = tl.ply_of_depth(tau, frac)
        fmats.append(m)
        n_surf = np.array([fn2d[k][0], fn2d[k][1], 0.0])
        oris.append(element_frame(zhat, n_surf, ang))
    names = sorted(set(fmats))
    materials = []
    for m in names:
        b = mat_block(blade, m)
        materials.append({"name": m, "E": b["elastic"]["E"], "G": b["elastic"]["G"],
                          "nu": b["elastic"]["nu"], "rho": b["density"]})
    return {"nodes": [["%.9f %.9f %.9f" % (p[0], p[1], z_end)] for p in P],
            "elements": [[" ".join(str(int(v) + 1) for v in f)] for f in faces2d],
            "sets": {"element": [{"name": m,
                                  "labels": [k + 1 for k in range(len(fmats)) if fmats[k] == m]}
                                 for m in names]},
            "elementOrientations": [[float(v) for v in o] for o in oris],
            "materials": materials}


def shell_boundary_payload(res, shell, cs1, cs2, si, blade, mat_block):
    """1-D shell cross-section (line mesh) at end si -- FEniCS ShellBounMesh format
    (string "x y z" nodes, string 1-BASED 2-node line connectivity, `elastic`-nested
    materials, per-region interpolated layup).  Beam axis +z; geometric frame."""
    from .orientation import element_frame
    s2 = shell["sec2d"]; NPs = s2["NPs"]; skin_loop = s2["skin_loop"]
    skin_region, skin_n = s2["skin_region"], s2["skin_n"]
    tl_by_region = shell["tl_by_region"]
    z_end = res["z1"] if si == 0 else res["z2"]
    tau = 0.0 if si == 0 else 1.0
    off = _end_layer(NPs, si, res["nsp"])
    P = shell["nodes"][off:off + NPs]                       # (NPs,3) end section coords
    zhat = np.array([0.0, 0.0, 1.0])

    sections, sec_index = [], {}

    def sec_id_for(lam):
        key = _lam_round_key(lam)
        if key not in sec_index:
            sec_index[key] = len(sections)
            sections.append([(m, float(t), float(a)) for (m, t, a) in lam])
        return sec_index[key]

    lines, lsec, oris = [], [], []
    NC = len(skin_loop)
    for i in range(NC):
        a, b = skin_loop[i], skin_loop[(i + 1) % NC]
        lines.append([a, b])
        lsec.append(sec_id_for(tl_by_region[skin_region[i]].at(tau)))
        n2d = 0.5 * (skin_n[i] + skin_n[(i + 1) % NC])
        oris.append(element_frame(zhat, np.array([n2d[0], n2d[1], 0.0]), 0.0))
    wnode = s2["wnode"]; NYs = s2["NYs"]
    for wi, NY in enumerate(NYs):
        lam_id = sec_id_for(tl_by_region[("web", wi)].at(tau))
        for m in range(NY):
            a, b = wnode(wi, m), wnode(wi, m + 1)
            lines.append([a, b]); lsec.append(lam_id)
            d = P[b][:2] - P[a][:2]
            oris.append(element_frame(zhat, np.array([-d[1], d[0], 0.0]), 0.0))
    used = sorted(set(lsec))
    mats = sorted({m for lam in sections for (m, _t, _a) in lam})
    return {"nodes": [["%.9f %.9f %.9f" % (p[0], p[1], z_end)] for p in P],
            "elements": [[" ".join(str(v + 1) for v in ln)] for ln in lines],
            "sets": {"element": [{"name": "layup_%d" % l,
                                  "labels": [k + 1 for k in range(len(lsec)) if lsec[k] == l]}
                                 for l in used]},
            "sections": [{"type": "shell", "elementSet": "layup_%d" % l,
                          "layup": [[m, float(t), float(a)] for (m, t, a) in sections[l]]}
                         for l in used],
            "elementOrientations": [[float(v) for v in o] for o in oris],
            "materials": [mat_block(blade, m) for m in mats]}


def _ply_at(lam, frac):
    """Ply (material, angle) at through-thickness fraction frac in laminate lam (outer->inner)."""
    T = _thick(lam); d = frac * T; acc = 0.0
    for (m, t, a) in lam:
        acc += t
        if d <= acc + 1e-12:
            return m, a
    return lam[-1][0], lam[-1][2]


def region_taper_laminates(cs1, cs2, skel):
    """Per-region TaperLaminate spanning the two stations, keyed by region identity
    (rigorous span-interpolated layup with ply drops).

    Regions are matched by LABEL, not arc position: the canonical skeleton's breakpoints
    are label-consistent across stations, so skeleton region k occupies breaks[0][k..k+1]
    at the root and breaks[1][k..k+1] at the tip -- the SAME physical band even though a
    smaller tip chord shifts its normalized arc.  (Matching by raw arc position instead
    mislabels the spar cap as a panel and fabricates spurious ply drops.)

    Returns a dict {("R", region_k): TaperLaminate, ("web", wi): TaperLaminate}."""
    from .layup import TaperLaminate
    out = {}
    breaks, kinds = skel["breaks"], skel["kinds"]
    for k in range(len(kinds)):
        if kinds[k][0] == "web":
            continue
        lams = []
        for si, cs in ((0, cs1), (1, cs2)):
            amid = 0.5 * (breaks[si][k] + breaks[si][k + 1])
            lams.append(list(_lam_tuple(cs, _set_at(cs, amid))))
        out[("R", k)] = TaperLaminate.from_stations(lams[0], lams[1])
    for wi, w in enumerate(cs1["webs"]):
        lam1 = list(_lam_tuple(cs1, w["lam"]))
        w2 = cs2["webs"][wi] if wi < len(cs2["webs"]) else w
        lam2 = list(_lam_tuple(cs2, w2["lam"]))
        out[("web", wi)] = TaperLaminate.from_stations(lam1, lam2)
    return out


def solid_yaml_payload(res, cs1, cs2=None):
    """Per-hex material + NuMAD-convention fiber orientation for export_solid_yaml.

    Each hex is assigned the ply of the SPAN-INTERPOLATED laminate at its through-
    thickness depth (ply drops handled), and a right-handed (e1=span root->tip,
    e3=inward normal, e2=e3 x e1) frame with the ply fiber angle rotated about e3
    (opensg_io.orientation.element_frame)."""
    from .orientation import element_frame
    cs2 = cs2 or cs1
    nodes, hexes, htag = res["nodes"], res["hexes"], res["htag"]
    sec = res["sec"]; nr = sec["nr"]; fn2d = sec["fn2d"]; fregion = sec["fregion"]
    z1, z2 = res["z1"], res["z2"]
    nf = len(sec["faces2d"])
    tl_by_region = region_taper_laminates(cs1, cs2, res["skel"])
    oris = np.zeros((len(hexes), 9))
    mats = []
    for k, (tag, hx) in enumerate(zip(htag, hexes)):
        f = k % nf
        zc = float(nodes[hx].mean(0)[2])
        tau = 0.0 if abs(z2 - z1) < 1e-12 else np.clip((zc - z1) / (z2 - z1), 0.0, 1.0)
        tl = tl_by_region[fregion[f]]
        if tag[0] == "skin":
            frac = (tag[2] + 0.5) / nr
        else:
            ncols = len(sec["wpair"][tag[1]][0]) - 1
            frac = (tag[2] + 0.5) / max(1, ncols)
        m, ang = tl.ply_of_depth(tau, frac)
        span = nodes[hx[4]] - nodes[hx[0]]
        n_surf = np.array([fn2d[f][0], fn2d[f][1], 0.0])
        oris[k] = element_frame(span, n_surf, ang)
        mats.append(m)
    return oris, mats
