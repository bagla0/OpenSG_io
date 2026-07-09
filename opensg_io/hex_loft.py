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
from .section_offset import offset_rings


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

    # through-thickness rings by robust PreVABS-style offset: signed-area orientation,
    # miter (bisector) vertex normals with the Clipper2 limit, and a smoothed thin-gap
    # clamp so rings never cross the opposite wall at a pinched trailing edge
    rings, fscale = offset_rings(oml, tnode, nr)
    return dict(hoop_s=hoop_s, hoop_kind=np.array(hoop_kind), oml=oml, rings=rings,
                fscale=fscale, tnode=tnode, NC=NC)


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

    # ---- shared quad topology
    faces2d, ftag = [], []
    hoop_kind = st[0]["hoop_kind"]
    for i in range(NC):
        ii = (i + 1) % NC
        kind = skel["kinds"][hoop_kind[i]]
        sid_lam = kind[1] if kind[0] == "skin" else _set_at(cs_list[0], float(st[0]["hoop_s"][i]) + 1e-9)
        for l in range(nr):
            faces2d.append([sid(i, l), sid(ii, l), sid(ii, l + 1), sid(i, l + 1)])
            ftag.append(("skin", sid_lam, l))
    for wi in range(len(webs0)):
        top, _b = wpair[wi]; NY = NYs[wi]
        for j in range(len(top) - 1):
            for m in range(NY):
                faces2d.append([wn(wi, j, m), wn(wi, j + 1, m), wn(wi, j + 1, m + 1), wn(wi, j, m + 1)])
                ftag.append(("web", wi, j))
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
    return dict(faces2d=faces2d, ftag=ftag, stations=stations, NP=NP, NC=NC,
                nr=nr, NYs=NYs, wpair=wpair, st=st, bands=bands)


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
    return dict(nodes=nodes, hexes=hexes, htag=htag, sec=sec, skel=skel)


def _ply_at(lam, frac):
    """Ply (material, angle) at through-thickness fraction frac in laminate lam (outer->inner)."""
    T = _thick(lam); d = frac * T; acc = 0.0
    for (m, t, a) in lam:
        acc += t
        if d <= acc + 1e-12:
            return m, a
    return lam[-1][0], lam[-1][2]


def solid_yaml_payload(res, cs1):
    """Per-hex fiber orientation + material sets for export_solid_yaml."""
    nodes, hexes, htag = res["nodes"], res["hexes"], res["htag"]
    sec = res["sec"]; nr = sec["nr"]
    lam_by_id = {sid: lam for lam, sid in cs1["laminates"].items()}
    cen = nodes[hexes].mean(1)
    oris = np.zeros((len(hexes), 9))
    mats = []
    for k, (tag, hx) in enumerate(zip(htag, hexes)):
        gen = nodes[hx[4]] - nodes[hx[0]]
        a1 = gen / np.linalg.norm(gen)                          # taper generator (span)
        e2r = nodes[hx[1]] - nodes[hx[0]]
        e2 = e2r - (e2r @ a1) * a1; e2 /= np.linalg.norm(e2)    # hoop / depth tangent
        e3 = np.cross(a1, e2); e3 /= np.linalg.norm(e3)
        if tag[0] == "skin":
            lam = lam_by_id[tag[1]]
            m, ang = _ply_at(lam, (tag[2] + 0.5) / nr)
        else:
            wlam = lam_by_id[cs1["webs"][tag[1]]["lam"]]
            ncols = len(sec["wpair"][tag[1]][0]) - 1
            m, ang = _ply_at(wlam, (tag[2] + 0.5) / max(1, ncols))
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        e1 = ca * a1 + sa * e2; e1 /= np.linalg.norm(e1)
        oris[k] = np.concatenate([e1, np.cross(e3, e1), e3])
        mats.append(m)
    return oris, mats
