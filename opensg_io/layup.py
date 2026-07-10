"""opensg_io.layup -- rigorous spanwise layup definition for tapered segments.

A blade laminate is an ordered stack of plies (outer -> inner), each ply a
(material, fiber_angle, thickness).  Between two span stations the laminate changes
by (a) plies getting thicker/thinner and (b) plies being ADDED or DROPPED -- the ply
drops that carry the load out of a tapering spar.  NuMAD encodes this as, per ply, a
spanwise ply-count curve n(s) with the ply existing only where n(s) >= its index
(BladeDef.findLayerExtents / ComponentDef): a ply "drops" where its count curve
crosses below its index.

For a TWO-station segment the same information reduces to: align the two stations'
ply stacks by ply IDENTITY (material, angle) preserving stacking order, then linearly
interpolate each matched ply's thickness across the span.  A ply present at only one
station is a **ply drop**: its thickness ramps linearly to zero at the other end.  This
is the rigorous, order-preserving, drop-aware definition -- not a frozen copy of one
station.

    TaperLaminate.from_stations(lam_left, lam_right)   # align + build
        .at(tau)   -> [(material, thickness, angle)]   # laminate at span fraction tau
        .thickness(tau) -> float                       # total wall thickness at tau

`lam_left` / `lam_right` are ordered lists (or tuples) of (material, thickness, angle),
outer surface first -- exactly the `_lam_tuple(cs, set_id)` output of build_cross_section.
"""
from dataclasses import dataclass
from typing import List, Tuple


def _key(ply, angle_tol=1.0):
    """Ply identity for alignment: (material, rounded angle).  Two plies match iff same
    material and fiber angle within angle_tol degrees."""
    m, _t, a = ply
    return (m, round(float(a) / angle_tol) * angle_tol)


def _lcs_align(seq_l, seq_r):
    """Longest-common-subsequence alignment of two ply-key sequences, order preserved.
    Returns a list of (il, ir) index pairs where il/ir is the index into seq_l/seq_r or
    None (a drop on that side).  Classic LCS backtrace -- robust to plies added/removed
    anywhere in the stack, not just at the ends."""
    nl, nr = len(seq_l), len(seq_r)
    dp = [[0] * (nr + 1) for _ in range(nl + 1)]
    for i in range(nl - 1, -1, -1):
        for j in range(nr - 1, -1, -1):
            dp[i][j] = (dp[i + 1][j + 1] + 1 if seq_l[i] == seq_r[j]
                        else max(dp[i + 1][j], dp[i][j + 1]))
    out, i, j = [], 0, 0
    while i < nl and j < nr:
        if seq_l[i] == seq_r[j]:
            out.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append((i, None)); i += 1                  # ply only on the left -> drop
        else:
            out.append((None, j)); j += 1                  # ply only on the right -> drop
    out += [(i2, None) for i2 in range(i, nl)]
    out += [(None, j2) for j2 in range(j, nr)]
    return out


@dataclass
class PlyGroup:
    """One ply through the segment: constant material + angle, thickness ramping
    linearly from t_left (station L) to t_right (station R).  t=0 at one end encodes a
    ply drop."""
    material: str
    angle: float
    t_left: float
    t_right: float

    def thickness(self, tau: float) -> float:
        return (1.0 - tau) * self.t_left + tau * self.t_right

    @property
    def is_drop(self) -> bool:
        return min(self.t_left, self.t_right) <= 1e-12 < max(self.t_left, self.t_right)


@dataclass
class TaperLaminate:
    """Ordered (outer->inner) list of PlyGroup spanning the segment."""
    plies: List[PlyGroup]

    @classmethod
    def from_stations(cls, lam_left, lam_right, angle_tol=1.0):
        L = [tuple(p) for p in lam_left]
        R = [tuple(p) for p in lam_right]
        kl = [_key(p, angle_tol) for p in L]
        kr = [_key(p, angle_tol) for p in R]
        groups = []
        for il, ir in _lcs_align(kl, kr):
            if il is not None and ir is not None:
                m, _t, a = L[il]
                groups.append(PlyGroup(m, float(a), float(L[il][1]), float(R[ir][1])))
            elif il is not None:                           # dropped toward the right end
                m, t, a = L[il]
                groups.append(PlyGroup(m, float(a), float(t), 0.0))
            else:                                          # added toward the right end
                m, t, a = R[ir]
                groups.append(PlyGroup(m, float(a), 0.0, float(t)))
        return cls(groups)

    def at(self, tau: float, tmin: float = 1e-9) -> List[Tuple[str, float, float]]:
        """Laminate at span fraction tau in [0,1]: [(material, thickness, angle)],
        outer->inner, with fully-dropped (t<tmin) plies omitted."""
        out = []
        for g in self.plies:
            t = g.thickness(tau)
            if t > tmin:
                out.append((g.material, t, g.angle))
        return out

    def thickness(self, tau: float) -> float:
        return float(sum(g.thickness(tau) for g in self.plies))

    def ply_of_depth(self, tau: float, frac: float) -> Tuple[str, float]:
        """(material, angle) of the ply at through-thickness fraction frac in [0,1]
        (outer->inner) of the laminate at span fraction tau -- for per-hex material
        assignment in the structured solid."""
        lam = self.at(tau)
        if not lam:
            return self.plies[-1].material, self.plies[-1].angle
        T = sum(t for (_m, t, _a) in lam)
        d = frac * T
        acc = 0.0
        for (m, t, a) in lam:
            acc += t
            if d <= acc + 1e-12:
                return m, a
        return lam[-1][0], lam[-1][2]

    def dropped(self):
        """The plies that drop within the segment (present at one end, gone at the
        other) -- the ply-drop manifest, for reporting."""
        return [g for g in self.plies if g.is_drop]

    # ---------------- ply-conforming through-thickness layer grouping ----------------
    #
    # A structured hex wall with nr EQUAL layers destroys a sandwich: a 3 mm skin on a
    # 76 mm wall is thinner than one 19 mm layer, so every layer-midpoint samples the
    # core and the skins vanish from the solid (measured: ALL panel hexes = foam ->
    # section GA/GJ collapse ~10x).  NuMAD solves this with ply-conforming guide
    # surfaces (editStacksForSolidMesh normalizes every stack to a fixed number of ply
    # GROUPS and offsets one guide surface per group, BladeDef.m:1571-1625/1666-1735).
    # Same idea here: group the plies into exactly nr contiguous groups with cuts AT
    # PLY BOUNDARIES (merging the thinnest neighbours when there are too many plies,
    # splitting the thickest group internally when too few), so hex layer l IS ply
    # group l and carries its dominant ply's material exactly -- for any nr.

    def group_cuts(self, nr: int, tau_ref: float = 0.5):
        """nr-1 interior cuts as (ply_index, alpha): cumulative depth of cut =
        sum(t_0..t_{i-1}) + alpha*t_i, evaluated with the ply thicknesses at any tau
        (so cuts move continuously along the span).  alpha=0 -> the ply's OUTER
        boundary.  Chosen on the tau_ref thicknesses."""
        t = [max(g.thickness(tau_ref), 0.0) for g in self.plies]
        n = len(t)
        # start with every ply its own group (cuts at all interior ply boundaries)
        cuts = [(i, 0.0) for i in range(1, n)]
        # too many groups -> merge the adjacent pair with the smallest combined
        # thickness (NuMAD's thin-ply merge), dropping the cut between them
        def group_spans(cuts_):
            pos = [0.0] + [sum(t[:i]) + a * t[i] for (i, a) in cuts_] + [sum(t)]
            return [(pos[k], pos[k + 1]) for k in range(len(pos) - 1)]
        while len(cuts) > nr - 1:
            spans = group_spans(cuts)
            pair = min(range(len(cuts)),
                       key=lambda k: (spans[k][1] - spans[k][0]) + (spans[k + 1][1] - spans[k + 1][0]))
            cuts.pop(pair)
        # too few -> split the thickest group at its mid-depth (inside one ply)
        while len(cuts) < nr - 1:
            spans = group_spans(cuts)
            k = max(range(len(spans)), key=lambda q: spans[q][1] - spans[q][0])
            mid = 0.5 * (spans[k][0] + spans[k][1])
            acc = 0.0
            for i in range(n):
                if acc + t[i] >= mid - 1e-15:
                    alpha = 0.0 if t[i] <= 1e-15 else (mid - acc) / t[i]
                    cuts.append((i, float(min(max(alpha, 0.0), 1.0))))
                    break
                acc += t[i]
            cuts.sort(key=lambda ia: sum(t[:ia[0]]) + ia[1] * t[ia[0]])
        return cuts

    def group_fractions(self, cuts, tau: float):
        """(nr+1,) cumulative through-thickness FRACTIONS of the group interfaces at
        span fraction tau (0=OML, 1=inner surface), from the given cuts."""
        t = [max(g.thickness(tau), 0.0) for g in self.plies]
        T = sum(t)
        if T <= 0:
            return [l / max(1, len(cuts) + 1) for l in range(len(cuts) + 2)]
        pos = [0.0] + [sum(t[:i]) + a * t[i] for (i, a) in cuts] + [T]
        return [p / T for p in pos]

    def group_material(self, cuts, l: int, tau: float):
        """(material, angle) of hex layer l = the DOMINANT (thickest) ply inside group
        l's depth span at span fraction tau."""
        t = [max(g.thickness(tau), 0.0) for g in self.plies]
        T = sum(t)
        pos = [0.0] + [sum(t[:i]) + a * t[i] for (i, a) in cuts] + [T]
        lo, hi = pos[l], pos[l + 1]
        best, bl = None, -1.0
        acc = 0.0
        for i, g in enumerate(self.plies):
            a, b = acc, acc + t[i]
            acc = b
            ov = min(b, hi) - max(a, lo)
            if ov > bl:
                bl, best = ov, g
        return best.material, best.angle
