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
