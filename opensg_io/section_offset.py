"""opensg_io.section_offset -- robust inward offset of a closed section contour.

PreVABS-style offset geometry (see prevabs src/geo/offset*.cpp, join.cpp), adapted to a
STRICTLY STRUCTURED setting (every ring keeps the same node count, so rings pair into
quads / hexes 1:1):

  * per-vertex MITER (angle-bisector) normals with the Clipper2 miter limit (2.0) --
    a raw nodal normal is only correct on smooth walls; at kinks (flatback TE corners,
    panel breaks) the offset point must sit on the bisector at distance d/cos(phi) so it
    is distance d from BOTH adjacent walls;
  * contour orientation from the SIGNED AREA (deterministic), never from a centroid
    test -- centroid orientation flips normals near thin trailing edges;
  * a THIN-REGION guard (PreVABS "Stage E" signedHalfThickness idea): each vertex
    ray-casts inward to find the opposite wall; where the requested total depth exceeds
    a fraction of the local half-gap the whole through-thickness stack at that vertex is
    scaled down (and the scale is smoothed along the hoop) so offset rings NEVER cross
    the opposite wall -- the mesh stays conforming and fold-free at a pinched TE where
    the nominal laminate physically cannot fit.
"""
import numpy as np


def ensure_ccw(xy):
    """Return (xy, flipped) with the closed polyline ordered counter-clockwise."""
    x, y = xy[:, 0], xy[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area2 < 0.0:
        return xy[::-1].copy(), True
    return xy, False


def miter_normals(oml, miter_limit=2.0):
    """Inward per-vertex miter normals of a closed CCW contour.

    Returns (m, s): m (N,2) unit bisector directions pointing INWARD, s (N,) miter
    scale so that p - d*s*m is at perpendicular distance d from both adjacent walls
    (s = 1/cos(phi/2), clamped at miter_limit exactly like Clipper2/PreVABS)."""
    e = np.roll(oml, -1, axis=0) - oml                     # edge k: v_k -> v_{k+1}
    L = np.linalg.norm(e, axis=1)
    L[L == 0.0] = 1e-300
    t = e / L[:, None]
    n_out = np.column_stack([t[:, 1], -t[:, 0]])           # CCW -> outward edge normal
    n_prev = np.roll(n_out, 1, axis=0)                     # edge arriving at vertex k
    b = n_prev + n_out
    nb = np.linalg.norm(b, axis=1)
    deg = nb < 1e-12                                       # 180-degree spike: fall back
    b[deg] = n_prev[deg]
    nb[deg] = np.linalg.norm(b[deg], axis=1)
    m_out = b / nb[:, None]
    cosphi = np.einsum("ij,ij->i", m_out, n_out)           # = cos(phi/2)
    cosphi = np.clip(cosphi, 1.0 / miter_limit, 1.0)       # miter limit
    return -m_out, 1.0 / cosphi                            # inward direction


def inward_gap(oml, m_in, skip=2):
    """Ray-cast p_i + u*m_in_i (u>0) against all non-adjacent edges of the closed
    contour; returns g (N,) = distance to the OPPOSITE wall (inf if no hit).
    skip = number of adjacent edges on each side excluded from the cast."""
    N = len(oml)
    a = oml
    d = np.roll(oml, -1, axis=0) - oml                     # edge vectors
    g = np.full(N, np.inf)
    for i in range(N):
        p = oml[i]; q = m_in[i]
        # solve p + u q = a_k + v d_k  for every edge k
        det = d[:, 0] * (-q[1]) - d[:, 1] * (-q[0])
        ok = np.abs(det) > 1e-14
        rhs = p - a
        v = (rhs[:, 0] * (-q[1]) - rhs[:, 1] * (-q[0])) / np.where(ok, det, 1.0)
        u = (d[:, 0] * rhs[:, 1] - d[:, 1] * rhs[:, 0]) / np.where(ok, det, 1.0)
        adj = (np.arange(N) - i) % N
        near = (adj <= skip) | (adj >= N - skip)
        hit = ok & ~near & (v >= -1e-12) & (v <= 1.0 + 1e-12) & (u > 1e-12)
        if hit.any():
            g[i] = float(u[hit].min())
    return g


def clamp_depths(depths, s, g, safety=0.45, smooth_win=3):
    """Scale factor f (N,) so that the deepest ring at each vertex stays within
    safety*g of the inward gap (each wall keeps to its own half of the gap with
    margin), smoothed along the hoop so rings stay fair.  depths: (N,) TOTAL
    requested depth per vertex (outermost ring depth * miter scale applied by
    caller via s)."""
    need = depths * s
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(need > 0, np.minimum(1.0, safety * g / need), 1.0)
    f[~np.isfinite(f)] = 1.0
    # windowed minimum (spread the clamp so neighbours agree) then linear smooth
    N = len(f)
    fm = f.copy()
    for w in range(1, smooth_win + 1):
        fm = np.minimum(fm, np.minimum(np.roll(f, w), np.roll(f, -w)))
    for _ in range(2):
        fm = 0.25 * np.roll(fm, 1) + 0.5 * fm + 0.25 * np.roll(fm, -1)
    return np.minimum(f, fm + 1e-12)


def offset_rings(oml, tnode, nr, miter_limit=2.0, safety=0.45):
    """Structured through-thickness rings of a closed CCW contour.

    oml (N,2): outer mold line nodes.  tnode (N,): local wall thickness at each node.
    nr: number of through-thickness layers (nr+1 rings, ring 0 = OML).

    Returns rings (nr+1, N, 2) and fscale (N,) -- the thin-TE clamp factor actually
    applied (1 everywhere the nominal laminate fits)."""
    oml, flipped = ensure_ccw(np.asarray(oml, float))
    m_in, s = miter_normals(oml, miter_limit)
    g = inward_gap(oml, m_in)
    f = clamp_depths(np.asarray(tnode, float), s, g, safety=safety)
    rings = np.empty((nr + 1, len(oml), 2))
    for l in range(nr + 1):
        d = (l / nr) * tnode * f * s
        rings[l] = oml - d[:, None] * m_in
    if flipped:
        rings = rings[:, ::-1]
    return rings, (f if not flipped else f[::-1])
