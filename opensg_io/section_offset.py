"""opensg_io.section_offset -- robust inward offset of a closed section contour.

PreVABS/NuMAD-grade offset geometry (studied from prevabs src/geo/offset*.cpp + NuMAD
BladeDef.m), adapted to a STRICTLY STRUCTURED setting: every ring keeps the same node
count, so rings pair into quads / hexes 1:1.

Devices, in the order they act:

  * contour orientation from the SIGNED AREA (deterministic; a centroid test flips
    normals near thin trailing edges);
  * per-vertex MITER (angle-bisector) normals with the Clipper2 miter limit (2.0): at
    kinks (flatback TE corners, panel breaks) the offset point must sit on the bisector
    at d/cos(phi) to be distance d from BOTH adjacent walls;
  * WALL CLEARANCE estimator (PreVABS signedHalfThickness idea): per-node distance to
    the nearest opposing wall (outward normals anti-parallel).  NOTE a ray cast along
    the miter direction is the WRONG estimator: at a sharp tail the bisector ray runs
    down the sliver centreline, nearly parallel to both walls, and misses the pinch;
  * a hard FOLD VERIFIER on the built rings -- non-positive hoop cells and inner-ring
    self-intersections are detected directly and corrected locally (NuMAD verifies with
    an element-determinant check, NuMesh3D.m:61-98).  Estimators have blind spots
    (e.g. steep wedges defeat the anti-parallel test); the verifier has none.

Two public strategies:

  offset_rings(...)   -- clamp mode: locally THINS the stack where it cannot fit
                         (geometry preserved, laminate scaled); verified fold-free.
  open_thin_gaps(...) -- FULL-ACCURACY mode (NuMAD expandBladeGeometryTEs): pushes the
                         OML outward where the full laminate of both walls cannot fit,
                         so nominal ply thickness is preserved exactly; verified.
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
    scale so that p + d*s*m is at perpendicular distance d from both adjacent walls
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


def wall_clearance(oml, m_in, opposing=-0.2):
    """Per-node distance to the nearest OPPOSING wall of the closed CCW contour
    (edges whose outward normal is roughly anti-parallel to the node's outward miter
    direction).  Along-contour neighbours (normals ~parallel) never qualify, so there
    is no index-window blindness at a sharp tail.  This is an ESTIMATOR -- steep
    wedges (walls far from parallel) can evade it, which is why the fold VERIFIER
    below always runs on the built rings."""
    P = np.asarray(oml, float)
    N = len(P)
    e = np.roll(P, -1, axis=0) - P
    L = np.linalg.norm(e, axis=1)
    L[L == 0.0] = 1e-300
    t = e / L[:, None]
    n_out = np.column_stack([t[:, 1], -t[:, 0]])
    m_out = -np.asarray(m_in, float)
    c = np.full(N, np.inf)
    for i in range(N):
        opp = (n_out @ m_out[i]) < opposing
        if not opp.any():
            continue
        w = P[i] - P[opp]
        proj = np.clip(np.einsum("ij,ij->i", w, e[opp]) / (L[opp] ** 2), 0.0, 1.0)
        d = w - proj[:, None] * e[opp]
        c[i] = float(np.linalg.norm(d, axis=1).min())
    return c


def _fold_nodes_stack(oml, depth, nsub):
    """Hard verifier over the WHOLE ring stack: nodes involved in any fold.

    Cell area is QUADRATIC in the layer index, so an intermediate sub-cell can invert
    (offset caustic where neighbouring normal rays cross inside the wall) while the
    full-depth cell stays positive -- every consecutive sub-ring pair must be checked,
    plus self-intersection of every sub-ring.  depth (N,2) = full-depth INWARD
    displacement vector per node (d*s*m_in), ADDED to the contour.

    Returns (local, crossing) bool masks: `local` = non-positive hoop cells and
    near-adjacent crossings (corner bowties where hoop spacing < depth at a kink --
    the CLAMP's domain); `crossing` = DISTANT inner-ring crossings (two different
    walls folding through each other -- a genuine gap deficit, the OPENING's domain)."""
    N = len(oml)
    local = np.zeros(N, bool)
    crossing = np.zeros(N, bool)
    rings = [oml + (l / nsub) * depth for l in range(nsub + 1)]
    for l in range(nsub):
        lo, cr = _fold_nodes(rings[l], rings[l + 1])
        local |= lo
        crossing |= cr
    return local, crossing


def _fold_nodes(outer, inner, far=4):
    """One band of the verifier: (a) non-positive hoop cells (outer_i, outer_j,
    inner_j, inner_i) -> local; (b) proper self-intersections of the inner ring
    polyline, split by contour separation: segments within `far` indices -> local
    (corner bowtie), farther -> crossing (two walls through each other).
    Returns (local, crossing) bool masks (N,)."""
    N = len(outer)
    j = (np.arange(N) + 1) % N
    local = np.zeros(N, bool)
    crossing = np.zeros(N, bool)
    # (a) signed area of each hoop cell between the two rings
    x1, y1 = outer[:, 0], outer[:, 1]
    x2, y2 = outer[j, 0], outer[j, 1]
    x3, y3 = inner[j, 0], inner[j, 1]
    x4, y4 = inner[:, 0], inner[:, 1]
    s2 = (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4)
    f = s2 <= 0.0
    local |= f
    local |= np.roll(f, 1)
    # (b) inner-ring self-intersection: proper crossing of non-adjacent segments
    A, B = inner, inner[j]
    D = B - A
    for i in range(N):
        d1 = D[i]
        c1 = d1[0] * (A[:, 1] - A[i, 1]) - d1[1] * (A[:, 0] - A[i, 0])
        c2 = d1[0] * (B[:, 1] - A[i, 1]) - d1[1] * (B[:, 0] - A[i, 0])
        u1 = D[:, 0] * (A[i, 1] - A[:, 1]) - D[:, 1] * (A[i, 0] - A[:, 0])
        u2 = D[:, 0] * (B[i, 1] - A[:, 1]) - D[:, 1] * (B[i, 0] - A[:, 0])
        kadj = (np.arange(N) - i) % N
        hit = (c1 * c2 < 0) & (u1 * u2 < 0) & (kadj > 1) & (kadj < N - 1)
        if not hit.any():
            continue
        near = hit & ((kadj <= far) | (kadj >= N - far))
        farh = hit & ~near
        for h, mask in ((near, local), (farh, crossing)):
            if h.any():
                mask[i] = mask[j[i]] = True
                mask |= h
                mask |= np.roll(h, 1)
    return local, crossing


def clamp_depths(depths, s, c, safety=0.45, smooth_win=3):
    """Estimator clamp: scale factor f (N,) keeping the deepest ring within safety*c of
    the wall clearance (safety < 0.5 keeps the two walls from meeting), smoothed along
    the hoop so rings stay fair."""
    need = depths * s
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(need > 0, np.minimum(1.0, safety * c / need), 1.0)
    f[~np.isfinite(f)] = 1.0
    N = len(f)
    fm = f.copy()
    for w in range(1, smooth_win + 1):
        fm = np.minimum(fm, np.minimum(np.roll(f, w), np.roll(f, -w)))
    for _ in range(2):
        fm = 0.25 * np.roll(fm, 1) + 0.5 * fm + 0.25 * np.roll(fm, -1)
    return np.minimum(f, fm + 1e-12)


def open_thin_gaps(oml, tnode, nr=4, miter_limit=2.0, eta=0.88, verify_iters=25,
                   smooth_win=4):
    """NuMAD-style trailing-edge OPENING (expandBladeGeometryTEs generalized;
    BladeDef.m:194-244) -- the FULL-ACCURACY strategy.

    ONE a-priori push computed entirely from the ORIGINAL geometry (like NuMAD's
    analytic thickness wedge): wherever the two walls cannot host their FULL nominal
    laminates (margin eta), the OML moves OUTWARD along the original outward miter
    normal by half the deficit, smoothed along the hoop.  Re-measuring the deformed
    contour and pushing again is a FEEDBACK LOOP (new corners -> new 'deficits' ->
    runaway); only the hard fold VERIFIER may add small corrections afterwards, with a
    cumulative cap of 0.5*t per node -- beyond that the residual is left to the
    downstream clamp (fscale < 1 then reports honestly that the laminate cannot fit).

    Returns (opened_oml (N,2), moved (N,) total outward displacement per node)."""
    P0 = np.asarray(oml, float)
    Pc, flipped = ensure_ccw(P0)
    t = np.asarray(tnode, float)
    tc = t[::-1].copy() if flipped else t.copy()
    moved = np.zeros(len(Pc))

    def _eikonal(delta, kappa, max_iter=1000):
        """Gradient-limited spread: amplitude never changes faster than kappa per unit
        arc length (discrete eikonal cone envelope)."""
        L = np.linalg.norm(np.roll(Pc, -1, axis=0) - Pc, axis=1)
        Lp = np.roll(L, 1)
        a = delta.copy()
        for _ in range(max_iter):
            a2 = np.maximum(a, np.maximum(np.roll(a, 1) - kappa * Lp,
                                          np.roll(a, -1) - kappa * L))
            if (a2 - a).max() <= 1e-15:
                break
            a = a2
        return a

    def smooth_max(delta, kappa=0.2, headroom=0.25, rounds=40):
        """C-smooth push profile: gradient-limited spread with ROUNDED crests.

        A bare eikonal cone has a sharp ridge (slope jumps +kappa -> -kappa) whose
        inside-concave radius is smaller than the laminate depth -- the offset stack
        then caustics exactly on the crest.  So: spread with a small headroom over the
        requirement, smooth heavily (binomial), and take the max against the true
        requirement -- the headroom keeps the smoothed profile above it, so the max
        almost never re-kinks.  This is the discrete counterpart of NuMAD's C2
        polynomial wedge (smooth in value AND slope, BladeDef.m:209-225)."""
        a = _eikonal(delta, kappa)
        if a.max() <= 0.0:
            return a
        b = a + np.where(a > 0, headroom * tc, 0.0)
        b = _eikonal(b, kappa)
        for _r in range(rounds):
            b = 0.25 * np.roll(b, 1) + 0.5 * b + 0.25 * np.roll(b, -1)
        return np.maximum(b, a)

    def push_dirs(m_in, passes=6):
        """Mollified OUTWARD push directions: the raw miter field is discontinuous at
        contour kinks, and pushing a smooth amplitude along a discontinuous direction
        field folds the wall over itself (overhanging S-shoulder).  A few binomial
        passes + renormalization give a C1-ish field; NuMAD's analytic wedge is smooth
        in both amplitude AND direction for the same reason."""
        d = -m_in
        for _p in range(passes):
            d = 0.25 * np.roll(d, 1, axis=0) + 0.5 * d + 0.25 * np.roll(d, -1, axis=0)
        n = np.linalg.norm(d, axis=1)
        n[n < 1e-12] = 1.0
        return d / n[:, None]

    # ---- one a-priori opening from the ORIGINAL geometry
    m_in0, s0 = miter_normals(Pc, miter_limit)
    c0 = wall_clearance(Pc, m_in0)
    a = smooth_max(0.5 * np.maximum(0.0, 2.0 * tc * s0 / eta - c0))
    if a.max() > 0.0:
        Pc = Pc + a[:, None] * push_dirs(m_in0)
        moved += a

    # ---- bounded verifier corrections wherever full-depth folds remain (the clearance
    # estimator is blind to steep wedges).  On a smooth ample contour nothing folds and
    # this is a strict no-op; at a genuinely pinched tip it blunts the knife edge (the
    # NuMAD >=3 mm TE rule); at a sharp-but-ample corner it may round the corner by at
    # most the 0.5*t cap -- the radius real manufactured parts have anyway.
    cap = moved + 0.5 * tc
    for _ in range(verify_iters):
        m_in, s = miter_normals(Pc, miter_limit)
        local, crossing = _fold_nodes_stack(Pc, (tc * s)[:, None] * m_in, nr)
        bad = local | crossing
        if not bad.any():
            break
        step = smooth_max(0.1 * tc * s * bad)
        step = np.minimum(step, np.maximum(0.0, cap - moved))
        if step.max() <= 1e-15:
            break                                          # cap reached -> clamp handles rest
        Pc = Pc + step[:, None] * push_dirs(m_in)
        moved += step
    if flipped:
        return Pc[::-1].copy(), moved[::-1].copy()
    return Pc, moved


def offset_rings(oml, tnode, nr, miter_limit=2.0, safety=0.45, verify_iters=40):
    """Structured through-thickness rings of a closed contour -- CLAMP strategy
    (geometry preserved; the stack is locally thinned where it cannot fit).

    oml (N,2): outer mold line nodes.  tnode (N,): local wall thickness at each node.
    nr: number of through-thickness layers (nr+1 rings, ring 0 = OML).

    Returns rings (nr+1, N, 2) and fscale (N,) -- the thin-gap scale actually applied
    (1 everywhere the nominal laminate fits).  The result is VERIFIED fold-free: any
    residual fold (estimator blind spot) shrinks fscale locally until clean."""
    oml, flipped = ensure_ccw(np.asarray(oml, float))
    t = np.asarray(tnode, float)
    tc = t[::-1].copy() if flipped else t.copy()
    m_in, s = miter_normals(oml, miter_limit)
    c = wall_clearance(oml, m_in)
    f = clamp_depths(tc, s, c, safety=safety)
    for _ in range(verify_iters):
        local, crossing = _fold_nodes_stack(oml, (tc * f * s)[:, None] * m_in, nr)
        bad = local | crossing                             # clamp handles every fold kind
        if not bad.any():
            break
        badw = bad | np.roll(bad, 1) | np.roll(bad, -1)
        f[badw] *= 0.75
    rings = np.empty((nr + 1, len(oml), 2))
    for l in range(nr + 1):
        d = (l / nr) * tc * f * s
        rings[l] = oml + d[:, None] * m_in                 # +m_in = move INWARD
    if flipped:
        return rings[:, ::-1], f[::-1]
    return rings, f
