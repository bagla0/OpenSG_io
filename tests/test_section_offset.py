"""Tests for opensg_io.section_offset.

Architecture under test: a smooth CLEARANCE ESTIMATOR (sees near-parallel thin walls --
the real blade-TE case) plus a hard FOLD VERIFIER on the built rings (catches everything,
including steep wedges that legitimately evade the anti-parallel estimator).  The tests
exercise each layer against its own contract, on two stress contours:

  * slit_contour  -- near-PARALLEL thin tail (estimator must see it),
  * wedge_contour -- STEEP sharp tail, walls ~70 deg apart (verifier must catch it).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.section_offset import (ensure_ccw, miter_normals, wall_clearance,
                                      offset_rings, open_thin_gaps)


def wedge_contour(n=200, L=1.0, h=0.30, tail=0.02):
    """Teardrop with a STEEP sharp thin tail (walls far from parallel near the tip)."""
    xs = np.linspace(0.0, 1.0, n // 2)
    half = 0.5 * tail + (h - 0.5 * tail) * (np.sin(np.pi * np.clip(xs, 0, 1)) ** 0.8)
    top = np.column_stack([L * xs, half])
    bot = np.column_stack([L * xs[::-1], -half[::-1]])
    return np.vstack([top, bot])


def slit_contour(n=240, L=1.0, h=0.30, tail=0.02, flat_from=0.65):
    """Blunt nose tapering to a near-PARALLEL thin tail of gap `tail` -- the realistic
    blade trailing-edge shape the clearance estimator is contracted to detect."""
    xs = np.linspace(0.0, 1.0, n // 2)
    ramp = np.clip((flat_from - xs) / flat_from, 0.0, 1.0)
    half = 0.5 * tail + (h - 0.5 * tail) * np.sin(0.5 * np.pi * ramp)
    top = np.column_stack([L * xs, half])
    bot = np.column_stack([L * xs[::-1], -half[::-1]])
    return np.vstack([top, bot])


def quad_areas(rings):
    """Signed areas of the (ring_l, ring_l+1) hoop cells -- all must stay positive."""
    nrp1, N, _ = rings.shape
    areas = []
    for l in range(nrp1 - 1):
        a, b = rings[l], rings[l + 1]
        for i in range(N):
            j = (i + 1) % N
            q = np.array([a[i], a[j], b[j], b[i]])
            s = 0.0
            for k in range(4):
                m = (k + 1) % 4
                s += q[k, 0] * q[m, 1] - q[m, 0] * q[k, 1]
            areas.append(0.5 * s)
    return np.array(areas)


def _oriented_areas(rings, contour):
    """quad_areas with a consistent sign for either input orientation."""
    a = quad_areas(rings)
    _pc, flipped = ensure_ccw(np.asarray(contour, float))
    return -a if flipped else a


def test_normals_and_estimator_on_parallel_tail():
    P = slit_contour(tail=0.02)
    Pc, _f = ensure_ccw(P)
    m, s = miter_normals(Pc)
    assert np.all(s >= 1.0) and np.all(s <= 2.0 + 1e-12)   # miter limit respected
    c = wall_clearance(Pc, m)
    assert c.min() > 0
    # near-parallel thin tail: the estimator MUST see the ~0.02 gap
    assert c.min() < 0.03, "estimator missed the parallel tail (got %.3f)" % c.min()


def test_clamp_estimator_thins_parallel_tail():
    P = slit_contour(tail=0.02)
    t = np.full(len(P), 0.04)                              # 4x the half-gap per wall
    rings, f = offset_rings(P, t, nr=3)
    assert f.min() < 0.5, "clamp should engage strongly on the parallel tail"
    assert _oriented_areas(rings, P).min() > 0, "clamped rings must not fold"


def test_verifier_catches_steep_wedge():
    P = wedge_contour(tail=0.02)
    t = np.full(len(P), 0.04)
    rings, f = offset_rings(P, t, nr=3)                    # estimator is blind here
    assert f.min() < 1.0, "verifier should have shrunk the fold region"
    assert _oriented_areas(rings, P).min() > 0, "verified rings must not fold"


def test_open_preserves_full_laminate_at_sharp_te():
    """The PHYSICAL blade-TE case (sharp local pinch): opening must restore the FULL
    nominal laminate everywhere -- zero ply thinning -- with a small smooth opening."""
    P = wedge_contour(tail=0.02)
    t = np.full(len(P), 0.04)
    Po, moved = open_thin_gaps(P, t)
    assert 1e-4 < moved.max() < 0.05, "small bounded opening expected"
    rings, f = offset_rings(Po, t, nr=3)
    assert f.min() > 0.999, "after opening the FULL laminate must fit (no thinning)"
    assert _oriented_areas(rings, Po).min() > 0, "opened rings must not fold"
    d = np.linalg.norm(rings[-1] - rings[0], axis=1)
    assert d.min() >= 0.04 - 1e-9, "full nominal depth everywhere"


def test_open_bounded_and_honest_on_oversized_laminate():
    """Pathological case (laminate 4x the gap over a LONG channel = design error, not a
    blade TE): opening must stay BOUNDED and fold-free, improve the fit substantially,
    and report the honest residual through fscale -- never run away or fold."""
    P = slit_contour(tail=0.02)
    t = np.full(len(P), 0.04)
    _rings0, f0 = offset_rings(P, t, nr=3)                 # without opening
    Po, moved = open_thin_gaps(P, t)
    assert moved.max() < 0.10, "opening must stay bounded (no runaway)"
    rings, f = offset_rings(Po, t, nr=3)
    assert _oriented_areas(rings, Po).min() > 0, "must stay fold-free"
    assert f.min() > 1.5 * f0.min(), "opening must substantially improve the fit"
    assert (f < 0.999).sum() < 0.4 * (f0 < 0.999).sum(), "thinned region must shrink a lot"
    assert (f < 0.999).sum() < 0.2 * len(P), "residual thinning must stay localized"


def test_open_noop_on_smooth_ample_contour():
    """Smooth contour with ample clearance: the opening is a STRICT no-op (the ellipse
    is the honest contract; sharp 90-deg corners with hoop spacing < depth may
    legitimately get rounded by <= 0.5*t, which is not a no-op violation)."""
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    P = np.column_stack([np.cos(th), 0.45 * np.sin(th)])
    t = np.full(len(P), 0.03)
    Po, moved = open_thin_gaps(P, t)
    assert moved.max() <= 1e-12
    assert np.allclose(Po, P)


if __name__ == "__main__":
    for fn in [test_normals_and_estimator_on_parallel_tail,
               test_clamp_estimator_thins_parallel_tail,
               test_verifier_catches_steep_wedge,
               test_open_preserves_full_laminate_at_sharp_te,
               test_open_bounded_and_honest_on_oversized_laminate,
               test_open_noop_on_smooth_ample_contour]:
        fn()
        print("PASS", fn.__name__)
