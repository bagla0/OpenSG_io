"""_hex_invert_debug.py -- ROOT CAUSE of the taper hex inversion, and why refinement can't fix it.

The loft (hex_loft.hex_between_sections) sets every span slice to a LINEAR blend of the two
section node positions:  P(tau) = (1-tau)*P1 + tau*P2 .  Faces are wound CCW using STATION-0
geometry only.  So if a face is positive-area at station 0 but the SAME node ordering is
negative-area at station 1, the linear morph passes through a zero-area (degenerate) then
negative-area (folded) configuration at some INTERIOR tau*.  That fold is a property of the
interpolation PATH -- independent of how many span/thickness/hoop elements we use.  This script
proves it: (A) counts faces that fold along the morph and where; (B) sweeps nsp/nr/mesh and shows
the inverted-hex count does NOT go to zero.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import section_skeleton, build_section_mesh, hex_between_sections, _hex_min_sj

W = "examples/data/IEA-22-280-RWT.yaml"
blade = load_blade(W)


def zof(r):
    try:
        ra = blade.osh["reference_axis"]["z"]
        return float(np.interp(r, ra["grid"], ra["values"]))
    except Exception:
        return r * 137.0                               # IEA-22 span length (matches the blade .dat)


def signed_areas(P, faces):
    q = P[faces, :2]                                   # (nf,4,2)
    a = np.zeros(len(faces))
    for k in range(4):
        b = (k + 1) % 4
        a += q[:, k, 0] * q[:, b, 1] - q[:, b, 0] * q[:, k, 1]
    return 0.5 * a


def analyze_pair(r1, r2, label, mesh=0.02, nw=3, nr=4):
    print("\n============================================================")
    print("PAIR %s :  r=%.4f -> %.4f   (chord %.2f -> %.2f,  span %.1f -> %.1f m)"
          % (label, r1, r2, blade.scalar("chord", r1), blade.scalar("chord", r2), zof(r1), zof(r2)))
    cs1 = build_cross_section(blade, r1, mesh_size=mesh)
    cs2 = build_cross_section(blade, r2, mesh_size=mesh)
    skel = section_skeleton([cs1, cs2], mesh_size=mesh, nw=nw)
    sec = build_section_mesh([cs1, cs2], skel, nr=nr)
    P1, P2 = sec["stations"]
    faces = sec["faces2d"]
    isweb = np.array([sec["fregion"][i][0] == "web" for i in range(len(faces))])

    # (A) fold analysis: min signed area along the linear morph tau in [0,1]
    A0 = signed_areas(P1, faces)
    A1 = signed_areas(P2, faces)
    Amin = np.full(len(faces), np.inf); tstar = np.zeros(len(faces))
    for t in np.linspace(0, 1, 201):
        A = signed_areas((1 - t) * P1 + t * P2, faces)
        upd = A < Amin; Amin[upd] = A[upd]; tstar[upd] = t
    fold = Amin < -1e-12
    print("  2-D faces: %d total (%d web, %d skin)" % (len(faces), isweb.sum(), (~isweb).sum()))
    print("  station-0 area>0 everywhere (by construction): %s ; station-1 area<0 for %d faces (%d web)"
          % (bool((A0 > -1e-12).all()), int((A1 < 0).sum()), int(((A1 < 0) & isweb).sum())))
    print("  faces that FOLD along the morph (min area<0): %d  ->  web=%d  skin=%d"
          % (fold.sum(), (fold & isweb).sum(), (fold & ~isweb).sum()))
    if fold.any():
        print("  fold location tau* (interior => refinement-independent): min=%.2f median=%.2f max=%.2f"
              % (tstar[fold].min(), np.median(tstar[fold]), tstar[fold].max()))

    # (B) empirical: does refinement reduce the inverted-hex count?
    z1, z2 = zof(r1), zof(r2)
    print("  -- span refinement (nsp), nr=4 mesh=0.02 --")
    for nsp in [1, 4, 12, 24, 48]:
        res = hex_between_sections(cs1, cs2, z1, z2, nr=4, nsp=nsp, nw=nw, mesh_size=mesh)
        sj = _hex_min_sj(res["nodes"], res["hexes"])
        print("     nsp=%3d: hexes=%7d  inverted=%6d (%.2f%%)  minSJ=%+.3f  [repair swapped=%d still=%d]"
              % (nsp, len(res["hexes"]), int((sj < 0).sum()), 100 * (sj < 0).mean(), sj.min(),
                 res["n_swapped"], res["n_still_inverted"]))
    print("  -- thickness refinement (nr), nsp=12 --")
    for nrr in [2, 4, 8]:
        res = hex_between_sections(cs1, cs2, z1, z2, nr=nrr, nsp=12, nw=nw, mesh_size=mesh)
        sj = _hex_min_sj(res["nodes"], res["hexes"])
        print("     nr=%2d : hexes=%7d  inverted=%6d (%.2f%%)  minSJ=%+.3f"
              % (nrr, len(res["hexes"]), int((sj < 0).sum()), 100 * (sj < 0).mean(), sj.min()))
    print("  -- hoop refinement (mesh_size), nsp=12 nr=4 --")
    for m in [0.04, 0.02, 0.01]:
        c1 = build_cross_section(blade, r1, mesh_size=m); c2 = build_cross_section(blade, r2, mesh_size=m)
        res = hex_between_sections(c1, c2, z1, z2, nr=4, nsp=12, nw=nw, mesh_size=m)
        sj = _hex_min_sj(res["nodes"], res["hexes"])
        print("     mesh=%.3f: hexes=%7d  inverted=%6d (%.2f%%)  minSJ=%+.3f"
              % (m, len(res["hexes"]), int((sj < 0).sum()), 100 * (sj < 0).mean(), sj.min()))


analyze_pair(0.1967, 0.2470, "MILD (adjacent stations)")
analyze_pair(0.1967, 0.3993, "STEEP (skips a station)")
