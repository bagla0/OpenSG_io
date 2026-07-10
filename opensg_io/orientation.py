"""opensg_io.orientation -- element material frames in the NuMAD / VABS / PreVABS
convention.

Every element carries a right-handed orthonormal triad (e1, e2, e3):

  * **e1** = the beam / span axis, oriented from the LEFT (root) end to the RIGHT (tip)
    end of the segment -- the reference from which the ply fiber angle is measured;
  * **e3** = the surface normal pointing INWARD (into the wall for the skin, the plate
    normal for a web);
  * **e2** = e3 x e1, the in-surface transverse direction (hoop for the skin, depth for
    a web).

For a SOLID element (one ply per element) the fiber angle theta rotates (e1, e2) about
e3 into the ply frame.  For a SHELL element the triad is purely geometric and the ply
angles live in the layup, so theta = 0.

The 9-vector stored per element is [e1(3), e2(3), e3(3)], matching the OpenSG
`elementOrientations` schema (FEniCS ShellSegmentMesh / SolidSegmentMesh read exactly
these nine numbers).
"""
import math

import numpy as np


def element_frame(span_vec, n_surface, angle_deg=0.0):
    """Return the 9-list [e1, e2, e3] for one element.

    span_vec (3,)   : the element's span/beam direction (root->tip); need not be unit.
    n_surface (3,)  : an INWARD surface-normal estimate (into the wall / plate); need
                      not be unit or exactly orthogonal to span_vec -- it is
                      Gram-Schmidt orthogonalized against e1.
    angle_deg       : ply fiber angle (deg) for a solid element; 0 for a shell frame.
    """
    e1 = np.asarray(span_vec, float)
    e1 = e1 / np.linalg.norm(e1)
    n = np.asarray(n_surface, float)
    e3 = n - np.dot(n, e1) * e1                            # inward normal, _|_ to span
    ne = np.linalg.norm(e3)
    if ne < 1e-12:                                         # degenerate: pick any _|_
        tmp = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e3 = tmp - np.dot(tmp, e1) * e1
        ne = np.linalg.norm(e3)
    e3 = e3 / ne
    e2 = np.cross(e3, e1)                                  # right-handed: e1 = e2 x e3
    e2 = e2 / np.linalg.norm(e2)
    if angle_deg:                                          # rotate fiber about e3
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
    return np.concatenate([e1, e2, e3])


def skin_inward_normal_2d(rings, l, i, ii):
    """Inward (OML->interior) in-plane normal of a skin face between ring l and l+1 at
    hoop nodes i, ii, from the station-0 ring stack rings (nr+1, NC, 2)."""
    inw = 0.5 * ((rings[l + 1, i] - rings[l, i]) + (rings[l + 1, ii] - rings[l, ii]))
    n = np.linalg.norm(inw)
    return inw / n if n > 1e-12 else np.array([1.0, 0.0])


def web_plate_normal_2d(p_m, p_mp1):
    """In-plane plate normal of a web face whose depth edge runs p_m -> p_mp1: the
    depth tangent rotated +90 deg (= span x depth projected into the section plane)."""
    d = np.asarray(p_mp1, float) - np.asarray(p_m, float)
    n = np.array([-d[1], d[0]])
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-12 else np.array([1.0, 0.0])
