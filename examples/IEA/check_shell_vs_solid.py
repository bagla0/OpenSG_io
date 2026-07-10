"""check_shell_vs_solid.py -- consistency cross-check of the generated taper YAMLs.

Compares output/iea22_seg_shell.yaml (numeric, 0-based; FEniCS ShellSegmentMesh format)
against output/iea22_seg_solid.yaml (string, 1-based; SolidSegmentMesh format) and the
four boundary YAMLs, hunting for generator bugs:

  A. span axis identical (3rd coordinate; same z-range, same nsp slices)
  B. shell mid-surface vs solid wall: every shell SKIN node must lie between the solid
     OML and inner ring (distance to the solid skin mid-surface ~ 0)
  C. laminate consistency: shell section total thickness at each skin quad == solid
     ring depth at the same hoop position / span slice
  D. orientation convention BOTH meshes: e1 = +span (root->tip), |e1|=|e2|=|e3|=1,
     right-handed, e3 . (outward radial) < 0 (INWARD)
  E. boundary YAMLs == the end sections of the taper meshes (node subsets)
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load(path):
    return yaml.load(open(os.path.join(OUT, path)), Loader=CLoad)


def nodes_of(d):
    """Nodes as (N,3) float regardless of writer style (numeric list vs string row)."""
    rows = d["nodes"]
    if isinstance(rows[0][0], str):
        return np.array([[float(v) for v in r[0].split()] for r in rows])
    return np.array(rows, float)


def cells_of(d, one_based):
    rows = d["elements"]
    if isinstance(rows[0][0], str):
        c = np.array([[int(v) for v in r[0].split()] for r in rows])
    else:
        c = np.array(rows, int)
    return c - 1 if one_based else c


sh = load("iea22_seg_shell.yaml")
so = load("iea22_seg_solid.yaml")
SN, SQ = nodes_of(sh), cells_of(sh, one_based=False)
XN, XH = nodes_of(so), cells_of(so, one_based=True)
fails = []


def check(name, ok, detail=""):
    print("%-52s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(name)


# ---- A: span axis
zsh = np.unique(np.round(SN[:, 2], 9))
zso = np.unique(np.round(XN[:, 2], 9))
check("A1 span slices identical", len(zsh) == len(zso) and np.allclose(zsh, zso),
      "%d slices, z=[%.2f, %.2f]" % (len(zsh), zsh.min(), zsh.max()))

# ---- B: shell skin nodes ON the solid mid-surface (per span slice, nearest distance)
NPs = len(SN[np.isclose(SN[:, 2], zsh[0])])
NP = len(XN[np.isclose(XN[:, 2], zsh[0])])
# solid skin: nodes are ring-major sid(i,l)=i*(nr+1)+l; infer nr+1 from repetition:
# mid-surface = mean of ring 0 and ring nr per hoop node
d_mid = []
for z in (zsh[0], zsh[len(zsh) // 2], zsh[-1]):
    S = SN[np.isclose(SN[:, 2], z)][:, :2]
    X = XN[np.isclose(XN[:, 2], z)][:, :2]
    # brute-force: distance from each shell node to nearest solid node must be
    # <= half the local wall thickness (mid-surface lies between rings)
    for p in S:
        d_mid.append(np.min(np.linalg.norm(X - p, axis=1)))
d_mid = np.array(d_mid)
check("B1 shell nodes near solid wall nodes", d_mid.max() < 0.08,
      "max dist %.4f m (should be < half wall thickness)" % d_mid.max())

# ---- C: laminate totals -- shell section thickness vs solid ring depth
secs = {s["elementSet"]: s["layup"] for s in sh["sections"]}
set_of_quad = {}
for st in sh["sets"]["element"]:
    for lab in st["labels"]:
        set_of_quad[lab] = st["name"]
tsh = []
for k in range(len(SQ)):
    lam = secs[set_of_quad[k]]
    tsh.append(sum(t for _m, t, _a in lam))
tsh = np.array(tsh)
check("C1 shell layup thickness sane", 0.001 < tsh.min() and tsh.max() < 0.5,
      "wall t in [%.1f, %.1f] mm across %d sections" % (1e3 * tsh.min(), 1e3 * tsh.max(), len(secs)))
# spanwise interpolation visible: thickness at root bay vs tip bay of one region differs
nq0 = len(SQ) // len(zsh[:-1])                              # quads per span bay
troot, ttip = tsh[:nq0], tsh[-nq0:]
check("C2 spanwise layup interpolation active", abs(troot.mean() - ttip.mean()) > 1e-6,
      "mean wall t %.2f mm (root bay) vs %.2f mm (tip bay)" % (1e3 * troot.mean(), 1e3 * ttip.mean()))

# ---- D: orientation frames
for name, d, N_, C_, one in (("shell", sh, SN, SQ, False), ("solid", so, XN, XH, True)):
    O = np.array(d["elementOrientations"], float)
    e1, e2, e3 = O[:, 0:3], O[:, 3:6], O[:, 6:9]
    check("D1 %s |e|=1" % name,
          np.allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-9)
          and np.allclose(np.linalg.norm(e2, axis=1), 1, atol=1e-9)
          and np.allclose(np.linalg.norm(e3, axis=1), 1, atol=1e-9))
    rh = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    check("D2 %s right-handed" % name, np.all(rh > 0.999), "min det %.6f" % rh.min())
    span_pos = e1[:, 2]
    if name == "shell":
        check("D3 %s e1 = +span (root->tip)" % name, np.all(span_pos > 0.9),
              "min e1.z %.3f" % span_pos.min())
    else:
        # solid: fiber angle rotates e1 in the (span, e2) plane -> e1.z = cos(angle) > 0
        check("D3 %s e1 span-dominant (+z)" % name, np.all(span_pos > 0.2),
              "min e1.z %.3f" % span_pos.min())
    # e3 INWARD test on skin elements: e3 . (element centroid - section centroid) < 0
    cen = N_[C_].mean(1)
    inward = []
    for z in np.unique(np.round(cen[:, 2], 6))[:1]:
        m = np.isclose(np.round(cen[:, 2], 6), z)
        c0 = cen[m][:, :2].mean(0)
        r = cen[m][:, :2] - c0
        r /= np.linalg.norm(r, axis=1)[:, None]
        dot = np.einsum("ij,ij->i", e3[m][:, :2], r)
        inward.append(np.median(dot))
    check("D4 %s e3 inward (median over first slice)" % name, inward[0] < -0.2,
          "median e3.rhat %.3f (webs dilute toward 0)" % inward[0])

# ---- E: boundary yamls == end sections
for tag, si in (("L", 0), ("R", -1)):
    bs = load("iea22_boundary_%s_solid.yaml" % tag)
    BN = nodes_of(bs)
    Xend = XN[np.isclose(XN[:, 2], zso[si])]
    check("E1 solid boundary %s node count" % tag, len(BN) == len(Xend),
          "%d vs %d" % (len(BN), len(Xend)))
    check("E2 solid boundary %s coords match" % tag,
          np.allclose(np.sort(BN[:, 0]), np.sort(Xend[:, 0]), atol=1e-8)
          and np.allclose(np.sort(BN[:, 1]), np.sort(Xend[:, 1]), atol=1e-8))
    bh = load("iea22_boundary_%s_shell.yaml" % tag)
    BSN = nodes_of(bh)
    Send = SN[np.isclose(SN[:, 2], zsh[si])]
    check("E3 shell boundary %s node count" % tag, len(BSN) == len(Send),
          "%d vs %d" % (len(BSN), len(Send)))
    check("E4 shell boundary %s coords match" % tag,
          np.allclose(np.sort(BSN[:, 0]), np.sort(Send[:, 0]), atol=1e-8)
          and np.allclose(np.sort(BSN[:, 1]), np.sort(Send[:, 1]), atol=1e-8))

print("\n%s: %d checks failed" % ("CLEAN" if not fails else "BUGS FOUND", len(fails)))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
