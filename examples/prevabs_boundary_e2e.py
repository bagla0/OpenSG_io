"""prevabs_boundary_e2e.py -- prove: windIO station -> PreVABS -> 2D-solid boundary YAML.

Runs entirely on the server (PreVABS is a linux binary).  Steps:
  1. load_blade(windio) + build_cross_section(r)          (opensg_io)
  2. emit_prevabs(cs)  -> {name}.dat, materials.xml, {name}.xml
  3. run the PreVABS binary  -> {name}.sg  (the robust quad/tri 2D cross-section mesh)
  4. convert_sg_to_yaml.py   -> {name}_solid_boundary.yaml  (FEniCS 2D-solid format)
  5. report node/element counts + element types + a quick conformity sanity check.
"""
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from opensg_io.converter import load_blade, build_cross_section, emit_prevabs

r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5336
windio = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "IEA-22-280-RWT.yaml")
mesh_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02

name = "iea_r%04d" % int(round(r * 1000))
work = os.path.join(HERE, "mesh_out", "pv_" + name)
os.makedirs(work, exist_ok=True)

print("### 1-2. windIO station -> cross-section -> PreVABS XML")
blade = load_blade(windio)
cs = build_cross_section(blade, r, mesh_size=0.01)
info = emit_prevabs(cs, work, name=name, mesh_size=mesh_size)
print("    r=%.4f chord=%.4f  layups=%d webs=%d  -> %s.xml" %
      (r, cs["chord"], info["n_layups"], info["n_webs"], name))

print("### 3. run PreVABS binary")
pv = (sorted(glob.glob(os.path.expanduser("~/OpenSG_io/third_party/prevabs_bin/**/prevabs"), recursive=True)) or
      sorted(glob.glob(os.path.expanduser("~/OpenSG_io/third_party/prevabs/**/prevabs"), recursive=True)))
PVBIN = pv[0]
PVDIR = os.path.dirname(PVBIN)
env = dict(os.environ, LD_LIBRARY_PATH="%s:%s" % (PVDIR, os.path.expanduser("~/miniconda3/envs/opensg_2_0/lib")))
res = subprocess.run([PVBIN, "-i", name + ".xml", "--vabs", "--hm"], cwd=work, env=env,
                     capture_output=True, text=True)
print("    PreVABS rc=%d  (%s)" % (res.returncode, os.path.basename(PVBIN)))
if res.returncode != 0:
    print("STDOUT tail:\n", res.stdout[-1800:])
    print("STDERR tail:\n", res.stderr[-1800:])
    sys.exit(1)
sg = os.path.join(work, name + ".sg")
print("    %s.sg written: %s" % (name, os.path.exists(sg)))

print("### 4. convert .sg -> 2D-solid YAML")
conv = os.path.join(ROOT, "scripts", "convert_sg_to_yaml.py")
outyaml = os.path.join(work, name + "_solid_boundary.yaml")
res2 = subprocess.run([sys.executable, conv, sg, outyaml], capture_output=True, text=True)
if res2.returncode != 0:
    print("convert stdout:\n", res2.stdout[-1200:])
    print("convert stderr:\n", res2.stderr[-1800:])
    sys.exit(1)
# echo the informative convert lines
for ln in res2.stdout.splitlines():
    if any(k in ln for k in ("nnode", "node-counts", "wrote (theta1+theta3", "non-zero theta3")):
        print("    " + ln.strip())

print("### 5. conformity sanity check on the 2D-solid YAML")
import yaml as _y
d = _y.safe_load(open(outyaml))
nodes = [[float(v) for v in row[0].split()] for row in d["nodes"]]
elems = [[int(v) - 1 for v in row[0].split()] for row in d["elements"]]
nn, ne = len(nodes), len(elems)
tri = sum(1 for e in elems if len(e) == 3); quad = sum(1 for e in elems if len(e) == 4)
# every node referenced by >=1 element (no orphans); no element references a missing node
ref = set(i for e in elems for i in e)
orphans = set(range(nn)) - ref
bad = [e for e in elems if max(e) >= nn or min(e) < 0]
# signed area of each 2D element (in the y-z plane; node = "x y z" with x=axial=0)
P = np.array(nodes)[:, 1:]
neg = 0
for e in elems:
    q = P[e]
    a = 0.0
    for k in range(len(q)):
        x1, y1 = q[k]; x2, y2 = q[(k + 1) % len(q)]
        a += x1 * y2 - x2 * y1
    if a <= 0:
        neg += 1
print("    nodes=%d elements=%d (tri=%d quad=%d)" % (nn, ne, tri, quad))
print("    orphan nodes=%d  bad-index elems=%d  non-positive-area elems=%d" % (len(orphans), len(bad), neg))
print("OK -> %s" % outyaml)
