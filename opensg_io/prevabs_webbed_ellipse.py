"""make_ellipse_prevabs.py -- emit a PreVABS 2.1 webbed-ellipse cross-section (ellipse.dat +
ellw.xml + materials.xml), run PreVABS to get the 2D quad/tri cross-section mesh (.sg).
Baseline = OML (mid ellipse a,b offset OUTWARD by t/2) so the [-45] skin built inward by t
has its mid-surface back at (a,b) -- matching the RM-shell contour.  3 vertical webs at x=c*a.
"""
import os, math, subprocess, glob, sys
import numpy as np

t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
a, b = 1.0, 0.6
W = os.path.expanduser("~/claude_tmp/pvwork/ellw")
os.makedirs(W, exist_ok=True)

# ellipse.dat: INNER baseline (mid ellipse offset INWARD by t/2); PreVABS builds the [-45]
# skin OUTWARD by t (direction="right"), so the skin mid-surface lands back on (a,b).
N = 260
th = np.linspace(0, 2 * math.pi, N, endpoint=True)
mx, my = a * np.cos(th), b * np.sin(th)
nx, ny = b * np.cos(th), a * np.sin(th); nn = np.hypot(nx, ny)
ox, oy = mx - (t / 2) * nx / nn, my - (t / 2) * ny / nn
with open(os.path.join(W, "ellipse.dat"), "w") as f:
    for x, y in zip(ox, oy):
        f.write("%.8f  %.8f\n" % (x, y))

open(os.path.join(W, "materials.xml"), "w").write("""<materials>
  <material name="ani" type="orthotropic">
    <density>1800</density>
    <elastic>
      <e1>3.7e10</e1><e2>9e9</e2><e3>9e9</e3>
      <g12>4e9</g12><g13>4e9</g13><g23>4e9</g23>
      <nu12>0.3</nu12><nu13>0.3</nu13><nu23>0.3</nu23>
    </elastic>
  </material>
  <lamina name="la_ani"><material>ani</material><thickness>%g</thickness></lamina>
</materials>""" % t)

open(os.path.join(W, "ellw.xml"), "w").write("""<cross_section name="ellw">
  <include><material>materials</material></include>
  <analysis><model>1</model></analysis>
  <general>
    <mesh_size>0.03</mesh_size>
    <element_type>linear</element_type>
  </general>
  <baselines>
    <line name="ln_ell" type="airfoil">
      <points data="file" format="1" header="0">ellipse.dat</points>
    </line>
    <point name="wp_m">-0.5 0</point>
    <line name="bl_web_m"><point>wp_m</point><angle>90</angle></line>
    <point name="wp_0">0.0 0</point>
    <line name="bl_web_0"><point>wp_0</point><angle>90</angle></line>
    <point name="wp_p">0.5 0</point>
    <line name="bl_web_p"><point>wp_p</point><angle>90</angle></line>
  </baselines>
  <layups>
    <layup name="layup_skin"><layer lamina="la_ani">-45</layer></layup>
    <layup name="layup_web"><layer lamina="la_ani">-45</layer></layup>
  </layups>
  <component name="surface">
    <segment><baseline>ln_ell</baseline><layup direction="right">layup_skin</layup></segment>
  </component>
  <component name="web_m" depend="surface"><segment><baseline>bl_web_m</baseline><layup>layup_web</layup></segment></component>
  <component name="web_0" depend="surface"><segment><baseline>bl_web_0</baseline><layup>layup_web</layup></segment></component>
  <component name="web_p" depend="surface"><segment><baseline>bl_web_p</baseline><layup>layup_web</layup></segment></component>
</cross_section>""")

PVBIN = glob.glob(os.path.expanduser("~/OpenSG_io/third_party/prevabs_bin/**/prevabs"), recursive=True)[0]
PVDIR = os.path.dirname(PVBIN)
env = dict(os.environ, LD_LIBRARY_PATH="%s:%s" % (PVDIR, os.path.expanduser("~/miniconda3/envs/opensg_2_0/lib")))
r = subprocess.run([PVBIN, "-i", "ellw.xml", "--vabs", "--hm"], cwd=W, env=env, capture_output=True, text=True)
print("=== RC", r.returncode, "===")
print("STDOUT tail:\n", r.stdout[-1500:])
if r.returncode != 0:
    print("STDERR tail:\n", r.stderr[-1500:])
print("outputs:", sorted(os.listdir(W)))
sg = os.path.join(W, "ellw.sg")
if os.path.exists(sg):
    with open(sg) as f:
        head = [next(f) for _ in range(6)]
    print("ellw.sg head:\n", "".join(head))
