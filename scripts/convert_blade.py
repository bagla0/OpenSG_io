"""GENERAL windIO -> OpenSG driver (portable). For ANY windIO blade (v1 or v2) emits, per span station,
a 1D-shell SG YAML, and with --solid a 2D-solid SG YAML via the repo-linked PreVABS (third_party/prevabs_bin)
-> .sg -> convert_sg_to_yaml.py. No blade-specific code; cross-platform (auto-finds prevabs / prevabs.exe).

  python scripts/convert_blade.py BAR_URC.yaml out --name bar --solid
  python scripts/convert_blade.py blade.yaml out --stations 0.3 0.5 0.7 --shell-ms 0.005
"""
import os, sys, glob, argparse, subprocess, platform
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs


def find_prevabs():
    exe = "prevabs.exe" if platform.system().lower().startswith("win") else "prevabs"
    hits = glob.glob(os.path.join(REPO, "third_party", "prevabs_bin", "**", exe), recursive=True)
    if hits:
        return sorted(hits)[-1]
    return os.environ.get("PREVABS_EXE")


def main():
    ap = argparse.ArgumentParser(description="windIO blade -> OpenSG 1D-shell (+ 2D-solid) per station")
    ap.add_argument("yaml"); ap.add_argument("outdir")
    ap.add_argument("--name", default=None, help="output naming stem (default: yaml basename)")
    ap.add_argument("--stations", type=float, nargs="+", default=None)
    ap.add_argument("--nstations", type=int, default=10)
    ap.add_argument("--shell-ms", type=float, default=0.01)
    ap.add_argument("--solid-ms", type=float, default=0.02)
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="shell reference-surface through-thickness fraction from the OML "
                         "(0.5 = mid-surface / centre reference; 0.0 = OML)")
    ap.add_argument("--solid", action="store_true", help="also build the 2D-solid via PreVABS")
    ap.add_argument("--prevabs", default=None, help="prevabs path (default: repo third_party/prevabs_bin)")
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.yaml))[0]
    os.makedirs(a.outdir, exist_ok=True)
    prevabs = a.prevabs or find_prevabs()
    convert = os.path.join(HERE, "convert_sg_to_yaml.py")
    stations = a.stations or [round((i + 1) / (a.nstations + 1), 4) for i in range(a.nstations)]
    blade = load_blade(a.yaml)
    print("reader=%s blade=%s stations=%s  prevabs=%s" % (type(blade).__name__, name, stations, prevabs), flush=True)
    if a.solid and not prevabs:
        sys.exit("no PreVABS found -- run scripts/fetch_prevabs.py or pass --prevabs")

    for r in stations:
        tag = "r%03d" % round(r * 100)
        try:
            cs = build_cross_section(blade, r, mesh_size=a.shell_ms)
        except Exception as e:
            print("r=%.2f BUILD FAILED: %s" % (r, repr(e)[:90]), flush=True); continue
        emit_opensg_yaml(cs, os.path.join(a.outdir, "shell_%s_%s.yaml" % (name, tag)),
                         fraction=a.fraction)
        msg = "r=%.2f chord=%.3f shell:%d elems [%d lam,%d webs]" % (
            r, cs["chord"], len(cs["elems"]), len(cs["laminates"]), len(cs["webs"]))
        if a.solid:
            pv = os.path.join(a.outdir, "prevabs_%s" % tag)
            try:
                emit_prevabs(cs, pv, name="%s_%s" % (name, tag), mesh_size=a.solid_ms)
                subprocess.run([prevabs, "-i", "%s_%s.xml" % (name, tag), "--vabs", "--hm"], cwd=pv,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                sg = os.path.join(pv, "%s_%s.sg" % (name, tag))
                solid_yaml = os.path.join(a.outdir, "solid_%s_%s.yaml" % (name, tag))
                subprocess.run([sys.executable, convert, sg, solid_yaml], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                import yaml
                d = yaml.safe_load(open(solid_yaml))
                msg += "  solid:%d nodes/%d elems OK" % (len(d["nodes"]), len(d["elements"]))
            except Exception as e:
                msg += "  SOLID FAILED: %s" % repr(e)[:80]
        print(msg, flush=True)
    print("DONE -> %s" % a.outdir, flush=True)


if __name__ == "__main__":
    main()
