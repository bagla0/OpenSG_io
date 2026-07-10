"""Foam-G bisect variants: stiffen ONLY the membrane G12 or ONLY the transverse G13/G23
of the foam (to the skin's), leaving the rest untouched.  Usage:
    python _foam_variant.py <in.yaml> <out.yaml> {g12|g23|all}
Material G convention: G = [G12, G13, G23]."""
import sys
import yaml

CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.load(open(src), Loader=CL)


def props(m):
    return m["elastic"] if "elastic" in m else m


Gs = list(props(next(m for m in d["materials"] if m["name"] == "glass_triax"))["G"])
for m in d["materials"]:
    if "foam" in m["name"].lower():
        p = props(m)
        G = list(p["G"])
        if mode in ("g12", "all"):
            G[0] = Gs[0]
        if mode in ("g23", "all"):
            G[1], G[2] = Gs[1], Gs[2]
        p["G"] = G
        print("foam G ->", G)
yaml.safe_dump(d, open(dst, "w"), default_flow_style=None, sort_keys=False)
