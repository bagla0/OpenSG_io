"""Write a variant YAML with the foam SHEAR modulus raised to the skin's (glass_triax),
everything else unchanged -- the controlled test for the soft-core hypothesis."""
import sys
import yaml

CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
src, dst = sys.argv[1], sys.argv[2]
d = yaml.load(open(src), Loader=CL)


def props(m):
    return m["elastic"] if "elastic" in m else m


triax = next(m for m in d["materials"] if m["name"] == "glass_triax")
Gs, Es, nus = (list(props(triax)[k]) for k in ("G", "E", "nu"))
for m in d["materials"]:
    if "foam" in m["name"].lower():
        p = props(m)
        p["G"] = list(Gs)          # stiffen ONLY the shear (isolates transverse shear)
print("stiffened foam G ->", Gs, "in", dst)
yaml.safe_dump(d, open(dst, "w"), default_flow_style=None, sort_keys=False)
