"""render_iea22_segment.py -- shaded renders of the generated IEA-22 r=0.2->0.3 segment
meshes, read back from the exported YAMLs: shell quad segment + structured solid HEX
segment.  Faces are shaded and element edges drawn (render3d), so the individual
hex/quad elements are visible; the solid's end caps expose the through-thickness
layers and the webs."""
import os
import sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from opensg_io.render3d import render_mesh_png

CLoad = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load(path):
    d = yaml.load(open(path), Loader=CLoad)
    nodes = np.array([[float(v) for v in r[0].split()] for r in d["nodes"]])
    cells = np.array([[int(v) - 1 for v in r[0].split()] for r in d["elements"]])
    setmap = np.zeros(len(cells), int)
    for si, s in enumerate(d["sets"]["element"]):
        for lab in s["labels"]:
            setmap[lab - 1] = si
    names = [s["name"] for s in d["sets"]["element"]]
    return nodes, cells, setmap, names


for kind, fname in (("shell", "iea22_seg_r020_r030_shell.yaml"),
                    ("solid", "iea22_seg_r020_r030_solid.yaml")):
    nodes, cells, setmap, names = load(os.path.join(HERE, fname))
    out = os.path.join(HERE, "iea22_seg_%s.png" % kind)
    title = ("IEA-22 segment r=0.2->0.3 -- %s mesh (%d nodes, %d %s)"
             % (kind.upper(), len(nodes), len(cells),
                "quads" if kind == "shell" else "hexes"))
    render_mesh_png(nodes, cells, "quad" if kind == "shell" else "hex",
                    setmap, out, title)
    print("wrote", out, flush=True)
