"""render_iea22_segment.py -- render the generated IEA-22 r=0.2->0.3 segment meshes,
read back from the exported YAMLs: shell quad segment + structured solid HEX segment."""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
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


def draw(ax, nodes, cells, edges, color_of, stride, lw):
    for k in range(0, len(cells), stride):
        c = cells[k]
        col = color_of(k)
        for a, b in edges:
            ax.plot(nodes[[c[a], c[b]], 2], nodes[[c[a], c[b]], 0], nodes[[c[a], c[b]], 1],
                    color=col, lw=lw)


PAL = ["#666666", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#d62728"]
QE = [(0, 1), (1, 2), (2, 3), (3, 0)]
HE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

for kind, fname, edges, stride, lw in (
        ("shell", "iea22_seg_r020_r030_shell.yaml", QE, 1, 0.3),
        ("solid", "iea22_seg_r020_r030_solid.yaml", HE, 3, 0.2)):
    nodes, cells, setmap, names = load(os.path.join(HERE, fname))
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, nodes, cells, edges, lambda k: PAL[setmap[k] % len(PAL)], stride, lw)
    ax.set_title("IEA-22 segment r=0.2$\\rightarrow$0.3 -- %s mesh (%d nodes, %d %s)"
                 % (kind.upper(), len(nodes), len(cells), "quads" if kind == "shell" else "hexes"))
    ax.set_xlabel("span z [m]")
    ax.view_init(16, -75)
    try:
        ax.set_box_aspect((2.4, 1, 0.45))
    except Exception:
        pass
    fig.tight_layout()
    out = os.path.join(HERE, "iea22_seg_%s.png" % kind)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out, flush=True)
