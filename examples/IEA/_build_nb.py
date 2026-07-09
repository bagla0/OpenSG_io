"""_build_nb.py -- build + EXECUTE the IEA-22 segment tutorial notebook.

Matches the OpenSG-TW tutorials: the notebook runs the standalone example
(examples/IEA/iea22_segment.py) inline, prints its gate output, and embeds the mesh
renders as stored cell outputs (myst-nb nb_execution_mode="off" renders them straight
from a fresh clone).  Writes docs/tutorials/iea22_tutorial.ipynb.

Run in an env with windIO + opensg_io + matplotlib + pyvista + nbformat/nbconvert/ipykernel:
    python examples/IEA/_build_nb.py
"""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TUT = os.path.join(REPO, "docs", "tutorials")

INTRO = r"""# IEA-22 — 3-D tapered segment (shell + structured-hex solid)

This tutorial builds the **3-D tapered segment of the IEA-22-280 blade between the
`r = 0.2` and `r = 0.3` span stations**, straight from the windIO definition bundled in
this repo (`examples/data/IEA-22-280-RWT.yaml`), producing **both** SG meshes:

* the **strictly structured 8-node HEX solid** — formed from the *layup information* of the
  two boundary cross-sections (through-thickness layers from each segment's local laminate;
  every hex carries the fiber frame of the ply at its depth), and
* the **equivalent mid-surface QUAD shell segment** — the same hoop skeleton and span
  stations, so shell-vs-solid comparisons are one-to-one.

Every export passes the **conformity gate** (watertight, faces shared by exactly two
cells) and, for the solid, a **positive min-scaled-Jacobian** check. The only external
requirement is the windIO v2 reader:

```bash
pip install windIO
```

The whole example is standalone — it runs `examples/IEA/iea22_segment.py`, which writes
its meshes and renders to `examples/IEA/output/`."""

HOWTO = r"""## How the general two-station hex loft works

The input each time is the *boundary cross-section definition at the two ends* (here
`build_cross_section(blade, r)` at `r = 0.2` and `0.3`; a PreVABS XML station resolves to the
same information). `opensg_io.hex_loft` then:

1. **Canonical hoop skeleton** — the union of layup-segment breakpoints and **web junction
   bands**, *label-matched between the two stations* so both get the **identical topology**;
   this is what makes a structured loft possible for *any* pair of compatible sections.
2. **Per-station realization** — the skin offsets inward through `nr` layers using the
   **local laminate thickness**, and the webs attach to the inner-skin band nodes (a
   watertight T-junction). The offset uses the PreVABS/NuMAD recipe — signed-area
   orientation, angle-bisector (miter) normals, a thin-gap clamp, and a **full-accuracy
   trailing-edge opening** that preserves the nominal laminate rather than thinning plies.
3. **Linear loft** — one 8-node hex per quad per span slice, with every 2-D face wound CCW
   so all hexes are right-handed (positive Jacobian).

The equivalent shell is the same skeleton taken on the wall mid-surface."""

SETUP = r'''import os, sys, runpy
def _root(d):
    d = os.path.abspath(d)
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "opensg_io")) and os.path.isdir(os.path.join(d, "examples")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()
ROOT = _root(os.getcwd())
EX = os.path.join(ROOT, "examples", "IEA")
OUT = os.path.join(EX, "output")
sys.path.insert(0, ROOT)
from IPython.display import Image, display
print("repo root:", ROOT)'''

BUILD = r'''# run the standalone example inline (default input = the bundled windIO)
sys.argv = ["iea22_segment.py"]
runpy.run_path(os.path.join(EX, "iea22_segment.py"), run_name="__main__")'''

SECTIONS = r'''# cross-sections at BOTH ends -- solid (through-thickness layers + webs) and shell
display(Image(filename=os.path.join(OUT, "iea22_sections.png")))'''

SOLID3D = r'''# the structured hex solid, shaded faces + element edges (colored by material)
display(Image(filename=os.path.join(OUT, "iea22_solid_3d.png")))'''

SHELL3D = r'''# the equivalent mid-surface shell (colored by layup)
display(Image(filename=os.path.join(OUT, "iea22_shell_3d.png")))'''

PEEK = r'''# the exported YAML is an OpenSG shell SG file -- show its STRUCTURE, not the mesh dump
import yaml
with open(os.path.join(OUT, "iea22_seg_shell.yaml")) as f:
    d = yaml.safe_load(f)
print("shell SG YAML  (keys: %s)" % ", ".join(d))
print("  nodes    : %d" % len(d["nodes"]))
print("  elements : %d" % len(d["elements"]))
print("  sets     : %d element sets (%s ...)" % (len(d["sets"]["element"]),
      ", ".join(s["name"] for s in d["sets"]["element"][:4])))
print("  sections : %d layups; e.g. %s -> %d plies" % (len(d["sections"]),
      d["sections"][0]["elementSet"], len(d["sections"][0]["layup"])))
print("  materials: %s" % ", ".join(m["name"] for m in d["materials"]))'''

NOTES = r"""## Notes

* **Strictly structured**: hoop skeleton × `nr` through-thickness layers × `nsp` span slices
  — no unstructured fill anywhere.
* The **shell conformity check differs from the solid one**: a branched mid-surface shell
  legitimately has T-junction edges shared by exactly **3** quads (skin-left + skin-right +
  web); the check asserts exactly `2 × n_webs × nsp` such edges and none shared by more.
* The two YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers
  (RM shell segment ↔ FEniCS 3-D solid) for the tapered Timoshenko 6×6.
* Refinement knobs (`iea22_segment.py`): `MESH` (hoop), `NR` (through-thickness), `NSP`
  (span), `NW` (junction band). Change the station radii with
  `python examples/IEA/iea22_segment.py <windio.yaml> <r1> <r2>`."""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(INTRO),
    nbf.v4.new_markdown_cell(HOWTO),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell("## Build the segment"),
    nbf.v4.new_code_cell(BUILD),
    nbf.v4.new_markdown_cell("## Cross-sections at both ends"),
    nbf.v4.new_code_cell(SECTIONS),
    nbf.v4.new_markdown_cell("## The structured hex solid"),
    nbf.v4.new_code_cell(SOLID3D),
    nbf.v4.new_markdown_cell("## The equivalent mid-surface shell"),
    nbf.v4.new_code_cell(SHELL3D),
    nbf.v4.new_markdown_cell("## The exported SG YAML"),
    nbf.v4.new_code_cell(PEEK),
    nbf.v4.new_markdown_cell(NOTES),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
print("executing iea22_tutorial ...", flush=True)
ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": REPO}})
out = os.path.join(TUT, "iea22_tutorial.ipynb")
nbf.write(nb, out)
print("wrote", out, flush=True)
