"""_build_iea22_nb.py -- build + EXECUTE the IEA-22 3-D-segment tutorial notebook.

Mirrors the OpenSG-TW RM_taper tutorials: the notebook runs the committed example
scripts inline, prints their output, and embeds the mesh renders as stored cell
outputs, so the Sphinx site (myst-nb, nb_execution_mode="off") shows text + images
straight from a fresh clone.  Writes docs/tutorials/iea22_tutorial.ipynb.

Run on the server (needs windIO + opensg_io + matplotlib + nbformat + nbconvert + ipykernel):
    python examples/_build_iea22_nb.py
"""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TUT = os.path.join(ROOT, "docs", "tutorials")

INTRO = r"""# IEA-22 — 3-D tapered segment (shell + structured-hex solid)

This tutorial builds the **3-D tapered segment of the IEA-22-280 blade between the
`r = 0.2` and `r = 0.3` span stations**, straight from the windIO definition bundled in
this repo (`examples/data/IEA-22-280-RWT.yaml`), producing **both** SG meshes:

* the **strictly structured 8-node HEX solid** — formed from the *layup information* of the
  two boundary cross-sections (through-thickness layers use each segment's local laminate
  thickness; every hex carries the fiber frame of the ply at its depth), and
* the **equivalent mid-surface QUAD shell segment** — the same hoop skeleton and span
  stations, so shell-vs-solid comparisons are one-to-one.

Every export passes the **mandatory conformity gate**. The only external requirement is the
windIO v2 reader:

```bash
pip install windIO
```

This notebook runs `examples/iea22_hex_segment.py` (generation) and
`examples/render_iea22_segment.py` (read-back renders) inline."""

HOWTO = r"""## How the general two-station hex loft works

The input each time is the *boundary cross-section definition at the two ends* (here
`build_cross_section(blade, r)` at `r = 0.2` and `0.3`; a PreVABS XML station resolves to the
same information). `opensg_io.hex_loft` then:

1. **Canonical hoop skeleton** — the union of layup-segment breakpoints and **web junction
   bands** (each web occupies a band of its own thickness on the skin, subdivided `nw` times =
   the junction refinement), *label-matched between the two stations* so both get the
   **identical topology**. Meshed independently the stations would differ (chord-driven node
   counts); the shared skeleton is what makes a structured loft possible for *any* pair of
   compatible sections.
2. **Per-station realization** — skin nodes on the OML offset inward through `nr` layers using
   the **local laminate thickness**; webs are plates whose across-thickness columns attach to
   the `nw+1` **inner-skin band nodes** (top/bottom rows *are* skin nodes → a watertight,
   conforming T-junction), with depth rows **cosine-clustered** so the mesh is refined at both
   web junctions.
3. **Linear loft** — one 8-node hex per quad per span slice, marching `r = 0.2 → 0.3`. A
   conforming quad section lofts to a conforming hex mesh by construction.

Two robustness devices adopted from the PreVABS / NuMAD mesh generators:

* **Miter-offset rings** (`opensg_io.section_offset`, after PreVABS `src/geo/offset*.cpp`):
  contour orientation from the signed area, per-vertex angle-bisector normals with the
  Clipper2 miter limit, and a smoothed **thin-gap clamp** (inward ray-cast to the opposite
  wall) so the through-thickness rings never fold or cross at the thin trailing edge.
* **Positive-Jacobian guarantee** (after NuMAD `NuMesh3D.m` det check): every 2-D face is
  canonically CCW-wound before extrusion, and the export asserts the **min scaled corner
  Jacobian** of all hexes is positive."""

SETUP = r'''import os, sys, runpy
def _root(d):
    d = os.path.abspath(d)
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "opensg_io")) and os.path.isdir(os.path.join(d, "examples")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()
ROOT = _root(os.getcwd())
EX = os.path.join(ROOT, "examples")
sys.path.insert(0, ROOT)
from IPython.display import Image, display
print("repo root:", ROOT)'''

BUILD = r'''# run the generation script inline (default input = the bundled windIO)
sys.argv = ["iea22_hex_segment.py"]
runpy.run_path(os.path.join(EX, "iea22_hex_segment.py"), run_name="__main__")'''

SHOW_HEX = r'''# the loft input (r=0.2 section with trailing-edge zoom: fold-free miter offset)
display(Image(filename=os.path.join(EX, "iea22_hex_segment.png")))
# the lofted structured hex, shaded faces + element edges (colored by material)
display(Image(filename=os.path.join(EX, "iea22_hex_3d.png")))'''

RENDER = r'''# read the two exported YAMLs back and render each mesh (colored by element set)
sys.argv = ["render_iea22_segment.py"]
runpy.run_path(os.path.join(EX, "render_iea22_segment.py"), run_name="__main__")
display(Image(filename=os.path.join(EX, "iea22_seg_shell.png")))
display(Image(filename=os.path.join(EX, "iea22_seg_solid.png")))'''

NOTES = r"""## Notes

* **Strictly structured**: hoop skeleton × `nr` through-thickness layers × `nsp` span slices —
  no unstructured fill anywhere.
* The **shell conformity check differs from the solid one**: a branched mid-surface shell
  legitimately has T-junction edges shared by exactly **3** quads (skin-left + skin-right +
  web); the script asserts exactly `2 × n_webs × nsp` such edges and none shared by more.
* The two YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers
  (RM shell segment ↔ FEniCS 3-D solid) for the tapered Timoshenko 6×6.
* Refinement knobs: `mesh_size` (hoop), `nr` (through-thickness), `nsp` (span), `nw` (junction
  band). The thin trailing-edge region is where to spend `mesh_size` if the downstream solve
  needs it."""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(INTRO),
    nbf.v4.new_markdown_cell(HOWTO),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell("## Build the segment"),
    nbf.v4.new_code_cell(BUILD),
    nbf.v4.new_markdown_cell("## The loft input and the structured hex"),
    nbf.v4.new_code_cell(SHOW_HEX),
    nbf.v4.new_markdown_cell("## The two exported segment meshes (read back from the YAMLs)"),
    nbf.v4.new_code_cell(RENDER),
    nbf.v4.new_markdown_cell(NOTES),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
print("executing iea22_tutorial ...", flush=True)
ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": ROOT}})
out = os.path.join(TUT, "iea22_tutorial.ipynb")
nbf.write(nb, out)
print("wrote", out, flush=True)
