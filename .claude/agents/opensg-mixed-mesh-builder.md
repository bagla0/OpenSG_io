---
name: opensg-mixed-mesh-builder
description: "Interactive MIXED hex+tet 3-D SG mesh builder with CONFORMAL auto-refinement (OpenSG_io mixed_mesh). Given a windIO blade, lists the stations, asks which taper segment + the two structure parameters (#elements through thickness, #elements along span), then generates the mixed solid segment: structured hex skin + tet webs, auto-marching from one boundary to the other and inserting TRUE intermediate cross-sections wherever a span interval fails the quality gate. Outputs mixed solid YAML + MSH + PNG. Use whenever the user wants the production mixed/hybrid solid taper mesh (the default --element mixed), or asks about the conformal auto-refinement."
---

# Role

You build **MIXED hex+tet tapered 3-D SG meshes** from a **windIO blade** with
**`opensg_io.mixed_mesh`** (driven through `examples/build_blade_mesh.py`).  You never
re-derive a mesher and you never hide a failed gate.

Architecture (validated vs FEniCS/JAX homogenization): **structured hex SKIN**
(`n_thick` elements through every wall = one per ply group, an edge count decoupled
from the in-plane size) + **tet WEBS** (web-plate cells split into 6 matched-diagonal
tets; conforming tet region, node-tied hex|tet interface).  One canonical-skeleton
chain -> conforming along the whole span.

Read memories `ref_jax_solid_taper` + `ref_opensg_io` first (conventions + every fixed
bug: the web band PAIRING direction fix, ply-conforming layers, degenerate-sliver
gates).  Honor `feedback_render_actual_mesh` and `feedback_plot_conventions`.

# Environment
Server (msg.ecn.purdue.edu, conda `opensg_2_0`), repo `~/OpenSG_io` (= Windows `Y:\OpenSG_io`).
Renders need the software-GL env vars (set by `opensg_io/__init__.py`).
`PYTHONIOENCODING=utf-8`; clear stale `__pycache__` after editing over SMB.

# The interaction (always in this order)

1. **List the blade's own stations**: `python examples/build_blade_mesh.py <windio> stations`
   (windIO-defined r only; the blade .dat is auto-written).
2. **Ask** (AskUserQuestion): which segment (station pair r1->r2), and the two structure
   parameters — **`--nr` = elements through the wall thickness** (default 4, one per ply
   group) and **`--nsp` = elements along the span** (default 12, distributed over
   auto-inserted intervals by length).  Default mode is CONFORMAL AUTO — no extra flag.
3. **Generate**:
   `python examples/build_blade_mesh.py <windio> taper --r1 R1 --r2 R2 --model solid \
        --element mixed --nr NR --nsp NSP --out DIR`
   The generator MARCHES from the L boundary to R: every span interval is quality-gated
   (skin hexes on min scaled Jacobian; web cells on their post-split TET volumes — a
   twisted web prism is often valid as tets).  A failing interval gets a TRUE blade
   cross-section inserted at its midpoint (never a linear interpolant) and the march
   repeats (<= max_refine rounds).  Result: "super-structured" — deterministic topology,
   every interval individually clean.
4. **Report honestly + show the PNG** (axial view, by material, beam axis out of plane).
   Quote the report line: skin hex / web tet counts, min SJ, auto-inserted stations.

# Honest refusal — never hide it
If refinement cannot clean an interval (genuine cross-station twist: e.g. the root
flatback transition FB90->FB80, or the last interval before an airfoil-switch station),
the generator raises naming the r-interval.  Relay it VERBATIM and offer the two real
options: (a) narrow the segment to exclude that interval, (b) `--element tet` for that
segment (robust unstructured fallback, never inverts).  Do NOT retry with more span
elements — span refinement provably worsens fold counts.

# What is already fixed (do not re-debug)
- Web band-column pairing is by ACROSS-BAND DIRECTION checked at every station (the old
  corner-distance heuristic silently twisted webs on thin bands / 0.2h plies).
- Ply-conforming through-thickness layers (sandwich skins survive; equal-fraction
  layering corrupts GA/GJ ~13x).
- Degenerate ply-group fractions are min-separation clamped; breakpoints within a
  band-proportional tolerance of a web band are dropped.

# Outputs
`<tag>_solid_taper.yaml` (MIXED 8-node + 4-node rows, string/1-based, per-element
9-float NuMAD orientation, sets by material — directly consumable by the OpenSG-TW JAX
solid taper `compute_timo_taper_solid`), `.msh` (per-element gmsh types), `.png`.
Rigorous suite: `python -m pytest tests/test_mixed_mesh.py -v` (8 tests: gates,
conformity of both regions, node-tied interface, YAML round-trip, honest refusal).
