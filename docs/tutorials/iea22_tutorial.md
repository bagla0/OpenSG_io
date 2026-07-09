# Worked example — IEA-22 3-D tapered segment (shell + solid hex)

This tutorial builds the **3-D tapered segment of the IEA-22-280 blade between the
r = 0.2 and r = 0.3 span stations** from the windIO definition bundled in this repo
([`examples/data/IEA-22-280-RWT.yaml`](https://github.com/bagla0/OpenSG_io/blob/main/examples/data/IEA-22-280-RWT.yaml)) —
producing **both** SG meshes:

* the **strictly structured 8-node HEX solid** — formed from the *layup information* of the
  two boundary cross-sections (through-thickness layers use each segment's local laminate
  thickness; every hex carries the fiber frame of the ply at its depth), and
* the **equivalent mid-surface QUAD shell segment** — the same hoop skeleton and span
  stations, so shell-vs-solid comparisons are one-to-one.

Every export passes the **mandatory conformity gate**.

```bash
pip install windIO          # windIO v2 reader
python examples/iea22_hex_segment.py
```

## How the general two-station hex loft works

The input each time is the *boundary cross-section definition at the two ends* (here
`build_cross_section(blade, r)` at r = 0.2 and 0.3; a PreVABS XML station resolves to the
same information). `opensg_io.hex_loft` then:

1. **Canonical hoop skeleton** — the union of layup-segment breakpoints and **web junction
   bands** (each web occupies a band of its own thickness on the skin, subdivided `nw`
   times = the junction refinement), *label-matched between the two stations* so both get
   the **identical topology**. Meshed independently the stations would differ (chord-driven
   node counts); the shared skeleton is what makes a structured loft possible for *any*
   pair of compatible sections.
2. **Per-station realization** — skin nodes on the OML offset inward through `nr` layers
   using the **local laminate thickness**; webs are plates whose across-thickness columns
   attach to the `nw+1` **inner-skin band nodes** (top/bottom rows *are* skin nodes → a
   watertight, conforming T-junction), with depth rows **cosine-clustered** so the mesh is
   refined at both web junctions.
3. **Linear loft** — one 8-node hex per quad per span slice, marching r = 0.2 → 0.3.
   A conforming quad section lofts to a conforming hex mesh by construction.

## Generation script

```{literalinclude} ../../examples/iea22_hex_segment.py
:language: python
:caption: examples/iea22_hex_segment.py
```

## Output

```text
stations: r=0.20 chord=7.200 (3 webs) | r=0.30 chord=6.703  ->  z=[27.40, 41.10] m
HEX: 10101 nodes, 7296 hexes  (section: 126 hoop nodes x 4 layers + webs NY=[12, 20, 20])
conformity gate (solid): PASS
wrote examples/iea22_seg_r020_r030_solid.yaml
conformity (shell, branched): PASS  (2275 nodes, 2136 quads; 72 T-junction edges as expected)
wrote examples/iea22_seg_r020_r030_shell.yaml
```

The loft input — the r = 0.2 quad cross-section with the three web junction bands — and the
lofted hex segment:

```{image} ../_static/iea22seg/iea22_hex_segment.png
:width: 100%
```

The two exported segment meshes, read back from their YAMLs (colored by element set):

```{image} ../_static/iea22seg/iea22_seg_shell.png
:width: 85%
```

```{image} ../_static/iea22seg/iea22_seg_solid.png
:width: 85%
```

## Notes

* **Strictly structured**: hoop skeleton × `nr` through-thickness layers × `nsp` span
  slices — no unstructured fill anywhere.
* The **shell conformity check differs from the solid one**: a branched mid-surface shell
  legitimately has T-junction edges shared by exactly **3** quads (skin-left + skin-right +
  web); the script asserts exactly `2 × n_webs × nsp` such edges and none shared by more.
* The two YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers
  (RM shell segment ↔ FEniCS 3-D solid) for the tapered Timoshenko 6×6.
* Refinement knobs: `mesh_size` (hoop), `nr` (through-thickness), `nsp` (span), `nw`
  (junction band). The thin trailing-edge region is the place to spend `mesh_size` if the
  downstream solve needs it.
