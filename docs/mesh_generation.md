# How the tapered SG mesh is generated (core walkthrough)

OpenSG_io builds the 3-D **solid (hex)** and equivalent **shell (quad)** Structure-Gene
meshes of a blade segment by a **two-station structured loft** — *not* by meshing a fixed
3-D boundary and filling it, and *not* with PreVABS. PreVABS is a separate, optional
*input* path (`opensg_io.prevabs_xml`) for reading an existing XML cross-section; it is
**not** used in the windIO taper pipeline below.

## The two inputs are cross-section *definitions*, not meshes

```python
cs1 = build_cross_section(blade, r1)     # section DEFINITION at span r1
cs2 = build_cross_section(blade, r2)     # section DEFINITION at span r2
res = hex_between_sections(cs1, cs2, z1, z2, ...)   # LOFT between them
```

`build_cross_section(blade, r)` (`opensg_io.converter`) reads the windIO blade at span
`r` and returns the 2-D **section definition** — it does **not** mesh anything:

- `xy`, `s_arc` — the OML (outer-mold-line) airfoil contour and its arc-length,
- `segments` — the chordwise layup regions (each with a `set_id` → ordered ply list),
- `laminates` — `{ply-tuple → set_id}` (each ply = `(material, thickness, angle)`),
- `webs` — the shear-web attachment arcs and their laminates,
- `chord`, `perim`, plus the material table.

So the pipeline starts from **two boundary section definitions** and **lofts** the 3-D
element mesh between them. `z1, z2 = blade_span_z(blade, r)` place the two sections along
the beam axis.

## The loft: `hex_between_sections`

```
hex_between_sections(cs1, cs2, z1, z2, nr, nsp, nw)
├─ section_skeleton([cs1, cs2])        # ONE canonical hoop topology for both ends
├─ build_section_mesh([cs1, cs2], skel)# realize the 2-D quad ring at each station
├─ loft: connect quad(s) → quad(s+1)   # 8-node hexes, nsp span slices
└─ _repair_inverted                    # per-hex positive-Jacobian fix
```

### `section_skeleton(cs_list, mesh_size, nw)`
Builds **one canonical hoop skeleton** shared by both stations, so the two ends have the
**identical node topology** (a prerequisite for a structured loft):
- `_station_breaks(cs)` — the labelled hoop breakpoints of a station: layup-segment
  boundaries **plus** a *web-junction band* (a strip of the web's own thickness) around
  each web attachment; a segment breakpoint that falls inside a band is dropped.
- The breakpoints of the two stations are **matched by label**; each hoop interval gets a
  fixed subdivision `counts[k]` (skin: `ceil(arc-length / mesh_size·chord)`; web band:
  `nw`), and a `kind` (`('skin', set_id)` or `('band', web, side)`). Raises if the two
  stations are not topology-compatible.

### `build_section_mesh(cs_list, skel, nr)`
Turns the skeleton into the **shared 2-D quad topology + per-station coordinates**:
- `region_taper_laminates(cs1, cs2, skel)` — per hoop region, a `TaperLaminate` aligning
  the two stations' ply stacks by identity (LCS on material+angle) with linear thickness
  interpolation and ply-drop = ramp-to-zero (`opensg_io.layup`). Matched by **label**,
  not arc position (so the spar cap is not mislabelled a panel).
- **ply-conforming layer fractions** — `TaperLaminate.group_cuts / group_fractions`
  group each region's laminate into exactly `nr` contiguous ply groups with cuts **at ply
  boundaries**, so a through-thickness hex layer *is* a ply group (a 3 mm sandwich skin on
  a 76 mm wall survives instead of being sampled away by an equal 19 mm layer).
- `build_station(cs, skel, si, nr, frac_int)` — realizes the skeleton at station `si`:
  - `_contour_pt(cs, s)` samples the OML at each hoop arc position,
  - the local wall thickness per hoop node = mean of the two adjacent regions' laminate
    thickness (NuMAD's node-averaged offset),
  - `open_thin_gaps(oml, tnode)` — NuMAD-style trailing-edge *opening* so the full
    laminate fits at a pinched TE (`opensg_io.section_offset`),
  - `offset_rings(oml, tnode, nr, fracs)` — the through-thickness rings by a
    **PreVABS-style miter offset**: signed-area orientation, angle-bisector normals with
    the Clipper2 miter limit, an inward-ray-cast **wall-clearance clamp** so rings never
    cross the opposite wall, and a **hard fold verifier** on the built rings.
  - webs are plates whose across-thickness columns attach to the inner-skin band nodes
    (a watertight T-junction) with cosine-clustered depth rows (refined at both junctions).
- The 2-D faces are **CCW-wound** (signed area at station 0) so the `+z` extrusion is
  right-handed; a per-face **inward surface normal** `fn2d` and **region key** are stored.

### Loft + repair
The shared 2-D quad set is stacked at `nsp+1` span stations (linear in `z`) and each quad
at slice `s` is connected to the *same* quad at `s+1` → one 8-node hex per quad per slice.
`_repair_inverted` applies NuMAD's determinant check (`NuMesh3D.m`): a hex the span loft
flipped is un-inverted by swapping its bottom/top faces (same nodes → conformity intact).

## Materials + orientation: `solid_yaml_payload(res, cs1, cs2)`
Per hex: the ply-group's **exact material** (`TaperLaminate.group_material`), and a
NuMAD/VABS **element frame** (`opensg_io.orientation.element_frame`): `e1` = beam axis
root→tip, `e3` = inward surface normal, `e2 = e3 × e1`, with the ply fiber angle rotated
about `e3`. `export_solid_yaml` writes it only after the conformity + Jacobian gates pass.

## The shell is the *same* loft on a surface: `shell_between_sections(res, cs1, cs2)`
It reuses the **same `hex_between_sections` result** (skeleton + stations), so shell and
solid are node-consistent. The skin sits on the **OML by default** (coincident with the
solid's outer hex ring; pair with an OML-referenced ABD, `frac=0`), each web is a strip of
mid-columns whose top/bottom rows are skin nodes (a branched T-junction), and every span
bay carries its own **span-interpolated layup** so the taper's stiffness change lives in
the shell layup. Boundary cross-sections (`solid_boundary_payload`,
`shell_boundary_payload`) are the loft's end stations exported in the FEniCS boundary
formats.

## One-line answers
- **PreVABS?** No — `build_cross_section` reads the windIO layup directly.
- **Loft or fill-a-boundary-mesh?** A **two-station structured loft**: build the section
  definition at `r1` and `r2`, realize a shared 2-D quad ring at each, connect ring→ring
  into hexes across `nsp` span slices.
- **Shell too?** Yes — same skeleton/stations, lofted on the mid/OML surface.
