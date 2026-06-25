# Inputs & outputs

OpenSG_io has one CLI -- `scripts/opensg_io.py <input> <outdir>` -- that auto-detects the input by extension
and content, and routes it to the right adapter. Every adapter produces the same two OpenSG artifacts:

- **1D-shell SG YAML** (`shell_*.yaml`) -- nodes, line elements, element sets = laminates, sections (layups),
  materials, and per-element `e1/e2/e3` orientations. Consumed by the MSG shell homogenizers (RM / Kirchhoff).
- **2D-solid SG YAML** (`solid_*.yaml`) -- the meshed cross-section (nodes + elements + per-element material
  and fiber angle). Consumed by the FEniCS 2D-solid (VABS-equivalent) reference.

## windIO blade (`*.yaml`)

`load_blade()` auto-detects the ontology version:

- **v2** -- `components.blade.outer_shape` + `structure` (anchors / layers / webs). Example: IEA-22-280-RWT.
- **v1** -- `outer_shape_bem` + `internal_structure_2d_fem`. Example: the NREL BAR designs.

Per spanwise station `r` it resolves chord, twist, the blended airfoil, the layup stack at each arc segment,
and the webs, then writes the 1D-shell YAML and (with `--solid`) the PreVABS XML it runs to get the 2D-solid.
See {doc}`tutorials/windio`.

## PreVABS cross-section (`*.xml`)

A PreVABS XML already encodes a cross-section (airfoil baseline, dividing points, layups, surface segments,
webs). OpenSG_io:

- **reconstructs the 1D-shell** directly from the XML midline + layups (no PreVABS run needed), and
- **builds the 2D-solid** by running PreVABS on the XML (`.sg` -> `convert_sg_to_yaml`).

See {doc}`tutorials/prevabs_xml`.

## OpenFAST blade data (ElastoDyn / BeamDyn)

OpenFAST blade files carry the *already-homogenized* beam properties, **not** the layup -- so no Structure
Gene can be built from them. OpenSG_io reads them as a **validation reference**:

- **ElastoDyn** blade file -> `BlFract`, `FlpStff` (EI flap = EI2), `EdgStff` (EI edge = EI3), `BMassDen`.
- **BeamDyn** blade file -> the full 6x6 sectional stiffness, reordered to the OpenSG Timoshenko convention
  `[EA, GA2, GA3, GJ, EI2, EI3]`.

It can also **write** a BeamDyn blade file from an OpenSG-homogenized 6x6, to drive an OpenFAST run.
See {doc}`tutorials/openfast`.
