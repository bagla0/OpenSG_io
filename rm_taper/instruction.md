# RM-Taper — reproducible mesh/result package

Supporting data for the paper **"Timoshenko Beam Modeling of Tapered Thin-Walled Composite
Structures Using the Reissner–Mindlin Model"** (Overleaf project `6a4be73fce5e3435a9333cf8`).
It validates the RM-shell equivalent-beam Timoshenko stiffness `C^b` (6×6) against a
**conforming 3-D solid** reference for three tapered geometries at two wall thicknesses.

## Contents

```
rm_taper/
  meshes/    <case>_solid.yaml     OpenSG 3-D solid mesh (structured 8-node hex)
             <case>_boundary.yaml  root (z=min) cross-section = boundary homogenization mesh
  figures/   <case>_solid_3d.png   clipped 3-D segment (shows webs)
             <case>_solid_xsec.png through-thickness cross-section
             <case>_boundary.png   boundary cross-section mesh
  instruction.md (this file)
```

`<case>` ∈ {circle, square, ellipse} × {thin, thick}.

## Geometry (all cases)

- Single `[-45°]` ply, orthotropic: E = [37, 9, 9] GPa, G = [4, 4, 4] GPa, ν = 0.3, ρ = 1800.
- Taper over span L = 2.0 m. Circle/square: R 1.0→0.65. Ellipse: a 1.0→0.65, b 0.60→0.42.
- Ellipse has **3 vertical shear webs** at x = c·a(z), c ∈ {−0.5, 0, +0.5} (multi-cell).
- Wall thickness: **thin t/R = 0.02**, **thick t/R = 0.20**. Centre (mid-surface) reference.

## Key results — conforming hex solid vs RM shell (diagonal `C^b`, ×10⁹)

Webbed ellipse (the demanding multi-cell case):

| t/R  | EA (solid/shell) | GA₂ | GA₃ | GJ | EI₂ | EI₃ |
|------|------------------|-----|-----|----|-----|-----|
| 0.02 | 1.63 / 1.658 (+1.7%) | 0.305/0.316 | 0.528/0.562 | 0.233/0.235 | 0.187/0.191 | 0.369/0.375 |
| 0.20 | 14.62 / 16.56 (+13.3%) | 3.97/3.82 | 5.00/5.63 | 2.47/2.50 | 1.76/1.98 | 3.51/3.78 |

**Thickness sweep — RM shell %err vs conforming solid (webbed ellipse):**

| t/R | EA | GA₂ | GA₃ | GJ | EI₂ | EI₃ |
|-----|----|-----|-----|----|-----|-----|
| 0.02 | +1.7 | +3.5 | +6.5 | +0.7 | +2.5 | +1.6 |
| 0.10 | +6.8 | +0.1 | +12.6 | +0.6 | +7.8 | +4.9 |
| 0.20 | +13.4 | −4.5 | +13.9 | +1.4 | +12.7 | +7.9 |

Circle/square (no webs) match the shell within a few percent at both thicknesses.

## Pipeline (how to regenerate)

### 1. Meshes — OpenSG_io
- **Structured HEX (preferred, resolves transverse shear):** `opensg_io.mesh3d.loft_to_hex` —
  loft a 2-D **quad** cross-section along the span (each quad → one 8-node hex per slice).
  Webbed ellipse: `opensg_io.mesh3d.webbed_ellipse_hex(t, nr, nsp, nw, nct)` builds the quad
  cross-section (skin annulus + web plates; web top/bottom rows ARE inner-skin nodes; constant
  web thickness under taper via per-station skin reparam) and lofts it.
- Circle/square: `gmsh_loft_struct.py <geom> <t> <npl> <nsp> <nc> hex` (transfinite → recombine).
- **Unstructured TET (robust, but P1 tets under-read shear ~25%):** `opensg_io.tapered_tet`
  (gmsh lofts the boundary surface → TetGen fills). Only for reference / arbitrary boundaries.
- **PreVABS 2-D cross-section** (`opensg_io.prevabs_webbed_ellipse`): emits ellipse `.dat` + XML,
  runs PreVABS 2.1 (vendored `third_party/prevabs`) to mesh/validate the cross-section.

### 2. Solid homogenization — FEniCS (`opensg-FEniCS`)
```python
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
sm = SolidSegmentMesh("meshes/ellipse_thick_solid.yaml")
mp, _ = sm.material_database
S = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0]  # 6×6
```
Element orientation per element: skin = ellipse tangent/normal + `[-45]` tilt; web = vertical.

### 3. RM shell 6×6 — mitc_rm_segment
`python run_shell_tR.py <T>`  (sets `run_ell3w.T`, generates the branched shell mesh via
`gen_ell3w`, solves with `run_indep.shell_solve_lagrange` — **dense** solver; the sparse one is
broken on webbed T-junctions). Full-integration segment is the paper default (MITC degenerates
by drilling-aliasing on the flat square webs).

### 4. Conformity gate — always
`opensg_io.conformity.assert_conforming(nodes, cells, celltype)` — refuses any mesh with a
hanging node / non-manifold interface. Run before every export (`export_solid_yaml` gates on it).

## Key findings

1. **Conformity matters.** The previous parametric webbed hex (`gen_ell3w`) was NON-conforming
   (124 hanging nodes at the web–skin T-junctions); its overlap spuriously over-counted stiffness
   and *masked* the shell's error. All meshes here pass the conformity gate.
2. **Shell over-predicts the webbed section ∝ t/R** (T-junction double-count: the shell carries
   each web mid-line × thickness t, overlapping the skin's own t). EA/EI: +1.7% (thin) → +13%
   (thick). In-plane shear GA₂ and torsion GJ stay within a few % at all thicknesses.
3. **Structured hex captures transverse shear; P1 tet does not** (constant-strain tets under-read
   the tapered shear gradient ~25%). Hence the hex loft is the reference, not TetGen.

## Caveats / where this was left off

- **Coupling sign:** the hex frame reproduces the diagonal and all couplings EXCEPT the
  extension-shear C₁₂/C₁₃ (A₁₆-type), whose sign flips vs the shell (a frame/ply convention that
  gen_ell3w and the shell agree on). The paper table reconciles just those two signs. A clean
  fix = match `emit_webhex_solid`'s fiber-tilt convention to `run_ell3w`'s (open).
- **Boundary meshes** here are the root (z=min) cross-section extracted from each solid; the paper's
  boundary tables use the RM ring + solid boundary homogenization (`run_ringboun.py`).
- **Thick C₁₃ magnitude** runs ~38% high (small coupling, order of magnitude below the diagonal).
- **Next:** reconcile the coupling frame convention; optionally regenerate circle/square with
  `loft_to_hex` for a uniform pipeline; add a t/R = 0.05 point to sharpen the thin-end trend.

## Provenance
OpenSG_io commits: conformity gate `a0f3d21`, tapered_tet+PreVABS `37e83d8`, loft_to_hex+gmsh
`39c3db0`. Paper (Overleaf) `68a5c8d`.
