# Mixed hex+tet taper mesh (conformal auto-refinement)

The **production solid-taper mesher**: `opensg_io.mixed_mesh` builds a tapered 3-D SG
between two blade stations as **structured hex skin + tet webs**, with a **conformal
auto-refinement march** — and it is the default (`--element mixed`) of the taper CLI.

## Architecture

* **Skin — structured hex8.** `n_thick` elements through *every* wall, one per **ply
  group** (the layer interfaces follow ply boundaries, so sandwich skins are meshed
  exactly).  `n_thick` is an *edge count*, decoupled from the in-plane mesh size — no
  isotropic element explosion on thin walls.
* **Webs — tet4.**  Each web-plate cell is split into 6 tets by the main-diagonal
  scheme; the implied face diagonals match between neighbours, so the tet region is
  conforming, and the hex|tet interface is node-tied (the standard hex-dominant
  transition).  A twisted web prism that would fail the trilinear-hex Jacobian is
  usually a *valid solid region as tets* — the quality gate accounts for this.
* **One canonical skeleton** spans the whole station chain, so every station plane has
  identical ring topology and the mesh is conforming along the span.

This is the mesh the OpenSG-TW **JAX solid taper** consumes directly (hex + tet
element batches; mixed quad+tri boundary sections extracted from the segment), with the
architecture validated against the FEniCS solid to < 1 %.

## Conformal auto-refinement (default)

The generator **marches from the L boundary to the R boundary**.  The chain starts as
`[r1, r2]`; every span interval is quality-gated:

* skin hexes → minimum scaled corner-Jacobian must stay positive;
* web cells → their **post-split tet volumes** must all be positive.

Where the gate fails — thin, complex, or strongly-morphing regions — a **true blade
cross-section** (never a linear interpolant) is inserted at the interval midpoint,
halving the shape morph, and the march repeats (up to `max_refine` rounds).  The result
is *super-structured*: deterministic topology, every interval individually clean —
**or an honest refusal** naming the interval that cannot be repaired (genuine
cross-station twist, e.g. the root flatback transition), with the robust `--element
tet` fallback suggested.  Span refinement is *never* used to mask a fold (it provably
makes folds worse).

## Python API

```python
from opensg_io.converter import load_blade
from opensg_io.mixed_mesh import mixed_taper_mesh, write_mixed_yaml, render_mixed_png

blade = load_blade("examples/data/IEA-22-280-RWT.yaml")
mesh  = mixed_taper_mesh(blade, 0.2, 0.3,
                         n_thick=4,      # elements through the wall (one per ply group)
                         n_span=12,      # span elements (distributed over intervals)
                         nw=3, mesh_size=0.02)
print(mesh["report"])
# {'stations': [0.2, 0.3], 'rounds': 1, 'n_hex': 6336, 'n_tet': 11880,
#  'min_sj_hex': 0.423, 'n_neg_tet': 0, ...}

write_mixed_yaml("r020_030_solid_taper.yaml", mesh)   # mixed 8-node + 4-node rows
render_mixed_png("r020_030_mixed.png", mesh)          # by material, beam axis out of plane
```

## CLI

```bash
python examples/build_blade_mesh.py examples/data/IEA-22-280-RWT.yaml taper \
    --r1 0.2 --r2 0.3 --model solid --element mixed --nr 4 --nsp 12 --out mesh_out
#  [mixed] round 0: stations r=['0.2000', '0.3000']  intervals=1  skin minSJ=+0.423  neg web tets=0  bad=[]
#  solid (MIXED): 6336 skin hex + 11880 web tet ; min SJ 0.423 ; 0 station(s) auto-inserted
```

Outputs: the mixed solid YAML (string/1-based, per-element 9-float NuMAD orientation,
sets by material), a `.msh` (per-element gmsh types), and the mesh PNG.

## Interacting through the agent

The repo ships a Claude Code agent, **`opensg-mixed-mesh-builder`**, that drives this
capability end-to-end.  A session looks like:

1. *you*: "mesh the IEA blade between two stations as a mixed solid" → the agent lists
   the blade's own windIO stations (and writes the blade `.dat` summary);
2. the agent asks which **segment** (r₁ → r₂) and the two structure parameters
   (**#elements through thickness**, **#elements along span**) — conformal auto mode is
   the default, no extra flag;
3. it generates, then shows the **axial-view render** (beam axis out of the plane,
   colored by material) and quotes the gate report verbatim;
4. if an interval is genuinely twisted, it relays the refusal and offers the two real
   options: narrow the segment, or `--element tet` for that span.

## Guarantees and honest limits

| gate | guarantee |
|---|---|
| skin hex scaled-Jacobian > 0 | no inverted structured cells |
| web tet volumes > 0 | no folded web tets |
| single skeleton chain | conforming station-to-station topology |
| hex-region / tet-region conformity | verified in `tests/test_mixed_mesh.py` |
| refusal names the r-interval | genuine twist is never silently meshed |

Known honest limits: the **root flatback transition** (FB90→FB80) twists skin cells at
every subdivision scale, and the last interval before an **airfoil-switch station**
can stay twisted — both are refused with the `tet` fallback suggested.  Layup
interpolation (ply drops) stays linear end-to-end between r₁ and r₂; auto-inserted
stations refine geometry only.

Rigorous suite: `python -m pytest tests/test_mixed_mesh.py -v` — 8 tests covering the
gates, conformity of both element regions, the node-tied interface, unit orientation
frames, the YAML round-trip, and the honest-refusal path.
