# WindIO_to_OpenSG

Convert a [**windIO v2**](https://github.com/IEAWindSystems/windIO) wind-turbine blade definition into
**OpenSG** cross-section inputs at any spanwise station, for both the **1D-shell** MSG homogenizers and the
**2D-solid** (VABS / FEniCS) reference.

```
                          ┌────────────────────────────────────────────────┐
   windIO v2 blade  ─────▶│  windio_to_opensg  (reads outer_shape+structure) │
   (e.g. IEA-22-280)      └───────────────┬──────────────────┬───────────────┘
                                          │                  │
                       1D-shell SG YAML ◀─┘                  └─▶ PreVABS XML + .dat + materials.xml
                       (nodes, line elems,                       │  prevabs.exe -i xsec.xml --vabs --hm
                        sets, layups, mats)                      ▼
                              │                              xsec.sg
              ┌───────────────┴───────────────┐                  │ convert_sg_to_yaml
              ▼                               ▼                  ▼
        JAX MSG-RM                      JAX MSG-Kirchhoff   2D-solid SG YAML ──▶ FEniCS solid (VABS-equiv)
       (Timoshenko 6×6)                (Timoshenko 6×6)
```

The 1D-shell YAML and the 2D-solid YAML feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW)
homogenizers so a single windIO blade can be cross-checked shell-vs-solid at every station.

## Why PreVABS for the solid mesh

We evaluated both. **PreVABS** is the chosen 2D-solid mesher because it is a self-contained, headless CLI
that emits a VABS `.sg` with per-element material + layup orientation and matches VABS to ~1e-5.

## Install

```bash
pip install windIO numpy pyyaml          # converter dependencies
# PreVABS (GPL v2) is vendored as a submodule + fetched binary -- see below
git clone --recurse-submodules https://github.com/bagla0/WindIO_to_OpenSG
python scripts/fetch_prevabs.py          # downloads the PreVABS release binary for your OS
```

windIO ships the IEA-22-280-RWT (v2) example, so no extra data download is needed.

## Usage

```bash
# Convert one station (r = non-dimensional blade length 0..1) -> 1D YAML + PreVABS XML
python scripts/convert_station.py --r 0.5 --mesh-size 0.01 --out out/
# All stations
python scripts/run_sweep.py
```

Or from Python:

```python
import windIO, os
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml, emit_prevabs

src = os.path.join(os.path.dirname(windIO.__file__), "examples/turbine/IEA-22-280-RWT.yaml")
blade = WindIOBlade(src)
cs = build_cross_section(blade, r=0.5, mesh_size=0.01)     # resolve the cross-section at mid-span
emit_opensg_yaml(cs, "iea22_r050.yaml")                    # 1D-shell SG  -> RM / Kirchhoff
emit_prevabs(cs, "prevabs_r050", name="iea22_r050")        # PreVABS XML  -> .sg -> 2D-solid
```

## Validation status

The IEA-22 converter is validated end-to-end:

- **1D-shell**: every station r = 0.1 … 0.95 builds (6–7 laminates incl. carbon spar caps, foam fillers,
  glass reinforcements, 3 biax/foam webs) and homogenizes; RM and Kirchhoff agree to **< 0.4 %** at mid-span.
- **2D-solid**: the PreVABS XML meshes cleanly (`prevabs.exe` → `.sg`, ~26 k elements at r=0.5) and converts
  to the OpenSG solid YAML.

Worked benchmark in [`examples/iea22/`](examples/iea22/).

## Licensing

The converter (`windio_to_opensg/`, `scripts/`) is MIT (see `LICENSE`). **PreVABS** and **gmsh** are
**GPL v2** third-party tools, included via submodule/fetch — see `NOTICE`. WindIO_to_OpenSG calls
`prevabs.exe` as a separate subprocess ("mere aggregation" under the GPL), so this repository's own code
remains MIT.
