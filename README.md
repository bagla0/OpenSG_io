# OpenSG_io

**A wrapper that converts cross-section inputs into [OpenSG](https://github.com/wenbinyugroup/OpenSG)
YAMLs** -- both the **1D-shell** SG (for the MSG shell homogenizers) and the **2D-solid** SG (VABS / FEniCS).
Its only job is to *prepare OpenSG input files*; it does **not** run windIO or OpenFAST -- it reads those
formats so you don't have to hand-build the SG.

Documentation & tutorials: **https://bagla0.github.io/OpenSG_io/**

**Worked example** → [Build a 3-D tapered blade segment (shell + structured-hex solid)](https://bagla0.github.io/OpenSG_io/tutorials/iea22_tutorial.html):
takes the IEA-22-280 windIO between `r = 0.2` and `r = 0.3` and produces both SG meshes end-to-end —
the generation code, printed output, and mesh renders are all in the notebook.

## Inputs

| input | what OpenSG_io produces |
|-------|-------------------------|
| **windIO** blade (`*.yaml`, v1 *or* v2 -- IEA-22, NREL BAR, ...) | 1D-shell YAML + 2D-solid YAML, per spanwise station |
| **PreVABS** cross-section (`*.xml`) | 1D-shell YAML (reconstructed from the XML) + 2D-solid YAML (runs PreVABS) |
| **OpenFAST** blade data (`*.fst`, ElastoDyn/BeamDyn `*.dat`) | the homogenized 6x6 / EI **reference** (OpenFAST has no layup, so no SG is built -- this is for *validation*) |

```
  windIO (v1/v2) --.
  PreVABS XML -----+--->  OpenSG_io  --->  1D-shell SG YAML  --> MSG-RM / Kirchhoff (Timoshenko 6x6)
  OpenFAST --------'                  \->  2D-solid SG YAML  --> FEniCS solid (VABS-equivalent)
                                       \-> (OpenFAST) 6x6 reference for validation
```

## Install

```bash
git clone --recurse-submodules https://github.com/bagla0/OpenSG_io
cd OpenSG_io
pip install numpy pyyaml          # core; add `windIO` only for windIO v2 inputs
python scripts/fetch_prevabs.py   # PreVABS release binary for your OS (needed for the 2D-solid)
```

## Usage -- one CLI, any input

```bash
# windIO blade -> 1D shell (+ 2D solid) at chosen stations
python scripts/opensg_io.py BAR_URC.yaml out --name bar --stations 0.3 0.5 0.7 --solid

# PreVABS XML -> 1D shell (reconstructed) + 2D solid
python scripts/opensg_io.py xsec.xml out --solid

# OpenFAST ElastoDyn/BeamDyn -> blade-data 6x6 / EI reference
python scripts/opensg_io.py BAR_URC_ElastoDyn_blade.dat out
```

From Python:

```python
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs
from opensg_io import prevabs_xml_to_shell, prevabs_xml_to_solid, read_elastodyn_blade

blade = load_blade("BAR_URC.yaml")                 # auto-detects windIO v1 / v2
cs = build_cross_section(blade, r=0.5)
emit_opensg_yaml(cs, "shell_r050.yaml")            # 1D-shell SG
emit_prevabs(cs, "prevabs_r050", name="r050")      # PreVABS XML -> .sg -> 2D-solid

prevabs_xml_to_shell("xsec.xml", "shell_xsec.yaml")   # PreVABS XML -> 1D shell
ed = read_elastodyn_blade("BAR_URC_ElastoDyn_blade.dat")  # OpenFAST reference (FlpStff/EdgStff/mass)
```

## Validated

- **windIO v2** (IEA-22-280-RWT) and **v1** (NREL BAR-URC) both convert end-to-end; the 1D-shell RM/Kirchhoff
  and the 2D-solid (FEniCS) agree within ~5% root-to-mid (worked benchmark in
  [`examples/iea22_hex_segment.py`](examples/iea22_hex_segment.py), rendered in the
  [tutorial](https://bagla0.github.io/OpenSG_io/tutorials/iea22_tutorial.html)).
- OpenFAST ElastoDyn/BeamDyn reading round-trips against the BAR blade data.

## Third-party tools

`opensg_io/` and `scripts/` are MIT (`LICENSE`). **PreVABS** + **gmsh** are GPL-2.0 and **OpenFAST** is
Apache-2.0, included as submodules / fetched binaries -- see `NOTICE`. OpenSG_io invokes them as separate
subprocesses (or just reads their files), so this repository's own code stays MIT.
