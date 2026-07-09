# OpenSG_io

**Prepare [OpenSG](https://github.com/wenbinyugroup/OpenSG) cross-section inputs from windIO,
PreVABS, or OpenFAST** — the **1-D shell SG** and the **2-D solid SG** at any spanwise station,
and the **3-D tapered segment** (structured hex solid + matching quad shell), without
hand-building a Structure Gene. It *reads* those formats; it does not run windIO or OpenFAST.

## Key features and capabilities

- **Three inputs, one converter** — a windIO **v1 or v2** blade (IEA-22, NREL BAR, …), a
  PreVABS cross-section XML, or OpenFAST ElastoDyn/BeamDyn blade data (the last as a
  homogenized $6\times6$ / $EI$ *reference* — no layup, so no SG).
- **Shell *and* solid from every station** — each station yields both the 1-D shell SG and
  the 2-D solid SG, so shell-vs-solid cross-checks are one-to-one.
- **3-D tapered segments** — a general two-station loft builds the **strictly structured
  8-node HEX solid** (through-thickness layers from the local laminate, per-hex ply fiber
  frames, refined conformal web junctions) plus the **equivalent mid-surface quad shell**.
- **Mesh quality guaranteed, not hoped for** — every export passes a conformity gate
  (watertight full-face adjacency), a positive **min-scaled-Jacobian** check, and a
  PreVABS/NuMAD-grade fold-free offset (miter normals, thin-gap clamp, full-accuracy
  trailing-edge opening).
- **Validated** — IEA-22 (windIO v2) and NREL BAR-URC (v1) convert end-to-end; the 1-D
  shell RM/Kirchhoff and the 2-D solid (FEniCS) agree within ~5 % root-to-mid.

| input | what OpenSG_io produces |
|---|---|
| **windIO** blade (`*.yaml`, v1 *or* v2) | 1-D shell + 2-D solid SG YAML per station; 3-D hex/shell segment between stations |
| **PreVABS** cross-section (`*.xml`) | 1-D shell SG (reconstructed) + 2-D solid SG (runs PreVABS) |
| **OpenFAST** blade data (`*.fst`, `*.dat`) | the homogenized $6\times6$ / $EI$ **reference** for validation |

## How it fits together

```text
  windIO (v1/v2) --.
  PreVABS XML -----+--->  OpenSG_io  -->  1D-shell SG YAML  -->  MSG-RM / Kirchhoff  (Timoshenko 6x6)
  OpenFAST --------'                 \->  2D-solid SG YAML  -->  FEniCS solid        (VABS-equivalent)
                                      \-> 3D SEGMENT: structured HEX + quad shell (tapered 6x6)
                                       \-> (OpenFAST) 6x6 reference for validation
```

The SG YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers.

```{toctree}
:hidden:
:caption: Introduction

installation
inputs
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/iea22_tutorial
tutorials/windio
tutorials/prevabs_xml
tutorials/openfast
```

```{toctree}
:hidden:
:caption: Reference

api
pipeline
```
