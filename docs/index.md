# OpenSG_io

**OpenSG_io** is a wrapper that converts cross-section inputs into [OpenSG](https://github.com/wenbinyugroup/OpenSG)
YAMLs -- both the **1D-shell** SG (for the MSG shell homogenizers) and the **2D-solid** SG (VABS / FEniCS).
Its job is to *prepare OpenSG input files*. It does **not** run windIO or OpenFAST; it only **reads** those
formats so you don't have to hand-build the Structure Gene.

```{list-table}
:header-rows: 1

* - Input
  - OpenSG_io produces
* - **windIO** blade (`*.yaml`, v1 or v2)
  - 1D-shell YAML + 2D-solid YAML, per spanwise station
* - **PreVABS** cross-section (`*.xml`)
  - 1D-shell YAML (reconstructed) + 2D-solid YAML (runs PreVABS)
* - **OpenFAST** blade data (ElastoDyn / BeamDyn)
  - homogenized 6x6 / EI **reference** (no SG -- OpenFAST has no layup)
```

One CLI handles all three -- it auto-detects the input type:

```bash
python scripts/opensg_io.py BAR_URC.yaml  out --name bar --stations 0.5 --solid   # windIO
python scripts/opensg_io.py xsec.xml       out --solid                            # PreVABS XML
python scripts/opensg_io.py BAR_URC_ElastoDyn_blade.dat out                        # OpenFAST reference
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
inputs
tutorials/iea22_tutorial
tutorials/windio
tutorials/prevabs_xml
tutorials/openfast
api
```

## How it fits together

```
  windIO (v1/v2) --.
  PreVABS XML -----+--->  OpenSG_io  -->  1D-shell SG YAML  -->  MSG-RM / Kirchhoff  (Timoshenko 6x6)
  OpenFAST --------'                 \->  2D-solid SG YAML  -->  FEniCS solid        (VABS-equivalent)
                                      \-> (OpenFAST) 6x6 reference for validation
```

The 1D-shell and 2D-solid YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers, so a
single blade can be cross-checked shell-vs-solid (and against OpenFAST) at every station.
