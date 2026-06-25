---
sd_hide_title: true
---

# OpenSG_io

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sd-pt-2 sd-pb-2

:::{grid-item}
:columns: 12 12 7 7

```{rst-class} display-4 sd-font-weight-bold sd-mb-2
OpenSG_io
```

**Prepare [OpenSG](https://github.com/wenbinyugroup/OpenSG) cross-section inputs from windIO, PreVABS, or
OpenFAST** — the **1D-shell SG** and the **2D-solid SG**, at any spanwise station, without hand-building a
Structure Gene. It *reads* those formats; it does not run windIO or OpenFAST.

```{button-ref} installation
:color: primary
:class: sd-px-4 sd-me-2
Get started
```
```{button-ref} tutorials/iea22_tutorial
:color: secondary
:outline:
:class: sd-px-4
Worked example
```
:::

:::{grid-item}
:columns: 12 12 5 5
:class: sd-d-flex-column sd-align-items-center

```{image} _static/logo.svg
:width: 150px
:class: sd-mb-2
```
:::
::::

---

## Pick your input

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`workflow;1.4em;sd-text-primary` windIO
:link: tutorials/windio
:link-type: doc
:class-card: sd-shadow-sm

A windIO **v1 or v2** blade (IEA-22, NREL BAR, …).
^^^
→ 1D-shell + 2D-solid SG **per spanwise station**.
:::

:::{grid-item-card} {octicon}`file-code;1.4em;sd-text-primary` PreVABS XML
:link: tutorials/prevabs_xml
:link-type: doc
:class-card: sd-shadow-sm

An existing PreVABS cross-section XML.
^^^
→ 1D-shell **reconstructed** from the XML + 2D-solid by running PreVABS.
:::

:::{grid-item-card} {octicon}`gear;1.4em;sd-text-primary` OpenFAST
:link: tutorials/openfast
:link-type: doc
:class-card: sd-shadow-sm

ElastoDyn / BeamDyn blade data.
^^^
→ the homogenized **6×6 / EI reference** (no layup → no SG; for validation).
:::
::::

## Why OpenSG_io

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`versions;1.2em;sd-text-secondary` Version-agnostic
One `load_blade()` reads windIO **v1** *and* **v2** — no blade-specific code; laminates follow the windIO
layer order.
:::
:::{grid-item-card} {octicon}`git-compare;1.2em;sd-text-secondary` Shell *and* solid
Every input yields both the 1D-shell and 2D-solid SG, so you can cross-check shell-vs-solid at each station.
:::
:::{grid-item-card} {octicon}`verified;1.2em;sd-text-secondary` Validated
IEA-22 (v2) and NREL BAR-URC (v1) convert end-to-end; shell and solid agree within ~5% root-to-mid.
:::
:::{grid-item-card} {octicon}`terminal;1.2em;sd-text-secondary` One CLI
`opensg_io.py <input> <outdir>` auto-detects the input type and routes it.
:::
::::

## How it fits together

```text
  windIO (v1/v2) --.
  PreVABS XML -----+--->  OpenSG_io  -->  1D-shell SG YAML  -->  MSG-RM / Kirchhoff  (Timoshenko 6x6)
  OpenFAST --------'                 \->  2D-solid SG YAML  -->  FEniCS solid        (VABS-equivalent)
                                      \-> (OpenFAST) 6x6 reference for validation
```

The 1D-shell and 2D-solid YAMLs feed the [OpenSG-TW](https://github.com/bagla0/OpenSG-TW) homogenizers.

```{toctree}
:hidden:
:caption: Getting started

installation
inputs
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/iea22_tutorial
tutorials/windio
tutorials/prevabs_xml
tutorials/openfast
```

```{toctree}
:hidden:
:caption: Reference

api
```
