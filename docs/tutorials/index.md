# Tutorials

The worked example is an **executed** notebook — committed pre-run, so the printed
conformity/quality gates and the mesh renders are the real outputs; the windIO input is
bundled at [`examples/data/`](https://github.com/bagla0/OpenSG_io/tree/main/examples/data),
clone and run. The three input guides cover each supported front-end.

## Worked example

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} ★ · IEA-22 3-D tapered segment (shell + solid hex)
:link: iea22_tutorial
:link-type: doc
The blade segment between $r=0.2$ and $r=0.3$ from the bundled windIO: strictly
structured hex solid formed from the layup + the equivalent quad shell, conformity and
min-scaled-Jacobian gates, shaded mesh renders.
:::
::::

## Input guides

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} windIO
:link: windio
:link-type: doc
A windIO **v1 or v2** blade → 1-D shell + 2-D solid SG per spanwise station.
:::

:::{grid-item-card} PreVABS XML
:link: prevabs_xml
:link-type: doc
An existing PreVABS cross-section XML → 1-D shell (reconstructed) + 2-D solid (runs
PreVABS).
:::

:::{grid-item-card} OpenFAST
:link: openfast
:link-type: doc
ElastoDyn / BeamDyn blade data → the homogenized $6\times6$ / $EI$ reference (no layup →
no SG; for validation).
:::
::::
