# Tutorial: PreVABS XML -> OpenSG

If you already have a **PreVABS cross-section XML** (airfoil baseline, dividing points, layups, segments,
webs), OpenSG_io turns it into OpenSG inputs directly -- no windIO needed.

## Convert (CLI)

```bash
# 1D shell only (reconstructed from the XML midline + layups)
python scripts/opensg_io.py xsec.xml out

# 1D shell + 2D solid (runs PreVABS on the XML)
python scripts/opensg_io.py xsec.xml out --solid
```

`xsec.xml` must sit next to its airfoil `.dat` and `materials.xml` (the files it `<include>`s), exactly as
PreVABS expects.

## Convert (Python)

```python
from opensg_io import prevabs_xml_to_shell, prevabs_xml_to_solid

# 1D shell -- parses the XML, maps dividing points to arc positions, rebuilds the midline mesh
prevabs_xml_to_shell("xsec.xml", "shell_xsec.yaml")

# 2D solid -- runs prevabs, then converts the .sg to the OpenSG solid YAML
prevabs_xml_to_solid("xsec.xml", "solid_xsec.yaml", prevabs="third_party/prevabs_bin/.../prevabs")
```

## How the 1D reconstruction works

`parse_prevabs_xml()` reads:

- `general/scale` -> chord; the airfoil `.dat` -> the OML contour (scaled by chord);
- each dividing `point` (`by="x2" which="top|bottom"`) -> an **arc position** `s` on the contour;
- each `layup` -> a ply stack `[(material, thickness, angle)]` (thickness = `count x lamina_thickness`);
- each surface `segment` (`baseline -> layup`) -> an arc range that carries that laminate;
- each `web` (`point + angle`) -> the two contour attachment nodes of the web line.

It then discretises the contour into nodes/line-elements with one element set per layup, and hands the result
to the same `emit_opensg_yaml()` writer used for windIO -- so the 1D-shell YAML is identical in format.

```{note}
The reconstruction targets the standard PreVABS airfoil-cross-section XML (the kind `emit_prevabs()` writes
and typical blade sections use). Very unusual baseline/point constructs may need adjustment.
```
