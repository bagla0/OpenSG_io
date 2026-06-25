# Tutorial: windIO blade -> OpenSG

This converts a windIO blade into OpenSG 1D-shell and 2D-solid YAMLs at chosen spanwise stations. It works
for **both** windIO v2 (e.g. IEA-22-280-RWT) and v1 (e.g. the NREL BAR-URC blade).

## 1. Get a blade

```bash
# v1 example: NREL BAR-URC (plain YAML, no windIO package needed)
curl -L -o BAR_URC.yaml \
  https://raw.githubusercontent.com/NREL/BAR_Designs/main/BAR_URC/BAR_URC.yaml

# v2 example: IEA-22 ships with the windIO package
python -c "import windIO,os,shutil; shutil.copy(os.path.join(os.path.dirname(windIO.__file__),'examples','turbine','IEA-22-280-RWT.yaml'),'.')"
```

## 2. Convert (CLI)

```bash
# 1D shell only, three stations
python scripts/opensg_io.py BAR_URC.yaml out --name bar --stations 0.3 0.5 0.7

# 1D shell + 2D solid (needs PreVABS -- see Installation)
python scripts/opensg_io.py BAR_URC.yaml out --name bar --stations 0.5 --solid
```

Output in `out/`:

```
shell_bar_r050.yaml      # 1D-shell SG
solid_bar_r050.yaml      # 2D-solid SG (with --solid)
prevabs_r050/            # the PreVABS XML/.dat/.sg used for the solid
```

## 3. Convert (Python)

```python
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs

blade = load_blade("BAR_URC.yaml")             # -> WindIOBladeV1 (v1) or WindIOBlade (v2)
cs = build_cross_section(blade, r=0.5, mesh_size=0.01)
print(cs["chord"], len(cs["laminates"]), "laminates")

emit_opensg_yaml(cs, "shell_r050.yaml")        # 1D-shell SG
emit_prevabs(cs, "prevabs_r050", name="r050")  # PreVABS XML -> run prevabs -> .sg -> 2D-solid
```

## Notes

- **nd_arc convention**: arc coordinate `s` runs TE(`0`) -> suction -> LE(`0.5`) -> pressure -> TE(`1`),
  matching the windIO airfoil ordering, for both v1 and v2.
- **Laminate order** = the order layers appear in the windIO file (outer -> inner) -- no blade-specific
  assumptions.
- **Thick walls** (e.g. BAR's foam-cored LE) need **PreVABS 2.1+** for the 2D mesh; OpenSG_io also snaps
  skin dividing points off the leading-edge nose and caps through-thickness plies so the mesher stays robust.
