# Tutorial: OpenFAST blade data (reference)

OpenFAST blade files carry the **already-homogenized** beam properties, not the layup. So you **cannot build
a Structure Gene from OpenFAST** -- but you *can* read it as a reference to validate the stiffness OpenSG
produces from the windIO/PreVABS section.

## Read an ElastoDyn blade file

```bash
python scripts/opensg_io.py BAR_URC_ElastoDyn_blade.dat out
# -> out/ref_elastodyn.dat : BlFract, FlpStff(EI2), EdgStff(EI3), BMassDen per station
```

```python
from opensg_io import read_elastodyn_blade, elastodyn_at

ed = read_elastodyn_blade("BAR_URC_ElastoDyn_blade.dat")
flap, edge, mass, twist = elastodyn_at(ed, 0.5)     # interpolate at mid-span
print("EI_flap=%.3e  EI_edge=%.3e" % (flap, edge))
```

## Read a BeamDyn blade file

BeamDyn gives the full 6x6; OpenSG_io reorders it from BeamDyn's `[shear_x, shear_y, axial, bend_x, bend_y,
torsion]` to the OpenSG Timoshenko `[EA, GA2, GA3, GJ, EI2, EI3]` (axial/torsion are exact; the shear and
bending pairs are matched by magnitude).

```python
from opensg_io import read_beamdyn_blade, beamdyn_to_timo
etas, Ks, Ms = read_beamdyn_blade("BAR_URC_BeamDyn_Blade.dat")
timo_6x6 = beamdyn_to_timo(Ks[10])                  # 11th station, OpenSG order
```

## Write a BeamDyn blade file (export)

Go the other way -- turn an OpenSG-homogenized 6x6 per station into a BeamDyn blade input, to drive an
OpenFAST aeroelastic run:

```python
from opensg_io import write_beamdyn_blade
write_beamdyn_blade(etas, list_of_timo_6x6, out_path="blade_BeamDyn_Blade.dat")
```

## A note on validation

When you compare, expect the **flapwise** EI to agree closely. A large **edgewise** mismatch usually points
to the trailing edge (flatback TE material at large chord-distance dominates edgewise EI) or a different
reference axis -- not necessarily a homogenization error, especially when your 1D-shell and 2D-solid agree
with each other.
