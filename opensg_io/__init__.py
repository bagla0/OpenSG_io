"""opensg_io: a wrapper that converts various cross-section inputs into OpenSG YAMLs (1D shell + 2D solid).

Inputs (OpenSG_io does NOT run windIO or OpenFAST -- it only READS their files to prepare OpenSG input):
  - windIO blade (v1 or v2)  -> load_blade(); build_cross_section()/emit_opensg_yaml()/emit_prevabs() per station.
  - PreVABS cross-section XML -> prevabs_xml.to_shell() (1D) and prevabs_xml.to_solid() (2D via PreVABS).
  - OpenFAST blade data       -> openfast_io.read_elastodyn_blade()/read_beamdyn_blade() as a validation
                                 reference (OpenFAST carries no layup, so no SG is built from it).
"""
import os as _os

# Force off-screen SOFTWARE OpenGL for PyVista/VTK renders.  This MUST be set before the first
# pyvista/VTK import anywhere in the process (tetgen_fill imports pyvista during meshing, well
# before any render), else VTK grabs the compute server's GPU (nouveau) and segfaults on large
# meshes.  Setting it here in the package __init__ guarantees it runs before any submodule loads.
_os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
_os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
_os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from .converter import (
    WindIOBlade,
    WindIOBladeV1,
    load_blade,
    build_cross_section,
    emit_opensg_yaml,
    emit_prevabs,
    interp,
    arc_param,
)
from .prevabs_xml import (
    parse_prevabs_xml,
    to_shell as prevabs_xml_to_shell,
    to_solid as prevabs_xml_to_solid,
)
from .openfast_io import (
    read_elastodyn_blade,
    elastodyn_at,
    read_beamdyn_blade,
    beamdyn_to_timo,
    timo_to_beamdyn,
    write_beamdyn_blade,
)

__all__ = [
    "WindIOBlade", "WindIOBladeV1", "load_blade", "build_cross_section",
    "emit_opensg_yaml", "emit_prevabs", "interp", "arc_param",
    "parse_prevabs_xml", "prevabs_xml_to_shell", "prevabs_xml_to_solid",
    "read_elastodyn_blade", "elastodyn_at", "read_beamdyn_blade",
    "beamdyn_to_timo", "timo_to_beamdyn", "write_beamdyn_blade",
]
__version__ = "0.3.0"
