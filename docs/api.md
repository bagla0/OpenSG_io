# API reference

## windIO adapter (`opensg_io.converter`)

```{eval-rst}
.. autofunction:: opensg_io.load_blade
.. autoclass:: opensg_io.WindIOBlade
.. autoclass:: opensg_io.WindIOBladeV1
.. autofunction:: opensg_io.build_cross_section
.. autofunction:: opensg_io.emit_opensg_yaml
.. autofunction:: opensg_io.emit_prevabs
```

## PreVABS XML adapter (`opensg_io.prevabs_xml`)

```{eval-rst}
.. autofunction:: opensg_io.parse_prevabs_xml
.. autofunction:: opensg_io.prevabs_xml_to_shell
.. autofunction:: opensg_io.prevabs_xml_to_solid
```

## OpenFAST bridge (`opensg_io.openfast_io`)

```{eval-rst}
.. autofunction:: opensg_io.read_elastodyn_blade
.. autofunction:: opensg_io.elastodyn_at
.. autofunction:: opensg_io.read_beamdyn_blade
.. autofunction:: opensg_io.beamdyn_to_timo
.. autofunction:: opensg_io.timo_to_beamdyn
.. autofunction:: opensg_io.write_beamdyn_blade
```
