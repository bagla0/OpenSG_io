# Installation

```bash
git clone --recurse-submodules https://github.com/bagla0/OpenSG_io
cd OpenSG_io
pip install numpy pyyaml          # core dependencies
```

For **windIO v2** inputs (the IEA reference blades), also `pip install windIO`. windIO **v1** blades
(e.g. the NREL BAR designs) need no extra package -- they are plain YAML.

## PreVABS (for the 2D-solid)

The 2D-solid path runs **PreVABS** (a GPL-2.0 cross-section mesher). Fetch the release binary for your OS:

```bash
python scripts/fetch_prevabs.py     # -> third_party/prevabs_bin/
```

`opensg_io` auto-discovers the binary under `third_party/prevabs_bin/`. On Linux, the binary dynamically
links `libgmsh` (bundled) and `libstdc++`; if you hit `GLIBCXX_... not found`, put a recent libstdc++ on the
path (e.g. a conda env's `lib/`):

```bash
export LD_LIBRARY_PATH="$(dirname $(find third_party/prevabs_bin -name prevabs)):$CONDA_PREFIX/lib"
```

## OpenFAST (optional)

OpenFAST is only needed to *run* an aeroelastic simulation; **reading** its blade-data files needs no binary.
To fetch it anyway:

```bash
python scripts/fetch_openfast.py    # -> third_party/openfast_bin/
```

## The OpenSG homogenizers

The YAMLs OpenSG_io produces are consumed by the MSG shell homogenizers and the FEniCS 2D-solid in
[OpenSG-TW](https://github.com/bagla0/OpenSG-TW). OpenSG_io itself only prepares the input files.
