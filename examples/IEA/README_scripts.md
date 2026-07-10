# IEA-22 tapered-segment scripts

All Python used to generate the meshes, run the homogenizers, and build every figure /
`.dat` in this folder. Run in the local `opensg_2_0_env` unless noted.

## Mesh generation (OpenSG_io)
- `iea22_segment.py` — build the r=0.2→0.3 segment from the bundled windIO: writes the
  **solid hex**, **shell quad** (OML reference), and the **four boundary** YAMLs to
  `output/`, plus renders. `IEA_NR=<n>` env sets through-thickness layers (default 4).

## Homogenizers (Timoshenko 6×6)
- `run_timo_shell.py` — OpenSG-TW JAX MITC-RM on the shell YAML → `output/timo_shell.npz`.
- `run_timo_solid_fenics.py` — OpenSG-FEniCS on the solid YAML → `output/timo_solid.npz`.
  Run on the server: `~/miniconda3/envs/opensg_env_v8/bin/python run_timo_solid_fenics.py`.

## Results / figures (read the YAMLs + npz)
- `export_dat_all.py` — the single combined `timo_shell_vs_solid_all.dat` (taper + both
  boundaries, solid + shell + %err 6×6).
- `compare_timo.py` — shell-vs-solid all-non-zero-terms table.
- `check_shell_vs_solid.py` — 20-point mesh consistency cross-check.
- `render_comparisons.py` — boundary rows (with nodes), taper comparison, boundary e1/e2/e3.
- `export_layup_views.py` — shell-by-layup + solid-by-material PyVista + boundary row.
- `export_deliverable.py` — mesh + e1/e2/e3 orientation images + Timo `.dat`.

## Soft-core debug (why GA2/GA3/GJ differ ~13×)
- `_stiff_foam.py` — write a variant YAML with foam shear modulus raised to the skin's.
- `_run_shell_stiff.py` / `_run_solid_stiff.py` — re-run each homogenizer on the variant.
- `_compare_stiff.py` — the original-vs-stiff gap table (13× → ~1× = soft-core proof).
  See `DEBUG_soft_core.txt`.
