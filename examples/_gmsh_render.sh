#!/usr/bin/env bash
# regenerate an IEA boundary + taper .msh, then render each with gmsh (xvfb, by subdomain)
set +e
cd ~/OpenSG_io
PY=~/miniconda3/envs/opensg_2_0/bin/python
export PYTHONIOENCODING=utf-8
W=examples/data/IEA-22-280-RWT.yaml
$PY examples/build_blade_mesh.py $W boundary --r 0.5336 --model both --out examples/mesh_out >/dev/null 2>&1
$PY examples/build_blade_mesh.py $W taper --r1 0.1967 --r2 0.2470 --model both --out examples/mesh_out 2>&1 | grep -iE 'inverted|solid:|shell:' | head -3
for f in examples/mesh_out/*.msh; do
  echo "render $f"
  xvfb-run -a $PY examples/render_msh_gmsh.py "$f" "${f%.msh}_gmsh.png" 2>&1 | grep -iE 'wrote|error' | head -1
done
