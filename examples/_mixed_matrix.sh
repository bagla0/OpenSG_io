#!/usr/bin/env bash
# Empirical IEA matrix for the MIXED conformal generator (uses ~/OpenSG_io directly).
cd ~/OpenSG_io
find ~/OpenSG_io -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
W=examples/data/IEA-22-280-RWT.yaml
for pair in "0.2 0.3 MILD" "0.247 0.3993 STEEP" "0.0487 0.0665 ROOT"; do
  set -- $pair
  echo "######## MIXED $3 r=$1 -> $2 ########"
  timeout 900 $PY examples/build_blade_mesh.py $W taper --r1 $1 --r2 $2 --model solid \
      --element mixed --out examples/mesh_out/mixed_$3 2>&1 \
      | grep -E '\[mixed\]|solid \(MIXED|OUTPUTS|Error|RuntimeError|Traceback' | head -10
done
