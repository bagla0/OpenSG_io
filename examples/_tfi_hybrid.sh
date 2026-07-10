#!/usr/bin/env bash
cd ~/OpenSG_io
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=~/miniconda3/envs/opensg_2_0/bin/python
# HEX-DOMINANT: fold web cells left free (tet/pyramid), rest transfinite hex. Worst adjacent pairs.
for pair in "0.0487 0.0665" "0.3993 0.5336" "0.7389 0.9800"; do
  set -- $pair
  echo "=== r=$1 -> $2 ==="
  R1=$1 R2=$2 HYBRID=1 timeout 250 $PY examples/_gmsh_tfi_full.py 2>&1 | grep -iE 'fold cells|transfinite FULL|Error|Exception'
done
