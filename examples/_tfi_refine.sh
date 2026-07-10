#!/usr/bin/env bash
cd ~/OpenSG_io
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=~/miniconda3/envs/opensg_2_0/bin/python
# does REFINING the hoop mesh drive gmsh-transfinite web folds to zero on the worst pairs?
for pair in "0.0487 0.0665" "0.3993 0.5336"; do
  set -- $pair
  echo "=== pair r=$1 -> $2 ==="
  for ms in 0.02 0.01 0.007; do
    printf "  mesh=%s : " $ms
    R1=$1 R2=$2 MS=$ms timeout 200 $PY examples/_gmsh_tfi_full.py 2>&1 | grep -iE 'transfinite FULL|Error' | sed "s/.*DONE.*: //"
  done
done
