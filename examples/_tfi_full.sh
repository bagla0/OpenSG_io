#!/usr/bin/env bash
cd ~/OpenSG_io
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=~/miniconda3/envs/opensg_2_0/bin/python
# every ADJACENT meshable windIO pair, gmsh transfinite hex, inverted-hex count
for pair in "0.0487 0.0665" "0.0665 0.0835" "0.0835 0.1022" "0.1022 0.1104" \
            "0.1104 0.1364" "0.1364 0.1556" "0.1556 0.1967" "0.1967 0.2470" \
            "0.2470 0.3993" "0.3993 0.5336" "0.5336 0.7389" "0.7389 0.9800"; do
  set -- $pair
  R1=$1 R2=$2 timeout 150 $PY examples/_gmsh_tfi_full.py 2>&1 | grep -iE 'transfinite FULL|Error' \
    | sed "s/DONE.*: /: /"
done
