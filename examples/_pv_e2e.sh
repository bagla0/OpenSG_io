#!/usr/bin/env bash
cd ~/OpenSG_io
export PYTHONIOENCODING=utf-8
PY=~/miniconda3/envs/opensg_2_0/bin/python
$PY examples/prevabs_boundary_e2e.py "${1:-0.5336}" "${2:-examples/data/IEA-22-280-RWT.yaml}" "${3:-0.02}"
