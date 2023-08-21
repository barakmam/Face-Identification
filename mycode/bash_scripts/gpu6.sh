#!/bin/bash

export PYTHONPATH=/tmp/bionicEye
ulimit -n 2048

mkdir tmp_run6
cp -r *[^tmp_run6]* tmp_run6
cd tmp_run6
python mycode/train.py "system.gpus=[6]" run_name=srfr
cd ..
rm -r tmp_run6



