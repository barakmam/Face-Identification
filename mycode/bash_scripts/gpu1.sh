#!/bin/bash

export PYTHONPATH=/tmp/bionicEye
ulimit -n 2048

mkdir tmp_run1
cp -r *[^tmp_run1]* tmp_run1
cd tmp_run1
python mycode/train.py -m "system.gpus=[1]" run_name=srfr data.batch_size=16 model.sr.lr=1e-5,1e-4,1e-3,1e-2 data.n_triplets=32,128,512,1024,2048
cd ..
rm -r tmp_run1



