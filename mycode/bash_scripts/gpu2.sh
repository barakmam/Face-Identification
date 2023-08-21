#!/bin/bash

export PYTHONPATH=/tmp/bionicEye
ulimit -n 2048

mkdir tmp_run2
cp -r *[^tmp_run2]* tmp_run2
cd tmp_run2
python mycode/train.py -m "system.gpus=[2]" run_name=srfr data.batch_size=32 model.sr.lr=1e-5,1e-4,1e-3,1e-2 data.n_triplets=32,128,512,1024,2048
cd ..
rm -r tmp_run2



