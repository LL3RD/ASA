#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29150}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH\
OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port=$PORT \
  --nproc_per_node=$GPUS $(dirname "$0")/train.py