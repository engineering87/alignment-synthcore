#!/usr/bin/env bash
# Development grid: seeds 61000-61003 (never used for confirmation).
cd "$(dirname "$0")"
S=61000,61001,61002,61003
for e in 0.05 0.01; do
  [ -f dev/mlp_s_e${e}_train.json ] || python3 train_sweep.py mlp_s $e $S 300,1200,2400,4800,9600 dev
  [ -f dev/mlp_l_e${e}_train.json ] || python3 train_sweep.py mlp_l $e $S 300,1200,2400 dev
done
for e in 0.05 0.01; do
  [ -f dev/cnn_l_e${e}_train.json ] || python3 train_sweep.py cnn_l $e $S 300,1200,2400 dev
done
echo ALL_DONE
