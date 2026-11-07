#!/usr/bin/env bash
# Development training, resumable: skips finished runs, resumes interrupted ones.
cd "$(dirname "$0")"
S=82000,82001,82002,82003
for pc in 0.3:1000 0.1:2000 0.03:4000; do
  p=${pc%%:*}; cap=${pc##*:}
  [ -f dev/cnn_l_p${p}_e0.05_train.json ] || python3 train_dense.py env_r7 cnn_l $p 0.05 $S $cap dev 2>&1 | grep --line-buffered -v Warn
done
echo DEV_DONE
