#!/usr/bin/env bash
# Development p sweep (seeds 81000-81003). Resumable: skips cells whose training log exists.
cd "$(dirname "$0")"
for p in "$@"; do
  [ -f dev/cnn_l_p${p}_e0.05_train.json ] || python3 train3.py env_r11 cnn_l $p 0.05 81000,81001,81002,81003 600,1200 dev 2>&1 | grep --line-buffered -v Warn
done
echo "DEV_P_DONE $*"
