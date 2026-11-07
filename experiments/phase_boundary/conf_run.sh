#!/usr/bin/env bash
# Confirmation training, one run per call; resumable (skips a run whose training log exists).
cd "$(dirname "$0")"; sha256sum -c freeze3.sha256 --quiet || exit 9
p=$1; cps=$2
S=$(python3 -c "import json;print(','.join(map(str,json.load(open('protocol3.json'))['training_seeds'])))")
[ -f confirm/cnn_l_p${p}_e0.05_train.json ] || python3 train3r.py env_r11 cnn_l $p 0.05 $S $cps confirm 2>&1 | grep --line-buffered -v Warn
echo "CONF_RUN_DONE $p"
