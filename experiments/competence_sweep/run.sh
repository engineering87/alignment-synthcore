#!/usr/bin/env bash
# Reproduce competence-sweep-v1 scoring from the stored policies. The frozen scripts expect
# ../src (repository root) and ../measure.py beside the sweep directory, so this builds that
# layout in a scratch directory rather than editing frozen files. Needs jax (batteries.py
# imports the vendored gw.py). Set RETRAIN=1 to retrain (weights are platform-dependent).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK/sweep/confirm"; ln -sfn "$REPO" "$WORK/src"
cp "$REPO/experiments/preference_measurement/measure.py" "$WORK/"
cp "$REPO"/experiments/competence_sweep/*.py "$REPO"/experiments/competence_sweep/*.json "$WORK/sweep/"
cd "$WORK/sweep" && sha256sum -c "$REPO/results/competence-sweep-v1/freeze_sweep.sha256"
if [ "${RETRAIN:-0}" = 1 ]; then
  S=$(python3 -c "import json;print(','.join(map(str,json.load(open('protocol_sweep.json'))['training_seeds'])))")
  python3 train_sweep.py mlp_l 0.05 $S 2400 confirm; python3 train_sweep.py mlp_s 0.05 $S 2400 confirm
  python3 train_sweep.py cnn_l 0.05 $S 300 confirm
else cp "$REPO"/results/competence-sweep-v1/models/*.npz confirm/; fi
python3 -u confirm_sweep.py   # about 75 minutes on 2 CPUs (NumPy CNN forward pass)
echo "outputs in $WORK/sweep"
