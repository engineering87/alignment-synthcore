#!/bin/bash
# Idempotent N2e chain: rerun after a container restart and it picks up where it stopped.
cd /home/claude/bench
set -e
V=n2e; L=18; H=11
if [ ! -f results/${V}_L${L}_h${H}.done ]; then
  python3 -u n2.py train --variant $V --layer $L --head $H --steps 140 --resume >> ${V}_train.out 2>&1
  touch results/${V}_L${L}_h${H}.done
fi
[ -f results/${V}_eval_L${L}_h${H}.json ] || python3 -u n2.py eval --variant $V --layer $L --head $H > ${V}_eval.out 2>&1
[ -f results/false_fire_${V}_L${L}_h${H}.json ] || python3 -u false_fire.py $V $L $H > ${V}_ff.out 2>&1
[ -f results/baselines_N2e.json ] || python3 -u baselines.py --plant results/${V}_L${L}_h${H}.pt --layer $L --head $H --out results/baselines_N2e.json > baselines_N2e.log 2>&1
echo CHAIN_DONE
