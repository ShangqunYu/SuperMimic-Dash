#!/bin/bash
# As checkpoints of dash_stairs_{edge,ctrl} appear: held-out stairs tracking with edge contact (nominal and DR);
# at 50k also the stairs-only course (gen_v6 drives the tracker) at 4 speeds, 3 seeds.
cd "$(dirname "$0")/.."

PY="env -u PYTHONPATH OMP_NUM_THREADS=4 systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 .venv/bin/python"
OUT=logs/stairs_eval; mkdir -p $OUT
for k in 10000 20000 30000 40000 50000; do
  c=$(printf "ckpt_%07d.pt" $k)
  for r in edge ctrl; do
    until [ -f logs/dash_stairs_$r/$c ]; do sleep 60; done
  done
  sleep 30
  for r in edge ctrl; do
    g=$([ $r = edge ] && echo 1 || echo 0)
    t=${r}_$((k / 1000))k
    [ -f $OUT/$t.json ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_edge.py logs/dash_stairs_$r/$c data/library/dash_stairs_v1_eval.pt $OUT/$t.json 2>&1 | grep -v -i 'warn\|INFO\|^[|+]\|Module' | tail -4 &
    [ -f $OUT/${t}_dr.json ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_edge.py logs/dash_stairs_$r/$c data/library/dash_stairs_v1_eval.pt $OUT/${t}_dr.json --randomize 2>&1 | grep -v -i 'warn\|INFO\|^[|+]\|Module' | tail -4 &
  done
  wait
  echo "$k done"
done
for s in 0 1 2; do
  for r in edge ctrl; do
    g=$([ $r = edge ] && echo 1 || echo 0)
    d=$OUT/course_${r}_50k_s$s
    [ -f $d/results.json ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/generator/course_demo.py logs/gen_v6/ckpt_0040000.pt logs/dash_stairs_$r/ckpt_0050000.pt $d \
      --rises 0.10 0.14 0.17 --gaps 0 --platforms 0 --speeds 0.5 1.0 1.5 2.0 --seconds 24 --dr --seed $s > $d.out 2>&1 &
  done
  wait
done
echo ALLDONE
