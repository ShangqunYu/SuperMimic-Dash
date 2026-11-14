#!/usr/bin/env bash
# Scene tests (kinematic + closed loop with the v6 teacher) on every checkpoint of a generator run as it appears.
#   scripts/generator/eval_checkpoints.sh RUN [LAST_STEP] [GPU]   -> logs/gen_eval/RUN_<step>k_{kin,loop}/, summary logs/gen_eval/RUN.summary
set -u
RUN=$1; LAST=${2:-200000}; GPU=${3:-1}
cd "$(dirname "$0")/../.."
TEACHER=logs/dash_teacher_v6/ckpt_0050000.pt
PY="env -u PYTHONPATH OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 .venv/bin/python"
step=${FIRST:-20000}
while [ $step -le $LAST ]; do
  ck=$(printf "logs/%s/ckpt_%07d.pt" "$RUN" $step)
  until [ -f "$ck" ]; do sleep 60; done
  sleep 30
  tag=logs/gen_eval/${RUN}_$((step / 1000))k
  $PY scripts/generator/rollout_generator.py "$ck" ${tag}_kin --samples 8 > ${tag}_kin.out 2>&1
  $PY scripts/generator/closed_loop.py "$ck" $TEACHER ${tag}_loop --samples 8 > ${tag}_loop.out 2>&1
  echo "$step kin $(grep -h '^success' ${tag}_kin.out) loop $(grep -h '^success' ${tag}_loop.out)" >> logs/gen_eval/${RUN}.summary
  step=$((step + 20000))
done
