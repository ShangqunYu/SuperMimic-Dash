#!/bin/bash
# Stairs-only trackers from scratch (2026-11-19): the v6 teacher's flags on data/library/dash_stairs_v1.pt, 50k
# iterations, with the edge penalty (GPU 1) and the identical control without it (GPU 0). Lab server: memory-capped.
cd "$(dirname "$0")/.."
FL="--robot dash --data.library data/library/dash_stairs_v1.pt --env.hardware-dr --env.terrain-ratio 0.4 --env.root-xy-threshold 1.0
    --env.contact-timeconst 0.01 --env.compact-library --env.policy-scan --world-model.state-spec.xi-hw-bodies 19
    --world-model.state-spec.height-scan --world-model.xy-threshold 1.0 --policy.num-joints 18 --policy.goal-dim 923 --policy.terrain-dim 403
    --policy-trainer.policy-scan --policy-trainer.goal-stride 5 --policy-trainer.goal-root-ori --schedule.lr-schedule cosine
    --schedule.lr-decay-fraction 1.0 --schedule.total-iterations 50000 --schedule.lr-decay-start 10000 --schedule.checkpoint-interval 10000"
PY="env -u PYTHONPATH OMP_NUM_THREADS=4 systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 .venv/bin/python -m supermimic.scripts.train"
CUDA_VISIBLE_DEVICES=1 nohup $PY $FL --run-name dash_stairs_edge --policy-trainer.edge-vol 33 > logs/dash_stairs_edge.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup $PY $FL --run-name dash_stairs_ctrl > logs/dash_stairs_ctrl.out 2>&1 &
