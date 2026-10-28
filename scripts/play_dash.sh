#!/usr/bin/env bash
# Watch a Dash policy track N random BONES-SEED clips side by side (each robot plays a random clip from the set, restarting on a new one).
#
#   scripts/play_dash.sh                          # 100 random clips, browser viewer
#   N=50 FILTER=dance scripts/play_dash.sh        # 50 random clips whose name contains "dance"
#   VIEWER=native scripts/play_dash.sh            # desktop window instead of the browser
#   CKPT=logs/other_run/ckpt_latest.pt SEED=3 scripts/play_dash.sh
#
# Browser viewer: open http://<server>:8080 (Tailscale) or tunnel with ssh -L 8080:localhost:8080.
# Tick "show all envs" in the viser panel to see every robot (otherwise one plus neighbours);
# in the native viewer press A for all envs, , / . to switch robot, R to toggle the ghost.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT=${CKPT:-logs/dash_e1_sim/ckpt_latest.pt}
NPZ_DIR=${NPZ_DIR:-data/npz/dash_bones_seed}
N=${N:-100}
FILTER=${FILTER:-}          # case-insensitive substring of the clip name, e.g. jump, dance, crawl, kick
SEED=${SEED:-0}
VIEWER=${VIEWER:-viser}     # viser | native | auto
GPU=${GPU:-1}

mapfile -t CLIPS < <(
    find "$NPZ_DIR" -maxdepth 1 -name '*.npz' -printf '%f\n' \
        | sed 's/\.npz$//' \
        | grep -i -- "${FILTER}" \
        | shuf -n "$N" --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED" -nosalt </dev/zero 2>/dev/null)
)
if [ "${#CLIPS[@]}" -eq 0 ]; then
    echo "no clips in $NPZ_DIR match '$FILTER'" >&2
    exit 1
fi

echo "[play_dash] ${#CLIPS[@]} clips from $NPZ_DIR (filter='${FILTER}', seed=$SEED), checkpoint $CKPT"
printf '  %s\n' "${CLIPS[@]}" | head -20
[ "${#CLIPS[@]}" -gt 20 ] && echo "  ... ($((${#CLIPS[@]} - 20)) more)"

CUDA_VISIBLE_DEVICES=$GPU exec env -u PYTHONPATH .venv/bin/python scripts/play.py \
    --ckpt "$CKPT" \
    --npz-dir "$NPZ_DIR" \
    --clips "${CLIPS[@]}" \
    --num-envs "${#CLIPS[@]}" \
    --viewer "$VIEWER"
