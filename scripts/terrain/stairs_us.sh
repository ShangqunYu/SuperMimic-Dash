#!/usr/bin/env bash
# Stair clips re-edited onto US building stairs (risers 12-20 cm, treads 25.4-32 cm; IBC: riser <= 17.8 cm,
# tread >= 27.9 cm; IRC: riser <= 19.7 cm, tread >= 25.4 cm). The source stairs are ~12.4 cm / 17.2 cm at
# Dash scale, so each clip is slowed by run / 0.172 (the source's speed along the stairs). Walking clips
# and their mirrors (jogging has too few footholds). Run from the repo root:
#   bash scripts/terrain/stairs_us.sh [parallel jobs] [clip name patterns...]
# then QA with make_manifest.py --scores logs/terrain_edit_us.
set -euo pipefail
JOBS=${1:-8}
shift || true
PATTERNS=("$@")
[ ${#PATTERNS[@]} -eq 0 ] && PATTERNS=(stairs_climbing_up_start stairs_climbing_down_stop stairs_climbing_down_loop)
OUT=logs/terrain_edit_us
SAVE=data/terrain/edited_us
mkdir -p "$OUT" "$SAVE"
for rise in 0.12 0.15 0.178 0.20; do
  for run in 0.254 0.279 0.32; do
    slow=$(python3 -c "print(round(max(1.0, $run / 0.172), 3))")
    for pat in "${PATTERNS[@]}"; do
      echo "$rise $run $slow $pat"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c 'name=$(echo "$3" | tr -c "A-Za-z0-9_\n" "x"); OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/terrain/terrain_edit.py \
  --raw data/terrain/raw --meta data/terrain/meta.csv --only "$3" --rise "$0" --run "$1" --slow "$2" --reach --follow-support \
  --tag "_us_r${0#0.}_t${1#0.}" --out '"$OUT"'/r$0_t$1_$name --save '"$SAVE"' --no-render > '"$OUT"'/r$0_t$1_$name.log 2>&1'
