#!/usr/bin/env bash
# The retarget processes leak ~0.1 GB/min each, so recycle the shards every
# 12 minutes until every clip in clip_list.csv has an output.
D="/media/simon/Crucial X10/dash_retarget"
N=${1:-8}
total=$(($(wc -l < "$D/clip_list.csv") - 1))
while true; do
  done_n=$(ls "$D/raw" | grep -vc '\.tmp\.npy$')
  echo "$(date +%T) $done_n / $total"
  [ "$done_n" -ge "$total" ] && break
  prev=$done_n
  timeout --kill-after=30 720 "$D/run_shards.sh" "$N"
  pkill -f '[b]in/retarget task=dash/soma'; sleep 5
  rm -f "$D"/raw/*.tmp.npy
  # Stop if a whole cycle made no progress (remaining clips all fail).
  [ "$(ls "$D/raw" | wc -l)" -le "$prev" ] && { echo "no progress, stopping"; break; }
done
