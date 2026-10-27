#!/usr/bin/env bash
# Retarget clip_list.csv into raw/ with N parallel single-core shards.
# Resumable: clips whose output exists are skipped. Usage: run_shards.sh [N] [LIMIT]
set -u
D="/media/simon/Crucial X10/dash_retarget"
N=${1:-14}; LIMIT=${2:-0}
cd ~/Repositories/mocap_retargeting
mkdir -p "$D/shards" "$D/raw" "$D/logs"
tail -n +2 "$D/clip_list.csv" | cut -d, -f1 > "$D/shards/all.txt"
[ "$LIMIT" -gt 0 ] && head -n "$LIMIT" "$D/shards/all.txt" > "$D/shards/tmp" && mv "$D/shards/tmp" "$D/shards/all.txt"
rm -f "$D"/shards/shard_*.txt
# Round-robin over the longest-first list keeps shard workloads even.
awk -v n="$N" -v d="$D/shards" '{print > (d "/shard_" (NR-1)%n ".txt")}' "$D/shards/all.txt"
for k in $(seq 0 $((N-1))); do
  [ -f "$D/shards/shard_$k.txt" ] || continue
  pixi run retarget task=dash/soma \
    "mocap_src_path='/media/simon/Crucial X10/bones-seed'" \
    "+mocap_list_file='$D/shards/shard_$k.txt'" \
    "+kin_output_dir='$D/raw'" +save_mocap=false +skip_existing=true \
    jit_recompile=false n_motion_chunk_workers=1 refine_pool_workers=1 \
    refine_save_plot=false \
    "+output_dir='$D/logs/shard_$k'" "hydra.run.dir='$D/logs/hydra_$k'" \
    > "$D/logs/shard_$k.log" 2>&1 &
done
wait
echo "done: $(ls "$D/raw" | wc -l) clips"
