#!/usr/bin/env bash
# One-off (user, 2026-10-16): move the large feature caches off the main SSD to /media/kavinder/hdd2/RAM.
# Runs while new jobs are held by outputs/experiments/QUEUE_PAUSE; waits for running jobs to finish first.
set -euo pipefail; cd "$(dirname "$0")/.."
DST=/media/kavinder/hdd2/RAM; OLD_COMPACT=/media/kavinder/hdd2/tfpose_cache
echo "[migrate] $(date +%T) waiting for running jobs to finish"
while pgrep -f '^/home/kavinder/RAM/pose-estimation/\.venv/bin/python scripts/' >/dev/null; do sleep 30; done
echo "[migrate] $(date +%T) start; free: $(df -h --output=avail /media/kavinder/hdd2 | tail -1)"
mkdir -p "$DST"
if [[ -d "$OLD_COMPACT" && ! -e "$DST/compact" ]]; then mv "$OLD_COMPACT" "$DST/compact"; echo "[migrate] compact caches moved (rename)"; fi
rsync -a outputs/shared_cache/ "$DST/shared_cache/"
# repoint links to the compact caches
find "$DST/shared_cache" -maxdepth 1 -type l | while read -r l; do
  t=$(readlink "$l"); n=${t/#$OLD_COMPACT/$DST/compact}; [[ "$t" != "$n" ]] && ln -sfn "$n" "$l"; done
# verify: every regular file present with identical size; links resolve
missing=$(rsync -a --dry-run --itemize-changes outputs/shared_cache/ "$DST/shared_cache/" | grep -v '^\.d\|^cL\|^\.L' | grep -c . || true)
broken=$(find "$DST/shared_cache" -maxdepth 1 -xtype l | wc -l)
echo "[migrate] verify: differing entries=$missing broken links=$broken"
[[ "$missing" == 0 && "$broken" == 0 ]] || { echo "[migrate] VERIFY FAILED - SSD copy kept, pause kept"; exit 1; }
grep -q '^cache_root:' configs/paths.yaml || printf '%s\n' "# Large derived caches (moved off the main SSD 2026-10-16)." "cache_root: $DST/shared_cache" "compact_cache_root: $DST/compact" >> configs/paths.yaml
.venv/bin/python -c "
import sys; sys.path.insert(0,'src'); from tfpose import featsets; from tfpose.cachepaths import CACHE
p=featsets.find_cache('dinov3_vitl16','full'); assert str(p).startswith('$DST'), p
import h5py; f=h5py.File(p/'features.h5'); print('[migrate] read check', p.name, f['image_relpath'].shape)
q=featsets.find_cache('moge2_vitl','exp109_valaug'); print('[migrate] compact link ok', q.resolve())"
mv outputs/shared_cache outputs/shared_cache.ssd_old && ln -s "$DST/shared_cache" outputs/shared_cache
rm -rf outputs/shared_cache.ssd_old
rm -f outputs/experiments/QUEUE_PAUSE
echo "[migrate] $(date +%T) DONE; SSD free: $(df -h --output=avail / | tail -1); hdd2 free: $(df -h --output=avail /media/kavinder/hdd2 | tail -1)"
