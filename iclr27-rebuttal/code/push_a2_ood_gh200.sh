#!/bin/bash
# Push what the Alpamayo 2 u40 OOD run needs to KISTI Hangang (iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md), from
# cvlab21 over the kisti6-dm socket (OTP: ssh -fN kisti6-dm). Everything goes under /appsdata/e1997a06/a2_ood (the group
# /scratch file quota is ~full; this tree is ~1.7k files):
#   code/amc   run_a2_openloop.py, a2_ood_gh200.sbatch, experiments/head_analysis/eval_lib.py, outputs/eval_sets/{ood_val,ood_train}
#   code/a2d   alpamayo2-distill src/alpamayo2_super + experiments/oodval (no install: the runner puts src/ on sys.path)
#   hub        HF cache of nvidia/Alpamayo2-Super rev 00554695 (29 blobs, 67 GB; symlinks kept)
#   data       physicalai_av/pre_processed/{ood_a2, ood_train_a2} (A2 7-camera caches)
#   masks      masks_{moop,fm_only,ce_only}.npz (u40, iclr27-rebuttal/code/make_a2_moop_masks.py)
# then checks sha256 of masks / manifests / code and the file counts and sizes of the model and caches (gate G0 local side).
#
#   bash iclr27-rebuttal/code/push_a2_ood_gh200.sh           # from the repo root
#   PARTS="code masks" bash ...                              # a subset: code masks hub oodval oodtrain
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down -- run: ssh -fN kisti6-dm"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
SAFE="-o BatchMode=yes -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive"
RS="ssh -o ControlPath=$CM $SAFE"
W=/appsdata/e1997a06/a2_ood
A2D=/home/cvlab21/project/chan/alpamayo2-distill
HUB=/mnt/nvme1n1/ad_vla/cache/hub/models--nvidia--Alpamayo2-Super
PRE=/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed
OUT=$(readlink -f outputs)
PARTS=${PARTS:-code masks hub oodval oodtrain}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
rsh() { $RS "$H" "$@"; }
has() { [[ " $PARTS " == *" $1 "* ]]; }

rsh "mkdir -p $W/code/amc $W/code/a2d $W/hub $W/data/physicalai_av/pre_processed $W/masks $W/outputs $W/queue $W/logs"

if has code; then
  stamp "code"
  mkdir -p "$TMP/amc/iclr27-rebuttal/code" "$TMP/amc/experiments/head_analysis" "$TMP/amc/outputs/eval_sets" "$TMP/a2d/experiments"
  cp iclr27-rebuttal/code/run_a2_openloop.py iclr27-rebuttal/code/a2_ood_gh200.sbatch "$TMP/amc/iclr27-rebuttal/code/"
  cp experiments/head_analysis/eval_lib.py "$TMP/amc/experiments/head_analysis/"
  cp "$OUT/eval_sets/ood_val.parquet" "$OUT/eval_sets/ood_train.parquet" "$TMP/amc/outputs/eval_sets/"
  mkdir -p "$TMP/a2d/src"
  cp -r "$A2D/src/alpamayo2_super" "$TMP/a2d/src/"
  cp -r "$A2D/experiments/oodval" "$TMP/a2d/experiments/"
  find "$TMP" -name __pycache__ -prune -exec rm -rf {} +
  rsync -a --delete -e "$RS" "$TMP/amc/" "$H:$W/code/amc/"
  rsync -a --delete -e "$RS" "$TMP/a2d/" "$H:$W/code/a2d/"
  (cd "$TMP" && find amc a2d -type f | sort | xargs md5sum) > "$TMP/code.local"
  rsh "cd $W/code && find amc a2d -type f | sort | xargs md5sum" > "$TMP/code.remote"
  diff -q "$TMP/code.local" "$TMP/code.remote" > /dev/null && stamp "code: md5 equal ($(wc -l < "$TMP/code.local") files)" \
    || { stamp "code: md5 MISMATCH"; diff "$TMP/code.local" "$TMP/code.remote" || true; exit 1; }
fi

if has masks; then
  stamp "masks"
  rsync -a -e "$RS" "$OUT"/iclr27-rebuttal/a2_moop/masks_{moop,fm_only,ce_only}.npz "$H:$W/masks/"
  (cd "$OUT/iclr27-rebuttal/a2_moop" && sha256sum masks_moop.npz masks_fm_only.npz masks_ce_only.npz) > "$TMP/m.local"
  rsh "cd $W/masks && sha256sum masks_moop.npz masks_fm_only.npz masks_ce_only.npz" > "$TMP/m.remote"
  diff -q "$TMP/m.local" "$TMP/m.remote" > /dev/null && stamp "masks: sha256 equal" || { stamp "masks: sha256 MISMATCH"; exit 1; }
fi

if has hub; then
  stamp "hub: Alpamayo2-Super 67 GB"
  rsync -a --partial --info=stats1 -e "$RS" "$HUB" "$H:$W/hub/"
  (cd "$HUB/blobs" && find . -type f -printf '%f %s\n' | sort) > "$TMP/h.local"
  rsh "cd $W/hub/models--nvidia--Alpamayo2-Super/blobs && find . -type f -printf '%f %s\n' | sort" > "$TMP/h.remote"
  diff -q "$TMP/h.local" "$TMP/h.remote" > /dev/null && stamp "hub: $(wc -l < "$TMP/h.local") blobs, names and sizes equal" \
    || { stamp "hub: blob MISMATCH"; exit 1; }
fi

push_cache() {  # <name> <expected samples>
  local n
  n=$(find "$PRE/$1/samples" -name '*.npz' | wc -l)
  [ "$n" -eq "$2" ] && [ -f "$PRE/$1/index.json" ] || { stamp "$1: local cache has $n / $2 samples -- not pushed"; return 1; }
  stamp "cache $1 ($n samples, $(du -sh "$PRE/$1" | cut -f1))"
  rsync -a --partial --info=stats1 -e "$RS" "$PRE/$1" "$H:$W/data/physicalai_av/pre_processed/"
  (cd "$PRE/$1" && find . -type f -printf '%P %s\n' | sort) > "$TMP/c.local"
  rsh "cd $W/data/physicalai_av/pre_processed/$1 && find . -type f -printf '%P %s\n' | sort" > "$TMP/c.remote"
  diff -q "$TMP/c.local" "$TMP/c.remote" > /dev/null && stamp "$1: $(wc -l < "$TMP/c.local") files, names and sizes equal" \
    || { stamp "$1: MISMATCH"; return 1; }
}
has oodval && push_cache ood_a2 262
has oodtrain && push_cache ood_train_a2 1271
rsh "lfs quota -g sa0713 /appsdata | tail -1" | awk '{printf "  /appsdata group files %s / %s\n", $6, $8}'
stamp "push done"
