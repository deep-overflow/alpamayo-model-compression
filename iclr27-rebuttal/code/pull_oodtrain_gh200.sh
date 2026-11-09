#!/bin/bash
# Bring the GH200 OOD-train rows of the ICLR27 arms back to cvlab21
# (iclr27-rebuttal/plans/2026-11-09_oodtrain-iclr27-arms-gh200.md, step 3): Hangang deletes /scratch files that
# nobody read for 15 days, so pull as soon as shards finish. Rows, summaries, configs, the queue logs and the
# holder logs; never the recipes. Needs the kisti6-dm socket. Safe to rerun (rsync only adds what changed).
#
#   bash iclr27-rebuttal/code/pull_oodtrain_gh200.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down -- run: ssh -fN kisti6-dm"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
SAFE="-o BatchMode=yes -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive"
S=/scratch/e1997a06
DST=$(readlink -f outputs)/iclr27-rebuttal/gh200
mkdir -p "$DST/logs"
R=(rsync -a --info=stats1 -e "ssh -o ControlPath=$CM $SAFE")
"${R[@]}" --exclude 'claimed/' --exclude 'running/' "$H:$S/outputs/iclr27-rebuttal/gh200/" "$DST/"
"${R[@]}" --include 'gh200_*oodtrain*' --include 'oodtrain_gh200-*' --exclude '*' \
  "$H:$S/project/alpamayo-model-compression/logs/" "$DST/logs/"
stamp "pulled into $DST"
for d in "$DST"/*_oodtrain; do
  [ -d "$d" ] && echo "  ${d##*/}: $(ls "$d"/summary_s*of*.txt 2>/dev/null | wc -l) shard summaries"
done
