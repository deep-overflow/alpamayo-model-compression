#!/bin/bash
# Bring the Alpamayo 2 u40 OOD rows back from KISTI Hangang (iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md):
# outputs (rows, summaries, configs, gate), the queue logs and the holder / task logs, never the model or caches.
# Needs the kisti6-dm socket. Safe to rerun (rsync only adds what changed).
#   bash iclr27-rebuttal/code/pull_a2_ood_gh200.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down -- run: ssh -fN kisti6-dm"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
SAFE="-o BatchMode=yes -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive"
W=/appsdata/e1997a06/a2_ood
DST=$(readlink -f outputs)/iclr27-rebuttal/a2_gh200
mkdir -p "$DST/logs" "$DST/queue"
R=(rsync -a --info=stats1 -e "ssh -o ControlPath=$CM $SAFE")
"${R[@]}" "$H:$W/outputs/" "$DST/"
"${R[@]}" "$H:$W/logs/" "$DST/logs/"
"${R[@]}" --exclude 'claimed/' "$H:$W/queue/" "$DST/queue/"
stamp "pulled into $DST"
for d in "$DST"/a2_ood*_*; do
  [ -d "$d" ] && echo "  ${d##*/}: $(ls "$d"/summary_s*of*.txt 2>/dev/null | wc -l) shard summaries"
done
