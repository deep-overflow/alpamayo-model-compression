#!/bin/bash
# Push the GH200 site's inputs to KISTI Hangang (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.3): push_neuron.sh
# pointed at hangang-dm, plus what its tiers leave out -- the OOD-train half of the ood namespace with its manifest,
# and tyr_off with its rewritten weights (recipe_list ships slim_meta.json only). Needs the kisti6-dm socket (OTP).
#
# Five rsyncs run side by side: one ssh stream to KISTI tops out near 8 MB/s (2026-11-05), and the link can sag to
# tens of kB/s for hours -- then use neuron_to_kisti6.sh instead. The code goes separately (a 2.4 MB tarball of
# experiments/, cvpr27/code, src/, pyproject.toml, uv.lock) unless TIERS_CODE=1.
cd "$(dirname "$0")/../.." || exit 1
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
# never a fresh login if the socket dies mid-run: BatchMode + keyboard-interactive only (no keys, no empty passwords)
SAFE="-o BatchMode=yes -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive"
S=/scratch/e1997a06
SRC=/mnt/nvme1n1/ad_vla
OUT=$(readlink -f outputs)
LOGD=${LOGD:-/tmp/push_kisti6_$$}
mkdir -p "$LOGD"
R=(rsync -aL --partial --info=stats1 -e "ssh -o ControlPath=$CM $SAFE")
pushn() { NEURON_DM=hangang-dm.ksc.re.kr NEURON_CM=$CM CODE_SRC=$PWD PUSH_PY=$SRC/venvs/alpamayo-mc/bin/python \
  bash experiments/transfer/push_neuron.sh e1997a06 "$@"; }
ssh -o ControlPath=$CM $SAFE $H "mkdir -p $S/outputs/cvpr27 $S/outputs/eval_sets $S/datasets/physicalai_av/pre_processed" || exit 1
stamp "logs in $LOGD"
[ -n "${TIERS_CODE:-}" ] && { pushn code > "$LOGD/code.log" 2>&1 || stamp "code tier failed"; }
pushn weights > "$LOGD/weights.log" 2>&1 & P1=$!
pushn a100eval > "$LOGD/a100eval.log" 2>&1 & P2=$!
pushn lingoqa > "$LOGD/lingoqa.log" 2>&1 & P3=$!
# the whole ood namespace: OOD-train 1,271 + OOD-Val 262 (a100eval also sends the 262 -- --ignore-existing, and rsync
# renames a finished temp file into place, so the overlap only costs bytes)
{ "${R[@]}" --ignore-existing "$SRC/data/physicalai_av/pre_processed/ood" "$H:$S/datasets/physicalai_av/pre_processed/" \
  && "${R[@]}" "$OUT/eval_sets/ood_train.parquet" "$H:$S/outputs/eval_sets/"; } > "$LOGD/ood.log" 2>&1 & P4=$!
"${R[@]}" "$OUT/cvpr27/slim_tyr_off" "$H:$S/outputs/cvpr27/" > "$LOGD/tyr_off.log" 2>&1 & P5=$!
rc=0
for p in "weights $P1" "a100eval $P2" "lingoqa $P3" "ood $P4" "tyr_off $P5"; do
  set -- $p
  if wait "$2"; then stamp "$1 ok"; else stamp "$1 FAILED (see $LOGD/$1.log)"; rc=1; fi
done
L=$(stat -c %s "$OUT/cvpr27/slim_tyr_off/slim_state.pt")
RS=$(ssh -o ControlPath=$CM $SAFE $H "stat -c %s $S/outputs/cvpr27/slim_tyr_off/slim_state.pt")
[ "$L" = "$RS" ] || { stamp "tyr_off slim_state size mismatch local $L remote $RS"; rc=1; }
stamp "remote sizes:"
ssh -o ControlPath=$CM $SAFE $H "du -sh --apparent-size $S/datasets/physicalai_av/pre_processed/* $S/datasets/lingoqa \
  $S/checkpoints/hub $S/outputs; ls $S/datasets/physicalai_av/pre_processed/test/samples | wc -l; \
  ls $S/datasets/physicalai_av/pre_processed/ood/samples | wc -l"
stamp "done rc=$rc"
exit $rc
