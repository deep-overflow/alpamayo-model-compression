#!/bin/bash
# Push the GH200 site's inputs to KISTI Hangang (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.3): push_neuron.sh
# pointed at hangang-dm, plus what its tiers leave out -- the OOD-train half of the ood namespace with its manifest,
# and tyr_off with its rewritten weights (recipe_list ships slim_meta.json only). Needs the kisti6-dm socket (OTP).
cd "$(dirname "$0")/../.." || exit 1
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
S=/scratch/e1997a06
SRC=/mnt/nvme1n1/ad_vla
OUT=$(readlink -f outputs)
R=(rsync -aL --partial --info=stats1 -e "ssh -o ControlPath=$CM")
stamp "push_neuron.sh tiers: code a100eval lingoqa weights"
NEURON_DM=hangang-dm.ksc.re.kr NEURON_CM=$CM CODE_SRC=$PWD PUSH_PY=$SRC/venvs/alpamayo-mc/bin/python \
  bash experiments/transfer/push_neuron.sh e1997a06 code a100eval lingoqa weights || { stamp "push_neuron.sh failed"; exit 1; }
stamp "ood namespace (OOD-train 1,271 + OOD-val 262; --ignore-existing skips the OOD-val files a100eval sent)"
"${R[@]}" --ignore-existing "$SRC/data/physicalai_av/pre_processed/ood" "$H:$S/datasets/physicalai_av/pre_processed/" || exit 1
"${R[@]}" "$OUT/eval_sets/ood_train.parquet" "$H:$S/outputs/eval_sets/" || exit 1
stamp "tyr_off recipe with slim_state.pt"
ssh -o ControlPath=$CM $H "mkdir -p $S/outputs/cvpr27" || exit 1
"${R[@]}" "$OUT/cvpr27/slim_tyr_off" "$H:$S/outputs/cvpr27/" || exit 1
L=$(stat -c %s "$OUT/cvpr27/slim_tyr_off/slim_state.pt")
RS=$(ssh -o ControlPath=$CM $H "stat -c %s $S/outputs/cvpr27/slim_tyr_off/slim_state.pt")
[ "$L" = "$RS" ] || { stamp "tyr_off slim_state size mismatch local $L remote $RS"; exit 1; }
stamp "remote sizes:"
ssh -o ControlPath=$CM $H "du -sh --apparent-size $S/project $S/datasets/physicalai_av/pre_processed/* $S/datasets/lingoqa $S/checkpoints/hub $S/outputs"
stamp "done"
