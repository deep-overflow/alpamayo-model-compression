#!/bin/bash
# Runs ON the NEURON datamover: copy the GH200 site's inputs to Hangang over KISTI's internal network
# (cvpr27/plans/2026-11-05_kisti6-gh200-site.md §2.3). On 2026-11-05 cvlab21's path to KISTI dropped to ~20-50 kB/s
# (RTT 300 ms, thousands of retransmits) while NEURON dm -> hangang-dm answered in 22 ms, and NEURON already holds
# every input: the pinned hub snapshot, test_2000 (= the whole `test` namespace, 2,000 clips), the ood namespace
# (OOD-train 1,271 + OOD-Val 262), LingoQA val, the manifests and the recipes.
#
# Needs a NEURON -> Hangang ControlMaster, opened once with Hangang's OTP from cvlab21:
#   ssh -t neuron-dm 'ssh -fNM -o ControlPath=/tmp/cm-e1997a06-hgdm -o ControlPersist=8h -o PubkeyAuthentication=no \
#       -o PreferredAuthentications=keyboard-interactive -o NumberOfPasswordPrompts=1 e1997a06@hangang-dm.ksc.re.kr'
# then, from cvlab21:  ssh neuron-dm /bin/bash /scratch/e1997a06/project/alpamayo-model-compression/cvpr27/code/neuron_to_kisti6.sh
# DRY=1 lists what would move.
set -euo pipefail
CP=/tmp/cm-e1997a06-hgdm
H=e1997a06@hangang-dm.ksc.re.kr
S=/scratch/e1997a06
stamp() { echo "$(date '+%F %T') KST $*"; }   # NEURON's clock is KST
ssh -o ControlPath=$CP -O check $H
R=(rsync -a --partial --info=stats1 -e "ssh -o ControlPath=$CP")
[ -n "${DRY:-}" ] && R+=(-n)
ssh -o ControlPath=$CP $H "mkdir -p $S/checkpoints/hub $S/datasets/physicalai_av/pre_processed $S/datasets/lingoqa \
  $S/project/LingoQA $S/outputs/eval_sets $S/outputs/cvpr27"
stamp "hub: Alpamayo (pinned snapshot) + the three repos its config/processor chain reaches + Lingo-Judge"
"${R[@]}" $S/checkpoints/hub/models--nvidia--Alpamayo-1.5-10B $S/checkpoints/hub/models--nvidia--Cosmos-Reason2-8B \
  $S/checkpoints/hub/models--Qwen--Qwen3-VL-2B-Instruct $S/checkpoints/hub/models--Qwen--Qwen3-VL-8B-Instruct \
  $S/checkpoints/hub/models--wayveai--Lingo-Judge $H:$S/checkpoints/hub/
stamp "caches: test (test_2000) + ood (OOD-train + OOD-Val)"
"${R[@]}" $S/datasets/physicalai_av/pre_processed/test $S/datasets/physicalai_av/pre_processed/ood \
  $H:$S/datasets/physicalai_av/pre_processed/
stamp "LingoQA val (train images stay) + benchmark package"
"${R[@]}" --exclude /train $S/datasets/lingoqa/ $H:$S/datasets/lingoqa/
"${R[@]}" $S/project/LingoQA/benchmark $H:$S/project/LingoQA/
stamp "manifests + recipes (MoOP, MoOP l0keep, tyr_off with slim_state.pt)"
"${R[@]}" $S/outputs/eval_sets/test_2000.parquet $S/outputs/eval_sets/ood_val.parquet \
  $S/outputs/eval_sets/ood_train.parquet $H:$S/outputs/eval_sets/
"${R[@]}" $S/outputs/slim_dual_u40_v2 $S/outputs/slim_dual_u40_l0keep_v2 $H:$S/outputs/
"${R[@]}" $S/outputs/cvpr27/slim_tyr_off $H:$S/outputs/cvpr27/
stamp "sizes on Hangang:"
ssh -o ControlPath=$CP $H "du -sh --apparent-size $S/checkpoints/hub $S/datasets/physicalai_av/pre_processed/test \
  $S/datasets/physicalai_av/pre_processed/ood $S/datasets/lingoqa $S/outputs"
stamp done
