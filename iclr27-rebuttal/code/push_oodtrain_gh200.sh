#!/bin/bash
# Push what the GH200 OOD-train run of the ICLR27 arms needs to KISTI Hangang
# (iclr27-rebuttal/plans/2026-11-09_oodtrain-iclr27-arms-gh200.md, step 1 and gate G0), from cvlab21.
#   - recipes kisti6 lacks: slim_meta.json + config.json of the selection-only arms (load_slim rebuilds them
#     from the base weights), and tyr_u40_r with its rewritten weights (slim_state.pt, 16 GB)
#   - the code as a 2.4 MB tarball (experiments/, cvpr27/code, src/, pyproject.toml, uv.lock); env.sh and .venv
#     on Hangang are left alone
# then compares sha256 of every pushed recipe file and md5 of the files the holder runs.
#
#   bash iclr27-rebuttal/code/push_oodtrain_gh200.sh            # from the repo root, kisti6-dm socket open
#   TYR_STATE=0 bash ...   skip tyr_u40_r/slim_state.pt (send it from NEURON with cvpr27/code/neuron_to_kisti6.sh)
#
# Needs the kisti6-dm socket (OTP): ssh -fN kisti6-dm. BatchMode everywhere, so a dead socket fails instead of
# trying a fresh login (KISTI locks the account after 5 failures).
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6-dm > /dev/null 2>&1 || { stamp "kisti6-dm socket down -- run: ssh -fN kisti6-dm"; exit 2; }
CM=$HOME/.ssh/cm-kisti6
H=e1997a06@hangang-dm.ksc.re.kr
SAFE="-o BatchMode=yes -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive"
S=/scratch/e1997a06
P=$S/project/alpamayo-model-compression
OUT=$(readlink -f outputs)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
rsh() { ssh -o ControlPath="$CM" $SAFE "$H" "$@"; }

# recipes, relative to outputs/ (moop is already on Hangang; listed so its sha256 is checked too)
RECIPES=(slim_dual_u40_v2 slim_dualexp_u40_em93p75 slim_traj_u40_v2 slim_coc_u40_v2 slim_wanda_u40_v2
         cvpr27/slim_lp_r50 slim_tyr_u40_r)
: > "$TMP/files.txt"
for r in "${RECIPES[@]}"; do
  for f in slim_meta.json config.json; do
    [ -f "$OUT/$r/$f" ] && echo "$r/$f" >> "$TMP/files.txt"
  done
done
[ "${TYR_STATE:-1}" = 1 ] && echo "slim_tyr_u40_r/slim_state.pt" >> "$TMP/files.txt"
stamp "recipes: $(wc -l < "$TMP/files.txt") files"
rsync -aL --partial --info=stats1,progress2 --files-from="$TMP/files.txt" \
  -e "ssh -o ControlPath=$CM $SAFE" "$OUT/" "$H:$S/outputs/"

stamp "code tarball"
CODE=(experiments cvpr27/code src pyproject.toml uv.lock)
tar czf "$TMP/code.tgz" --exclude=__pycache__ "${CODE[@]}"
rsync -a -e "ssh -o ControlPath=$CM $SAFE" "$TMP/code.tgz" "$H:$S/code_oodtrain_gh200.tgz"
rsh "cd $P && tar xzf $S/code_oodtrain_gh200.tgz && rm -f $S/code_oodtrain_gh200.tgz"

stamp "G0: sha256 of the recipes, md5 of the code the holder runs"
(cd "$OUT" && xargs -a "$TMP/files.txt" sha256sum) > "$TMP/local.sha"
rsh "cd $S/outputs && sha256sum $(tr '\n' ' ' < "$TMP/files.txt")" > "$TMP/remote.sha"
RUNS=(cvpr27/code/eval_a100.sbatch cvpr27/code/run_nll_gtcoc.py experiments/evaluation/run_baseline.py
      experiments/head_analysis/slim_lib.py experiments/evaluation/sample_cache.py)
md5sum "${RUNS[@]}" > "$TMP/local.md5"
rsh "cd $P && md5sum ${RUNS[*]}" > "$TMP/remote.md5"
rc=0
diff "$TMP/local.sha" "$TMP/remote.sha" > /dev/null && stamp "recipes: sha256 equal" \
  || { stamp "recipes: sha256 MISMATCH"; diff "$TMP/local.sha" "$TMP/remote.sha" || true; rc=1; }
diff "$TMP/local.md5" "$TMP/remote.md5" > /dev/null && stamp "code: md5 equal" \
  || { stamp "code: md5 MISMATCH"; diff "$TMP/local.md5" "$TMP/remote.md5" || true; rc=1; }
stamp "done rc=$rc"
exit $rc
