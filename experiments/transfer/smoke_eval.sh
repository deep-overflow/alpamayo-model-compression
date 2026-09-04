#!/bin/bash
# Prove a NEURON allocation can actually run the evaluation, before a long job claims it.
#
# Runs inside an existing allocation:
#   srun --jobid=<id> --gres=gpu:1 --pty /bin/bash experiments/transfer/smoke_eval.sh
#
# Checks in cost order, and does NOT stop at the first failure -- one run should report
# every problem, because each round trip through the queue is expensive here.
#
#   1 card      what the scheduler actually gave us, and how much VRAM it has (A100 is
#               40 or 80 GB; --reserve-gb has to match)
#   2 imports   timed. /scratch is Lustre and an earlier job died at 1 h having never
#               left the import phase, so a slow number here is the finding.
#   3 offline   the config chain under HF_HUB_OFFLINE=1: Alpamayo's config builds a
#               Cosmos-Reason2-8B processor which chains to Qwen3-VL-8B and -2B, so
#               four repos must be cached, not one.
#   4 baseline  two clips end to end through run_baseline.py -- cache path, model load,
#               rollout, scoring.
#   5 slim      the same for a selection-only recipe, which exercises load_slim
#               rebuilding weights from slim_meta.json with no slim_state.pt present.
#   6 lingoqa   judge.py imports from LINGOQA_BENCH and the val set reads from
#               LINGOQA_DATA; two questions through run_lingo_vqa.
set -u
cd "${ALPAMAYO_REPO:-/scratch/$USER/project/alpamayo-model-compression}"
# shellcheck disable=SC1091
source env.sh
PY=$ALPAMAYO_REPO/.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
fail=0
note() { printf '\n=== %s ===\n' "$*"; }

note "1 card"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail=1

note "2 imports (timed)"
/usr/bin/time -f "  torch+alpamayo1_5 import: %e s" "$PY" -c "
import torch, alpamayo1_5
print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'bf16',
      torch.cuda.is_bf16_supported())" || fail=1

note "3 offline config chain"
"$PY" experiments/transfer/preflight.py --ckpt outputs/slim_dual_u40_v2 2>&1 | tail -12 || fail=1

note "4 baseline, 2 clips"
"$PY" experiments/evaluation/run_baseline.py --set indist --model baseline \
    --limit 2 --exp-id smoke_baseline --gpu 0 --reserve-gb 40 2>&1 | tail -6 || fail=1

note "5 slim_dual_u40_v2, 2 clips (load_slim from slim_meta.json only)"
"$PY" experiments/evaluation/run_baseline.py --set indist --model outputs/slim_dual_u40_v2 \
    --limit 2 --exp-id smoke_dual --gpu 0 --reserve-gb 40 2>&1 | tail -6 || fail=1

note "6 lingoqa paths + 2 questions"
"$PY" -c "
import os, sys
from pathlib import Path
b = Path(os.environ['LINGOQA_BENCH']); d = Path(os.environ['LINGOQA_DATA'])
print('  bench', b, b.is_dir(), '| judge.py', (b / 'judge.py').exists())
print('  data ', d, d.is_dir(), '| val.parquet', (d / 'val.parquet').exists())
sys.path.insert(0, str(b))
import judge  # noqa: F401
print('  judge module imports OK')" || fail=1
"$PY" experiments/lingoqa/run_lingo_vqa.py --arm baseline --limit 2 \
    --exp-id smoke_lingo --gpu 0 2>&1 | tail -6 || fail=1

note "verdict"
[ "$fail" -eq 0 ] && echo "SMOKE OK" || echo "SMOKE FAILED (see above)"
exit "$fail"
