# Moving recovery training to another server

What has to travel for `experiments/recovery/train_recover.py` to keep running, what
deliberately does not, and the one thing that must not be measured on the new box.

## The short version

```bash
# on this box -- LIST_ONLY resolves every file list and prints volumes, touching nothing
LIST_ONLY=1 bash experiments/transfer/push.sh <user@host> /path/on/target/ad_vla core resume
bash experiments/transfer/push.sh <user@host> /path/on/target/ad_vla core resume

# on the target, after cloning the repo at branch lingoqa-reasoning-probe
bash experiments/transfer/bootstrap.sh /path/on/target/ad_vla
source env.sh
python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2 --load --gpu 0
```

`core` is ~37 GB, `resume` adds ~1.8 GB per in-flight run. A naive copy of everything
the training paths *touch* would be ~150 GB; the difference is entirely the three
reductions in the table below.

## What travels

| item | source | size | why |
|---|---|---:|---|
| Alpamayo-1.5-10B, **revision `7aba8293` only** | `ad_vla/cache/hub` | 21 GB | `slim_lib.MODEL_REV` is pinned; the cache holds six snapshots and the other five are dead weight |
| Cosmos-Reason2-8B, **configs/tokenizer only** | same | 10 MB | see caveat 2 |
| `pre_processed/train/samples` (1,200) | `ad_vla/data/physicalai_av` | 6.8 GB | official half of the CE training set |
| `pre_processed/ood/samples` (1,533) | same | 8.5 GB | OOD-train 1,271 (CE) + ood_val 262 (probe) |
| `pre_processed/eval` — **238 of 18,868** | same | 0.7 GB | only the clips `val_official_238.parquet` names |
| `outputs/eval_sets` + `outputs/recovery_sets` | `ad_vla/outputs/chan` | 4 MB | the manifests that define every set |
| `slim_meta.json` + `config.json` + `summary.txt` per slim arm | same | 160 MB (51 arms) | the complete pruning recipe for every config the repo has built |
| the whole run dir (`resume` tier) | same | 2.1 GB/run | `state_last.pt` + both adapters; `--resume auto` is the trainer default |
| code | GitHub `origin/lingoqa-reasoning-probe` | 45 MB | already pushed at this commit — clone, don't rsync |

`core` totals **37 GB**: 21.0 (weights) + 6.8 (train) + 8.5 (ood) + 0.7 (probe) + 0.16
(recipes) + 0.004 (manifests). `openloop` adds 4.1 GB.

## What does not, and why

- **`slim_state.pt`, 14 GB per arm.** `slim_lib.load_slim` treats it as optional:
  `apply_surgery` slices the base weights in place, so the surgically-modified skeleton
  already *is* the slim checkpoint and the state load is an identity restated for
  safety. The 3 MB `slim_meta.json` reconstructs it bit-for-bit (verified
  tensor-by-tensor). 51 recipes weigh 160 MB against ~714 GB of state files, so the
  target gets *every* config the repo has built for less than the cost of one.
- **The other 18,630 clips of the eval cache, 51 GB.** Training touches the eval
  namespace only for the 238-clip official probe half. Add the `openloop` tier if the
  target is also to run open-loop evaluation — but read caveat 1 first.
- **Raw camera chunk zips.** Every training and probe sample loads through
  `sample_cache.load_cached`, never `load_physical_aiavdataset`. The zips are needed
  only to *build* a cache, which `build_cache.py` already did.
- **`.venv`, 8.1 GB.** `bootstrap.sh` rebuilds it with `uv sync` plus the
  `alpamayo1.5@f42e594` pin. Copy it with the `venv` tier only if the target is also
  CUDA 12.8 — the source venv is `torch 2.8.0+cu128`.
- **`reports/`, `plans/`, alpasim drivers and runs.** Not on the training path.

## Caveats

1. **Evaluation numbers do not transfer across GPU architectures.** Two runs agree
   bitwise only *within* one architecture: the same clip and seed gave 0.286 on Ada and
   0.291 on Blackwell, and 3–4% of clips produce different CoC text. Every published arm
   (`baseline_ada_*`, `dual_u40`, `dual_u55`, u70) was measured on **Ada**. So the new
   box can train freely, but any `run_baseline.py` number it produces is paired-
   comparable only against other arms measured on that same box. The in-training probe
   is exempt — it selects a checkpoint and its absolute value is never reported, which
   is already true of u55 v2 and u70 whose probes ran on Blackwell.

2. **Cosmos-Reason2-8B ships without its weights.** `expert_per_clip` patches the hub
   to force `local_files_only=True` for that repo, so anything it *does* request must be
   in the local cache. Observed reads on the recovery path are the tokenizer and configs
   only — the four safetensors have not been touched since 2026-08-11. If a target run
   fails at model load looking for them, `push.sh ... cosmos` sends the 21 GB, or
   `hf download nvidia/Cosmos-Reason2-8B` fetches them with a gated-access token.

3. **`HF_HOME` and `HF_HUB_CACHE` are separate on purpose.** `HF_HOME` locates the
   token, `HF_HUB_CACHE` locates the blobs. `bootstrap.sh` keeps them apart; never set
   only one. Only the `full_right` token has gated-repo access.

4. **Paths that are now overridable, and paths that still are not.** This transfer made
   three things portable: `sample_cache.AV` reads `$AD_VLA_DATA`, and both retry
   launchers read `$ALPAMAYO_MC_REPO`. Still hardcoded to `/mnt/nvme1n1/ad_vla` and
   irrelevant to training, but they will break if the target ever runs them:
   `build_cache.py`, `build_ood_cache.py`, `make_eval_sets.py`, `eval_ood.py`,
   `make_train_set.py`, `lingo_lib.py`, `run_vqa_importance.py`, `test_quant_lib.py`,
   and the three `launch_alpasim_*.sh`. `build_report.py` additionally hardcodes
   `REPO = /workspace/...`, so pass it absolute paths.

5. **Uncommitted work does not travel.** At the time of writing, the source tree has
   local modifications to `CLAUDE.md`, `launch_arms.sh`, `prune_lib.py`,
   `run_importance.py`, `analyze_recovery.py`, plus untracked `plans/` and `reports/`
   files. `push.sh` sends none of it — commit and push, or rsync the repo directly.

## Restarting an interrupted run on the target

`train_recover.py` defaults to `--resume auto` and reads `state_last.pt`, so the `resume`
tier is enough to continue mid-run:

```bash
source env.sh
nohup bash experiments/recovery/run_ddp_retry.sh 60 "0 1 2 3" \
    experiments/recovery/train_recover.py --ckpt outputs/slim_coc_u55_v2 \
    --exp-id recover_coc_u55 --steps 1200 --accum 4 --lr 1.4e-4 --warmup 50 \
    --val-every 150 --lambda-traj 0.5 --val-fm-clips 100 \
    > logs/recover_coc_u55.log 2>&1 &
```

Global batch is `world_size x --accum`, so a target with a different card count needs
`--accum` adjusted to hold it at 16 — otherwise the run is no longer the same recipe as
the arm it is meant to be compared against.
