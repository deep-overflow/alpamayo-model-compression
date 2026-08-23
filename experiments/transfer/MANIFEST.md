# Split sites: train on KISTI NEURON, evaluate here

Training moves to NEURON; evaluation stays on this box. That split is not a preference,
it is what keeps the results comparable — see caveat 1. This file says exactly what has
to travel for `experiments/recovery/train_recover.py` to run there, what deliberately
does not, and what breaks if the split is ignored.

## The short version

```bash
# 1. one authenticated connection, reused by everything below (NEURON uses OTP)
ssh -fNM -S ~/.ssh/cm-neuron -o ControlPersist=8h <USER>@neuron-dm.ksc.re.kr

# 2. preview, then send  (~37 GB)
LIST_ONLY=1 bash experiments/transfer/push_neuron.sh <USER>
bash experiments/transfer/push_neuron.sh <USER>

# 3. on the NEURON *login* node -- the Datamover neither builds envs nor submits jobs
ssh <USER>@neuron.ksc.re.kr
cd /scratch/$USER/project/alpamayo-model-compression
bash experiments/transfer/bootstrap_neuron.sh
source env.sh
python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2
sinfo                                    # pick a partition with >=40 GB per GPU
sbatch experiments/transfer/train_recover.sbatch --ckpt outputs/slim_coc_u55_v2 \
    --exp-id recover_coc_u55 --steps 1200

# 4. pull the adapter back here to evaluate it
rsync -avP <USER>@neuron-dm.ksc.re.kr:/scratch/<USER>/outputs/recover_coc_u55/ \
    outputs/recover_coc_u55/
```

## What training needs

Two questions, answered concretely.

### Which code

| path | role |
|---|---|
| `experiments/recovery/train_recover.py` | the trainer (entry point) |
| `experiments/recovery/recover_lib.py` | KI-LoRA: insulated FM loss, `load_slim_lora` |
| `experiments/head_analysis/slim_lib.py` | rebuilds the slim model from `slim_meta.json` |
| `experiments/head_analysis/analysis_lib.py` | fused prompt construction, `build_inputs`, `gt_actions` |
| `experiments/head_analysis/expert_per_clip.py` | imported first by every runner — installs the gated-hub patch and `reserve_gpu` |
| `experiments/head_analysis/eval_lib.py` | probe metrics, `coc_degenerate` |
| `experiments/evaluation/sample_cache.py` | reads the npz caches; `clip_seed` |
| `experiments/head_analysis/run_eval.py` | `eval_config_samples`, used by the probe |
| `experiments/recovery/rebuild_merged.py` | merges the adapter afterwards (can also run here) |
| the `alpamayo1_5` package | pinned to `@f42e594`, installed into `.venv`, not in `pyproject.toml` |

`train_recover.py` puts `experiments/{recovery,head_analysis,evaluation}` on `sys.path`,
so those three directories have to arrive together.

Two ways to get them there:

- **`git clone` on the Datamover** (the guide's method A, and the node it sanctions for
  `git`). Clone **`server-transfer`**, not `lingoqa-reasoning-probe`: the former branches
  off the latter at `ed9105f`, so it is a superset — one clone brings the research code
  *and* this toolkit. The repo is private, so the Datamover needs either a read-only
  deploy key or a fine-grained read-only PAT.
- **`push_neuron.sh <USER> code`**, which rsyncs the working tree (90 MB with `.venv`,
  `outputs`, `logs`, `wandb` and worktrees excluded).

The difference is uncommitted work. At the time of writing the source tree has local
edits to `CLAUDE.md`, `launch_arms.sh`, `prune_lib.py`, `run_importance.py` and
`analyze_recovery.py`, plus untracked `plans/` and `reports/` — and **none of them is on
the training path**. `prune_lib` appears in `recover_lib.py` only in comments describing
where the FM convention came from, never as an import. So a clone is sufficient to
train; use rsync instead if you want the working tree mirrored for other reasons.

### Which data

| item | source | size | why |
|---|---|---:|---|
| `pre_processed/train/samples` (1,200) | `ad_vla/data/physicalai_av` | 6.8 GB | official half of the CE training set |
| `pre_processed/ood/samples` (1,533) | same | 8.5 GB | OOD-train 1,271 (CE) + ood_val 262 (probe) — one namespace, ships whole |
| `pre_processed/eval` — **238 of 18,868** | same | 0.7 GB | official half of the checkpoint-selection probe, named by `val_official_238.parquet` |
| Alpamayo-1.5-10B, **revision `7aba8293` only** | `ad_vla/cache/hub` | 21 GB | `slim_lib.MODEL_REV` is pinned; the cache holds six snapshots |
| Cosmos-Reason2-8B, **configs/tokenizer only** | same | 10 MB | see caveat 2 |
| `outputs/recovery_sets` + `outputs/eval_sets` | `ad_vla/outputs/chan` | 4 MB | the manifests that define train and probe |
| `slim_meta.json` (+`config.json`, `summary.txt`) per arm | same | 160 MB (51 arms) | the complete pruning recipe |

**~37 GB total.** A naive copy of everything the training paths touch is ~150 GB.

## What does not travel, and why

- **`slim_state.pt`, 14 GB per arm.** `slim_lib.load_slim` treats it as optional:
  `apply_surgery` slices the base weights in place, so the surgically-modified skeleton
  already *is* the slim checkpoint and the state load is an identity restated for
  safety. The 3 MB `slim_meta.json` reconstructs it bit-for-bit (verified
  tensor-by-tensor). 51 recipes weigh 160 MB against ~714 GB of state files, so NEURON
  gets *every* config the repo has built for less than the cost of one.
- **`val_500` and `test_500`, 4.1 GB, and the other 18,630 eval clips, 51 GB.**
  Evaluation stays here. Training touches the eval namespace only for the 238-clip
  probe half.
- **Raw camera chunk zips.** Every training and probe sample loads through
  `sample_cache.load_cached`, never `load_physical_aiavdataset`. The zips are needed
  only to *build* a cache, which `build_cache.py` already did.
- **`.venv`, 8.1 GB.** `torch 2.8.0+cu128` will not match NEURON's site CUDA stack.
  `bootstrap_neuron.sh` prints both the uv and conda routes; either is fine as long as
  `$REPO/.venv/bin/{python,torchrun}` end up existing, because that is what the
  launchers call.
- **alpasim, `reports/`, closed-loop drivers.** Not on the training path.

## Caveats

1. **Evaluation cannot move, which is the whole reason for the split.** Two runs agree
   bitwise only *within* one GPU architecture: the same clip and seed gave 0.286 on Ada
   and 0.291 on Blackwell, and 3–4% of clips produce different CoC text. Every published
   arm (`baseline_ada_*`, `dual_u40`, `dual_u55`, u70) was measured on **Ada 4–7 here**,
   so a NEURON minADE would not be paired-comparable with any of them. The in-training
   probe is exempt — it selects a checkpoint and its absolute value is never reported,
   which is already true of the u55 v2 and u70 arms whose probes ran on Blackwell.
   Bring the adapter back and evaluate it on Ada.

2. **Cosmos-Reason2-8B ships without its weights.** `expert_per_clip` patches the hub to
   force `local_files_only=True` for that repo, so anything it *does* request must be in
   the local cache. Observed reads on the recovery path are the tokenizer and configs
   only — the four safetensors have not been touched since 2026-08-11. If a NEURON run
   fails at model load looking for them, `push_neuron.sh <USER> cosmos` sends the 21 GB.
   Note that a compute node with no outbound network cannot fall back to a download.

3. **Global batch must stay 16.** It is `world_size x --accum`, and the u55 arms it will
   be compared against used 16. `train_recover.sbatch` derives `--accum` from the GPU
   count and refuses to launch when the count cannot divide 16 exactly, rather than
   silently changing the recipe.

4. **GPU memory.** Each rank held ~43 GB on the source box's 48 GB cards. A 32 GB V100
   partition will OOM at load; pick 80 GB-class cards.

5. **`/scratch` is not backed up** and files untouched for 15 days are purge candidates.
   Pull adapters back here as they are produced rather than leaving them as the only copy.

6. **Paths that are now overridable, and paths that still are not.** `sample_cache.AV`
   reads `$AD_VLA_DATA`, and both retry launchers read `$ALPAMAYO_MC_REPO`; defaults are
   unchanged, so nothing on this box moved. Still hardcoded to `/mnt/nvme1n1/ad_vla` and
   irrelevant to training, but they will break if NEURON ever runs them:
   `build_cache.py`, `build_ood_cache.py`, `make_eval_sets.py`, `eval_ood.py`,
   `make_train_set.py`, `lingo_lib.py`, `run_vqa_importance.py`, `test_quant_lib.py`,
   and the three `launch_alpasim_*.sh`. `build_report.py` additionally hardcodes
   `REPO = /workspace/...`, so pass it absolute paths.

7. **`run_ddp_retry.sh` is not for NEURON.** It polls `nvidia-smi` until cards go idle,
   which is what a shared interactive box needs and a scheduler does not. Use
   `train_recover.sbatch`.

## Files

| file | side | what it does |
|---|---|---|
| `push_neuron.sh` | source | tiered rsync into `/scratch/$USER`, over one multiplexed OTP session |
| `bootstrap_neuron.sh` | NEURON login | layout check, `outputs` symlink, env build guidance, writes `env.sh` |
| `train_recover.sbatch` | NEURON login | the SLURM job; derives `--accum`, forces `WANDB_MODE=offline` |
| `preflight.py` | either | verifies manifests, all 3,271 sample npz, the pinned snapshot, the recipe; `--load` rebuilds the model |
| `push.sh` | source | generic variant for a target that mirrors this box's layout |
| `lists.sh` | — | shared file-list generators, sourced by both push scripts |
