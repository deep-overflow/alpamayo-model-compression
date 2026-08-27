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
/bin/bash experiments/transfer/bootstrap_neuron.sh    # /bin/bash, not bash -- caveat 8
# build the venv as bootstrap prints, then
source env.sh                            # not optional -- caveat 11
.venv/bin/python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2
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
   reads `$AD_VLA_DATA`, and both retry launchers read `$ALPAMAYO_REPO`; defaults are
   unchanged, so nothing on this box moved. (This branch first called the variable
   `ALPAMAYO_MC_REPO` while `main` had independently introduced `ALPAMAYO_REPO` for the
   same purpose; it now uses main's name everywhere. An `env.sh` written by an older
   `bootstrap_neuron.sh` still exports the old name — re-run bootstrap, or rename the
   export, before the next `sbatch`.) Still hardcoded to `/mnt/nvme1n1/ad_vla` and
   irrelevant to training, but they will break if NEURON ever runs them:
   `build_cache.py`, `build_ood_cache.py`, `make_eval_sets.py`, `eval_ood.py`,
   `make_train_set.py`, `lingo_lib.py`, `run_vqa_importance.py`, `test_quant_lib.py`,
   and the three `launch_alpasim_*.sh`. `build_report.py` additionally hardcodes
   `REPO = /workspace/...`, so pass it absolute paths.

7. **`github.com:22` is firewalled from NEURON**, so `git clone git@github.com:...`
   hangs until it times out. Port 443 is open on the Datamover, so either clone over
   HTTPS or add `HostName ssh.github.com` / `Port 443` to `~/.ssh/config` there. The
   `code` tier sidesteps GitHub auth entirely and is what shipped here.

8. **NEURON shadows `bash` with a shell function** -- `bash () { /bin/bash --login; }` --
   which discards every argument and starts an interactive login shell. So
   `bash some_script.sh` exits 0 having run nothing at all, with no output and no error.
   Spell out `/bin/bash some_script.sh`, or `./some_script.sh`. This is the single most
   confusing failure on that machine: it looks exactly like a script that silently
   succeeded.

9. **`run_ddp_retry.sh` is not for NEURON.** It polls `nvidia-smi` until cards go idle,
   which is what a shared interactive box needs and a scheduler does not. Use
   `train_recover.sbatch`.

10. **The login node enforces a 2-hour CPU limit, and preflight scales with set size.**
    `ulimit -t` on glogin is 7200 s, and the site guidance is that heavy work belongs on
    a compute node — GPU work always, and anything sustained. Measured for the
    50,000-sample manifest: **22.5 s CPU (15.1 user + 7.3 sys) against 465 s wall**, so
    it is I/O-bound, not CPU-bound, and sits at 0.3% of the cap. The cost that does
    scale is Lustre metadata: one `.exists()` per sample, 51,771 stats for that run,
    against a metadata server the whole machine shares. At 50k it is a few minutes of
    stat traffic; ten times that would not be a neighbourly thing to do on a login node.
    So: `--load` **must** go in an allocation (it builds the 11B model on a GPU), and the
    plain check is fine on the login node at this size but belongs in `srun` as sets grow:

    ```bash
    srun -p amd_a100nv_8 --gres=gpu:1 -t 0:10:00 \
         --comment="field=efficientai;appl=pytorch" \
         .venv/bin/python experiments/transfer/preflight.py --ckpt <ckpt> [--load]
    ```

    The Datamover has no CPU limit and works for the non-`--load` check, but the site
    reserves it for transfers, so prefer `srun` over moving the load there.

11. **A non-interactive `ssh` does not source `env.sh`.** `ssh <USER>@neuron "cd repo &&
    .venv/bin/python experiments/transfer/preflight.py ..."` runs with `AD_VLA_DATA`
    unset, which falls back to this box's `/mnt/nvme1n1/ad_vla/data` — a path that does
    not exist there — so every cache check misses and preflight reports the entire set
    as missing. The header prints what it resolved; read those three lines before
    believing a "50000 missing". Prefix the command with `. ./env.sh &&`.

## Files

| file | side | what it does |
|---|---|---|
| `push_neuron.sh` | source | tiered rsync into `/scratch/$USER`, over one multiplexed OTP session |
| `bootstrap_neuron.sh` | NEURON login | layout check, `outputs` symlink, env build guidance, writes `env.sh` |
| `train_recover.sbatch` | NEURON login | the SLURM job; derives `--accum`, forces `WANDB_MODE=offline` |
| `preflight.py` | NEURON, `srun` for `--load` | verifies manifests, every sample npz the manifest names, the pinned snapshot, the recipe; `--load` rebuilds the model on a GPU. Needs `env.sh` sourced — caveats 10 and 11 |
| `push.sh` | source | generic variant for a target that mirrors this box's layout |
| `lists.sh` | — | shared file-list generators, sourced by both push scripts |
