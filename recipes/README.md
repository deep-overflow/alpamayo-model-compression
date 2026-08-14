# Pruning recipes — how to reproduce and evaluate a compressed Alpamayo 1.5

A **recipe** is `slim_meta.json`: the list of Q-head and MLP-channel indices kept in every
layer of both towers. It is 3.4 MB and reconstructs the pruned model **exactly** — the
16.8 GB `slim_state.pt` in `outputs/slim_*/` is not needed and never has to be copied.

That is not an approximation. `apply_surgery()` slices the base weights in place, so
`base + surgery(meta)` *is* the checkpoint. Verified tensor-by-tensor against the saved
state file: 1,159 tensors, 8,421,074,162 params, **0 mismatches**
(`VERDICT: IDENTICAL`).

## What is here

| recipe | criterion | needs reference CoC text |
|---|---|---|
| `traj_u40_v2` | `I_traj` — trajectory flow-matching Taylor | no |
| `coc_u40_v2` | `I_CoC` — CoC NLL Taylor | **yes** |
| `j_u40_v2` | `J` — J-lens, label-free | no |
| `dual_u40_v2` | `max(rank I_traj, rank I_CoC)` | **yes** |
| `jtraj_u40_v2` | `max(rank I_traj, rank J)` | no |

All five are the same size: 11,078,526,194 → 8,421,074,162 params (**−24.0%**), uniform
across depth — every one of the 36 VLM layers keeps 19/32 Q heads and 7390/12288 MLP
channels. The expert tower and all KV groups are untouched. They differ **only** in which
units are kept. `dual_u40_v2` is the strongest arm in both open- and closed-loop.

`config.json` in each directory records the provenance: base revision, param counts, and
the smoke-test result from the build.

## Prerequisites on this box

Nothing has to be downloaded. The base weights are already in the shared cache and the
recipe is in this repo.

```bash
export HF_HOME=$HOME/.cache/huggingface              # locates your token
export HF_HUB_CACHE=/mnt/nvme1n1/ad_vla/cache/hub    # locates the 22 GB of blobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID                  # --gpu N == nvidia-smi -i N
```

Both variables are needed and they point at **different** places: the shared cache holds
the blobs but not a token with gated access. If you have your own HF account with
`nvidia/Alpamayo-1.5-10B` approved, `HF_HOME` should point at your own cache directory.
If you do not, the blobs are already local, so `HF_HUB_OFFLINE=1` works instead of a
token.

A venv with `alpamayo1_5 @ f42e594`, `transformers==4.57.1`, `torch 2.8.0+cu128` lives at
`/mnt/nvme1n1/ad_vla/venvs/alpamayo-mc` (this repo's `.venv` symlink). It is world-readable;
`uv sync` in your own checkout builds an equivalent one.

Inference needs **≥24 GB of VRAM**. `run_baseline.py --reserve-gb 26` reserves it up front.

### `outputs/` in a fresh clone

A new checkout has no `outputs/`, and `run_baseline.py` reads its manifests from
`outputs/eval_sets/` and writes results to `outputs/<exp-id>/`. Give yourself your own
results directory and link only the shared inputs into it, so runs never land in someone
else's tree:

```bash
mkdir -p /mnt/nvme1n1/ad_vla/outputs/<you> && ln -s /mnt/nvme1n1/ad_vla/outputs/<you> outputs
ln -s /mnt/nvme1n1/ad_vla/outputs/chan/eval_sets outputs/eval_sets   # manifests, read-only
```

`/mnt/nvme1n1` is shared and was 87% full (933 GB free) on 2026-08-13 — check `df` before
a run that writes much, and note that `du` over that tree double-counts heavily because a
lot of it is hardlinked.

## Loading a recipe

```python
import sys
sys.path.insert(0, "experiments/head_analysis")
import expert_per_clip          # MUST be imported first -- installs a hub patch that
                                # forces local_files_only for a second gated repo
import slim_lib as sl

model = sl.load_slim("recipes/dual_u40_v2", device="cuda")   # ~1 min, no state file
```

`load_slim` pins the base revision (`slim_lib.MODEL_REV`) because several snapshots of
the gated repo are cached on this box and an unpinned `from_pretrained` picks whichever
one resolves first.

`from_pretrained` / `save_pretrained` do **not** work on these checkpoints: layers can
have different shapes and the HF format cannot express that. `load_slim` is the only
entry point.

## Evaluating

```bash
bash experiments/head_analysis/run_retry_host.sh 480 \
  experiments/evaluation/run_baseline.py \
  --set test --model recipes/dual_u40_v2 --exp-id mytag_test \
  --gpu 4 --reserve-gb 26
```

`--set` is `indist` (official val, 500 clips), `test` (official test, 500), or `ood`
(1,533 long-tail clips carrying curated reference reasoning). `--model baseline` runs the
unpruned model. Results land in `outputs/<exp-id>/` as `config.json` / `metrics.json` /
`summary.txt` / `plots/*.png`.

Per-clip inputs are pre-cached under
`/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/` (63 GB, ~0.4 s per clip), so a
500-clip set is ~15 min on one card and the full 2,533 clips ~2 h on four.

To split a set across GPUs, use the queue rather than assigning cards by hand:

```bash
bash experiments/evaluation/launch_arms.sh init 4 mytag=recipes/dual_u40_v2
bash experiments/evaluation/launch_arms.sh worker 4 &   # one per free card
bash experiments/evaluation/launch_arms.sh status
```

Compare against the unpruned model with `analyze_baseline.py --compare`, or several arms
at once with `analyze_arms.py` (which also evaluates the pre-registered gates).

## Four things that will bite

1. **Determinism holds only within one GPU architecture.** The same clip and seed gave
   0.286 on Ada and 0.291 on Blackwell, and 3–4% of clips produce different CoC text
   across architectures. This box has Ada on cards 4–7 and Blackwell on 0–3. Keep every
   arm of a comparison on one architecture and record which — all published numbers in
   this track are **Ada**.
2. **`expert_per_clip` must be imported before anything touches the hub.** It patches
   `transformers.utils.hub.cached_files` to force `local_files_only=True` for
   `nvidia/Cosmos-Reason2-8B`, which is gated and served from the local cache. Import it
   first or the load fails with a 401 that looks unrelated.
3. **Seeds come from the clip, not the loop index.**
   `sample_cache.clip_seed(seed, clip_id) = sha256(f"{seed}:{clip_id}")[:4]`, and sample
   *k* uses `base + k`. A K′-sample run is therefore a strict prefix of a K-sample run, and
   sharding does not change any clip's result.
4. **minADE here is a full rollout, not teacher-forced.** The model generates its own CoC
   and the trajectory is conditioned on that, so a criterion that damages reasoning shows
   up in minADE too. On the OOD set each clip is evaluated twice — own rollout, and the
   curated `gt_coc` teacher-forced — which is the only place the two channels can be told
   apart.

## Rebuilding the 16.8 GB state file (usually unnecessary)

Only needed to hand a checkpoint to a machine that cannot reach the base weights, or to
hardlink a driver directory for closed-loop alpasim runs:

```bash
bash experiments/head_analysis/run_retry_host.sh 20 \
  experiments/head_analysis/make_slim.py \
  --config dual_u40_v2 --importance importance_v2 --jlens jlens_v2 \
  --gpu 7 --out outputs/slim_dual_u40_v2
```

~12 min, of which the surgery is 0 s and the save 37 s; the rest is loading the base
model. This path recomputes the masks from the calibration measurements rather than
reading the recipe, which is how the recipe was produced in the first place.

## Where the numbers are

- `reports/evaluation/2026-08-12_results-tables.html` — every open-loop and closed-loop
  number for all six arms, tables only.
- `reports/evaluation/2026-08-12_criterion-decomposition.html` — what the five arms mean.
- `outputs/eval_sets/EVAL_SETS.md` — how the evaluation sets were drawn.
