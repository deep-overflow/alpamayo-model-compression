"""KI-LoRA recovery trainer for slim Alpamayo-1.5 checkpoints (single- or multi-GPU).

Per micro-step: one grad VLM forward over [prompt + CoC + cot_end + traj_future_start]
(use_cache=True) yields CoC logits and the KV cache; CE on the CoC span trains VLM LoRA.
The cache is detached (recover_lib.fm_loss_insulated), then one flow-matching step with
t ~ U(0,1) trains expert LoRA. The trajectory gradient never reaches the VLM -- that is
the point (Knowledge Insulation); the --ki-check gate at startup asserts it numerically.

Multi-GPU is manual data parallelism, not the DDP wrapper: the forward path calls
submodules directly (vlm.model -> detach -> expert), which DDP's reducer cannot trace.
Each rank accumulates grads on its own shard, then the LoRA grads (~440 MB) are
all-reduce-averaged before the optimizer step -- mathematically identical to one big
batch. The probe is sharded across ranks too and gathered, so probe wall-clock divides
by the world size. Global batch = world_size x --accum.

Data: outputs/recovery_sets/train_official_<n>.parquet (full-model CoC, `train` cache) +
OOD-train rows of outputs/eval_sets/ood.parquet (curated gt_coc, `ood` cache). CE learns
the terminal tokens too, which is the direct lever on the u55 failure mode (CoC
degeneracy 0.714 -- runaway text that never stops).

Probe (every --val-every steps + step 0): K=1 rollout on ood_val 262 + official 238,
mean/median minADE and CoC degeneracy by source. Selection: overall mean minADE.
Adapter-only saves; rebuild_merged.py materializes the merged checkpoint.

Pre-registered gates (plans/2026-08-19_recovery-training.md): G1 recovered test_500
minADE@6 <= 1.6, G2 CoC degen <= 0.05, G3 closed-loop paired d_score CI lower > -0.080.

Usage (4 GPUs, global batch 16 = 4 x accum 4):
  bash experiments/recovery/run_ddp_retry.sh 60 "4 5 6 7" \
      experiments/recovery/train_recover.py --ckpt outputs/slim_dual_u55_v2 \
      --exp-id recover_dual_u55 --steps 300 --accum 4 --lr 1.4e-4 --warmup 25 \
      --val-every 75
  # single GPU (global batch 8):
  bash experiments/head_analysis/run_retry_host.sh 40 experiments/recovery/train_recover.py \
      --ckpt outputs/slim_dual_u55_v2 --exp-id recover_dual_u55 --gpu 7
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import eval_lib as el
import recover_lib as rl
import sample_cache as sc
from alpamayo1_5 import helper
from expert_per_clip import reserve_gpu
from run_eval import eval_config_samples
from transformers import get_cosine_schedule_with_warmup


def dist_init(device_index=None):
    """(rank, world, local_rank); world=1 when not launched by torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        kw = {}
        if device_index is not None:
            torch.cuda.set_device(device_index)
            kw["device_id"] = torch.device(f"cuda:{device_index}")
        dist.init_process_group("nccl", timeout=timedelta(hours=2), **kw)
        return dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    return 0, 1, 0


def wandb_init(args, extra):
    """Rank-0 wandb run; returns a log(dict, step) callable that no-ops on failure."""
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.exp_id,
                         config={**vars(args), **extra})
        return lambda d, step: run.log(d, step=step), run
    except Exception as e:  # noqa: BLE001  logging must never kill training
        print(f"wandb disabled: {e}", flush=True)
        return lambda d, step: None, None


def load_samples(repo):
    """[(cache, clip_id, t0_us, coc_text, source)] for train; probe list likewise."""
    sets = repo / "outputs" / "recovery_sets"
    train = []
    off = pd.read_parquet(max(sets.glob("train_official_*.parquet")))
    for r in off.itertuples():
        train.append(("train", r.clip_id, int(r.t0_us), r.coc, "official"))
    ood = pd.read_parquet(repo / "outputs" / "eval_sets" / "ood.parquet")
    for r in ood[ood.split == "train"].itertuples():
        train.append(("ood", r.clip_id, int(r.t0_us), r.gt_coc, "ood"))

    probe = []
    voff = pd.read_parquet(max(sets.glob("val_official_*.parquet")))
    for r in voff.itertuples():
        probe.append(("eval", r.clip_id, int(r.t0_us), "official"))
    oodv = pd.read_parquet(repo / "outputs" / "eval_sets" / "ood_val.parquet")
    for r in oodv.itertuples():
        probe.append(("ood", r.clip_id, int(r.t0_us), "ood"))
    # fixed shuffle so any --probe-limit prefix mixes both sources
    probe = [probe[i] for i in np.random.default_rng(0).permutation(len(probe))]
    return train, probe


def prepare(base, processor, tok, sample, max_coc, device="cuda"):
    """One training sample -> (inputs, x1, seq_tf, coc_start, coc_end)."""
    cache_ns, clip_id, t0_us, coc_text, _ = sample
    data = sc.load_cached(sc.path_for(cache_ns, clip_id, t0_us))
    inputs = lib.build_inputs(base, processor, data, device)
    x1 = lib.gt_actions(base, data, device).to(torch.float32)  # (1, 64, 2)
    ids = rl.coc_ids(tok, coc_text, max_tokens=max_coc)
    coc = torch.tensor([ids], device=device)  # (1, L_coc)
    seq_tf = torch.cat([inputs["input_ids"], coc], dim=1)  # (1, prompt+coc)
    return inputs, x1, seq_tf, inputs["input_ids"].shape[1], seq_tf.shape[1]


def micro_step(base, inputs, x1, seq_tf, cs, ce, t_val, noise, lam_ce):
    """Insulated losses for one sample. Returns (loss, l_fm, l_ce) tensors."""
    out, cache, rope_deltas = rl.vlm_forward(base, inputs, seq_tf)
    l_ce = rl.ce_loss(base, out, seq_tf, cs, ce)
    prefill = cache.get_seq_length()
    l_fm = rl.fm_loss_insulated(base, cache, rope_deltas, x1, prefill, t_val, noise)
    return l_fm + lam_ce * l_ce, l_fm, l_ce


@torch.no_grad()
def probe_rows(base, processor, tok, samples, seed, max_gen):
    """K=1 rollout probe rows for this rank's shard of the probe set."""
    rows = []
    for cache_ns, clip_id, t0_us, source in samples:
        data = sc.load_cached(sc.path_for(cache_ns, clip_id, t0_us))
        inputs = lib.build_inputs(base, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        cs = sc.clip_seed(seed, clip_id)
        torch.manual_seed(cs)
        torch.cuda.manual_seed_all(cs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(base, inputs, max_generation_length=max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_gen = roll["sequences"][:, :coc_end].clone()
        gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
        gen_coc = tok.decode([t for t in gen_ids if t not in (rl.COT_END, rl.TFS)],
                             skip_special_tokens=True)
        del roll
        ade_k, _, _, _ = eval_config_samples(base, inputs, seq_gen, prompt_len, coc_end,
                                             gt_xy, [cs])
        rows.append({"source": source, "minADE": float(ade_k[0]),
                     "degen": el.coc_degenerate(gen_coc)["degenerate"]})
    return rows


def probe_stats(rows):
    out = {}
    for tag in ("all", "official", "ood"):
        sub = [r for r in rows if tag == "all" or r["source"] == tag]
        if sub:
            out[tag] = {"n": len(sub),
                        "minADE_mean": float(np.mean([r["minADE"] for r in sub])),
                        "minADE_median": float(np.median([r["minADE"] for r in sub])),
                        "degen": float(np.mean([r["degen"] for r in sub]))}
    return out


def ki_check(base, processor, tok, sample, max_coc, vlm_params, expert_params):
    """FM-only backward must leave VLM LoRA untouched; CE-only likewise for the expert."""
    inputs, x1, seq_tf, cs, ce = prepare(base, processor, tok, sample, max_coc)
    noise = torch.randn(x1.shape, device=x1.device)

    _, l_fm, l_ce = micro_step(base, inputs, x1, seq_tf, cs, ce, 0.5, noise, 0.0)
    l_fm.backward()
    leak = [n for n, p in vlm_params if p.grad is not None and p.grad.abs().max() > 0]
    assert not leak, f"KI violated: FM gradient reached VLM LoRA ({leak[:3]})"
    got = sum(1 for _, p in expert_params if p.grad is not None and p.grad.abs().max() > 0)
    assert got > 0, "FM gradient reached no expert LoRA param"

    # l_ce still pins the whole VLM activation graph -- drop it before building a
    # second one, or the two together OOM a 47 GiB card
    del l_fm, l_ce
    for _, p in vlm_params + expert_params:
        p.grad = None
    _, l_fm, l_ce = micro_step(base, inputs, x1, seq_tf, cs, ce, 0.5, noise, 1.0)
    l_ce.backward()
    leak = [n for n, p in expert_params if p.grad is not None and p.grad.abs().max() > 0]
    assert not leak, f"CE gradient reached expert LoRA ({leak[:3]})"
    got = sum(1 for _, p in vlm_params if p.grad is not None and p.grad.abs().max() > 0)
    assert got > 0, "CE gradient reached no VLM LoRA param"
    for _, p in vlm_params + expert_params:
        p.grad = None
    print(f"KI CHECK PASS  (fm={l_fm.item():.4f} ce={l_ce.item():.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--accum", type=int, default=8, help="micro-steps per rank per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-coc", type=int, default=256)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=150)
    ap.add_argument("--probe-limit", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="one entry per rank under torchrun (LOCAL_RANK indexes it)")
    ap.add_argument("--wandb-project", type=str, default="alpamayo-recovery")
    args = ap.parse_args()

    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_env > 1:
        assert args.gpus and len(args.gpus) >= world_env, "--gpus must list one card per rank"
        rank, world, local = dist_init(args.gpus[int(os.environ["LOCAL_RANK"])])
        device = reserve_gpu(args.reserve_gb, devices=[args.gpus[local]])
    else:
        rank, world, local = dist_init()
        device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    r0 = rank == 0
    print(f"[rank {rank}/{world}] using {device}", flush=True)

    out_dir = REPO / "outputs" / args.exp_id
    if r0:
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    peft_model, base, meta = rl.load_slim_lora(REPO / args.ckpt, r=args.r, alpha=args.alpha,
                                               device="cuda")
    lib.set_vlm_attn_impl(base, "sdpa")
    lib.set_expert_attn_impl(base, "sdpa")
    processor = helper.get_processor(base.tokenizer)
    tok = base.tokenizer
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    vlm_params = [(n, p) for n, p in trainable if "language_model.layers" in n]
    expert_params = [(n, p) for n, p in trainable if "expert.layers" in n]
    n_train_p, _ = rl.param_summary(peft_model)
    assert len(vlm_params) + len(expert_params) == len(trainable)

    train, probe_samples = load_samples(REPO)
    if args.probe_limit is not None:
        probe_samples = probe_samples[: args.probe_limit]
    # fixed global shuffle for shard balance, then rank-strided shard
    perm = np.random.default_rng(args.seed).permutation(len(train))
    shard = [train[i] for i in perm][rank::world]
    probe_shard = probe_samples[rank::world]
    n_off = sum(1 for s in train if s[4] == "official")
    if r0:
        print(f"config={meta['config']}  trainable={n_train_p:,}  world={world}  "
              f"global_batch={world * args.accum}", flush=True)
        print(f"train samples: {len(train)} (official {n_off}, ood {len(train) - n_off}); "
              f"{len(shard)}/rank  probe: {len(probe_samples)} ({len(probe_shard)}/rank)",
              flush=True)

    if r0:
        ki_check(base, processor, tok, train[0], args.max_coc, vlm_params, expert_params)
    if world > 1:
        dist.barrier()

    if r0:
        (out_dir / "config.json").write_text(json.dumps({
            "ckpt": args.ckpt, "config": meta["config"],
            "insulation": "fm->expert, ce->vlm",
            "steps": args.steps, "accum": args.accum, "world": world,
            "global_batch": world * args.accum, "gpus": args.gpus or [args.gpu],
            "lr": args.lr, "warmup": args.warmup, "lambda_ce": args.lambda_ce,
            "r": args.r, "alpha": args.alpha, "max_coc": args.max_coc,
            "n_train": len(train), "n_official": n_off, "n_probe": len(probe_samples),
            "trainable_params": n_train_p, "seed": args.seed,
            "gpu": torch.cuda.get_device_name(device),
        }, indent=2))

    wlog, wrun = (wandb_init(args, {"config_name": meta["config"],
                                    "global_batch": world * args.accum,
                                    "trainable_params": n_train_p})
                  if r0 else (lambda d, step: None, None))

    params = [p for _, p in trainable]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    rng = np.random.default_rng([args.seed, rank])
    order, ptr = rng.permutation(len(shard)).tolist(), 0

    def next_sample():
        nonlocal order, ptr
        if ptr >= len(order):
            order, ptr = rng.permutation(len(shard)).tolist(), 0
        s = shard[order[ptr]]
        ptr += 1
        return s

    log = {"step": [], "l_fm": [], "l_ce": [], "lr": [], "val": []}

    def run_probe(step):
        t0 = time.time()
        rows = probe_rows(base, processor, tok, probe_shard, args.seed, args.max_gen)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            rows = [r for part in gathered for r in part]
        res = probe_stats(rows)
        res["step"] = step
        log["val"].append(res)
        if r0:
            wlog({f"val/{s}_{k}": res[s][k] for s in ("all", "official", "ood") if s in res
                  for k in ("minADE_mean", "minADE_median", "degen")}, step)
            a = res["all"]
            by_src = "  ".join(f"{s} {res[s]['minADE_mean']:.4f}/{res[s]['degen']:.3f}"
                               for s in ("official", "ood") if s in res)
            print(f"[val step {step}] minADE {a['minADE_mean']:.4f} "
                  f"(med {a['minADE_median']:.4f}) degen {a['degen']:.3f}  [{by_src}] "
                  f"({(time.time() - t0) / 60:.1f}m)", flush=True)
        return res["all"]["minADE_mean"]

    best, bad = run_probe(0), 0
    if r0:
        rl.save_adapter(peft_model, out_dir / "adapter_best.pt",
                        {"step": 0, "val_minADE": best, "r": args.r, "alpha": args.alpha,
                         "ckpt": args.ckpt})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        fm_acc, ce_acc = 0.0, 0.0
        for _ in range(args.accum):
            sample = next_sample()
            inputs, x1, seq_tf, cs, ce = prepare(base, processor, tok, sample, args.max_coc)
            t_val = float(rng.uniform())
            noise = torch.randn(x1.shape, device=x1.device)
            loss, l_fm, l_ce = micro_step(base, inputs, x1, seq_tf, cs, ce, t_val, noise,
                                          args.lambda_ce)
            (loss / args.accum).backward()
            fm_acc += l_fm.item() / args.accum
            ce_acc += l_ce.item() / args.accum
        if world > 1:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        # the two channels live on disjoint params; clip them separately so one loss's
        # scale cannot rescale the other's update
        torch.nn.utils.clip_grad_norm_([p for _, p in vlm_params], 1.0)
        torch.nn.utils.clip_grad_norm_([p for _, p in expert_params], 1.0)
        opt.step()
        sched.step()

        if step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[rank {rank}] MEMORY GATE: peak {peak:.1f} GB", flush=True)
            assert peak < 46.0, f"rank {rank} peak {peak:.1f} GB exceeds gate"

        if world > 1:
            t_ = torch.tensor([fm_acc, ce_acc], device="cuda")
            dist.all_reduce(t_, op=dist.ReduceOp.AVG)
            fm_acc, ce_acc = float(t_[0]), float(t_[1])
        log["step"].append(step)
        log["l_fm"].append(fm_acc)
        log["l_ce"].append(ce_acc)
        log["lr"].append(sched.get_last_lr()[0])
        if r0:
            wlog({"train/l_fm": fm_acc, "train/l_ce": ce_acc,
                  "train/lr": sched.get_last_lr()[0],
                  "train/sec_per_step": time.time() - t0}, step)
        if r0 and (step % 10 == 0 or step == 1):
            print(f"[step {step}/{args.steps}] L_fm={fm_acc:.5f} L_ce={ce_acc:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s/step)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            v = run_probe(step)
            if r0:
                rl.save_adapter(peft_model, out_dir / "adapter_last.pt",
                                {"step": step, "val_minADE": v, "r": args.r,
                                 "alpha": args.alpha, "ckpt": args.ckpt})
            if v < best - 1e-4:
                best, bad = v, 0
                if r0:
                    rl.save_adapter(peft_model, out_dir / "adapter_best.pt",
                                    {"step": step, "val_minADE": v, "r": args.r,
                                     "alpha": args.alpha, "ckpt": args.ckpt})
                    print(f"  *best -> adapter_best.pt (step {step})", flush=True)
            else:
                bad += 1
                if bad >= args.patience:
                    if r0:
                        print(f"early stop at step {step} (patience {args.patience})",
                              flush=True)
                    break
            if r0:
                (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))

    if r0:
        (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))
        vals = log["val"]
        (out_dir / "summary.txt").write_text(
            f"KI-LoRA recovery {meta['config']} "
            f"steps={log['step'][-1] if log['step'] else 0} "
            f"world={world} global_batch={world * args.accum} lambda_ce={args.lambda_ce}\n"
            f"val minADE(K1): start {vals[0]['all']['minADE_mean']:.4f} -> best {best:.4f}\n"
            f"val degen: start {vals[0]['all']['degen']:.3f} -> "
            f"last {vals[-1]['all']['degen']:.3f}\n"
            f"adapter_best.pt / adapter_last.pt; merge with rebuild_merged.py\n")
        _plot(out_dir, log)
        if wrun is not None:
            wrun.finish()
        print("done ->", out_dir, flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _plot(out_dir, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax[0].plot(log["step"], log["l_fm"], color="#2a78d6", label="L_fm (expert)")
    ax[0].plot(log["step"], log["l_ce"], color="#eda100", label="L_ce (vlm)")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss")
    ax[0].legend(frameon=False)
    ax[0].set_title("training loss (insulated channels)")
    vs = [v["step"] for v in log["val"]]
    for i, (key, tt) in enumerate([("minADE_mean", "val minADE (K=1 rollout)"),
                                   ("degen", "val CoC degeneracy")], start=1):
        for src, c in (("all", "#29261B"), ("official", "#008300"), ("ood", "#e87ba4")):
            ax[i].plot(vs, [v[src][key] for v in log["val"]], "o-", ms=3.5, color=c, label=src)
        ax[i].set_xlabel("step")
        ax[i].set_title(tt)
        ax[i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "training.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
