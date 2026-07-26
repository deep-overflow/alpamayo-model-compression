"""LoRA recovery trainer (Design B) for slim Alpamayo-1.5 checkpoints.

Per micro-step: one grad VLM forward over [prompt + teacher CoC] (use_cache=True) yields
an attached KV cache; the expert runs a single flow-matching step on it; a combined
backward L_fm (+ lambda*L_coc) sends the trajectory gradient through expert LoRA AND VLM
LoRA via the cache. No gradient checkpointing (needs use_cache); measured peak ~35 GB.

Teacher CoC (full-model, from make_teacher_refs) is the cache context for every config,
matching the eval protocol and avoiding a slow per-step rollout. CoC CE (cocsafe only)
uses the same teacher tokens as target to protect reasoning.

After training, LoRA is merged into the slim weights and re-saved in slim_lib format, so
graphed latency and every eval path apply unchanged.

Usage (PYTORCH_CUDA_ALLOC_CONF unset is fine here; no CUDA graph):
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/train_lora.py \
      --ckpt outputs/slim_integrated_mag --loss fm --exp-id lora_int --gpu 0
  ... --ckpt outputs/slim_cocsafe_r20 --loss fm_coc --lambda-coc 0.5 --exp-id lora_coc20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import lora_lib as ll  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  gated-repo hub patch
from run_eval import eval_config  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.token_utils import to_special_token  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

REPO = Path("/workspace/alpamayo-model-compression")


def load_refs(refs_dir, split):
    """Merge teacher-ref shards for a split into {clip_id: {prompt_len, coc_ids}}."""
    refs = {}
    for f in sorted((REPO / "outputs" / refs_dir).glob(f"{split}_*.json")):
        refs.update(json.loads(f.read_text()))
    return refs


def prepare_clip(model, processor, clip_id, ref, device):
    """Build (inputs, x1, seq_tf, coc_start, coc_end, gt_xy) for one clip from a teacher ref."""
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    inputs = lib.build_inputs(model, processor, data, device)
    prompt_len = inputs["input_ids"].shape[1]
    x1 = lib.gt_actions(model, data, device).to(torch.float32)  # (1, 64, 2)
    coc = torch.tensor([ref["coc_ids"]], device=device)  # (1, L_coc)
    seq_tf = torch.cat([inputs["input_ids"], coc], dim=1)  # (1, prompt+coc)
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
    return inputs, x1, seq_tf, prompt_len, seq_tf.shape[1], gt_xy


def train_micro(base, inputs, x1, seq_tf, coc_start, coc_end, eos_id, lam_coc):
    """One Design-B micro-step. Returns (L_fm, L_coc-or-None) as floats after backward setup."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = base.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        l_fm = ll.ph21.flow_matching_loss(base, cache, rope_deltas, seq_tf, eos_id, x1)
        loss = l_fm
        l_coc = None
        if lam_coc > 0:
            logits = base.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
            l_coc = F.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end])
            loss = l_fm + lam_coc * l_coc
    return loss, l_fm, l_coc


@torch.no_grad()
def val_probe(base, processor, refs, clips, eos_id, seed=123, k=1):
    """K=1 minADE + NLL over a few val clips (teacher CoC context)."""
    ades, nlls = [], []
    for ci, clip_id in enumerate(clips):
        if clip_id not in refs:
            continue
        inputs, _, seq_tf, cs, ce, gt_xy = prepare_clip(base, processor, clip_id,
                                                        refs[clip_id], "cuda")
        ade, _, nll = eval_config(base, inputs, seq_tf, cs, ce, gt_xy, [seed + ci])
        ades.append(ade)
        nlls.append(nll)
    return float(np.mean(ades)), float(np.mean(nlls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--loss", choices=["fm", "fm_coc"], default="fm")
    ap.add_argument("--lambda-coc", type=float, default=0.5)
    ap.add_argument("--refs", type=str, default="teacher_refs")
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-clips", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve-gb", type=float, default=44.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    lam_coc = args.lambda_coc if args.loss == "fm_coc" else 0.0
    peft_model, base, meta = ll.load_slim_lora(REPO / args.ckpt, r=args.r, alpha=args.alpha,
                                               device="cuda")
    lib.set_vlm_attn_impl(base, "sdpa")
    lib.set_expert_attn_impl(base, "sdpa")
    processor = helper.get_processor(base.tokenizer)
    eos_id = base.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    n_train, n_total = ll.param_summary(peft_model)
    print(f"config={meta['config']}  loss={args.loss} lam={lam_coc}  trainable={n_train:,}",
          flush=True)

    train_refs = load_refs(args.refs, "train")
    val_refs = load_refs(args.refs, "val")
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    train_clips = [c for c in split["train"] if c in train_refs]
    val_clips = [c for c in split["val"] if c in val_refs][: args.val_clips]
    print(f"train clips with refs: {len(train_clips)}  val: {len(val_clips)}", flush=True)
    assert len(train_clips) > 0 and len(val_clips) > 0, "missing teacher refs"

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train_clips)).tolist()
    ptr = 0

    def next_clip():
        nonlocal ptr, order
        if ptr >= len(order):
            order = rng.permutation(len(train_clips)).tolist()
            ptr = 0
        c = train_clips[order[ptr]]
        ptr += 1
        return c

    (out_dir / "config.json").write_text(json.dumps({
        "ckpt": args.ckpt, "config": meta["config"], "loss": args.loss, "lambda_coc": lam_coc,
        "steps": args.steps, "accum": args.accum, "lr": args.lr, "r": args.r, "alpha": args.alpha,
        "n_train_params": n_train, "n_train_clips": len(train_clips), "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    log = {"step": [], "l_fm": [], "l_coc": [], "lr": [], "val_step": [], "val_ade": [],
           "val_nll": []}
    best_ade, best_state, bad = float("inf"), None, 0
    base_ade, base_nll = val_probe(base, processor, val_refs, val_clips, eos_id, k=1)
    print(f"[val step 0 | slim] minADE(K1)={base_ade:.4f} NLL={base_nll:.4f}", flush=True)
    log["val_step"].append(0); log["val_ade"].append(base_ade); log["val_nll"].append(base_nll)

    for step in range(1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        fm_acc, coc_acc = 0.0, 0.0
        for _ in range(args.accum):
            clip_id = next_clip()
            inputs, x1, seq_tf, cs, ce, _ = prepare_clip(base, processor, clip_id,
                                                         train_refs[clip_id], "cuda")
            loss, l_fm, l_coc = train_micro(base, inputs, x1, seq_tf, cs, ce, eos_id, lam_coc)
            (loss / args.accum).backward()
            fm_acc += l_fm.item() / args.accum
            coc_acc += (l_coc.item() if l_coc is not None else 0.0) / args.accum
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        if step == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"MEMORY GATE: peak {peak:.1f} GB", flush=True)
            assert peak < 46.0, f"peak {peak:.1f} GB exceeds gate"

        log["step"].append(step); log["l_fm"].append(fm_acc)
        log["l_coc"].append(coc_acc); log["lr"].append(sched.get_last_lr()[0])
        if step % 20 == 0 or step == 1:
            print(f"[step {step}/{args.steps}] L_fm={fm_acc:.5f} L_coc={coc_acc:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s/step)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            v_ade, v_nll = val_probe(base, processor, val_refs, val_clips, eos_id, k=1)
            log["val_step"].append(step); log["val_ade"].append(v_ade); log["val_nll"].append(v_nll)
            improved = v_ade < best_ade - 1e-4
            print(f"[val step {step}] minADE(K1)={v_ade:.4f} NLL={v_nll:.4f} "
                  f"{'*best' if improved else ''}", flush=True)
            if improved:
                best_ade = v_ade
                best_state = {k: v.detach().cpu().clone()
                              for k, v in peft_model.state_dict().items() if "lora_" in k}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at step {step} (patience {args.patience})", flush=True)
                    break
            (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))

    # restore best adapters, merge, save
    if best_state is not None:
        sd = peft_model.state_dict()
        sd.update({k: v.to("cuda") for k, v in best_state.items()})
        peft_model.load_state_dict(sd)
    merged_dir = out_dir / "merged"
    ll.merge_save(peft_model, base, meta, merged_dir)
    (out_dir / "metrics.json").write_text(json.dumps(log, indent=2))
    (out_dir / "summary.txt").write_text(
        f"LoRA recovery {meta['config']} loss={args.loss} lam={lam_coc}\n"
        f"slim minADE(K1) {base_ade:.4f} -> best {best_ade:.4f} (val {len(val_clips)} clips)\n"
        f"merged -> {merged_dir}\n")

    _plot(out_dir, log, base_ade)
    print(f"done. merged slim+LoRA -> {merged_dir}", flush=True)


def _plot(out_dir, log, base_ade):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log["step"], log["l_fm"], color="#2a78d6", label="L_fm")
    if any(log["l_coc"]):
        ax[0].plot(log["step"], log["l_coc"], color="#eda100", label="L_coc")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(frameon=False)
    ax[0].set_title("training loss")
    ax[1].plot(log["val_step"], log["val_ade"], "o-", color="#008300")
    ax[1].axhline(base_ade, ls="--", color="#6B6555", label="slim (start)")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("val minADE (K1)"); ax[1].legend(frameon=False)
    ax[1].set_title("val recovery")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "training.png", dpi=150)


if __name__ == "__main__":
    main()
