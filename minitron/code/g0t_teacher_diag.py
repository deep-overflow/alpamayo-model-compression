"""G0-T: does the dense teacher give the distillation targets the Minitron KD will use?

Plan minitron/plans/2026-10-23_plan.md, D2 and D6. Both questions are about the unpruned
Alpamayo-1.5-10B, the Minitron teacher.

(a) The 128 discrete trajectory tokens get a logit KL in the KD loss (D2), but the release
    never decodes them -- the action comes from the expert, and the release masks this
    vocabulary range during CoC decoding. Is the head alive? Teacher-forced on the GT tokens:
    NLL over the 3000-bin range, top-1 bin accuracy, mean bin error. Free-running: greedy
    128 tokens restricted to the bin range -> traj_tokenizer.decode -> ADE against GT, next
    to the expert's own sample on the same CoC and the round-trip ADE of the GT bins
    (the quantisation floor of the discrete representation).
(b) Teacher correction was skipped (D6). Recorded, not gated: the teacher's GT
    flow-matching loss on the inference grid t = 0, 0.1, ..., 0.9 (the KD query points,
    D1-2) and its trajectory-token NLL, on distillation-train windows vs official val.

Both sets get a fresh seeded CoC rollout (seed from the clip, sc.clip_seed), re-encoded
with recover_lib.coc_ids exactly as the KD will build its sequences, so the CoC source is
the same on both sides. Train windows follow the KD rule R1 (mt_data.r1_windows) and come
from the existing `train` cache; val is outputs/recovery_sets/val_official_238 (official
val, disjoint from val500).

Pre-registered gates:
  G0-T(a) PASS iff on BOTH sets
          (i)  median per-sample trajectory-token NLL (3000-bin range) < H_marg - 1 nat,
               H_marg = plug-in entropy of the pooled GT bin marginal, averaged over the two
               action dims (accel, curvature) -- what a head that knew only bin
               frequencies would score; and
          (ii) median greedy-discrete ADE <= 2 x median single-sample expert ADE.
          FAIL -> take the trajectory-token KL (D2) back to the user before train_kd.
  G0-T(b) recorded only: relative train-vs-val gap of mean FM loss and mean trajectory NLL,
          flagged when |gap| > 10%.

Usage (from a worktree, which needs its own .venv / outputs links):
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 30 \\
      minitron/code/g0t_teacher_diag.py --gpu 4
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))

import analysis_lib as lib
import eval_lib as el
import mt_data
import recover_lib as rl
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import MODEL_REV, seed_all, set_determinism
from run_eval import eval_config_samples

T_GRID = tuple(round(0.1 * i, 1) for i in range(10))  # inference Euler grid 0.0 .. 0.9
K_EXPERT = 6  # expert samples per clip; [0] is the single-sample reference, min is @6


class BinRangeOnly(LogitsProcessor):
    """Greedy over the future-trajectory bins only (the inverse of ExpertLogitsProcessor)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __call__(self, input_ids, scores):
        scores[:, : self.lo] = -float("inf")
        scores[:, self.hi:] = -float("inf")
        return scores


def windows(n_train, n_val, seed):
    """[(set, cache namespace, clip_id, t0_us)]: R1 train windows in cache + official val."""
    rng = np.random.default_rng(seed)
    r1 = mt_data.r1_windows()
    r1 = r1[[sc.path_for("train", c, int(t)).exists() for c, t in zip(r1.clip_id, r1.t0_us)]]
    # shuffle, then the first window of each clip: a random window per clip, random clip order
    one = r1.iloc[rng.permutation(len(r1))].drop_duplicates("clip_id")[:n_train]
    out = [("train", "train", c, int(t)) for c, t in zip(one.clip_id, one.t0_us)]
    val = pd.read_parquet(REPO / "outputs" / "recovery_sets" / "val_official_238.parquet")
    out += [("val", "eval", c, int(t)) for c, t in zip(val.clip_id[:n_val], val.t0_us[:n_val])]
    return out


@torch.no_grad()
def measure(model, processor, tok, ns, clip_id, t0_us, seed, max_gen):
    data = sc.load_cached(sc.path_for(ns, clip_id, t0_us))
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
    cs = sc.clip_seed(seed, clip_id)

    seed_all(cs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    coc_end = roll["eos_pos"] + 1
    gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
    gen_coc = tok.decode([t for t in gen_ids if t not in (rl.COT_END, rl.TFS)],
                         skip_special_tokens=True)
    del roll
    coc = torch.tensor([rl.coc_ids(tok, gen_coc)], device="cuda")  # (1, L_coc)
    seq_tf = torch.cat([inputs["input_ids"], coc], dim=1)  # (1, P + L_coc)
    ce = seq_tf.shape[1]

    # (a) teacher-forced trajectory tokens
    lo = model.future_token_start_idx
    hi = lo + model.traj_tokenizer.num_bins
    traj = rl.traj_token_ids(model, data, "cuda")  # (1, 128), GT bins + lo
    out, cache, rope_deltas = rl.vlm_forward(model, inputs, torch.cat([seq_tf, traj], dim=1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.vlm.lm_head(out.last_hidden_state[:, ce - 1 : ce + 127])  # (1, 128, V)
    logits = logits[0].float()  # (128, V)
    tgt = traj[0]  # (128,)
    nll_full = F.cross_entropy(logits, tgt, reduction="none")  # (128,)
    sub = logits[:, lo:hi]  # (128, 3000)
    nll_traj = F.cross_entropy(sub, tgt - lo, reduction="none")  # (128,)
    err = (sub.argmax(-1) - (tgt - lo)).abs().float()  # (128,) bins; even = accel, odd = curv
    del out, logits, sub

    # (b) GT flow-matching loss on the inference grid, KI path (cache cropped to the CoC end)
    if cache.get_seq_length() > ce:
        cache.crop(ce)
    x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    gen = torch.Generator().manual_seed(cs)
    noise = torch.randn(x1.shape, generator=gen).to("cuda")  # (1, 64, 2)
    fm = [float(rl.fm_loss_insulated(model, cache, rope_deltas, x1, ce, t, noise))
          for t in T_GRID]
    del cache

    # (a) free-running greedy decode of the 128 bins -> trajectory
    seq_out = model.vlm.generate(
        input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
        pixel_values=inputs["tokenized_data"]["pixel_values"],
        image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
        do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=128, min_new_tokens=128,
        logits_processor=LogitsProcessorList([BinRangeOnly(lo, hi)]),
        return_dict_in_generate=True, pad_token_id=tok.pad_token_id,
    ).sequences
    bins = seq_out[:, ce : ce + 128] - lo  # (1, 128)
    hx = data["ego_history_xyz"][:, 0].to("cuda").float()  # (1, 16, 3)
    hr = data["ego_history_rot"][:, 0].to("cuda").float()  # (1, 16, 3, 3)
    fut_xyz, _, _ = model.traj_tokenizer.decode(hx, hr, bins)  # (1, 64, 3)
    ade_d, fde_d = el.ade_fde(fut_xyz[:, :, :2].cpu().numpy(), gt_xy)
    gt_xyz, _, _ = model.traj_tokenizer.decode(hx, hr, tgt[None] - lo)  # (1, 64, 3)
    ade_rt, _ = el.ade_fde(gt_xyz[:, :, :2].cpu().numpy(), gt_xy)

    # expert on the same CoC: [0] single sample, min = minADE@6
    seeds = [cs + k for k in range(K_EXPERT)]
    ade_k, _, nll_coc, _ = eval_config_samples(model, inputs, seq_tf, prompt_len, ce,
                                               gt_xy, seeds)
    return {
        "clip_id": clip_id, "t0_us": t0_us, "gen_len": len(gen_ids),
        "degenerate": el.coc_degenerate(gen_coc)["degenerate"], "nll_coc": float(nll_coc),
        "nll_traj": float(nll_traj.mean()), "nll_traj_accel": float(nll_traj[0::2].mean()),
        "nll_traj_curv": float(nll_traj[1::2].mean()), "nll_full": float(nll_full.mean()),
        "top1": float((err == 0).float().mean()), "bin_err_accel": float(err[0::2].mean()),
        "bin_err_curv": float(err[1::2].mean()), "fm_t": fm, "fm_mean": float(np.mean(fm)),
        "ade_disc": float(ade_d[0]), "fde_disc": float(fde_d[0]), "ade_gt_roundtrip": float(ade_rt[0]),
        "ade_expert_1": float(ade_k[0]), "minade6_expert": float(ade_k.min()),
        "gt_bins": (tgt - lo).tolist(),
    }


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def summarize(rows, n_bins):
    bins = np.array([r["gt_bins"] for r in rows])  # (N, 128)
    h_marg = np.mean([entropy(np.bincount(bins[:, d::2].ravel(), minlength=n_bins))
                      for d in (0, 1)])
    out = {"H_marg": h_marg, "sets": {}}
    for s in ("train", "val"):
        rs = [r for r in rows if r["set"] == s]
        if not rs:
            continue
        col = {k: np.array([r[k] for r in rs]) for k in (
            "nll_traj", "nll_full", "top1", "bin_err_accel", "bin_err_curv", "fm_mean",
            "ade_disc", "ade_gt_roundtrip", "ade_expert_1", "minade6_expert", "degenerate")}
        st = {f"{k}_mean": float(v.mean()) for k, v in col.items()}
        st.update({f"{k}_median": float(np.median(v)) for k, v in col.items()})
        st["fm_by_t"] = np.mean([r["fm_t"] for r in rs], axis=0).tolist()
        st["n"] = len(rs)
        st["gate_a_nll"] = bool(st["nll_traj_median"] < h_marg - 1.0)
        st["gate_a_ade"] = bool(st["ade_disc_median"] <= 2.0 * st["ade_expert_1_median"])
        out["sets"][s] = st
    sets = out["sets"]
    if {"train", "val"} <= sets.keys():
        out["G0T_a"] = "PASS" if all(sets[s]["gate_a_nll"] and sets[s]["gate_a_ade"]
                                     for s in ("train", "val")) else "FAIL"
        gaps = {k: (sets["train"][f"{k}_mean"] - sets["val"][f"{k}_mean"]) / sets["val"][f"{k}_mean"]
                for k in ("fm_mean", "nll_traj")}
        out["G0T_b_gap"] = gaps
        out["G0T_b_flag"] = {k: abs(v) > 0.10 for k, v in gaps.items()}
    return out


def plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = out_dir / "plots"
    pdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for s in ("train", "val"):
        rs = [r["fm_t"] for r in rows if r["set"] == s]
        if rs:
            ax.plot(T_GRID, np.mean(rs, axis=0), marker="o", label=f"{s} (n={len(rs)})")
    ax.set_xlabel("t (Euler grid)")
    ax.set_ylabel("teacher FM loss vs GT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "fm_by_t.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    for s in ("train", "val"):
        rs = [r for r in rows if r["set"] == s]
        ax.scatter([r["ade_expert_1"] for r in rs], [r["ade_disc"] for r in rs], s=8,
                   alpha=0.6, label=s)
    lim = max(1.0, max(max(r["ade_expert_1"], r["ade_disc"]) for r in rows))
    ax.plot([0, lim], [0, lim], color="gray", lw=0.8)
    ax.plot([0, lim / 2], [0, lim], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("expert single-sample ADE (m)")
    ax.set_ylabel("greedy discrete-token ADE (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "ade_discrete_vs_expert.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="g0t_teacher_diag")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=238)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N windows only")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / "minitron" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r["set"], r["clip_id"]) for r in rows}

    todo = windows(args.n_train, args.n_val, args.seed)
    if args.limit:
        todo = todo[: args.limit]
    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{args.exp_id} | {len(todo)} windows ({len(done)} done) | {gpu_name}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer

    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "model": "nvidia/Alpamayo-1.5-10B", "revision": MODEL_REV,
        "t_grid": T_GRID, "k_expert": K_EXPERT, "gpu": gpu_name,
        "train_rule": "R1 (mt_data.r1_windows), one cached window per clip",
        "val_set": "outputs/recovery_sets/val_official_238.parquet",
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }, indent=2))

    t_start = time.time()
    for i, (s, ns, clip_id, t0_us) in enumerate(todo):
        if (s, clip_id) in done:
            continue
        rec = measure(model, processor, tok, ns, clip_id, t0_us, args.seed, args.max_gen)
        rows.append({"set": s, **rec})
        if len(rows) % 10 == 0 or i == len(todo) - 1:
            rows_path.write_text(json.dumps(rows))
            el_s = time.time() - t_start
            print(f"[{len(rows)}/{len(todo)}] {s} nll_traj {rec['nll_traj']:.3f} "
                  f"ade_disc {rec['ade_disc']:.2f} ade_exp {rec['ade_expert_1']:.2f} "
                  f"fm {rec['fm_mean']:.4f} | {el_s / 60:.1f} min", flush=True)
    rows_path.write_text(json.dumps(rows))

    summ = summarize(rows, model.traj_tokenizer.num_bins)
    (out_dir / "metrics.json").write_text(json.dumps(summ, indent=2))
    plots(rows, out_dir)

    lines = [f"G0-T teacher diagnostic ({gpu_name}); H_marg = {summ['H_marg']:.3f} nats"]
    for s, st in summ["sets"].items():
        lines.append(
            f"{s:5s} n={st['n']:3d} | traj NLL median {st['nll_traj_median']:.3f} "
            f"(gate < {summ['H_marg'] - 1:.3f}: {st['gate_a_nll']}) top1 {st['top1_mean']:.3f} "
            f"bin err a/k {st['bin_err_accel_mean']:.1f}/{st['bin_err_curv_mean']:.1f} | "
            f"ADE disc {st['ade_disc_median']:.3f} vs expert-1 {st['ade_expert_1_median']:.3f} "
            f"(gate <= 2x: {st['gate_a_ade']}), GT round-trip {st['ade_gt_roundtrip_median']:.3f}, "
            f"minADE@6 {st['minade6_expert_median']:.3f} | FM mean {st['fm_mean_mean']:.4f} | "
            f"degen {st['degenerate_mean']:.3f}")
    if "G0T_a" in summ:
        lines.append(f"G0-T(a): {summ['G0T_a']}")
        lines.append("G0-T(b) train-vs-val gap: " + ", ".join(
            f"{k} {v:+.1%}{' FLAG' if summ['G0T_b_flag'][k] else ''}"
            for k, v in summ["G0T_b_gap"].items()))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
