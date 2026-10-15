"""Open-loop ego-history perturbation: does a model continue a deceleration it sees in its history?

In closed loop the simulator hands the car over with -8 m/s2 commanded, every arm loses a median
1.06 m/s in the first 0.1 s, and only the unpruned VLM then plans a further slowdown
(`analyze_handover.py`). Here the same history is injected in the open-loop harness, where the
scene, the frames and the seeds stay fixed (plans/2026-10-08_history-perturbation-openloop.md).

Conditions on `ego_history_xyz` (16 poses, 0.1 s apart, t0 frame, last pose the origin); u is the
direction of the last step and d = min(drop, 0.9 v_last) * 0.1 s:

  clean    nothing                                        the plan at the hand-over
  drop1    poses 0-14 moved by +d u (last step shorter)   the history 0.1 s after it
  drop2    poses 0-13 by +2d u, pose 14 by +d u           0.2 s after, where closed loop peaked
  rise1    poses 0-14 by -d u (last step longer)          mirror control
  slow10   all poses x 0.9                                a slower history with no braking event
  dropN@cleanCoC   dropN history, the clean rollout's CoC teacher-forced

The action space integrates the predicted acceleration from a speed v0 it fits to the history, so
a perturbed history lowers the planned speed in every arm before the model has said anything.
The model's response is therefore the **predicted acceleration**; planned speed is stored beside
it. Seeds come from `clip_seed` and are the same in every condition, so each comparison is paired
within clip and seed.

Gates (fixed before any run): S0 the perturbation touches only the intended history tokens and
`clean` reproduces `lib.build_inputs`; H1 dense's mean predicted acceleration over the first 2 s
drops under drop2 (clip CI excluding 0); H2 it drops more for dense than for MoOP; H3 the drop
orders the arms by how much of the VLM was pruned; H4 dense keeps >= 2/3 of it with the clean CoC
teacher-forced.

Usage:
  ALPAMAYO_REPO=$PWD bash experiments/head_analysis/run_retry_host.sh 3 \
      experiments/evaluation/run_history_perturb.py --model baseline --gpu 4
  ... --model outputs/slim_dual_u40_v2 --gpu 5
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import eval_lib as el
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_baseline import COT_END, MODEL_REV, TFS, seed_all, set_determinism
from sample_cache import clip_seed

DT = 0.1
CONDS = ["clean", "drop1", "drop2", "rise1", "slow10"]
TF_CONDS = ["drop1", "drop2", "sim2"]  # also evaluated with the clean rollout's CoC teacher-forced
# Added mid-run (plan addendum): m/s taken off each of the last five steps. `sim5` is the history a
# dense-driven car has 0.5 s after the closed-loop hand-over; `brake5` is a uniform -3 m/s2.
SUSTAINED = {"sim5": (1.06, 1.29, 1.45, 1.66, 1.80), "brake5": (0.3, 0.6, 0.9, 1.2, 1.5),
             # second addendum: the median history every arm has 0.2 s after the hand-over (the second
             # step is slower again, which `drop2` leaves out), and the dense car's 0.3 s after
             "sim2": (1.06, 1.28), "sim3": (1.06, 1.28, 1.38)}


def perturb(hist_xyz, cond, drop):
    """hist_xyz (1, 1, 16, 3), t0 frame, last pose the origin -> (perturbed copy, metres per step)."""
    h = hist_xyz.clone()
    step = h[0, 0, -1, :2] - h[0, 0, -2, :2]              # (2,) the last 0.1 s
    u = step / step.norm()                                # (2,) direction of travel
    d = min(drop, 0.9 * float(step.norm()) / DT) * DT     # never reverses the last step
    if cond in SUSTAINED:
        cut = torch.tensor(SUSTAINED[cond]) * DT          # (n,) metres off the last n steps, oldest first
        n = len(cut)
        length = (h[0, 0, -n:, :2] - h[0, 0, -n - 1:-1, :2]).norm(dim=1)   # (n,) their lengths
        cut = cut * min(1.0, float((0.9 * length / cut).min()))            # at most 90% of any step
        shift = torch.zeros(16)                           # pose i moves by all that is cut after it
        shift[15 - n:15] = torch.flip(torch.cumsum(torch.flip(cut, [0]), 0), [0])
        shift[:15 - n] = cut.sum()
        h[0, 0, :, :2] += shift[:, None] * u
        return h, float(cut[0])
    if cond == "drop1":
        h[0, 0, :-1, :2] += d * u
    elif cond == "drop2":
        h[0, 0, :-2, :2] += 2 * d * u
        h[0, 0, -2, :2] += d * u
    elif cond == "rise1":
        h[0, 0, :-1, :2] -= d * u
    elif cond == "slow10":
        h[0, 0] *= 0.9
    return h, d


def build_raw(processor, data, device):
    """First half of lib.build_inputs: the 16 frames are processed once per clip, not per condition."""
    messages = helper.create_message(frames=data["image_frames"].flatten(0, 1),
                                     camera_indices=data["camera_indices"])
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                           continue_final_message=True, return_dict=True, return_tensors="pt")
    tokenized = helper.to_device({"tokenized_data": inputs}, device)["tokenized_data"]
    raw_ids = tokenized.pop("input_ids")                  # (1, T_prompt), history slots still padding
    return raw_ids, dict(tokenized)


def fuse(model, raw_ids, tokenized, hist_xyz, hist_rot):
    """Second half: write this condition's history tokens into the prompt."""
    ids = model.fuse_traj_tokens(raw_ids.clone(), {"ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot})
    return {"input_ids": ids, "tokenized_data": tokenized, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}


def plan_samples(model, inputs, seq_tf, coc_start, coc_end, seeds):
    """run_eval.eval_config_samples, keeping the action itself.

    Returns v0 (m/s, from the history), accel (K, 64) in m/s2, pred_xy (K, 64, 2), CoC NLL.
    """
    attention_mask = torch.ones_like(seq_tf)
    sp = model.action_space
    hx = inputs["ego_history_xyz"][:, -1].float()         # (1, 16, 3)
    hr = inputs["ego_history_rot"][:, -1].float()         # (1, 16, 3, 3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=attention_mask,
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]).item()
        cache = out.past_key_values
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device=seq_tf.device)
        prefix_mask = torch.ones(1, prefill, device=seq_tf.device, dtype=torch.long)
        acts, preds = [], []
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, out.rope_deltas, offset, prefix_mask,
                                            seed=s).float()  # (1, 64, 2), cropped back to prefill each call
            pred_xyz, _ = sp.action_to_traj(action, hx, hr)
            acts.append(action[0, :, 0])                  # (64,) normalised acceleration
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    accel = torch.stack(acts) * sp.accel_std.to(acts[0].device) + sp.accel_mean.to(acts[0].device)   # (K, 64)
    v0 = float(sp.estimate_t0_states(hx, hr)["v"].flatten()[0])
    del out, cache
    return v0, accel.cpu().numpy(), np.stack(preds), nll


def summarise(v0, accel, pred, gt_xy):
    """v0 scalar, accel (K, 64) m/s2, pred (K, 64, 2) -> second-by-second means per sample."""
    v = v0 + np.cumsum(accel * DT, axis=1)                # (K, 64) planned speed at 0.1 .. 6.4 s
    ade, _ = el.ade_fde(pred, gt_xy)
    r = {"v0": round(v0, 4), "ade": [round(float(x), 4) for x in ade],
         "a_prof": [round(float(x), 4) for x in accel.mean(0)]}     # (64,) mean over the K samples
    for name, sl in (("01", slice(0, 10)), ("12", slice(10, 20)), ("23", slice(20, 30))):
        r[f"a{name}"] = [round(float(x), 4) for x in accel[:, sl].mean(1)]
        r[f"v{name}"] = [round(float(x), 4) for x in v[:, sl].mean(1)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--exp-id", default="histperturb_v1")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--manifest", default="test_500")
    ap.add_argument("--cache", default="test")
    ap.add_argument("--n-first", type=int, default=150, help="first N clips of the manifest")
    ap.add_argument("--min-speed", type=float, default=2.0, help="skip clips slower than this at t0")
    ap.add_argument("--drop", type=float, default=1.06, help="m/s taken out of one history step")
    ap.add_argument("--conds", nargs="+", default=CONDS, help="'clean' first; see CONDS and SUSTAINED")
    ap.add_argument("--k", type=int, default=4, help="trajectory samples per condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="stop after N evaluated clips")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}.json"
    man = pd.read_parquet(REPO / "outputs" / args.sets_id / f"{args.manifest}.parquet").iloc[: args.n_first]

    set_determinism(warn_only=True)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{args.exp_id} {tag} | first {len(man)} of {args.manifest} | {gpu_name}", flush=True)

    if args.model == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.model, device="cuda")

    processor = helper.get_processor(model.tokenizer)
    tok = model.tokenizer
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / f"config_{tag}.json").write_text(json.dumps({
        "model": args.model, "manifest": args.manifest, "n_first": args.n_first,
        "min_speed": args.min_speed, "drop_mps": args.drop, "conditions": args.conds,
        "sustained_mps": {c: SUSTAINED[c] for c in args.conds if c in SUSTAINED},
        "teacher_forced_clean_coc": [c for c in TF_CONDS if c in args.conds], "k": args.k, "seed": args.seed,
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample, same in every condition",
        "max_gen": args.max_gen, "gpu": gpu_name, "model_revision": MODEL_REV,
    }, indent=2))

    t_start, n_skip, checked = time.time(), 0, False
    for i, m in enumerate(man.itertuples()):
        if m.clip_id in done:
            continue
        if args.limit and len(rows) >= args.limit:
            break
        t0 = time.time()
        data = sc.load_cached(sc.path_for(args.cache, m.clip_id, int(m.t0_us)))
        hist = data["ego_history_xyz"]                    # (1, 1, 16, 3)
        v_last = float((hist[0, 0, -1, :2] - hist[0, 0, -2, :2]).norm()) / DT
        if v_last < args.min_speed:
            n_skip += 1
            continue
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        raw_ids, tokenized = build_raw(processor, data, "cuda")
        hist_rot = data["ego_history_rot"].to("cuda")     # (1, 1, 16, 3, 3)
        base = clip_seed(args.seed, m.clip_id)
        traj_seeds = [base + k for k in range(args.k)]
        rec = {"clip_id": m.clip_id, "seed": base, "bucket": el.bucket(gt_xy), "v_last": round(v_last, 4)}

        clean_ids = clean_tail = None
        for cond in args.conds:
            hx, d = perturb(hist, cond, args.drop)
            inputs = fuse(model, raw_ids, tokenized, hx.to("cuda"), hist_rot)
            ids = inputs["input_ids"]
            prompt_len = ids.shape[1]
            if cond == "clean":
                clean_ids = ids
                rec["delta_m"] = round(d, 4)
                if not checked:                           # S0: this path equals lib.build_inputs
                    ref = lib.build_inputs(model, processor, data, "cuda")["input_ids"]
                    assert torch.equal(ref, ids), "clean prompt differs from lib.build_inputs"
                    checked = True
            changed = torch.nonzero(ids[0] != clean_ids[0]).flatten()
            hist_pos = torch.nonzero(lib.compute_spans(model, clean_ids)["hist"]).flatten()
            # history tokens that differ from clean, as indices into the 48-token history span
            rel = [int((hist_pos == c).nonzero()[0, 0]) for c in changed.cpu()]

            seed_all(base)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq = roll["sequences"][:, :coc_end].clone()
            gen_ids = seq[0, prompt_len:].tolist()
            gen_coc = tok.decode([t for t in gen_ids if t not in (COT_END, TFS)], skip_special_tokens=True)
            del roll
            with torch.no_grad():
                v0, accel, pred, nll = plan_samples(model, inputs, seq, prompt_len, coc_end, traj_seeds)
            rec[cond] = {**summarise(v0, accel, pred, gt_xy), "nll_self": nll, "gen_coc": gen_coc,
                         "gen_len": len(gen_ids), "hist_tokens_changed": rel}
            if cond == "clean":
                clean_tail = seq[:, prompt_len:].clone()  # the clean CoC, cot_end and traj_future_start
            elif cond in TF_CONDS:
                seq_tf = torch.cat([ids, clean_tail], dim=1)
                with torch.no_grad():
                    v0, accel, pred, nll = plan_samples(model, inputs, seq_tf, prompt_len, seq_tf.shape[1],
                                                        traj_seeds)
                rec[f"{cond}@cleanCoC"] = {**summarise(v0, accel, pred, gt_xy), "nll_cleancoc": nll}

        rows.append(rec)
        if len(rows) % 10 == 0:
            rows_path.write_text(json.dumps(rows))
        a = {c: float(np.mean(rec[c]["a01"]) + np.mean(rec[c]["a12"])) / 2 for c in args.conds}
        last = args.conds[-1]
        print(f"[{i + 1}/{len(man)}] {m.clip_id[:8]} v={v_last:5.1f} a(0-2s) "
              + " ".join(f"{c} {a[c]:+.2f}" for c in args.conds[:3])
              + f" v0 {rec['clean']['v0']:.2f}->{rec[last]['v0']:.2f} ({last}) tok"
              f" {rec[args.conds[1]]['hist_tokens_changed']} ({time.time() - t0:.0f}s)", flush=True)

    rows_path.write_text(json.dumps(rows))
    ref = "drop2" if "drop2" in args.conds else args.conds[1]
    da = np.array([(np.mean(r[ref]["a01"]) + np.mean(r[ref]["a12"])) / 2
                   - (np.mean(r["clean"]["a01"]) + np.mean(r["clean"]["a12"])) / 2 for r in rows])
    line = (f"{tag}: {len(rows)} clips ({n_skip} skipped below {args.min_speed} m/s), "
            f"mean change in predicted acceleration over 0-2 s under {ref}: {da.mean():+.4f} m/s2")
    (out_dir / f"summary_{tag}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
