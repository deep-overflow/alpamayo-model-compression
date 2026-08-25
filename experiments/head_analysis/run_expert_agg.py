"""D stage: does a better aggregation of the SAME gradients beat the shipped criterion?

The step decomposition (run_step_importance.py) showed two aggregation defects in the
shipped expert score `traj_exp_* = mean_clips |sum_s dL_s/dg|`:

  step axis  steps rank units differently (Q head rho 0.717 against a 0.929 noise floor)
             and their mass differs 7.7x, so the sum is a few steps' opinion;
  clip axis  the mean is outlier-driven -- at step 0 one clip out of 100 carries 49% of
             the mass, and trimming the top 10% moves 14% of the selection.

Neither defect needs new gradients: every arm below is a re-aggregation of the per-clip,
per-step arrays already measured, so this run only pays for mask evaluation.

Arms (all per-layer uniform, expert tower only, KV and VLM untouched):
  sum             mean_clips |sum_s g|            -- the shipped score, must reproduce it
  sumabs          mean_clips sum_s |g|            -- predicted to be a no-op (G5d)
  trimclip        10%-trimmed clip mean of |sum_s g|
  znorm           per-step within-layer z-score, averaged over steps
  maxrank         per-step within-layer rank, maxed over steps
  trimclip_znorm  both axes at once
  damagewt        steps weighted by the measured dev(s) damage curve
  infer           the inference-path score (measurement B)
  magnitude       weight norm -- the reference this whole track is about

Evaluated on indist_500[60:], never the 60 clips stepmask_v1 used: damagewt's weights come
from that run, and reusing those clips would tune one arm on its own evaluation set.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 60 \
      experiments/head_analysis/run_expert_agg.py --gpu 6 --exp-id stepagg_v1 \
      --shard 0 --n-shards 2
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
UNITS = ("q", "mlp")


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# score aggregations
# ---------------------------------------------------------------------------


def zscore_layers(a):
    """Within-layer z-score of a (L, U) map. Steps carry different mass; this removes it."""
    m = a.mean(1, keepdims=True)
    s = a.std(1, keepdims=True)
    return (a - m) / np.where(s > 0, s, 1.0)


def rank_layers(a):
    """Within-layer rank in [0, 1] of a (L, U) map. 1 = most important."""
    order = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    return order / max(a.shape[1] - 1, 1)


def trimmed_mean(a, frac=0.10):
    """Per-entry trimmed mean over axis 0, dropping the largest `frac` of clips.

    Only the upper tail is dropped: the failure mode is a handful of clips whose gradients
    are orders of magnitude larger than the rest, not clips that are unusually small.
    """
    n = a.shape[0]
    keep = max(int(round(n * (1 - frac))), 1)
    return np.sort(a, axis=0)[:keep].mean(0)


def build_scores(agg, pc, infer_agg, mag, damage_w):
    """arm -> {"q": (L,H), "mlp": (L,I)}; every arm is a re-aggregation of the same grads."""
    arms = {}

    def per_unit(fn):
        return {u: fn(u) for u in UNITS}

    arms["sum"] = per_unit(lambda u: agg[f"{u}_shipped"].astype(np.float64))
    arms["sumabs"] = per_unit(lambda u: agg[f"{u}_abs_step"].astype(np.float64).sum(0))
    arms["trimclip"] = per_unit(
        lambda u: trimmed_mean(pc[f"{u}_shipped"].astype(np.float64)))
    arms["znorm"] = per_unit(
        lambda u: np.mean([zscore_layers(a) for a in agg[f"{u}_abs_step"].astype(np.float64)],
                          axis=0))
    arms["maxrank"] = per_unit(
        lambda u: np.max([rank_layers(a) for a in agg[f"{u}_abs_step"].astype(np.float64)],
                         axis=0))
    # left in fp32: the per-clip MLP array is (100, 10, 36, 8256) and upcasting it plus the
    # sort copy would be ~5 GB of RAM for no precision the ranking can use
    arms["trimclip_znorm"] = per_unit(
        lambda u: np.mean([zscore_layers(a.astype(np.float64))
                           for a in trimmed_mean(pc[f"{u}_abs_step"])], axis=0))
    arms["damagewt"] = per_unit(
        lambda u: np.tensordot(damage_w, agg[f"{u}_abs_step"].astype(np.float64), axes=(0, 0)))
    if infer_agg is not None:
        arms["infer"] = per_unit(
            lambda u: infer_agg[f"{u}_abs_step"].astype(np.float64).sum(0))
    arms["magnitude"] = {"q": mag[0].astype(np.float64), "mlp": mag[1].astype(np.float64)}
    return arms


def damage_weights(stepmask_dir, criterion="traj"):
    """w_s proportional to the measured dev(s) of a mask applied at step s only."""
    m = json.loads((stepmask_dir / "metrics.json").read_text())
    meta, rows = m["meta"], m["rows"]
    sel = sorted((meta[n]["step"], n) for n in m["configs"]
                 if meta[n]["kind"] == "only" and meta[n].get("criterion") == criterion)
    w = np.array([np.mean([np.mean(r["configs"][n]["dev_k"]) for r in rows]) for _, n in sel])
    return w / w.sum()


def check_integrity(arms, imp, ratio, lines):
    """G5a: the re-aggregated `sum` must be the shipped criterion, not merely close to it."""
    ok = True
    for u, key in (("q", "traj_exp_q"), ("mlp", "traj_exp_mlp")):
        a, b = arms["sum"][u], imp[key]
        rho = float(np.mean([spearmanr(a[i], b[i])[0] for i in range(a.shape[0])]))
        layers = list(range(a.shape[0]))
        ka = ml.select_mask(a, ratio, layers) == 1
        kb = ml.select_mask(b, ratio, layers) == 1
        ov = float((ka & kb).sum() / max(ka.sum(), 1))
        good = rho > 0.999 and ov > 0.999
        ok = ok and good
        lines.append(f"  G5a {u:4s} per-layer rho {rho:.6f}  kept-overlap {ov:.6f}  "
                     f"{'PASS' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------


@torch.no_grad()
def denoise_paths(model, inputs, cache, rope_deltas, offset, prefix_mask, seeds):
    """K denoisings on a prebuilt cache -> (K, 64, 2) predicted xy."""
    preds = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in seeds:
            action = lib.denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask,
                                            seed=s)
            pred_xyz, _ = model.action_space.action_to_traj(
                action.float(), inputs["ego_history_xyz"][:, -1].float(),
                inputs["ego_history_rot"][:, -1].float(),
            )
            preds.append(pred_xyz[0, :, :2].cpu().numpy())
    return np.stack(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=str, required=True)
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--clip-offset", type=int, default=60,
                    help="skip the clips stepmask_v1 used, so damagewt is not tuned on them")
    ap.add_argument("--manifest", default="indist_500")
    ap.add_argument("--cache", default="eval")
    ap.add_argument("--stepimp", default="stepimp_fm_perstep_v2",
                    help="the per-step decomposition every arm re-aggregates")
    ap.add_argument("--infer-run", default="stepimp_infer_v2")
    ap.add_argument("--stepmask", default="stepmask_v1", help="source of the damage weights")
    ap.add_argument("--importance", default="importance_v2_ada", help="G5a reference")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.40])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{args.manifest}.parquet")
    man = [(r.clip_id, int(r.t0_us)) for r in df.itertuples()]
    man = man[args.clip_offset : args.clip_offset + args.num_clips]
    man = man[args.shard :: args.n_shards]

    agg = dict(np.load(REPO / "outputs" / args.stepimp / "step_importance.npz"))
    pc = dict(np.load(REPO / "outputs" / args.stepimp / "step_importance_perclip.npz"))
    infer_agg = None
    if args.infer_run:
        infer_agg = dict(np.load(REPO / "outputs" / args.infer_run / "step_importance.npz"))
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    dw = damage_weights(REPO / "outputs" / args.stepmask)

    # the fp16 accident: per-clip arrays must reproduce the fp64 aggregate they came from,
    # or trimclip is computed from underflowed garbage
    lines = []
    for u in UNITS:
        got = pc[f"{u}_abs_step"].astype(np.float64).mean(0)
        rel = np.abs(got - agg[f"{u}_abs_step"]) / (np.abs(agg[f"{u}_abs_step"]) + 1e-30)
        if np.median(rel) > 1e-3:
            raise ValueError(f"per-clip {u} does not reproduce the aggregate "
                             f"(median rel {np.median(rel):.2e}); re-run run_step_importance")
        lines.append(f"  per-clip {u} fidelity OK (median rel {np.median(rel):.2e})")

    set_determinism()
    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    ec = model.expert.config
    masks = ml.PruneMasks(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                          ec.intermediate_size, "cuda")
    mag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                              ec.intermediate_size)
    arms = build_scores(agg, pc, infer_agg, mag, dw)
    ok = check_integrity(arms, imp, args.ratios[-1], lines)
    print("\n".join(lines), flush=True)
    if not ok:
        raise ValueError("G5a failed: the re-aggregated `sum` is not the shipped criterion")

    layers = list(range(ec.num_hidden_layers))
    cfgs = [("baseline", {"kind": "baseline"}, None, None)]
    for r in args.ratios:
        for name, sc_ in arms.items():
            q = ml.select_mask(sc_["q"], r, layers)
            m = ml.select_mask(sc_["mlp"], r, layers)
            cfgs.append((f"{name}_r{int(round(r * 100))}",
                         {"kind": "arm", "arm": name, "ratio": r},
                         torch.as_tensor(q, dtype=torch.float32, device="cuda"),
                         torch.as_tensor(m, dtype=torch.float32, device="cuda")))
    print(f"{len(cfgs)} configs x {len(man)} clips x K={args.k}", flush=True)

    # kept-set overlaps between arms, at each ratio -- cheap and needed for interpretation
    overlaps = {}
    for r in args.ratios:
        names = list(arms)
        ov = {}
        for a in names:
            for b in names:
                ka = ml.select_mask(arms[a]["q"], r, layers) == 1
                kb = ml.select_mask(arms[b]["q"], r, layers) == 1
                ov[f"{a}|{b}"] = float((ka & kb).sum() / max(ka.sum(), 1))
        overlaps[f"r{int(round(r * 100))}"] = ov

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV, "tower": "expert",
        "purpose": "does a better aggregation of the same gradients beat the shipped score?",
        "manifest": args.manifest, "cache": args.cache, "clip_offset": args.clip_offset,
        "n_clips": len(man), "clip_ids": [c for c, _ in man], "ratios": args.ratios,
        "k": args.k, "seed": args.seed, "shard": [args.shard, args.n_shards],
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4], +k per sample",
        "stepimp": args.stepimp, "infer_run": args.infer_run, "stepmask": args.stepmask,
        "damage_weights": [float(x) for x in dw], "arms": list(arms),
        "kept_overlap_q": overlaps, "integrity": lines,
        "configs": [{"name": n, **m} for n, m, _, _ in cfgs],
        "gpu": torch.cuda.get_device_name(device),
    }, indent=2))

    rows_path = out_dir / f"rows_s{args.shard}of{args.n_shards}.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}

    for ci, (clip_id, t0_us) in enumerate(man):
        if clip_id in done:
            continue
        t_start = time.time()
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0_us))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
        base = sc.clip_seed(args.seed, clip_id)
        seeds = [base + k for k in range(args.k)]

        masks.reset()
        seed_all(base)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
            coc_end = roll["eos_pos"] + 1
            seq_tf = roll["sequences"][:, :coc_end].clone()
            del roll
            out = model.vlm.model(
                input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
                pixel_values=inputs["tokenized_data"]["pixel_values"],
                image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
            )
            logits = model.vlm.lm_head(
                out.last_hidden_state[:, prompt_len - 1 : coc_end - 1]).float()
            nll = torch.nn.functional.cross_entropy(
                logits[0], seq_tf[0, prompt_len:coc_end]).item()
        cache, rope_deltas = out.past_key_values, out.rope_deltas
        prefill = cache.get_seq_length()
        offset = torch.tensor([prefill], device="cuda")
        prefix_mask = torch.ones(1, prefill, device="cuda", dtype=torch.long)
        del out

        # the expert cannot change the CoC, so its NLL is a per-clip datum, not per config
        rec = {"clip_id": clip_id, "bucket": el.bucket(gt_xy), "seed": base,
               "coc_len": int(coc_end - prompt_len), "nll": nll, "configs": {}}
        base_pred = None
        for name, _, q, m in cfgs:
            if q is None:
                masks.reset()
            else:
                masks.q_mask.copy_(q)
                masks.mlp_mask.copy_(m)
            pred_k = denoise_paths(model, inputs, cache, rope_deltas, offset, prefix_mask,
                                   seeds)  # (K, 64, 2)
            if base_pred is None:
                base_pred = pred_k
            ade, fde = el.ade_fde(pred_k, gt_xy)
            entry = {"ade_k": [round(float(x), 6) for x in ade],
                     "fde_k": [round(float(x), 6) for x in fde],
                     "dev_k": [round(float(x), 6) for x in
                               np.linalg.norm(pred_k - base_pred, axis=2).mean(1)]}
            # per-sample horizon arrays, as the fixed protocol stores them: the full-horizon
            # scalars cannot be re-reduced later
            for h in (16, 32):
                a_h, f_h = el.ade_fde(pred_k[:, :h], gt_xy[:h])
                entry[f"ade_k_h{h}"] = [round(float(x), 6) for x in a_h]
                entry[f"fde_k_h{h}"] = [round(float(x), 6) for x in f_h]
            rec["configs"][name] = entry
        masks.reset()
        rows.append(rec)
        del cache, inputs

        b = rec["configs"]["baseline"]["ade_k"]
        print(f"[{ci + 1}/{len(man)}] {clip_id} {rec['bucket']:10s} "
              f"baseMinADE@6={min(b[:6]):.3f} ({time.time() - t_start:.0f}s)", flush=True)
        rows_path.write_text(json.dumps(rows))
        if (ci + 1) % 10 == 0 or ci + 1 == len(man):
            save(out_dir, cfgs, rows, args)
    save(out_dir, cfgs, rows, args)
    print("saved ->", out_dir, flush=True)


def save(out_dir, cfgs, rows, args):
    (out_dir / f"metrics_s{args.shard}of{args.n_shards}.json").write_text(json.dumps({
        "n_clips": len(rows), "configs": [n for n, _, _, _ in cfgs],
        "meta": {n: m for n, m, _, _ in cfgs}, "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
