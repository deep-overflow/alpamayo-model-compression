"""Dense-reference CoC NLL: how well an arm predicts the unpruned model's own reasoning.

The in-distribution sets carry no reference reasoning text, so `nll_gtcoc` exists only
on OOD. This scores each arm against the next best thing on val500 / test500: the CoC the
**unpruned** model rolled out on that clip, read from a stored `run_baseline.py` run
(`--ref-run`, default `baseline_ada_ps_<set>`). The number is a fidelity to the dense
model's reasoning distribution, not a correctness score -- the reference is one sample,
so an arm is penalised for a different but valid reasoning and rewarded for repeating the
dense model's mistakes. It is also the quantity `I_CoC` was fitted on (own-rollout NLL
over calib_100), so `coc` and `dual` are structurally favoured: read it as a
consistency check beside `nll_gtcoc`, never in its place.

Same span rule as `nll_gtcoc`: the text is re-tokenised and appended to the prompt with
`gt_coc_seq` (`[prompt] + tokens + <cot_end> + <traj_future_start>`), one forward pass,
mean per-token cross-entropy in nats over that span. No rollout and no trajectory
sampling, so ~1-2 s/clip against ~8 s/clip for `run_baseline.py`, and seed-independent.

`nll_self` in the stored rows used the same span: the rollout stops on
<traj_future_start> (`analysis_lib.run_rollout`, `eos_pos`), and `sequences[:,
prompt_len:eos_pos + 1]` therefore ends in the same two closing tokens `gt_coc_seq`
appends. So on `--model baseline` `nll_dense` must land on the stored `nll_self`, and the
residual is only the decode->encode round trip of the reference text (`gen_coc` was
decoded with the closing tokens stripped, then re-tokenised here). The summary reports
that gap and the number of clips whose re-tokenised span length differs from `gen_len`;
both are printed for every arm because the reference text is the same.

Usage:
  python experiments/evaluation/run_nll_dense.py --set test --model baseline --gpu 4
  python experiments/evaluation/run_nll_dense.py --set test --model outputs/slim_dual_u40_v2 --gpu 4
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
from run_baseline import MODEL_REV, gt_coc_seq, load_manifest, set_determinism


def load_reference(ref_dir):
    """clip_id -> stored row of the reference run, merging its shards."""
    rows = []
    for p in sorted(ref_dir.glob("*.json")):
        if p.name == "config.json":
            continue
        rows += json.loads(p.read_text())
    if not rows:
        raise SystemExit(f"no rows under {ref_dir}")
    return {r["clip_id"]: r for r in rows}


@torch.no_grad()
def span_nll(model, inputs, seq, coc_start, coc_end):
    """Per-token NLL (nats) of seq[coc_start:coc_end] given the fused prompt. (L,)"""
    attention_mask = torch.ones_like(seq)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq, attention_mask=attention_mask,
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=False,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1 : coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq[0, coc_start:coc_end],
                                                reduction="none")
    del out
    return nll.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", required=True, choices=["indist", "test"])
    ap.add_argument("--model", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--ref-run", default=None,
                    help="outputs/<dir> of the run whose gen_coc is the reference; "
                         "default baseline_ada_ps_<set>")
    ap.add_argument("--exp-id", default=None, help="default nll_dense_<tag>_<set>")
    ap.add_argument("--sets-id", default="eval_sets")
    ap.add_argument("--n-indist", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N clips of the shard")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--manifest", default=None, help="override the manifest stem")
    ap.add_argument("--cache", default=None, help="override the pre_processed cache name")
    ap.add_argument("--strict-deterministic", action="store_true",
                    help="error instead of warn on a non-deterministic kernel")
    args = ap.parse_args()

    tag = "baseline" if args.model == "baseline" else Path(args.model).name
    ref_run = args.ref_run or f"baseline_ada_ps_{args.which}"
    exp_id = args.exp_id or f"nll_dense_{tag}_{args.which}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{tag}_s{args.shard}of{args.n_shards}.json"

    ref = load_reference(REPO / "outputs" / ref_run)
    man = load_manifest(args.which, REPO / "outputs" / args.sets_id, args.n_indist,
                        args.manifest, args.cache)
    missing = [m["clip_id"] for m in man if m["clip_id"] not in ref]
    if missing:
        raise SystemExit(f"{len(missing)} manifest clips have no reference row in {ref_run}, "
                         f"e.g. {missing[:3]}")
    man = man[args.shard::args.n_shards]          # strided, so any shard count covers all
    if args.limit:
        man = man[: args.limit]

    set_determinism(warn_only=not args.strict_deterministic)
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{exp_id} {tag} | set={args.which} ref={ref_run} shard {args.shard}/{args.n_shards} "
          f"| {len(man)} clips | {gpu_name}", flush=True)

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

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": args.model, "set": args.which, "ref_run": ref_run, "n_clips": len(man),
        "shard": [args.shard, args.n_shards], "sets_id": args.sets_id, "gpu": gpu_name,
        "model_revision": MODEL_REV, "manifest": args.manifest, "cache": args.cache,
        "span": "[prompt] + tokenize(ref gen_coc) + cot_end + traj_future_start, "
                "cross_entropy mean over the appended span",
        "deterministic": {"use_deterministic_algorithms": True,
                          "warn_only": not args.strict_deterministic,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    for i, m in enumerate(man):
        if m["clip_id"] in done:
            continue
        t0 = time.time()
        r = ref[m["clip_id"]]
        data = sc.load_cached(sc.path_for(m["cache"], m["clip_id"], m["t0_us"]))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)

        seq = gt_coc_seq(tok, inputs["input_ids"], r["gen_coc"], "cuda")  # (1, L)
        per_tok = span_nll(model, inputs, seq, prompt_len, seq.shape[1])  # (n_tok,)
        rec = {"clip_id": m["clip_id"], "bucket": el.bucket(gt_xy),
               "nll_dense": float(per_tok.mean()),
               "n_tok": len(per_tok),
               "ref_len": int(r["gen_len"]), "ref_nll_self": float(r["nll_self"]),
               "ref_empty": bool(r.get("coc_empty", False)),
               "ref_degenerate": bool(r.get("coc_degenerate", False)),
               "ref_coc": r["gen_coc"]}
        rows.append(rec)
        if len(rows) % 10 == 0 or i + 1 == len(man):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(man)}] {m['clip_id'][:8]} {rec['bucket']:10s} "
              f"nll={rec['nll_dense']:.3f} n_tok={rec['n_tok']} ({time.time() - t0:.1f}s)",
              flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    a = np.array([r["nll_dense"] for r in rows])
    line = (f"{tag} {args.which} vs {ref_run} shard {args.shard}/{args.n_shards}: "
            f"{len(rows)} clips, nll_dense mean {a.mean():.4f} median {np.median(a):.4f}")
    if rows:
        # the round-trip gate: on baseline this is decode->encode drift only (same span
        # as nll_self), and it is reported for every arm because the reference is the
        # same text
        gap = np.array([r["nll_dense"] - r["ref_nll_self"] for r in rows])
        tok_gap = np.array([r["n_tok"] - r["ref_len"] for r in rows])
        line += (f", nll_dense - ref nll_self mean {gap.mean():+.4f} max|.| "
                 f"{np.abs(gap).max():.4f}, re-tokenised span length off on "
                 f"{int((tok_gap != 0).sum())} clips")
    (out_dir / f"summary_s{args.shard}of{args.n_shards}.txt").write_text(line + "\n")
    print(f"\n{line}\n{(time.time() - t_start) / 60:.1f} min -> {rows_path}", flush=True)


if __name__ == "__main__":
    main()
