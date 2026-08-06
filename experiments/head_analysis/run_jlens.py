"""Stage A/B: build the J-lens for the VLM tower and score pruning units by it.

Per clip: roll out the CoC, then one teacher-forced forward whose graph is reused
for S backwards -- one per dictionary token, each yielding that token's J-lens
vector at *every* layer. Unit scores follow by matmul (see jlens_lib), so no
per-unit backward is needed.

Gates this run has to answer:
  G1  do the lens readouts name driving concepts, and is there a middle-layer
      kurtosis plateau (the paper's workspace band)?
  G2  is the resulting unit score different from magnitude, and closer to the
      CoC objective than to the trajectory objective?

Start with --smoke to measure one backward before committing to a full S.

Usage:
  bash run_retry.sh 20 experiments/head_analysis/run_jlens.py --gpu 0 --smoke
  bash run_retry.sh 20 experiments/head_analysis/run_jlens.py --gpu 0 \
      --exp-id jlens_v1 --num-clips 2 --n-freq 512 --n-random 512
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib  # noqa: E402
import jlens_lib as jl  # noqa: E402
import mask_lib as ml  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

# resolved from this file so the script runs the same on the host and in the container
REPO = Path(__file__).resolve().parents[2]
READOUT_LAYERS = [0, 6, 12, 18, 24, 30, 35]


def source_span(model, inputs, prompt_len, coc_start, coc_end, t_total, mode):
    """Positions the Jacobian is averaged over. Vision/traj/sink always excluded.

    mode="coc" is the default because Alpamayo's prompt text is a *constant*
    instruction template (prompt_len is 3086 for every clip), and it outnumbers
    the generated CoC ~157 to ~13. Averaging over it makes J_l describe how the
    model chews through boilerplate rather than where its reasoning lives, which
    is what the first pass measured: the readout inverted past layer 20 because
    the late-layer Jacobian predicts the next *template* token, and MLP scores
    failed to reproduce across disjoint clips.

    mode="text_coc" keeps the original span for comparison.
    """
    spans = lib.compute_spans(model, inputs["input_ids"])
    src = torch.zeros(t_total, dtype=torch.bool)
    if mode == "text_coc":
        src[:prompt_len] = spans["text"]
    src[coc_start:coc_end] = True
    return torch.nonzero(src).flatten(), spans


def load_coc_refs(exp_id):
    """CoC token ids the full model already generated, from make_teacher_refs.

    Train shards only -- the calibration clips are a subset of train, so this
    keeps test clean. Using the stored CoC instead of a fresh rollout removes
    sampling noise from the Jacobian and makes the run reproducible.
    """
    refs = {}
    for shard in sorted((REPO / "outputs" / exp_id).glob("train_*.json")):
        refs.update(json.loads(shard.read_text()))
    return refs


def rollout_clip(model, processor, data, args, seed):
    """CoC rollout, used only when a clip is absent from the teacher refs."""
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end].cpu()  # (1, T)
    del roll
    torch.cuda.empty_cache()
    return {"seq_tf": seq_tf, "prompt_len": prompt_len,
            "coc_start": coc_start, "coc_end": coc_end}


def seq_from_ref(inputs, ref):
    """Teacher-forced sequence built from a stored CoC continuation."""
    prompt_len = inputs["input_ids"].shape[1]
    if prompt_len != ref["prompt_len"]:
        raise RuntimeError(
            f"prompt length {prompt_len} != stored {ref['prompt_len']}; the refs were "
            "generated with a different prompt build and cannot be teacher-forced"
        )
    coc = torch.tensor(ref["coc_ids"], dtype=inputs["input_ids"].dtype,
                       device=inputs["input_ids"].device).view(1, -1)
    seq_tf = torch.cat([inputs["input_ids"], coc], dim=1)  # (1, T)
    return {"seq_tf": seq_tf, "prompt_len": prompt_len,
            "coc_start": prompt_len, "coc_end": seq_tf.shape[1]}


def jacobian_clip(model, processor, data, ref, wu, taps, wstats, args, probe_acc):
    """One forward + S backwards. Returns (2, L, S, d) on cuda and a record."""
    mem0 = torch.cuda.memory_allocated() / 1024**3
    inputs = lib.build_inputs(model, processor, data, "cuda")
    roll = (seq_from_ref(inputs, ref) if ref is not None
            else rollout_clip(model, processor, data, args, args.seed))
    seq_tf = roll["seq_tf"].to("cuda")
    t_total = seq_tf.shape[1]
    coc_len = roll["coc_end"] - roll["coc_start"]
    span, spans = source_span(
        model, inputs, roll["prompt_len"], roll["coc_start"], roll["coc_end"], t_total,
        args.span,
    )
    n_src, n_vision = int(len(span)), int(spans["vision"].sum())
    span = span.to("cuda")
    taps.set_span(span)
    wstats.set_span(span)

    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf,
            attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"],
            use_cache=False,
        )
    last_hidden = out.last_hidden_state  # (1, T, d), post final RMSNorm
    wstats.add_positions(len(span))
    t_fwd = time.time() - t0

    # probe activations for the kurtosis / readout diagnostics (G1)
    sel = span[torch.linspace(0, len(span) - 1, min(args.n_probe, len(span))).long()]
    probes = torch.stack([taps.out[li][0, sel].detach().float() for li in range(taps.n_layers)])
    probe_acc.append(probes.cpu())  # (L, n_probe, d)

    t0 = time.time()
    v = jl.build_jlens(taps, last_hidden, wu, chunk_log=args.log_every)
    t_bwd = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    del out, last_hidden, inputs, seq_tf, roll, probes, span, sel
    taps.clear_tensors()
    torch.cuda.empty_cache()
    rec = {"t_total": t_total, "n_src": n_src, "coc_len": coc_len,
           "n_vision": n_vision, "fwd_s": t_fwd, "bwd_s": t_bwd,
           "bwd_per_token_s": t_bwd / max(wu.shape[0], 1), "peak_gb": peak,
           "held_gb_before": mem0, "held_gb_after": torch.cuda.memory_allocated() / 1024**3}
    return v, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=8)
    ap.add_argument("--clip-offset", type=int, default=0,
                    help="start of the calibration slice; a disjoint slice gives an "
                         "independent estimate to check the Jacobian has converged")
    ap.add_argument("--exp-id", type=str, default="jlens_v1")
    ap.add_argument("--coc-refs", type=str, default="teacher_refs",
                    help="make_teacher_refs output; supplies the CoC corpus and the "
                         "teacher-forced continuations")
    ap.add_argument("--n-freq", type=int, default=512)
    ap.add_argument("--n-random", type=int, default=512)
    ap.add_argument("--n-probe", type=int, default=64)
    ap.add_argument("--span", choices=["coc", "text_coc"], default="coc",
                    help="Jacobian source positions; see source_span")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--reserve-gb", type=float, default=42.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=64)
    ap.add_argument("--save-vectors", action="store_true",
                    help="dump the (2,L,S,d) J-lens tensor (~600 MB at S=1024)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 clip, S=16 -- measures one backward before committing")
    args = ap.parse_args()
    if args.smoke:
        args.num_clips, args.n_freq, args.n_random = 1, 8, 8
        args.exp_id = "jlens_smoke"
        args.log_every = 4

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    clips = split["calib"][args.clip_offset : args.clip_offset + args.num_clips]

    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    # every weight is frozen, so without this the residual stream carries no
    # autograd graph and the taps would receive nothing
    model.vlm.enable_input_require_grads()

    layers = model.vlm.model.language_model.layers
    tc = model.vlm.config.text_config

    # ---- dictionary from the stored CoC corpus ------------------------------
    # ~900 train clips of this model's own reasoning output, so the frequent half
    # is grounded in what it actually says rather than in a hand-picked word list
    refs = load_coc_refs(args.coc_refs)
    corpus = [t for r in refs.values() for t in r["coc_ids"]]
    token_ids, n_freq = jl.select_tokens(corpus, model, args.n_freq, args.n_random,
                                         seed=args.seed)
    wu = jl.unembed_rows(model, token_ids)  # (S, d)
    s_dict = wu.shape[0]
    print(f"dictionary: {s_dict} tokens = {n_freq} CoC + {s_dict - n_freq} random "
          f"(corpus {len(corpus)} tokens from {len(refs)} clips)", flush=True)

    datas = []
    for clip_id in clips:
        t0 = time.time()
        datas.append(load_physical_aiavdataset(clip_id, t0_us=5_100_000))
        print(f"[data {len(datas)}/{len(clips)}] {clip_id} ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: Jacobian ---------------------------------------------------
    taps = jl.ProbeTaps(layers)
    wstats = jl.WriteStats(layers, tc.num_attention_heads, tc.head_dim,
                           tc.intermediate_size, "cuda")
    v_acc = torch.zeros(2, len(layers), s_dict, tc.hidden_size, device="cuda",
                        dtype=torch.float32)
    # odd-clip partial sum: v_even = v_acc - v_odd gives a split-half estimate, so a
    # single run reports its own Jacobian noise floor without a paired run
    v_odd = torch.zeros_like(v_acc)
    n_odd = 0
    probe_acc, records = [], []
    for ci, data in enumerate(datas):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        v, rec = jacobian_clip(model, processor, data, refs.get(clips[ci]), wu, taps,
                               wstats, args, probe_acc)
        v_acc += v
        if ci % 2:
            v_odd += v
            n_odd += 1
        del v
        rec["clip_id"] = clips[ci]
        records.append(rec)
        print(f"[jac {ci + 1}/{len(clips)}] T={rec['t_total']} src={rec['n_src']} "
              f"bwd={rec['bwd_s']:.0f}s ({rec['bwd_per_token_s'] * 1000:.0f}ms/token) "
              f"peak={rec['peak_gb']:.1f}GB held {rec['held_gb_before']:.1f}->"
              f"{rec['held_gb_after']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
    n_even = len(clips) - n_odd
    v_acc /= len(clips)
    taps.remove()
    mlp_sq, head_cov = wstats.finalize()
    wstats.remove()

    # ---- scores + diagnostics ----------------------------------------------
    scores = jl.unit_jscores(model, v_acc, mlp_sq, head_cov)
    n_p = min(p.shape[1] for p in probe_acc)  # clips can differ if a CoC is short
    probes = torch.stack([p[:, :n_p] for p in probe_acc]).mean(0)  # (L, n_probe, d)
    # random half only: the CoC half is frequency-selected and would make the
    # readout bimodal by construction (see excess_kurtosis)
    kurt = np.array([jl.excess_kurtosis(v_acc[1, li, n_freq:], probes[li].to("cuda"))
                     for li in range(len(layers))])
    cka = jl.cka_matrix(v_acc[1])
    auc = np.array([jl.freq_auc(v_acc[1, li], probes[li].to("cuda"), n_freq)
                    for li in range(len(layers))])
    readouts = {
        str(li): jl.readout(v_acc[1, li], probes[li].mean(0).to("cuda"), token_ids,
                            model.tokenizer)
        for li in READOUT_LAYERS if li < len(layers)
    }

    # split-half: same write stats both sides, so this isolates Jacobian noise
    split_half = {}
    if n_odd and n_even:
        s_odd = jl.unit_jscores(model, v_odd / n_odd, mlp_sq, head_cov)
        s_even = jl.unit_jscores(model, (v_acc * len(clips) - v_odd) / n_even, mlp_sq, head_cov)
        for k in ("q_j", "mlp_j"):
            rho = [spearmanr(s_odd[k][li], s_even[k][li]).statistic
                   for li in range(len(layers))]
            split_half[k] = np.array(rho)
        del s_odd, s_even
    del v_odd

    # weight-norm magnitude while the model is loaded: G2(a) compares against it
    mag_q, mag_mlp = ml.magnitude_scores(layers, tc.num_attention_heads, tc.head_dim,
                                         tc.intermediate_size)
    np.savez(out_dir / "jlens.npz", token_ids=np.array(token_ids), kurtosis=kurt, cka=cka,
             freq_auc=auc, n_freq=n_freq, mag_q=mag_q, mag_mlp=mag_mlp,
             mlp_sq=mlp_sq.cpu().numpy(), head_cov=head_cov.cpu().numpy(),
             **{f"split_{k}": v for k, v in split_half.items()},
             **{k: v for k, v in scores.items()})
    if args.save_vectors:
        torch.save({"v": v_acc.half().cpu(), "token_ids": token_ids},
                   out_dir / "jlens_vectors.pt")

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B",
        "purpose": "J-lens over the VLM tower + J-space unit scores (Stage A/B)",
        "reference": "transformer-circuits.pub/2026/workspace (2026-07-06)",
        "num_clips": len(clips), "clip_ids": clips, "seed": args.seed,
        "coc_refs": args.coc_refs, "teacher_forced": True,
        "dict_tokens": s_dict, "n_freq": n_freq, "n_random": s_dict - n_freq,
        "source_span": args.span,
        "taps": {"mid": "post_attention_layernorm input (Q heads)",
                 "out": "decoder layer output (MLP channels)"},
        "gpu": torch.cuda.get_device_name(device),
        "shapes": {k: list(v.shape) for k, v in scores.items()},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "per_clip": records, "kurtosis": kurt.tolist(), "freq_auc": auc.tolist(),
        "readouts": readouts,
    }, indent=2))
    write_summary(out_dir, scores, kurt, auc, readouts, records, s_dict, split_half)
    print("saved ->", out_dir, flush=True)


def write_summary(out_dir, scores, kurt, auc, readouts, records, s_dict, split_half):
    n_layers = kurt.shape[0]
    lo, hi = int(0.38 * n_layers), int(0.92 * n_layers)
    lines = [
        f"J-lens over {s_dict} dictionary tokens, {len(records)} clips",
        f"backward cost: {np.mean([r['bwd_per_token_s'] for r in records]) * 1000:.0f} ms/token, "
        f"peak {max(r['peak_gb'] for r in records):.1f} GB",
        "",
        "G1 -- layer bands (paper: sensory 0-38%, workspace 38-92%, motor 92-100%)",
        f"  predicted workspace band for {n_layers} layers: {lo}-{hi}",
        f"  mean excess kurtosis  sensory={kurt[:lo].mean():.2f}  "
        f"workspace={kurt[lo:hi].mean():.2f}  motor={kurt[hi:].mean():.2f}",
        f"  plateau present: {kurt[lo:hi].mean() > kurt[:lo].mean()}",
        "",
        "G1 -- CoC vocabulary vs random vocabulary in the readout (AUC, 0.5 = no signal)",
        f"  sensory={np.nanmean(auc[:lo]):.3f}  workspace={np.nanmean(auc[lo:hi]):.3f}  "
        f"motor={np.nanmean(auc[hi:]):.3f}  best layer={int(np.nanargmax(auc))} "
        f"({np.nanmax(auc):.3f})",
        "",
        "G1 -- readouts (top dictionary tokens at the mean text-span activation)",
    ]
    for li, toks in readouts.items():
        lines.append(f"  layer {li:>2}: " + " ".join(repr(t) for t, _ in toks[:10]))
    lines += ["", "unit scores (per-layer mean)"]
    for k in ("q_j", "q_w", "mlp_j", "mlp_w"):
        a = scores[k]
        lines.append(f"  {k:<6} mean={a.mean():.4e}  min={a.min():.4e}  max={a.max():.4e}")
    jf_q = scores["q_j"] / np.clip(scores["q_w"], 1e-12, None)
    jf_m = scores["mlp_j"] / np.clip(scores["mlp_w"], 1e-12, None)
    lines.append(f"  jfrac  q: mean={jf_q.mean():.4f} sd={jf_q.std():.4f}   "
                 f"mlp: mean={jf_m.mean():.4f} sd={jf_m.std():.4f}")
    if split_half:
        lines += ["", "split-half reproducibility of the J-score (odd vs even clips)",
                  "  this is the noise floor: any G2 margin smaller than 1-rho is unreadable"]
        for k, r in split_half.items():
            lines.append(f"  {k:<6} median rho={np.nanmedian(r):+.3f}  "
                         f"min={np.nanmin(r):+.3f}  max={np.nanmax(r):+.3f}")
    lines += ["", "G2 requires analyze_jspace.py (correlations vs magnitude / coc / traj)."]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
