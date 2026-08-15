"""VQA-context Taylor importance, and the CoC control on the same images.

Tests the bottleneck hypothesis from the 2026-08-14 LingoQA study: the language
criteria (`I_CoC`, `J`) fail not because their token dictionary is narrow, nor because
100 calibration clips is too few, but because the *positions* where the objective is
observed are narrow. `prune_lib.coc_nll` scores the ~14 tokens of a terse
"action, justification" line; `jlens_lib` restricts source positions to text and
generated-CoC tokens. A unit that only fires when answering "what colour is the traffic
light" leaves no gradient there and is pruned.

Two objectives are accumulated over the same segments, so their selections are
directly comparable:

  vqa   `create_vqa_message` with a LingoQA train question, the reference answer
        teacher-forced, cross-entropy over the answer positions only.
  coc   the stock CoC prompt on the same images, the model's own rollout
        teacher-forced -- exactly `run_importance.py`'s objective 1, but on LingoQA
        images rather than PhysicalAI-AV ones.

That second one is the domain control. LingoQA is Wayve UK footage while the existing
`importance_v*` were computed on NVIDIA PhysicalAI-AV, so `I_VQA` differing from the
shipped `I_CoC` could be either the context or the domain. `coc` here shares the domain
and differs only in the objective, which separates the two.

The trajectory objective is absent by necessity: LingoQA ships no ego trajectory, so
`lib.gt_actions` has nothing to read. Only the VLM tower's Q heads and MLP channels are
scored, which is the pool the u40_v2 family prunes anyway.

Calibration draws from LingoQA **train** only. `val.parquet` is the held-out benchmark
and is never touched here.

Usage:
  python experiments/lingoqa/run_vqa_importance.py --num-clips 100 --gpu 7
  python experiments/lingoqa/run_vqa_importance.py --num-clips 4 --exp-id vqa_imp_smoke
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))

# expert_per_clip must import first: it installs the transformers hub patch forcing
# local_files_only for the gated Cosmos repo, as every runner in this repo does
# isort: off
import expert_per_clip  # noqa: F401  imported for the hub patch, not for a name
import analysis_lib as lib
import lingo_lib as ll
import prune_lib as pl
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
# isort: on

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
TRAIN = Path("/mnt/nvme1n1/ad_vla/data/lingoqa_train")


def load_train_manifest(n_clips, n_questions, seed):
    """Segments and their questions from LingoQA train, drawn deterministically.

    Sampling is by a hash of the id rather than a shuffled index, so the draw does not
    move when the parquet's row order changes and a smaller --num-clips is a prefix of
    a larger one.
    """
    df = pd.read_parquet(TRAIN / "train.parquet")

    def h(s):
        return int.from_bytes(hashlib.sha256(f"{seed}:{s}".encode()).digest()[:8], "big")

    segments = sorted(df.segment_id.unique(), key=h)[:n_clips]
    keep = df[df.segment_id.isin(set(segments))]
    out = []
    for seg in segments:
        sub = keep[keep.segment_id == seg]
        # one row per question; a train question carries a single reference answer
        qs = sub.drop_duplicates("question_id").sort_values("question_id", key=lambda c: c.map(h))
        out.append({"segment_id": seg,
                    "questions": [{"question": r.question, "answer": r.answer}
                                  for r in qs.head(n_questions).itertuples()]})
    return out


def vqa_nll(model, processor, data, question, answer, device):
    """CE over the reference answer's tokens, under the native VQA prompt.

    Mirrors `prune_lib.coc_nll` -- hidden at p predicts token p+1 -- but the scored
    span is the answer rather than the CoC, which is the whole point of the run.
    Returns None when the prompt cannot be built for this row.
    """
    msgs = helper.create_vqa_message(
        frames=data["image_frames"].flatten(0, 1),
        question=question,
        camera_indices=data["camera_indices"])
    enc = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt")
    enc = helper.to_device(dict(enc), device)
    prompt_ids = enc.pop("input_ids")                       # (1, T_prompt)

    tok = model.tokenizer
    ans = tok(str(answer), add_special_tokens=False)["input_ids"]
    end = tok.convert_tokens_to_ids("<|answer_end|>")
    if end is not None and end >= 0:
        ans = ans + [end]
    if not ans:
        return None, None
    ans_t = torch.tensor([ans], device=device)
    seq_tf = torch.cat([prompt_ids, ans_t], dim=1)          # (1, T_prompt + T_ans)
    start, stop = prompt_ids.shape[1], seq_tf.shape[1]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, _, _ = pl.vlm_forward_with_grad(model, seq_tf, enc, use_cache=False)
    h = hidden[:, start - 1: stop - 1]
    logits = model.vlm.lm_head(h).float()
    return F.cross_entropy(logits[0], seq_tf[0, start:stop]), len(ans)


def coc_nll_on(model, processor, data, max_gen, seed, device):
    """The stock CoC objective, on these images. Domain control for the comparison."""
    inputs = lib.build_inputs(model, processor, data, device)
    prompt_len = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=max_gen)
    coc_end = roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]
    del roll
    if coc_end <= prompt_len:
        return None, None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, _, _ = pl.vlm_forward_with_grad(
            model, seq_tf, inputs["tokenized_data"], use_cache=False)
    return pl.coc_nll(model, hidden, seq_tf, prompt_len, coc_end), int(coc_end - prompt_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-questions", type=int, default=4,
                    help="questions per segment for the vqa objective")
    ap.add_argument("--exp-id", default="importance_vqa")
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", default="4,5,6,7")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    man = load_train_manifest(args.num_clips, args.num_questions, args.seed)

    device = ll.pick_gpu(args.reserve_gb, [int(g) for g in args.gpu.split(",")])
    print(f"{args.exp_id} | {len(man)} segments x {args.num_questions} questions | "
          f"{torch.cuda.get_device_name(device)}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # the gates sit downstream of the embeddings; without this the early layers carry
    # no graph at all (see prune_lib.retain_cache_grads)
    model.vlm.enable_input_require_grads()
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "eager")
    lib.set_expert_attn_impl(model, "eager")

    cfg = model.vlm.config.text_config
    gates = pl.UnitGates(model.vlm.model.language_model.layers, cfg.num_attention_heads,
                         cfg.head_dim, cfg.intermediate_size, "cuda", torch.float32)

    L, H, I = len(model.vlm.model.language_model.layers), cfg.num_attention_heads, \
        cfg.intermediate_size
    acc = {o: {"vlm_q": np.zeros((L, H)), "vlm_mlp": np.zeros((L, I))} for o in ("vqa", "coc")}
    counts = {"vqa": 0, "coc": 0}
    records = []

    t0 = time.time()
    for i, m in enumerate(man):
        seg = m["segment_id"]
        data = ll.load_segment(seg, data=TRAIN, device="cuda", split="train")
        rec = {"segment_id": seg, "n_questions": len(m["questions"])}

        # ---- objective: VQA answer NLL, one backward per question ----
        for q in m["questions"]:
            loss, n_ans = vqa_nll(model, processor, data, q["question"], q["answer"], "cuda")
            if loss is None:
                continue
            loss.backward()
            acc["vqa"]["vlm_q"] += gates.q_scores()
            acc["vqa"]["vlm_mlp"] += gates.mlp_scores()
            gates.zero_grads()
            counts["vqa"] += 1
            rec.setdefault("vqa_nll", []).append(round(float(loss), 4))
            rec.setdefault("vqa_len", []).append(n_ans)
            del loss

        # ---- control: the stock CoC objective on the same images ----
        loss, n_coc = coc_nll_on(model, processor, data, args.max_gen, args.seed + i, "cuda")
        if loss is not None:
            loss.backward()
            acc["coc"]["vlm_q"] += gates.q_scores()
            acc["coc"]["vlm_mlp"] += gates.mlp_scores()
            gates.zero_grads()
            counts["coc"] += 1
            rec.update({"coc_nll": round(float(loss), 4), "coc_len": n_coc})
            del loss

        records.append(rec)
        torch.cuda.empty_cache()
        print(f"[{i + 1}/{len(man)}] {seg[:8]} vqa={len(rec.get('vqa_nll', []))} "
              f"coc_len={rec.get('coc_len')} ({time.time() - t0:.0f}s)", flush=True)

        if (i + 1) % 10 == 0 or i + 1 == len(man):
            np.savez(out_dir / "importance.npz",
                     **{f"{o}_{k}": v / max(counts[o], 1) for o, d in acc.items()
                        for k, v in d.items()})
            (out_dir / "records.json").write_text(json.dumps(records, indent=2))

    gates.remove()
    np.savez(out_dir / "importance.npz",
             **{f"{o}_{k}": v / max(counts[o], 1) for o, d in acc.items()
                for k, v in d.items()})
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "purpose": "VQA-context Taylor importance + CoC control on the same images",
        "objectives": {"vqa": "reference-answer NLL under create_vqa_message",
                       "coc": "own-rollout CoC NLL under create_message"},
        "pool": "VLM tower Q heads and MLP channels (no traj: LingoQA has no ego GT)",
        "data": "LingoQA scenery TRAIN; val.parquet untouched",
        "num_clips": len(man), "num_questions": args.num_questions,
        "backwards": counts, "seed": args.seed, "max_gen": args.max_gen,
        "segment_ids": [m["segment_id"] for m in man],
    }, indent=2))
    print(f"done: {counts} backwards, {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
