"""Stage 2 of the LingoQA reasoning probe: frozen reader answers, Lingo-Judge scores.

Three conditions, selected with --condition:

  coc    the reader sees the arm's generated CoC text plus the question. This is the
         measurement. The reader never sees the frames, so a correct answer can only
         come from information that survived into the reasoning text.
  blind  the reader sees the question alone. The language-prior floor. LingoQA
         questions carry strong priors ("Are there any pedestrians?" -> "No" is often
         right), so without this floor the `coc` accuracy is uninterpretable.
  vlm    Qwen3-VL-8B answers from the frames directly. The ceiling: how much of the
         gap is the CoC versus the task itself.

`blind` and `vlm` are arm-independent -- run once, reused for every arm.

The reader is instructed to always commit to an answer rather than abstain, so that
`coc` and `blind` produce the same response style and their difference is a clean
information-gain measure rather than a difference in hedging behaviour.

Scoring is LingoQA's own `LingoJudge` at its own threshold (logit > 0), taking the max
over the two human references per question, imported from the LingoQA checkout rather
than reimplemented.

Usage:
  python experiments/lingoqa/run_lingo_judge.py --condition coc --coc-run lingo_pilot_baseline
  python experiments/lingoqa/run_lingo_judge.py --condition blind --limit 20
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
LINGOQA = Path("/home/cvlab21/project/chan/LingoQA/benchmark")
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(LINGOQA))

# expert_per_clip must import first: it installs the transformers hub patch forcing
# local_files_only for the gated Cosmos repo, as every runner in this repo does
# isort: off
import expert_per_clip  # noqa: F401  imported for the hub patch, not for a name
import lingo_lib as ll
from judge import LingoJudge  # LingoQA's own metric
# isort: on

READER = "Qwen/Qwen3-8B"
VLM_READER = "Qwen/Qwen3-VL-8B-Instruct"

SYS_COC = (
    "You are reading the internal reasoning notes that a self-driving system wrote "
    "about a driving scene. Answer the question about that scene using the notes as "
    "your only evidence. Reply with one short sentence, in the style of a direct "
    "answer such as 'Yes, there is one pedestrian.' or 'No, the road is clear.'. "
    "Never say that you are unsure or that the notes do not say -- if the notes do "
    "not cover it, give your single most likely answer for an ordinary driving scene."
)
SYS_BLIND = (
    "You are answering questions about an ordinary driving scene that you cannot see. "
    "Reply with one short sentence, in the style of a direct answer such as "
    "'Yes, there is one pedestrian.' or 'No, the road is clear.'. Never say that you "
    "are unsure or that you cannot see -- always give your single most likely answer."
)
SYS_VLM = (
    "You are answering questions about a driving scene shown in four consecutive "
    "front-camera frames. Reply with one short sentence, in the style of a direct "
    "answer such as 'Yes, there is one pedestrian.' or 'No, the road is clear.'."
)


def build_prompts(questions, coc_by_seg, condition):
    """(system, user) chat pairs, one per question, in manifest order."""
    out = []
    for q in questions:
        if condition == "coc":
            user = (f"Reasoning notes:\n\"\"\"\n{coc_by_seg[q['segment_id']]}\n\"\"\"\n\n"
                    f"Question: {q['question']}")
            out.append((SYS_COC, user))
        elif condition == "blind":
            out.append((SYS_BLIND, f"Question: {q['question']}"))
        else:
            out.append((SYS_VLM, f"Question: {q['question']}"))
    return out


@torch.inference_mode()
def run_reader_text(prompts, device, batch_size, max_new_tokens):
    """Greedy batched generation with Qwen3-8B, thinking mode off."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(READER, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.bfloat16).to(device).eval()
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for s, u in prompts
    ]
    answers = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(batch)):
            gen = out[j, enc["input_ids"].shape[1]:]
            answers.append(tok.decode(gen, skip_special_tokens=True).strip())
        print(f"  reader {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return answers


@torch.inference_mode()
def run_reader_vlm(questions, prompts, device, max_new_tokens):
    """Ceiling condition: Qwen3-VL-8B answering from the frames themselves."""
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(VLM_READER)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_READER, dtype=torch.bfloat16).to(device).eval()
    answers = []
    for n, (q, (sysmsg, user)) in enumerate(zip(questions, prompts)):
        d = ll.DATA / "images" / "val" / q["segment_id"]
        imgs = [Image.open(d / f"{i}.jpg").convert("RGB") for i in range(5 - ll.N_FRAMES, 5)]
        msgs = [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                {"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                 + [{"type": "text", "text": user}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        answers.append(proc.decode(out[0, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        if (n + 1) % 25 == 0:
            print(f"  vlm {n + 1}/{len(questions)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["coc", "blind", "vlm"])
    ap.add_argument("--coc-run", default=None,
                    help="exp_id of the Stage 1 run (required for --condition coc)")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N segments")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--reserve-gb", type=float, default=18.0)
    ap.add_argument("--gpu", default="4,5,6,7",
                    help="comma-separated cards to try, in order")
    args = ap.parse_args()

    if args.condition == "coc" and not args.coc_run:
        ap.error("--condition coc requires --coc-run")

    questions, segments = ll.load_manifest()
    coc_by_seg, coc_cfg = {}, None
    if args.condition == "coc":
        coc_dir = REPO / "outputs" / args.coc_run
        rows = json.loads((coc_dir / "coc.json").read_text())
        coc_by_seg = {r["segment_id"]: r["gen_coc"] for r in rows}
        coc_cfg = json.loads((coc_dir / "config.json").read_text())
        segments = [s for s in segments if s in coc_by_seg]  # Stage 1 may be a partial run
    if args.limit:
        segments = segments[: args.limit]
    keep = set(segments)
    questions = [q for q in questions if q["segment_id"] in keep]

    tag = args.coc_run if args.condition == "coc" else args.condition
    exp_id = args.exp_id or f"lingo_judge_{tag}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ll.pick_gpu(args.reserve_gb, [int(g) for g in args.gpu.split(",")])
    print(f"{exp_id} | condition={args.condition} | {len(questions)} questions over "
          f"{len(segments)} segments | {torch.cuda.get_device_name(device)}", flush=True)

    t0 = time.time()
    prompts = build_prompts(questions, coc_by_seg, args.condition)
    if args.condition == "vlm":
        answers = run_reader_vlm(questions, prompts, device, args.max_new_tokens)
    else:
        answers = run_reader_text(prompts, device, args.batch_size, args.max_new_tokens)

    judge = LingoJudge().eval().to(device)
    scores = judge.compute([q["question"] for q in questions],
                           [q["references"] for q in questions], answers)

    rows = []
    for q, a, s in zip(questions, answers, scores.tolist()):
        rows.append({"question_id": q["question_id"], "segment_id": q["segment_id"],
                     "question": q["question"], "references": q["references"],
                     "answer": a, "score": s, "correct": bool(s > 0.0)})
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))

    acc = sum(r["correct"] for r in rows) / len(rows)
    per_seg = {}
    for r in rows:
        per_seg.setdefault(r["segment_id"], []).append(r["correct"])
    metrics = {"condition": args.condition, "n_questions": len(rows),
               "n_segments": len(per_seg), "accuracy": acc,
               "mean_score": float(scores.mean()),
               "segment_accuracy": {k: sum(v) / len(v) for k, v in per_seg.items()}}
    summary = (f"LingoQA reasoning probe -- condition={args.condition} tag={tag}\n"
               f"questions {len(rows)} over {len(per_seg)} segments\n"
               f"Lingo-Judge accuracy {acc * 100:.1f}%  (threshold logit>0)\n"
               f"mean judge logit {float(scores.mean()):+.4f}\n"
               f"reader {VLM_READER if args.condition == 'vlm' else READER}, greedy\n"
               f"NOTE: paired deltas between arms are the claim; the absolute number is\n"
               f"not a LingoQA leaderboard score (out-of-domain, synthetic ego history).\n")
    ll.write_outputs(out_dir, {
        "condition": args.condition, "coc_run": args.coc_run, "coc_config": coc_cfg,
        "reader": VLM_READER if args.condition == "vlm" else READER,
        "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
        "judge": "wayveai/Lingo-Judge", "judge_threshold": "logit > 0.0",
        "n_questions": len(rows), "n_segments": len(per_seg),
    }, metrics, summary)
    print(summary + f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
