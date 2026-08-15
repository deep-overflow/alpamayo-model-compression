"""Standard-protocol LingoQA benchmark: Alpamayo answers the questions itself.

This is the *reportable* LingoQA number, unlike the CoC-as-evidence probe in
`run_lingo_coc.py` + `run_lingo_judge.py`. The difference matters:

  probe      model never sees the question; a frozen reader answers from the CoC.
             Measures how much scene information survived into the reasoning text.
             Absolute value is NOT a LingoQA score -- only arm-to-arm deltas are.
  this       model gets video + question and writes the answer itself, exactly as
             the benchmark specifies. The scored sentence is Alpamayo's own.

Alpamayo 1.5 ships a native VQA interface (`helper.create_vqa_message`): the question
goes in as `<|question_start|>...<|question_end|>` and the assistant turn is primed
with `<|answer_start|>`. Crucially that template carries **no traj-history
placeholder**, so unlike the CoC path this needs no synthetic ego trajectory -- the
one adaptation the probe could not avoid disappears here.

Where the resulting number belongs, from Table 5 of the paper, which reports fine-tuned
and zero-shot models as separate categories with the frame count disclosed per row:

  humans, 5 frames                        96.6      (1 frame: 81.8)
  fine-tuned on LingoQA train, 5 frames   60.8      LLaVA 59.0, BLIP-2 52.2 (1 frame)
  zero-shot, GPT-4V 5 frames              59.6      LLaVA 49.4, FUYU 45.4 (1 frame)

Alpamayo is zero-shot here, so it belongs in the third row and must not be compared
against the fine-tuned block. Remaining caveats to state wherever it is reported:
  - out-of-domain: LingoQA is Wayve UK footage, Alpamayo is trained on NVIDIA
    PhysicalAI-AV. The zero-shot models in Table 5 are also out-of-domain, so this is
    a property of the comparison rather than a defect unique to us.
  - LingoQA gives 5 front-camera frames over 4 s; Alpamayo consumes 4 per camera, so
    the last 4 are used, mapped to camera index 1 ("Front camera"). Frame count is a
    disclosed per-row property in Table 5 (models there range 0 to 5), and it matters:
    human accuracy falls 96.6 -> 81.8 going from 5 frames to 1.

Writes `predictions.csv` with the columns LingoQA's own `evaluate.py` expects
(question_id, segment_id, answer), 500 rows. Score it with the upstream script:

  python /home/cvlab21/project/chan/LingoQA/benchmark/evaluate.py \
      --predictions_path outputs/<exp_id>/predictions.csv

Usage:
  python experiments/lingoqa/run_lingo_vqa.py --arm baseline --gpu 5
  python experiments/lingoqa/run_lingo_vqa.py --arm outputs/slim_dual_u40_v2 --limit 20
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

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
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
# isort: on

MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"

# Appendix E of the paper (Table 8) reports zero-shot models under distinct response-style
# conditions and shows Lingo-Judge over-rates long-form answers: it scores FUYU at 45.4%
# against 17.69% from human raters, and agrees with humans only on short answers (GPT-4V
# concise 59.6 vs 56.67 human). Alpamayo's answers run ~22 words against a 9-word human
# median, i.e. squarely in the inflating regime, so both conditions are measured. The paper
# names the conditions but does not publish their wording, so this suffix is ours and is
# recorded in config.json rather than left implicit.
CONCISE_SUFFIX = " Answer concisely in one short sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--max-gen", type=int, default=128,
                    help="answers run ~25 words; 64 truncated them mid-sentence")
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    ap.add_argument("--gpu", default="4,5,6,7",
                    help="comma-separated cards to try, in order. Defaults to the Ada "
                         "cards: every arm of a comparison must stay on one architecture")
    ap.add_argument("--style", default="unprompted", choices=["unprompted", "concise"],
                    help="response-style condition, mirroring the paper's Table 8. "
                         "'unprompted' passes the benchmark question verbatim and is the "
                         "primary, non-gameable number; 'concise' appends a brevity "
                         "instruction to control the paper's documented long-answer bias")
    ap.add_argument("--strict-deterministic", action="store_true")
    args = ap.parse_args()

    tag = "baseline" if args.arm == "baseline" else Path(args.arm).name
    if args.style != "unprompted":
        tag = f"{tag}_{args.style}"
    exp_id = args.exp_id or f"lingo_vqa_{tag}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.json"

    questions, _ = ll.load_manifest()
    if args.limit:
        questions = questions[: args.limit]

    torch.use_deterministic_algorithms(True, warn_only=not args.strict_deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = ll.pick_gpu(args.reserve_gb, [int(g) for g in args.gpu.split(",")])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{exp_id} | {len(questions)} questions | {gpu_name}", flush=True)

    if args.arm == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.arm, device="cuda")

    tok = model.tokenizer
    processor = helper.get_processor(tok)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    ans_end = tok.convert_tokens_to_ids("<|answer_end|>")
    gc = model.vlm.generation_config
    gc.do_sample = False           # greedy: the reported number must not carry sampling noise
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.num_return_sequences = 1
    gc.max_new_tokens = args.max_gen
    gc.output_logits = False
    gc.return_dict_in_generate = True
    gc.pad_token_id = tok.pad_token_id
    gc.eos_token_id = ans_end

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["question_id"] + "|" + r["segment_id"] for r in rows}

    (out_dir / "config.json").write_text(json.dumps({
        "arm": args.arm, "tag": tag, "protocol": "standard LingoQA VQA",
        "message_builder": "helper.create_vqa_message", "synthetic_ego_history": False,
        "n_questions": len(questions), "max_gen": args.max_gen, "decoding": "greedy",
        "style": args.style,
        "style_suffix": CONCISE_SUFFIX if args.style == "concise" else "",
        "gpu": gpu_name, "model_revision": MODEL_REV,
        "frames_per_segment": ll.N_FRAMES, "camera_index": ll.FRONT_CAMERA,
        "zero_shot": True,
        "deterministic": {"use_deterministic_algorithms": True,
                          "warn_only": not args.strict_deterministic,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    seg_cache = {}
    for i, q in enumerate(questions):
        key = q["question_id"] + "|" + q["segment_id"]
        if key in done:
            continue
        t0 = time.time()
        if q["segment_id"] not in seg_cache:
            seg_cache.clear()          # questions are sorted by segment; keep one resident
            seg_cache[q["segment_id"]] = ll.load_segment(q["segment_id"], device="cuda")
        data = seg_cache[q["segment_id"]]

        asked = q["question"] + (CONCISE_SUFFIX if args.style == "concise" else "")
        msgs = helper.create_vqa_message(
            frames=data["image_frames"].flatten(0, 1),
            question=asked,
            camera_indices=data["camera_indices"])
        enc = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt")
        enc = helper.to_device(dict(enc), "cuda")
        ids = enc.pop("input_ids")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm.generate(input_ids=ids, generation_config=gc, **enc)
        gen = out.sequences[0, ids.shape[1]:]
        n_gen = int(gen.shape[0])
        answer = tok.decode(gen, skip_special_tokens=True).strip()

        rows.append({"question_id": q["question_id"], "segment_id": q["segment_id"],
                     "question": q["question"], "asked": asked, "answer": answer,
                     "references": q["references"], "gen_len": n_gen,
                     # a run that hit the cap was cut off mid-sentence, which the judge
                     # will score as written -- track it rather than silently accept it
                     "truncated": n_gen >= args.max_gen})
        if len(rows) % 25 == 0 or i + 1 == len(questions):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(questions)}] {n_gen:3d}tok ({time.time() - t0:.1f}s) "
              f"{answer[:80]!r}", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))

    # exactly the three columns LingoQA's evaluate.py reads; 500 rows expected
    pd.DataFrame([{"question_id": r["question_id"], "segment_id": r["segment_id"],
                   "answer": r["answer"]} for r in rows]).to_csv(
        out_dir / "predictions.csv", index=False)

    trunc = sum(r["truncated"] for r in rows)
    print(f"done: {len(rows)} answers, {trunc} truncated at the {args.max_gen}-token cap, "
          f"{time.time() - t_start:.0f}s -> {out_dir / 'predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
