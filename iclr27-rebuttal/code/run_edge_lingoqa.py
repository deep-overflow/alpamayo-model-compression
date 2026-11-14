"""LingoQA (standard VQA protocol) for Alpamayo 2 Edge, dense or with VLM unit masks (plans/2026-11-14_a2-edge-moop-u40.md).

The Edge twin of run_a2_lingoqa.py: the model answers each of the 500 LingoQA val questions itself, zero-shot, greedy, max 128
new tokens; predictions.csv is scored by experiments/lingoqa/score_lingo_vqa.py (Lingo-Judge).

Prompt: Edge's native VQA conversation (chat_template.build_vqa_conversation + helper._tokenize_messages without the closed
think block, as helper.prepare_vqa_inputs does) with the front camera only: the last 4 of the segment's 5 frames, camera label
"Front camera:", time labels <t=-3.0s> .. <t=0.0s> (LingoQA: a 4-second clip at 1 Hz, 5 frames -> 1 s spacing, inferred), pixel
bounds 163,840-196,608. Edge's VQA input check wants 6 cameras x 4 frames, so the conversation is built directly, as the A2
port did (front camera, 4 frames). Decoding = text_tasks.generate_vqa_answer (greedy; stops at EOS or the trajectory anchor;
trajectory vocab masked), re-implemented only to also return the generated length.

--gate N answers the first N questions with no hooks, all-ones hooks (must be identical) and the masks.

Usage:
  CUDA_VISIBLE_DEVICES=2 /mnt/nvme1n1/ad_vla/venvs/alpamayo2-edge/bin/python iclr27-rebuttal/code/run_edge_lingoqa.py \
      --exp-id iclr27-rebuttal/edge_lingo_dense [--masks outputs/iclr27-rebuttal/edge_moop/masks_moop.npz] [--gate 3]
"""

import os
import sys

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to one card")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "lingoqa"))
import edge_eval_lib as eel
import lingo_lib as ll
from transformers import LogitsProcessorList

FRONT_BOUNDS = (163840, 196608)
TIMESTAMPS_US = torch.tensor([-3_000_000, -2_000_000, -1_000_000, 0], dtype=torch.int64)  # last 4 of 5 frames at 1 Hz


def encode(model, processor, frames, question):
    """frames (4, 3, H, W) uint8 -> tokenized VQA inputs on cuda and the prompt text."""
    from alpamayo2_edge import helper
    from alpamayo2_edge.chat_template import build_vqa_conversation

    data = {"image_frames": frames.unsqueeze(1),  # (4, 1, 3, H, W)
            "camera_indices": torch.full((4,), ll.FRONT_CAMERA, dtype=torch.int64),
            "absolute_timestamps": TIMESTAMPS_US, "camera_pixel_bounds": {ll.FRONT_CAMERA: FRONT_BOUNDS}}
    messages = build_vqa_conversation(data, question, include_camera_ids=model.config.include_camera_ids,
                                      system_prompt=getattr(model.config, "system_prompt", None))
    td = helper._tokenize_messages(messages, processor, include_closed_thinking=False)
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in td.items()}, model.tokenizer.decode(td["input_ids"][0])


def make_generation(model, max_gen):
    """text_tasks.generate_vqa_answer's generation config and logits processor."""
    from alpamayo2_edge.models.alpamayo2_edge import MaskDiscreteTrajectoryLogitsProcessor

    g = copy.deepcopy(model.vlm.generation_config)
    g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    g.num_beams, g.num_return_sequences, g.max_new_tokens = 1, 1, max_gen
    g.pad_token_id = model.tokenizer.pad_token_id
    g.return_dict_in_generate, g.output_scores, g.output_logits = False, False, False
    eos = g.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    g.eos_token_id = list(dict.fromkeys([*eos, model.config.traj_ids["future_start"]]))
    tr = model.config.traj_ids
    lp = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=min(tr["history_id0"], tr["future_id0"]), traj_vocab_size=model.config.traj_vocab_size)])
    return g, lp


@torch.inference_mode()
def answer(model, processor, gen_cfg, lp, frames, question):
    from alpamayo2_edge.chat_template import CLOSED_THINKING

    td, _ = encode(model, processor, frames, question)
    plen = td["input_ids"].shape[1]
    seq = model.vlm.generate(**td, generation_config=gen_cfg, logits_processor=lp)
    comp = seq[:, plen:]  # (1, n)
    text = model.tokenizer.batch_decode(comp, skip_special_tokens=True)[0].strip()
    return text.removeprefix(CLOSED_THINKING).strip(), int(comp.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--max-gen", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--plan", default="iclr27-rebuttal/plans/2026-11-14_a2-edge-moop-u40.md")
    args = ap.parse_args()
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    questions, _ = ll.load_manifest()
    if args.limit:
        questions = questions[: args.limit]
    if args.gate:
        questions = questions[: args.gate]

    eel.set_determinism()
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)
    gen_cfg, lp = make_generation(model, args.max_gen)
    gpu = torch.cuda.get_device_name(0)
    masks = None
    if args.masks:
        z = np.load(args.masks)
        masks = (z["q_mask"], z["mlp_mask"])
    seg_cache = {}

    def frames_of(seg):
        if seg not in seg_cache:
            seg_cache.clear()  # questions are sorted by segment; keep one resident
            seg_cache[seg] = ll.load_segment(seg)["image_frames"][0]  # (4, 3, H, W) uint8
        return seg_cache[seg]

    if args.gate:
        assert masks is not None, "--gate needs --masks"
        res = {}
        for mode in ("none", "ones", "masks"):
            hooks = None
            if mode == "ones":
                hooks = eel.EdgeVlmMasks(model, np.ones_like(masks[0]), np.ones_like(masks[1]))
            elif mode == "masks":
                hooks = eel.EdgeVlmMasks(model, *masks)
            res[mode] = [answer(model, processor, gen_cfg, lp, frames_of(q["segment_id"]), q["question"])
                         for q in questions]
            if hooks:
                hooks.remove()
            print(f"gate {mode}: " + " | ".join(f"{a[:60]!r} ({n}tok)" for a, n in res[mode]), flush=True)
        ident = res["none"] == res["ones"]
        changed = res["none"] != res["masks"]
        sane = all(n < args.max_gen and a for a, n in res["none"])
        _, prompt = encode(model, processor, frames_of(questions[0]["segment_id"]), questions[0]["question"])
        (out_dir / "gate.json").write_text(json.dumps({
            "ones_equal_none": ident, "masks_change_output": changed, "dense_answers_complete": sane, "answers": res,
            "questions": [q["question"] for q in questions], "prompt_example": prompt}, indent=2))
        print(f"gate ones==none: {ident} | masks change: {changed} | dense answers complete: {sane}", flush=True)
        print("prompt example:", prompt.replace("<|image_pad|>", "")[:600], flush=True)
        sys.exit(0 if ident and changed and sane else 1)

    hooks = eel.EdgeVlmMasks(model, *masks) if masks is not None else None
    rows_path = out_dir / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["question_id"] + "|" + r["segment_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": ec.MODEL_ID, "model_revision": ec.MODEL_REV, "protocol": "standard LingoQA VQA (zero-shot)",
        "message_builder": "alpamayo2_edge.chat_template.build_vqa_conversation, front camera only, no closed think block",
        "time_labels_us": TIMESTAMPS_US.tolist(), "pixel_bounds": FRONT_BOUNDS, "n_questions": len(questions),
        "max_gen": args.max_gen, "decoding": "greedy (text_tasks.generate_vqa_answer settings)",
        "frames_per_segment": ll.N_FRAMES, "camera_index": ll.FRONT_CAMERA, "masks": args.masks,
        "mask_placement": "forward pre-hooks: o_proj input per Q head, mlp.fc2 input per MLP channel",
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": gpu, "plan": args.plan}, indent=2))
    t_start = time.time()
    for i, q in enumerate(questions):
        if q["question_id"] + "|" + q["segment_id"] in done:
            continue
        t0 = time.time()
        text, n_gen = answer(model, processor, gen_cfg, lp, frames_of(q["segment_id"]), q["question"])
        rows.append({"question_id": q["question_id"], "segment_id": q["segment_id"], "question": q["question"],
                     "answer": text, "references": q["references"], "gen_len": n_gen, "truncated": n_gen >= args.max_gen})
        if len(rows) % 25 == 0 or i + 1 == len(questions):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(questions)}] {n_gen:3d}tok ({time.time() - t0:.1f}s) {text[:80]!r}", flush=True)
    rows_path.write_text(json.dumps(rows, indent=2))
    if hooks:
        hooks.remove()
    pd.DataFrame([{"question_id": r["question_id"], "segment_id": r["segment_id"], "answer": r["answer"]}
                  for r in rows]).to_csv(out_dir / "predictions.csv", index=False)
    trunc = sum(r["truncated"] for r in rows)
    print(f"done: {len(rows)} answers, {trunc} truncated at {args.max_gen} tokens, {time.time() - t_start:.0f}s on {gpu} "
          f"-> {out_dir / 'predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
