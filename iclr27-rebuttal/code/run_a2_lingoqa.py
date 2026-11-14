"""LingoQA (standard VQA protocol) for Alpamayo 2 Super, dense or with VLM unit masks (MoOP / FM-only / CE-only).

Plan: iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md. The A2 twin of experiments/lingoqa/run_lingo_vqa.py (A1.5, the paper's
Table 4): the model answers each of the 500 LingoQA val questions itself, zero-shot, from the last 4 of the segment's 5 front-camera
frames, greedy, max 128 new tokens; predictions.csv is scored by experiments/lingoqa/score_lingo_vqa.py (Lingo-Judge, LingoQA's
own metric). Alpamayo 2 Super's native VQA task builds the prompt (alpamayo2_super.text_tasks.build_text_task_messages, task
"vqa": camera name + frame numbers + images + the question, no trajectory tokens); the camera is index 1 = "Front camera", as the
A1.5 port maps LingoQA's frames. Masks are applied as in run_a2_openloop.py (forward pre-hooks on o_proj / down_proj).

--gate N answers the first N questions with no hooks and with all-ones hooks (must be identical) and with the masks.

--split train (2026-11-14, the confirmatory replication in plans/2026-11-13_a2-lingoqa.md): a fixed random sample of LingoQA's
train split instead of val -- --n-seg segments (seeded by --sample-seed) x --per-seg questions each; one human reference per question.
A2's importance never saw LingoQA, so for the pruning comparison this is fresh data; whether A2 itself was trained on LingoQA train
is unknown, which can lift every arm's absolute accuracy but not the paired arm-to-arm comparison. The sample is written to
manifest.json; each row carries its references, which score_a2_lingo_train.py uses.

Usage (cards from the shell, model from the local cache):
  CUDA_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1 /mnt/nvme1n1/ad_vla/venvs/alpamayo2/bin/python \\
      iclr27-rebuttal/code/run_a2_lingoqa.py --exp-id iclr27-rebuttal/a2_lingo_dense
  ... --masks outputs/iclr27-rebuttal/a2_moop/masks_moop.npz --exp-id iclr27-rebuttal/a2_lingo_moop
"""

import os
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise SystemExit("set CUDA_VISIBLE_DEVICES to this replica's cards, e.g. 4,5")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
A2D = Path(os.environ.get("A2_DISTILL", "/home/cvlab21/project/chan/alpamayo2-distill"))
sys.path.insert(0, str(A2D / "src"))
sys.path.insert(0, str(A2D / "experiments" / "oodval"))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "lingoqa"))
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
import lingo_lib as ll
import run_oodval as ro
from alpamayo2_super import helper
from alpamayo2_super.models.alpamayo2_super import (
    Alpamayo2Super,
    MaskDiscreteTrajectoryLogitsProcessor,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.text_tasks import build_text_task_messages
from run_a2_openloop import VlmMasks
from transformers.generation.logits_process import LogitsProcessorList

TRAIN = Path(os.environ.get("LINGOQA_TRAIN", "/mnt/nvme1n1/ad_vla/data/lingoqa/train"))


def train_sample(n_seg, per_seg, seed, exclude=()):
    """n_seg random train segments x per_seg random questions each, sorted by segment (a fixed, seeded draw); segments in
    `exclude` are never drawn (a sample disjoint from an earlier one)."""
    df = pd.read_parquet(TRAIN / "train.parquet", columns=["question_id", "segment_id", "question", "answer"])
    g = df.groupby(["question_id", "segment_id", "question"])["answer"].apply(list).reset_index()
    rng = np.random.default_rng(seed)
    segs = sorted(set(g.segment_id.unique()) - set(exclude))
    pick = sorted(rng.choice(segs, size=n_seg, replace=False).tolist())
    qs = []
    for s in pick:
        sub = g[g.segment_id == s].sort_values("question_id")
        for r in sub.iloc[sorted(rng.choice(len(sub), size=min(per_seg, len(sub)), replace=False))].itertuples():
            qs.append({"question_id": r.question_id, "segment_id": r.segment_id, "question": r.question,
                       "references": list(r.answer)})
    return qs


def encode(processor, model, frames, cams, question):
    """frames (1, 4, 3, H, W) uint8, cams (1,) -> tokenized inputs on cuda:0 (prepare_text_generation_inputs, processor cached)"""
    data = {"image_frames": frames, "camera_indices": cams, "question": question}
    messages = build_text_task_messages(data, model.config, "vqa")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
    images = frames.flatten(0, 1).float() / 255.0  # (4, 3, H, W)
    td = dict(processor(text=text, images=images, videos=None, padding=False, return_tensors="pt", do_rescale=False))
    return {k: v.to("cuda:0") for k, v in td.items()}, text


def answer(model, processor, gc, lp, frames, cams, question):
    td, _ = encode(processor, model, frames, cams, question)
    plen = td["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.vlm.generate(**td, generation_config=gc, logits_processor=lp)
    gen = out.sequences[:, plen:]  # (1, n)
    text = extract_text_tokens(model.tokenizer, gen)["answer"][0]
    return text, int(gen.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="outputs/<exp-id> in this repo (or an absolute path)")
    ap.add_argument("--masks", default=None, help="npz with q_mask (64, 64), mlp_mask (64, 25600); none = dense")
    ap.add_argument("--max-gen", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first-share", type=float, default=0.47)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--plan", default="iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--n-seg", type=int, default=400, help="train: segments sampled")
    ap.add_argument("--per-seg", type=int, default=5, help="train: questions per sampled segment")
    ap.add_argument("--sample-seed", type=int, default=20261114)
    ap.add_argument("--exclude-manifest", default=None, help="train: never draw the segments of this earlier manifest.json")
    args = ap.parse_args()
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    n_gpus = torch.cuda.device_count()
    assert n_gpus == len(gpus.split(",")) >= 1, (n_gpus, gpus)

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "train":
        excl = ({q["segment_id"] for q in json.loads(Path(args.exclude_manifest).read_text())["questions"]}
                if args.exclude_manifest else set())
        questions = train_sample(args.n_seg, args.per_seg, args.sample_seed, excl)
        assert not excl & {q["segment_id"] for q in questions}
        (out_dir / "manifest.json").write_text(json.dumps(
            {"split": "train", "n_seg": args.n_seg, "per_seg": args.per_seg, "seed": args.sample_seed,
             "excluded_segments_from": args.exclude_manifest, "n_excluded": len(excl), "questions": questions}, indent=1))
    else:
        questions, _ = ll.load_manifest()
    if args.gate:
        questions = questions[: args.gate]
    elif args.limit:
        questions = questions[: args.limit]

    ro.set_determinism()
    if n_gpus == 1:
        device_map, per_gpu = {"": 0}, [ro.N_LAYERS]
    else:
        share = args.first_share if n_gpus == 2 else 1.0 / n_gpus - 0.03
        device_map, per_gpu = ro.layer_device_map(n_gpus, share)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"{args.exp_id} | {len(questions)} questions | cards {gpus} ({gpu_names[0]}) | masks {args.masks}", flush=True)
    model = Alpamayo2Super.from_pretrained(ro.MODEL_ID, revision=ro.MODEL_REV, dtype=torch.bfloat16, device_map=device_map)
    model.eval()
    processor = helper.get_processor(model.tokenizer, model.config)

    gc = copy.deepcopy(model.vlm.generation_config)
    gc.do_sample = False  # greedy, as the A1.5 LingoQA runs: the reported number must not carry sampling noise
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.num_return_sequences = 1
    gc.max_new_tokens = args.max_gen
    gc.output_logits = False
    gc.return_dict_in_generate = True
    gc.pad_token_id = model.tokenizer.pad_token_id
    lp = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
        traj_token_offset=min(model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"]),
        traj_vocab_size=model.config.traj_vocab_size)])
    masks = None
    if args.masks:
        z = np.load(args.masks)
        masks = (z["q_mask"], z["mlp_mask"])

    seg_cache = {}

    def frames_of(seg):
        if seg not in seg_cache:
            seg_cache.clear()  # questions are sorted by segment; keep one resident
            d = ll.load_segment(seg, data=TRAIN, split="train") if args.split == "train" else ll.load_segment(seg)
            seg_cache[seg] = (d["image_frames"], d["camera_indices"])  # (1, 4, 3, H, W) uint8, (1,) = [1]
        return seg_cache[seg]

    if args.gate:
        res = {}
        for mode in ("none", "ones", "masks"):
            hooks = None
            if mode == "ones":
                hooks = VlmMasks(model, np.ones_like(masks[0]), np.ones_like(masks[1]))
            elif mode == "masks":
                hooks = VlmMasks(model, *masks)
            res[mode] = [answer(model, processor, gc, lp, *frames_of(q["segment_id"]), q["question"]) for q in questions]
            if hooks:
                hooks.remove()
            print(f"gate {mode}: " + " | ".join(f"{a[:60]!r} ({n}tok)" for a, n in res[mode]), flush=True)
        ident = res["none"] == res["ones"]
        changed = res["none"] != res["masks"]
        sane = all(n < args.max_gen and a for a, n in res["none"])
        _, prompt = encode(processor, model, *frames_of(questions[0]["segment_id"]), questions[0]["question"])
        (out_dir / "gate.json").write_text(json.dumps({
            "ones_equal_none": ident, "masks_change_output": changed, "dense_answers_complete": sane,
            "answers": res, "prompt_example": prompt}, indent=2))
        print(f"gate ones==none: {ident} | masks change: {changed} | dense answers complete: {sane}", flush=True)
        sys.exit(0 if ident and changed and sane else 1)

    hooks = VlmMasks(model, *masks) if masks is not None else None
    rows_path = out_dir / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["question_id"] + "|" + r["segment_id"] for r in rows}
    (out_dir / "config.json").write_text(json.dumps({
        "model": ro.MODEL_ID, "model_revision": ro.MODEL_REV, "protocol": "standard LingoQA VQA (zero-shot)",
        "message_builder": "alpamayo2_super.text_tasks.build_text_task_messages(task='vqa')", "n_questions": len(questions),
        "split": args.split, "sample": ({"n_seg": args.n_seg, "per_seg": args.per_seg, "seed": args.sample_seed}
                                        if args.split == "train" else None),
        "max_gen": args.max_gen, "decoding": "greedy", "frames_per_segment": ll.N_FRAMES, "camera_index": ll.FRONT_CAMERA,
        "masks": args.masks, "mask_placement": "forward pre-hooks: o_proj input per Q head, down_proj input per MLP channel",
        "gpus": gpus, "gpu_names": gpu_names, "layers_per_gpu": per_gpu, "plan": args.plan, "a2_distill": str(A2D),
        "deterministic": {"use_deterministic_algorithms": True, "warn_only": True, "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}}, indent=2))
    t_start = time.time()
    for i, q in enumerate(questions):
        if q["question_id"] + "|" + q["segment_id"] in done:
            continue
        t0 = time.time()
        text, n_gen = answer(model, processor, gc, lp, *frames_of(q["segment_id"]), q["question"])
        rows.append({"question_id": q["question_id"], "segment_id": q["segment_id"], "question": q["question"],
                     "answer": text, "references": q["references"], "gen_len": n_gen, "truncated": n_gen >= args.max_gen})
        if len(rows) % 25 == 0 or i + 1 == len(questions):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(questions)}] {n_gen:3d}tok ({time.time() - t0:.1f}s) {text[:80]!r}", flush=True)
    rows_path.write_text(json.dumps(rows, indent=2))
    if hooks:
        hooks.remove()
    # the three columns LingoQA's evaluate.py (and score_lingo_vqa.py) read; 500 rows expected
    pd.DataFrame([{"question_id": r["question_id"], "segment_id": r["segment_id"], "answer": r["answer"]}
                  for r in rows]).to_csv(out_dir / "predictions.csv", index=False)
    trunc = sum(r["truncated"] for r in rows)
    print(f"done: {len(rows)} answers, {trunc} truncated at {args.max_gen} tokens, {time.time() - t_start:.0f}s "
          f"-> {out_dir / 'predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
