"""Shared library for the LingoQA reasoning probe.

The probe asks a question the existing metrics cannot: is the CoC text still
*informative*? `nll_gtcoc` scores it against a curated reference and
`coc_degenerate()` catches token soup, but a CoC can pass both while having
stopped tracking the scene.

Design is CoC-as-evidence. Alpamayo runs completely unmodified -- the stock CoC
prompt from `helper.create_message`, no question injected -- and a frozen
text-only reader answers the LingoQA question from the generated CoC alone.
The reader never sees the frames, so the score measures how much scene
information reached the text. A reader with image access would answer around a
damaged CoC and hide exactly the effect we are looking for.

LingoQA is out of domain (Wayve UK footage, front camera only, no ego history).
That is tolerable because every confound here -- the synthetic ego history, the
domain gap, the frame choice, the reader's own errors -- is identical across
arms and cancels in the paired delta. Only the paired delta is claimed; the
absolute number is not a LingoQA leaderboard score and must not be reported as
one.

Data layout under DATA: `val.parquet` (1000 QA pairs = 500 unique
(question_id, segment_id) with 2 references each, over 100 segments x 5
questions) and `images/val/<segment_id>/{0..4}.jpg` at 2048x1280.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# LINGOQA_DATA so the same code runs off-box (NEURON puts it under /scratch); the
# default is this box's copy, so nothing here changes.
DATA = Path(os.environ.get("LINGOQA_DATA", "/mnt/nvme1n1/ad_vla/data/lingoqa"))

# LingoQA ships a forward-facing camera; camera index 1 is "Front camera" in
# helper.CAMERA_DISPLAY_NAMES, which is the name the model saw in training.
FRONT_CAMERA = 1
N_FRAMES = 4  # helper.create_message's num_frames_per_camera; LingoQA gives 5, we take the last 4

# Median per-step distance and identity rotation over 80 cached eval clips: the ego
# history of a typical straight, constant-speed clip. LingoQA has no ego trajectory,
# so one fixed history stands in for every segment. It is identical across arms and
# segments, so it cannot produce a paired difference -- it only shifts the common mode.
HIST_STEP_M = 0.7156
N_HIST = 16  # 16 poses -> the 48 traj tokens create_message reserves


def pick_gpu(need_gb, devices):
    """Reserve `need_gb` on the first of `devices` that has room.

    Wraps `expert_per_clip.reserve_gpu` one card at a time because that function
    only catches `torch.OutOfMemoryError`: on a card with no free memory at all the
    driver raises `torch.AcceleratorError` instead, which aborts its whole scan
    rather than skipping that card. On this shared box cards 0-3 are routinely at
    100%, so an unpinned scan dies before it ever reaches the free Ada cards.
    """
    import torch as _torch
    from expert_per_clip import reserve_gpu

    last = None
    for idx in devices:
        try:
            return reserve_gpu(need_gb, devices=[idx])
        except (_torch.OutOfMemoryError, _torch.AcceleratorError) as exc:
            last = exc
            print(f"  cuda:{idx}: unavailable ({type(exc).__name__})", flush=True)
            _torch.cuda.empty_cache()
    raise _torch.OutOfMemoryError(f"no GPU in {devices} with {need_gb} GiB free") from last


def synthetic_ego_history(device=None):
    """One fixed straight-line, constant-speed history ending at the current pose.

    Matches the cached loader's shapes: xyz (1, 1, 16, 3), rot (1, 1, 16, 3, 3),
    oldest pose first, last pose at the origin, rotations identity.
    """
    xyz = np.zeros((N_HIST, 3), dtype=np.float32)
    xyz[:, 0] = (np.arange(N_HIST, dtype=np.float32) - (N_HIST - 1)) * HIST_STEP_M
    rot = np.tile(np.eye(3, dtype=np.float32), (N_HIST, 1, 1))
    out = {
        "ego_history_xyz": torch.from_numpy(xyz)[None, None],        # (1, 1, 16, 3)
        "ego_history_rot": torch.from_numpy(rot)[None, None],        # (1, 1, 16, 3, 3)
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def load_manifest(data=DATA):
    """The 500 evaluation questions, grouped as LingoQA's own evaluate.py groups them.

    Returns (questions, segments):
      questions: list of dicts with question_id, segment_id, question, references (2)
      segments:  ordered list of the 100 unique segment_ids

    The 1000 rows collapse to 500 unique (question_id, segment_id) carrying 2 human
    references each; `LingoJudge.compute` takes the max over them, so a prediction
    matching either one counts.
    """
    df = pd.read_parquet(data / "val.parquet")
    g = (df.groupby(["question_id", "segment_id", "question"])["answer"]
           .apply(list).reset_index())
    questions = [{"question_id": r.question_id, "segment_id": r.segment_id,
                  "question": r.question, "references": list(r.answer)}
                 for r in g.itertuples()]
    questions.sort(key=lambda q: (q["segment_id"], q["question_id"]))
    segments = sorted({q["segment_id"] for q in questions})
    return questions, segments


def load_segment(segment_id, data=DATA, device=None, split="val"):
    """Frames for one segment in the loader's return format.

    Returns image_frames (1, 4, 3, H, W) uint8 for the single front camera plus the
    synthetic ego history, i.e. the four keys `analysis_lib.build_inputs` reads.
    The last 4 of LingoQA's 5 frames are used -- the ones nearest the present, matching
    how the cached clips end at t0.

    `split` selects the image tree: "val" is the held-out benchmark, "train" the
    calibration source for importance runs. The two live under different `data` roots.
    """
    d = data / "images" / split / segment_id
    frames = []
    for i in range(5 - N_FRAMES, 5):
        img = np.asarray(Image.open(d / f"{i}.jpg").convert("RGB"))   # (H, W, 3)
        frames.append(np.transpose(img, (2, 0, 1)))                   # (3, H, W)
    out = {
        "image_frames": torch.from_numpy(np.stack(frames))[None],     # (1, 4, 3, H, W)
        "camera_indices": torch.tensor([FRONT_CAMERA], dtype=torch.int64),
        **synthetic_ego_history(),
    }
    if device is not None:
        for k in ("image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot"):
            out[k] = out[k].to(device)
    return out


def run_coc_greedy(model, inputs, max_generation_length=256):
    """Greedy CoC rollout. Returns (text, n_tokens).

    A greedy twin of `analysis_lib.run_rollout`, which samples at temperature 0.6.
    Sampling noise would have to be averaged away with repeated rollouts; greedy makes
    each arm's text a deterministic function of the segment, so the paired comparison
    is exact. Kept here rather than as a flag on `run_rollout` because that function
    mutates `model.vlm.generation_config` in place and every existing result in the
    repo was produced through it.
    """
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    gc = model.vlm.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.num_return_sequences = 1
    gc.max_new_tokens = max_generation_length
    gc.output_logits = False
    gc.return_dict_in_generate = True
    gc.pad_token_id = model.tokenizer.pad_token_id

    from alpamayo1_5.models.token_utils import to_special_token
    eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    out = model.vlm.generate(
        input_ids=inputs["input_ids"],
        generation_config=gc,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_id)]),
        logits_processor=LogitsProcessorList([ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size)]),
        **inputs["tokenized_data"],
    )
    sequences = replace_padding_after_eos(token_ids=out.sequences, eos_token_id=eos_id,
                                          pad_token_id=model.tokenizer.pad_token_id)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = sequences[0, prompt_len:].tolist()
    cot_end = model.tokenizer.convert_tokens_to_ids(to_special_token("cot_end"))
    keep = [t for t in gen_ids if t not in (eos_id, cot_end, model.tokenizer.pad_token_id)]
    return model.tokenizer.decode(keep, skip_special_tokens=True), len(keep)


def write_outputs(out_dir, config, metrics, summary):
    """outputs/<exp_id>/ with the repo's config/metrics/summary convention."""
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "summary.txt").write_text(summary)
