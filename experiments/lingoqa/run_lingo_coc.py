"""Stage 1 of the LingoQA reasoning probe: generate one CoC per segment, per arm.

Alpamayo runs unmodified here -- stock CoC prompt, no question injected. The question
is only applied in Stage 2, by a frozen reader that sees this text and nothing else.

Cost is small because the CoC depends on the segment, not the question: 100 segments
per arm, not 500 questions. Decoding is greedy, so an arm's text is a deterministic
function of the segment and the paired comparison against another arm is exact.

Stage 1 and Stage 2 are separate scripts so the reader or its prompt can be revised
without re-running Alpamayo, the same split every run_*/analyze_* pair here uses.

Keep every arm on one GPU architecture -- Ada cards 4-7 on this box. 3-4% of clips
generate different CoC text across architectures, which would swamp the effect being
measured.

Usage:
  python experiments/lingoqa/run_lingo_coc.py --arm baseline --gpu 5 --limit 2
  python experiments/lingoqa/run_lingo_coc.py --arm outputs/slim_dual_u40_v2 --gpu 6
"""

import os

# must precede any CUDA context creation for deterministic cuBLAS reductions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time
from pathlib import Path

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
import eval_lib as el
import lingo_lib as ll
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
# isort: on

# pinned so a repo update cannot silently change the weights under a run; the same
# revision every result in the evaluation track was produced with
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline", help="'baseline' or a slim ckpt dir")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N segments")
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    ap.add_argument("--gpu", default="4,5,6,7",
                    help="comma-separated cards to try, in order. Defaults to the Ada "
                         "cards: every arm of a comparison must stay on one architecture")
    ap.add_argument("--strict-deterministic", action="store_true")
    args = ap.parse_args()

    tag = "baseline" if args.arm == "baseline" else Path(args.arm).name
    exp_id = args.exp_id or f"lingo_coc_{tag}"
    out_dir = REPO / "outputs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "coc.json"

    questions, segments = ll.load_manifest()
    if args.limit:
        segments = segments[: args.limit]

    torch.use_deterministic_algorithms(True, warn_only=not args.strict_deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = ll.pick_gpu(args.reserve_gb, [int(g) for g in args.gpu.split(",")])
    gpu_name = torch.cuda.get_device_name(device)
    print(f"{exp_id} | {len(segments)} segments, {len(questions)} questions | {gpu_name}",
          flush=True)

    if args.arm == "baseline":
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        import slim_lib as sl
        model = sl.load_slim(REPO / args.arm, device="cuda")

    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["segment_id"] for r in rows}

    (out_dir / "config.json").write_text(json.dumps({
        "arm": args.arm, "tag": tag, "n_segments": len(segments),
        "n_questions": len(questions), "max_gen": args.max_gen,
        "decoding": "greedy", "gpu": gpu_name, "model_revision": MODEL_REV,
        "frames_per_segment": ll.N_FRAMES, "camera_index": ll.FRONT_CAMERA,
        "ego_history": {"synthetic": True, "step_m": ll.HIST_STEP_M, "n_poses": ll.N_HIST},
        "deterministic": {"use_deterministic_algorithms": True,
                          "warn_only": not args.strict_deterministic,
                          "cudnn_deterministic": True, "tf32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    }, indent=2))

    t_start = time.time()
    for i, seg in enumerate(segments):
        if seg in done:
            continue
        t0 = time.time()
        data = ll.load_segment(seg, device="cuda")
        inputs = lib.build_inputs(model, processor, data, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            text, n_tok = ll.run_coc_greedy(model, inputs, max_generation_length=args.max_gen)
        rows.append({"segment_id": seg, "gen_coc": text, "gen_len": n_tok,
                     "prompt_len": int(inputs["input_ids"].shape[1]),
                     **{f"coc_{k}": v for k, v in el.coc_degenerate(text).items()}})
        if len(rows) % 10 == 0 or i + 1 == len(segments):
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(segments)}] {seg[:8]} {n_tok:4d}tok "
              f"({time.time() - t0:.0f}s) {text[:90]!r}", flush=True)

    rows_path.write_text(json.dumps(rows, indent=2))
    degen = sum(r["coc_degenerate"] for r in rows) / max(len(rows), 1)
    print(f"done: {len(rows)} segments, degenerate {degen:.3f}, "
          f"{time.time() - t_start:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
