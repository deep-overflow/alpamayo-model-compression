"""Tyr-the-Pruner algorithm 2 (sparsity-distribution search), following the official repo.

cvpr27/plans/2026-10-31_tyr-a100.md. Official code: Tyr-the-Pruner/search_sparsity_dist.py and
src/metric_utils.py (compute_sparse_kl_div), settings from script/example.sh.

What follows the official code exactly:
  - (1+lambda) evolution from the supernet centres (all levels 0); per generation 128 offspring;
    num_flips = min(randint(1,4), randint(1,4)); each flip lowers one module whose level-1 exists
    (drawn uniformly over all modules) and raises one module OF THE SAME TYPE whose level+1 exists
    (the parity rule), so heads and channels are each conserved; duplicates in the offspring list
    are skipped (the parent itself is not excluded)
  - staged selection with survivors 16 / 4 / 2, a fresh random minibatch per stage, the parent
    re-added before the last stage, the best final survivor becomes the parent; 50 generations
  - fitness: dense-teacher top-8192 sparse KL, log-softmax over the k-subset on both sides,
    KL(teacher || student), position-weighted mean over the minibatch (compute_sparse_kl_div)
Where it differs (decided by the user, 2026-10-31):
  - data is calib_100: each sequence is the clip's fused prompt + the dense model's own CoC
    (clip seed, max_gen 256), teacher-forced; KL is taken at the positions whose next token is
    text (prompt text or CoC, not vision / trajectory-history / sink) -- the analogue of "every
    text position but the last" on FineWeb
  - stage sizes: the official 2048 / 16384 / 131072 tokens are converted to whole clips with the
    mean teacher sequence length (a fused prompt cannot be cut mid-image)

Outputs (outputs/<exp-id>/): config.json, search_log.json, final_config.json ({module: level}),
summary.txt. Teacher cache: outputs/<teacher-id>/<clip>.npz.

  .venv/bin/python cvpr27/code/tyr_off/search.py --teacher-only --teacher-id cvpr27/tyr_off/teacher
  .venv/bin/python cvpr27/code/tyr_off/search.py --supernet cvpr27/tyr_off/supernet_r1 \
      --exp-id cvpr27/tyr_off/search_r1 --workers 4 --gpu 0,1,2,3
  # smoke: --generations 2 --offspring 4 --num-clips 4
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import analysis_lib as lib
import sample_cache as sc
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu
from run_tyr_search import tf_forward
from slim_lib import MODEL_REV

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model():
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    return model


@torch.no_grad()
def build_teacher(model, processor, calib, args, teacher_dir):
    """Dense rollout of each clip's CoC, then top-k logits at the text-target positions."""
    teacher_dir.mkdir(parents=True, exist_ok=True)
    for ci, (clip_id, t0) in enumerate(calib):
        path = teacher_dir / f"{clip_id}.npz"
        if path.exists():
            continue
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        text = lib.compute_spans(model, inputs["input_ids"])["text"]       # (T_prompt,)
        seed = sc.clip_seed(args.seed, clip_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_end = roll["eos_pos"] + 1
        seq = roll["sequences"][:, :coc_end]                               # (1, T)
        del roll
        # next-token targets j in [1, T): prompt text or CoC (prompt_len .. coc_end - 1)
        tgt = torch.zeros(coc_end, dtype=torch.bool)
        tgt[:prompt_len] = text
        tgt[prompt_len:] = True
        tgt[0] = False
        pred_pos = (torch.nonzero(tgt).flatten() - 1).numpy()              # logits positions
        item = {"pixel_values": inputs["tokenized_data"]["pixel_values"].cpu(),
                "image_grid_thw": inputs["tokenized_data"]["image_grid_thw"].cpu()}
        hidden, _, _ = tf_forward(model, seq, item, use_cache=False)
        topv, topi = [], []
        for chunk in np.array_split(pred_pos, max(1, len(pred_pos) // 256)):
            logits = model.vlm.lm_head(hidden[0, torch.from_numpy(chunk).cuda()]).float()
            v, ix = logits.topk(args.topk, dim=-1)
            topv.append(v.cpu().numpy().astype(np.float16))
            topi.append(ix.cpu().numpy().astype(np.int32))
        np.savez(path, seq=seq[0].cpu().numpy().astype(np.int32), prompt_len=prompt_len,
                 coc_end=coc_end, pred_pos=pred_pos.astype(np.int32),
                 topv=np.concatenate(topv), topi=np.concatenate(topi))
        del hidden
        print(f"[teacher {ci + 1}/{len(calib)}] {clip_id} len={coc_end} "
              f"kl_positions={len(pred_pos)} coc={coc_end - prompt_len}", flush=True)


class Supernet:
    """Level files + current model state (search_sparsity_dist.load_layers semantics)."""

    def __init__(self, model, sup_dir):
        meta = json.loads((sup_dir / "metadata.json").read_text())
        self.dir = sup_dir
        self.names = meta["layer_names"]                  # per layer: mlp.down_proj, self_attn.o_proj
        self.levels = {n: {int(k) for k in meta["levels"][n]} for n in self.names}
        self.keep = {n: {int(k): v["keep"] for k, v in meta["levels"][n].items()} for n in self.names}
        self.modules, self.state = {}, {n: None for n in self.names}
        if model is not None:
            layers = model.vlm.model.language_model.layers
            for n in self.names:
                i = int(n.split(".")[1])
                self.modules[n] = layers[i].mlp.down_proj if "mlp" in n else layers[i].self_attn.o_proj

    def load(self, cand):
        for n, lv in zip(self.names, cand):
            if self.state[n] != lv:
                self.modules[n].weight.data.copy_(torch.load(self.dir / n / f"{lv}.pth",
                                                             map_location="cuda"))
                self.state[n] = lv


@torch.no_grad()
def kl_sum(model, sup, cand, items, teacher):
    """(sum of per-position KL, number of positions) over the clips -- compute_sparse_kl_div."""
    sup.load(cand)
    tot, n = 0.0, 0
    for item in items:
        t = teacher[item["clip_id"]]
        seq = torch.from_numpy(t["seq"]).long().unsqueeze(0).cuda()
        hidden, _, _ = tf_forward(model, seq, item, use_cache=False)
        pos = torch.from_numpy(t["pred_pos"]).long().cuda()
        topi = torch.from_numpy(t["topi"]).long().cuda()
        topv = torch.from_numpy(t["topv"]).float().cuda()
        for sl in torch.split(torch.arange(len(pos), device="cuda"), 256):
            logits = model.vlm.lm_head(hidden[0, pos[sl]]).float().gather(-1, topi[sl])
            tot += float(F.kl_div(logits.log_softmax(-1), topv[sl].log_softmax(-1),
                                  log_target=True, reduction="sum"))
        n += len(pos)
        del hidden
    return tot, n


def mutate(parent, sup):
    """search_sparsity_dist.py:253-286 (num_flips, decrement anywhere, increment same type)."""
    offspring = list(parent)
    num_flips = min(random.randint(1, 4), random.randint(1, 4))
    for _ in range(num_flips):
        while True:
            decr_id = random.randint(0, len(offspring) - 1)
            if offspring[decr_id] - 1 in sup.levels[sup.names[decr_id]]:
                break
        while True:
            incr_id = random.randint(0, len(offspring) - 1)
            if (decr_id + incr_id) % 2 != 0:
                continue
            if offspring[incr_id] + 1 in sup.levels[sup.names[incr_id]]:
                break
        offspring[decr_id] -= 1
        offspring[incr_id] += 1
    return offspring


def load_worker_state(args, gpu):
    reserve_gpu(args.reserve_gb, devices=[gpu])
    model = load_model()
    processor = helper.get_processor(model.tokenizer)
    sup = Supernet(model, Path(args.supernet_path))
    teacher_dir = REPO / "outputs" / args.teacher_id
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    items, teacher = [], {}
    for clip_id, t0 in calib:
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0))
        inp = lib.build_inputs(model, processor, data, "cuda")
        items.append({"clip_id": clip_id,
                      "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
                      "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu()})
        teacher[clip_id] = dict(np.load(teacher_dir / f"{clip_id}.npz"))
    return model, sup, items, teacher


def worker_main(gpu, args, task_q, res_q):
    try:
        model, sup, items, teacher = load_worker_state(args, gpu)
        res_q.put(("ready", gpu, None, None))
        while True:
            task = task_q.get()
            if task is None:
                break
            idx, cand, clip_idx = task
            s, n = kl_sum(model, sup, cand, [items[i] for i in clip_idx], teacher)
            res_q.put(("res", idx, s, n))
    except Exception as e:
        res_q.put(("err", gpu, repr(e), None))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supernet", default=None, help="outputs-relative supernet dir")
    ap.add_argument("--supernet-path", default=None,
                    help="absolute copy of the supernet to read level files from (e.g. node-local /tmp)")
    ap.add_argument("--teacher-id", default="cvpr27/tyr_off/teacher")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--topk", type=int, default=8192)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--offspring", type=int, default=128)
    ap.add_argument("--tokens-per-selection", type=int, nargs="+", default=[2048, 16384, 131072])
    ap.add_argument("--survivors-per-selection", type=int, nargs="+", default=[16, 4, 2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    teacher_dir = REPO / "outputs" / args.teacher_id

    if args.teacher_only:
        reserve_gpu(args.reserve_gb, devices=[int(args.gpu.split(",")[0])])
        model = load_model()
        build_teacher(model, helper.get_processor(model.tokenizer), calib, args, teacher_dir)
        print("teacher cache complete ->", teacher_dir, flush=True)
        return

    assert args.supernet and args.exp_id
    args.supernet_path = args.supernet_path or str(REPO / "outputs" / args.supernet)
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c, _ in calib if not (teacher_dir / f"{c}.npz").exists()]
    assert not missing, f"teacher cache incomplete ({len(missing)}), run --teacher-only"
    seq_len = [int(np.load(teacher_dir / f"{c}.npz")["seq"].shape[0]) for c, _ in calib]
    mean_len = float(np.mean(seq_len))
    stage_clips = [max(1, round(t / mean_len)) for t in args.tokens_per_selection]
    sup = Supernet(None, Path(args.supernet_path))

    gpus = [int(x) for x in args.gpu.split(",")][: args.workers]
    assert len(gpus) == args.workers, "need one --gpu id per worker"
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(g, args, task_q, res_q), daemon=True) for g in gpus]
    for pr in procs:
        pr.start()
    for _ in procs:
        msg = res_q.get()
        assert msg[0] == "ready", msg

    def fitness(cands, clip_idx):
        for i, c in enumerate(cands):
            task_q.put((i, c, clip_idx))
        out = [None] * len(cands)
        for _ in cands:
            msg = res_q.get()
            assert msg[0] == "res", msg
            out[msg[1]] = msg[2] / max(msg[3], 1)
        return out

    parent = [0] * len(sup.names)
    all_idx = list(range(len(calib)))
    fit0 = fitness([parent], all_idx)[0]
    print(f"R centre: KL={fit0:.6f} over all {len(calib)} clips; stage clips {stage_clips} "
          f"(mean seq len {mean_len:.0f})", flush=True)
    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "model_revision": MODEL_REV, "clip_ids": [c for c, _ in calib],
        "mean_seq_len": mean_len, "stage_clips": stage_clips, "kl_centre_all_clips": fit0,
        "fitness": "top-k sparse KL(teacher || student) at text-target positions, position-weighted",
    }, indent=2))

    log = []
    for gen in range(args.generations):
        t0 = time.time()
        offspring = []
        while len(offspring) < args.offspring:
            child = mutate(parent, sup)
            if child in offspring:
                continue
            offspring.append(child)
        for si, (surv, ncl) in enumerate(zip(args.survivors_per_selection, stage_clips)):
            if si == len(stage_clips) - 1 and parent not in offspring:
                offspring.append(parent)  # elitist
            clip_idx = random.sample(all_idx, min(ncl, len(all_idx)))
            fits = fitness(offspring, clip_idx)
            order = sorted(range(len(offspring)), key=lambda k: fits[k])[:surv]
            offspring, fits = [offspring[k] for k in order], [fits[k] for k in order]
        parent, fit_p = offspring[0], fits[0]
        log.append({"gen": gen, "fitness": fit_p, "sec": round(time.time() - t0, 1), "levels": parent})
        (out_dir / "search_log.json").write_text(json.dumps(log))
        (out_dir / "final_config.json").write_text(
            json.dumps(dict(zip(sup.names, parent)), indent=1))
        print(f"[gen {gen + 1}/{args.generations}] fit={fit_p:.6f} {time.time() - t0:.0f}s", flush=True)

    for _ in procs:
        task_q.put(None)
    for pr in procs:
        pr.join(timeout=60)
    # G1: the per-type totals cannot move (type-conserving flips); check it on the kept counts
    keep_q = sum(sup.keep[n][lv] for n, lv in zip(sup.names, parent) if "self_attn" in n)
    keep_m = sum(sup.keep[n][lv] for n, lv in zip(sup.names, parent) if "mlp" in n)
    keep_q0 = sum(sup.keep[n][0] for n in sup.names if "self_attn" in n)
    keep_m0 = sum(sup.keep[n][0] for n in sup.names if "mlp" in n)
    assert (keep_q, keep_m) == (keep_q0, keep_m0), ((keep_q, keep_m), (keep_q0, keep_m0))
    (out_dir / "summary.txt").write_text(
        f"tyr_off search ({args.workers} workers): {args.generations} gens x {args.offspring} offspring, "
        f"stage clips {stage_clips}, survivors {args.survivors_per_selection}\n"
        f"KL at centre {fit0:.6f} (all clips); final parent fitness {fit_p:.6f} (last minibatch)\n"
        f"kept {keep_q} heads / {keep_m} channels (= centre)\n"
        f"levels mha {[lv for n, lv in zip(sup.names, parent) if 'self_attn' in n]}\n"
        f"levels mlp {[lv for n, lv in zip(sup.names, parent) if 'mlp' in n]}\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
