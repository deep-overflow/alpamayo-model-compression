"""Tyr baseline, algorithm 2: evolutionary sparsity-distribution search.

Two stages, both label-free (the teacher is the dense model itself):

  teacher   For each calib clip, the DENSE model rolls out its CoC (clip seed),
            and we store: the teacher-forced sequence, top-k logits at the CoC
            positions, and the expert vector field at fixed (x_t, t) points
            (x_t anchored on the GT trajectory, target is the teacher field).
  search    Candidates are per-layer level vectors over the supernet. Mutation
            moves one level quantum between two layers OF THE SAME MODULE TYPE
            (the upstream parity trick, made explicit), so the per-type budget
            -- and hence the removed-parameter total -- is conserved exactly.
            Fitness is the dual-teacher distance, each term normalized by the
            uniform-init value:  fit = KL_coc / KL0 + MSE_vf / MSE0
            with KL_coc the sparse top-k KL at CoC positions (upstream
            compute_sparse_kl_div restricted to CoC) and MSE_vf the expert
            vector-field MSE against the dense field. Staged selection on
            growing clip minibatches with an elitist parent.

Gates (plans/2026-08-20_tyr-baseline.md): T0 budget conservation, T1/T2 judged
after the slim builds; the search log records per-generation fitness so the
monotone-improvement check in T0 is auditable.

Usage:
  bash experiments/head_analysis/run_retry_host.sh 720 \
      experiments/head_analysis/run_tyr_search.py --gpu 4,5,6,7
  # teacher cache only: --teacher-only ; smoke: --generations 2 --offspring 4 \
  #   --stage-clips 2 --survivors 1 --num-clips 4
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib
import mask_lib as ml
import prune_lib as pl
import sample_cache as sc
import tyr_lib as tyr
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch
from slim_lib import MODEL_REV

REPO = Path(__file__).resolve().parents[2]


def expert_field(model, cache, rope_deltas, prefill_len, x_t, t_val):
    """One expert forward: vector field at (x_t, t) on the given VLM cache."""
    device = x_t.device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    offset = torch.tensor([prefill_len], device=device)
    prefix_mask = torch.ones(1, prefill_len, device=device, dtype=torch.long)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill_len,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    fk = {"is_causal": False} if model.config.expert_non_causal_attention else {}
    t = torch.full((1, 1, 1), t_val, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        emb = model.action_in_proj(x_t.to(torch.bfloat16), t)
        if emb.dim() == 2:
            emb = emb.view(1, n_tok, -1)
        out = model.expert(
            inputs_embeds=emb, position_ids=position_ids, past_key_values=cache,
            attention_mask=attention_mask, use_cache=True, **fk,
        )
        cache.crop(prefill_len)
        pred = model.action_out_proj(out.last_hidden_state[:, -n_tok:])
    return pred.float().view(1, n_tok, -1)


def tf_forward(model, seq, item, use_cache):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return pl.vlm_forward_with_grad(
            model, seq,
            {"pixel_values": item["pixel_values"].cuda(),
             "image_grid_thw": item["image_grid_thw"].cuda()},
            use_cache=use_cache,
        )


def build_teacher(model, processor, calib, args, teacher_dir):
    teacher_dir.mkdir(parents=True, exist_ok=True)
    for ci, (clip_id, t0) in enumerate(calib):
        path = teacher_dir / f"{clip_id}.npz"
        if path.exists():
            continue
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0))
        inputs = lib.build_inputs(model, processor, data, "cuda")
        prompt_len = inputs["input_ids"].shape[1]
        seed = sc.clip_seed(args.seed, clip_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll = lib.run_rollout(model, inputs, max_generation_length=args.max_gen)
        coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end]
        del roll
        item = {"pixel_values": inputs["tokenized_data"]["pixel_values"].cpu(),
                "image_grid_thw": inputs["tokenized_data"]["image_grid_thw"].cpu()}
        hidden, cache, rope_deltas = tf_forward(model, seq_tf, item, use_cache=True)
        prefill_len = cache.get_seq_length()
        logits = model.vlm.lm_head(hidden[:, coc_start - 1: coc_end - 1]).float()
        topv, topi = logits.topk(args.topk, dim=-1)  # (1, Tc, k)

        x1 = lib.gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
        gen = torch.Generator().manual_seed(seed)
        xts, tvals, fields = [], [], []
        for _ in range(args.noise_draws):
            noise = torch.randn(x1.shape, generator=gen).to(x1.device)
            for t_val in args.t_grid:
                x_t = (1.0 - t_val) * noise + t_val * x1
                v = expert_field(model, cache, rope_deltas, prefill_len, x_t, t_val)
                xts.append(x_t.cpu().numpy()[0])
                tvals.append(t_val)
                fields.append(v.cpu().numpy()[0])
        np.savez(path,
                 seq=seq_tf[0].cpu().numpy().astype(np.int32),
                 coc_start=coc_start, coc_end=coc_end,
                 topv=topv[0].cpu().numpy().astype(np.float16),
                 topi=topi[0].cpu().numpy().astype(np.int32),
                 xt=np.stack(xts), tv=np.array(tvals), vd=np.stack(fields))
        del hidden, cache
        print(f"[teacher {ci + 1}/{len(calib)}] {clip_id} coc={coc_end - coc_start}",
              flush=True)


class MaskSupernet:
    """Mask-only supernet for the dual-global arm: a level vector maps to dual-rank
    masks (select_mask_ratios rule) applied through PruneMasks -- no weight files,
    kept weights stay original. Level 0 is dual_u40_v2 exactly."""

    def __init__(self, model, imp_path, head_cut=13, head_step=1, mlp_cut=4898,
                 mlp_step=256, num_levels=9):
        imp = dict(np.load(imp_path))
        self.sq, self.sm = tyr.dual_scores(imp)
        n_layers, n_heads = self.sq.shape
        inter = self.sm.shape[1]
        self.keeps_q = tyr.level_keeps(n_heads, head_cut, head_step, num_levels)
        self.keeps_m = tyr.level_keeps(inter, mlp_cut, mlp_step, num_levels)
        self.meta = {"head_cut": head_cut, "head_step": head_step, "mlp_cut": mlp_cut,
                     "mlp_step": mlp_step, "num_levels": num_levels}
        self.names, self.levels = [], {}
        for i in range(n_layers):
            for suf, ks in (("mlp.down_proj", self.keeps_m), ("self_attn.o_proj", self.keeps_q)):
                n = f"layers.{i:02d}.{suf}"
                self.names.append(n)
                self.levels[n] = sorted(ks)
        self.masks = None
        if model is not None:
            tc = model.vlm.config.text_config
            self.masks = ml.PruneMasks(model.vlm.model.language_model.layers,
                                       tc.num_attention_heads, tc.head_dim,
                                       tc.intermediate_size, "cuda")
        self.state = None

    def load(self, cand):
        if cand == self.state:
            return
        q = np.ones_like(self.sq)
        m = np.ones_like(self.sm)
        for n, lv in cand.items():
            i = int(n.split(".")[1])
            if "mlp" in n:
                m[i, tyr.cut_lowest(self.sm[i], self.sm.shape[1] - self.keeps_m[lv])] = 0
            else:
                q[i, tyr.cut_lowest(self.sq[i], self.sq.shape[1] - self.keeps_q[lv])] = 0
        self.masks.set(q=q, mlp=m)
        self.state = dict(cand)


def make_supernet(args, model):
    if args.selection == "dual":
        return MaskSupernet(model, REPO / "outputs" / args.importance / "importance.npz")
    sup_dir = REPO / "outputs" / args.supernet
    meta = json.loads((sup_dir / "metadata.json").read_text())
    return Supernet(model, sup_dir, meta)


class Supernet:
    """Weight store + current model state, upstream load_layers semantics."""

    def __init__(self, model, sup_dir, meta):
        self.dir = sup_dir
        self.names = meta["layer_names"]
        self.levels = {n: sorted(int(k) for k in
                                 (meta["levels_mlp"] if "mlp" in n else meta["levels_q"]))
                       for n in self.names}
        self.modules = {}
        if model is not None:
            layers = model.vlm.model.language_model.layers
            for n in self.names:
                i = int(n.split(".")[1])
                self.modules[n] = (layers[i].mlp.down_proj if "mlp" in n
                                   else layers[i].self_attn.o_proj)
        self.state = {n: None for n in self.names}

    def load(self, cand):
        for n, lv in cand.items():
            if self.state[n] != lv:
                w = torch.load(self.dir / n / f"{lv}.pth", map_location="cuda")
                self.modules[n].weight.data.copy_(w)
                self.state[n] = lv


def fitness_terms(model, sup, cand, clip_items, teacher):
    sup.load(cand)
    kls, mses = [], []
    for item in clip_items:
        t = teacher[item["clip_id"]]
        seq = torch.from_numpy(t["seq"]).long().unsqueeze(0).cuda()
        cs, ce = int(t["coc_start"]), int(t["coc_end"])
        hidden, cache, rope_deltas = tf_forward(model, seq, item, use_cache=True)
        prefill_len = cache.get_seq_length()
        logits = model.vlm.lm_head(hidden[:, cs - 1: ce - 1]).float()
        topi = torch.from_numpy(t["topi"]).long().unsqueeze(0).cuda()
        topv = torch.from_numpy(t["topv"]).float().unsqueeze(0).cuda()
        gathered = logits.gather(-1, topi)
        kl = F.kl_div(gathered.log_softmax(-1).flatten(0, 1),
                      topv.log_softmax(-1).flatten(0, 1),
                      log_target=True, reduction="batchmean")
        kls.append(float(kl))
        mse = 0.0
        for p in range(t["tv"].shape[0]):
            x_t = torch.from_numpy(t["xt"][p]).unsqueeze(0).cuda()
            v = expert_field(model, cache, rope_deltas, prefill_len, x_t,
                             float(t["tv"][p]))
            mse += float(F.mse_loss(v, torch.from_numpy(t["vd"][p]).unsqueeze(0).cuda()))
        mses.append(mse / t["tv"].shape[0])
        del hidden, cache
    return float(np.mean(kls)), float(np.mean(mses))


def mutate(parent, sup, names_by_type):
    child = dict(parent)
    flips = min(random.randint(1, 4), random.randint(1, 4))
    for _ in range(flips):
        typ = random.choice(("mha", "mlp"))
        names = names_by_type[typ]
        cands_d = [n for n in names if child[n] - 1 in sup.levels[n]]
        if not cands_d:
            continue
        nd = random.choice(cands_d)
        child[nd] -= 1
        cands_i = [n for n in names if child[n] + 1 in sup.levels[n]]
        if not cands_i:
            child[nd] += 1
            continue
        child[random.choice(cands_i)] += 1
    return child


def load_eval_state(args, devices):
    """One worker's world: model on its card, supernet handles, inputs, teacher."""
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"worker using {device}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    sup = make_supernet(args, model)
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
    torch.manual_seed(args.seed)
    try:
        model, sup, items, teacher = load_eval_state(args, [gpu])
        res_q.put(("ready", gpu, None, None))
        while True:
            task = task_q.get()
            if task is None:
                break
            idx, cand, clip_idx = task
            batch = [items[i] for i in clip_idx]
            kl, mse = fitness_terms(model, sup, cand, batch, teacher)
            res_q.put(("res", idx, kl, mse))
    except Exception as e:  # surface the death; the driver asserts on it
        res_q.put(("err", gpu, repr(e), None))
        raise


def driver_main(args):
    """Multi-GPU search: candidates fan out to persistent per-card workers.

    The evaluations are identical to the single-GPU path (same candidates, same
    minibatch, same math) -- only their placement changes, so --workers is a pure
    wall-clock knob."""
    gpus = [int(x) for x in args.gpu.split(",")][: args.workers]
    assert len(gpus) == args.workers, "need one --gpu id per worker"
    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sup = make_supernet(args, None)
    names_by_type = {"mha": [n for n in sup.names if "self_attn" in n],
                     "mlp": [n for n in sup.names if "mlp" in n]}
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]
    teacher_dir = REPO / "outputs" / args.teacher_id
    missing = [c for c, _ in calib if not (teacher_dir / f"{c}.npz").exists()]
    assert not missing, f"teacher cache incomplete ({len(missing)}), run --teacher-only"

    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(g, args, task_q, res_q), daemon=True)
             for g in gpus]
    for pr in procs:
        pr.start()
    for _ in procs:
        msg = res_q.get()
        assert msg[0] == "ready", msg

    def eval_batch(cands, clip_idx):
        for i, c in enumerate(cands):
            task_q.put((i, c, clip_idx))
        out = [None] * len(cands)
        for _ in cands:
            msg = res_q.get()
            assert msg[0] == "res", msg
            out[msg[1]] = (msg[2], msg[3])
        return out

    parent = {n: 0 for n in sup.names}
    all_idx = list(range(len(calib)))
    (kl0, mse0), = eval_batch([parent], all_idx)
    kl0, mse0 = max(kl0, 1e-8), max(mse0, 1e-8)
    print(f"norms at uniform init: KL0={kl0:.6f} MSE0={mse0:.6f}", flush=True)
    (out_dir / "config.json").write_text(json.dumps({
        **{k: v for k, v in vars(args).items()},
        "model_revision": MODEL_REV, "clip_ids": [c for c, _ in calib],
        "kl0": kl0, "mse0": mse0,
        "fitness": "KL_coc/KL0 + MSE_vf/MSE0, dual-teacher, label-free",
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4] for teacher rollouts",
    }, indent=2))
    if args.selection == "dual":
        (out_dir / "mask_meta.json").write_text(json.dumps(sup.meta, indent=2))

    log = []
    fit_parent = 2.0
    for gen in range(args.generations):
        t0 = time.time()
        offspring = []
        tries = 0
        while len(offspring) < args.offspring and tries < args.offspring * 50:
            tries += 1
            child = mutate(parent, sup, names_by_type)
            if child != parent and child not in offspring:
                offspring.append(child)
        assert offspring, "mutation produced no valid offspring"
        for si, (surv, ncl) in enumerate(zip(args.survivors, args.stage_clips)):
            if si == len(args.survivors) - 1 and parent not in offspring:
                offspring.append(parent)  # elitist
            clip_idx = random.sample(all_idx, min(ncl, len(all_idx)))
            terms = eval_batch(offspring, clip_idx)
            scored = sorted(((kl / kl0 + mse / mse0, kl, mse, c)
                             for (kl, mse), c in zip(terms, offspring)),
                            key=lambda x: x[0])
            offspring = [c for *_, c in scored[:surv]]
        fit_parent, kl_p, mse_p, parent = scored[0][:4]
        log.append({"gen": gen, "fitness": fit_parent, "kl": kl_p, "mse": mse_p,
                    "sec": round(time.time() - t0, 1),
                    "levels": [parent[n] for n in sup.names]})
        (out_dir / "search_log.json").write_text(json.dumps(log, indent=2))
        (out_dir / "final_config.json").write_text(json.dumps(parent, indent=2))
        print(f"[gen {gen + 1}/{args.generations}] fit={fit_parent:.4f} "
              f"(kl {kl_p:.5f}, mse {mse_p:.6f}) {time.time() - t0:.0f}s", flush=True)

    for _ in procs:
        task_q.put(None)
    for pr in procs:
        pr.join(timeout=60)
    mha_sum = sum(parent[n] for n in names_by_type["mha"])
    mlp_sum = sum(parent[n] for n in names_by_type["mlp"])
    assert mha_sum == 0 and mlp_sum == 0, (mha_sum, mlp_sum)
    (out_dir / "summary.txt").write_text(
        f"tyr search ({args.workers} workers): {args.generations} gens x "
        f"{args.offspring} offspring, final fitness {fit_parent:.4f} (init 2.0)\n"
        f"levels mha {[parent[n] for n in names_by_type['mha']]}\n"
        f"levels mlp {[parent[n] for n in names_by_type['mlp']]}\n")
    print("saved ->", out_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supernet", default="tyr_supernet_u40")
    ap.add_argument("--teacher-id", default="tyr_teacher_u40")
    ap.add_argument("--exp-id", default="tyr_search_u40")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--calib-manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--t-grid", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--noise-draws", type=int, default=2)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--offspring", type=int, default=32)
    ap.add_argument("--stage-clips", type=int, nargs="+", default=[4, 16, 48])
    ap.add_argument("--survivors", type=int, nargs="+", default=[8, 2, 1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-only", action="store_true")
    ap.add_argument("--selection", choices=["weights", "dual"], default="weights",
                    help="weights: supernet weight files (Tyr); dual: mask-only supernet "
                         "from the dual ranking (dual-global arm)")
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--workers", type=int, default=1,
                    help=">1: multi-GPU candidate-parallel search, one worker per "
                         "--gpu id; pure wall-clock knob, same evaluations")
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    ap.add_argument("--gpu", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.workers > 1:
        assert not args.teacher_only, "--teacher-only is single-GPU"
        driver_main(args)
        return

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = REPO / "outputs" / args.teacher_id
    calib = sc.calib_samples(REPO, args.calib_manifest)[: args.num_clips]

    devices = None if args.gpu is None else [int(x) for x in args.gpu.split(",")]
    device = reserve_gpu(args.reserve_gb, devices=devices)
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    # ---- stage A: teacher cache (dense model; must run before any supernet load)
    build_teacher(model, processor, calib, args, teacher_dir)
    if args.teacher_only:
        print("teacher cache complete ->", teacher_dir, flush=True)
        return

    sup = make_supernet(args, model)
    names_by_type = {"mha": [n for n in sup.names if "self_attn" in n],
                     "mlp": [n for n in sup.names if "mlp" in n]}

    print(f"preloading {len(calib)} clip inputs + teacher caches...", flush=True)
    items, teacher = [], {}
    for clip_id, t0 in calib:
        data = sc.load_cached(sc.path_for(args.cache, clip_id, t0))
        inp = lib.build_inputs(model, processor, data, "cuda")
        items.append({"clip_id": clip_id,
                      "pixel_values": inp["tokenized_data"]["pixel_values"].cpu(),
                      "image_grid_thw": inp["tokenized_data"]["image_grid_thw"].cpu()})
        teacher[clip_id] = dict(np.load(teacher_dir / f"{clip_id}.npz"))

    parent = {n: 0 for n in sup.names}
    kl0, mse0 = fitness_terms(model, sup, parent, items, teacher)
    kl0, mse0 = max(kl0, 1e-8), max(mse0, 1e-8)
    print(f"norms at uniform init: KL0={kl0:.6f} MSE0={mse0:.6f}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        **{k: v for k, v in vars(args).items()},
        "model_revision": MODEL_REV, "clip_ids": [c for c, _ in calib],
        "kl0": kl0, "mse0": mse0,
        "fitness": "KL_coc/KL0 + MSE_vf/MSE0, dual-teacher, label-free",
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4] for teacher rollouts",
    }, indent=2))
    if args.selection == "dual":
        (out_dir / "mask_meta.json").write_text(json.dumps(sup.meta, indent=2))

    log = []
    fit_parent = 2.0  # by construction at init
    for gen in range(args.generations):
        t0 = time.time()
        offspring = []
        tries = 0
        while len(offspring) < args.offspring and tries < args.offspring * 50:
            tries += 1
            child = mutate(parent, sup, names_by_type)
            if child != parent and child not in offspring:
                offspring.append(child)
        assert offspring, "mutation produced no valid offspring"

        for si, (surv, ncl) in enumerate(zip(args.survivors, args.stage_clips)):
            if si == len(args.survivors) - 1 and parent not in offspring:
                offspring.append(parent)  # elitist
            batch = random.sample(items, min(ncl, len(items)))
            scored = []
            for cand in offspring:
                kl, mse = fitness_terms(model, sup, cand, batch, teacher)
                scored.append((kl / kl0 + mse / mse0, kl, mse, cand))
            scored.sort(key=lambda x: x[0])
            offspring = [c for _, _, _, c in scored[:surv]]
        fit_parent, kl_p, mse_p, parent = scored[0][:4]
        log.append({"gen": gen, "fitness": fit_parent, "kl": kl_p, "mse": mse_p,
                    "sec": round(time.time() - t0, 1),
                    "levels": [parent[n] for n in sup.names]})
        (out_dir / "search_log.json").write_text(json.dumps(log, indent=2))
        (out_dir / "final_config.json").write_text(json.dumps(parent, indent=2))
        print(f"[gen {gen + 1}/{args.generations}] fit={fit_parent:.4f} "
              f"(kl {kl_p:.5f}, mse {mse_p:.6f}) {time.time() - t0:.0f}s", flush=True)

    mha_sum = sum(parent[n] for n in names_by_type["mha"])
    mlp_sum = sum(parent[n] for n in names_by_type["mlp"])
    assert mha_sum == 0 and mlp_sum == 0, (mha_sum, mlp_sum)  # budget conserved
    (out_dir / "summary.txt").write_text(
        f"tyr search: {args.generations} gens x {args.offspring} offspring, "
        f"final fitness {fit_parent:.4f} (uniform init = 2.0)\n"
        f"levels mha {[parent[n] for n in names_by_type['mha']]}\n"
        f"levels mlp {[parent[n] for n in names_by_type['mlp']]}\n")
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
