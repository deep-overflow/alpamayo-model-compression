"""Tyr (the ICLR paper's port, row `tyr_u40_r` / u50 `tyr_u50_r`) on Alpamayo 2 Edge (plans/2026-11-15_a2-edge-baselines-u50.md).
Edge venv.

Stages (the A1.5 scripts run_tyr_supernet.py / run_tyr_search.py / make_slim tyr_u*_r; tyr_lib imported unchanged):
  teacher    per calib_100 clip, the DENSE model: own CoC rollout (clip seed) -> teacher-forced sequence, top-1024 logits at the CoC
             positions, expert vector field at 2 noise draws x t in {0.1, 0.5, 0.9} with x_t = (1 - t) noise + t x1, x1 = GT action
  supernet   block-sequential over the 28 VLM layers: Hessians H = sum x x^T at the inputs of o_proj and mlp.fc2 (Edge relu^2 MLP;
             A1.5 down_proj) over the 100 prompt prefills (all tokens, no labels); OSSCAR removal + least-squares reconstruction at
             9 levels (tyr_lib.prune_levels, damp 1e-2; o_proj groups = 16 heads x 128 columns, 1 round; fc2 groups = single
             channels, 16 rounds); the level-0 weight is written back before the next block (median error accumulation)
             u50 level 0 = MoOP u50's cut: 8/16 heads, 4,608/9,216 channels; steps 1 head / 192 channels (A1.5 256 of 12,288,
             scaled to Edge's width), so any type-conserving level map keeps the removed total at 645,922,816
  search     candidates = {56 module names -> level}; mutation moves one level between two modules of the same type (budget
             conserved); 20 generations x 32 offspring, staged selection on 4 / 16 / 48 random clips with 8 / 2 / 1 survivors,
             elitist parent; fitness = KL_coc / KL0 + MSE_vf / MSE0 (uniform init = 2.0), KL = KL(teacher || student) over the
             teacher's top-1024 at CoC positions (student teacher-forced on the teacher's sequence), MSE = student expert field
             (reading the student's own cache) vs the teacher field
  assemble   chosen level weights -> weights_tyr_u50.pt (o_proj / fc2 per layer, bf16) + masks_tyr_u50.npz from the exact-zero
             columns; evaluation loads the weights (run_edge_openloop.py --weights) and the masks
Edge inputs: calib_edge cache, the no-nav prompt, mm_token_type_ids.

Usage (Edge venv; CUDA_VISIBLE_DEVICES = the card(s) to use):
  python iclr27-rebuttal/code/edge_tyr.py teacher
  python iclr27-rebuttal/code/edge_tyr.py supernet
  python iclr27-rebuttal/code/edge_tyr.py search --workers 4          # CUDA_VISIBLE_DEVICES=4,5,6,7
  python iclr27-rebuttal/code/edge_tyr.py assemble
"""

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import edge_common as ec

ec.setup_hf_env(offline=True)

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import edge_eval_lib as eel
import edge_grad_lib as gl
import tyr_lib as tyr

OUT = REPO / "outputs" / "iclr27-rebuttal"
EDGE_MOOP = OUT / "edge_moop"


def load_edge():
    from alpamayo2_edge import helper

    model, _ = ec.load_model("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, helper.get_processor(model.tokenizer, model.config)


def calib():
    return [(r.clip_id, int(r.t0_us)) for r in pd.read_parquet(ec.CALIB_MANIFEST).itertuples()]


def build_td(model, processor, clip_id, t0, keep_data=False):
    """Fused prompt inputs (tokenized_data) for one calib clip, on CPU; optionally the raw sample too."""
    data = ec.load_cached(ec.path_for(clip_id, t0))
    inputs, _ = ec.prepare_inputs_no_nav(data, model.config, model.tokenizer, processor)
    td = gl.fused_tokenized(model, inputs)
    keep = ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    item = {"clip_id": clip_id, **{k: td[k].cpu() for k in keep}}
    return (item, data, inputs) if keep_data else item


def to_cuda(item):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in item.items()}


def tf_forward(model, seq, item, use_cache):
    """Teacher-forced VLM forward over seq (prompt + CoC); returns (hidden, cache, rope_deltas)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if use_cache:
            return gl.vlm_forward_with_grad(model, seq, to_cuda(item))
        td = to_cuda(item)
        pl = td["input_ids"].shape[1]
        mm = torch.cat([td["mm_token_type_ids"], td["mm_token_type_ids"].new_zeros((1, seq.shape[1] - pl))], 1)
        out = model.vlm.model(input_ids=seq, attention_mask=torch.ones_like(seq), pixel_values=td["pixel_values"],
                              image_grid_thw=td["image_grid_thw"], mm_token_type_ids=mm, use_cache=False)
        return out.last_hidden_state, None, None


def expert_field(model, cache, rope_deltas, prefill, x_t, t_val):
    """One expert forward: the vector field at (x_t, t) on the given VLM cache (run_tyr_search.expert_field, Edge modules)."""
    from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask

    dev = x_t.device
    dims = model.expert.action_space.get_action_space_dims()
    n_tok = dims[0]
    pos, mask = build_expert_pos_ids_and_attn_mask(
        offset=torch.tensor([prefill], device=dev), rope_deltas=rope_deltas.to(dev), kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=dev, prefix_mask=torch.ones(1, prefill, device=dev, dtype=torch.long))
    t = torch.full((1, 1, 1), t_val, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        emb = model.expert.action_in_proj(x_t, t).reshape(1, n_tok, -1)
        out = model.expert.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache, attention_mask=mask,
                                  use_cache=True, is_causal=False)
        cache.crop(prefill)
        pred = model.expert.action_out_proj(out.last_hidden_state).reshape(1, *dims)
    return pred.float()  # (1, 64, 2)


# ------------------------------------------------------------------ teacher
def cmd_teacher(args):
    model, processor = load_edge()
    tdir = OUT / args.teacher_id
    tdir.mkdir(parents=True, exist_ok=True)
    for ci, (clip_id, t0) in enumerate(calib()):
        path = tdir / f"{clip_id}.npz"
        if path.exists():
            continue
        item, data, inputs = build_td(model, processor, clip_id, t0, keep_data=True)
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        inputs["tokenized_data"] = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs["tokenized_data"].items()}
        seed = ec.clip_seed(args.seed, clip_id)
        eel.seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seq, prompt_len, _, cot, _ = gl.rollout_sequence(model, inputs, args.max_gen)
        seq = seq.clone()
        cs, ce = prompt_len, seq.shape[1]
        hidden, cache, rope = tf_forward(model, seq, item, use_cache=True)
        prefill = cache.get_seq_length()
        with torch.no_grad():
            logits = model.vlm.lm_head(hidden[:, cs - 1: ce - 1]).float()
        topv, topi = logits.topk(args.topk, dim=-1)  # (1, Tc, k)
        x1 = gl.gt_actions(model, data, "cuda")  # (1, 64, 2)
        gen = torch.Generator().manual_seed(seed)
        xts, tvals, fields = [], [], []
        for _ in range(args.noise_draws):
            noise = torch.randn(x1.shape, generator=gen).to(x1.device)
            for t_val in args.t_grid:
                x_t = (1.0 - t_val) * noise + t_val * x1
                fields.append(expert_field(model, cache, rope, prefill, x_t, t_val).cpu().numpy()[0])
                xts.append(x_t.cpu().numpy()[0])
                tvals.append(t_val)
        np.savez(path, seq=seq[0].cpu().numpy().astype(np.int32), coc_start=cs, coc_end=ce,
                 topv=topv[0].cpu().numpy().astype(np.float16), topi=topi[0].cpu().numpy().astype(np.int32),
                 xt=np.stack(xts), tv=np.array(tvals), vd=np.stack(fields))
        del hidden, cache
        print(f"[teacher {ci + 1}/100] {clip_id[:8]} coc={ce - cs} | {cot[:50]!r}", flush=True)


# ------------------------------------------------------------------ supernet
def cmd_supernet(args):
    model, processor = load_edge()
    out_dir = OUT / args.supernet
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = model.config.vlm_config.text_config
    layers = model.vlm.model.language_model.layers
    keeps_q = tyr.level_keeps(tc.num_attention_heads, args.head_cut, args.head_step, args.num_levels)
    keeps_m = tyr.level_keeps(tc.intermediate_size, args.mlp_cut, args.mlp_step, args.num_levels)
    print(f"preloading {len(calib())} clip inputs...", flush=True)
    store = [build_td(model, processor, c, t) for c, t in calib()]
    names = [f"layers.{i:02d}.{s}" for i in range(len(layers)) for s in ("mlp.fc2", "self_attn.o_proj")]
    (out_dir / "metadata.json").write_text(json.dumps({
        "model": ec.MODEL_ID, "model_revision": ec.MODEL_REV, "num_clips": len(store), "clip_ids": [s["clip_id"] for s in store],
        "head_cut": args.head_cut, "head_step": args.head_step, "mlp_cut": args.mlp_cut, "mlp_step": args.mlp_step,
        "num_levels": args.num_levels, "damp": args.damp, "selection": "osscar",
        "levels_q": {str(k): v for k, v in keeps_q.items()}, "levels_mlp": {str(k): v for k, v in keeps_m.items()},
        "layer_names": names, "hessian_tokens": "full fused no-nav prompt prefill, no labels",
        "error_accumulation": "median (level-0 weight written back per block)"}, indent=2))
    t_all = time.time()
    for i, layer in enumerate(layers):
        t0 = time.time()
        o_proj, fc2 = layer.self_attn.o_proj, layer.mlp.fc2
        hooks = {"o": tyr.HessianHook(o_proj), "m": tyr.HessianHook(fc2)}
        for item in store:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                td = to_cuda(item)
                model.vlm.model(input_ids=td["input_ids"], attention_mask=torch.ones_like(td["input_ids"]),
                                pixel_values=td["pixel_values"], image_grid_thw=td["image_grid_thw"],
                                mm_token_type_ids=td["mm_token_type_ids"], use_cache=False)
        for h in hooks.values():
            h.remove()
        t_fwd = time.time() - t0
        for mod, hook, keeps, n_groups, upd, suffix in (
                (o_proj, hooks["o"], keeps_q, tc.num_attention_heads, 1, "self_attn.o_proj"),
                (fc2, hooks["m"], keeps_m, tc.intermediate_size, 16, "mlp.fc2")):
            name = f"layers.{i:02d}.{suffix}"
            gs = mod.in_features // n_groups
            sols = tyr.prune_levels(mod, hook.H, n_groups, sorted(set(keeps.values())), update_iter=upd, damp=args.damp)
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            for lv, keep in keeps.items():
                w = sols[keep]
                n_zero = int((w.abs().sum(0).reshape(n_groups, gs).sum(1) == 0).sum())
                assert n_zero == n_groups - keep, (name, lv, n_zero, keep)
                torch.save(w.to(torch.bfloat16).cpu(), d / f"{lv}.pth")
            mod.weight.data.copy_(sols[keeps[0]].to(mod.weight.dtype))  # error accumulation
        del hooks
        torch.cuda.empty_cache()
        print(f"[block {i + 1}/{len(layers)}] fwd {t_fwd:.0f}s total {time.time() - t0:.0f}s", flush=True)
    (out_dir / "summary.txt").write_text(
        f"Edge tyr supernet: {len(layers)} blocks x {len(keeps_q)}/{len(keeps_m)} levels, {len(store)} clips, "
        f"{time.time() - t_all:.0f}s; level 0 = cut {args.head_cut}/{tc.num_attention_heads} heads, "
        f"{args.mlp_cut}/{tc.intermediate_size} channels per layer\n")
    print("saved ->", out_dir, flush=True)


# ------------------------------------------------------------------ search
class Supernet:
    """Weight store + current model state (run_tyr_search.Supernet, Edge modules)."""

    def __init__(self, model, sup_dir, meta):
        self.dir, self.names = sup_dir, meta["layer_names"]
        self.levels = {n: sorted(int(k) for k in (meta["levels_mlp"] if "mlp" in n else meta["levels_q"])) for n in self.names}
        self.modules = {}
        if model is not None:
            layers = model.vlm.model.language_model.layers
            for n in self.names:
                i = int(n.split(".")[1])
                self.modules[n] = layers[i].mlp.fc2 if "mlp" in n else layers[i].self_attn.o_proj
        self.state = {n: None for n in self.names}

    def load(self, cand):
        for n, lv in cand.items():
            if self.state[n] != lv:
                self.modules[n].weight.data.copy_(torch.load(self.dir / n / f"{lv}.pth", map_location="cuda"))
                self.state[n] = lv


def fitness_terms(model, sup, cand, items, teacher):
    """run_tyr_search.fitness_terms on Edge."""
    sup.load(cand)
    kls, mses = [], []
    for item in items:
        t = teacher[item["clip_id"]]
        seq = torch.from_numpy(t["seq"]).long().unsqueeze(0).cuda()
        cs, ce = int(t["coc_start"]), int(t["coc_end"])
        hidden, cache, rope = tf_forward(model, seq, item, use_cache=True)
        prefill = cache.get_seq_length()
        with torch.no_grad():
            logits = model.vlm.lm_head(hidden[:, cs - 1: ce - 1]).float()
        topi = torch.from_numpy(t["topi"]).long().unsqueeze(0).cuda()
        topv = torch.from_numpy(t["topv"]).float().unsqueeze(0).cuda()
        kl = F.kl_div(logits.gather(-1, topi).log_softmax(-1).flatten(0, 1), topv.log_softmax(-1).flatten(0, 1),
                      log_target=True, reduction="batchmean")
        kls.append(float(kl))
        mse = 0.0
        for p in range(t["tv"].shape[0]):
            x_t = torch.from_numpy(t["xt"][p]).unsqueeze(0).cuda()
            v = expert_field(model, cache, rope, prefill, x_t, float(t["tv"][p]))
            mse += float(F.mse_loss(v, torch.from_numpy(t["vd"][p]).unsqueeze(0).cuda()))
        mses.append(mse / t["tv"].shape[0])
        del hidden, cache
    return float(np.mean(kls)), float(np.mean(mses))


def mutate(parent, sup, names_by_type):
    """run_tyr_search.mutate: move one level quantum between two modules of the same type."""
    child = dict(parent)
    flips = min(random.randint(1, 4), random.randint(1, 4))
    for _ in range(flips):
        names = names_by_type[random.choice(("mha", "mlp"))]
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


def worker_main(gpu, args, task_q, res_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.manual_seed(args.seed)
    try:
        model, processor = load_edge()
        sup_dir = OUT / args.supernet
        sup = Supernet(model, sup_dir, json.loads((sup_dir / "metadata.json").read_text()))
        items = [build_td(model, processor, c, t) for c, t in calib()]
        teacher = {it["clip_id"]: dict(np.load(OUT / args.teacher_id / f"{it['clip_id']}.npz")) for it in items}
        res_q.put(("ready", gpu, None, None))
        while True:
            task = task_q.get()
            if task is None:
                break
            idx, cand, clip_idx = task
            kl, mse = fitness_terms(model, sup, cand, [items[i] for i in clip_idx], teacher)
            res_q.put(("res", idx, kl, mse))
    except Exception as e:  # surface a worker death to the driver, which asserts on it
        res_q.put(("err", gpu, repr(e), None))
        raise


def cmd_search(args):
    """run_tyr_search.driver_main on Edge: candidates fan out to one persistent worker per card."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")][: args.workers]
    assert len(gpus) == args.workers, "need one visible card per worker"
    out_dir = OUT / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sup_dir = OUT / args.supernet
    sup = Supernet(None, sup_dir, json.loads((sup_dir / "metadata.json").read_text()))
    names_by_type = {"mha": [n for n in sup.names if "self_attn" in n], "mlp": [n for n in sup.names if "mlp" in n]}
    clips = calib()
    missing = [c for c, _ in clips if not (OUT / args.teacher_id / f"{c}.npz").exists()]
    assert not missing, f"teacher cache incomplete ({len(missing)})"
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(g, args, task_q, res_q), daemon=True) for g in gpus]
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
    all_idx = list(range(len(clips)))
    (kl0, mse0), = eval_batch([parent], all_idx)
    kl0, mse0 = max(kl0, 1e-8), max(mse0, 1e-8)
    print(f"norms at uniform init: KL0={kl0:.6f} MSE0={mse0:.6f}", flush=True)
    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "model": ec.MODEL_ID, "model_revision": ec.MODEL_REV, "clip_ids": [c for c, _ in clips],
        "kl0": kl0, "mse0": mse0, "fitness": "KL_coc/KL0 + MSE_vf/MSE0, dual-teacher, label-free",
        "seed_rule": "sha256(f'{seed}:{clip_id}')[:4] for teacher rollouts"}, indent=2, default=str))
    log, fit_parent, t_start = [], 2.0, time.time()
    for gen in range(args.generations):
        t0 = time.time()
        offspring, tries = [], 0
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
            scored = sorted(((kl / kl0 + mse / mse0, kl, mse, c) for (kl, mse), c in zip(terms, offspring)),
                            key=lambda x: x[0])
            offspring = [c for *_, c in scored[:surv]]
        fit_parent, kl_p, mse_p, parent = scored[0][:4]
        log.append({"gen": gen, "fitness": fit_parent, "kl": kl_p, "mse": mse_p, "sec": round(time.time() - t0, 1),
                    "levels": [parent[n] for n in sup.names]})
        (out_dir / "search_log.json").write_text(json.dumps(log, indent=2))
        (out_dir / "final_config.json").write_text(json.dumps(parent, indent=2))
        print(f"[gen {gen + 1}/{args.generations}] fit={fit_parent:.4f} (kl {kl_p:.5f}, mse {mse_p:.6f}) "
              f"{time.time() - t0:.0f}s", flush=True)
    for _ in procs:
        task_q.put(None)
    for pr in procs:
        pr.join(timeout=60)
    mha_sum = sum(parent[n] for n in names_by_type["mha"])
    mlp_sum = sum(parent[n] for n in names_by_type["mlp"])
    assert mha_sum == 0 and mlp_sum == 0, (mha_sum, mlp_sum)  # budget conserved
    (out_dir / "summary.txt").write_text(
        f"Edge tyr search ({args.workers} workers): {args.generations} gens x {args.offspring} offspring, final fitness "
        f"{fit_parent:.4f} (init 2.0), {time.time() - t_start:.0f}s\nlevels mha {[parent[n] for n in names_by_type['mha']]}\n"
        f"levels mlp {[parent[n] for n in names_by_type['mlp']]}\n")
    print("saved ->", out_dir, flush=True)


# ------------------------------------------------------------------ assemble
def cmd_assemble(args):
    """make_slim tyr_u*_r on Edge: copy each module's chosen level, derive masks from the exact-zero columns."""
    sup_dir = OUT / args.supernet
    meta = json.loads((sup_dir / "metadata.json").read_text())
    cfg = json.loads((OUT / args.exp_id / "final_config.json").read_text())
    n_layers = len({n.split(".")[1] for n in meta["layer_names"]})
    H, I, hd, hidden = 16, 9216, 128, 2048
    q = np.ones((n_layers, H), dtype=np.float32)
    m = np.ones((n_layers, I), dtype=np.float32)
    weights = {}
    for n in meta["layer_names"]:
        lv = cfg[n]
        w = torch.load(sup_dir / n / f"{lv}.pth", map_location="cpu")  # bf16 (hidden, in)
        weights[n] = w
        i = int(n.split(".")[1])
        col = w.float().abs().sum(0)  # (in,)
        if "mlp" in n:
            m[i] = (col != 0).float().numpy()
            assert int((m[i] == 0).sum()) == I - meta["levels_mlp"][str(lv)], n
        else:
            q[i] = (col.view(H, hd).sum(1) != 0).float().numpy()
            assert int((q[i] == 0).sum()) == H - meta["levels_q"][str(lv)], n
    removed = int((q == 0).sum() * 2 * hd * hidden + (m == 0).sum() * 2 * hidden)
    torch.save(weights, EDGE_MOOP / f"weights_{args.name}.pt")
    np.savez(EDGE_MOOP / f"masks_{args.name}.npz", q_mask=q, mlp_mask=m)
    info = {"name": args.name, "supernet": str(sup_dir), "search": str(OUT / args.exp_id), "removed_params": removed,
            "heads_cut_per_layer": [int(x) for x in (q == 0).sum(1)], "mlp_cut_per_layer": [int(x) for x in (m == 0).sum(1)],
            "weights": str(EDGE_MOOP / f"weights_{args.name}.pt"), "masks": str(EDGE_MOOP / f"masks_{args.name}.npz")}
    (OUT / args.exp_id / "assembly.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "supernet", "search", "assemble"])
    ap.add_argument("--teacher-id", default="edge_tyr_teacher")
    ap.add_argument("--supernet", default="edge_tyr_supernet_u50")
    ap.add_argument("--exp-id", default="edge_tyr_search_u50")
    ap.add_argument("--name", default="tyr_u50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--t-grid", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--noise-draws", type=int, default=2)
    ap.add_argument("--num-levels", type=int, default=9)
    ap.add_argument("--head-cut", type=int, default=8)
    ap.add_argument("--head-step", type=int, default=1)
    ap.add_argument("--mlp-cut", type=int, default=4608)
    ap.add_argument("--mlp-step", type=int, default=192)
    ap.add_argument("--damp", type=float, default=1e-2)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--offspring", type=int, default=32)
    ap.add_argument("--stage-clips", type=int, nargs="+", default=[4, 16, 48])
    ap.add_argument("--survivors", type=int, nargs="+", default=[8, 2, 1])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.stage != "assemble" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES")
    {"teacher": cmd_teacher, "supernet": cmd_supernet, "search": cmd_search, "assemble": cmd_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
