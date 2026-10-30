"""Minitron knowledge-distillation recovery for a slim Alpamayo-1.5 student (plan D1-D3, D10).

Loss per micro-batch (minitron/plans/2026-10-23_plan.md section 3, decided 2026-10-24):
  L = KL(p_T || p_S)  forward, tau = 1, full vocabulary, mean over the supervised tokens:
                      the CoC, cot_end, traj_future_start and the 128 GT discrete
                      trajectory tokens
    + MSE(v_S, v_T)   expert velocity (64 x 2) at a GT interpolant
                      x_t = (1 - t) noise + t x1, one t per sample, uniform on the
                      inference grid {0, 0.1, ..., 0.9}
  Plain sum: under KI the two terms update disjoint parameters, so a weight would only
  rescale one tower's AdamW update.
KI (Alpamayo-R1's own training recipe): the expert reads a student cache computed without
gradient, so the velocity loss trains the expert and the action projections only and the
KL trains the VLM only. Teacher = the dense Alpamayo-1.5-10B, frozen, run online or -- with
--teacher-targets -- precomputed once by make_teacher_targets.py: full-vocabulary logits at
the supervised positions (2407.14679: top-K with K <= 100 hurts) and the velocity at
mt_data.kd_draw(window, epoch). Both modes query the expert at the same (noise, t).

Context: text = teacher-generated CoC (kd_r1.parquet, R1 windows); trajectory tokens = GT
discretised by traj_tokenizer. The expert never sees the trajectory tokens: each sample's
offset is its CoC end, and the release's mask builder hides every cache position from
there on (trajectory tokens and right padding alike).

The student runs its VLM twice per step: once with gradient and activation checkpointing
for the KL (use_cache=False -- HF disables the cache under checkpointing), and once in
eval mode under no_grad with use_cache=True for the cache the expert reads. KI keeps that
cache gradient-free anyway, so the second pass changes no gradient; it costs a forward.

Sharding is FSDP2 (fully_shard) on every decoder layer / ViT block and on every module the
step calls directly -- embed_tokens, the final norm, visual, lm_head, expert,
action_in_proj, action_out_proj -- so no parameter is read outside its own forward
(--shard-group N: HSDP, sharded inside groups of N ranks and replicated across them). The
student keeps fp32 master weights with bf16 compute; the teacher is bf16 and sharded the same
way. Both are loaded on CPU one after the other and moved to GPU shard by shard, which keeps
host memory at one model per rank. AdamW (betas 0.9 / 0.95, weight decay 0.1 -- Megatron
defaults; the Minitron papers do not state theirs) over all student parameters; gradient
norm clipped per tower (VLM vs expert + action projections), since a joint clip would couple
the insulated towers again.

Schedule (D10-2): peak LR 1e-4, cosine to 1e-5, warmup 40 steps, global batch 64 samples
= world x micro-batch x accumulation, 3 epochs. 64 KD clips are held out; their KL and
velocity loss are logged every --eval-every steps. A full bf16 checkpoint (slim_state.pt +
the student's slim_meta.json) is written after every epoch, for the offline probe on
val_official_238 that picks the checkpoint.

mtw24_kd2 additions (minitron/plans/2026-10-26_kisti-training-plan.md, all in one run):
  --kl-groups     KL = mean over the trajectory group (trajectory vocabulary + traj_future_start)
                  + mean over the rest, the split of the release SFT loss
  --vqa-manifest  LingoQA QA mixed in, `--vqa-per-step` of every global batch; the GT answer is the
                  context, KL on answer + <|answer_end|>, no velocity term (online teacher only)
  --heldout-clips hold out every window of the listed clips (kd_r5 has five per clip)
  --freeze-vlm    (mtw24_kd2ef) VLM frozen: velocity loss only, expert + action projections train
  --save-fracs    checkpoints at fractions of the run instead of per epoch; each one but the last
                  also replaces <exp>/state (sharded fp32 weights + AdamW moments, ~100 GB)
  --deadline      stop before a step that would end past this unix time, after writing <exp>/state;
                  --resume continues from <exp>/state (NEURON jobs end after two days)

Usage (smoke on four cards):
  ALPAMAYO_REPO=$PWD bash experiments/recovery/run_ddp_retry.sh 3 "0 1 2 3" \\
      minitron/code/train_kd.py --student outputs/minitron/slim_mtw24 --exp-id kd_smoke \\
      --micro-batch 1 --max-steps 3 --no-save
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import functools
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))
sys.path.insert(0, str(REPO / "experiments" / "lingoqa"))

import analysis_lib as lib
import lingo_lib as ll
import mt_data
import recover_lib as rl
import sample_cache as sc
import slim_lib as sl
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from run_baseline import MODEL_REV

T_GRID = tuple(round(0.1 * i, 1) for i in range(10))  # inference Euler grid 0.0 .. 0.9
N_TOK = 64  # diffusion tokens = waypoints


# ---------------------------------------------------------------- distributed / models


def dist_init(gpus, offload=False):
    local = int(os.environ.get("LOCAL_RANK", "0"))
    dev = gpus[local] if gpus else local
    torch.cuda.set_device(dev)  # before init, or NCCL opens contexts on every card
    # gloo for CPU tensors: with --cpu-offload the gradient-norm reduction runs on CPU.
    # Only then -- DeviceMesh (HSDP) cannot split a group whose backend is "cpu:gloo,cuda:nccl"
    dist.init_process_group("cpu:gloo,cuda:nccl" if offload else "nccl",
                            timeout=timedelta(hours=2), device_id=torch.device(f"cuda:{dev}"))
    return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{dev}")


def shard_modules(model):
    """Every module whose forward the step calls, innermost first."""
    vlm = model.vlm.model
    return [*vlm.language_model.layers, *vlm.visual.blocks, *model.expert.layers,
            vlm.language_model.embed_tokens, vlm.language_model.norm, vlm.visual,
            model.vlm.lm_head, model.expert, model.action_in_proj, model.action_out_proj]


def shard(model, device, mp, dtype=None, offload=False, mesh=None):
    kw = {"offload_policy": CPUOffloadPolicy()} if offload else {}
    for m in shard_modules(model):
        if dtype is not None:  # cast only what this unit will own, one unit at a time
            for p in m.parameters():
                if not isinstance(p, DTensor):
                    p.data = p.data.to(dtype)
        fully_shard(m, mp_policy=mp, mesh=mesh, **kw)
    for m in model.modules():  # buffers outside any shard unit (rotary inv_freq, ...)
        for name, b in list(m.named_buffers(recurse=False)):
            if b is not None and b.device != device:
                m._buffers[name] = b.to(device)
    loose = [n for n, p in model.named_parameters() if not isinstance(p, DTensor)]
    assert not loose, f"parameters outside every FSDP unit: {loose[:5]}"


def load_teacher(device, mesh=None):
    t = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV,
                                    dtype=torch.bfloat16)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    shard(t, device, MixedPrecisionPolicy(param_dtype=torch.bfloat16), mesh=mesh)
    return t


def load_student(ckpt, device, offload=False, mesh=None, freeze_vlm=False):
    s = sl.load_slim(REPO / ckpt, device="cpu")
    for p in s.parameters():
        p.requires_grad_(True)
    if freeze_vlm:
        for p in s.vlm.parameters():
            p.requires_grad_(False)
    s.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    shard(s, device, MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                          reduce_dtype=torch.float32),
          dtype=torch.float32, offload=offload, mesh=mesh)  # fp32 master weights, cast unit by unit
    s.train()
    return s


def towers(model):
    vlm = list(model.vlm.parameters())
    act = (list(model.expert.parameters()) + list(model.action_in_proj.parameters())
           + list(model.action_out_proj.parameters()))
    return vlm, act


# ---------------------------------------------------------------- data


def load_raw(row, ns="train", tgt_dir=None):
    """(npz data, CoC text, manifest row, precomputed teacher targets or None)."""
    data = sc.load_cached(sc.path_for(ns, row.clip_id, int(row.t0_us)))
    tgt = None
    if tgt_dir is not None:
        tgt = torch.load(Path(tgt_dir) / f"{row.clip_id}__t0_{int(row.t0_us)}.pt",
                         map_location="cpu", weights_only=True)
    return data, row.coc, row, tgt


def load_vqa_raw(row, root):
    """(LingoQA segment: last 4 front-camera frames, manifest row) -- run_lingo_vqa.py's input."""
    return ll.load_segment(row.segment_id, data=Path(root), split="train"), row


def make_batch(ref, processor, tok, raws, max_coc, device):
    """Right-padded teacher-forced batch [prompt + CoC + terminals + 128 GT traj tokens]."""
    seqs, offsets, prompt_lens, pvs, thws, x1s = [], [], [], [], [], []
    for data, coc_text, *_ in raws:
        inputs = lib.build_inputs(ref, processor, data, device)
        coc = torch.tensor(rl.coc_ids(tok, coc_text, max_tokens=max_coc), device=device)
        traj = rl.traj_token_ids(ref, data, device)[0]  # (128,)
        seqs.append(torch.cat([inputs["input_ids"][0], coc, traj]))
        prompt_lens.append(inputs["input_ids"].shape[1])
        offsets.append(prompt_lens[-1] + coc.numel())  # CoC end = expert offset
        pvs.append(inputs["tokenized_data"]["pixel_values"])
        thws.append(inputs["tokenized_data"]["image_grid_thw"])
        x1s.append(lib.gt_actions(ref, data, device).float()[0])  # (64, 2)
    b = pad_batch(seqs, prompt_lens, pvs, thws, tok, device)
    return {**b, "kind": "drive", "offsets": torch.tensor(offsets, device=device),
            "x1": torch.stack(x1s)}


def make_vqa_batch(processor, tok, raws, max_answer, device):
    """Right-padded teacher-forced VQA batch [video + question, primed <|answer_start|> + GT answer
    + <|answer_end|>]: helper.create_vqa_message as in run_lingo_vqa.py, which stops on answer_end."""
    ans_end = tok.convert_tokens_to_ids("<|answer_end|>")
    seqs, prompt_lens, pvs, thws = [], [], [], []
    for data, row in raws:
        msgs = helper.create_vqa_message(frames=data["image_frames"].flatten(0, 1),
                                         question=str(row.question),
                                         camera_indices=data["camera_indices"])
        enc = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                            continue_final_message=True, return_dict=True,
                                            return_tensors="pt")
        ans = tok(str(row.answer), add_special_tokens=False)["input_ids"][:max_answer]
        prompt = enc["input_ids"][0].to(device)
        seqs.append(torch.cat([prompt, torch.tensor(ans + [ans_end], device=device)]))
        prompt_lens.append(prompt.numel())
        pvs.append(enc["pixel_values"].to(device))
        thws.append(enc["image_grid_thw"].to(device))
    return {**pad_batch(seqs, prompt_lens, pvs, thws, tok, device), "kind": "vqa"}


def pad_batch(seqs, prompt_lens, pvs, thws, tok, device):
    t_max = max(s.numel() for s in seqs)
    ids = torch.full((len(seqs), t_max), tok.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : s.numel()] = s
        mask[i, : s.numel()] = 1
    # supervised targets: every position from the prompt end to the sequence end
    b_idx, h_idx = [], []
    for i, s in enumerate(seqs):
        pos = torch.arange(prompt_lens[i], s.numel(), device=device)
        b_idx.append(torch.full_like(pos, i))
        h_idx.append(pos - 1)  # the hidden state that predicts token `pos`
    return {"ids": ids, "mask": mask, "pv": torch.cat(pvs), "thw": torch.cat(thws),
            "b_idx": torch.cat(b_idx), "h_idx": torch.cat(h_idx), "t_max": t_max}


def traj_group(b, traj_range):
    """(N,) True where the supervised target is a trajectory token or traj_future_start -- the
    `future_traj` group of the release SFT loss (alpamayo-recipes sft_base_model.py:388-407)."""
    tgt = b["ids"][b["b_idx"], b["h_idx"] + 1]  # (N,) the token each position predicts
    return ((tgt >= traj_range[0]) & (tgt < traj_range[1])) | (tgt == rl.TFS)


def batches(rows, micro, pool, load):
    """Yield raw micro-batches, prefetching the next one's files on a thread pool."""
    chunks = [rows[i : i + micro] for i in range(0, len(rows), micro)]
    nxt = [pool.submit(load, r) for r in chunks[0].itertuples()] if chunks else []
    for k in range(len(chunks)):
        cur = [f.result() for f in nxt]
        if k + 1 < len(chunks):
            nxt = [pool.submit(load, r) for r in chunks[k + 1].itertuples()]
        yield cur


# ---------------------------------------------------------------- forward pieces


def vlm_forward(model, b, use_cache):
    return model.vlm.model(input_ids=b["ids"], attention_mask=b["mask"], pixel_values=b["pv"],
                           image_grid_thw=b["thw"], use_cache=use_cache)


def sup_logits(model, out, b):
    h = out.last_hidden_state[b["b_idx"], b["h_idx"]]  # (N, H)
    return model.vlm.lm_head(h)  # (N, V)


def velocity(model, cache, rope_deltas, b, x_t, t):
    """Expert velocity at (x_t, t) reading `cache` up to each sample's CoC end. (B, 64, 2)"""
    bsz = x_t.shape[0]
    # get_rope_index takes each delta against the padded row length, so a right-padded sample
    # comes back `pad` short and its expert positions would shift by -pad; the release's
    # prefill has no padding. Add the padding back (checked against batch-1 passes).
    rope_deltas = rope_deltas + (b["t_max"] - b["mask"].sum(1, keepdim=True))  # (B, 1)
    pos, att = model._build_expert_pos_ids_and_attn_mask(
        offset=b["offsets"], rope_deltas=rope_deltas, kv_cache_seq_len=b["t_max"],
        n_diffusion_tokens=N_TOK, b_star=bsz, device=x_t.device)
    kw = {"is_causal": False} if model.config.expert_non_causal_attention else {}
    emb = model.action_in_proj(x_t.to(torch.bfloat16), t)  # (B, 64, 2048)
    if emb.dim() == 2:
        emb = emb.view(bsz, N_TOK, -1)
    out = model.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache,
                       attention_mask=att, use_cache=True, **kw)
    cache.crop(b["t_max"])
    return model.action_out_proj(out.last_hidden_state[:, -N_TOK:]).float()


def kl_fwd(logits_t, logits_s, grp=None):
    """Forward KL per supervised position, averaged over all of them, or -- with `grp` (N,)
    marking the trajectory group -- averaged within each group and summed (groups 1:1)."""
    lp_t = F.log_softmax(logits_t.float(), dim=-1)
    lp_s = F.log_softmax(logits_s.float(), dim=-1)
    kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1)  # (N,)
    if grp is None:
        return kl.mean()
    return sum(kl[m].mean() for m in (grp, ~grp) if m.any())


class TeacherStep(nn.Module):
    """FSDP root for the teacher: supervised logits and the velocity targets.

    A module whose forward runs with no FSDP ancestor becomes an FSDP root, and roots keep
    their parameters gathered after forward. Calling the shard units directly made every
    decoder layer a root (one smoke step held ~30 GB more than the activations); one root
    per model above all units makes them reshard after their forward as intended.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, b, x_t, t):
        drive = b["kind"] == "drive"
        out = vlm_forward(self.m, b, use_cache=drive)
        lg = sup_logits(self.m, out, b)
        if not drive:  # VQA: no trajectory, the expert does not run
            return lg, None
        v = velocity(self.m, out.past_key_values, out.rope_deltas, b, x_t, t)
        return lg, v


class StudentStep(nn.Module):
    """FSDP root for the student: (KL, velocity MSE) for one micro-batch."""

    def __init__(self, model, freeze_vlm=False):
        super().__init__()
        self.m = model
        self.freeze_vlm = freeze_vlm

    def forward(self, b, x_t, t, lg_t, v_t, train):
        if self.freeze_vlm:  # constant VLM: no KL pass, the cache pass below is its only forward
            l_kl = torch.zeros((), device=b["ids"].device)
        else:
            with torch.set_grad_enabled(train):
                out_s = vlm_forward(self.m, b, use_cache=False)
                l_kl = kl_fwd(lg_t, sup_logits(self.m, out_s, b), b.get("grp"))
                del out_s
        if b["kind"] != "drive":  # VQA: KL only, no cache pass and no expert
            return l_kl, torch.zeros((), device=l_kl.device)
        self.m.eval()  # the cache pass must not run through checkpointing
        with torch.no_grad():
            out_c = vlm_forward(self.m, b, use_cache=True)
        if train:
            self.m.train()
        with torch.set_grad_enabled(train):
            v_s = velocity(self.m, out_c.past_key_values, out_c.rope_deltas, b, x_t, t)
            l_v = F.mse_loss(v_s, v_t)
        return l_kl, l_v


def kd_losses(t_step, s_step, b, raws, slot, train):
    """(KL, velocity MSE) for one micro-batch; gradient only when `train`.

    The query (noise, t) comes from mt_data.kd_draw(window, slot) -- slot = epoch in training,
    0 for the held-out set -- so it is the draw the precomputed teacher velocity was taken at.
    With precomputed targets (raws[i][3]) the teacher does not run at all (t_step unused).
    VQA micro-batches (online teacher only) have no draw and no velocity term.
    """
    if b["kind"] == "vqa":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                lg_t, _ = t_step(b, None, None)
            return s_step(b, None, None, lg_t, None, train)
    device = b["x1"].device
    rows, tgts = [r[2] for r in raws], [r[3] for r in raws]
    draws = [mt_data.kd_draw(r.clip_id, r.t0_us, slot) for r in rows]
    noise = torch.stack([d[0] for d in draws]).to(device)  # (B, 64, 2)
    t_idx = [d[1] for d in draws]
    t = torch.tensor([T_GRID[i] for i in t_idx], device=device).view(-1, 1, 1)
    x_t = (1.0 - t) * noise + t * b["x1"]  # (B, 64, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if tgts[0] is None:
            with torch.no_grad():
                lg_t, v_t = t_step(b, x_t, t)
        else:
            assert all(int(g["t_idx"][slot]) == i for g, i in zip(tgts, t_idx)), "draw mismatch"
            lg_t = torch.cat([g["logits"] for g in tgts]).to(device, non_blocking=True)
            assert lg_t.shape[0] == b["b_idx"].numel(), "supervised positions mismatch"
            v_t = torch.stack([g["v"][slot] for g in tgts]).to(device)  # (B, 64, 2)
        return s_step(b, x_t, t, lg_t, v_t, train)


# ---------------------------------------------------------------- loop


def lr_at(step, total, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, total - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, prog)))


def mean_all(x):
    x = x.detach().clone()
    dist.all_reduce(x)
    return float(x) / dist.get_world_size()


@torch.no_grad()
def heldout_eval(t_step, s_step, held, held_vqa, args, ref, processor, tok, rank, world, pool,
                 device, traj_range):
    """Held-out (driving KL, velocity MSE, VQA KL or None). Both sets are cut to a multiple of the
    world size beforehand, so every rank runs the same micro-batches, as FSDP collectives need."""
    s_step.m.eval()
    tot = torch.zeros(5, device=device)
    load = functools.partial(load_raw, tgt_dir=args.teacher_targets)
    for raws in batches(held.iloc[rank::world], args.micro_batch, pool, load):
        b = make_batch(ref, processor, tok, raws, args.max_coc, device)
        if args.kl_groups:
            b["grp"] = traj_group(b, traj_range)
        l_kl, l_v = kd_losses(t_step, s_step, b, raws, 0, train=False)
        n = torch.tensor(float(len(raws)), device=device)
        tot[:3] += torch.stack([l_kl * n, l_v * n, n])
    if held_vqa is not None:
        load = functools.partial(load_vqa_raw, root=args.lingo_root)
        for raws in batches(held_vqa.iloc[rank::world], args.micro_batch_vqa, pool, load):
            b = make_vqa_batch(processor, tok, raws, args.max_answer, device)
            if args.kl_groups:
                b["grp"] = traj_group(b, traj_range)
            l_kl, _ = kd_losses(t_step, s_step, b, raws, 0, train=False)
            n = torch.tensor(float(len(raws)), device=device)
            tot[3:] += torch.stack([l_kl * n, n])
    dist.all_reduce(tot)
    s_step.m.train()
    vqa = float(tot[3] / tot[4]) if held_vqa is not None else None
    return float(tot[0] / tot[2]), float(tot[1] / tot[2]), vqa


def save_ckpt(student, args, out_dir, tag, rank):
    sd = get_model_state_dict(student, options=StateDictOptions(full_state_dict=True,
                                                                cpu_offload=True))
    if rank == 0:
        d = out_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.to(torch.bfloat16) for k, v in sd.items()}, d / "slim_state.pt")
        shutil.copy(REPO / args.student / "slim_meta.json", d / "slim_meta.json")
    dist.barrier()


def save_state(student, opt, out_dir, step, rank):
    """Resumable state: every rank writes its own shards to state.tmp, which then replaces state."""
    msd, osd = get_state_dict(student, opt)
    tmp, dst = out_dir / "state.tmp", out_dir / "state"
    if rank == 0 and tmp.exists():
        shutil.rmtree(tmp)
    dist.barrier()
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=tmp)
    if rank == 0:
        (tmp / "step.json").write_text(json.dumps({"step": step}))
        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    dist.barrier()


def load_state(student, opt, out_dir):
    msd, osd = get_state_dict(student, opt)  # initialises the AdamW moments (lr 0 step)
    sd = {"model": msd, "optim": osd}
    dcp.load(sd, checkpoint_id=out_dir / "state")
    set_state_dict(student, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    return json.loads((out_dir / "state" / "step.json").read_text())["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="slim ckpt dir (slim_meta.json)")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--manifest", default="outputs/minitron/kd_sets/kd_r1.parquet")
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-coc", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N steps")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="student params / grads / AdamW state on CPU (fits 48 GB cards, slow)")
    ap.add_argument("--shard-group", type=int, default=0,
                    help="HSDP: shard over groups of N consecutive ranks, replicate across "
                         "groups (cvlab21: 4 = one CPU socket); 0 = shard over all ranks")
    ap.add_argument("--teacher-targets", default=None,
                    help="dir from make_teacher_targets.py; the teacher is then not loaded")
    ap.add_argument("--heldout-clips", default=None,
                    help="json list of clip ids held out with all their windows (else --heldout "
                         "random rows)")
    ap.add_argument("--kl-groups", action="store_true",
                    help="KL = mean over trajectory-group positions + mean over the rest (the "
                         "release SFT split) instead of one mean over every position")
    ap.add_argument("--vqa-manifest", default=None, help="LingoQA QA rows (make_vqa_set.py)")
    ap.add_argument("--vqa-heldout", default=None)
    ap.add_argument("--vqa-per-step", type=int, default=16, help="VQA samples per global batch")
    ap.add_argument("--micro-batch-vqa", type=int, default=None)
    ap.add_argument("--lingo-root", default=os.environ.get(
        "LINGOQA_TRAIN", "/mnt/nvme1n1/ad_vla/data/lingoqa/train"))
    ap.add_argument("--max-answer", type=int, default=96)
    ap.add_argument("--save-fracs", type=float, nargs="*", default=None,
                    help="save at these fractions of the run (e.g. 0.25 0.5 0.75 1) instead of "
                         "after every epoch")
    ap.add_argument("--freeze-vlm", action="store_true",
                    help="train the expert + action projections only, on the frozen VLM's cache")
    ap.add_argument("--deadline", type=float, default=0,
                    help="unix time: write <exp>/state and stop before a step that would pass it")
    ap.add_argument("--resume", action="store_true", help="continue from <exp>/state if present")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.teacher_targets:
        args.teacher_targets = str(REPO / args.teacher_targets)
        tcfg = json.loads((Path(args.teacher_targets) / "config.json").read_text())
        assert tcfg["epochs"] >= args.epochs, f"targets hold {tcfg['epochs']} velocity slots"

    rank, world, device = dist_init(args.gpus, args.cpu_offload)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_dir = REPO / "outputs" / "minitron" / args.exp_id
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    args.micro_batch_vqa = args.micro_batch_vqa or args.micro_batch

    rows = pd.read_parquet(REPO / args.manifest)
    rows = rows[[sc.path_for("train", c, int(t)).exists()
                 for c, t in zip(rows.clip_id, rows.t0_us)]].reset_index(drop=True)
    if args.teacher_targets:
        absent = [f"{c}__t0_{int(t)}" for c, t in zip(rows.clip_id, rows.t0_us)
                  if not (Path(args.teacher_targets) / f"{c}__t0_{int(t)}.pt").exists()]
        assert not absent, f"{len(absent)} windows have no teacher targets, e.g. {absent[:3]}"
    if args.heldout_clips:
        hc = set(json.loads((REPO / args.heldout_clips).read_text()))
        held = rows[rows.clip_id.isin(hc)].reset_index(drop=True)
        train = rows[~rows.clip_id.isin(hc)].reset_index(drop=True)
    else:
        perm = np.random.default_rng(args.seed).permutation(len(rows))
        held = rows.iloc[perm[: args.heldout]].reset_index(drop=True)
        train = rows.iloc[perm[args.heldout :]].reset_index(drop=True)
    held = held.iloc[: len(held) // world * world]  # same micro-batch count on every rank
    vqa = held_vqa = None
    if args.vqa_manifest:
        assert not args.teacher_targets, "VQA KD runs with the online teacher only"
        vqa = pd.read_parquet(REPO / args.vqa_manifest)
        if args.vqa_heldout:
            held_vqa = pd.read_parquet(REPO / args.vqa_heldout)
            held_vqa = held_vqa.iloc[: len(held_vqa) // world * world]
    # Every global batch has the same composition, split evenly over ranks, and each rank runs
    # its driving micro-batches before its VQA ones: FSDP needs every rank to call the same
    # modules in the same order, and only driving micro-batches run the expert.
    n_vqa_step = args.vqa_per_step if vqa is not None else 0
    n_drive_step = args.global_batch - n_vqa_step
    assert n_drive_step % world == 0 and n_vqa_step % world == 0, "batch must split over ranks"
    d_rank, v_rank = n_drive_step // world, n_vqa_step // world
    accum = (math.ceil(d_rank / args.micro_batch)
             + (math.ceil(v_rank / args.micro_batch_vqa) if v_rank else 0))
    steps_per_epoch = len(train) // n_drive_step
    if vqa is not None:
        steps_per_epoch = min(steps_per_epoch, len(vqa) // n_vqa_step)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    save_steps = ({max(1, round(total * f)): f"ckpt_q{round(f * 100)}" for f in args.save_fracs}
                  if args.save_fracs else {})

    def mem(tag):  # smoke runs only: where the memory goes, for sizing the A100 run
        if args.max_steps and rank == 0:
            print(f"[mem] {tag}: {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB "
                  f"(peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f})", flush=True)

    # HSDP: cvlab21's sockets talk at ~1 GB/s vs ~5 GB/s inside one, so parameters are gathered
    # inside a socket and gradients cross it once per step (set_requires_all_reduce below)
    mesh = None
    if args.shard_group:
        mesh = init_device_mesh("cuda", (world // args.shard_group, args.shard_group),
                                mesh_dim_names=("replicate", "shard"))
    teacher = t_step = None
    if not args.teacher_targets:
        teacher = load_teacher(device, mesh)  # one model on host at a time
        mem("teacher sharded")
    student = load_student(args.student, device, offload=args.cpu_offload, mesh=mesh,
                           freeze_vlm=args.freeze_vlm)
    mem("student sharded")
    # one FSDP root per model above all shard units (see TeacherStep)
    if teacher is not None:
        t_step = TeacherStep(teacher)
        fully_shard(t_step, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                          cast_forward_inputs=False), mesh=mesh)
    # batch building reads tokenizers / action space only, which the student shares
    ref = teacher if teacher is not None else student
    s_step = StudentStep(student, freeze_vlm=args.freeze_vlm)
    fully_shard(s_step, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                      reduce_dtype=torch.float32,
                                                      cast_forward_inputs=False), mesh=mesh,
                **({"offload_policy": CPUOffloadPolicy()} if args.cpu_offload else {}))
    processor = helper.get_processor(ref.tokenizer)
    tok = ref.tokenizer
    traj_range = (ref.future_token_start_idx, ref.future_token_start_idx + ref.config.traj_vocab_size)
    vlm_p, act_p = towers(student)
    # foreach=False: the multi-tensor path allocates temporaries for every parameter at
    # once (+16 GiB peak per rank in the smoke run); per-parameter updates do not
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr,
                            betas=tuple(args.betas),
                            weight_decay=args.weight_decay, eps=1e-8, foreach=False)
    n_student = sum(p.numel() for p in student.parameters())
    step, log = 0, []
    if args.resume and (out_dir / "state").exists():
        step = load_state(student, opt, out_dir)
        log = [r for r in json.loads((out_dir / "metrics.json").read_text())["log"]
               if r["step"] <= step]
    n_missing = int(len(pd.read_parquet(REPO / args.manifest)) - len(rows))
    if rank == 0:
        (out_dir / "config.json").write_text(json.dumps({
            **vars(args), "world": world, "accum": accum, "n_train": len(train),
            "n_heldout": len(held), "n_missing": n_missing, "resumed_from": step,
            "steps_per_epoch": steps_per_epoch, "total_steps": total,
            "student_params": n_student, "gpu": torch.cuda.get_device_name(device),
            "t_grid": T_GRID, "teacher": "nvidia/Alpamayo-1.5-10B", "revision": MODEL_REV,
            "teacher_mode": "precomputed" if args.teacher_targets else "online",
            "draw": "mt_data.kd_draw(clip, t0, epoch); held-out slot 0",
            "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
            "n_vqa_train": len(vqa) if vqa is not None else 0,
            "n_vqa_heldout": len(held_vqa) if held_vqa is not None else 0,
            "per_step": {"drive": n_drive_step, "vqa": n_vqa_step}, "traj_group": list(traj_range),
            "save_steps": save_steps,
            "loss": ("MSE(v_S, v_T) only; VLM frozen, expert + action projections train"
                     if args.freeze_vlm else
                     "KL(p_T||p_S) over CoC+terminals+128 traj tokens"
                     + (" as mean(traj group) + mean(rest)" if args.kl_groups else "")
                     + (" and over VQA answer + answer_end" if vqa is not None else "")
                     + " + MSE(v_S, v_T) on driving samples; KI"),
        }, indent=2))
        print(f"{args.exp_id} | world {world} micro {args.micro_batch}/{args.micro_batch_vqa} "
              f"micro-batches/step {accum} | train {len(train)} drive + "
              f"{len(vqa) if vqa is not None else 0} vqa, held {len(held)} + "
              f"{len(held_vqa) if held_vqa is not None else 0} | {n_missing} manifest rows without "
              f"cache | {total} steps{f', resuming after {step}' if step else ''} | "
              f"student {n_student:,} params", flush=True)

    pool = ThreadPoolExecutor(4)
    drive_load = functools.partial(load_raw, tgt_dir=args.teacher_targets)
    vqa_load = functools.partial(load_vqa_raw, root=args.lingo_root)
    n_rank = d_rank + v_rank  # samples per rank per step: each micro-batch weighs its share
    t_start, stopped = time.time(), False
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + 1 + epoch).permutation(len(train))
        order_v = (np.random.default_rng(args.seed + 101 + epoch).permutation(len(vqa))
                   if vqa is not None else None)
        for g in range(steps_per_epoch):
            if step >= total:
                break
            if epoch * steps_per_epoch + g < step:  # done before --resume
                continue
            if args.deadline:  # every rank must agree, or the others wait in a collective
                late = torch.tensor(float(time.time() + 1.2 * (log[-1]["s_per_step"] if log else 0)
                                          > args.deadline), device=device)
                dist.all_reduce(late, op=dist.ReduceOp.MAX)
                if late.item():
                    stopped = True
                    break
            gidx = order[g * n_drive_step : (g + 1) * n_drive_step]
            mine = train.iloc[gidx[rank::world]]
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, args)
            t0 = time.time()
            sums = torch.zeros(3, device=device)  # driving KL, velocity MSE, VQA KL
            m = 0
            for raws in batches(mine, args.micro_batch, pool, drive_load):
                b = make_batch(ref, processor, tok, raws, args.max_coc, device)
                if args.kl_groups:
                    b["grp"] = traj_group(b, traj_range)
                if mesh is not None:  # sharded grads accumulate; replicas sync on the last
                    s_step.set_requires_all_reduce(m == accum - 1)
                l_kl, l_v = kd_losses(t_step, s_step, b, raws, epoch, train=True)
                mem(f"step {step} micro {m} after forward")
                ((l_kl + l_v) * len(raws) / n_rank).backward()
                mem(f"step {step} micro {m} after backward")
                sums[:2] += torch.stack([l_kl.detach(), l_v.detach()]) * len(raws) / d_rank
                m += 1
            if vqa is not None:
                vidx = order_v[g * n_vqa_step : (g + 1) * n_vqa_step]
                for raws in batches(vqa.iloc[vidx[rank::world]], args.micro_batch_vqa, pool,
                                    vqa_load):
                    b = make_vqa_batch(processor, tok, raws, args.max_answer, device)
                    if args.kl_groups:
                        b["grp"] = traj_group(b, traj_range)
                    if mesh is not None:
                        s_step.set_requires_all_reduce(m == accum - 1)
                    l_kl, _ = kd_losses(t_step, s_step, b, raws, epoch, train=True)
                    mem(f"step {step} micro {m} (vqa) after forward")
                    (l_kl * len(raws) / n_rank).backward()
                    sums[2] += l_kl.detach() * len(raws) / v_rank
                    m += 1
            gn_vlm = torch.nn.utils.clip_grad_norm_(vlm_p, args.clip)
            gn_act = torch.nn.utils.clip_grad_norm_(act_p, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            mem(f"step {step} after optimizer")
            step += 1
            rec = {"step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                   "kl": mean_all(sums[0]), "v_mse": mean_all(sums[1]),
                   **({"kl_vqa": mean_all(sums[2])} if vqa is not None else {}),
                   "gn_vlm": float(gn_vlm.full_tensor() if isinstance(gn_vlm, DTensor) else gn_vlm),
                   "gn_act": float(gn_act.full_tensor() if isinstance(gn_act, DTensor) else gn_act),
                   "s_per_step": time.time() - t0,
                   "peak_gb": torch.cuda.max_memory_allocated(device) / 2**30}
            if step % args.eval_every == 0 or step == total:
                h_kl, h_v, h_vqa = heldout_eval(t_step, s_step, held, held_vqa, args, ref,
                                                processor, tok, rank, world, pool, device,
                                                traj_range)
                rec.update({"held_kl": h_kl, "held_v_mse": h_v,
                            **({"held_kl_vqa": h_vqa} if h_vqa is not None else {})})
            log.append(rec)
            if rank == 0:
                print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                  for k, v in rec.items()}), flush=True)
                (out_dir / "metrics.json").write_text(json.dumps({"log": log}))
            if not args.no_save and step in save_steps:
                save_ckpt(student, args, out_dir, save_steps[step], rank)
                if step < total:
                    save_state(student, opt, out_dir, step, rank)
        if not args.no_save and not save_steps and step >= (epoch + 1) * steps_per_epoch:
            save_ckpt(student, args, out_dir, f"ckpt_epoch{epoch + 1}", rank)
        if step >= total or stopped:
            break

    if stopped:
        save_state(student, opt, out_dir, step, rank)
        if rank == 0:
            print(f"deadline: stopped after step {step}/{total}, state saved -- rerun with --resume",
                  flush=True)
    elif rank == 0:
        held = [r for r in log if "held_kl" in r]
        lines = [(f"{args.exp_id}: {step} steps in {(time.time() - t_start) / 3600:.2f} h "
                  f"(world {world}, micro {args.micro_batch}, accum {accum})"),
                 (f"train KL {log[0]['kl']:.4f} -> {log[-1]['kl']:.4f}, "
                  f"velocity MSE {log[0]['v_mse']:.4f} -> {log[-1]['v_mse']:.4f}"),
                 (f"peak memory {max(r['peak_gb'] for r in log):.1f} GiB/rank, "
                  f"{np.median([r['s_per_step'] for r in log]):.1f} s/step")]
        if held:
            lines.append(f"held-out KL {held[0]['held_kl']:.4f} -> {held[-1]['held_kl']:.4f}, "
                         f"velocity MSE {held[0]['held_v_mse']:.4f} -> {held[-1]['held_v_mse']:.4f}")
            if "held_kl_vqa" in held[0]:
                lines.append(f"held-out VQA KL {held[0]['held_kl_vqa']:.4f} -> "
                             f"{held[-1]['held_kl_vqa']:.4f}")
        if "kl_vqa" in log[0]:
            lines.append(f"train VQA KL {log[0]['kl_vqa']:.4f} -> {log[-1]['kl_vqa']:.4f}")
        (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
