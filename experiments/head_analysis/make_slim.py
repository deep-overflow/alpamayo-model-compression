"""Build a physically-slimmed Alpamayo-1.5 checkpoint from a validated mask config.

Configs:
  integrated_mag   -- VLM graded 30/50 + layer35(kv-only) + KV1 (trajectory Taylor),
                      expert early40/late10 (magnitude). Expected -3.25B.
  cocsafe_full_r20 -- VLM dual max(rank_traj, rank_coc) 20% + KV1(dual),
                      expert early40/late10 (magnitude). Expected -2.01B.
  *_u40_v2         -- the one-factor family, all at the grid's dual_uniform cell
                      (uniform matched budget, expert untouched, no KV drop); only the
                      within-layer score differs. Combined: dual = max(rank_traj,
                      rank_coc), j_traj = max(rank_traj, rank_J). Single-criterion
                      controls: traj, coc, j; wanda / wandatxt (gradient-free
                      |W|*||X||_2 baseline, all tokens / text+CoC tokens).
                      Operator ablation: dualsum / dualprod
                      keep dual's halves but combine by rank-sum / rank-product.
                      Expected -2.66B each.
  tyr_u40 / tyr_uniform_u40 -- Tyr-the-Pruner baseline: OSSCAR-reconstructed
                      supernet weights at the searched / uniform level assignment,
                      same -2.66B budget by construction.
  dual2nd_u40_v2   -- dual with the DIAGONAL SECOND-ORDER score: mean_clips (dL/dg)^2
                      instead of mean_clips |dL/dg| (LLM-Pruner's `param_second` on our
                      activation gates). Same budget, allocation and axes.
  dualscope_u40    -- dual's ranking and dual_u40_v2's exact budget, but confined to
                      LLM-Pruner's layer scope (4..34); only WHERE the cut lands
                      differs. Expected -2.66B.
  expert_u<N>      -- expert tower only, uniform N%, VLM and KV untouched; the
                      within-layer score is the importance file's traj_exp_* (so the
                      step aggregation is chosen by --importance). Expected -0.53B at
                      N=25.
  dualexp_u40_e<N> -- dual_u40_v2's VLM half + expert_u<N>'s expert half in one
                      checkpoint (first config pruning both towers). Expected -3.19B
                      at N=25.
  znorm11_u40_v2   -- one score from ELEVEN losses: CoC NLL + the ten flow-matching step
                      losses, each z-scored within a layer and averaged (1/11 each). Needs
                      --stepvlm (per-step VLM gradients). Kept set overlaps dual's by
                      87-88%. plans/2026-08-31_znorm11-criterion.md.
  dualfix_u40_v2   -- dual with the degenerate-layer guard: a layer whose half is constant
                      (the last layer's trajectory importance is structurally zero) no
                      longer contributes its INDEX ORDER to max(rank, rank).
  maxstep11_u40_v2 -- the same ELEVEN losses as znorm11, combined with the UNION operator
                      instead of the mean: max over the eleven within-layer ranks. Isolates
                      "per-step trajectory scores" from "mean instead of max", which znorm11
                      changed together. Carries dualfix's guard. Needs --stepvlm.
  meandual_u40_v2  -- dual's two halves under znorm11's operator: mean of z(I_traj) and
                      z(I_CoC). The fourth cell of the 2x2 (operator x step axis).
                      plans/2026-09-03_union-step-criterion.md.
  dualexp_u40_em<M> -- the same VLM half + expert MLP-ONLY at M% (expert Q heads and KV
                      untouched); M may carry a decimal written with `p` (em93p75 =
                      93.75%). The expert score comes from --expert-importance, so the
                      kept set matches the dualrc_u40_s<N>_em<M> ladder unit for unit.
                      plans/2026-08-31_dualrwl-expert-mlp.md.
  dualrc_u40_s<N>  -- dual_u40_v2's selection everywhere + cache-targeted OSSCAR refit of
                      o_proj / down_proj in layers >= N (run_cache_recon.py supernet via
                      --tyr-supernet; expert-attention-weighted prefill + own-CoC Hessian).
                      Needs slim_state.pt. plans/2026-08-29_cache-targeted-reconstruction.md.
  dualrc_u40_s<N>_em<M>  the same VLM half plus expert MLP-only pruning at M% (expert Q
                      heads and KV untouched), i.e. the union of the reconstructed VLM and
                      expertm_u<M>; the two halves' removed params add. M may carry a
                      decimal written with `p` (em87p5 = 87.5%).
                      plans/2026-08-31_dualrwl-expert-mlp.md.
  expert{q,m}_u<N> -- ONE expert axis only: q = Q heads, m = MLP channels, N% of that
  expert{q,m}_c<N>    axis per layer (u) or exactly N units per layer (c, the
                      parameter-matched control); VLM, KV and the other axis untouched.
                      Same score and selection as expert_u<N>, so
                      expertq_u25 | expertm_u25 == expert_u25 unit for unit.
                      plans/2026-08-28_expert-axis-ablation.md.
  dual{q,m}_u40_v2 -- the same one-axis-at-a-time decomposition on the VLM, at
  dualm_c<N>          dual_u40_v2's exact budget and score; dualq | dualm ==
                      dual_u40_v2 unit for unit, and dualm_c1109 is the
                      parameter-matched control for dualq (0.03%).
                      plans/2026-08-30_axis-taylor-comparability.md.

The mask recipes are imported from run_integrated / run_cocsafe -- no duplicated math.
Writes slim_state.pt + slim_meta.json to --out, then smoke-tests one val clip
(short rollout, teacher-forced NLL, one denoise, cache-shape assert).

Usage:
  bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/make_slim.py \
      --config integrated_mag --gpu 4 --out outputs/slim_integrated_mag
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import analysis_lib as lib  # noqa: E402
import mask_lib as ml  # noqa: E402
import sample_cache as sc  # noqa: E402
import slim_lib as sl  # noqa: E402
import tyr_lib as tyr  # noqa: E402
from expert_per_clip import reserve_gpu  # noqa: E402  also installs the gated-repo hub patch
from run_cocsafe import rank_norm  # noqa: E402
from run_eval import eval_config  # noqa: E402
from run_grid import allocations, grid_configs  # noqa: E402
from run_integrated import expert_masks, vlm_combined_masks  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# pinned to the blobs every result in this track was produced with, matching run_baseline
MODEL_REV = sl.MODEL_REV  # single source, so a build and a load can never drift apart


def scope_matched_counts(scope_len, target_removed, tc, ratio):
    """Per-layer cut counts that fit `target_removed` params into `scope_len` layers.

    Narrowing the scope while holding the budget means cutting deeper in the layers that
    remain: the head count follows the depth-scaled ratio, and the MLP count is solved so
    the total lands on `target_removed` to the parameter -- an approximate budget would
    stop the comparison against dual_u40_v2 from being one-factor. The leftover channels
    are spread one per layer over the front of the scope, deterministically.

    Returns (heads_cut, mlp_base, n_layers_with_one_extra).
    """
    attn_cost = 2 * tc.hidden_size * tc.head_dim          # q_proj rows + o_proj cols
    mlp_cost = 3 * tc.hidden_size                          # gate/up rows + down col
    heads_cut = round(ratio * tc.num_attention_heads)
    rest = target_removed - scope_len * heads_cut * attn_cost
    ch_total, rem = divmod(rest, mlp_cost)
    assert rem == 0 and 0 < ch_total < scope_len * tc.intermediate_size, (ch_total, rem)
    base, extra = divmod(ch_total, scope_len)
    return heads_cut, base, extra


def build_masks(cfg_name, imp, model, jlens="jlens_v2", vqa_imp="importance_vqa",
                stepvlm="importance_stepvlm_v1",
                wanda_run="wanda_v1", wanda_txt_run="wanda_txt_v1",
                tyr_supernet="tyr_supernet_u40",
                tyr_config="tyr_search_u40/final_config.json", scope=(4, 34),
                imp_run="importance_v2", cache_imp="cachejlens_v1",
                expert_imp="importance_stepexp_znorm"):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    emag = ml.magnitude_scores(model.expert.layers, ec.num_attention_heads, ec.head_dim,
                               ec.intermediate_size)
    eq, em = expert_masks(imp, emag, ec.num_hidden_layers, "magnitude")
    it = re.match(r"^(.+)_u40_it(\d+)$", cfg_name)
    # the optional _qcut<N> trades Q heads for MLP channels at the SAME parameter
    # budget: maxstep11_u40_qcut4_v2 cuts 4 heads per layer instead of 13 and puts
    # the difference into channels (plans/2026-09-05_axis-allocation.md)
    uni = re.match(r"^(.+)_u(\d+)(?:_qcut(\d+))?_v2$", cfg_name)
    exp_only = re.match(r"^expert_u(\d+)$", cfg_name)
    dualexp = re.match(r"^dualexp_u40_e(\d+)$", cfg_name)
    dualexp_m = re.match(r"^dualexp_u40_em(\d+(?:p\d+)?)$", cfg_name)
    axis = re.match(r"^expert([qm])_([uc])(\d+)$", cfg_name)
    vaxis = re.match(r"^dual([qm])_u40_v2$|^dualm_c(\d+)$", cfg_name)
    dualrc = re.match(r"^dualrc_u40_s(\d+)(?:_em(\d+(?:p\d+)?))?$", cfg_name)
    if vaxis:
        # The VLM twin of the expert-axis decomposition
        # (plans/2026-08-30_axis-taylor-comparability.md). dual_u40_v2's own masks are
        # already axis-separable -- vq and vm come from two independent
        # select_mask_ratios calls -- so dualq | dualm == dual_u40_v2 unit for unit and
        # the shipped dual_u40_v2 runs serve as the additivity arm at no cost.
        # Everything but the axis is held: same dual score, same 0.3985632694 budget,
        # expert and KV untouched. `dualm_c<N>` cuts exactly N channels per layer, the
        # parameter-matched control for dualq (1109 ch = 13 heads to 0.03%).
        # Must be matched before `uni`, whose (.+)_u(\d+)_v2 also accepts dualq_u40_v2.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                tc.num_attention_heads, tc.intermediate_size, 0.5)
        rq, rm = allocs["uniform"]  # the matched 0.3985632694, never 0.40
        sq, sm = tyr.dual_scores(imp)
        which = vaxis.group(1)
        vq = np.ones((tc.num_hidden_layers, tc.num_attention_heads))  # (36, 32)
        vm = np.ones((tc.num_hidden_layers, tc.intermediate_size))  # (36, 12288)
        if which == "q":
            vq = ml.select_mask_ratios(sq, rq)
        elif which == "m":
            vm = ml.select_mask_ratios(sm, rm)
        else:  # dualm_c<N>: N channels per layer, same score and per-layer rule
            n_ch = int(vaxis.group(2))
            vm = ml.select_mask_ratios(
                sm, np.full(tc.num_hidden_layers, n_ch / tc.intermediate_size))
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif dualrc:
        # Cache-targeted reconstruction (plans/2026-08-29_cache-targeted-reconstruction.md):
        # dual_u40_v2's selection in every layer, and run_cache_recon.py's refitted
        # o_proj / down_proj written into layers >= s<N> only. The refit supernet is read
        # from --tyr-supernet; its masks derived from the exactly-zero columns must equal
        # dual's, so the usual surgery slices the reconstructed values (state required).
        start = int(dualrc.group(1))
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                tc.num_attention_heads, tc.intermediate_size, 0.5)
        rq, rm = allocs["uniform"]
        sq, sm = tyr.dual_scores(imp)
        vq = ml.select_mask_ratios(sq, rq)  # (36, 32)
        vm = ml.select_mask_ratios(sm, rm)  # (36, 12288)
        sup = REPO / "outputs" / tyr_supernet
        smeta = json.loads((sup / "metadata.json").read_text())
        assert smeta.get("selection") == "dual", smeta.get("selection")
        vlayers = model.vlm.model.language_model.layers
        nh, hd = tc.num_attention_heads, tc.head_dim
        written = 0
        for n in smeta["layer_names"]:
            i = int(n.split(".")[1])
            if i < start:
                continue
            mod = (vlayers[i].mlp.down_proj if "mlp" in n else vlayers[i].self_attn.o_proj)
            w = torch.load(sup / n / "0.pth", map_location="cuda")
            col = (w.abs().sum(0).float() > 0).cpu().numpy().astype(float)
            derived = col.reshape(nh, hd).max(1) if "mlp" not in n else col
            assert np.array_equal(derived, vm[i] if "mlp" in n else vq[i]), n
            mod.weight.data.copy_(w.to(mod.weight.dtype))
            written += 1
        assert written == 2 * (tc.num_hidden_layers - start), (written, start)
        print(f"dualrc: refitted weights written into layers {start}..{tc.num_hidden_layers - 1} "
              f"({written} modules) from {tyr_supernet}", flush=True)
        eq, em = np.ones_like(eq), np.ones_like(em)
        if dualrc.group(2):
            # expert MLP-only on top of the reconstructed VLM: Q heads, KV and head_dim stay
            # whole, so the expert kept set is bit-identical to expertm_u<M> and the two
            # halves' removed-parameter counts simply add (different matrices). The expert
            # score comes from its OWN run -- the axis ablation used the step-normalised
            # aggregation while the VLM half needs importance_v2's dual keys, so two
            # importance files are live at once. plans/2026-08-31_dualrwl-expert-mlp.md.
            ez = dict(np.load(REPO / "outputs" / expert_imp / "importance.npz"))
            # `p` stands in for the decimal point: em87p5 = 87.5%, which keeps 1032 of
            # 8256 channels and continues the halving ladder past the integer grid
            pct = float(dualrc.group(2).replace("p", "."))
            em = ml.select_mask(ez["traj_exp_mlp"], pct / 100,
                                list(range(ec.num_hidden_layers)))  # (36, 8256)
        kvonly = ()
    elif axis:
        # One expert axis at a time (plans/2026-08-28_expert-axis-ablation.md): does the
        # expert's cost come from the step-specialised Q heads or from the parameter
        # mass in the MLP? `u<N>` cuts N% of the chosen axis per layer, `c<N>` exactly N
        # units per layer (expertm_c341 removes what expertq_u25 removes, to 0.1%).
        which, mode, n = axis.group(1), axis.group(2), int(axis.group(3))
        all_e = list(range(ec.num_hidden_layers))
        n_units = ec.num_attention_heads if which == "q" else ec.intermediate_size
        ratio = n / 100 if mode == "u" else n / n_units  # select_mask rounds n_units*ratio
        eq = np.ones((ec.num_hidden_layers, ec.num_attention_heads))  # (36, 16)
        em = np.ones((ec.num_hidden_layers, ec.intermediate_size))  # (36, 8256)
        if which == "q":
            eq = ml.select_mask(imp["traj_exp_q"], ratio, all_e)
        else:
            em = ml.select_mask(imp["traj_exp_mlp"], ratio, all_e)
        vq = np.ones((tc.num_hidden_layers, tc.num_attention_heads))
        vm = np.ones((tc.num_hidden_layers, tc.intermediate_size))
        kvonly = ()
    elif exp_only:
        # Expert tower only, uniform ratio, VLM and KV untouched -- the shape the D-stage
        # aggregation arms were measured in (run_expert_agg.py evaluated exactly this as a
        # runtime mask). The within-layer score is whatever `traj_exp_*` the importance file
        # carries, so the aggregation is selected by --importance rather than by a new stem:
        #   --importance importance_v2_ada        the shipped |sum_s| rule
        #   --importance importance_stepexp_znorm the step-normalised one
        # This exists so a mask-level open-loop result can be carried into alpasim, which
        # needs a real slim_state.pt and cannot take runtime masks.
        ratio = int(exp_only.group(1)) / 100
        all_e = list(range(ec.num_hidden_layers))
        eq = ml.select_mask(imp["traj_exp_q"], ratio, all_e)
        em = ml.select_mask(imp["traj_exp_mlp"], ratio, all_e)
        vq = np.ones((tc.num_hidden_layers, tc.num_attention_heads))
        vm = np.ones((tc.num_hidden_layers, tc.intermediate_size))
        kvonly = ()
    elif dualexp:
        # dual_u40_v2's VLM half + expert_u<N>'s expert half, verbatim: the first config
        # that prunes both towers with their individually-validated recipes. Gate G0
        # requires the VLM kept sets bit-identical to slim_dual_u40_v2 (so --importance
        # must carry importance_v2's VLM keys, i.e. the Blackwell-anchored drop-in) and
        # the expert kept sets bit-identical to slim_expert_znorm_r25.
        # plans/2026-08-26_dual-plus-znorm.md.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                tc.num_attention_heads, tc.intermediate_size, 0.5)
        rq, rm = allocs["uniform"]  # the matched 0.3985632694, never 0.40
        sq, sm = tyr.dual_scores(imp)
        vq = ml.select_mask_ratios(sq, rq)  # (36, 32)
        vm = ml.select_mask_ratios(sm, rm)  # (36, 12288)
        ratio = int(dualexp.group(1)) / 100
        all_e = list(range(ec.num_hidden_layers))
        eq = ml.select_mask(imp["traj_exp_q"], ratio, all_e)  # (36, 16)
        em = ml.select_mask(imp["traj_exp_mlp"], ratio, all_e)  # (36, 8256)
        kvonly = ()
    elif dualexp_m:
        # dual_u40_v2's VLM half + expert MLP-only. The expert Q heads carry the whole
        # cost of an expert cut (reports/evaluation/2026-08-28_expert-axis.html), so this
        # leaves them whole and takes only the width, which the dualr_wl ladder showed is
        # free to 516 channels open- and closed-loop. Same budget as dualrc_u40_s0_em<M>,
        # so the two differ only in the VLM half -- a one-factor test of whether the
        # expert cut transfers to dual's (unrewritten) selection.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                tc.num_attention_heads, tc.intermediate_size, 0.5)
        rq, rm = allocs["uniform"]  # the matched 0.3985632694, never 0.40
        sq, sm = tyr.dual_scores(imp)
        vq = ml.select_mask_ratios(sq, rq)  # (36, 32)
        vm = ml.select_mask_ratios(sm, rm)  # (36, 12288)
        ez = dict(np.load(REPO / "outputs" / expert_imp / "importance.npz"))
        pct = float(dualexp_m.group(1).replace("p", "."))
        eq = np.ones((ec.num_hidden_layers, ec.num_attention_heads))  # (36, 16)
        em = ml.select_mask(ez["traj_exp_mlp"], pct / 100,
                            list(range(ec.num_hidden_layers)))  # (36, 8256)
        kvonly = ()
    elif it:
        # Staged re-calibration masks from run_iter_prune.py: same budget, allocation
        # and axes as the *_u40_v2 family (verified by its R0 gate), only the score
        # measurement schedule differs. The masks are precomputed, so this branch just
        # loads them; plans/2026-08-16_iterative-recalibration.md.
        z = np.load(REPO / "outputs" / f"iter_{it.group(1)}_u40" / "final_masks.npz")
        vq, vm = z["vq"], z["vm"]
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name == "dualscope_u40":
        # Same criterion, calibration, axes and budget as dual_u40_v2 -- only the layer
        # scope changes, from all 36 layers to LLM-Pruner's 4..34 (its
        # --block_attention_layer_start/end convention, which leaves the first four and
        # last two layers intact). The closed-loop split that motivates this is in
        # plans/2026-08-22_scope-matched-dual.md: at a matched budget our 36-layer cut
        # halves collisions but deviates further from the GT path (d2gt 3.32 vs 2.88),
        # and the layer scope is the one structural difference left to test.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                tc.num_attention_heads, tc.intermediate_size, 0.5)
        rq_u, rm_u = allocs["uniform"]
        attn_cost = 2 * tc.hidden_size * tc.head_dim
        mlp_cost = 3 * tc.hidden_size
        # dual_u40_v2's realized budget, from the same source the u40 family draws on
        target = tc.num_hidden_layers * (
            round(float(rq_u[0]) * tc.num_attention_heads) * attn_cost
            + round(float(rm_u[0]) * tc.intermediate_size) * mlp_cost)
        layers_in = list(range(scope[0], scope[1]))
        depth_ratio = float(rq_u[0]) * tc.num_hidden_layers / len(layers_in)
        heads_cut, mlp_base, extra = scope_matched_counts(
            len(layers_in), target, tc, depth_ratio)
        rq = np.zeros(tc.num_hidden_layers)
        rm = np.zeros(tc.num_hidden_layers)
        for k, li in enumerate(layers_in):
            rq[li] = heads_cut / tc.num_attention_heads
            rm[li] = (mlp_base + (1 if k < extra else 0)) / tc.intermediate_size
        sq, sm = tyr.dual_scores(imp)
        vq = ml.select_mask_ratios(sq, rq)
        vm = ml.select_mask_ratios(sm, rm)
        cut_q = int((1.0 - vq).sum())
        cut_m = int((1.0 - vm).sum())
        removed = cut_q * attn_cost + cut_m * mlp_cost
        assert removed == target, (removed, target)
        assert vq[: scope[0]].all() and vq[scope[1]:].all(), "cut leaked outside the scope"
        print(f"scope {scope[0]}..{scope[1]} ({len(layers_in)} layers): "
              f"{heads_cut}/{tc.num_attention_heads} heads and "
              f"{mlp_base}(+1 x {extra})/{tc.intermediate_size} channels per layer, "
              f"removing {removed:,} == dual_u40_v2", flush=True)
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name.startswith("dualg_u40"):
        # dual-global: dual ranking within layers, per-layer cut counts from the searched
        # level vector (plans/2026-08-21_dual-global.md); original weights, masks only
        cfg_path = REPO / "outputs" / tyr_config
        levels = json.loads(cfg_path.read_text())
        mm = json.loads((cfg_path.parent / "mask_meta.json").read_text())
        sq, sm = tyr.dual_scores(imp)
        kq = tyr.level_keeps(tc.num_attention_heads, mm["head_cut"], mm["head_step"],
                             mm["num_levels"])
        km = tyr.level_keeps(tc.intermediate_size, mm["mlp_cut"], mm["mlp_step"],
                             mm["num_levels"])
        vq = np.ones((tc.num_hidden_layers, tc.num_attention_heads))
        vm = np.ones((tc.num_hidden_layers, tc.intermediate_size))
        for n, lv in levels.items():
            i = int(n.split(".")[1])
            if "mlp" in n:
                vm[i, tyr.cut_lowest(sm[i], tc.intermediate_size - km[lv])] = 0.0
            else:
                vq[i, tyr.cut_lowest(sq[i], tc.num_attention_heads - kq[lv])] = 0.0
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name.startswith(("tyr_u40", "tyr_uniform_u40", "tyr_sel_u40",
                              "dualr_u40", "dualgr_u40")):
        # Tyr baseline (plans/2026-08-20_tyr-baseline.md). The supernet stores
        # OSSCAR-reconstructed o_proj / down_proj weights per sparsity level; this
        # branch WRITES the chosen level's weights into the model and derives the
        # 0/1 masks from their exactly-zero columns, so the usual surgery slices
        # the reconstructed values. tyr_uniform_u40 = level 0 everywhere (uniform
        # allocation + reconstruction); tyr_u40 = the searched distribution. Both
        # keep u40_v2's removed-parameter total by type-conserving levels.
        sup = REPO / "outputs" / tyr_supernet
        smeta = json.loads((sup / "metadata.json").read_text())
        levels = {n: 0 for n in smeta["layer_names"]}
        uniform_levels = cfg_name.startswith(("tyr_uniform", "tyr_sel")) or cfg_name == "dualr_u40"
        if not uniform_levels:
            levels = json.loads((REPO / "outputs" / tyr_config).read_text())
        # tyr_sel_u40: Tyr's OSSCAR *selection* at level 0 with the ORIGINAL weights
        # (no reconstruction) -- separates "which units" from "how the rest is rewritten"
        write_weights = not cfg_name.startswith("tyr_sel")
        vlayers = model.vlm.model.language_model.layers
        nh, hd = tc.num_attention_heads, tc.head_dim
        vq = np.ones((tc.num_hidden_layers, nh))
        vm = np.ones((tc.num_hidden_layers, tc.intermediate_size))
        for n, lv in levels.items():
            i = int(n.split(".")[1])
            mod = (vlayers[i].mlp.down_proj if "mlp" in n
                   else vlayers[i].self_attn.o_proj)
            w = torch.load(sup / n / f"{lv}.pth", map_location="cuda")
            if write_weights:
                mod.weight.data.copy_(w.to(mod.weight.dtype))
            col = w.abs().sum(0).float()
            if "mlp" in n:
                vm[i] = (col > 0).cpu().numpy().astype(float)
            else:
                vq[i] = (col.reshape(nh, hd).sum(1) > 0).cpu().numpy().astype(float)
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif uni:
        # The one-factor family. Everything is held at the grid's dual_uniform cell --
        # uniform allocation, expert untouched, no KV drop -- so the only things that
        # vary are the within-layer score and, across the ratio sweep, the budget.
        # `dual`/`j_traj` are the combined criteria max(rank I_traj, rank X);
        # `traj`/`coc`/`j` are the single-criterion controls that say what each half of
        # that max() does on its own.
        stem, pct = uni.group(1), int(uni.group(2))
        if pct == 40:
            # u40 is NOT 0.40: it is the matched target 0.3985632694 that
            # run_grid.allocations() derives from slim_integrated_mag's realized budget.
            # Rounding it to 0.40 moves 17 MLP channels per layer, so the shipped
            # checkpoints would no longer regenerate bit-identically.
            ref_meta = json.loads(
                (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
            allocs, _ = allocations(imp, ref_meta, tc.num_hidden_layers,
                                    tc.num_attention_heads, tc.intermediate_size, 0.5)
            rq, rm = allocs["uniform"]
        else:
            # the sweep points mean exactly what their name says
            rq = np.full(tc.num_hidden_layers, pct / 100)
            rm = np.full(tc.num_hidden_layers, pct / 100)

        if uni.group(3) is not None:
            # Same removed-parameter total, different split between the axes. One Q head
            # costs 2*head_dim*hidden (q_proj rows + o_proj cols); one MLP channel costs
            # 3*hidden (gate/up rows + down col), so a head is worth 85.33 channels here.
            # The channel count is DERIVED from whatever budget rq/rm just set rather than
            # written down, and the assert refuses anything that does not divide evenly --
            # a config that is "almost" the same size would silently stop being a
            # one-factor comparison, which is the whole point of this arm.
            n_q = int(uni.group(3))
            attn_cost = 2 * tc.head_dim * tc.hidden_size
            mlp_cost = 3 * tc.hidden_size
            per_layer = (round(rq[0] * tc.num_attention_heads) * attn_cost
                         + round(rm[0] * tc.intermediate_size) * mlp_cost)
            n_m, rem = divmod(per_layer - n_q * attn_cost, mlp_cost)
            assert rem == 0, (
                f"_qcut{n_q} leaves {rem} parameters over; only cuts that divide evenly "
                f"keep the budget identical")
            assert 0 <= n_q < tc.num_attention_heads and 0 < n_m < tc.intermediate_size
            rq = np.full(tc.num_hidden_layers, n_q / tc.num_attention_heads)
            rm = np.full(tc.num_hidden_layers, n_m / tc.intermediate_size)
            print(f"qcut{n_q}: cut {n_q}/{tc.num_attention_heads} heads and "
                  f"{n_m}/{tc.intermediate_size} channels per layer "
                  f"({per_layer:,} params, unchanged)", flush=True)

        def half(name):
            if name == "traj":
                return imp["traj_vlm_q"], imp["traj_vlm_mlp"]
            if name == "coc":
                return imp["coc_vlm_q"], imp["coc_vlm_mlp"]
            if name in ("vqa", "coclingo"):
                # VQA-context importance and its same-images CoC control come from their
                # own run: they are measured on LingoQA train, which ships no ego
                # trajectory, so that npz holds only the two VLM arrays per objective and
                # cannot supply traj / KV / expert scores. Those still come from `imp`.
                z = dict(np.load(REPO / "outputs" / vqa_imp / "importance.npz"))
                pre = "vqa" if name == "vqa" else "coc"
                return z[f"{pre}_vlm_q"], z[f"{pre}_vlm_mlp"]
            if name in ("wanda", "wandatxt"):
                # gradient-free baseline: |W| * ||X||_2 aggregated per unit, official
                # WrappedGPT stats (plans/2026-08-20_wanda-baseline.md). `wandatxt`
                # accumulates ||X|| over text + own-CoC tokens only (--tokens text)
                run = wanda_run if name == "wanda" else wanda_txt_run
                z = dict(np.load(REPO / "outputs" / run / "wanda.npz"))
                return z["q_w"], z["mlp_w"]
            if name in ("traj2", "coc2"):
                # Diagonal second-order (empirical Fisher) of the same gate: the stored
                # per-clip arrays are |dL/dg|, so squaring them gives (dL/dg)^2 exactly.
                # First order takes mean|g| (abs inside the clip sum); this takes mean g^2,
                # which up-weights units whose damage concentrates in a few clips.
                z = dict(np.load(REPO / "outputs" / imp_run / "importance_perclip.npz"))
                pre = name[:-1]
                return ((z[f"{pre}_vlm_q"] ** 2).mean(0),
                        (z[f"{pre}_vlm_mlp"] ** 2).mean(0))
            if name == "znorm11":
                # CoC NLL + the ten flow-matching step losses, each z-scored WITHIN a layer
                # and averaged with weight 1/11. The per-step VLM gradients come from
                # run_step_importance_vlm (step_importance_vlm.npz); the CoC half is the
                # reference file's, so both halves are the same clips and architecture.
                # plans/2026-08-31_znorm11-criterion.md.
                st = np.load(REPO / "outputs" / stepvlm / "step_importance_vlm.npz")

                def zs(x):  # (L, U) -> within-layer z-score
                    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-12)

                zq = sum(zs(st["q_abs_step"][i]) for i in range(st["q_abs_step"].shape[0]))
                zm = sum(zs(st["mlp_abs_step"][i]) for i in range(st["mlp_abs_step"].shape[0]))
                n = st["q_abs_step"].shape[0] + 1
                return (zq + zs(imp["coc_vlm_q"])) / n, (zm + zs(imp["coc_vlm_mlp"])) / n
            if name in ("max11", "meandual"):
                # The 2x2 that znorm11 collapsed into one move. Both read the same eleven
                # losses' materials as znorm11 -- the ten per-step flow-matching gradients
                # from run_step_importance_vlm plus the reference file's CoC NLL -- and
                # differ only in how they are combined:
                #   max11    max over the eleven WITHIN-LAYER RANKS (union, like dual)
                #   meandual mean of z(I_traj) and z(I_CoC), i.e. dual's two halves under
                #            znorm11's operator
                # sum_s(q_abs_step) reproduces the summed traj_vlm_* it factorises to
                # within-layer Spearman 0.99, so the two axes really are separable.
                # plans/2026-09-03_union-step-criterion.md.
                def zs(x):  # (L, U) -> within-layer z-score
                    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-12)

                if name == "meandual":
                    return ((zs(imp["traj_vlm_q"]) + zs(imp["coc_vlm_q"])) / 2,
                            (zs(imp["traj_vlm_mlp"]) + zs(imp["coc_vlm_mlp"])) / 2)
                st = np.load(REPO / "outputs" / stepvlm / "step_importance_vlm.npz")

                def guarded_rank(x):
                    # a layer whose scores are all equal must not contribute its index
                    # order; the last layer is structurally constant on every FM step
                    r = rank_norm(x)
                    r[np.ptp(x, axis=1) == 0] = -np.inf
                    return r

                nq = st["q_abs_step"].shape[0]
                sq = np.maximum.reduce([guarded_rank(st["q_abs_step"][i]) for i in range(nq)]
                                       + [guarded_rank(imp["coc_vlm_q"])])
                sm = np.maximum.reduce([guarded_rank(st["mlp_abs_step"][i]) for i in range(nq)]
                                       + [guarded_rank(imp["coc_vlm_mlp"])])
                return sq, sm
            if name == "cache":
                # cache-Jacobian importance (plans/2026-08-30_cache-jlens-criterion.md):
                # E ||d cache / d g||^2 weighted by the expert's per-(layer, group)
                # sensitivity, from run_cache_jlens.py. Label-free and second-order, so it
                # is not an |dL/dg| Taylor score; it enters through rank_norm like the rest.
                z = dict(np.load(REPO / "outputs" / cache_imp / "importance.npz"))
                return z["cache_vlm_q"], z["cache_vlm_mlp"]
            jl = dict(np.load(REPO / "outputs" / jlens / "jlens.npz"))
            return jl["q_j"], jl["mlp_j"]

        parts = {"dual": ("traj", "coc"), "dualfix": ("traj", "coc"),
                 "maxstep11": ("max11",),
                 "dual2nd": ("traj2", "coc2"),
                 "j_traj": ("traj", "j"),
                 "trajvqa": ("traj", "vqa"), "dualsum": ("traj", "coc"),
                 "dualprod": ("traj", "coc"),
                 "cachedual": ("cache", "coc"), "cacheonly": ("cache",)}.get(stem, (stem,))
        # dualsum/dualprod are the operator ablation: same halves as dual, only the
        # combination differs (plans/2026-08-20_combination-operator-ablation.md)
        op = {"dualsum": np.add, "dualprod": np.multiply}.get(stem, np.maximum)
        # select_mask_ratios ranks within a layer, so rank_norm is a no-op for a single
        # criterion -- it only matters when two scores have to share one scale
        def rank_or_nan(x):
            """rank_norm, except a layer whose scores are all equal contributes nothing.

            The VLM's LAST layer has trajectory importance identically zero -- its o_proj /
            down_proj outputs never reach the KV cache the expert reads -- and
            rank_norm(zeros) is the INDEX ORDER, so `dualfix` avoids letting
            max(rank traj, rank coc) keep the highest-numbered units there
            (74% of dual's layer-35 Q keeps are just high indices).
            plans/2026-08-31_znorm11-criterion.md.
            """
            r = rank_norm(x)
            flat = np.ptp(x, axis=1) == 0
            r[flat] = -np.inf
            return r

        rank = rank_or_nan if stem.startswith("dualfix") else rank_norm
        sq, sm = half(parts[0])
        for p in parts[1:]:
            oq, om = half(p)
            sq = op(rank(sq), rank(oq))
            sm = op(rank(sm), rank(om))
        vq = ml.select_mask_ratios(sq, rq)
        vm = ml.select_mask_ratios(sm, rm)
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name == "dual_uniform":
        # the grid's practical winner: dual criterion, uniform layerwise budget, VLM only.
        # Masks come straight from grid_configs so the closed-loop checkpoint is bit-identical
        # to the cell that was evaluated open-loop; expert and KV are deliberately untouched
        # so this tests exactly one factor combination and nothing else.
        ref_meta = json.loads(
            (REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
        cfgs, _ = grid_configs(imp, ref_meta, tc.num_hidden_layers, tc.num_attention_heads,
                               tc.intermediate_size, 0.5)
        vq, vm = next((q, m) for n, _, q, m in cfgs if n == "dual_uniform")
        eq, em = np.ones_like(eq), np.ones_like(em)
        kvonly = ()
    elif cfg_name == "integrated_mag":
        vq, vm = vlm_combined_masks(imp, tc.num_hidden_layers, tc.num_attention_heads,
                                    tc.intermediate_size)
        kvonly = (tc.num_hidden_layers - 1,)
    elif cfg_name.startswith("j_traj_full"):
        # Label-free twin of cocsafe_full: identical structure, identical ratio, identical
        # expert/KV axes -- only the reasoning half of the criterion changes, from the CoC
        # NLL Taylor score to the J-lens score. That makes the closed-loop comparison a
        # one-factor test of "do we still need CoC reference text?".
        ratio = 0.30 if cfg_name.endswith("r30") else 0.20
        all_l = list(range(tc.num_hidden_layers))
        jl = dict(np.load(REPO / "outputs" / "jlens_coc" / "jlens.npz"))
        n_per_group = tc.num_attention_heads // tc.num_key_value_heads  # GQA: 4
        # the J-lens scores Q heads and MLP channels but not KV groups, so a group's
        # J-mass is the summed squared J-score of the Q heads it feeds (group h ->
        # VLM Q heads [4h, 4h+4)) -- the same coupling mask_lib uses to remove a group
        j_kv = (jl["q_j"] ** 2).reshape(tc.num_hidden_layers, tc.num_key_value_heads,
                                        n_per_group).sum(-1)
        dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(jl["q_j"]))
        dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(jl["mlp_j"]))
        dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                             rank_norm(j_kv))
        vq = ml.select_mask(dual_q, ratio, all_l) * ml.kv_group_mask(
            dual_kv, 1, tc.num_attention_heads, all_l)
        vm = ml.select_mask(dual_m, ratio, all_l)
        kvonly = ()
    else:  # cocsafe_full_r20 / cocsafe_full_r30 -- dual width r% + KV1(dual) + expert magnitude
        ratio = 0.30 if cfg_name == "cocsafe_full_r30" else 0.20
        all_l = list(range(tc.num_hidden_layers))
        dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
        dual_m = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
        dual_kv = np.maximum(rank_norm(imp["traj_kv_k"] + imp["traj_kv_v"]),
                             rank_norm(imp["coc_kv_k"] + imp["coc_kv_v"]))
        vq = ml.select_mask(dual_q, ratio, all_l) * ml.kv_group_mask(
            dual_kv, 1, tc.num_attention_heads, all_l)
        vm = ml.select_mask(dual_m, ratio, all_l)
        kvonly = ()
    return vq, vm, eq, em, kvonly


def expected_removed(model, vq, vm, eq, em, kvonly):
    tc = model.vlm.config.text_config
    ec = model.expert.config
    p_vq = 2 * tc.hidden_size * tc.head_dim
    p_vm = 3 * tc.hidden_size
    p_eq = 2 * ec.hidden_size * ec.head_dim
    p_em = 3 * ec.hidden_size
    removed = (int((vq == 0).sum()) * p_vq + int((vm == 0).sum()) * p_vm
               + int((eq == 0).sum()) * p_eq + int((em == 0).sum()) * p_em)
    # kv-only layers also lose q_norm (head_dim) and post_attention_layernorm (hidden)
    removed += len(kvonly) * (tc.head_dim + tc.hidden_size)
    return removed


def smoke(model, processor, clip_id, t0_us, seed=42):
    """Short rollout + TF eval + cache-shape assert on the slim model.

    Reads the per-clip cache rather than the camera chunk zip: the val chunks were
    deleted in 2026-08, so `load_physical_aiavdataset` would re-download here.
    """
    data = sc.load_cached(sc.path_for("eval", clip_id, t0_us))
    inputs = lib.build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = lib.run_rollout(model, inputs, max_generation_length=64)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end].clone()
    for i in range(len(model.vlm.model.language_model.layers)):
        k, _ = lib.cache_layer_kv(roll["past_key_values"], i)
        assert k.shape[1] == 8, f"layer {i}: cache heads {k.shape[1]}"
    del roll
    ade, fde, nll = eval_config(model, inputs, seq_tf, coc_start, coc_end, gt_xy, [seed])
    return {"clip_id": clip_id, "coc_len": coc_end - coc_start,
            "minADE_k1": ade, "minFDE_k1": fde, "nll": nll}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="a named config, or <criterion>_u<pct>_v2 for the uniform family "
                         "(criterion in traj|coc|j|dual|j_traj; pct 40 means the matched "
                         "0.3985632694, any other pct means exactly pct/100), or "
                         "expert_u<N> / expert{q,m}_{u,c}<N> / dualexp_u40_e<N> for the expert-tower cuts; "
                         "dual{q,m}_u40_v2 / dualm_c<N> for the VLM axis split")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--importance", type=str, default="importance_v1")
    ap.add_argument("--stepvlm", type=str, default="importance_stepvlm_v1",
                    help="per-step VLM importance run for the znorm11 criterion")
    ap.add_argument("--jlens", type=str, default="jlens_v2",
                    help="J-lens run supplying q_j/mlp_j for the j_traj configs")
    ap.add_argument("--cache-importance", type=str, default="cachejlens_v1",
                    help="run_cache_jlens.py run supplying cache_vlm_* for the cachedual / "
                         "cacheonly configs")
    ap.add_argument("--expert-importance", type=str, default="importance_stepexp_znorm",
                    help="run supplying traj_exp_mlp for the dualrc_u40_s<N>_em<M> "
                         "expert-MLP-only half (importance_stepexp_znorm is what the "
                         "expert-axis ablation selected with)")
    ap.add_argument("--vqa-importance", type=str, default="importance_vqa",
                    help="run supplying vqa_vlm_* / coc_vlm_* for the vqa, coclingo and "
                         "trajvqa configs (measured on LingoQA train)")
    ap.add_argument("--wanda", type=str, default="wanda_v1",
                    help="run supplying q_w/mlp_w for the wanda config")
    ap.add_argument("--wanda-txt", type=str, default="wanda_txt_v1",
                    help="run supplying q_w/mlp_w for the wandatxt config")
    ap.add_argument("--scope-start", type=int, default=4,
                    help="dualscope_u40: first layer that may be cut (LLM-Pruner's "
                         "--block_attention_layer_start)")
    ap.add_argument("--scope-end", type=int, default=34,
                    help="dualscope_u40: one past the last layer that may be cut")
    ap.add_argument("--tyr-supernet", type=str, default="tyr_supernet_u40",
                    help="supernet dir for the tyr configs")
    ap.add_argument("--tyr-config", type=str,
                    default="tyr_search_u40/final_config.json",
                    help="searched level assignment for tyr_u40 (outputs-relative)")
    ap.add_argument("--sets-id", type=str, default="eval_sets")
    ap.add_argument("--no-state", action="store_true",
                    help="write only slim_meta.json, skipping the 16.8 GB slim_state.pt. "
                         "load_slim() reconstructs the identical model from the meta "
                         "(base weights are sliced in place, verified tensor-by-tensor), "
                         "so this is enough for evaluation; the state file is only needed "
                         "to hand the checkpoint to a machine without the base weights.")
    ap.add_argument("--reserve-gb", type=float, default=30.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / args.out
    if (args.config.startswith(("tyr_u40", "tyr_uniform_u40", "dualr_u40", "dualgr_u40",
                                "dualrc_u40"))
            and args.no_state):
        # the tyr configs REWRITE o_proj/down_proj (OSSCAR reconstruction); load_slim
        # rebuilds a --no-state recipe from the base weights, which would silently
        # evaluate selection-only -- that is what tyr_sel_u40 is for
        raise SystemExit("tyr configs need slim_state.pt: drop --no-state")
    device = reserve_gpu(args.reserve_gb, devices=None if args.gpu is None else [args.gpu])
    print(f"using {device}", flush=True)

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = helper.get_processor(model.tokenizer)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    vq, vm, eq, em, kvonly = build_masks(args.config, imp, model, args.jlens,
                                         args.vqa_importance, args.stepvlm, args.wanda,
                                         args.wanda_txt,
                                         args.tyr_supernet, args.tyr_config,
                                         (args.scope_start, args.scope_end),
                                         args.importance, args.cache_importance,
                                         args.expert_importance)

    full_total = sl.n_params(model)
    t0 = time.time()
    meta = sl.apply_surgery(model, vq, vm, eq, em, kvonly_layers=kvonly)
    meta["config"] = args.config
    meta["importance_from"] = args.importance
    slim_total = sl.n_params(model)
    removed = expected_removed(model, vq, vm, eq, em, kvonly)
    print(f"surgery {time.time() - t0:.0f}s: {full_total:,} -> {slim_total:,} "
          f"(-{full_total - slim_total:,}, {(full_total - slim_total) / full_total * 100:.1f}%)",
          flush=True)
    assert slim_total == full_total - removed, (slim_total, full_total - removed)
    sl.check_slim(model)
    meta["params"] = {"full": full_total, "slim": slim_total, "removed": removed}

    t0 = time.time()
    sl.save_slim(model, meta, out_dir, write_state=not args.no_state)
    kind = "recipe" if args.no_state else "checkpoint"
    print(f"saved {kind} in {time.time() - t0:.0f}s -> {out_dir}", flush=True)

    man = pd.read_parquet(REPO / "outputs" / args.sets_id / "indist_500.parquet")
    result = smoke(model, processor, man.clip_id.iloc[0], int(man.t0_us.iloc[0]))
    print(f"smoke: {result}", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model": "nvidia/Alpamayo-1.5-10B", "model_revision": MODEL_REV,
        "config": args.config,
        "importance_from": args.importance, "jlens_from": args.jlens,
        "params": meta["params"],
        "kvonly_layers": list(kvonly),
        "kept_q_per_layer": {"vlm": [len(m["q"]) for m in meta["vlm"]],
                             "expert": [len(m["q"]) for m in meta["expert"]]},
        "smoke": result, "gpu": torch.cuda.get_device_name(device),
    }, indent=2))
    (out_dir / "summary.txt").write_text(
        f"slim checkpoint {args.config}\n"
        f"params {full_total:,} -> {slim_total:,} "
        f"(-{removed:,}, {removed / full_total * 100:.1f}%)\n"
        f"smoke clip {result['clip_id']}: NLL {result['nll']:.4f} "
        f"minADE@1 {result['minADE_k1']:.3f}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
