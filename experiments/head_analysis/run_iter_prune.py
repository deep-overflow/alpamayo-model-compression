"""Staged re-calibration: build iteratively re-scored pruning masks for the u40 family.

One-shot pruning scores every unit once on the intact model and cuts 39.86% of each
layer in one go; the calibration diagnostic showed that single-unit scores measured
there do not compose at that scale (group damage is 1/5 to 1/55 of the singles sum).
This orchestrator splits the same per-layer budget into stages and re-measures the
scores on the masked model between stages, so each cut is chosen by the network that
actually receives it. Everything else -- criterion, calib_100, uniform allocation,
final budget (Q 13/32, MLP 4898/12288 per layer), expert/KV untouched -- matches the
shipped `slim_<crit>_u40_v2`, so `it3 - oneshot` is a one-factor contrast of staging.

The process itself needs no GPU: it computes cuts on CPU and launches the scoring
passes (`run_importance.py --mask`, `run_jlens.py --mask`) as subprocesses under
`run_retry_host.sh`, which waits for a free card. Stage-1 scores are the stored
importance_v2 / jlens_v2 (the intact-model measurement, identical to the one-shot
starting point). Both stores were measured on Blackwell, so scoring subprocesses
should be pinned to Blackwell cards (--gpus) to keep the stage-1/stage-2 scores on
one architecture. A completed stage's scores are reused on restart.

Selection detail: combined criteria are composed exactly as make_slim.build_masks
does (rank_norm over the full layer, then max), and cuts extend among survivors
only -- at stage >= 2 a masked unit's re-measured score is exactly zero (mask hooks
precede gate/stat hooks), so it sits at the bottom of both halves and cannot
re-enter. With --stages 1 the selection reduces to make_slim's exact argsort path,
which is what --verify checks bit-for-bit against the shipped checkpoint (gate R0).

Gates are pre-registered in plans/2026-08-16_iterative-recalibration.md.

Usage:
  .venv/bin/python experiments/head_analysis/run_iter_prune.py --criterion dual --verify
  nohup .venv/bin/python experiments/head_analysis/run_iter_prune.py \
      --criterion dual --gpus 0,1,2,3 >> logs/iter_dual.log 2>&1 &
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

N_LAYERS, N_Q, N_MLP = 36, 32, 12288
CUT_Q, CUT_MLP = 13, 4898  # the realized u40_v2 per-layer cut
SHIPPED = {c: f"slim_{c}_u40_v2" for c in ("traj", "coc", "dual", "jtraj", "j")}
NEEDS = {"traj": ("imp",), "coc": ("imp",), "dual": ("imp",),
         "j": ("jl",), "jtraj": ("imp", "jl")}


def rank_norm(scores):
    """Per-layer rank in [0, 1]; verbatim from run_cocsafe (not imported: that module
    pulls the full model stack and this process must stay CPU-only)."""
    out = np.zeros_like(scores, dtype=float)
    n = scores.shape[1]
    for i in range(scores.shape[0]):
        out[i] = np.argsort(np.argsort(scores[i])) / max(n - 1, 1)
    return out


def combined_scores(crit, imp, jl):
    """The five within-layer scores exactly as make_slim.build_masks composes them."""
    if crit == "traj":
        return imp["traj_vlm_q"], imp["traj_vlm_mlp"]
    if crit == "coc":
        return imp["coc_vlm_q"], imp["coc_vlm_mlp"]
    if crit == "j":
        return jl["q_j"], jl["mlp_j"]
    if crit == "dual":
        return (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
                np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"])))
    return (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(jl["q_j"])),
            np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(jl["mlp_j"])))


def stage_counts(k_final, n_stages):
    return [int(np.floor(k_final * s / n_stages + 0.5)) for s in range(1, n_stages + 1)]


def extend_cut(scores, mask, k_target):
    """Cut down to k_target per layer: among surviving units, drop the lowest-scoring.

    With an all-ones mask this is `np.argsort(scores[li])[:k]` -- the exact
    select_mask_ratios expression, same default sort kind -- which R0 relies on.
    """
    mask = mask.copy()
    for li in range(scores.shape[0]):
        alive = np.where(mask[li] > 0)[0]
        need = k_target - (scores.shape[1] - len(alive))
        if need > 0:
            order = np.argsort(scores[li][alive])
            mask[li][alive[order[:need]]] = 0.0
    return mask


def run_scoring(kind, crit, stage, mask_path, args):
    """Launch one scoring pass under run_retry_host and block until it finishes."""
    exp_id = f"iter_{crit}_s{stage}_{kind}"
    out = REPO / "outputs" / exp_id
    if kind == "imp":
        done = ((out / "importance.npz").exists() and (out / "metrics.json").exists()
                and json.loads((out / "metrics.json").read_text())["n_clips"] == 100)
        script, result = "run_importance.py", out / "importance.npz"
        extra = ["--num-clips", "100", "--reserve-gb", "44"]
    else:
        done = (out / "jlens.npz").exists()
        script, result = "run_jlens.py", out / "jlens.npz"
        # jlens_v2's exact dictionary request (the corpus caps the freq half at 202)
        extra = ["--num-clips", "100", "--n-freq", "512", "--n-random", "512",
                 "--reserve-gb", "44"]
    if done:
        print(f"[{crit}] stage {stage} {kind}: already measured, reusing {exp_id}",
              flush=True)
        return dict(np.load(result))
    cmd = ["bash", str(REPO / "experiments/head_analysis/run_retry_host.sh"),
           str(args.retries), f"experiments/head_analysis/{script}",
           "--exp-id", exp_id, "--mask", str(mask_path), "--gpu", args.gpus] + extra
    print(f"[{crit}] stage {stage} {kind}: launching {' '.join(cmd[3:])}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO, check=True)
    if not result.exists():
        raise SystemExit(f"[{crit}] stage {stage} {kind}: pass ended without {result}")
    print(f"[{crit}] stage {stage} {kind}: done ({(time.time() - t0) / 60:.0f} min)",
          flush=True)
    return dict(np.load(result))


def verify_oneshot(crit, q_mask, m_mask):
    """R0: the 1-stage cut must reproduce the shipped checkpoint's kept sets exactly."""
    meta = json.loads((REPO / "outputs" / SHIPPED[crit] / "slim_meta.json").read_text())
    bad = []
    for li, ent in enumerate(meta["vlm"]):
        kq = np.where(q_mask[li] > 0)[0].tolist()
        km = np.where(m_mask[li] > 0)[0].tolist()
        if sorted(kq) != sorted(ent["q"]) or sorted(km) != sorted(ent["mlp"]):
            bad.append(li)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", required=True, choices=list(NEEDS))
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--gpus", type=str, default="0,1,2,3",
                    help="cards the scoring subprocesses may use; keep on Blackwell, "
                         "where importance_v2 and jlens_v2 were measured")
    ap.add_argument("--retries", type=int, default=2000)
    ap.add_argument("--importance", default="importance_v2")
    ap.add_argument("--jlens", default="jlens_v2")
    ap.add_argument("--verify", action="store_true",
                    help="R0 only: CPU check that --stages 1 reproduces the shipped "
                         "kept sets bit-for-bit, then exit")
    args = ap.parse_args()
    crit = args.criterion

    out_dir = REPO / "outputs" / f"iter_{crit}_u40"
    out_dir.mkdir(parents=True, exist_ok=True)
    kq_s = stage_counts(CUT_Q, args.stages)
    km_s = stage_counts(CUT_MLP, args.stages)

    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    jl = dict(np.load(REPO / "outputs" / args.jlens / "jlens.npz"))

    if args.verify:
        sq, sm = combined_scores(crit, imp, jl)
        q1 = extend_cut(sq, np.ones((N_LAYERS, N_Q)), CUT_Q)
        m1 = extend_cut(sm, np.ones((N_LAYERS, N_MLP)), CUT_MLP)
        bad = verify_oneshot(crit, q1, m1)
        print(f"[{crit}] R0 one-shot reproduction: "
              + ("PASS (36/36 layers bit-identical)" if not bad
                 else f"FAIL at layers {bad}"))
        raise SystemExit(1 if bad else 0)

    q_mask = np.ones((N_LAYERS, N_Q))
    m_mask = np.ones((N_LAYERS, N_MLP))
    stage_log = []
    for s in range(1, args.stages + 1):
        if s > 1:
            mask_path = out_dir / f"stage{s - 1}_mask.npz"
            if "imp" in NEEDS[crit]:
                imp = run_scoring("imp", crit, s, mask_path, args)
            if "jl" in NEEDS[crit]:
                jl = run_scoring("jl", crit, s, mask_path, args)
        sq, sm = combined_scores(crit, imp, jl)
        q_mask = extend_cut(sq, q_mask, kq_s[s - 1])
        m_mask = extend_cut(sm, m_mask, km_s[s - 1])
        np.savez(out_dir / f"stage{s}_mask.npz", q_mask=q_mask, mlp_mask=m_mask)
        stage_log.append({"stage": s, "cut_q": kq_s[s - 1], "cut_mlp": km_s[s - 1],
                          "kept_q": float(q_mask.sum()), "kept_mlp": float(m_mask.sum())})
        print(f"[{crit}] stage {s}: cut {kq_s[s - 1]}/{km_s[s - 1]} per layer "
              f"(kept {int(q_mask[0].sum())} q, {int(m_mask[0].sum())} mlp)", flush=True)

    np.savez(out_dir / "final_masks.npz", vq=q_mask, vm=m_mask)
    # churn: how much did staged re-scoring change the selection vs the one-shot cut?
    sq1, sm1 = combined_scores(crit, dict(np.load(
        REPO / "outputs" / args.importance / "importance.npz")), dict(np.load(
        REPO / "outputs" / args.jlens / "jlens.npz")))
    q1 = extend_cut(sq1, np.ones((N_LAYERS, N_Q)), CUT_Q)
    m1 = extend_cut(sm1, np.ones((N_LAYERS, N_MLP)), CUT_MLP)
    overlap_q = float((q_mask * q1).sum() / q_mask.sum())
    overlap_m = float((m_mask * m1).sum() / m_mask.sum())
    (out_dir / "config.json").write_text(json.dumps({
        "criterion": crit, "stages": args.stages,
        "stage_cuts_q": kq_s, "stage_cuts_mlp": km_s, "stage_log": stage_log,
        "importance_stage1": args.importance, "jlens_stage1": args.jlens,
        "scoring_gpus": args.gpus, "oneshot_ref": SHIPPED[crit],
        "kept_overlap_vs_oneshot": {"q": overlap_q, "mlp": overlap_m},
        "plan": "plans/2026-08-16_iterative-recalibration.md",
    }, indent=2))
    print(f"[{crit}] final masks -> {out_dir}  "
          f"overlap vs one-shot: q {overlap_q:.3f}, mlp {overlap_m:.3f}", flush=True)


if __name__ == "__main__":
    main()
