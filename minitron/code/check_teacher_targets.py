"""Check stored teacher targets against a fresh teacher pass (plan D10-5).

For the first --n windows that have a target file: rebuild the window alone (batch 1, so the
padding differs from the precompute batch), rerun the teacher, and compare
  - logits: max |diff|, argmax agreement, KL(stored || fresh);
  - velocity per epoch slot: max |diff|;
  - NLL of the actual next tokens under the stored logits, CoC part vs trajectory part
    (G0-T measured trajectory NLL ~1.9 on the teacher; an off-by-one in h_idx would not).
Writes outputs/minitron/teacher_targets_check/metrics.json.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import mt_data
import train_kd as tk
from make_teacher_targets import OUT, target_path


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/minitron/kd_sets/kd_r1.parquet")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    from expert_per_clip import reserve_gpu
    from run_baseline import set_determinism

    set_determinism(warn_only=True)
    reserve_gpu(30.0, devices=None if args.gpu is None else [args.gpu])
    rows = pd.read_parquet(tk.REPO / args.manifest)
    rows = rows[[target_path(c, t).exists() for c, t in zip(rows.clip_id, rows.t0_us)]]
    rows = rows.iloc[: args.n]
    teacher = tk.Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=tk.MODEL_REV,
                                             dtype=torch.bfloat16).to("cuda").eval()
    processor = tk.helper.get_processor(teacher.tokenizer)
    tok = teacher.tokenizer
    recs = []
    for r in rows.itertuples():
        raw = tk.load_raw(r, tgt_dir=OUT)
        g = raw[3]
        b = tk.make_batch(teacher, processor, tok, [raw], 256, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tk.vlm_forward(teacher, b, use_cache=True)
            lg = tk.sup_logits(teacher, out, b).float()  # (N, V)
            dv = []
            for slot in range(g["v"].shape[0]):
                noise, ti = mt_data.kd_draw(r.clip_id, r.t0_us, slot)
                assert ti == int(g["t_idx"][slot])
                t = torch.tensor(tk.T_GRID[ti], device="cuda").view(1, 1, 1)
                x_t = (1.0 - t) * noise.cuda()[None] + t * b["x1"]
                v = tk.velocity(teacher, out.past_key_values, out.rope_deltas, b, x_t, t)
                dv.append(float((v[0].cpu() - g["v"][slot]).abs().max()))
        st = g["logits"].float().cuda()  # (N, V)
        tgt = b["ids"][0, b["h_idx"] + 1]  # (N,) the token each position predicts
        nll = F.cross_entropy(st, tgt, reduction="none")
        n_traj = 128
        kl = (F.softmax(st, -1) * (F.log_softmax(st, -1) - F.log_softmax(lg, -1))).sum(-1)
        recs.append({
            "window": f"{r.clip_id}__t0_{int(r.t0_us)}", "n_sup": int(g["n_sup"]),
            "n_sup_fresh": int(lg.shape[0]), "logit_maxdiff": float((st - lg).abs().max()),
            "argmax_agree": float((st.argmax(-1) == lg.argmax(-1)).float().mean()),
            "kl_stored_fresh": float(kl.mean()), "v_maxdiff": dv,
            "nll_coc": float(nll[:-n_traj].mean()), "nll_traj": float(nll[-n_traj:].mean()),
            "file_mb": target_path(r.clip_id, r.t0_us).stat().st_size / 2**20})
        print(json.dumps(recs[-1]), flush=True)
    d = tk.REPO / "outputs" / "minitron" / "teacher_targets_check"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
