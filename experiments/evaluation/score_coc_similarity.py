"""How close is each generated CoC to the curated one? Two text measures, per clip.

Reads the rows run_coc_ablation.py wrote (gen_coc, gt_coc) and scores every pair with

  judge  Qwen/Qwen3-8B, thinking off. Two separate questions -- does the candidate name
         the same ego ACTION as the reference, and the same CAUSE -- each answered with one
         digit 0/1/2. The score is the expectation over the next-token probabilities of
         "0"/"1"/"2", so it is continuous and needs no parsing.
  emb    sentence-transformers/all-mpnet-base-v2 loaded through transformers (mean pooling,
         L2-normalised), cosine similarity.

Validity controls written alongside: every reference against ITSELF (must top the scale)
and against the generated CoC of ANOTHER clip (a fixed derangement; must score low).

No driving model is involved, so this is not bound to the Ada cards.

Usage:
  python experiments/evaluation/score_coc_similarity.py --gpu 0
"""

import argparse
import configparser
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JUDGE = "Qwen/Qwen3-8B"
EMB = "sentence-transformers/all-mpnet-base-v2"

SYS = ("You grade driving-reasoning sentences. Each sentence states what the ego vehicle "
       "does and why. Answer with a single digit and nothing else.")
Q = {
    "action": (
        "Does the candidate describe the same ego-vehicle ACTION (maneuver) as the reference? "
        "Ignore the stated reason.\n"
        "2 = same maneuver (e.g. both stop, both nudge left, both keep lane)\n"
        "1 = related but not the same (e.g. slow down vs stop, same direction but different "
        "maneuver)\n"
        "0 = different or contradictory maneuver"),
    "cause": (
        "Does the candidate attribute the action to the same CAUSE (the same object, agent or "
        "road condition) as the reference? Ignore the action itself.\n"
        "2 = same cause\n"
        "1 = related or partially overlapping cause\n"
        "0 = different cause, or the candidate gives no cause"),
}


def load_rows(exp_dir):
    rows = {}
    for f in sorted(exp_dir.glob("baseline_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r
    return [rows[k] for k in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="coc_ablation_ood")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cp = configparser.ConfigParser()
    cp.read(Path.home() / ".cache/huggingface/stored_tokens")
    os.environ.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(args.gpu),
                       "HF_HOME": str(Path.home() / ".cache/huggingface"),
                       "HF_HUB_CACHE": "/mnt/nvme1n1/ad_vla/cache/hub",
                       "HF_HUB_ENABLE_HF_TRANSFER": "0",
                       "HF_TOKEN": cp["full_right"]["hf_token"], "OMP_NUM_THREADS": "8"})
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    out_dir = REPO / "outputs" / args.exp_id
    rows = load_rows(out_dir)[: args.limit]
    n = len(rows)
    gt = [str(r["gt_coc"]).strip() for r in rows]
    gen = [str(r["gen_coc"]).strip() for r in rows]
    perm = np.roll(np.arange(n), n // 2)  # fixed derangement: clip i gets clip i+n/2's CoC
    pairs = {"own": gen, "shuffled": [gen[j] for j in perm], "identity": gt}

    # embedding cosine
    tok = AutoTokenizer.from_pretrained(EMB)
    enc = AutoModel.from_pretrained(EMB).cuda().eval()

    @torch.no_grad()
    def embed(texts):
        outs = []
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to("cuda")
            h = enc(**b).last_hidden_state  # (B, T, 768)
            m = b["attention_mask"][..., None].float()  # (B, T, 1)
            e = (h * m).sum(1) / m.sum(1)  # (B, 768)
            outs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(outs)

    e_gt = embed(gt)  # (N, 768)
    emb = {k: (e_gt * embed(v)).sum(-1).numpy() for k, v in pairs.items()}  # (N,) each
    del enc
    torch.cuda.empty_cache()
    print("emb cos mean", {k: round(float(v.mean()), 4) for k, v in emb.items()}, flush=True)

    # judge
    jt = AutoTokenizer.from_pretrained(JUDGE)
    jt.padding_side = "left"
    jm = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).cuda().eval()
    digit_ids = [jt.encode(d, add_special_tokens=False) for d in "012"]
    assert all(len(d) == 1 for d in digit_ids)
    digit_ids = torch.tensor([d[0] for d in digit_ids], device="cuda")  # (3,)

    def prompt(question, ref, cand):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"{question}\n\nReference: {ref}\nCandidate: "
                                            f"{cand if cand else '(empty)'}\n\nDigit:"}]
        return jt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)

    @torch.no_grad()
    def judge(question, cands):
        exp, mass = [], []
        for i in range(0, n, args.batch):
            texts = [prompt(question, gt[j], cands[j]) for j in range(i, min(i + args.batch, n))]
            b = jt(texts, padding=True, return_tensors="pt").to("cuda")
            p = jm(**b).logits[:, -1].float().softmax(-1)[:, digit_ids]  # (B, 3)
            mass.append(p.sum(-1).cpu())
            p = p / p.sum(-1, keepdim=True)
            exp.append((p * torch.arange(3, device="cuda")).sum(-1).cpu())  # (B,)
        return torch.cat(exp).numpy(), torch.cat(mass).numpy()

    res = {"clip_id": [r["clip_id"] for r in rows], "perm": perm.tolist(),
           "judge_model": JUDGE, "emb_model": EMB, "questions": Q}
    for k, cands in pairs.items():
        res[f"emb_{k}"] = [round(float(x), 5) for x in emb[k]]
        for qn, qt in Q.items():
            s, mass = judge(qt, cands)
            res[f"judge_{qn}_{k}"] = [round(float(x), 5) for x in s]
            print(f"judge {qn:6s} {k:8s} mean {s.mean():.3f}  digit mass {mass.mean():.4f} "
                  f"(min {mass.min():.3f})", flush=True)
    (out_dir / "coc_similarity.json").write_text(json.dumps(res))
    print("->", out_dir / "coc_similarity.json")


if __name__ == "__main__":
    main()
