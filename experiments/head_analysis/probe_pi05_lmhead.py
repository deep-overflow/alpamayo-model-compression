"""Does the released pi0.5 checkpoint still have a usable language channel?

Decides whether pi0.5 can serve as the manipulation-side platform for the
reasoning-preserving-compression study. The paper describes two-stage inference
(predict a high-level subtask in text, then act), but the open checkpoint was
trained with knowledge insulation (VLM supervised on FAST *action* tokens) and
LeRobot's loader remaps `lm_head.weight` onto `embed_tokens.weight`, with no
decoding path implemented.

Gemma ties input/output embeddings, so logits = hidden @ embed.T is still
available. This probe runs that projection on a text-only prompt and reports what
the top tokens actually are:

  * fluent continuation of the prompt      -> language channel intact  (GOOD)
  * ids concentrated in a narrow high range -> FAST action tokens       (BAD)
  * incoherent / degenerate                 -> language lost in training (BAD)

Text-only is deliberate: it isolates the language head from the robot input
pipeline (no images/state needed), so a negative result cannot be blamed on
malformed observations.

Usage:
    python probe_pi05_lmhead.py [--device cuda:3]
"""

import argparse
import glob
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors.torch import safe_open

PREFIX = "paligemma_with_expert.paligemma."
PROMPTS = [
    "The robot should",
    "To clean the table, the robot first needs to",
    "Task: put the dishes in the sink. Next subtask:",
]


def load_lm(repo, dtype=torch.float32):
    """Language tower weights straight out of the safetensors shard.

    The checkpoint stores no `embed_tokens`: Gemma ties input/output embeddings and
    the single [vocab, hidden] matrix is saved under `lm_head.weight` (LeRobot
    remaps it to embed_tokens on load). We use it for both roles.
    """
    # HF_HUB_CACHE follows HF_HUB_CACHE / HF_HOME, so this keeps working after the
    # blob cache moved off the root filesystem (2026-08-06)
    pattern = (f"{HF_HUB_CACHE}/"
               f"models--{repo.replace('/', '--')}/snapshots/*/model.safetensors")
    local = glob.glob(pattern)
    path = Path(local[0]) if local else Path(snapshot_download(repo)) / "model.safetensors"
    want = {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        lm_keys = [k for k in keys if k.startswith(PREFIX + "model.language_model.")]
        for k in lm_keys + [PREFIX + "lm_head.weight"]:
            want[k] = f.get_tensor(k).to(dtype)
    return want, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lerobot/pi05_base")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer, GemmaForCausalLM
    from transformers.models.gemma.configuration_gemma import GemmaConfig

    w, all_keys = load_lm(args.repo)
    n_layer = 1 + max(int(k.split("layers.")[1].split(".")[0])
                      for k in w if ".layers." in k)
    emb = w[PREFIX + "lm_head.weight"]
    vocab, hidden = emb.shape
    L0 = PREFIX + "model.language_model.layers.0."
    inter = w[L0 + "mlp.up_proj.weight"].shape[0]
    q_out = w[L0 + "self_attn.q_proj.weight"].shape[0]
    k_out = w[L0 + "self_attn.k_proj.weight"].shape[0]
    head_dim = args.head_dim
    n_head, n_kv = q_out // head_dim, k_out // head_dim
    print(f"language tower: layers={n_layer} hidden={hidden} inter={inter} "
          f"q_out={q_out} k_out={k_out} head_dim={head_dim} "
          f"heads={n_head} kv={n_kv} vocab={vocab}")

    cfg = GemmaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
                      num_hidden_layers=n_layer, num_attention_heads=n_head,
                      num_key_value_heads=n_kv, head_dim=head_dim,
                      tie_word_embeddings=True)
    model = GemmaForCausalLM(cfg)
    sd = {k.replace(PREFIX + "model.language_model.", "model."): v
          for k, v in w.items() if k != PREFIX + "lm_head.weight"}
    sd["model.embed_tokens.weight"] = emb
    sd["lm_head.weight"] = emb
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print("  missing sample:", missing[:5])
    if unexpected[:5]:
        print("  unexpected sample:", unexpected[:5])
    model.eval().to(args.device)

    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    out = {"repo": args.repo, "layers": n_layer, "vocab": vocab, "samples": []}
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            logits = model(ids).logits[0, -1]
        top = torch.topk(logits, 10)
        text = tok.decode(gen[0], skip_special_tokens=True)
        toks = [tok.decode([i]) for i in top.indices.tolist()]
        print(f"\nprompt: {prompt!r}")
        print(f"  greedy: {text!r}")
        print(f"  top-10 next ids: {top.indices.tolist()}")
        print(f"  top-10 tokens  : {toks}")
        out["samples"].append({"prompt": prompt, "greedy": text,
                               "top_ids": top.indices.tolist(), "top_tokens": toks})

    ids_all = [i for s in out["samples"] for i in s["top_ids"]]
    frac_high = sum(i > vocab - 2048 for i in ids_all) / max(len(ids_all), 1)
    out["frac_top_ids_in_last_2048"] = frac_high
    print(f"\ntop-token ids in the last 2048 vocab slots (FAST-token region): {frac_high:.2f}")
    print("verdict hint: high fraction => action tokens, not language")
    Path("pi05_lmhead_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
