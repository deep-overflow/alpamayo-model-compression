"""Verify a transferred box can actually run recovery training, before it burns a GPU.

Checks in order of cost, stopping at the first failure:

1. manifests    outputs/{eval_sets,recovery_sets} parquets read, and the row counts
                match what train_recover.load_samples expects (1200 + 1271 train,
                238 + 262 probe).
2. samples      every npz those manifests name resolves under AD_VLA_DATA. A missing
                clip is silent in training until that step comes up hours in, so this
                is the check that matters most after an interrupted rsync.
3. weights      the pinned Alpamayo revision resolves from the local hub cache, and
                the slim recipe (slim_meta.json) is present for --ckpt.
4. --load       (opt-in, needs a GPU and ~24 GB VRAM) reconstruct the slim model from
                slim_meta.json alone -- no slim_state.pt on the target -- and attach
                LoRA to it. That exercises the whole load path the trainer uses; the
                numerical KI insulation gate stays where it is, in train_recover's
                own --ki-check at startup.

Usage:
  python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2
  python experiments/transfer/preflight.py --ckpt outputs/slim_coc_u55_v2 --load --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))

FAILED = []


def check(label, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(label)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/slim_coc_u55_v2")
    ap.add_argument("--load", action="store_true", help="also build the model and step once")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    print(f"repo         {REPO}")
    print(f"AD_VLA_DATA  {os.environ.get('AD_VLA_DATA', '(unset -> /mnt/nvme1n1/ad_vla/data)')}")
    print(f"HF_HUB_CACHE {os.environ.get('HF_HUB_CACHE', '(unset)')}")
    print()

    import sample_cache as sc

    # 1. manifests
    sets = REPO / "outputs" / "recovery_sets"
    esets = REPO / "outputs" / "eval_sets"
    for p in (sets / "train_official_1200.parquet", sets / "val_official_238.parquet",
              esets / "ood.parquet", esets / "ood_val.parquet"):
        if not check(f"manifest {p.name}", p.exists(), str(p)):
            return report()

    off = pd.read_parquet(sets / "train_official_1200.parquet")
    voff = pd.read_parquet(sets / "val_official_238.parquet")
    ood = pd.read_parquet(esets / "ood.parquet")
    oodv = pd.read_parquet(esets / "ood_val.parquet")
    ood_tr = ood[ood.split == "train"]
    check("train rows  1200 official + 1271 ood",
          len(off) == 1200 and len(ood_tr) == 1271, f"{len(off)} + {len(ood_tr)}")
    check("probe rows   238 official +  262 ood",
          len(voff) == 238 and len(oodv) == 262, f"{len(voff)} + {len(oodv)}")

    # 2. samples -- the check an interrupted rsync fails
    groups = [("train", off), ("ood", ood_tr), ("eval", voff), ("ood", oodv)]
    for ns, df in groups:
        missing = [r.clip_id for r in df.itertuples()
                   if not sc.path_for(ns, r.clip_id, int(r.t0_us)).exists()]
        check(f"{ns} cache ({len(df)} clips)", not missing,
              "all present" if not missing else f"{len(missing)} missing, e.g. {missing[:2]}")
    if FAILED:
        return report()

    # 3. weights + recipe
    ckpt = REPO / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    check(f"slim recipe {ckpt.name}/slim_meta.json", (ckpt / "slim_meta.json").exists(),
          str(ckpt))

    import slim_lib as sl
    from huggingface_hub import constants as hfc
    hub = Path(hfc.HF_HUB_CACHE)
    snap = hub / "models--nvidia--Alpamayo-1.5-10B" / "snapshots" / sl.MODEL_REV
    shards = sorted(snap.glob("*.safetensors")) if snap.is_dir() else []
    check(f"base weights rev {sl.MODEL_REV[:8]}", len(shards) == 5,
          f"{len(shards)} shards in {snap}")
    cos = hub / "models--nvidia--Cosmos-Reason2-8B"
    check("Cosmos tokenizer/config", any(cos.rglob("tokenizer_config.json")), str(cos))

    # The `.no_exist` markers are what make HF_HUB_OFFLINE usable: they record that an
    # optional file is genuinely absent upstream. Without them huggingface_hub cannot
    # tell "not cached" from "does not exist" and raises LocalEntryNotFoundError, which
    # is how NEURON jobs 890894/890895 died -- the loader probes for adapter_config.json
    # and model.safetensors, neither of which the Alpamayo repo has.
    marks = hub / "models--nvidia--Alpamayo-1.5-10B" / ".no_exist" / sl.MODEL_REV
    have = sorted(p.name for p in marks.iterdir()) if marks.is_dir() else []
    check("offline markers for the pinned rev", "model.safetensors" in have,
          ", ".join(have) if have else f"missing {marks}")
    if FAILED:
        return report()

    # 4. the expensive one
    if args.load:
        import expert_per_clip  # noqa: F401  -- installs the gated-hub patch first
        import recover_lib as rl
        import torch

        dev = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
        print(f"\nloading {ckpt.name} on {dev} from slim_meta.json (no slim_state.pt needed) ...")
        peft_model, base, _ = rl.load_slim_lora(ckpt, device=dev)
        n_lora = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        check("LoRA attached", n_lora > 0, f"{n_lora/1e6:.1f}M trainable")
        check("model on device", next(base.parameters()).is_cuda,
              str(next(base.parameters()).device))
        del peft_model, base
        torch.cuda.empty_cache()

    return report()


def report():
    print()
    if FAILED:
        print(f"PREFLIGHT FAILED: {len(FAILED)} check(s) -- {', '.join(FAILED)}")
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
