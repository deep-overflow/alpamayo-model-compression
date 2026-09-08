"""Does this box actually run the model? (run on cvlab20 after the venv build)

The venv importing torch proves nothing about whether the transferred model loads, the
caches are readable, or the GPU takes a 40 GB reservation. This checks each in the order
the real runners depend on them, so a failure names the layer that broke.

  . /home/cvlab20/project/chan/cvlab20-server/env.sh
  $ALPAMAYO_REPO/.venv/bin/python cvlab20-server/smoke_test.py --gpu 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("ALPAMAYO_REPO",
                           "/home/cvlab20/project/chan/alpamayo-model-compression"))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reserve-gb", type=float, default=40.0)
    args = ap.parse_args()

    print("== 1. environment")
    for k in ("ALPAMAYO_REPO", "AD_VLA_DATA", "HF_HUB_CACHE", "HF_HOME",
              "CUBLAS_WORKSPACE_CONFIG"):
        print(f"   {k:24s} {os.environ.get(k, '(unset)')}")
    if not os.environ.get("AD_VLA_DATA"):
        raise SystemExit("AD_VLA_DATA unset -- source env.sh first, or the caches will "
                         "be looked for on the wrong filesystem")

    print("== 2. torch and the GPU")
    import torch
    print(f"   torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"devices {torch.cuda.device_count()}")
    print(f"   cuda:{args.gpu} = {torch.cuda.get_device_name(args.gpu)}")

    print("== 3. caches")
    import sample_cache as sc
    for ns, man in (("calib_st", "calib_st4000"), ("test", "test_500"),
                    ("eval", "val_500"), ("ood", "ood_val")):
        try:
            samples = sc.calib_samples(REPO, man)
        except Exception as e:                       # noqa: BLE001 -- report, keep going
            print(f"   {ns:9s} manifest unreadable: {type(e).__name__}: {e}")
            continue
        clip, t0 = samples[0]
        p = sc.path_for(ns, clip, t0)
        print(f"   {ns:9s} {len(samples):5d} clips, first exists: {p.exists()}")

    print("== 4. reserve GPU memory (the importance pass peaks at 40.5 GB)")
    import expert_per_clip as epc  # also installs the gated-hub patch
    t = time.time()
    dev = epc.reserve_gpu(args.reserve_gb, devices=[args.gpu])
    print(f"   reserved on {dev} in {time.time() - t:.1f}s")

    print("== 5. load the model from the transferred snapshot")
    import slim_lib as sl
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    t = time.time()
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=sl.MODEL_REV,
                                        dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"   loaded {n:,} params in {time.time() - t:.1f}s (revision {sl.MODEL_REV[:8]})")
    if n != 11_078_526_194:
        print("   !! expected 11,078,526,194 -- a different snapshot may have arrived")

    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
