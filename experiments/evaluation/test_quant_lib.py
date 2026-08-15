"""Unit tests for quant_lib. No GPU and no model download needed.

Run: .venv/bin/python experiments/evaluation/test_quant_lib.py
"""

import glob
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import quant_lib as ql

SNAP = glob.glob("/mnt/nvme1n1/ad_vla/cache/hub/models--nvidia--Alpamayo-1.5-10B"
                 "/snapshots/7aba8293*")[0]
with open(SNAP + "/model.safetensors.index.json") as _fh:
    WMAP = json.load(_fh)["weight_map"]

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")


def load(n):
    with safe_open(SNAP + "/" + WMAP[n], framework="pt") as fh:
        return fh.get_tensor(n)


print("== quantize_dequantize ==")
W = load("vlm.model.language_model.layers.17.mlp.gate_proj.weight")  # (12288, 4096) bf16

e16, e8, e4, e2 = (ql.rel_mse(W, b) for b in (16, 8, 4, 2))
check("bits=16 is exact", e16 == 0.0, f"rel_mse={e16}")
check("error is monotone in bits", e16 < e8 < e4 < e2,
      f"8:{e8:.2e} 4:{e4:.2e} 2:{e2:.2e}")

q0 = ql.quantize_dequantize(W[:8], 0)
check("bits=0 zeroes the row", bool((q0 == 0).all()))
q16 = ql.quantize_dequantize(W[:8], 16)
check("bits=16 round-trips bit-exactly", bool(torch.equal(q16, W[:8])))

a = ql.quantize_dequantize(W[:64], 4)
b = ql.quantize_dequantize(W[:64], 4)
check("deterministic", bool(torch.equal(a, b)))
check("dtype preserved", a.dtype == W.dtype, str(a.dtype))

print("\n== per-row mixed bit-widths ==")
sub = W[:32].clone()
mix = torch.tensor([16] * 8 + [8] * 8 + [4] * 8 + [0] * 8)
qm = ql.quantize_dequantize(sub, mix)
same = [bool(torch.equal(qm[i * 8:(i + 1) * 8],
                         ql.quantize_dequantize(sub[i * 8:(i + 1) * 8], bit)))
        for i, bit in enumerate((16, 8, 4, 0))]
check("mixed spec == per-group uniform applied separately", all(same), str(same))

print("\n== padding (in_features not divisible by group) ==")
V = load("vlm.model.visual.blocks.13.mlp.linear_fc1.weight")  # (4304, 1152) -> in=1152 ok
V2 = load("vlm.model.visual.blocks.13.mlp.linear_fc2.weight")  # (1152, 4304) -> in=4304 pads
check("padded tensor keeps shape", ql.quantize_dequantize(V2, 4).shape == V2.shape,
      f"{tuple(V2.shape)} in={V2.shape[1]} 4304%64={4304 % 64}")
pe4, pe8 = ql.rel_mse(V2, 4), ql.rel_mse(V2, 8)
check("padded tensor error still monotone", pe8 < pe4, f"8:{pe8:.2e} 4:{pe4:.2e}")
check("padded error comparable to unpadded", pe4 < 5 * ql.rel_mse(V, 4),
      f"pad:{pe4:.2e} nopad:{ql.rel_mse(V, 4):.2e}")

print("\n== QVLA per-row layout (group=0) reproduces the measured penalty ==")
r4 = ql.rel_mse(W, 4, group=0)
g4 = ql.rel_mse(W, 4, group=64)
check("per-row scale is worse than g64 at 4 bits", r4 > 2 * g4, f"row:{r4:.2e} g64:{g4:.2e}")
H = load("vlm.lm_head.weight")
hr, hg = ql.rel_mse(H, 4, group=0), ql.rel_mse(H, 4, group=64)
check("lm_head per-row 4-bit is catastrophic", hr > 10 * hg, f"row:{hr:.2e} g64:{hg:.2e}")

print("\n== storage accounting ==")
shapes = {"a": (4096, 4096), "b": (12288, 4096), "c": (4096, 12288)}
sp8 = ql.uniform_spec(shapes, 8)
eb8, _, _ = ql.effective_bits(shapes, sp8)
check("uniform W8 effective bits = 8 + (16+8)/64", abs(eb8 - (8 + 24 / 64)) < 1e-6,
      f"{eb8:.4f}")
eb16, _, _ = ql.effective_bits(shapes, ql.uniform_spec(shapes, 16))
check("uniform 16-bit costs exactly 16", abs(eb16 - 16.0) < 1e-9, f"{eb16:.4f}")
eb0, _, _ = ql.effective_bits(shapes, ql.uniform_spec(shapes, 0))
check("all-zero spec costs 0", eb0 == 0.0)

print("\n== pool selection ==")


class Fake(torch.nn.Module):
    """Mimics the real module-name tree without loading 22 GB."""

    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Module()
        self.vlm.model = torch.nn.Module()
        self.vlm.model.language_model = torch.nn.Module()
        self.vlm.model.language_model.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8, bias=False),
                                 "down_proj": torch.nn.Linear(8, 8, bias=False)})
            for _ in range(2)])
        self.vlm.model.language_model.embed_tokens = torch.nn.Linear(8, 8, bias=False)
        self.vlm.model.visual = torch.nn.Module()
        self.vlm.model.visual.blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"fc1": torch.nn.Linear(8, 8, bias=False)})])
        self.vlm.lm_head = torch.nn.Linear(8, 8, bias=False)
        self.expert = torch.nn.Module()
        self.expert.layers = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8, bias=False)})
             for _ in range(2)])
        self.action_in_proj = torch.nn.Linear(8, 8, bias=False)


names = set(ql.pool_modules(Fake()))
check("expert excluded", not any("expert" in n for n in names))
check("embed_tokens excluded", not any("embed_tokens" in n for n in names))
check("action_in_proj excluded", not any("action_in_proj" in n for n in names))
check("lm_head included", "vlm.lm_head" in names)
check("vlm text layers included", sum("language_model.layers" in n for n in names) == 4)
check("vit included", sum("visual" in n for n in names) == 1)
check("pool size is exactly 6", len(names) == 6, str(sorted(names)))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
