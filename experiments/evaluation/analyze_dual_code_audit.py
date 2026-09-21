"""Read-only CPU audit of the principal dual experiments and stored artifacts.

Run from the repository root with .venv/bin/python. No model inference is run.
Writes only reports/evaluation/2026-09-10_dual-code-audit.json.
Original experiment code, scores, checkpoints and metrics remain unchanged.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if Path.cwd() != REPO:
    raise SystemExit("Run this audit from the repository root.")
torch.set_num_threads(4)
OUT = REPO / "reports/evaluation/2026-09-10_dual-code-audit.json"
result = {}


def record(name, value):
    result[name] = value
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(name, json.dumps(value, ensure_ascii=False)[:1400], flush=True)


def functions_from(path, names, env):
    tree = ast.parse(Path(path).read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), env)
    return env


record(
    "scope",
    {
        "arms": [
            "baseline",
            "dual",
            "traj",
            "coc",
            "dualr",
            "dualr_rep",
            "dualr_wl",
            "dualrwl_em50",
            "dualrwl_em75",
            "dualrwl_em87p5",
            "dualrwl_em93p75",
            "dualexp_em93p75",
        ],
        "method": "CPU artifact/code audit; no GPU rerun; no experiment mutation",
    },
)
imp = np.load("outputs/importance_v2/importance.npz")
rank = functions_from("experiments/head_analysis/run_cocsafe.py", ["rank_norm"], {"np": np})[
    "rank_norm"
]
meta = json.loads(Path("outputs/slim_dual_u40_v2/slim_meta.json").read_text())
rank_check = {}
for key, mkey, keep in [("q", "q", 19), ("mlp", "mlp", 7390)]:
    a, b = imp[f"traj_vlm_{key}"], imp[f"coc_vlm_{key}"]
    ra, rb = rank(a), rank(b)
    current = np.maximum(ra, rb)
    neutral = ra.copy()
    neutral[np.ptp(a, axis=1) == 0] = 0
    alternative = np.maximum(neutral, rb)
    layers = []
    for layer_idx in range(36):
        mask = np.ones(a.shape[1], bool)
        mask[np.argsort(current[layer_idx])[: a.shape[1] - keep]] = False
        kept = np.flatnonzero(mask)
        assert kept.tolist() == meta["vlm"][layer_idx][mkey]
        alt = np.ones(a.shape[1], bool)
        alt[np.argsort(alternative[layer_idx])[: a.shape[1] - keep]] = False
        if not np.array_equal(mask, alt):
            layers.append(
                {
                    "layer": layer_idx,
                    "replacement_count": int(np.sum(mask & ~alt)),
                    "retained_overlap": float(np.sum(mask & alt) / keep),
                }
            )
    rank_check[key] = {
        "constant_traj_layers": np.flatnonzero(np.ptp(a, axis=1) == 0).tolist(),
        "saved_recipe_matches_current_code": True,
        "neutral_zero_branch_changes": layers,
        "zero_row_rank_range": [float(ra[-1].min()), float(ra[-1].max())],
    }
record("dual_zero_rank", rank_check)

# Check the actual gate hooks against an independent activation-gradient contraction.

gates_env = functions_from(
    "experiments/head_analysis/prune_lib.py", ["UnitGates"], {"torch": torch, "np": np}
)
toy = SimpleNamespace(
    self_attn=SimpleNamespace(o_proj=torch.nn.Linear(6, 4, bias=False).double()),
    mlp=SimpleNamespace(down_proj=torch.nn.Linear(5, 4, bias=False).double()),
)
gates = gates_env["UnitGates"]([toy], 2, 3, 5, "cpu", torch.float64)
gen = torch.Generator().manual_seed(42)
head_input = torch.randn((1, 3, 6), generator=gen, dtype=torch.float64, requires_grad=True)
mlp_input = torch.randn((1, 3, 5), generator=gen, dtype=torch.float64, requires_grad=True)
loss = toy.self_attn.o_proj(head_input).square().sum() + toy.mlp.down_proj(mlp_input).square().sum()
loss.backward()
head_expected = (head_input * head_input.grad).reshape(1, 3, 2, 3).sum((0, 1, 3))
mlp_expected = (mlp_input * mlp_input.grad).sum((0, 1))
assert torch.allclose(gates.q_gates[0].grad, head_expected)
assert torch.allclose(gates.mlp_gates[0].grad, mlp_expected)
gates.remove()
record("gate_gradient_check", {"head_matches_contraction": True, "mlp_matches_contraction": True})

# Replay the original z-score function and verify all expert ladder masks.
env = functions_from("experiments/head_analysis/run_expert_agg.py", ["zscore_layers"], {"np": np})
z = np.load("outputs/stepimp_fm_perstep_v2/step_importance.npz")["mlp_abs_step"].astype(np.float64)
scores = np.mean([env["zscore_layers"](a) for a in z], axis=0)
stored = np.load("outputs/importance_stepexp_znorm/importance.npz")["traj_exp_mlp"]
assert np.array_equal(scores, stored)
wl = json.loads(Path("outputs/slim_dualr_wl_u40/slim_meta.json").read_text())
ladder = {}
for suffix, r in [
    ("50", 0.5),
    ("75", 0.75),
    ("87p5", 0.875),
    ("93p75", 0.9375),
    ("96p875", 0.96875),
    ("98p4375", 0.984375),
    ("100", 1.0),
]:
    root = Path(f"outputs/slim_dualrwl_em{suffix}_u40")
    m = json.loads((root / "slim_meta.json").read_text())
    checks = []
    for layer_idx in range(36):
        mask = np.ones(8256, bool)
        mask[np.argsort(scores[layer_idx])[: round(8256 * r)]] = False
        checks.append(
            np.flatnonzero(mask).tolist() == m["expert"][layer_idx]["mlp"]
            and m["expert"][layer_idx]["q"] == list(range(16))
        )
    ladder[suffix] = {
        "all_masks_match": all(checks),
        "vlm_indices_match_wl": m["vlm"] == wl["vlm"],
        "state_exists": (root / "slim_state.pt").exists(),
        "kept_per_layer": len(m["expert"][0]["mlp"]),
    }
record("expert_znorm", {"score_exact_match": True, "ladder": ladder})

# Evaluate token weight functions without importing GPU/model runners.
rc = functions_from(
    "experiments/head_analysis/run_cache_recon.py",
    ["token_weights", "WeightedHessianHook"],
    {
        "torch": torch,
        "SPAN_SHARE": {"vision": 0.7223, "text": 0.1656, "hist": 0.0418, "sink": 0.011},
    },
)
spans = {k: torch.arange(4) == i for i, k in enumerate(["vision", "text", "hist", "sink"])}
cfg = json.loads(Path("outputs/dualr_wl_supernet_u40/metadata.json").read_text())
lingo = cfg["lingo"]
drive = 1 - cfg["decode_share"] - lingo["prefill_share"] - lingo["answer_share"]
a = rc["token_weights"](spans, 4, 2, cfg["decode_share"], True, 1)
a[:4] *= drive / (1 - cfg["decode_share"])
vs = {k: torch.arange(3) == i for i, k in enumerate(["vision", "text", "sink"])}
b = rc["token_weights"](
    vs,
    3,
    2,
    lingo["answer_share"],
    True,
    1,
    prefill_share=lingo["prefill_share"],
    span_share={"vision": 0.7223, "text": 0.1656 + 0.0418, "sink": 0.011},
)
load = functions_from(
    "experiments/lingoqa/run_vqa_importance.py",
    ["load_train_manifest"],
    {"pd": pd, "hashlib": hashlib, "TRAIN": Path(lingo["source"])},
)["load_train_manifest"]
train = load(lingo["segments"], lingo["questions"], lingo["seed"])
nqa = sum(len(m["questions"]) for m in train)
counts = np.array([cfg["num_clips"]] * 2 + [nqa] * 2)
weights = np.array([float(a[:4].sum()), float(a[4:].sum()), float(b[:3].sum()), float(b[3:].sum())])
masses = counts * weights
val = pd.read_parquet("/mnt/nvme1n1/ad_vla/data/lingoqa/val.parquet")
record(
    "wl_mixture",
    {
        "stream_order": ["driving_prefill", "own_coc", "lingo_prefill", "lingo_answer"],
        "driving_clips": cfg["num_clips"],
        "qa_samples": nqa,
        "per_sample_weights": weights.tolist(),
        "actual_mass": masses.tolist(),
        "actual_normalized_coefficients": (masses / masses.sum()).tolist(),
        "lingo_segment_overlap_eval": len({m["segment_id"] for m in train} & set(val.segment_id)),
    },
)

# Open-loop stored row coverage and provenance.
spec = importlib.util.spec_from_file_location(
    "pn", REPO / "experiments/evaluation/paper_numbers.py"
)
pn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pn)
calib = set(pd.read_parquet("outputs/eval_sets/calib_100.parquet").clip_id)
expected = {
    s: set(pd.read_parquet("outputs/eval_sets/" + n + ".parquet").clip_id)
    for s, n in [("indist", "val_500"), ("test", "test_500"), ("oodval", "ood_val")]
}
coverage = {}
loaded = {}
for arm in result["scope"]["arms"]:
    loaded[arm] = {}
    for s, (name, filterval) in pn.ARMS[arm].items():
        files = sorted(Path("outputs", name).glob("*_s*of*.json"))
        rows = [
            r
            for f in files
            for r in json.loads(f.read_text())
            if not filterval or r.get("split") == "val"
        ]
        ids = [r["clip_id"] for r in rows]
        c = Counter(ids)
        duplicates = {i: n for i, n in c.items() if n > 1}
        conflicts = [
            i
            for i in duplicates
            if len({json.dumps(r, sort_keys=True) for r in rows if r["clip_id"] == i}) > 1
        ]
        records = {r["clip_id"]: r for r in rows}
        loaded[arm][s] = records
        conf = json.loads(Path("outputs", name, "config.json").read_text())
        bad = []
        for r in rows:
            seed = int.from_bytes(hashlib.sha256(f"42:{r['clip_id']}".encode()).digest()[:4], "big")
            if r.get("seed") != seed or any(
                len(r[k]) < 6 or not np.isfinite(r[k]).all()
                for k in ["ade_rollout_k", "fde_rollout_k"]
            ):
                bad.append(r["clip_id"])
        coverage[f"{arm}/{s}"] = {
            "n_unique": len(records),
            "raw_rows": len(rows),
            "files": len(files),
            "duplicate_ids": len(duplicates),
            "conflicting_duplicates": len(conflicts),
            "missing_manifest": len(expected[s] - set(ids)),
            "extra_manifest": len(set(ids) - expected[s]),
            "calib_overlap": len(set(ids) & calib),
            "bad_seed_or_metrics": len(bad),
            "gpu_config": conf["gpu"],
            "k_config": conf["k"],
            "model_config": conf["model"],
            "revision": conf.get("model_revision"),
            "mean_minADE6": float(np.mean([pn.at6(r, "ade_rollout_k") for r in records.values()])),
        }
record("open_loop_coverage", coverage)
consistency = {}
for arm, ref in [
    ("dualrwl_em50", "dualr_wl"),
    ("dualrwl_em75", "dualr_wl"),
    ("dualrwl_em87p5", "dualr_wl"),
    ("dualrwl_em93p75", "dualr_wl"),
    ("dualexp_em93p75", "dual"),
]:
    for s in expected:
        a, b = loaded[arm][s], loaded[ref][s]
        common = set(a) & set(b)
        consistency[f"{arm}/{s}"] = {
            "paired_n": len(common),
            "coc_identical": sum(a[i]["gen_coc"] == b[i]["gen_coc"] for i in common),
        }
record("coc_consistency", consistency)

# Closed-loop count and aggregation invariants.
runs = Path("/home/cvlab21/project/chan/alpasim-runs")
closed = {}
for tag, run in [
    ("baseline", "m2601_merged_baseline"),
    ("dual", "m2601_merged_slim_dual_u40_v2"),
    ("dualr_wl", "m2601_merged_slim_dualr_wl_u40"),
    ("em93p75", "m2601_merged_slim_dualrwl_em93p75_u40"),
    ("dualexp", "m2601_merged_slim_dualexp_u40_em93p75"),
]:
    p = runs / run / "aggregate/results-summary.json"
    d = json.loads(p.read_text())
    rolls = d["rollouts"]
    counts = Counter(r["clipgt_id"] for r in rolls)
    closed[tag] = {
        "scenes": len(counts),
        "rollouts": len(rolls),
        "per_scene_counts": dict(Counter(counts.values())),
        "null_scores": sum(r["score"] is None for r in rolls),
        "mean_score": float(
            np.nanmean([r["score"] if r["score"] is not None else np.nan for r in rolls])
        ),
        "scene_ids": sorted(counts),
    }
common = set.intersection(*[set(d["scene_ids"]) for d in closed.values()])
for d in closed.values():
    d.pop("scene_ids")
record("closed_loop_coverage", {"common_scenes": len(common), "arms": closed})

# Read only the affected matrix blocks from memory-mapped checkpoints.
base = torch.load(
    "outputs/slim_dual_u40_v2/slim_state.pt", map_location="cpu", weights_only=True, mmap=True
)
parent = torch.load(
    "outputs/slim_dualr_wl_u40/slim_state.pt", map_location="cpu", weights_only=True, mmap=True
)
child = torch.load(
    "outputs/slim_dualrwl_em93p75_u40/slim_state.pt",
    map_location="cpu",
    weights_only=True,
    mmap=True,
)
twin = torch.load(
    "outputs/slim_dualexp_u40_em93p75/slim_state.pt",
    map_location="cpu",
    weights_only=True,
    mmap=True,
)
keys = [
    k
    for k in parent
    if k.startswith("vlm.model.language_model.layers.")
    and k.endswith(("o_proj.weight", "down_proj.weight"))
]
checks = {
    "wl_em93_vlm_refit_blocks": {k: torch.equal(parent[k], child[k]) for k in keys},
    "dualexp_vlm_blocks": {k: torch.equal(base[k], twin[k]) for k in keys},
}
for label, check in checks.items():
    assert all(check.values()), label
exkeys = [k for k in child if k.startswith("expert.")]
assert all(torch.equal(child[k], twin[k]) for k in exkeys)
old = torch.load(
    "outputs/slim_dualr_u40/slim_state.pt", map_location="cpu", weights_only=True, mmap=True
)
k = "vlm.model.language_model.layers.35.self_attn.o_proj.weight"
record(
    "state_checks",
    {
        "wl_em93_refit_blocks_equal": len(keys),
        "dualexp_dual_blocks_equal": len(keys),
        "expert_em93_twin_tensors_equal": len(exkeys),
        "old_dualr_layer35_o_maxabs": float(old[k].abs().max()),
        "wl_layer35_o_maxabs": float(parent[k].abs().max()),
        "dual_layer35_o_maxabs": float(base[k].abs().max()),
    },
)
# Existing constant-layer control: median zero does not imply identical per-clip outputs.
fix_checks = {}
for subset in expected:

    def read_rows(run):
        return {
            r["clip_id"]: r
            for path in sorted(Path("outputs", run).glob("*_s*of*.json"))
            for r in json.loads(path.read_text())
        }

    fixed = read_rows(f"dualfix_u40_v2_{subset}")
    control = read_rows(f"dual_ada_u40_v2_{subset}")
    ids = sorted(set(fixed) & set(control))
    delta = np.array(
        [pn.at6(fixed[i], "ade_rollout_k") - pn.at6(control[i], "ade_rollout_k") for i in ids]
    )
    fix_checks[subset] = {
        "n": len(ids),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(delta.mean()),
        "changed_minADE6": int(np.count_nonzero(delta)),
        "coc_same": sum(fixed[i]["gen_coc"] == control[i]["gen_coc"] for i in ids),
    }
record("existing_dualfix_control", fix_checks)

lingo_checks = {}
val_keys = set(zip(val.question_id, val.segment_id))
for run in [
    "lingo_vqa_baseline",
    "lingo_vqa_slim_dual_u40_v2",
    "lingo_vqa_slim_dualr_u40",
    "lingo_vqa_slim_dualr_rep_u40",
    "lingo_vqa_slim_dualr_wl_u40",
]:
    rows = json.loads(Path("outputs", run, "scored.json").read_text())
    keys = [(r["question_id"], r["segment_id"]) for r in rows]
    cfg = json.loads(Path("outputs", run, "config.json").read_text())
    lingo_checks[run] = {
        "n": len(rows),
        "unique_keys": len(set(keys)),
        "missing_eval_keys": len(val_keys - set(keys)),
        "extra_eval_keys": len(set(keys) - val_keys),
        "accuracy": float(np.mean([r["score"] > 0 for r in rows])),
        "correct_matches_threshold": all(r["correct"] == (r["score"] > 0) for r in rows),
        "generation_config": {
            k: v
            for k, v in cfg.items()
            if k in ["style", "max_gen", "max_new_tokens", "temperature", "seed", "gpu"]
        },
    }
record("lingo_scoring", lingo_checks)
print("DONE", OUT, flush=True)
