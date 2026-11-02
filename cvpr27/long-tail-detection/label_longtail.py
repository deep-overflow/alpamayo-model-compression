"""Label driving clips as long-tail or nominal with Claude, from the cached model inputs.

Claude is shown what the driving model is shown at t0: the t0 frame of each of the four
cameras and the ego's past 1.5 s of motion. It never sees the GT future path -- a selection
that saw the answer would call "hard" whatever has a large curvature, which the existing
manoeuvre buckets already cover -- and on OOD clips never the curated reasoning or cluster
stored in the same npz, because those are the labels the pilot checks Claude against.

Taxonomy = NVIDIA's nine `event_cluster` names (reasoning/ood_reasoning.parquet) + NOMINAL,
so OOD clips give a recall check for free.

Claude has no sampling seed: `--repeats 2` labels every clip twice, and the agreement
between the two is the label noise floor. Results append to labels.jsonl, keyed by
(clip_id, rep), so a rerun only sends what is missing.

Runs from the light labeler venv (no torch):
  /mnt/nvme1n1/ad_vla/venvs/longtail/bin/python cvpr27/long-tail-detection/label_longtail.py \
      --exp-id pilot50 --set ood_val:ood:25 --set test_2000:test:25 --repeats 2
"""

import argparse
import base64
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRE = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed")
OUT_ROOT = REPO / "outputs" / "cvpr27" / "long_tail_detection"

# $/MTok, claude-opus-5-5 first-party; Batch API halves both
PRICE = {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20}
TOK = ("in_tok", "out_tok", "cache_write_tok", "cache_read_tok")

CATS = [
    "NOMINAL",
    "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
    "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
    "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
    "CYCLISTS_AND_MICROMOBILITY_COMPLEX",
    "COMPLEX_INTERSECTION_INTERACTION",
    "EMERGENCY_INCIDENT_SCENE",
    "ROAD_DEBRIS_OR_SAFETY_TRACES",
    "ANIMALS_BIRDS_ROADKILL",
    "OTHER_LONGTAIL",
]

CAM_NAMES = {0: "cross-left (120 deg FOV)", 1: "front-wide (120 deg FOV)",
             2: "cross-right (120 deg FOV)", 6: "front-tele (30 deg FOV)"}
SHOW_ORDER = (1, 6, 0, 2)

SYSTEM = """You review driving scenes recorded by a car's forward-facing cameras, to find long-tail situations: the rare, non-routine scenes on which an autonomous driving model is most likely to fail.

For each scene you get the frame at the current moment from four cameras (front-wide, front-tele, cross-left, cross-right) and a summary of the ego car's motion over the last 1.5 seconds. Judge only what is visible now and the ego's recent motion. Do not guess what happens later.

Pick the one category that most affects what the ego car has to do next:

- NOMINAL: ordinary driving. Normal traffic, signals, parked cars, pedestrians on sidewalks away from the road, routine turns and stops. Most scenes are NOMINAL; only leave it when something specific makes the scene non-routine for the ego.
- WORK_ZONES_TEMP_TRAFFIC_CONTROL: construction or maintenance zones, cones, barrels, barriers, temporary signs, lane closures or shifts, flaggers, temporary signals.
- PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY: many pedestrians near the road, or pedestrians on, entering or very close to the ego's path.
- SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR: unusual vehicles (oversized, towed loads, construction or utility vehicles, stopped buses or trucks blocking a lane) or unusual behaviour (double parking, reversing, cutting in, wrong-way driving, U-turns, vehicles stopped in the travel lane).
- CYCLISTS_AND_MICROMOBILITY_COMPLEX: cyclists, scooters or other micromobility sharing or crossing the ego's path.
- COMPLEX_INTERSECTION_INTERACTION: unprotected turns, unclear right of way, several agents negotiating an intersection, unusual intersection layout.
- EMERGENCY_INCIDENT_SCENE: emergency vehicles with lights, police, crashes, incident response.
- ROAD_DEBRIS_OR_SAFETY_TRACES: objects, debris, spills or damage on the road surface.
- ANIMALS_BIRDS_ROADKILL: animals or birds on or near the road, roadkill.
- OTHER_LONGTAIL: clearly non-routine for driving but none of the above (e.g. flooded or blocked road, fallen tree, unusual road geometry, severely degraded visibility that changes how the car must drive).

Also list any further categories that apply (secondary_categories, may be empty), and rate driving difficulty for the next few seconds:
1 trivial (open road, nothing to react to)
2 routine (normal traffic and signals)
3 needs attention to specific agents or road features
4 needs a non-routine manoeuvre or negotiation (nudging around an obstruction, yielding in an unclear situation, following temporary lanes)
5 safety-critical or highly ambiguous

hazards: a few short phrases naming the specific things that drive the rating (empty for a plain NOMINAL scene). rationale: one or two sentences.

Rules: refer to people only by their road role (pedestrian, cyclist, worker, officer). Do not describe or infer age, gender, ethnicity, health or any other personal attribute of anyone, and do not read or mention licence plates."""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATS},
        "secondary_categories": {"type": "array", "items": {"type": "string", "enum": CATS}},
        "difficulty": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "hazards": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["category", "secondary_categories", "difficulty", "hazards", "rationale"],
    "additionalProperties": False,
}


def load_key(env_file):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in Path(env_file).read_text().splitlines():
        k, _, v = line.strip().partition("=")
        if k.strip().removeprefix("export ").strip() == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")
            return
    raise SystemExit(f"no ANTHROPIC_API_KEY in {env_file}")


def pick(df, n, seed):
    """n clips; round-robin over `cluster` when the manifest has one, so rare clusters appear."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n is None or n >= len(df):
        return df
    if "cluster" in df.columns:
        df = df.assign(_r=df.groupby("cluster").cumcount()).sort_values(["_r", "cluster"], kind="stable")
        df = df.drop(columns="_r")
    return df.head(n).reset_index(drop=True)


def parse_sets(specs, seed):
    clips = []
    for spec in specs:
        parts = spec.split(":")
        manifest, cache = parts[0], parts[1]
        n = int(parts[2]) if len(parts) > 2 else None
        df = pick(pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{manifest}.parquet"), n, seed)
        for r in df.itertuples():
            clips.append({"set": manifest, "cache": cache, "clip_id": r.clip_id, "t0_us": int(r.t0_us),
                          "gt_cluster": getattr(r, "cluster", None)})
    return clips


def t0_images(z, long_side):
    """The t0 frame of each camera, downscaled, as (camera name, base64 JPEG)."""
    n_frame = int(z["n_frame"])
    cams = z["camera_indices"].tolist()
    out = []
    for ci in SHOW_ORDER:
        c = cams.index(ci)
        img = Image.open(io.BytesIO(z["jpeg_bytes"][c * n_frame + n_frame - 1]))  # (1080, 1920) RGB
        img.thumbnail((long_side, long_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        out.append((CAM_NAMES[ci], base64.standard_b64encode(buf.getvalue()).decode()))
    return out


def ego_summary(z):
    h = z["ego_history_xyz"][0, 0]                                   # (16, 3), t0 rig frame, h[-1] = 0
    R = z["ego_history_rot"][0, 0]                                   # (16, 3, 3)
    v = np.linalg.norm(np.diff(h[:, :2], axis=0), axis=1) / 0.1      # (15,) m/s at 10 Hz
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))             # (16,), yaw[-1] = 0
    dt = 0.1 * (len(v) - 1)
    return (f"Ego motion over the last {dt:.1f} s: speed {v[-1]:.1f} m/s now, {v[0]:.1f} m/s "
            f"{dt:.1f} s ago (mean acceleration {(v[-1] - v[0]) / dt:+.1f} m/s^2); heading changed "
            f"{yaw[-1] - yaw[0]:+.1f} deg (positive = turning left).")


def build_content(clip, long_side):
    # only the image and ego fields are read; gt_coc / cluster in OOD npz never leave this process
    z = np.load(PRE / clip["cache"] / "samples" / f"{clip['clip_id']}__t0_{clip['t0_us']}.npz",
                allow_pickle=True)
    content = []
    for name, b64 in t0_images(z, long_side):
        content.append({"type": "text", "text": f"Camera: {name}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    content.append({"type": "text", "text": ego_summary(z) + "\n\nClassify this scene."})
    return content


def request_params(args, content):
    return {
        "model": args.model,
        "max_tokens": 16000,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": args.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": content}],
    }


def label_one(client, args, clip, rep):
    import anthropic

    rec = {**clip, "rep": rep}
    t = time.time()
    try:
        resp = client.beta.messages.create(**request_params(args, build_content(clip, args.long_side)),
                                           betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except anthropic.APIStatusError as e:
        return {**rec, "status": f"api_error_{e.status_code}", "error": str(e.message)[:300]}
    except anthropic.APIConnectionError as e:
        return {**rec, "status": "connection_error", "error": str(e)[:300]}
    u = resp.usage
    rec.update({"served_model": resp.model, "stop_reason": resp.stop_reason, "request_id": resp._request_id,
                "in_tok": u.input_tokens, "out_tok": u.output_tokens,
                "cache_read_tok": u.cache_read_input_tokens or 0,
                "cache_write_tok": u.cache_creation_input_tokens or 0, "sec": round(time.time() - t, 1)})
    if resp.stop_reason != "end_turn":
        rec["status"] = resp.stop_reason
        if resp.stop_reason == "refusal" and resp.stop_details:
            rec["error"] = f"{resp.stop_details.category}: {resp.stop_details.explanation}"
        return rec
    text = "".join(b.text for b in resp.content if b.type == "text")
    rec.update({"status": "ok", **json.loads(text)})
    return rec


def cost(df):
    return (df.in_tok * PRICE["input"] + df.out_tok * PRICE["output"]
            + df.cache_write_tok * PRICE["cache_write"] + df.cache_read_tok * PRICE["cache_read"]).sum() / 1e6


def summarize(out_dir):
    rows = [json.loads(line) for line in (out_dir / "labels.jsonl").read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "labels.parquet", index=False)
    ok = df[df.status == "ok"].copy()
    ok["longtail"] = ok.category != "NOMINAL"
    m = {"n_requests": len(df), "status": df.status.value_counts().to_dict(),
         "served_models": df.served_model.value_counts().to_dict() if "served_model" in df else {},
         "tokens_mean": {k: float(df[k].mean()) for k in ("in_tok", "out_tok", "cache_read_tok") if k in df},
         "cost_usd_standard": float(cost(df[list(TOK)].fillna(0))) if "in_tok" in df else 0.0,
         "sets": {}}
    m["cost_usd_per_request"] = m["cost_usd_standard"] / max(len(df), 1)
    for s, g in ok.groupby("set"):
        first = g[g.rep == 0]
        d = {"n_clips": int(first.clip_id.nunique()),
             "longtail_share": float(first.longtail.mean()),
             "category": first.category.value_counts().to_dict(),
             "difficulty": {int(k): int(v) for k, v in first.difficulty.value_counts().sort_index().items()}}
        if first.gt_cluster.notna().any():
            # recall against NVIDIA's curated cluster: primary match, and primary-or-secondary match
            prim = first.category == first.gt_cluster
            anyc = [gc == c or gc in sec for gc, c, sec in
                    zip(first.gt_cluster, first.category, first.secondary_categories)]
            d.update({"gt_primary_match": float(prim.mean()), "gt_any_match": float(np.mean(anyc)),
                      "per_cluster": {c: {"n": len(gg), "primary": float((gg.category == c).mean()),
                                          "longtail": float(gg.longtail.mean())}
                                      for c, gg in first.groupby("gt_cluster")}})
        reps = g.pivot_table(index="clip_id", columns="rep", values=["category", "difficulty", "longtail"],
                             aggfunc="first")
        if g.rep.nunique() > 1:
            reps = reps.dropna()
            d["repeat"] = {"n_pairs": len(reps),
                           "same_category": float((reps["category"][0] == reps["category"][1]).mean()),
                           "same_longtail": float((reps["longtail"][0] == reps["longtail"][1]).mean()),
                           "mean_abs_d_difficulty": float((reps["difficulty"][0] - reps["difficulty"][1])
                                                          .abs().mean())}
        m["sets"][s] = d
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    lines = [f"requests {m['n_requests']}  status {m['status']}  served {m['served_models']}",
             (f"tokens/request in {m['tokens_mean'].get('in_tok', 0):.0f} out {m['tokens_mean'].get('out_tok', 0):.0f}"
             f" cache_read {m['tokens_mean'].get('cache_read_tok', 0):.0f}"),
             f"cost ${m['cost_usd_standard']:.2f} standard (${m['cost_usd_per_request']:.4f}/request; batch = half)"]
    for s, d in m["sets"].items():
        lines.append(f"\n[{s}] {d['n_clips']} clips  long-tail {d['longtail_share']:.0%}  difficulty {d['difficulty']}")
        lines.append(f"  categories {d['category']}")
        if "gt_primary_match" in d:
            lines.append(f"  vs NVIDIA cluster: primary {d['gt_primary_match']:.0%}, primary-or-secondary "
                         f"{d['gt_any_match']:.0%}")
            for c, v in d["per_cluster"].items():
                lines.append(f"    {c:40s} n={v['n']:3d} primary {v['primary']:.0%} long-tail {v['longtail']:.0%}")
        if "repeat" in d:
            r = d["repeat"]
            lines.append(f"  repeat ({r['n_pairs']} pairs): same category {r['same_category']:.0%}, same long-tail "
                         f"flag {r['same_longtail']:.0%}, mean |d difficulty| {r['mean_abs_d_difficulty']:.2f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    help="manifest:cache[:n], manifest = outputs/eval_sets/<stem>.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--long-side", type=int, default=1280, help="image long side in px (cache is 1920x1080)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", default=str(HERE / ".env"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir)
        return

    load_key(args.env_file)
    import anthropic

    client = anthropic.Anthropic(max_retries=6)
    clips = parse_sets(args.sets, args.seed)
    (out_dir / "config.json").write_text(json.dumps({
        "sets": args.sets, "seed": args.seed, "repeats": args.repeats, "model": args.model,
        "effort": args.effort, "thinking": "adaptive (model default)", "long_side": args.long_side,
        "fallbacks": "default (server-side-fallback-2026-07-01)", "n_clips": len(clips),
        "inputs": "t0 frame x 4 cameras + ego past 1.5 s summary; no GT future, no gt_coc/cluster",
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16], "system": SYSTEM,
        "schema": SCHEMA, "price_per_mtok": PRICE,
    }, indent=2))

    path = out_dir / "labels.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["clip_id"], r["rep"]))
    todo = [(c, k) for k in range(args.repeats) for c in clips if (c["clip_id"], k) not in done]
    print(f"{args.exp_id}: {len(clips)} clips x {args.repeats} = {len(clips) * args.repeats}, "
          f"{len(todo)} to send", flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as ex, path.open("a") as f:
        futs = [ex.submit(label_one, client, args, c, k) for c, k in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"[{i}/{len(todo)}] {rec['set']} {rec['clip_id'][:8]} r{rec['rep']} {rec['status']} "
                  f"{rec.get('category', '')} d={rec.get('difficulty', '')} gt={rec.get('gt_cluster') or '-'}",
                  flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
