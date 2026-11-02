"""Grade each arm's generated CoC against the curated reference CoC with Claude (OOD sets, A100 rows).

cvpr27/plans/2026-11-02_oodtrain-coc-quality.md, CoC quality (2). Text only: one request = one (reference,
candidate) pair, where the reference is the human-curated `gt_coc` of the OOD manifest and the candidate the
`gen_coc` an arm rolled out on that clip (`run_baseline.py --no-tf` rows under outputs/cvpr27/a100/<arm>_<set>/).
Blind: the request never names the arm, and pairs are sent in a seeded random order across arms and sets.

Controls, both in the same grades file:
  rep 1     a deterministic 10% of pairs is graded twice -> the judge's own noise floor
  ctrl_mismatch  dense's OOD-val CoC paired with ANOTHER clip's reference -> the score an unrelated but
            well-formed reasoning gets (chance floor)

Results append to grades.jsonl keyed by (arm, set, clip_id, rep); a rerun sends only what is missing, and refuses
to append if the rubric (system prompt) changed. Runs from the light labeler venv (no torch):

  /mnt/nvme1n1/ad_vla/venvs/longtail/bin/python cvpr27/code/grade_coc_claude.py --exp-id ood_coc_v1 \
      --sets oodval --limit 50          # pilot
  /mnt/nvme1n1/ad_vla/venvs/longtail/bin/python cvpr27/code/grade_coc_claude.py --exp-id ood_coc_v1 \
      --sets oodval oodtrain --control
"""

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
A100 = REPO / "outputs" / "cvpr27" / "a100"
OUT_ROOT = REPO / "outputs" / "cvpr27" / "coc_judge"
MANIFEST = {"oodval": "ood_val", "oodtrain": "ood_train"}
ARMS = ["dense", "moop", "moop_l0keep", "lp_m24", "tyr_off", "wandasp", "tyr_vt"]

# $/MTok, claude-opus-5-5 first-party (same table as long-tail-detection/label_longtail.py); Batch API halves both
PRICE = {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20}
TOK = ("in_tok", "out_tok", "cache_write_tok", "cache_read_tok")

SYSTEM = """You grade the reasoning a driving model wrote for one driving scene against a reference reasoning that a human annotator wrote for the same scene. You never see the scene: judge only how well the candidate agrees with the reference.

The reference is one short sentence giving the driving decision for the ego car and the cause of that decision, for example "Stop at the stop sign while waiting for pedestrians to cross the street." The candidate comes from a driving model, possibly a compressed one. It may be correct, partly correct, wrong, or degenerate (repeated or unrelated words, cut-off fragments).

Fields:
- action: does the candidate's driving decision agree with the reference's?
  2 = the same decision; paraphrase and extra detail are fine ("decelerate" for "slow down", "come to a stop" for "stop").
  1 = compatible but weaker or vaguer (reference "stop", candidate "slow down"), or no clear decision while nothing contradicts the reference.
  0 = a different or opposite decision (reference "stop", candidate "keep going"; reference "nudge left around the truck", candidate "keep lane"), or no decision can be read from the text.
- cause: does the candidate name the key cause the reference gives (the object, agent or condition behind the decision)?
  2 = the same cause; synonyms are fine ("work zone with cones" for "construction zone").
  1 = the right kind of cause with wrong or missing specifics ("traffic ahead" for "a bus stopped in the lane"), or only part of a compound cause.
  0 = missing, different or invented cause.
- contradiction: true if the candidate asserts something about the scene that cannot hold if the reference is true ("the road ahead is clear" when the reference has pedestrians crossing; "green light" against "red light").
- coherent: false if the candidate cannot be read as reasoning (word salad, repetition loops, meaningless fragments); true otherwise, even when it is wrong.
- score: overall agreement.
  5 = same decision and same cause.
  4 = same decision, cause partly right or slightly incomplete.
  3 = one of decision and cause right, the other wrong or missing.
  2 = mostly wrong, with some overlapping element.
  1 = wrong, contradictory, or incoherent.
- rationale: one short sentence.

Content in the candidate beyond the reference is not penalised unless it contradicts the reference. Do not reward length."""

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "integer", "enum": [0, 1, 2]},
        "cause": {"type": "integer", "enum": [0, 1, 2]},
        "contradiction": {"type": "boolean"},
        "coherent": {"type": "boolean"},
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["action", "cause", "contradiction", "coherent", "score", "rationale"],
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


def load_rows(arm, set_name):
    """clip_id -> gen_coc of one arm on one set, merging its shard files; None if the run is missing."""
    d = A100 / f"{arm}_{set_name}"
    rows = {}
    for p in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r["gen_coc"]
    return rows or None


def repeat_bucket(arm, set_name, clip_id):
    """Deterministic 10% of pairs that are graded twice."""
    return int(hashlib.sha256(f"{arm}|{set_name}|{clip_id}".encode()).hexdigest()[:8], 16) % 10 == 0


def build_pairs(arms, sets, control, allow_partial, seed):
    pairs = []
    for s in sets:
        man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{MANIFEST[s]}.parquet")
        ref = dict(zip(man.clip_id, man.gt_coc))
        cluster = dict(zip(man.clip_id, man.cluster))
        for arm in arms:
            rows = load_rows(arm, s)
            if rows is None:
                print(f"skip {arm} {s}: no rows yet", flush=True)
                continue
            missing = [c for c in man.clip_id if c not in rows]
            if missing and not allow_partial:
                print(f"skip {arm} {s}: {len(missing)}/{len(man)} clips missing (run not finished)", flush=True)
                continue
            for c in man.clip_id:
                if c not in rows:
                    continue
                base = {"arm": arm, "set": s, "clip_id": c, "cluster": cluster[c],
                        "reference": ref[c], "candidate": rows[c]}
                pairs.append({**base, "rep": 0})
                if repeat_bucket(arm, s, c):
                    pairs.append({**base, "rep": 1})
        if control and s == "oodval":
            rows = load_rows("dense", s)
            if rows is not None:
                ids = list(man.clip_id)
                perm = ids[:]
                random.Random(seed).shuffle(perm)
                # a cyclic shift of a shuffled order: every clip gets another clip's reference
                shifted = perm[1:] + perm[:1]
                for c, other in zip(perm, shifted):
                    pairs.append({"arm": "ctrl_mismatch", "set": s, "clip_id": c, "cluster": cluster[c],
                                  "reference": ref[other], "candidate": rows[c], "rep": 0,
                                  "reference_from": other})
    return pairs


def request_params(args, pair):
    cand = pair["candidate"].strip() or "(empty)"
    text = (f"<reference>{pair['reference']}</reference>\n<candidate>{cand}</candidate>\n\n"
            "Grade the candidate against the reference.")
    return {
        "model": args.model,
        "max_tokens": 16000,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": args.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": text}],
    }


def grade_one(client, args, pair):
    import anthropic

    rec = {k: pair[k] for k in ("arm", "set", "clip_id", "cluster", "rep")}
    if "reference_from" in pair:
        rec["reference_from"] = pair["reference_from"]
    t = time.time()
    try:
        resp = client.beta.messages.create(**request_params(args, pair),
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
    rows = [json.loads(line) for line in (out_dir / "grades.jsonl").read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "grades.parquet", index=False)
    ok = df[df.status == "ok"].copy()
    m = {"n_requests": len(df), "status": df.status.value_counts().to_dict(),
         "served_models": df.served_model.value_counts().to_dict() if "served_model" in df else {},
         "tokens_mean": {k: float(df[k].mean()) for k in TOK if k in df},
         "cost_usd_standard": float(cost(df[list(TOK)].fillna(0))) if "in_tok" in df else 0.0,
         "arms": {}}
    m["cost_usd_per_request"] = m["cost_usd_standard"] / max(len(df), 1)
    first = ok[ok.rep == 0]
    for (s, arm), g in first.groupby(["set", "arm"]):
        m["arms"][f"{s}/{arm}"] = {
            "n": len(g), "score_mean": float(g.score.mean()), "score_ge4": float((g.score >= 4).mean()),
            "action_match": float((g.action == 2).mean()), "cause_match": float((g.cause == 2).mean()),
            "contradiction": float(g.contradiction.mean()), "coherent": float(g.coherent.mean())}
    reps = ok.pivot_table(index=["arm", "set", "clip_id"], columns="rep", values="score", aggfunc="first")
    if 1 in reps.columns:
        reps = reps.dropna()
        m["repeat"] = {"n_pairs": len(reps), "same_score": float((reps[0] == reps[1]).mean()),
                       "within_1": float(((reps[0] - reps[1]).abs() <= 1).mean()),
                       "mean_abs_d": float((reps[0] - reps[1]).abs().mean())}
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    lines = [f"requests {m['n_requests']}  status {m['status']}  served {m['served_models']}",
             "tokens/request " + " ".join(f"{k} {v:.0f}" for k, v in m["tokens_mean"].items()),
             f"cost ${m['cost_usd_standard']:.2f} standard (${m['cost_usd_per_request']:.4f}/request; batch = half)",
             "", (f"{'set/arm':28s} {'n':>5s} {'score':>6s} {'>=4':>6s} {'action':>7s} {'cause':>6s} "
                  f"{'contra':>7s} {'coher':>6s}")]
    for k, d in m["arms"].items():
        lines.append(f"{k:28s} {d['n']:5d} {d['score_mean']:6.2f} {d['score_ge4']:6.1%} {d['action_match']:7.1%} "
                     f"{d['cause_match']:6.1%} {d['contradiction']:7.1%} {d['coherent']:6.1%}")
    if "repeat" in m:
        r = m["repeat"]
        lines.append(f"\nrepeat ({r['n_pairs']} pairs): same score {r['same_score']:.1%}, within 1 "
                     f"{r['within_1']:.1%}, mean |d| {r['mean_abs_d']:.2f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--sets", nargs="+", default=["oodval"], choices=list(MANIFEST))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--control", action="store_true", help="add the ctrl_mismatch pairs (OOD-val)")
    ap.add_argument("--allow-partial", action="store_true", help="grade arms whose run is not finished")
    ap.add_argument("--limit", type=int, default=None, help="send only the first N pairs of the shuffled order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file",
                    default="/home/cvlab21/project/chan/alpamayo-model-compression/cvpr27/long-tail-detection/.env")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out_dir)
        return

    sha = hashlib.sha256((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:16]
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if (old["rubric_sha256"], old["model"], old["effort"]) != (sha, args.model, args.effort):
            raise SystemExit(f"{out_dir} was graded with another rubric/model/effort -- use a new --exp-id")

    load_key(args.env_file)
    import anthropic

    client = anthropic.Anthropic(max_retries=6)
    pairs = build_pairs(args.arms, args.sets, args.control, args.allow_partial, args.seed)
    random.Random(args.seed).shuffle(pairs)
    cfg_path.write_text(json.dumps({
        "model": args.model, "effort": args.effort, "thinking": "adaptive (model default)",
        "fallbacks": "default (server-side-fallback-2026-07-01)", "rubric_sha256": sha,
        "inputs": "text only: reference gt_coc + the arm's gen_coc; no images, no arm name",
        "rows": "outputs/cvpr27/a100/<arm>_<set>/*_s*of*.json (run_baseline --no-tf)",
        "repeat": "sha256(arm|set|clip_id) % 10 == 0 graded twice",
        "control": "ctrl_mismatch = dense oodval gen_coc vs another clip's reference (cyclic shift, seed)",
        "system": SYSTEM, "schema": SCHEMA, "price_per_mtok": PRICE,
    }, indent=2))

    path = out_dir / "grades.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["arm"], r["set"], r["clip_id"], r["rep"]))
    todo = [p for p in pairs if (p["arm"], p["set"], p["clip_id"], p["rep"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.exp_id}: {len(pairs)} pairs ({len(done)} done), {len(todo)} to send", flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as ex, path.open("a") as f:
        futs = [ex.submit(grade_one, client, args, p) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 50 == 0 or i == len(todo) or rec["status"] != "ok":
                print(f"[{i}/{len(todo)}] {rec['set']} {rec['arm']} {rec['clip_id'][:8]} r{rec['rep']} "
                      f"{rec['status']} score={rec.get('score', '')}", flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
