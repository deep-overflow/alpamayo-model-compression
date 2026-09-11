"""지금까지의 alpasim 폐루프 결과를 한 파일로 모은다 -- 씬 집합(suite)별로.

`analyze_alpasim.py` 는 한 번에 한 묶음의 arm 만 보고, 그 산출물이 `outputs/` 여기저기
흩어져 있다. 전체를 나란히 보려면 병합 런에서 직접 다시 집계하는 편이 낫다: 점수와 게이트는
`aggregate/results-summary.json` 이 원본이고, arm 이 늘어도 재실행이 필요 없다.

**CoC 는 config 이름만으로 찾으면 안 된다.** 같은 config 가 여러 씬 집합에서 돌았고
(`slim_tyr_u40_r` 은 150씬 0.0592, hard100 0.0633), `outputs/*/metrics.json` 은 이름만
같으면 구별되지 않는다. 그래서 그 파일의 `n_scenes` 가 이 suite 와 맞을 때만 가져온다.

점수는 rollout -> 씬 평균 -> arm 평균 순으로 접고, baseline 대비 페어드 차이는 두 arm 이
모두 돈 씬에서만 낸다. suite 안의 모든 arm 이 같은 씬을 돌았는지 확인해 결과에 적는다.

Usage:
  .venv/bin/python experiments/evaluation/collect_alpasim_all.py --out outputs/alpasim_all
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RUNS_DEFAULT = Path("/home/cvlab21/project/chan/alpasim-runs")

# (prefix, 표시 이름, 기대 씬 수, 설명)
SUITES = [
    ("m2601_merged_", "150씬 매트릭스", 150,
     "public_2601 을 scene_id 로 정렬한 첫 150씬. 913씬 전체보다 쉽다."),
    ("h100_merged_", "hard100", 100,
     "hard_score 상위 100씬. 150씬 매트릭스와 서로소."),
]
GATES = ("collision_at_fault", "offroad", "wrong_lane")

# 접두사 규칙에서 벗어나 있지만 같은 씬을 돈 런. 지금은 sangoh 의 expert-pruning arm 하나로,
# `fm-expert-pruning/scripts/make_slim.py --arm G_default_r40` 이 만든 체크포인트다.
# 우리 arm 이 아니므로 external 로 표시해 표에서 구분한다.
EXTRA = {"m2601_merged_": [("fmp_G_default_r40_merged", "G_default_r40", "sangoh, expert pruning")]}


def load(run_dir):
    """rollout 리스트 -> 씬별 점수/게이트.

    게이트는 rollout 최상위가 아니라 `metrics` 안에 있다 -- 최상위에서 찾으면 조용히 0/0 이
    되어 표가 전부 '-' 로 나온다. 값이 하나도 안 잡히면 예외로 세운다.
    """
    rows = json.loads((run_dir / "aggregate/results-summary.json").read_text())["rollouts"]
    per_scene = collections.defaultdict(list)
    hits = {g: 0 for g in GATES}
    miss = {g: 0 for g in GATES}
    for r in rows:
        per_scene[r["clipgt_id"]].append(r["score"])
        m = r.get("metrics") or r.get("score_metrics") or {}
        for g in GATES:
            v = m.get(g, r.get(g))
            if v is None:
                miss[g] += 1
            elif float(v) > 0:
                hits[g] += 1
    if all(h == 0 for h in hits.values()) and all(m == len(rows) for m in miss.values()):
        raise SystemExit(f"{run_dir.name}: 게이트를 하나도 읽지 못했습니다 "
                         f"(rollout 키: {sorted(rows[0])})")
    # 분모는 전체 rollout 으로 둔다 -- 기존 리포트들이 그렇게 냈고, 값이 없는 rollout 을
    # 분모에서 빼면 비율이 조용히 올라간다. 결측 수는 따로 남겨 표에 각주로 쓴다.
    gates = {g: [hits[g], len(rows)] for g in GATES}
    return ({s: float(np.mean(v)) for s, v in per_scene.items()}, gates, len(rows),
            {g: miss[g] for g in GATES if miss[g]})


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def harvest_coc(n_scenes):
    """이 suite 와 씬 수가 같은 analyze_alpasim 산출물에서만 CoC 를 가져온다."""
    out = {}
    for f in sorted((REPO / "outputs").glob("*/metrics.json")):
        try:
            m = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # outputs/ 에는 중단된 실행이 남긴 반쪽짜리 metrics.json 도 있다. CoC 수확은
            # 부수적이므로 못 읽는 파일은 건너뛴다.
            continue
        if not isinstance(m, dict) or m.get("n_scenes") != n_scenes or "coc" not in m:
            continue
        for cfg, v in m["coc"].items():
            if isinstance(v, dict) and "mean_degenerate_frac" in v:
                out.setdefault(cfg, dict(v, _from=f.parent.name))
    return out


def removed_params(cfg):
    f = REPO / "outputs" / cfg / "config.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("params", {}).get("removed")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_DEFAULT)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/alpasim_all")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    blob = {"suites": []}
    for prefix, label, n_expect, desc in SUITES:
        runs = sorted(d for d in args.runs_root.iterdir()
                      if d.name.startswith(prefix) and (d / "aggregate/results-summary.json").exists())
        if not runs:
            print(f"[{label}] 런 없음 -- 건너뜀")
            continue
        arms, external = {}, {}
        for d in runs:
            cfg = d.name[len(prefix):]
            arms[cfg] = load(d)
        for run_name, cfg, note in EXTRA.get(prefix, []):
            d = args.runs_root / run_name
            if (d / "aggregate/results-summary.json").exists():
                arms[cfg] = load(d)
                external[cfg] = note
        scene_sets = {c: set(v[0]) for c, v in arms.items()}
        common = sorted(set.intersection(*scene_sets.values()))
        ragged = {c: len(s) for c, s in scene_sets.items() if len(s) != len(common)}
        coc = harvest_coc(n_expect)
        print(f"[{label}] arm {len(arms)}개, 공통 씬 {len(common)}"
              + (f", 씬 수가 다른 arm {ragged}" if ragged else "")
              + f", CoC 수확 {len(coc)}개")

        base = np.array([arms["baseline"][0][s] for s in common]) if "baseline" in arms else None
        entries = []
        for cfg, (ps, gates, n_roll, gmiss) in sorted(arms.items()):
            x = np.array([ps[s] for s in common])
            lo, hi = boot_ci(x)
            e = {"config": cfg, "score": float(x.mean()), "ci_lo": lo, "ci_hi": hi,
                 "median": float(np.median(x)), "n_rollouts": n_roll,
                 "n_scenes_own": len(ps),
                 "pass_frac": float((x > 0).mean()),
                 "gates": gates, "gate_missing": gmiss, "removed": removed_params(cfg),
                 "coc": coc.get(cfg, {}).get("mean_degenerate_frac"),
                 "coc_empty": coc.get(cfg, {}).get("mean_empty_frac"),
                 "coc_soup": coc.get(cfg, {}).get("mean_soup_frac"),
                 "coc_from": coc.get(cfg, {}).get("_from"),
                 "external": external.get(cfg)}
            if base is not None and cfg != "baseline":
                d = x - base
                dlo, dhi = boot_ci(d)
                nz = d[d != 0]
                e["delta"] = float(d.mean())
                e["delta_lo"], e["delta_hi"] = dlo, dhi
                e["wilcoxon_p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) else None
                e["wlt"] = [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]
            entries.append(e)
        blob["suites"].append({"prefix": prefix, "label": label, "desc": desc,
                               "n_scenes": len(common), "ragged": ragged, "arms": entries})

    (out / "metrics.json").write_text(json.dumps(blob, indent=1))
    print(f"\n-> {out}/metrics.json")


if __name__ == "__main__":
    main()
