"""Render the 15-minute meeting's tables from stored experiment results.

No model inference or shared-output writes. Run from the repository root:
    MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_meeting_tables.py

PNG, PDF, SVG, CSV and the source manifest are written to ppt-fig/ by default.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
LP = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/llm-pruner")
CL = Path("/home/cvlab21/project/chan/alpasim-runs")
LP_CL = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-runs")
FULL = 11_078_526_194
SOURCES = {}
INK, MUTED, TEAL = "#142D3E", "#526575", "#006F73"


def source_bytes(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(source_bytes(path))


def clip_values(directory, pattern="*_s*of*.json", ood=False, metric="ade"):
    rows = {}
    for path in sorted(directory.glob(pattern)):
        batch = read(path)
        if not isinstance(batch, list):
            continue
        for row in batch:
            if ood and row.get("split") != "val":
                continue
            key = row["clip_id"]
            assert key not in rows, (directory, "duplicate clip", key)
            samples = np.asarray(row[f"{metric}_rollout_k"], dtype=float)
            assert len(samples) >= 6 and np.isfinite(samples).all()
            rows[key] = float(samples[:6].min())
    assert len(rows) == (262 if ood else 500), (directory, len(rows))
    return rows


def scene_values(path, metric="score", n_scenes=150):
    groups, seen = {}, set()
    for row in read(path)["rollouts"]:
        key = (row["clipgt_id"], row["rollout_id"])
        assert key not in seen, (path, key)
        seen.add(key)
        value = row["score"] if metric == "score" else row["metrics"][metric]
        assert value is not None and np.isfinite(value), (path, metric, key)
        if metric in ("collision_at_fault", "offroad"):
            assert value in (0, 1), (path, metric, value)
        groups.setdefault(key[0], []).append(value)
    assert len(groups) == n_scenes and all(len(v) == 2 for v in groups.values()), path
    return {k: float(np.mean(v)) for k, v in groups.items()}


def mean(values):
    return float(np.mean(list(values.values())))


def closed_metrics(path, n_scenes=150):
    # Equal two-rollout scene counts make this also the mean over all rollouts.
    return {metric: mean(scene_values(path, metric, n_scenes))
            for metric in ("collision_at_fault", "offroad", "progress_clipped_rel")}


def paired(a, b, n_boot=10000):
    assert a.keys() == b.keys(), "Unpaired evaluation sets"
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(0)
    draws = delta[rng.integers(0, len(delta), (n_boot, len(delta)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(delta.mean()), "lo": float(lo), "hi": float(hi)}


def lingo(directory, exp_id):
    rows = read(directory / "metrics.json")["runs"]
    row, = [r for r in rows if r["exp_id"] == exp_id]
    assert row["n_matched"] == 500
    return 100 * row["accuracy"]


def collect_hard100(main, main_scenes):
    suite_path = CL / "hard100/hard100_suite.csv"
    suite_rows = list(csv.DictReader(source_bytes(suite_path).decode().splitlines()))
    suite = {row["scene_id"] for row in suite_rows}
    assert len(suite_rows) == len(suite) == 100
    assert {row["test_suite_id"] for row in suite_rows} == {"public_2601_hard100"}
    assert suite.isdisjoint(main_scenes["Dense"]), "Hard100 overlaps the main 150 scenes"
    reference = {row["method"]: row for row in main}
    rows, scores, observed_by_method = [], {}, {}
    metrics = ("collision_at_fault", "offroad", "progress_clipped_rel")
    for label, checkpoint in [("Dense", "baseline"), ("LLM-Pruner", "lp_r50"),
                              ("Týr-the-Pruner", "slim_tyr_u40_r"),
                              ("Týr-K", "slim_tyrK"), ("Dual", "slim_dual_u40_v2")]:
        path = CL / f"h100_merged_{checkpoint}/aggregate/results-summary.json"
        scenes = scene_values(path, n_scenes=100)
        assert scenes.keys() == suite, (label, "Hard100 suite mismatch")
        scores[label] = scenes
        rollouts = read(path)["rollouts"]
        observed, missing = {}, []
        for rollout in rollouts:
            values = {key: rollout["metrics"].get(key) for key in metrics}
            if any(value is None for value in values.values()):
                assert all(value is None for value in values.values())
                assert rollout["status"] == "fail" and rollout["score"] == 0
                missing.append({"scene_id": rollout["clipgt_id"],
                                "rollout_id": rollout["rollout_id"],
                                "reason": rollout["failure_reason"]})
                continue
            assert all(np.isfinite(value) for value in values.values())
            assert all(values[key] in (0, 1) for key in metrics[:2])
            observed.setdefault(rollout["clipgt_id"], []).append(rollout)
        assert all(len(group) == 2 for group in observed.values())
        observed_by_method[label] = observed
        valid = [r for group in observed.values() for r in group]
        # Guard the fixed missing-observation counts stated in the exported captions.
        assert len(valid) == (192 if label == "Dual" else 196), (label, len(valid))
        counts = {key: int(sum(r["metrics"][key] for r in valid)) for key in metrics[:2]}
        removed = (read(OUTPUTS / checkpoint / "slim_meta.json")["params"]["removed"]
                   if label == "Týr-K" else reference[label]["removed"])
        rows.append({"method": label, "checkpoint": checkpoint,
                     "removed": removed, "removed_pct": 100 * removed / FULL,
                     "closed_score": mean(scenes),
                     # Preserve the historical report's recorded-event / scheduled denominator.
                     # Missing metrics remain explicitly missing, not imputed observations.
                     **{key: count / len(rollouts) for key, count in counts.items()},
                     "progress_clipped_rel": float(np.mean([
                         r["metrics"]["progress_clipped_rel"] for r in valid])),
                     "recorded_event_counts": counts, "scheduled_rollouts": len(rollouts),
                     "observed_rollouts": len(valid), "missing_metrics": missing,
                     "closed_delta_vs_dense": None if label == "Dense" else paired(
                         scenes, scores["Dense"])})

    common = set.intersection(*(set(values) for values in observed_by_method.values()))
    assert len(common) == 96, len(common)
    common_rows = []
    for row in rows:
        label = row["method"]
        valid = [r for scene in sorted(common) for r in observed_by_method[label][scene]]
        common_rows.append({"method": label, "checkpoint": row["checkpoint"],
                            "removed_pct": row["removed_pct"], "observed_rollouts": len(valid),
                            "closed_score": float(np.mean([r["score"] for r in valid])),
                            **{key: float(np.mean([r["metrics"][key] for r in valid]))
                               for key in metrics}})
    return {"suite": "public_2601_hard100", "n_scenes": 100, "rollouts_per_scene": 2,
            "scene_ids": sorted(suite), "rows": rows,
            "aggregation": "Scene score: all 100 scenes, including stored zero scores on "
                           "failed rollouts. Event rates: recorded events / 200 scheduled "
                           "rollouts, matching the historical report. Progress: mean over "
                           "observed rollouts only (Dual 192; other methods 196).",
            "common_observed": {"n_scenes": len(common), "scene_ids": sorted(common),
                                "excluded_scene_ids": sorted(suite - common),
                                "rows": common_rows,
                                "aggregation": "Every metric uses the same scenes with "
                                               "complete observations for all five methods."},
            "dual_comparisons": {label: paired(scores["Dual"], values)
                                 for label, values in scores.items() if label != "Dual"},
            "unavailable": ["Wanda", "Traj", "CoC", "Dual + expert MLP"]}


def collect():
    # Tyr's searched, reconstructed checkpoint is used consistently across all metrics.
    specs = [
        ("Dense", "baseline", "baseline_ada_ps", "oodval", "lingo_vqa_scores"),
        ("Wanda", "slim_wanda_u40_v2", "wanda_u40_v2", "oodval",
         "lingo_vqa_scores_wanda_u40_v2"),
        ("LLM-Pruner", "lp_r50", None, None, None),
        ("Týr-the-Pruner", "slim_tyr_u40_r", "tyr_u40_r", "oodval", "lingo_vqa_scores_tyr"),
        ("Traj", "slim_traj_u40_v2", "traj_u40_v2", "ood", "lingo_vqa_scores_extra"),
        ("CoC", "slim_coc_u40_v2", "coc_u40_v2", "ood", "lingo_vqa_scores_extra"),
        ("Dual", "slim_dual_u40_v2", "dual_u40_v2_ps", "ood", "lingo_vqa_scores"),
    ]
    main, all_clips, all_fde, all_scenes = [], {}, {}, {}
    lp_lingo = read(LP / "lingo_vqa_analysis/metrics.json")["per_run"]
    for label, checkpoint, prefix, ood_suffix, lingo_dir in specs:
        if label == "LLM-Pruner":
            clip_sources = {
                split: (LP / f"lp_k6_{suffix}", "lp_r50_s*of*.json", split == "ood")
                for split, suffix in (("val", "indist"), ("test", "test"), ("ood", "ood"))
            }
            closed_path = LP_CL / "cl150_merged_lp_r50/aggregate/results-summary.json"
            qa = lp_lingo["lingo_vqa_lp_r50"]
            assert qa["n_questions"] == 500 and qa["arm"] == "outputs/lp_r50"
            accuracy = qa["accuracy"] * 100
            meta = read(LP / "lp_r50/slim_meta.json")
            assert meta["config"]["arm"] == "coc_param_first"
            removed = meta["params"]["removed"]
        else:
            clip_sources = {
                split: (OUTPUTS / f"{prefix}_{suffix}", "*_s*of*.json", split == "ood")
                for split, suffix in (("val", "indist"), ("test", "test"), ("ood", ood_suffix))
            }
            closed_path = CL / f"m2601_merged_{checkpoint}/aggregate/results-summary.json"
            accuracy = lingo(OUTPUTS / lingo_dir, f"lingo_vqa_{checkpoint}")
            removed = 0 if checkpoint == "baseline" else read(
                OUTPUTS / checkpoint / "slim_meta.json"
            )["params"]["removed"]
        clips = {split: clip_values(*source) for split, source in clip_sources.items()}
        fde = {split: clip_values(*source, metric="fde")
               for split, source in clip_sources.items()}
        for split in clips:
            assert clips[split].keys() == fde[split].keys()
        scenes = scene_values(closed_path)
        all_clips[label], all_fde[label], all_scenes[label] = clips, fde, scenes
        main.append({"method": label, "checkpoint": checkpoint, "removed": removed,
                     "removed_pct": 100 * removed / FULL,
                     **{f"{k}_ade6": mean(v) for k, v in clips.items()},
                     **{f"{k}_fde6": mean(v) for k, v in fde.items()},
                     "closed_score": mean(scenes), **closed_metrics(closed_path),
                     "lingo_pct": accuracy})
    for label, splits in all_clips.items():
        for split, values in splits.items():
            assert values.keys() == all_clips["Dense"][split].keys(), (label, split)
        assert all_scenes[label].keys() == all_scenes["Dense"].keys(), label

    # A clearly labelled backup variant; never splice its LingoQA into the lp_r50 row.
    lp_dual = {split: clip_values(LP / f"lp_k6_{suffix}", "lp_r50_dual_s*of*.json", split == "ood")
               for split, suffix in (("val", "indist"), ("test", "test"), ("ood", "ood"))}
    lp_dual_summary = {**{f"{k}_ade6": mean(v) for k, v in lp_dual.items()},
                       "lingo_pct": 100 * lp_lingo["lingo_vqa_lp_r50_dual"]["accuracy"],
                       "closed_score": mean(scene_values(
                           LP_CL / "cl150_merged_lp_r50_dual/aggregate/results-summary.json"))}

    performance_keys = [f"{split}_{metric}6" for split in ("val", "test", "ood")
                        for metric in ("ade", "fde")]
    performance_keys += ["closed_score", "collision_at_fault", "offroad", "progress_clipped_rel"]
    composition = [{"method": "Dual", "kept": 8256, "removed_pct": main[-1]["removed_pct"],
                    **{key: main[-1][key] for key in performance_keys},
                    "closed_delta": None, "open_delta": None, "open_fde_delta": None}]
    for rung, label, kept in [("75", "Dual + MLP75", 2064),
                              ("87p5", "Dual + MLP87.5", 1032),
                              ("93p75", "Dual + MLP93.75", 516)]:
        clips = {split: clip_values(OUTPUTS / f"dualexp_em{rung}_{suffix}", ood=split == "ood")
                 for split, suffix in (("val", "indist"), ("test", "test"), ("ood", "oodval"))}
        fde = {split: clip_values(OUTPUTS / f"dualexp_em{rung}_{suffix}",
                                 ood=split == "ood", metric="fde")
               for split, suffix in (("val", "indist"), ("test", "test"), ("ood", "oodval"))}
        for split in clips:
            assert clips[split].keys() == fde[split].keys()
        checkpoint = f"slim_dualexp_u40_em{rung}"
        closed_path = CL / f"m2601_merged_{checkpoint}/aggregate/results-summary.json"
        scenes = scene_values(closed_path)
        params = read(OUTPUTS / checkpoint / "slim_meta.json")["params"]
        composition.append({"method": label, "kept": kept,
                            "removed_pct": 100 * params["removed"] / FULL,
                            **{f"{k}_ade6": mean(v) for k, v in clips.items()},
                            **{f"{k}_fde6": mean(v) for k, v in fde.items()},
                            "closed_score": mean(scenes), **closed_metrics(closed_path),
                            "closed_delta": paired(scenes, all_scenes["Dual"]),
                            "open_delta": {s: paired(v, all_clips["Dual"][s], 5000)
                                           for s, v in clips.items()},
                            "open_fde_delta": {s: paired(v, all_fde["Dual"][s], 5000)
                                               for s, v in fde.items()}})

    axis = read(OUTPUTS / "expert_axis_analysis/metrics.json")
    assert axis["g0"]["pass"]
    axis_rows = []
    for label, key, a, b in [
        ("Q25 − MLP25\nAda · equal unit fraction", "G1 expert_q25-expert_m25",
         "expert_q25", "expert_m25"),
        ("Q25 − MLP4.13\nAda · matched parameters", "G4 expert_q25-expert_m_pm",
         "expert_q25", "expert_m_pm"),
        ("Q50 − MLP50\nBlackwell · equal unit fraction", "G3x expert_q50-expert_m50",
         "expert_q50", "expert_m50"),
    ]:
        axis_rows.append({"comparison": label, "q_removed": axis["g0"][a]["removed"],
                          "mlp_removed": axis["g0"][b]["removed"],
                          **axis["sets"]["indist"]["contrasts"][key]})
    calibration = read(OUTPUTS / "calibsize_eval/metrics.json")
    return {"main": main, "llm_pruner_dual_backup": lp_dual_summary,
            "metric_definitions": {
                "open_loop": "Mean over clips of independent per-metric minima over samples 0:6; "
                             "ADE/FDE in meters, trajectory horizon 6.4 s.",
                "collision_at_fault": "Fraction of rollouts with collision_at_fault=1; "
                                      "stored as fraction, tables show percent.",
                "offroad": "Fraction of rollouts with offroad=1; "
                           "stored as fraction, tables show percent.",
                "progress": "Mean progress_clipped_rel over all rollouts; "
                            "ratio relative to the full GT trajectory, not a percent.",
                "closed_score": "Average the two rollout scores in each scene, then average "
                                "over scenes. Main/expert MLP: 150 scenes (300 rollouts); "
                                "Hard100: 100 scenes (200 rollouts).",
                "hard100_missing_metrics": "Hard100 contains missing observations. See "
                                           "hard100.aggregation for the historical denominator "
                                           "and progress convention; common_observed uses "
                                           "96 scenes (192 rollouts) for every metric.",
                "lingo_pct": "Native VQA, unprompted, Lingo-Judge accuracy over 500 questions.",
                "composition_parentheses": "Performance cells show absolute mean (signed change "
                                           "from Dual). Rate changes are percentage points; "
                                           "other changes use the metric's original unit.",
            },
            "dual_vs_llm_closed": paired(all_scenes["Dual"], all_scenes["LLM-Pruner"]),
            "hard100": collect_hard100(main, all_scenes),
            "expert_axis": axis_rows, "composition": composition,
            "calibration": {k: {x: v[x] for x in ["score", "ci_lo", "ci_hi"]}
                            for k, v in calibration["arms"].items()}}


def signed(value):
    return f"{value:+.4f}".replace("-", "−")


def interval(value, lo="lo", hi="hi", point="mean"):
    return f"{signed(value[point])}\n[{signed(value[lo])}, {signed(value[hi])}]"


def table(out, name, headers, rows, widths, notes, highlight=(), separators=(), fontsize=19,
          figure_width=13.33, groups=()):
    """Draw a spacious table with fixed column widths and no rasterized text."""
    row_height = 0.90 if any("\n" in cell for row in rows for cell in row) else 0.59
    group_height = 0.55 if groups else 0
    height = 1.4 + group_height + len(rows) * row_height + 0.27 * len(notes)
    fig, ax = plt.subplots(figsize=(figure_width, height))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.98, bottom=0.02)
    ax.set(xlim=(0, 1), ylim=(0, height))
    ax.axis("off")
    bounds = np.r_[0, np.cumsum(widths)]
    assert np.isclose(bounds[-1], 1)
    top = height - 0.15
    header_bottom = top - group_height - 0.90
    ax.plot([0, 1], [top, top], color=INK, lw=1.5)
    cells = []

    def cell(text, col, y, size, bold=False, color=INK):
        left, right = bounds[col:col + 2]
        x = left + 0.012 if col == 0 else (left + right) / 2
        artist = ax.text(x, y, text, ha="left" if col == 0 else "center", va="center",
                         fontsize=size, weight="bold" if bold else "normal", color=color,
                         linespacing=1.40)
        cells.append((artist, left, right))

    for start, end, label in groups:
        left, right = bounds[start], bounds[end]
        artist = ax.text((left + right) / 2, top - 0.25, label, ha="center", va="center",
                         fontsize=fontsize, weight="bold", color=INK)
        cells.append((artist, left, right))
        ax.plot([left + 0.005, right - 0.005], [top - 0.50] * 2, color="#B8C5CD", lw=0.7)
    for j, header in enumerate(headers):
        cell(header, j, top - group_height - 0.44, fontsize - 1, bold=True)
    ax.plot([0, 1], [header_bottom, header_bottom], color=INK, lw=0.8)
    for i, row in enumerate(rows):
        y = header_bottom - (i + 0.5) * row_height
        if i in highlight:
            ax.add_patch(Rectangle((0, y - row_height / 2), 1, row_height,
                                   facecolor="#E7F3F2", edgecolor="none"))
        elif i == 0 and row[0] in ("Dense", "Dual"):
            ax.add_patch(Rectangle((0, y - row_height / 2), 1, row_height,
                                   facecolor="#F2F5F7", edgecolor="none"))
        if i in separators:
            ax.plot([0, 1], [y + row_height / 2, y + row_height / 2], color="#B8C5CD", lw=0.6)
        for j, text in enumerate(row):
            cell(text, j, y, fontsize, bold=i in highlight, color=TEAL if i in highlight else INK)
    bottom = header_bottom - len(rows) * row_height
    ax.plot([0, 1], [bottom, bottom], color=INK, lw=1.0)
    for i, note in enumerate(notes):
        artist = ax.text(0.012, bottom - 0.28 - 0.27 * i, note, fontsize=11.8, color=MUTED,
                         ha="left", va="center")
        cells.append((artist, 0, 1))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, left, right in cells:
        box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        assert box.x0 >= left - 0.003 and box.x1 <= right + 0.003, (name, artist.get_text())
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    with (out / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        prefixes = {col: label + " / " for start, end, label in groups
                    for col in range(start, end)}
        writer.writerow([(prefixes.get(j, "") + h).replace("\n", " ")
                         for j, h in enumerate(headers)])
        writer.writerows([[cell.replace("\n", " ") for cell in row] for row in rows])
    print(name)


def render(data, out):
    main = data["main"]
    method_note = "LLM-Pruner: CoC / param_first, layers 4–33. " \
                  "Týr: searched allocation + reconstruction."
    open_note = "Clip means of minADE@6 / minFDE@6 (m); " \
                "each minimum is taken independently over the first 6 samples. Horizon: 6.4 s."
    closed_note = "150 scenes × 2 rollouts. Collision@fault / offroad: affected rollouts (%). " \
                  "Progress: mean progress_clipped_rel (ratio)."
    qa_note = "Lingo-Judge accuracy on 500 questions; native VQA, unprompted."
    open_headers = ["minADE@6 ↓\n(m)", "minFDE@6 ↓\n(m)"]
    compact_open_headers = ["minADE@6\n↓ (m)", "minFDE@6\n↓ (m)"]
    closed_headers = ["Scene\nscore ↑", "Collision@fault ↓\n(%)", "Offroad ↓\n(%)",
                      "Progress ↑\n(ratio)"]

    def base_cells(row):
        # User-requested display override in every performance table; keep raw metadata intact.
        removed = "24.0" if row["method"] == "LLM-Pruner" else f'{row["removed_pct"]:.1f}'
        return [row["method"], removed]

    def open_cells(row, splits):
        return [f'{row[f"{s}_{metric}6"]:.4f}' for s in splits for metric in ("ade", "fde")]

    def closed_cells(row):
        return [f'{row["closed_score"]:.4f}', f'{100 * row["collision_at_fault"]:.1f}',
                f'{100 * row["offroad"]:.1f}', f'{row["progress_clipped_rel"]:.4f}']

    expert_9375, = [row for row in data["composition"] if row["method"] == "Dual + MLP93.75"]
    driving_rows = [*main, expert_9375]
    expert_note = "Dual + MLP93.75: 93.75% expert MLP removal (516 channels/layer); " \
                  "all expert Q heads retained. Full-model removal: 39.4%."
    table(out, "table_01_open_loop", ["Method", "Removed\n(%)", *open_headers * 2],
          [[*base_cells(r), *open_cells(r, ("test", "ood"))] for r in driving_rows],
          [0.25, 0.11, 0.16, 0.16, 0.16, 0.16], [open_note, method_note, expert_note],
          groups=((2, 4, "Test500"), (4, 6, "OOD-val262")),
          highlight=(6, 7), separators=(1, 4, 6, 7), fontsize=18)
    table(out, "table_01_closed_loop", ["Method", "Removed\n(%)", *closed_headers],
          [[*base_cells(r), *closed_cells(r)] for r in driving_rows],
          [0.25, 0.11, 0.15, 0.19, 0.14, 0.16], [closed_note, method_note, expert_note],
          highlight=(6, 7), separators=(1, 4, 6, 7), fontsize=18)
    table(out, "table_01_lingoqa", ["Method", "Removed\n(%)", "LingoQA ↑\nAccuracy (%)"],
          [[*base_cells(r), f'{r["lingo_pct"]:.1f}'] for r in main],
          [0.50, 0.22, 0.28], [qa_note, method_note],
          highlight=(6,), separators=(1, 4, 6), fontsize=20)

    # The full comparison is a reference image; the three separate tables fit slides.
    table(out, "table_01_main_results",
          ["Method", "Removed\n(%)", *compact_open_headers * 2, *closed_headers,
           "LingoQA ↑\nAcc. (%)"],
          [[*base_cells(r), *open_cells(r, ("test", "ood")), *closed_cells(r),
            f'{r["lingo_pct"]:.1f}'] for r in main],
          [0.18, 0.065, *[0.075] * 4, 0.08, 0.105, 0.08, 0.085, 0.105],
          [open_note, closed_note, qa_note, method_note],
          groups=((2, 4, "Test500"), (4, 6, "OOD-val262"),
                  (6, 10, "Closed-loop · 150 scenes × 2 rollouts")),
          highlight=(6,), separators=(1, 4, 6), fontsize=15, figure_width=20)
    table(out, "table_04_main_all_splits",
          ["Method", "Removed\n(%)", *compact_open_headers * 3, *closed_headers,
           "LingoQA ↑\nAcc. (%)"],
          [[*base_cells(r), *open_cells(r, ("val", "test", "ood")), *closed_cells(r),
            f'{r["lingo_pct"]:.1f}'] for r in main],
          [0.16, 0.06, *[0.07] * 6, 0.075, 0.09, 0.065, 0.065, 0.065],
          [open_note, closed_note, qa_note, method_note],
          groups=((2, 4, "Val500"), (4, 6, "Test500"), (6, 8, "OOD-val262"),
                  (8, 12, "Closed-loop · 150 scenes × 2 rollouts")),
          highlight=(6,), separators=(1, 4, 6), fontsize=15, figure_width=23.5)
    table(out, "table_02_expert_axis",
          ["Comparison", "Removed (M)\nQ / MLP", "Mean ΔminADE@6\n95% CI (m)",
           "Median ΔminADE@6\n95% CI (m)"],
          [[r["comparison"], f'{r["q_removed"] / 1e6:.1f} / {r["mlp_removed"] / 1e6:.1f}',
            interval(r, "mlo", "mhi"), interval(r, point="med")] for r in data["expert_axis"]],
          [0.35, 0.19, 0.23, 0.23],
          ["Dense VLM; val500. Δ = Q-head pruning minus MLP-channel pruning; "
           "positive means Q pruning has higher error.",
           "Each pair uses the same GPU architecture and identical generated CoC. "
           "Q25 removes 4/16 heads; Q50 removes 8/16.",
           "Parameter-matched MLP removes 341/8256 channels. "
           "Mean and median are distinct estimands."], fontsize=16)
    composition = data["composition"]
    parent = composition[0]
    composition_headers = ["Model", "Kept MLP\nch. / layer", "Removed\n(%)"]
    composition_note = "MLP75 / 87.5 / 93.75: expert MLP removal (%). " \
                       "All expert Q heads retained; plain Dual VLM, no refit."
    delta_note = "Absolute value (change from Dual); ref. = Dual reference. " \
                 "Rate changes use percentage points (pp)."

    def composition_base(row):
        return [row["method"], f'{row["kept"]:,}', f'{row["removed_pct"]:.1f}']

    def absolute_delta(row, key, rate=False):
        scale, digits = (100, 1) if rate else (1, 4)
        value = row[key] * scale
        change = (row[key] - parent[key]) * scale
        delta = f"{change:+.{digits}f}".replace("-", "−") + (" pp" if rate else "")
        suffix = "ref." if row["method"] == "Dual" else delta
        return f"{value:.{digits}f}\n({suffix})"

    def composition_open(row):
        return [absolute_delta(row, f"{split}_{metric}6")
                for split in ("test", "ood") for metric in ("ade", "fde")]

    def composition_closed(row):
        return [absolute_delta(row, key, rate=key in ("collision_at_fault", "offroad"))
                for key in ("closed_score", "collision_at_fault", "offroad",
                            "progress_clipped_rel")]

    table(out, "table_03_expert_open_loop", [*composition_headers, *open_headers * 2],
          [[*composition_base(r), *composition_open(r)] for r in composition],
          [0.23, 0.11, 0.10, *[0.14] * 4],
          ["Absolute value (change from Dual); ref. = Dual reference.",
           open_note, composition_note],
          groups=((3, 5, "Test500"), (5, 7, "OOD-val262")), fontsize=16)
    table(out, "table_03_expert_closed_loop", [*composition_headers, *closed_headers],
          [[*composition_base(r), *composition_closed(r)] for r in composition],
          [0.22, 0.11, 0.10, 0.13, 0.18, 0.12, 0.14],
          [delta_note, closed_note, composition_note], fontsize=16)
    table(out, "table_03_dual_expert_mlp",
          [*composition_headers, *compact_open_headers * 2, *closed_headers],
          [[*composition_base(r), *composition_open(r), *composition_closed(r)]
           for r in composition],
          [0.16, 0.07, 0.06, 0.09, 0.09, 0.09, 0.09, 0.08, 0.10, 0.08, 0.09],
          [delta_note, open_note, closed_note, composition_note],
          groups=((3, 5, "Test500"), (5, 7, "OOD-val262"),
                  (7, 11, "Closed-loop · 150 scenes × 2 rollouts")),
          fontsize=16, figure_width=21)

    table(out, "table_06_hard100_closed_loop", ["Method", "Removed\n(%)", *closed_headers],
          [[*base_cells(r), *closed_cells(r)] for r in data["hard100"]["rows"]],
          [0.25, 0.11, 0.15, 0.19, 0.14, 0.16],
          ["Hard100: 100 scenes × 2 scheduled rollouts. Scene score includes stored zeros "
           "for route-check failures.",
           "Event rates: recorded events / 200 (%). Missing metrics: Dual 8 rollouts; "
           "other methods 4 rollouts each.",
           "Progress: mean progress_clipped_rel over observed rollouts only "
           "(Dual: 192; other methods: 196).",
           "Týr: KL + MSE search fitness. Týr-K: KL-only search fitness. "
           "Both use searched allocation + reconstruction."],
          highlight=(4,), separators=(1, 4), fontsize=18)
    common = data["hard100"]["common_observed"]
    common_rows = [row for row in common["rows"] if row["method"] != "Týr-K"]
    table(out, "table_06_hard100_common_valid",
          ["Method", "Removed\n(%)", *closed_headers],
          [[*base_cells(r), *closed_cells(r)] for r in common_rows],
          [0.25, 0.11, 0.15, 0.19, 0.14, 0.16],
          [f'Hard100 common observations: {common["n_scenes"]} scenes × 2 rollouts. '
           "All columns use the same scenes for all four methods.",
           "Scenes with missing metrics in any method are excluded. "
           "Event rates use observed rollouts; progress is a ratio.",
           "Týr: KL + MSE search fitness; searched allocation + reconstruction."],
          highlight=(3,), separators=(1, 3), fontsize=18)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    keys = ["baseline", "calib100", "st2000_a", "st2000_b"]
    labels = ["Dense", "Dual\n100 clips", "Dual\n2,000 clips A", "Dual\n2,000 clips B"]
    for i, key in enumerate(keys):
        value = data["calibration"][key]
        score = value["score"]
        color = TEAL if key == "calib100" else MUTED
        ax.errorbar(i, score, yerr=[[score - value["ci_lo"]], [value["ci_hi"] - score]],
                    fmt="o", color=color, capsize=6, markersize=8, linewidth=1.8)
        ax.text(i, value["ci_hi"] + 0.012, f"{score:.3f}", ha="center", fontsize=17,
                color=color)
    ax.axhline(data["calibration"]["baseline"]["score"], color=MUTED, ls="--", lw=1)
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Closed-loop scene score ↑",
           ylim=(0.62, 0.92), xlim=(-0.5, 3.5))
    ax.tick_params(labelsize=15)
    ax.yaxis.label.set_size(16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5EBEF", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.10, 0.035,
             "150 scenes × 2 rollouts; 95% scene-bootstrap CI. All Dual arms remove 24.0%.\n"
             "Calibration size and clip membership both change; "
             "this is not a size-only experiment.",
             fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.23)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_05_calibration_closedloop.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("fig_05_calibration_closedloop")


def copy_analysis_figures(out):
    """Collect the existing publication panels without altering their artwork."""
    names = ["fig1_depth_q_head", "fig1_depth_mlp", "fig1_rank_agreement", "fig1_kept_overlap",
             "fig2_steps_q_head", "fig2_steps_mlp", "fig2_mass_curve", "fig2_stability_vs_mass"]
    assets = []
    for name in names:
        for ext in ("png", "pdf", "svg"):
            source = REPO / "figures" / f"{name}.{ext}"
            destination = out / source.name
            if source.resolve() != destination.resolve():
                destination.write_bytes(source_bytes(source))
            assets.append({"file": destination.name, "source": str(source)})
    print(f"Copied {len(names)} analysis figures (PNG/PDF/SVG)")
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "ppt-fig")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.style": "normal",
                         "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
                         "text.usetex": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path"})
    data = collect()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render(data, args.out_dir)
    data["analysis_figures"] = copy_analysis_figures(args.out_dir)
    data["sources_sha256"] = SOURCES
    (args.out_dir / "meeting_table_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated {len(SOURCES)} source files; saved to {args.out_dir}")


if __name__ == "__main__":
    main()
