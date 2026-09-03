"""Stack the three collected tables into one CSV that Google Drive turns into one sheet.

Drive converts an uploaded CSV to a single-tab spreadsheet, so the three families are
written as three labelled blocks separated by a blank row -- the same layout the manual
"AD VLA Evaluation" sheet already uses. Each block keeps its own header row, so a reader
can select a block and sort it in place.

Only columns that carry information are dropped: none. Floats already arrive rounded to
6 significant digits from `collect_results.py`.

Usage:
  .venv/bin/python experiments/evaluation/make_sheet_csv.py
  .venv/bin/python experiments/evaluation/make_sheet_csv.py --index results_index
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BLOCKS = [
    ("OPEN LOOP  (PhysicalAI-AV; rollout only; minADE@k/minFDE@k means)", "openloop"),
    ("CLOSED LOOP  (alpasim; per-scene mean, paired vs baseline)", "closedloop"),
    ("LINGOQA  (Lingo-Judge; judge_pct = accuracy x 100)", "lingoqa"),
]
# the sheet is a view: provenance the CSVs keep in full (gpu string, tag, absolute param
# counts, the medians of the non-headline @8 metric, coc_empty/soup) is dropped here so
# the upload fits in one request. outputs/<index>/*.csv stays complete.
VIEW = {
    "openloop": ["arm", "set", "n_clips", "arch", "prune_pct", "minADE6_mean",
                 "minFDE6_mean", "minADE8_mean", "coc_degen", "d_minADE6_median",
                 "d_sig"],
    "closedloop": ["run", "n_scenes", "config", "score", "passed",
                   "collision_at_fault", "collision_any", "offroad", "wrong_lane",
                   "progress_clipped_rel", "dist_to_gt_trajectory", "coc_degen",
                   "d_score_mean", "wilcoxon_p"],
    "lingoqa": ["exp_id", "protocol", "style", "n_questions", "judge_pct",
                "mean_logit", "truncated_frac"],
}


def shorten(v):
    """4 significant digits -- what the reports quote, and it halves the payload."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return v if float(v).is_integer() and abs(f) < 1e6 else f"{f:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results_index")
    ap.add_argument("--name", default="sheet_upload.csv")
    args = ap.parse_args()

    index_dir = REPO / "outputs" / args.index
    cfg = json.loads((index_dir / "config.json").read_text())
    kst = datetime.timezone(datetime.timedelta(hours=9))
    stamp = datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M KST")

    out = index_dir / args.name
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([(f"AD VLA compression results - generated {stamp} by"
                     " experiments/evaluation/collect_results.py")])
        w.writerow([(f"protocol: {cfg.get('protocol', '')} | k={cfg.get('k')} |"
                     f" paired baseline: {cfg.get('paired_baseline', '')}")])
        w.writerow([("OOD rows are cut to split=='val' (262 clips). Runs predating"
                     " per-sample storage leave the @6 columns blank and keep"
                     " minADE8_*.")])
        w.writerow([("Ada and Blackwell are not bitwise comparable - compare within"
                     " one arch (see arch, baseline_ref).")])
        w.writerow([("this sheet is a view; the complete column set is in"
                     f" outputs/{args.index}/{{openloop,closedloop,lingoqa}}.csv")])
        for title, name in BLOCKS:
            with (index_dir / f"{name}.csv").open() as src:
                rows = list(csv.DictReader(src))
            cols = VIEW[name]
            w.writerow([])
            w.writerow([title, f"{len(rows)} rows"])
            w.writerow(cols)
            for r in rows:
                w.writerow([shorten(r.get(c, "")) for c in cols])
    print(f"{out}  {out.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
