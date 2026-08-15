"""Score predictions.csv with LingoQA's own metric, mirroring their evaluate.py.

Why not just run `LingoQA/benchmark/evaluate.py`: it imports `click` and `datasets`,
neither of which is in this repo's venv, and installing them is not free -- a --target
install pulled a broken pyarrow that shadowed the venv's working one, and installing
into the venv itself would perturb the environment other experiments are running against.

What is reproduced versus what is not:

  the metric      `LingoJudge` is imported verbatim from the LingoQA checkout. Nothing
                  about the scoring is reimplemented: the model, the
                  "[CLS]\\nQuestion:...\\nAnswer:...\\nStudent:..." serialization, the
                  lowercase/strip preprocessing, and the max over the two references
                  (Eq. 1 of the paper) all come from their class.
  the threshold   `score > 0.0` on the raw logit, exactly as evaluate.py's
                  `evaluate_question` does. Equivalent to the README demo's
                  sigmoid > 0.5.
  the score       correct / total, as `dataset_filtered.num_rows / dataset_evaluated.num_rows`.
  the plumbing    evaluate.py wraps the merged frame in a HF `Dataset` and uses
                  .map(batched=True, batch_size=1).filter(...). With the default
                  batch_size that is a per-row loop, so iterating the dataframe gives
                  identical results. That wrapper is the only thing rewritten here.

The reference join is theirs too: group the 1,000-row parquet by
(question_id, segment_id, question) with .agg(list) so the two human answers collapse
into one `references` list, then inner-join the predictions on
(question_id, segment_id). `matched` must be 500; anything less means rows were dropped
and the score covers a subset, which evaluate.py only warns about.

Usage:
  python experiments/lingoqa/score_lingo_vqa.py --runs lingo_vqa_baseline lingo_vqa_...
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LINGOQA = Path("/home/cvlab21/project/chan/LingoQA/benchmark")
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(LINGOQA))

import lingo_lib as ll
from judge import LingoJudge

EXPECTED = 500


def load_references():
    """The reference side of evaluate.py's join, from the local copy of val.parquet."""
    ref = pd.read_parquet(ll.DATA / "val.parquet")
    ref = ref[["question_id", "segment_id", "question", "answer"]]
    ref = ref.groupby(["question_id", "segment_id", "question"]).agg(list)
    ref = ref.rename({"answer": "references"}, axis=1).reset_index()
    return ref


def score_run(judge, exp_id, ref, device):
    d = REPO / "outputs" / exp_id
    pred = pd.read_csv(d / "predictions.csv")
    pred = pred.rename({"answer": "prediction"}, axis=1)
    merged = pd.merge(pred, ref, on=["question_id", "segment_id"])

    rows = []
    for r in merged.itertuples():
        s = judge.compute([r.question], [list(r.references)], [r.prediction])
        rows.append({"question_id": r.question_id, "segment_id": r.segment_id,
                     "question": r.question, "prediction": r.prediction,
                     "references": list(r.references), "score": float(s[0]),
                     "correct": bool(float(s[0]) > 0.0)})
    acc = sum(x["correct"] for x in rows) / len(rows)
    (d / "scored.json").write_text(json.dumps(rows, indent=2))

    cfg = json.loads((d / "config.json").read_text())
    gen = json.loads((d / "rows.json").read_text())
    words = [len(g["answer"].split()) for g in gen]
    return {"exp_id": exp_id, "arm": cfg.get("arm"), "style": cfg.get("style"),
            "n_predictions": len(pred), "n_matched": len(merged),
            "accuracy": acc, "mean_logit": float(sum(x["score"] for x in rows) / len(rows)),
            "answer_words_median": float(pd.Series(words).median()),
            "answer_words_mean": float(pd.Series(words).mean()),
            "truncated": int(sum(g["truncated"] for g in gen)),
            "truncated_frac": sum(g["truncated"] for g in gen) / len(gen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exp-id", default="lingo_vqa_scores")
    ap.add_argument("--gpu", default="4,5,6,7")
    args = ap.parse_args()

    device = ll.pick_gpu(2.0, [int(g) for g in args.gpu.split(",")])
    judge = LingoJudge().eval().to(device)
    ref = load_references()
    print(f"Loaded {len(ref)} references.", flush=True)

    out = []
    for exp_id in args.runs:
        m = score_run(judge, exp_id, ref, device)
        out.append(m)
        flag = "" if m["n_matched"] == EXPECTED else "  <-- NOT 500, subset only!"
        print(f"{exp_id:<42} matched {m['n_matched']:>4}  "
              f"acc {m['accuracy'] * 100:5.1f}%  trunc {m['truncated_frac'] * 100:5.1f}%{flag}",
              flush=True)

    lines = ["LingoQA zero-shot benchmark -- Lingo-Judge accuracy (threshold logit > 0)",
             "metric imported verbatim from LingoQA/benchmark/judge.py", "",
             f"{'run':<42}{'style':<12}{'matched':>8}{'acc':>8}{'words':>7}{'trunc':>8}"]
    for m in out:
        lines.append(f"{m['exp_id']:<42}{m['style'] or '-':<12}{m['n_matched']:>8}"
                     f"{m['accuracy'] * 100:>7.1f}%{m['answer_words_median']:>7.0f}"
                     f"{m['truncated_frac'] * 100:>7.1f}%")
    summary = "\n".join(lines) + "\n"
    ll.write_outputs(REPO / "outputs" / args.exp_id,
                     {"runs": args.runs, "threshold": "logit > 0.0",
                      "judge": "wayveai/Lingo-Judge",
                      "reference_join": "groupby(question_id, segment_id, question).agg(list)"},
                     {"runs": out}, summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
