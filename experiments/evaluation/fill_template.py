"""Fill {{path|fmt}} tokens in a report template from a metrics JSON, so no number in a
report is typed by hand.

  {{g2.step_marginal_move.9|.3f}}   -> metrics["g2"]["step_marginal_move"][9], formatted
  {{g0.all_block_move.0|+.2f}}      -> sign-forced float
  {{n_clips}}                       -> default: ints as-is, floats with 4 decimals
  {{g3.skippable_share|.1%}}        -> percent
A path that does not resolve is an error, not a blank cell.

Usage:
  .venv/bin/python experiments/evaluation/fill_template.py <metrics.json> <template.html> <out.html>
"""

import json
import re
import sys
from pathlib import Path


def resolve(metrics, path):
    cur = metrics
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def render(metrics, text):
    def sub(m):
        tok = m.group(1).strip()
        path, _, fmt = tok.partition("|")
        v = resolve(metrics, path.strip())
        if fmt:
            return format(v, fmt.strip())
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


def main():
    metrics_path, template, out = (Path(a) for a in sys.argv[1:4])
    metrics = json.loads(metrics_path.read_text())
    out.write_text(render(metrics, template.read_text()))
    print(f"rendered {template} -> {out}")


if __name__ == "__main__":
    main()
