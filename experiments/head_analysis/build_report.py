"""Build a self-contained HTML report by inlining plots as base64 data URIs.

Placeholders in the template look like PLOT::<dir-key>::<filename.png>, where the
dir-key maps to an outputs/ subdirectory passed on the command line.

Usage:
  python build_report.py <template.html> <out.html> key=outputs/dir [key=... ...]
"""

import base64
import re
import sys
from pathlib import Path

REPO = Path("/workspace/alpamayo-model-compression")


def main():
    template, out = Path(sys.argv[1]), Path(sys.argv[2])
    dirs = {}
    for arg in sys.argv[3:]:
        key, path = arg.split("=", 1)
        dirs[key] = Path(path) if Path(path).is_absolute() else REPO / path

    html = template.read_text()
    n = 0

    def repl(m):
        nonlocal n
        key, fname = m.group(1), m.group(2)
        p = dirs[key] / fname
        b64 = base64.b64encode(p.read_bytes()).decode()
        n += 1
        return f"data:image/png;base64,{b64}"

    html = re.sub(r"PLOT::([\w-]+)::([\w.-]+\.png)", repl, html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    left = len(re.findall(r"PLOT::", html))
    print(f"{n} images embedded -> {out} {out.stat().st_size / 1024**2:.2f} MB")
    print(f"remaining placeholders: {left}")


if __name__ == "__main__":
    main()
