"""Colour the master table and pack it as .xlsx, which Drive converts with fills intact.

A CSV upload cannot carry formatting, so the analysis sheet arrived as undifferentiated
grey. This writes the same table as a styled workbook: two tabs, `master` and `legend`.

Colour is applied only where it carries a reading, and the readings do not share a
direction -- that is the whole reason to encode it:

  val/test/ood_d   minADE delta, LOWER is better; green <= 0, red as it grows
  cl_dscore        alpasim score delta, HIGHER is better; the scale is INVERTED
  degen            CoC degeneracy rate, lower is better
  lingo_pct        Lingo-Judge %, higher is better (unpruned reference is 73.2)

A delta whose bootstrap CI includes zero is greyed rather than coloured, so "no
significant change" never reads as a result. Track groups alternate a faint band across
the identity columns so the blocks separate without competing with the metric colours.

Written against the stdlib -- the venv has no openpyxl -- reusing the cell helpers from
`make_results_xlsx.py`.

Usage:
  .venv/bin/python experiments/evaluation/make_master_xlsx.py
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from make_results_xlsx import CONTENT_TYPES, ROOT_RELS, col_name, is_number

INK, MUTED = "FF29261B", "FF8A8578"
# pale enough that black text stays readable in either Sheets theme
GOOD2, GOOD1 = "FFCDE9CD", "FFE8F4E8"
WARN = "FFFAEFC9"
BAD1, BAD2, BAD3 = "FFFBE3DE", "FFF4C2B8", "FFE79B8B"
BAND = "FFF4F2ED"
HEADER = "FF29261B"

IDENT = ["track", "method", "ckpt", "role", "budget", "prune_pct", "arch", "vs"]
NUM3 = {"val_ade6", "val_fde6", "val_degen", "val_d", "test_ade6", "test_fde6",
        "test_degen", "test_d", "ood_ade6", "ood_fde6", "ood_degen", "ood_d",
        "cl_score", "cl_dscore", "cl_collision", "cl_offroad", "cl_progress"}
DELTA = {"val_d": "val_sig", "test_d": "test_sig", "ood_d": "ood_sig"}
DEGEN = {"val_degen", "test_degen", "ood_degen"}
LOWER_BETTER = {"cl_collision", "cl_offroad"}


class StyleBook:
    """Interns (bold, colour, fill, numfmt) into the cellXfs table xlsx wants."""

    def __init__(self):
        self.fonts = [(False, INK), (True, INK), (False, MUTED), (True, "FFFFFFFF")]
        self.fills = [None, HEADER, BAND, GOOD2, GOOD1, WARN, BAD1, BAD2, BAD3]
        self.numfmts = ["General", "0.000", "0.0"]
        self.xfs, self.index = [], {}
        self.get(0, 0, 0)                       # xf 0 must be the default

    def get(self, font, fill, numfmt):
        key = (font, fill, numfmt)
        if key not in self.index:
            self.index[key] = len(self.xfs)
            self.xfs.append(key)
        return self.index[key]

    def xml(self):
        fonts = "".join(f'<font><sz val="11"/><name val="Calibri"/>'
                        f'{"<b/>" if b else ""}<color rgb="{c}"/></font>'
                        for b, c in self.fonts)
        fills = ['<fill><patternFill patternType="none"/></fill>',
                 '<fill><patternFill patternType="gray125"/></fill>']
        for f in self.fills[1:]:
            fills.append(f'<fill><patternFill patternType="solid">'
                         f'<fgColor rgb="{f}"/><bgColor indexed="64"/>'
                         f"</patternFill></fill>")
        nfs = "".join(f'<numFmt numFmtId="{164 + i}" formatCode="{c}"/>'
                      for i, c in enumerate(self.numfmts[1:]))
        xfs = []
        for font, fill, nf in self.xfs:
            # fill 0 is "none"; our solid fills start at index 2 because xlsx reserves 1
            fid = 0 if fill == 0 else fill + 1
            nid = 0 if nf == 0 else 163 + nf
            xfs.append(f'<xf numFmtId="{nid}" fontId="{font}" fillId="{fid}" '
                       f'borderId="0" xfId="0" applyFont="1" applyFill="1" '
                       f'applyNumberFormat="1"/>')
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/'
                'spreadsheetml/2006/main">'
                f'<numFmts count="{len(self.numfmts) - 1}">{nfs}</numFmts>'
                f'<fonts count="{len(self.fonts)}">{fonts}</fonts>'
                f'<fills count="{len(fills)}">{"".join(fills)}</fills>'
                '<borders count="1"><border/></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
                'borderId="0"/></cellStyleXfs>'
                f'<cellXfs count="{len(self.xfs)}">{"".join(xfs)}</cellXfs>'
                "</styleSheet>")


def short(v):
    """4 significant digits; non-numeric text passes through untouched."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return v if f.is_integer() and abs(f) < 1e6 else f"{f:.4g}"


def band(v, edges, fills):
    """First fill whose edge v falls under; fills has one more entry than edges."""
    for e, f in zip(edges, fills):
        if v < e:
            return f
    return fills[-1]


def fill_for(col, value, row):
    """The colour a metric cell earns, or None. Direction is per column, not global."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if col in DELTA:
        if not row.get(DELTA[col]):
            return None                      # CI includes zero -> not a result
        if v <= 0:
            return 3                         # GOOD2: at or better than the reference
        return band(v, (0.01, 0.05, 0.15), (4, 5, 6, 8))
    if col == "cl_dscore":                   # higher is better: the scale flips here
        if v >= 0.05:
            return 3
        if v >= 0.01:
            return 4
        if v > -0.01:
            return None
        return 6 if v > -0.05 else 8
    if col in DEGEN:
        return band(v, (0.02, 0.10, 0.30), (None, 5, 6, 8))
    if col in LOWER_BETTER:
        return band(v, (0.05, 0.10), (4, 5, 6))
    if col == "lingo_pct":
        return band(v, (20, 40, 60, 70), (8, 6, 5, 4, 3))
    if col == "cl_score":
        return band(v, (0.60, 0.75), (6, 5, 3))
    return None


def sheet_xml(rows, header, sb, styler, freeze_col=2):
    n_col = len(header)
    widths = [10] * n_col
    for r in [header] + rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], min(len(str(v)) + 2, 34))
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
                   for i, w in enumerate(widths))
    body = []
    for ri, r in enumerate([header] + rows, start=1):
        cells = []
        for ci, v in enumerate(r):
            if v == "" or v is None:
                continue
            ref = f"{col_name(ci)}{ri}"
            s = sb.get(3, 1, 0) if ri == 1 else styler(ri - 2, ci, v)
            if ri > 1 and is_number(v):
                cells.append(f'<c r="{ref}" s="{s}"><v>{v}</v></c>')
            else:
                cells.append(f'<c r="{ref}" s="{s}" t="inlineStr">'
                             f"<is><t>{escape(str(v))}</t></is></c>")
        body.append(f'<row r="{ri}">{"".join(cells)}</row>')
    dim = f"A1:{col_name(n_col - 1)}{len(rows) + 1}"
    pane = (f'<pane xSplit="{freeze_col}" ySplit="1" '
            f'topLeftCell="{col_name(freeze_col)}2" activePane="bottomRight" '
            'state="frozen"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dim}"/><sheetViews><sheetView workbookViewId="0">'
            f"{pane}</sheetView></sheetViews>"
            '<sheetFormatPr defaultRowHeight="15"/>'
            f"<cols>{cols}</cols><sheetData>{''.join(body)}</sheetData>"
            f'<autoFilter ref="{dim}"/></worksheet>')


def write_xlsx(path, tabs, sb):
    n = len(tabs)
    overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n + 1))
    sheets = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                     for i, (name, _) in enumerate(tabs, start=1))
    rels = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n + 1))
    rels += (f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/'
             'officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES.format(sheets=overrides))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
                   ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/'
                   f'relationships"><sheets>{sheets}</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
                   f'2006/relationships">{rels}</Relationships>')
        z.writestr("xl/styles.xml", sb.xml())
        for i, (_, xml) in enumerate(tabs, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", xml)


LEGEND = [
    ["colour", "means", "applies to"],
    ["green", "at or better than the reference", "val/test/ood_d <= 0, cl_dscore >= +0.05"],
    ["pale green", "slightly worse / slightly better", "d < 0.01 m; cl_dscore +0.01..+0.05"],
    ["yellow", "0.01-0.05 m worse", "val/test/ood_d"],
    ["orange", "0.05-0.15 m worse", "val/test/ood_d"],
    ["red", "> 0.15 m worse", "val/test/ood_d"],
    ["grey text", "bootstrap CI includes zero - not a result", "any *_d with empty *_sig"],
    ["", "", ""],
    ["DIRECTION", "minADE delta: LOWER is better. alpasim score delta: HIGHER is better.",
     "the two scales are deliberately inverted"],
    ["", "", ""],
    ["degen", "CoC degeneracy rate; yellow >0.02, orange >0.10, red >0.30", "*_degen"],
    ["lingo_pct", "Lingo-Judge %; unpruned reference is 73.2", "green >=70, red <20"],
    ["cl_score", "alpasim scene score; baseline is 0.750 over 150 scenes", "green >=0.75"],
    ["", "", ""],
    ["band", "alternating faint band separates track groups", "identity columns"],
    ["has", "O=open loop, C=closed loop, L=LingoQA", "which evaluations exist"],
    ["role", "anchor=unpruned, arm=candidate, control=isolates one factor", ""],
    ["vs", "the run each paired delta is measured against", ""],
    ["", "", ""],
    ["protocol", "rollout only; OOD cut to split=='val' (262 clips); minADE@6 mean", ""],
    ["arch", "Ada and Blackwell are not bitwise comparable - compare within one", ""],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results_index")
    ap.add_argument("--name", default="master.xlsx")
    args = ap.parse_args()

    index_dir = REPO / "outputs" / args.index
    lines = (index_dir / "master.csv").read_text().split("\n")
    reader = csv.reader(lines[5:])            # skip the CSV preamble; it becomes `legend`
    full = next(reader)
    dicts = [dict(zip(full, r)) for r in reader if r]
    # provenance columns stay in the CSV; the coloured view keeps `ckpt` as the identity
    # and 4 significant digits, which is the precision the reports quote anyway
    header = [c for c in full if c not in ("openloop_dir", "prune_pct")]
    rows = [[short(d.get(c, "")) for c in header] for d in dicts]

    # a track's rows share a band, and the band flips when the track changes
    bands, cur, flip = [], None, False
    for d in dicts:
        if d["track"] != cur:
            cur, flip = d["track"], not flip
        bands.append(flip)

    sb = StyleBook()

    def styler(ri, ci, value):
        col = header[ci]
        d = dicts[ri]
        numfmt = 1 if col in NUM3 else (2 if col in ("lingo_pct", "prune_pct") else 0)
        if col in IDENT or col == "has":
            bold = 1 if (col == "method" and d["role"] == "anchor") else 0
            font = 2 if d["role"] in ("control", "no open loop") else bold
            return sb.get(font, 2 if bands[ri] else 0, numfmt)
        fill = fill_for(col, value, d)
        muted = col in DELTA and not d.get(DELTA[col])
        return sb.get(2 if muted else 0, fill or 0, numfmt)

    tabs = [("master", sheet_xml(rows, header, sb, styler)),
            ("legend", sheet_xml(LEGEND[1:], LEGEND[0], sb,
                                 lambda ri, ci, v: sb.get(0, 0, 0), freeze_col=0))]
    out = index_dir / args.name
    write_xlsx(out, tabs, sb)
    print(f"{out}  {out.stat().st_size / 1024:.1f} KB  {len(rows)} rows  "
          f"{len(sb.xfs)} cell styles", flush=True)


if __name__ == "__main__":
    main()
