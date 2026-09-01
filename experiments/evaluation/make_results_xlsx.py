"""Pack the collected CSVs into one .xlsx so Google Drive converts it to one
spreadsheet with one tab per result family.

Drive turns an uploaded CSV into a single-tab spreadsheet, so three CSVs would become
three separate files. An uploaded .xlsx converts with its tabs intact, which is the only
way to land openloop / closedloop / lingoqa side by side without the Sheets API.

Written against the stdlib on purpose -- an .xlsx is a zip of XML, and the research venv
has neither openpyxl nor xlsxwriter. Numbers are written as numbers (so the sheet can
sort and chart them), the header row is bold and frozen, and every sheet gets an
autofilter.

Usage:
  .venv/bin/python experiments/evaluation/make_results_xlsx.py
  .venv/bin/python experiments/evaluation/make_results_xlsx.py --index results_index
"""

import argparse
import csv
import datetime
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
SHEETS = ["README", "openloop", "closedloop", "lingoqa"]
MAX_WIDTH = 44

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{sheets}
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

# two fonts (plain, bold) -> cellXfs 0 = body, 1 = header
STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="1"><fill><patternFill patternType="none"/></fill></fills>
<borders count="1"><border/></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>
</styleSheet>"""


def col_name(i):
    """0 -> A, 26 -> AA."""
    s = ""
    while True:
        s = chr(ord("A") + i % 26) + s
        i = i // 26 - 1
        if i < 0:
            return s


def is_number(v):
    """Numeric cells sort and chart; everything else stays an inline string."""
    if v == "" or v is None:
        return False
    s = str(v)
    try:
        float(s)
    except ValueError:
        return False
    return not (len(s) > 1 and s[0] == "0" and s[1] != ".")   # keep ids like 0042 as text


def sheet_xml(rows, sst):
    """One worksheet: bold frozen header, autofilter, numbers as numbers.

    Strings are indices into the shared-string table -- `arch` and `set` repeat on every
    row, and inline strings made the workbook a third larger for no gain."""
    n_col = max((len(r) for r in rows), default=1)
    widths = [10] * n_col
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], min(len(str(v)) + 2, MAX_WIDTH))
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
                   for i, w in enumerate(widths))
    body = []
    for ri, r in enumerate(rows, start=1):
        cells = []
        for ci, v in enumerate(r):
            if v == "" or v is None:
                continue
            ref = f"{col_name(ci)}{ri}"
            style = ' s="1"' if ri == 1 else ""
            if ri > 1 and is_number(v):
                cells.append(f'<c r="{ref}"{style}><v>{v}</v></c>')
            else:
                cells.append(f'<c r="{ref}"{style} t="s"><v>{sst[str(v)]}</v></c>')
        body.append(f'<row r="{ri}">{"".join(cells)}</row>')
    dim = f"A1:{col_name(n_col - 1)}{max(len(rows), 1)}"
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dim}"/>'
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            "</sheetView></sheetViews>"
            '<sheetFormatPr defaultRowHeight="15"/>'
            f"<cols>{cols}</cols><sheetData>{''.join(body)}</sheetData>"
            f'<autoFilter ref="{dim}"/></worksheet>')


def build_sst(tabs):
    """string -> index, in first-seen order; header cells are strings too."""
    sst = {}
    for ri_tabs, (_, rows) in enumerate(tabs):
        for ri, r in enumerate(rows):
            for v in r:
                if v == "" or v is None:
                    continue
                s = str(v)
                if (ri == 0 or not is_number(v)) and s not in sst:
                    sst[s] = len(sst)
    return sst


def sst_xml(sst):
    items = "".join(f'<si><t xml:space="preserve">{escape(s)}</t></si>' for s in sst)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            f'count="{len(sst)}" uniqueCount="{len(sst)}">{items}</sst>')


def write_xlsx(path, tabs):
    """tabs: list of (name, rows); rows[0] is the header."""
    n = len(tabs)
    sst = build_sst(tabs)
    overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n + 1))
    overrides += ('\n<Override PartName="/xl/sharedStrings.xml" ContentType="application/'
                  'vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>')
    sheets = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                     for i, (name, _) in enumerate(tabs, start=1))
    rels = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n + 1))
    rels += (f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/'
             'officeDocument/2006/relationships/styles" Target="styles.xml"/>'
             f'<Relationship Id="rId{n + 2}" Type="http://schemas.openxmlformats.org/'
             'officeDocument/2006/relationships/sharedStrings" '
             'Target="sharedStrings.xml"/>')
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
        z.writestr("xl/styles.xml", STYLES)
        z.writestr("xl/sharedStrings.xml", sst_xml(sst))
        for i, (_, rows) in enumerate(tabs, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows, sst))


def readme_rows(index_dir, cfg, counts):
    """A first tab that says what the numbers are, so the file travels on its own."""
    kst = datetime.timezone(datetime.timedelta(hours=9))
    stamp = datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M KST")
    return [
        ["field", "value"],
        ["generated", stamp],
        ["generated by", ("experiments/evaluation/collect_results.py"
                          " + make_results_xlsx.py")],
        ["source", (f"outputs/{index_dir.name}/*.csv, recomputed from the stored"
                    " per-clip rows")],
        ["", ""],
        ["open-loop protocol", cfg.get("protocol", "")],
        ["minADE@k", (f"k = {cfg.get('k')}, reduced from ade_rollout_k[:k]; seeds are"
                      " base+k so a prefix is a real k-sample run")],
        ["OOD", ("every OOD run is cut to split=='val' (262 clips), so full-OOD and"
                 " ood_val-manifest runs share one row shape")],
        ["paired delta", (cfg.get("paired_baseline", "")
                          + "; d_minADE6_median is the median per-clip delta,"
                            " d_sig='*' when the bootstrap CI excludes zero")],
        ["older runs", ("runs predating per-sample storage leave the @6 columns blank"
                        " and keep minADE8_*")],
        ["", ""],
        ["rows: openloop", counts["openloop"]],
        ["rows: closedloop", counts["closedloop"]],
        ["rows: lingoqa", counts["lingoqa"]],
        ["", ""],
        ["note", ("arm names follow paper_numbers.ARMS where it names the run"
                  " directory; unregistered runs keep their directory name")],
        ["note", ("Ada and Blackwell are not bitwise comparable -- compare within one"
                  " arch, see the arch and baseline_ref columns")],
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results_index", help="dir written by collect_results")
    ap.add_argument("--name", default="results.xlsx")
    args = ap.parse_args()

    index_dir = REPO / "outputs" / args.index
    cfg = json.loads((index_dir / "config.json").read_text())
    tables, counts = {}, {}
    for name in SHEETS[1:]:
        with (index_dir / f"{name}.csv").open() as fh:
            rows = list(csv.reader(fh))
        tables[name] = rows
        counts[name] = len(rows) - 1

    tabs = [("README", readme_rows(index_dir, cfg, counts))]
    tabs += [(n, tables[n]) for n in SHEETS[1:]]
    out = index_dir / args.name
    write_xlsx(out, tabs)
    print(f"{out}  {out.stat().st_size / 1024:.1f} KB  "
          + " ".join(f"{n}={counts[n]}" for n in SHEETS[1:]), flush=True)


if __name__ == "__main__":
    main()
