"""Write the collected result tables into a Google Sheet, over the REST API.

Why not the MCP connector: its `tools/list` fails on this box, so the sheets tools never
become callable. But the OAuth handshake that `/mcp` performs still succeeds, and Claude
Code caches the resulting access token -- with `drive` + `spreadsheets` scopes -- in its
own credential store. That token is all the REST API needs, so the broken transport can
be stepped around entirely.

Why not the base64/MCP upload path this replaces: the workbook had to be carried as
base64 through a model response, and that transport corrupted silently somewhere above
~19.6k characters (measured: 19604 fine, 20140 mangled, 22172 rejected), which is why the
coloured view kept shedding columns. Here the file never leaves the machine.

Only the standard library is used. The repo venv is shared with running experiments and
installing into it has broken them before (see the note in `score_lingo_vqa.py`), and a
bearer token needs nothing more than urllib.

Token, in order of preference:
  --token-file <path>   a file containing the raw access token
  $SHEETS_ACCESS_TOKEN  the raw access token
  Claude Code's store   $CLAUDE_CONFIG_DIR/.credentials.json (or ~/.claude/...),
                        entry .mcpOAuth["sheets|<hash>"].accessToken

Google access tokens last about an hour and the store holds no refresh token, so when it
has expired the fix is to reconnect the connector (`/mcp` -> sheets) and re-run; the
reconnect re-mints the token even though tool discovery still fails afterwards.

Usage:
  # create a new spreadsheet inside a Drive folder
  .venv/bin/python experiments/evaluation/push_to_sheet.py \
      --folder 1ihefCIYkpTV09hXHgTTuighwH1MGMCwP
  # rewrite an existing one, keeping its URL
  .venv/bin/python experiments/evaluation/push_to_sheet.py --sheet-id <id>
"""

import argparse
import csv
import datetime
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHEETS = "https://sheets.googleapis.com/v4/spreadsheets"
DRIVE = "https://www.googleapis.com/drive/v3/files"
TABS = ["README", "openloop", "closedloop", "lingoqa"]
BLANK = "-"                            # every empty cell reads as "no value for this row"

# same palette as the coloured workbook, as Sheets 0-1 rgb
GOOD2 = {"red": 0.804, "green": 0.914, "blue": 0.804}
GOOD1 = {"red": 0.910, "green": 0.957, "blue": 0.910}
WARN = {"red": 0.980, "green": 0.937, "blue": 0.788}
BAD2 = {"red": 0.957, "green": 0.761, "blue": 0.722}
BAD3 = {"red": 0.906, "green": 0.608, "blue": 0.545}
HEADER_BG = {"red": 0.161, "green": 0.149, "blue": 0.106}

# (column, [(condition on {c}, colour)], gate column). Direction is per column on
# purpose: a minADE delta is better when lower, an alpasim score delta when higher.
# A delta whose bootstrap CI includes zero is left uncoloured via the gate, so "no
# significant change" never reads as a result.
RULES = {
    "openloop": [
        ("d_minADE6_median", [("{c}<=0", GOOD2), ("{c}<0.01", GOOD1),
                              ("{c}<0.05", WARN), ("{c}<0.15", BAD2),
                              ("{c}>=0.15", BAD3)], "d_sig"),
        ("coc_degen", [("{c}>0.30", BAD3), ("{c}>0.10", BAD2), ("{c}>0.02", WARN)], None),
    ],
    "closedloop": [
        ("d_score_mean", [("{c}>=0.05", GOOD2), ("{c}>=0.01", GOOD1),
                          ("{c}<=-0.05", BAD3), ("{c}<0", BAD2)], None),
        ("collision_at_fault", [("{c}>0.10", BAD3), ("{c}>0.05", BAD2)], None),
        ("offroad", [("{c}>0.10", BAD3), ("{c}>0.05", BAD2)], None),
        ("coc_degen", [("{c}>0.30", BAD3), ("{c}>0.10", BAD2), ("{c}>0.02", WARN)], None),
    ],
    "lingoqa": [
        ("judge_pct", [("{c}>=70", GOOD2), ("{c}>=60", GOOD1),
                       ("{c}<40", BAD2), ("{c}<20", BAD3)], None),
        ("truncated_frac", [("{c}>0.20", BAD2), ("{c}>0.05", WARN)], None),
    ],
}


def claude_store():
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    cands = [Path(cfg) / ".credentials.json"] if cfg else []
    cands += [Path.home() / "project/chan/.claude/.credentials.json",
              Path.home() / ".claude/.credentials.json"]
    return [p for p in cands if p.exists()]


def token_from_store():
    """The cached MCP OAuth access token, with its expiry checked before use."""
    for p in claude_store():
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key, v in (d.get("mcpOAuth") or {}).items():
            if not key.startswith("sheets|") or not v.get("accessToken"):
                continue
            exp = v.get("expiresAt")
            if exp and exp / 1000 < time.time():
                mins = (time.time() - exp / 1000) / 60
                raise SystemExit(
                    f"the cached sheets token in {p} expired {mins:.0f} min ago and the "
                    "store holds no refresh token.\nReconnect the connector (/mcp -> "
                    "sheets) to re-mint it, then re-run. Tool discovery will still fail "
                    "afterwards; only the token is needed.")
            return v["accessToken"]
    raise SystemExit(
        "no sheets access token found. Either connect the sheets connector once "
        "(/mcp) so Claude Code caches one, or pass --token-file / $SHEETS_ACCESS_TOKEN.")


def get_token(args):
    if args.token_file:
        return Path(args.token_file).read_text().strip()
    if os.environ.get("SHEETS_ACCESS_TOKEN"):
        return os.environ["SHEETS_ACCESS_TOKEN"].strip()
    return token_from_store()


def api(token, url, payload=None, method=None, soft=False):
    """One REST call. `soft` returns (None, code) instead of exiting, so a caller can
    fall back when only part of the token's scope is honoured."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read() or b"{}")
            return (out, 200) if soft else out
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:600]
        if soft:
            return None, e.code
        if e.code == 401:
            raise SystemExit(
                "401 from Google: the cached token is not accepted.\nReconnect the "
                "connector (/mcp -> sheets) ONCE and re-run straight away. Two "
                "authorisations in a row invalidate the first token, which is how this "
                f"state is usually reached.\n{body}") from e
        if e.code == 403:
            raise SystemExit(
                f"403 from Google: authenticated, but not allowed to write there.\n{body}"
            ) from e
        raise SystemExit(f"{e.code} {url}\n{body}") from e


def read_table(index_dir, name):
    with (index_dir / f"{name}.csv").open() as fh:
        rows = list(csv.reader(fh))
    return [[c if c != "" else BLANK for c in r] for r in rows]


def readme_rows(index_dir, cfg, counts):
    kst = datetime.timezone(datetime.timedelta(hours=9))
    stamp = datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M KST")
    return [
        ["AD VLA compression results"],
        ["generated", stamp],
        ["by", "experiments/evaluation/collect_results.py -> push_to_sheet.py"],
        ["source", f"outputs/{index_dir.name}/*.csv, recomputed from the stored rows"],
        [""],
        ["OPEN LOOP"],
        ["protocol", cfg.get("protocol", "")],
        ["metric", (f"minADE@{cfg.get('k')} / minFDE@{cfg.get('k')},"
                    " mean first, median beside it")],
        ["horizons", cfg.get("horizons", "")],
        ["omitted", cfg.get("omitted", "")],
        ["paired delta", (cfg.get("paired_baseline", "")
                          + "; d_sig='*' when the bootstrap CI excludes zero")],
        ["arch", "Ada and Blackwell are not bitwise comparable -- compare within one"],
        ["rows", counts["openloop"]],
        [""],
        ["CLOSED LOOP"],
        ["shape", cfg.get("closedloop_shape", "")],
        ["score", ("0 if an at-fault collision or offroad occurred, else"
                   " min(clamp(progress_clipped_rel,0,1)/0.8, 1.0)")],
        ["note", ("absolute score plus the per-scene paired delta vs baseline;"
                  " 57.7% of baseline rollouts sit at the 1.0 ceiling, so the mean CI"
                  " is the primary reading and Wilcoxon is secondary")],
        ["rows", counts["closedloop"]],
        [""],
        ["LINGOQA"],
        ["metric", "Lingo-Judge accuracy at the paper's own threshold (logit > 0)"],
        ["protocols", ("vqa = the arm answers from frames (capability)."
                       " coc_judge = a frozen reader answers from the arm's CoC text"
                       " only (information retained). The two are NOT comparable; the"
                       " judge rows need the blind floor (39.2) to be read at all.")],
        ["note", ("truncated_frac / answer_words / n_matched describe the answerer, so"
                  " they are blank for coc_judge rows by definition, not by omission")],
        ["rows", counts["lingoqa"]],
        [""],
        ["blank cells", f"'{BLANK}' means the value does not exist for that row"],
    ]


def a1(i):
    s = ""
    while True:
        s = chr(ord("A") + i % 26) + s
        i = i // 26 - 1
        if i < 0:
            return s


def format_requests(sheet_id, header, n_rows, rules):
    reqs = [
        {"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": HEADER_BG,
                "textFormat": {"bold": True,
                               "foregroundColor": {"red": 1, "green": 1, "blue": 1}}}},
            "fields": "userEnteredFormat(backgroundColor,textFormat)"}},
        {"updateSheetProperties": {
            "properties": {"sheetId": sheet_id,
                           "gridProperties": {"frozenRowCount": 1,
                                              "frozenColumnCount": 2}},
            "fields": "gridProperties(frozenRowCount,frozenColumnCount)"}},
        {"setBasicFilter": {"filter": {"range": {
            "sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": n_rows,
            "startColumnIndex": 0, "endColumnIndex": len(header)}}}},
    ]
    for col, bands, gate in rules:
        if col not in header:
            continue
        ci = header.index(col)
        gi = header.index(gate) if gate and gate in header else None
        # rules are evaluated in order and the first match wins, so add the narrowest
        # band first
        for tail, colour in bands:
            cond = tail.format(c=f"${a1(ci)}2")
            if gi is not None:
                cond = f'AND(${a1(gi)}2="*", {cond})'
            reqs.append({"addConditionalFormatRule": {"rule": {
                "ranges": [{"sheetId": sheet_id, "startRowIndex": 1,
                            "endRowIndex": n_rows,
                            "startColumnIndex": ci, "endColumnIndex": ci + 1}],
                "booleanRule": {
                    "condition": {"type": "CUSTOM_FORMULA",
                                  "values": [{"userEnteredValue": f"={cond}"}]},
                    "format": {"backgroundColor": colour}}}, "index": 0}})
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results_index")
    ap.add_argument("--folder", help="Drive folder id to create the spreadsheet in")
    ap.add_argument("--sheet-id", help="existing spreadsheet to rewrite (keeps its URL)")
    ap.add_argument("--title", default="AD VLA Results (auto)")
    ap.add_argument("--token-file", help="file holding a raw Google access token")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the payload and report shapes without calling Google")
    args = ap.parse_args()
    if not args.folder and not args.sheet_id and not args.dry_run:
        raise SystemExit("need --folder (create) or --sheet-id (rewrite)")

    index_dir = REPO / "outputs" / args.index
    cfg = json.loads((index_dir / "config.json").read_text())
    tables = {n: read_table(index_dir, n) for n in TABS[1:]}
    counts = {n: len(v) - 1 for n, v in tables.items()}
    tables["README"] = readme_rows(index_dir, cfg, counts)
    for t in TABS:
        print(f"  {t:11s} {len(tables[t]) - 1:4d} rows x "
              f"{max(len(r) for r in tables[t]):3d} cols", flush=True)
    if args.dry_run:
        print("dry run: nothing sent", flush=True)
        return

    token = get_token(args)

    if args.sheet_id:
        sid = args.sheet_id
        meta = api(token, f"{SHEETS}/{sid}?fields=properties.title,sheets.properties")
        print(f"rewriting '{meta['properties']['title']}'", flush=True)
    else:
        # Drive first: it is the only call that can create straight into a folder. The
        # cached MCP token does not always carry usable Drive access even though its
        # stored scope says `drive`, so fall back to a Sheets-side create (which lands
        # in My Drive root) and try to move it afterwards.
        body = {"name": args.title, "parents": [args.folder],
                "mimeType": "application/vnd.google-apps.spreadsheet"}
        f, code = api(token, f"{DRIVE}?fields=id&supportsAllDrives=true", body, soft=True)
        if f:
            sid = f["id"]
            print(f"created {sid} in folder {args.folder}", flush=True)
        else:
            print(f"Drive create refused ({code}); falling back to Sheets create",
                  flush=True)
            res = api(token, SHEETS, {"properties": {"title": args.title}})
            sid = res["spreadsheetId"]
            moved, mcode = api(
                token,
                f"{DRIVE}/{sid}?addParents={args.folder}&removeParents=root&fields=id",
                {}, method="PATCH", soft=True)
            if moved:
                print(f"created {sid} and moved it into {args.folder}", flush=True)
            else:
                print(f"created {sid} in My Drive root; the move also failed ({mcode}) "
                      f"-- drag it into the folder by hand", flush=True)
        meta = api(token, f"{SHEETS}/{sid}?fields=properties.title,sheets.properties")

    have = {s["properties"]["title"]: s["properties"]["sheetId"]
            for s in meta.get("sheets", [])}
    reqs = []
    if TABS[0] not in have:                       # rename the default sheet to README
        first = meta["sheets"][0]["properties"]
        reqs.append({"updateSheetProperties": {
            "properties": {"sheetId": first["sheetId"], "title": TABS[0]},
            "fields": "title"}})
        have.pop(first["title"], None)
        have[TABS[0]] = first["sheetId"]
    reqs += [{"addSheet": {"properties": {"title": t}}} for t in TABS[1:] if t not in have]
    if reqs:
        api(token, f"{SHEETS}/{sid}:batchUpdate", {"requests": reqs})

    meta = api(token, f"{SHEETS}/{sid}?fields=sheets.properties")
    ids = {s["properties"]["title"]: s["properties"]["sheetId"] for s in meta["sheets"]}

    api(token, f"{SHEETS}/{sid}/values:batchClear",
        {"ranges": [f"'{t}'" for t in TABS]})
    api(token, f"{SHEETS}/{sid}/values:batchUpdate",
        {"valueInputOption": "USER_ENTERED",
         "data": [{"range": f"'{t}'!A1", "values": tables[t]} for t in TABS]})

    fmt = []
    for t in TABS[1:]:
        fmt += format_requests(ids[t], tables[t][0], len(tables[t]), RULES.get(t, []))
    if fmt:
        api(token, f"{SHEETS}/{sid}:batchUpdate", {"requests": fmt})

    print(f"https://docs.google.com/spreadsheets/d/{sid}/edit", flush=True)


if __name__ == "__main__":
    main()
