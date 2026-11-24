"""Candidate gallery for picking closed-loop showcase scenes: MoOP vs FM-only vs CE-only, BEV only (no GPU).

Candidates: scenes where MoOP passes BOTH rollouts while FM-only and CE-only (each its worse rollout) both
fail the same way (at-fault collision, or off-road), over the u40 and u50 runs on Hard100 and Origin150.
Per candidate: cl_bev.py frames for the three arms (the map is loaded once per scene), a side-by-side BEV
video (cl_compose.py), and one HTML page with filters, CoC captions synced to each video, a pick checkbox
and a copyable list of picks. Data goes to gallery_data.js so the page also opens from file://.

  taskset -c 0-31 python closed-loop-viz/cl_gallery.py --out /mnt/.../closedloop_gallery --work /tmp/... -j 6
  python closed-loop-viz/serve_range.py 8780 /mnt/.../closedloop_gallery     # then open http://localhost:8780
"""
import argparse
import asyncio
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cl_bev import frame_cocs, load_map, read_rollout, render_arm, verdict_of
from cl_compose import compose, entry_data

RUNS = Path("/home/cvlab21/project/chan/advla/alpasim-runs")
SETS = {("u40", "Hard100"): "h100_merged_slim_{}_u40_v2", ("u40", "Origin150"): "m2601_merged_slim_{}_u40_v2",
        ("u50", "Hard100"): "u50h_merged_slim_{}_u50_v2", ("u50", "Origin150"): "u50o_merged_slim_{}_u50_v2"}
ARMS = [("moop", "MoOP", "dual"), ("fm", "FM-only", "traj"), ("ce", "CE-only", "coc")]
ON_PAGE = {("u40", s) for s in ("fb6d1a18", "166fe701", "93982b71", "065dcac9")}    # MoOP page, commit 7472b89


def candidates(budgets):
    out, skipped = [], []
    for (bud, sset), pat in SETS.items():
        if bud not in budgets:
            continue
        rolls = {}
        for key, _, k in ARMS:
            d = json.loads((RUNS / pat.format(k) / "aggregate/results-summary.json").read_text())
            for r in d["rollouts"]:
                rolls.setdefault(r["clipgt_id"], {}).setdefault(key, []).append(r)
        for scene, a in sorted(rolls.items()):
            if set(a) != {"moop", "fm", "ce"} or any(verdict_of(r) != "pass" for r in a["moop"]):
                continue
            worst = {key: min(a[key], key=lambda r: float(r["score"])) for key in a}
            v = {key: verdict_of(worst[key]) for key in ("fm", "ce")}
            if v["fm"] == v["ce"] != "pass":
                short = scene.split("-")[1]
                # some summaries list a rollout whose log was never written; nothing to draw then
                logs = [RUNS / pat.format(k) / "rollouts" / scene / worst[key]["rollout_id"] / f
                        for key, _, k in ARMS for f in ("rollout.asl", "metrics.parquet")]
                if not all(p.exists() for p in logs):
                    skipped.append(f"{bud} {sset} {short} ({v['fm']}): rollout log missing")
                    continue
                out.append({"id": f"{bud}_{sset.lower()}_{short}", "budget": bud, "set": sset, "scene": scene,
                            "short": short, "kind": v["fm"], "on_page": (bud, short) in ON_PAGE,
                            "cfgs": {key: str(RUNS / pat.format(k)) for key, _, k in ARMS}})
    out.sort(key=lambda c: (c["kind"] != "collision", c["budget"], c["set"], c["short"]))
    return out, skipped


def bev_scene(job):
    scene, cands, work = job
    vm = None
    for c in cands:
        for key, _, _ in ARMS:
            d = Path(work) / c["id"] / key
            if (d / "bev_meta.json").exists():
                continue
            if vm is None:
                vm = load_map(scene)
            render_arm(vm, scene, c["cfgs"][key], d)
    return scene


def gallery_entry(c, work):
    entry = {k: v for k, v in c.items() if k != "cfgs"}
    entry["arms"] = [{"label": label, "dir": str(Path(work) / c["id"] / key)} for key, label, _ in ARMS]
    return entry


def compose_one(job):
    c, work, out = job
    return compose(gallery_entry(c, work), Path(out) / "videos" / f"{c['id']}.mp4", tmp_root=work)


def refresh_coc(job):
    """Recompute the per-frame CoC of existing BEV dumps from the logs, without re-rendering frames."""
    c, work = job
    for key, _, _ in ARMS:
        p = Path(work) / c["id"] / key / "bev_meta.json"
        meta = json.loads(p.read_text())
        dat = asyncio.run(read_rollout(Path(c["cfgs"][key]) / "rollouts" / c["scene"] / meta["rollout_id"] / "rollout.asl"))
        meta["coc"] = frame_cocs(dat, sorted(dat["frames"]))
        p.write_text(json.dumps(meta))
    return c["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="gallery directory (index.html, videos/)")
    ap.add_argument("--work", type=Path, required=True, help="BEV frame dumps (~9 MB per arm)")
    ap.add_argument("--budgets", nargs="+", default=["u40", "u50"])
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--data-only", action="store_true",
                    help="keep frames and videos; recompute CoC from the logs and rewrite the page and data")
    args = ap.parse_args()

    cands, skipped = candidates(args.budgets)
    print(f"{len(cands)} candidates: " + ", ".join(f"{k} {sum(c['kind'] == k for c in cands)}"
                                                  for k in ("collision", "offroad")), flush=True)
    for s in skipped:
        print(f"  skipped {s}", flush=True)
    by_scene = {}
    for c in cands:
        by_scene.setdefault(c["scene"], []).append(c)
    with Pool(args.jobs) as pool:
        if args.data_only:
            for _ in pool.imap_unordered(refresh_coc, [(c, str(args.work)) for c in cands]):
                pass
            entries = [entry_data(gallery_entry(c, args.work), args.out / "videos" / f"{c['id']}.mp4") for c in cands]
        else:
            for i, s in enumerate(pool.imap_unordered(bev_scene, [(s, cs, str(args.work)) for s, cs in by_scene.items()])):
                print(f"  BEV {i + 1}/{len(by_scene)} scenes ({s})", flush=True)
            entries = []
            for e in pool.imap(compose_one, [(c, str(args.work), str(args.out)) for c in cands]):
                entries.append(e)
                print(f"  video {len(entries)}/{len(cands)} {e['id']}", flush=True)
    for e in entries:
        moop = e["arms"][0]
        e["route_m"] = round(json.loads((args.work / e["id"] / "moop" / "bev_meta.json").read_text())["gt_dist_m"])
        e["moop_score"] = round(moop["score"], 3)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gallery_data.js").write_text(
        "window.GALLERY = " + json.dumps({"fps": 10, "scenes": entries}, separators=(",", ":"), ensure_ascii=False) + ";\n")
    (args.out / "index.html").write_text(PAGE)
    (args.out / "candidates.json").write_text(json.dumps(cands, indent=1))     # run dirs, for camera renders of picks
    print(f"wrote {args.out / 'index.html'} ({len(entries)} scenes)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-loop candidates</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --ink:#18202b; --muted:#5b6573; --line:#dde2e8; --good:#0a8a5a; --bad:#c8402f; --soft:#e8f0fb; }
@media (prefers-color-scheme: dark) { :root { --bg:#11151c; --card:#1a2029; --ink:#e8ecf2; --muted:#a2acba; --line:#2a3240; --good:#4cc79a; --bad:#f08a72; --soft:#1d2a3d; } }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 24px; margin: 0 0 6px; } p { color: var(--muted); margin: 6px 0; }
.bar { position: sticky; top: 0; z-index: 5; background: var(--bg); padding: 10px 0; border-bottom: 1px solid var(--line); display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; }
.grp { display:inline-flex; gap:4px; flex-wrap:wrap; align-items:center; } .grp b { font-size:13px; color:var(--muted); margin-right:4px; }
button { border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:999px; padding:3px 11px; font:600 13px/1.4 inherit; cursor:pointer; }
button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
.picks { display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:13px; } .picks code { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:2px 6px; max-width: 640px; overflow-wrap:anywhere; }
.key { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12.5px; color:var(--muted); margin:10px 0 4px; }
.key span { display:inline-flex; align-items:center; gap:5px; } .key i { position:relative; width:24px; height:14px; border-radius:3px; background:#15181d; display:inline-block; }
.key i::after { content:""; position:absolute; left:4px; right:4px; top:6px; height:2px; }
.k-road::after{top:3px!important;height:8px!important;background:#373d48} .k-edge::after{background:#b4bcc7} .k-flag::after{height:4px!important;top:5px!important;background:#f0605a}
.k-plan::after{height:3px!important;background:#ff9f43} .k-gt::after{height:0!important;border-top:2px dashed #6fa8ff} .k-trail::after{height:3px!important;background:#e3e6ea}
.k-ego::after{left:8px!important;right:8px!important;top:2px!important;height:10px!important;background:#fff;box-shadow:inset 0 0 0 2px #3fbf8a} .k-traf::after{left:8px!important;right:8px!important;top:2px!important;height:10px!important;background:#8d939c}
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; margin-top:16px; }
.card.picked { outline: 2px solid var(--good); }
.head { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; align-items:baseline; margin-bottom:8px; }
.head .t { font-weight:700; } .head .m { color:var(--muted); font-size:13px; }
.tag { display:inline-block; font-size:12px; font-weight:700; border-radius:6px; padding:0 6px; margin-left:4px; }
.tag.pass { color:var(--good); background:color-mix(in srgb, var(--good) 14%, transparent); } .tag.fail { color:var(--bad); background:color-mix(in srgb, var(--bad) 14%, transparent); }
.tag.page { color:var(--ink); background:color-mix(in srgb, var(--ink) 10%, transparent); }
video { width:100%; display:block; border-radius:8px; background:#0c0c0e; }
.caps { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:6px; margin-top:6px; }
.cap { border:1px solid var(--line); border-radius:8px; padding:6px 8px; font-size:13px; min-height: 4.4em; } .cap.ours { background:var(--soft); } .cap.hit { border-color:var(--bad); }
.cap b { display:block; font-size:12.5px; } .flag { color:var(--bad); font-weight:700; font-size:11.5px; margin-left:4px; }
label.pick { font-size:13px; font-weight:600; cursor:pointer; user-select:none; }
@media (max-width: 700px) { .caps { grid-template-columns: 1fr; } }
</style></head><body><div class="wrap">
<h1>Closed-loop candidates: MoOP vs FM-only vs CE-only</h1>
<p>Scenes where MoOP passes <b>both</b> of its rollouts while FM-only and CE-only (each its worse rollout) <b>both</b>
fail the same way. Each video shows those rollouts side by side as a heading-up bird's-eye view on the scene's own map;
red marks only what the scorer flagged at that moment. Tick the scenes you want; the list at the top is what to send back.</p>
<div class="key"><span><i class="k-road"></i>lane surface</span><span><i class="k-edge"></i>road edge</span>
<span><i class="k-flag"></i>flagged: road edge touched (off-road) / vehicle hit (collision)</span><span><i class="k-ego"></i>the car (red outline = this drive fails)</span>
<span><i class="k-traf"></i>other traffic</span><span><i class="k-plan"></i>6.4 s plan held now</span><span><i class="k-gt"></i>human path and position</span><span><i class="k-trail"></i>driven path</span></div>
<div class="bar">
  <span class="grp" data-f="budget"><b>budget</b><button data-v="">all</button><button data-v="u40">u40 (24.0%)</button><button data-v="u50">u50 (30.0%)</button></span>
  <span class="grp" data-f="set"><b>set</b><button data-v="">all</button><button data-v="Hard100">Hard100</button><button data-v="Origin150">Origin150</button></span>
  <span class="grp" data-f="kind"><b>failure</b><button data-v="">all</button><button data-v="collision">collision</button><button data-v="offroad">off-road</button></span>
  <span class="grp" data-f="picked"><b>show</b><button data-v="">all</button><button data-v="1">picked</button></span>
  <span class="picks"><span id="count"></span><code id="pickcode">(none)</code><button id="copy">copy</button><button id="clear">clear</button></span>
</div>
<div id="list"></div>
</div>
<script src="gallery_data.js"></script>
<script>
(function () {
  const D = window.GALLERY, fps = D.fps, KEY = "clgallery-picks";
  const VTXT = { pass: "no incident", collision: "at-fault collision", offroad: "off-road" };
  const KTXT = { collision: "At-fault collision", offroad: "Off-road" };
  let picks = new Set(); try { picks = new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch (e) {}
  const filt = { budget: "", set: "", kind: "", picked: "" };
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify([...picks])); } catch (e) {} };
  const cocAt = (runs, f) => { let t = runs[0] ? runs[0][1] : ""; for (const [s, x] of runs) { if (s > f) break; t = x; } return t; };
  const setCoc = (el, t, wait) => { const k = t.indexOf("<|"); el.textContent = (k < 0 ? t : t.slice(0, k).trim()) || (wait || "(empty reasoning output)");
    if (k >= 0) { const s = document.createElement("span"); s.className = "flag"; s.textContent = "degenerate continuation"; el.appendChild(s); } };
  const cards = D.scenes.map((sc) => {
    const c = document.createElement("div"); c.className = "card"; c.dataset.id = sc.id;
    const fails = sc.arms.slice(1).map((a) => `<span class="tag fail">${a.name} ${VTXT[a.verdict]} ${a.event_t.toFixed(1)} s</span>`).join("");
    const m = sc.arms[0];
    c.innerHTML = `<div class="head"><div><span class="t">${KTXT[sc.kind]} · ${sc.budget} · ${sc.set} · ${sc.short}</span>
      ${sc.on_page ? '<span class="tag page">on the page now</span>' : ""}<span class="tag pass">MoOP ${VTXT.pass} · ${m.dist_m} m · score ${sc.moop_score}</span>${fails}</div>
      <div class="m"><span class="clock">t = ${sc.t_offset.toFixed(1)} s</span> · route ${sc.route_m} m · ${sc.scene}
      <label class="pick"><input type="checkbox"> pick</label></div></div>
      <video src="videos/${sc.video}" poster="videos/${sc.poster}" controls muted loop playsinline preload="none"></video>
      <div class="caps">${sc.arms.map((a) => `<div class="cap${a.name === "MoOP" ? " ours" : ""}"><b>${a.name}</b><span></span></div>`).join("")}</div>`;
    const v = c.querySelector("video"), caps = [...c.querySelectorAll(".cap")], clock = c.querySelector(".clock");
    let last = -1;
    const upd = () => { const f = Math.min(Math.floor(v.currentTime * fps + 1e-3), sc.n_frames - 1); if (f === last) return; last = f;
      clock.textContent = `t = ${(sc.t_offset + f / fps).toFixed(1)} s`;
      sc.arms.forEach((a, j) => { const g = Math.min(f, a.n_frames - 1); const first = a.coc.find((r) => r[1]);
        const wait = first && g < first[0] ? `(no reasoning yet: the first arrives at ${(sc.t_offset + first[0] / fps).toFixed(1)} s)` : "";
        setCoc(caps[j].querySelector("span"), cocAt(a.coc, g), wait);
        caps[j].classList.toggle("hit", a.event_frame !== null && g >= a.event_frame); }); };
    const loop = () => { upd(); if (!v.paused) requestAnimationFrame(loop); };
    v.addEventListener("play", () => { document.querySelectorAll("video").forEach((o) => { if (o !== v) o.pause(); }); loop(); });
    v.addEventListener("seeked", upd); v.addEventListener("timeupdate", upd); upd();
    const cb = c.querySelector("input"); cb.checked = picks.has(sc.id); c.classList.toggle("picked", cb.checked);
    cb.addEventListener("change", () => { cb.checked ? picks.add(sc.id) : picks.delete(sc.id); c.classList.toggle("picked", cb.checked); save(); render(); });
    document.getElementById("list").appendChild(c);
    return { c, sc };
  });
  function render() {
    let n = 0;
    cards.forEach(({ c, sc }) => { const ok = (!filt.budget || sc.budget === filt.budget) && (!filt.set || sc.set === filt.set)
      && (!filt.kind || sc.kind === filt.kind) && (!filt.picked || picks.has(sc.id)); c.style.display = ok ? "" : "none"; n += ok; });
    document.getElementById("count").textContent = `${n} shown · ${picks.size} picked:`;
    document.getElementById("pickcode").textContent = picks.size ? [...picks].sort().join(" ") : "(none)";
    document.querySelectorAll(".grp").forEach((g) => g.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(filt[g.dataset.f] === b.dataset.v))));
  }
  document.querySelectorAll(".grp").forEach((g) => g.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; filt[g.dataset.f] = b.dataset.v; render(); }));
  document.getElementById("copy").addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText([...picks].sort().join(" ")));
  document.getElementById("clear").addEventListener("click", () => { picks.clear(); save(); cards.forEach(({ c }) => { c.classList.remove("picked"); c.querySelector("input").checked = false; }); render(); });
  render();
})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
