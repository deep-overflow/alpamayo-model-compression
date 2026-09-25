"""CoC token-level importance by word category, from the per-position anatomy (run A of
plans/2026-09-25_coc-position-content.md; outputs/coc_posanat_v1_s*/).

Per clip c and generated-CoC position p, T^L_{c,p} = sum over the band's Q heads of
|o_{u,p} . dL/do_{u,p}| for L = CE (CoC NLL) and FM (the shipped 10-step expert loss), normalised
to sum to one over the clip's CoC positions (words + the two special tokens; a words-only
normalisation is reported alongside). Subword tokens are merged into words (a token starting with
a space or newline starts a word); the two special tokens <|cot_end|> and <|traj_future_start|>
are units of their own.

Word categories (rule-based; the lists are printed so the mapping can be audited or re-mapped from
the per-word table in metrics.json):
  decision  the ego action clause: from the first word up to the first connective
            (since / because / due / for / as / at / through / with / after / but / then / and / or /
            while / when / until) or up to a "to" that is followed by a determiner (unless the head
            verb is nudge / turn / change), minus function words
  scene     every remaining word after the clause that is neither spatial nor a function word
            (scene entities AND their states: vehicle, light, red, merges, bends, clear, ...)
  spatial   ahead / directly / in / into / from / behind / front / side / left / right / rightward /
            leftward / across / along / under / over / near / next / beside / onto / toward(s) /
            forward / at / on / through (after the clause)
  other     function words and punctuation (the / a / an / our / to / of / it / is / has / ...,
            the connectives themselves)
  cot_end, traj_start   the two special tokens

Reports, per band (0-21 / 22-34):
  1. category importance share, token share, enrichment = importance share / token share, CE and
     FM; main test: decision enrichment FM - CE, clip-paired bootstrap 95% CI (pre-specified)
  2. the same within four relative-position quarters of the words (position control)
  3. per layer the 6 heads with the largest rank(I_traj@CoC) - rank(I_CoC@CoC) and the 6 smallest
     (the census groups): share of each group's I_FM (and I_CE) on decision words and on the two
     special tokens

Usage:
  python experiments/head_analysis/analyze_coc_token_importance.py --shards coc_posanat_v1_s0 ... --out coc_tokimp_v1
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
LAST = 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}
CATS = ("decision", "scene", "spatial", "other", "cot_end", "traj_start")
WORD_CATS = CATS[:4]
CONNECTIVES = {"since", "because", "due", "for", "as", "at", "through", "with", "after", "but", "then", "and", "or", "while", "when", "until"}
DETERMINERS = {"the", "a", "an", "our", "its", "their", "this", "that", "any"}
FUNCTION = DETERMINERS | CONNECTIVES | {"to", "of", "it", "is", "are", "has", "have", "be", "been", "was", "were", "by", "in", "on", "no", "not", "so", "if", "which", "who", "we", "us", "them", "they", "there", "here", "such", "these", "those", "than", "also"}
SPATIAL = {"ahead", "directly", "in", "into", "from", "behind", "front", "side", "left", "right", "rightward", "leftward", "across", "along", "under", "over", "near", "next", "beside", "adjacent", "onto", "toward", "towards", "forward", "at", "on", "through", "alongside", "nearby", "opposite", "downstream", "upstream", "lateral", "laterally"}
DIRECTION_VERBS = {"nudge", "turn", "change", "steer", "move", "pull"}
PREPOSITIONS = {"to", "from", "behind", "toward", "towards", "in", "into", "on", "following", "past", "around"}
DECISION_VERBS = {"keep", "adapt", "adjust", "stop", "slow", "accelerate", "decelerate", "yield", "wait", "nudge", "proceed", "maintain",
                  "remain", "stay", "resume", "change", "match", "follow", "pass", "increase", "allow", "progress", "turn", "merge", "cross",
                  "continue", "brake", "hold", "prepare", "reduce", "creep", "coast", "overtake", "go", "drive", "move", "pull", "steer", "back", "let"}
SPECIAL = {"<|cot_end|>": "cot_end", "<|traj_future_start|>": "traj_start"}
COC = 3
K_GROUP = 6
rng = np.random.default_rng(0)


def merge_words(tokens):
    """(units, spans): unit strings and, per unit, its token indices. Special tokens stand alone."""
    units, spans, cur, cur_idx = [], [], "", []
    for i, t in enumerate(tokens):
        if t in SPECIAL:
            if cur_idx:
                units.append(cur)
                spans.append(cur_idx)
                cur, cur_idx = "", []
            units.append(t)
            spans.append([i])
            continue
        if t.startswith((" ", "\n")) and cur_idx:
            units.append(cur)
            spans.append(cur_idx)
            cur, cur_idx = "", []
        cur += t.strip()
        cur_idx.append(i)
    if cur_idx:
        units.append(cur)
        spans.append(cur_idx)
    return units, spans


def classify(units, strict=False):
    """One category per unit, by the rules in the module docstring. strict=True: the action clause is
    the first two words only (the head-clause convention of coc_action_consistency.py)."""
    norm = [u.lower().strip(".,;:!?") for u in units]
    words = [(i, w) for i, w in enumerate(norm) if units[i] not in SPECIAL]
    cats = [None] * len(units)
    for i, u in enumerate(units):
        if u in SPECIAL:
            cats[i] = SPECIAL[u]
    head = words[0][1] if words else ""
    end = len(words) if not strict else min(2, len(words))
    for j, (i, w) in enumerate(words):
        if j == 0 or strict:
            continue
        if w in CONNECTIVES and not (w == "for" and j == 1 and head == "wait"):
            end = j
            break
        nxt = words[j + 1][1] if j + 1 < len(words) else ""
        if w in PREPOSITIONS and nxt not in DECISION_VERBS and not (head in DIRECTION_VERBS and (nxt in DETERMINERS or nxt in ("left", "right"))):
            end = j
            break
    for j, (i, w) in enumerate(words):
        if w == "" or all(ch in ".,;:!?-/" for ch in w):
            cats[i] = "other"
        elif j < end:
            cats[i] = "other" if w in FUNCTION else "decision"
        elif w in FUNCTION:
            cats[i] = "other"
        elif w in SPATIAL:
            cats[i] = "spatial"
        else:
            cats[i] = "scene"
    return cats, norm


def ci_of(vals, n=3000):
    v = np.asarray(vals, float)
    boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rows, per = [], {}
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = np.load(REPO / "outputs" / s / "posanat_perclip.npz")
        rows += m["per_clip"]
        for k in ("q_ce_pos", "q_fm_pos"):
            per.setdefault(k, []).append(z[k])
    q = {L: np.abs(np.nan_to_num(np.concatenate(per[k]).astype(np.float64))) for L, k in (("CE", "q_ce_pos"), ("FM", "q_fm_pos"))}  # (N, PAD, L, H)
    N = len(rows)
    clips = [r["clip_id"] for r in rows]

    # units, categories, per-unit importance per band (sum of |contribution| over the unit's tokens and the band's heads)
    unit_cat, unit_word, unit_qpos, imp = [], [], [], {L: {b: [] for b in BANDS} for L in q}
    for i, r in enumerate(rows):
        units, spans = merge_words(r["tokens"])
        cats, norm = classify(units)
        unit_cat.append(cats)
        unit_word.append(norm)
        n_words = sum(u not in SPECIAL for u in units)
        wi, qpos = 0, []
        for u in units:
            if u in SPECIAL:
                qpos.append(-1)
            else:
                qpos.append(min(3, int(4 * wi / n_words)))
                wi += 1
        unit_qpos.append(qpos)
        for L in q:
            for b, sl in BANDS.items():
                tok = q[L][i, :, sl, :].sum((1, 2))  # (PAD,)
                imp[L][b].append(np.array([tok[sp].sum() for sp in spans]))
    lines = [f"CoC token-level importance by word category -- {N} clips, shards {', '.join(args.shards)}", ""]
    # audit: words per category
    wc = {c: Counter() for c in WORD_CATS}
    for cats, norm in zip(unit_cat, unit_word):
        for c, w in zip(cats, norm):
            if c in wc:
                wc[c][w] += 1
    lines.append("word lists (count over the clips):")
    for c in WORD_CATS:
        lines.append(f"  {c:9s} n={sum(wc[c].values()):4d}: " + ", ".join(f"{w} {n}" for w, n in wc[c].most_common(60)))
    lines.append("")

    metrics = {"n_clips": N, "clips": clips, "bands": {}, "word_table": {}}
    for b in BANDS:
        mb = {"norm_all": {}, "norm_words": {}, "quarters": {}, "groups": {}}
        lines.append(f"===== band {b} =====")
        for normname, keep_special in (("all CoC positions (words + special tokens)", True), ("words only", False)):
            share = {L: {c: np.zeros(N) for c in CATS} for L in q}
            tshare = {c: np.zeros(N) for c in CATS}
            for i in range(N):
                cats = np.array(unit_cat[i])
                sel = np.ones(len(cats), bool) if keep_special else ~np.isin(cats, ("cot_end", "traj_start"))
                for L in q:
                    v = imp[L][b][i] * sel
                    v = v / max(v.sum(), 1e-30)
                    for c in CATS:
                        share[L][c][i] = v[cats == c].sum()
                for c in CATS:
                    tshare[c][i] = (cats[sel] == c).mean() if sel.any() else 0.0
            lines.append(f"[{b}] normalisation: {normname}")
            lines.append(f"  {'category':10s} {'tokens':>7s} | {'CE share':>9s} {'enrich':>7s} | {'FM share':>9s} {'enrich':>7s} | FM-CE enrichment [paired bootstrap CI]")
            key = "norm_all" if keep_special else "norm_words"
            for c in CATS:
                if not keep_special and c in ("cot_end", "traj_start"):
                    continue
                ts = tshare[c].mean()
                e = {L: share[L][c].mean() / max(ts, 1e-30) for L in q}
                boot = []
                for _ in range(3000):
                    idx = rng.integers(0, N, N)
                    t_ = tshare[c][idx].mean()
                    boot.append((share["FM"][c][idx].mean() - share["CE"][c][idx].mean()) / max(t_, 1e-30))
                lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
                d = share["FM"][c] - share["CE"][c]
                p = float(wilcoxon(d[d != 0])[1]) if (d != 0).sum() > 5 else 1.0
                mb[key][c] = {"token_share": float(ts), "CE_share": float(share["CE"][c].mean()), "FM_share": float(share["FM"][c].mean()),
                              "CE_enrich": float(e["CE"]), "FM_enrich": float(e["FM"]), "d_enrich": float(e["FM"] - e["CE"]),
                              "d_enrich_ci": [float(lo), float(hi)], "wilcoxon_share_p": p}
                lines.append(f"  {c:10s} {ts:7.3f} | {share['CE'][c].mean():9.3f} {e['CE']:7.2f} | {share['FM'][c].mean():9.3f} {e['FM']:7.2f} | {e['FM'] - e['CE']:+.2f} [{lo:+.2f}, {hi:+.2f}]  (share diff Wilcoxon p={p:.2g})")
            mb[key]["main_test_decision"] = mb[key]["decision"]
        # position control: quarters of the words, importance normalised within the quarter
        lines.append(f"[{b}] position control (words only, four relative-position quarters; importance normalised within the quarter)")
        lines.append(f"  {'quarter':8s} {'n words':>8s} {'dec tok':>8s} | {'CE dec share':>12s} {'enrich':>7s} | {'FM dec share':>12s} {'enrich':>7s} | FM-CE [CI]  | categories present")
        for qq in range(4):
            sh = {L: [] for L in q}
            ts, nw, present = [], 0, Counter()
            for i in range(N):
                cats = np.array(unit_cat[i]); qp = np.array(unit_qpos[i])
                sel = qp == qq
                if not sel.any():
                    continue
                nw += int(sel.sum())
                present.update(cats[sel].tolist())
                ts.append((cats[sel] == "decision").mean())
                for L in q:
                    v = imp[L][b][i][sel]
                    v = v / max(v.sum(), 1e-30)
                    sh[L].append(v[cats[sel] == "decision"].sum())
            ts = np.array(ts); shc, shf = np.array(sh["CE"]), np.array(sh["FM"])
            n_dec = present["decision"]
            if len(ts) == 0 or ts.mean() == 0:
                lines.append(f"  Q{qq + 1:<7d} {nw:8d} {0:8.3f} | no decision words")
                continue
            if n_dec < 20:
                lines.append(f"  Q{qq + 1:<7d} {nw:8d} {ts.mean():8.3f} | only {n_dec} decision words in this quarter -- not testable; " + ", ".join(f"{c} {n}" for c, n in present.most_common()))
                mb["quarters"][f"Q{qq + 1}"] = {"n_words": nw, "n_decision": int(n_dec), "testable": False, "categories": dict(present)}
                continue
            boot = []
            for _ in range(3000):
                idx = rng.integers(0, len(ts), len(ts))
                boot.append((shf[idx].mean() - shc[idx].mean()) / max(ts[idx].mean(), 1e-30))
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            ec, ef = shc.mean() / ts.mean(), shf.mean() / ts.mean()
            mb["quarters"][f"Q{qq + 1}"] = {"n_words": nw, "decision_token_share": float(ts.mean()), "CE_share": float(shc.mean()), "FM_share": float(shf.mean()),
                                            "CE_enrich": float(ec), "FM_enrich": float(ef), "d_enrich": float(ef - ec), "d_enrich_ci": [float(lo), float(hi)],
                                            "categories": dict(present)}
            lines.append(f"  Q{qq + 1:<7d} {nw:8d} {ts.mean():8.3f} | {shc.mean():12.3f} {ec:7.2f} | {shf.mean():12.3f} {ef:7.2f} | {ef - ec:+.2f} [{lo:+.2f}, {hi:+.2f}] | " + ", ".join(f"{c} {n}" for c, n in present.most_common()))
        # special tokens and the last word, for reference (all-position normalisation)
        lines.append(f"[{b}] special tokens and the last word (share of the clip's CoC total, mean over clips): "
                     + " | ".join(f"{c}: CE {mb['norm_all'][c]['CE_share']:.3f} FM {mb['norm_all'][c]['FM_share']:.3f}" for c in ("cot_end", "traj_start")))
        lw = {L: [] for L in q}
        for i in range(N):
            cats = np.array(unit_cat[i])
            widx = np.where(~np.isin(cats, ("cot_end", "traj_start")))[0]
            for L in q:
                v = imp[L][b][i]
                lw[L].append(v[widx[-1]] / max(v.sum(), 1e-30))
        lines.append(f"  last word: CE {np.mean(lw['CE']):.3f} FM {np.mean(lw['FM']):.3f}")
        # sensitivity: strict decision definition (first two words), words-only normalisation
        share = {L: np.zeros(N) for L in q}
        ts = np.zeros(N)
        for i in range(N):
            cats = np.array(classify(merge_words(rows[i]["tokens"])[0], strict=True)[0])
            sel = ~np.isin(cats, ("cot_end", "traj_start"))
            ts[i] = (cats[sel] == "decision").mean()
            for L in q:
                v = imp[L][b][i] * sel
                v = v / max(v.sum(), 1e-30)
                share[L][i] = v[cats == "decision"].sum()
        boot = []
        for _ in range(3000):
            idx = rng.integers(0, N, N)
            boot.append((share["FM"][idx].mean() - share["CE"][idx].mean()) / max(ts[idx].mean(), 1e-30))
        ec, ef = share["CE"].mean() / ts.mean(), share["FM"].mean() / ts.mean()
        mb["strict_decision_words_only"] = {"token_share": float(ts.mean()), "CE_share": float(share["CE"].mean()), "FM_share": float(share["FM"].mean()),
                                            "CE_enrich": float(ec), "FM_enrich": float(ef), "d_enrich": float(ef - ec), "d_enrich_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        lines.append(f"[{b}] sensitivity, strict decision = first two words (words-only normalisation): tokens {ts.mean():.3f} | CE share {share['CE'].mean():.3f} enrich {ec:.2f} | FM share {share['FM'].mean():.3f} enrich {ef:.2f} | FM-CE {ef - ec:+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")
        metrics["bands"][b] = mb

    # 3. head groups (census definition) -- share of each group's importance on decision words / special tokens
    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    Tc = np.abs(pq["fm_full"][:, COC].astype(np.float64)).mean(0)
    Cc = np.abs(pq["ce"][:, COC].astype(np.float64)).mean(0)
    H = Tc.shape[1]
    rk = lambda x: np.argsort(np.argsort(x, axis=1), axis=1) / (H - 1)
    diff = rk(Tc) - rk(Cc)
    lines.append("\n===== head groups (per layer: 6 largest and 6 smallest rank(I_traj@CoC) - rank(I_CoC@CoC)) =====")
    lines.append("  share of the group's own CoC-position mass: 'all' = over words + special tokens, 'words' = over words only")
    lines.append(f"  {'band':6s} {'loss':4s} {'group':12s} | {'decision (all) [CI]':>26s} {'special':>8s} {'scene':>6s} | {'decision (words) [CI]':>26s} {'scene':>6s} {'spatial':>8s} {'other':>6s}")
    for b, sl in BANDS.items():
        gm = {}
        groups = {}
        for name in ("FM-favoured", "CE-favoured"):
            mask = np.zeros((LAST, H), bool)
            for l in range(sl.start, sl.stop):
                order = np.argsort(diff[l])
                mask[l, order[-K_GROUP:] if name == "FM-favoured" else order[:K_GROUP]] = True
            groups[name] = mask
        keys = ("decision", "special", "scene", "decision_w", "scene_w", "spatial_w", "other_w")
        per_clip = {name: {L: {c: np.zeros(N) for c in keys} for L in q} for name in groups}
        for i in range(N):
            cats = np.array(unit_cat[i])
            units, spans = merge_words(rows[i]["tokens"])
            is_word = ~np.isin(cats, ("cot_end", "traj_start"))
            for name, mask in groups.items():
                for L in q:
                    tok = (q[L][i, :, :LAST, :] * mask[None]).sum((1, 2))  # (PAD,)
                    u = np.array([tok[sp].sum() for sp in spans])
                    tot, totw = max(u.sum(), 1e-30), max(u[is_word].sum(), 1e-30)
                    per_clip[name][L]["decision"][i] = u[cats == "decision"].sum() / tot
                    per_clip[name][L]["special"][i] = u[~is_word].sum() / tot
                    per_clip[name][L]["scene"][i] = u[cats == "scene"].sum() / tot
                    for c in ("decision", "scene", "spatial", "other"):
                        per_clip[name][L][f"{c}_w"][i] = u[cats == c].sum() / totw
        for L in ("FM", "CE"):
            for name in groups:
                m, lo, hi = ci_of(per_clip[name][L]["decision"])
                mw, low, hiw = ci_of(per_clip[name][L]["decision_w"])
                gm.setdefault(name, {})[L] = {c: float(per_clip[name][L][c].mean()) for c in keys} | {"decision_ci": [lo, hi], "decision_w_ci": [low, hiw]}
                lines.append(f"  {b:6s} {L:4s} {name:12s} | {m:.3f} [{lo:.3f}, {hi:.3f}]".ljust(56) + f" {per_clip[name][L]['special'].mean():8.3f} {per_clip[name][L]['scene'].mean():6.3f} | "
                             + f"{mw:.3f} [{low:.3f}, {hiw:.3f}]".rjust(26) + f" {per_clip[name][L]['scene_w'].mean():6.3f} {per_clip[name][L]['spatial_w'].mean():8.3f} {per_clip[name][L]['other_w'].mean():6.3f}")
            for c, lab in (("decision", "all"), ("decision_w", "words")):
                d = per_clip["FM-favoured"][L][c] - per_clip["CE-favoured"][L][c]
                m, lo, hi = ci_of(d)
                p = float(wilcoxon(d[d != 0])[1]) if (d != 0).sum() > 5 else 1.0
                gm[f"{L}_{c}_FMfav_minus_CEfav"] = {"mean": m, "ci": [lo, hi], "wilcoxon_p": p}
                lines.append(f"  {b:6s} {L:4s} decision share ({lab}), FM-favoured - CE-favoured: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (Wilcoxon p={p:.2g})")
        metrics["bands"][b]["groups"] = gm

    # per-word table (all-position normalisation, both bands): count, mean share per occurrence
    wt = defaultdict(lambda: {"count": 0, "cat": None, **{f"{L}_{b}": 0.0 for L in q for b in BANDS}})
    for i in range(N):
        units = merge_words(rows[i]["tokens"])[0]
        for L in q:
            for b in BANDS:
                v = imp[L][b][i] / max(imp[L][b][i].sum(), 1e-30)
                for u, w, c, val in zip(units, unit_word[i], unit_cat[i], v):
                    key = u if u in SPECIAL else w
                    wt[key][f"{L}_{b}"] += float(val)
                    wt[key]["cat"] = c
        for u, w in zip(units, unit_word[i]):
            wt[u if u in SPECIAL else w]["count"] += 1
    table = {}
    for w, d in wt.items():
        table[w] = {"count": d["count"], "cat": d["cat"]} | {k: d[k] / d["count"] for k in d if k not in ("count", "cat")}
    metrics["word_table"] = table
    lines.append("\nper-word mean share per occurrence (all-position normalisation), words with >= 5 occurrences, sorted by FM share in 0-21:")
    lines.append(f"  {'word':22s} {'cat':10s} {'n':>4s} | {'CE 0-21':>8s} {'FM 0-21':>8s} | {'CE 22-34':>8s} {'FM 22-34':>8s}")
    for w, d in sorted(table.items(), key=lambda kv: -kv[1]["FM_0-21"]):
        if d["count"] >= 5:
            lines.append(f"  {w:22s} {d['cat']:10s} {d['count']:4d} | {d['CE_0-21']:8.4f} {d['FM_0-21']:8.4f} | {d['CE_22-34']:8.4f} {d['FM_22-34']:8.4f}")

    # plots: category shares and enrichment per band
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, b in zip(axes, BANDS):
        m = metrics["bands"][b]["norm_all"]
        x = np.arange(len(CATS))
        ax.bar(x - 0.2, [m[c]["token_share"] for c in CATS], 0.2, color="#bbbbbb", label="token share")
        ax.bar(x, [m[c]["CE_share"] for c in CATS], 0.2, color="#ECAE3C", label="I_CoC share")
        ax.bar(x + 0.2, [m[c]["FM_share"] for c in CATS], 0.2, color="#618BC8", label="I_traj share")
        ax.set_xticks(x); ax.set_xticklabels(CATS, rotation=30, ha="right", fontsize=8); ax.set_title(f"layers {b}", fontsize=9)
    axes[0].set_ylabel("share of the clip's CoC total"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "category_shares.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, b in zip(axes, BANDS):
        m = metrics["bands"][b]["norm_words"]
        x = np.arange(len(WORD_CATS))
        ax.bar(x - 0.15, [m[c]["CE_enrich"] for c in WORD_CATS], 0.3, color="#ECAE3C", label="I_CoC")
        ax.bar(x + 0.15, [m[c]["FM_enrich"] for c in WORD_CATS], 0.3, color="#618BC8", label="I_traj")
        ax.axhline(1, color="black", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(WORD_CATS, fontsize=8); ax.set_title(f"layers {b}, words only", fontsize=9)
    axes[0].set_ylabel("enrichment = importance share / token share"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "category_enrichment.png", dpi=150); plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "anatomy": args.anatomy, "k_group": K_GROUP,
                                                 "rules": {"connectives": sorted(CONNECTIVES), "spatial": sorted(SPATIAL), "function": sorted(FUNCTION), "direction_verbs": sorted(DIRECTION_VERBS)}}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
