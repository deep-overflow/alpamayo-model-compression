"""Curate stored Dual/expert evidence without rerunning a model or changing source figures.

Reuses the original analysis plotting functions and changes presentation only. The six
existing paper panels are preserved. Run from the repository root with .venv/bin/python.
"""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import fig_why_differs as source
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
FORMATS = ("png", "pdf", "svg")
EXISTING = {
    "fig1_depth_mlp": "VLM MLP: 두 objective의 레이어별 중요도",
    "fig1_depth_q_head": "VLM Q head: 두 objective의 레이어별 중요도",
    "fig1_rank_agreement": "같은 레이어 내부의 Traj–CoC 순위 상관",
    "fig2_mass_curve": "Expert: 낮은 중요도부터 제거한 raw importance 비율",
    "fig2_steps_mlp": "Expert MLP: step 사이의 순위 안정성",
    "fig2_steps_q_head": "Expert Q head: step 사이의 순위 안정성",
}
ADDITIONS = {
    "fig4_same_token_q_head": ("", "토큰 조건을 맞췄을 때의 Q-head 순위 상관"),
    "fig4_same_token_mlp": ("", "토큰 조건을 맞췄을 때의 VLM MLP 순위 상관"),
    "fig5_probe_ratio_mlp": ("", "무작위 readout에서도 재현되는 출력 경로의 영향"),
    "fig5_damage_by_band_fm": ("", "핵심 개입: 대역별 pruning에 따른 FM loss 변화"),
    "fig5_damage_by_band_nll": ("", "핵심 개입: 대역별 pruning에 따른 CoC NLL 변화"),
    "fig5_dual_saves": ("supplementary", "Dual이 남긴 unit을 실제로 제거한 결과"),
    "fig4_token_handoff_mlp": ("supplementary", "Vision 토큰의 중요도 기여가 깊이에 따라 감소"),
    "fig4_itraj_ports_mlp": ("supplementary", "VLM importance로 들어오는 cache별 gradient 기여"),
    "fig5_vv_window_profile": ("supplementary", "Vision interaction 차단과 중요도 프로파일 비교"),
}
TABLES = {
    "table_01_open_loop": "Open-loop: Test500 및 OOD-val262",
    "table_01_closed_loop": "Closed-loop: 150 scenes × 2 rollouts",
    "table_01_lingoqa": "LingoQA reasoning benchmark",
    "table_02_expert_axis": "Expert Q/MLP 실제 제거 비교",
    "table_03_expert_open_loop": "Dual + expert MLP: open-loop 압축 사다리",
    "table_03_expert_closed_loop": "Dual + expert MLP: closed-loop 압축 사다리",
}
INPUTS = [
    "outputs/importance_v2/importance.npz",
    "outputs/stepimp_fm_perstep_v2/step_importance.npz",
    "outputs/gradanat_v1/metrics_analysis.json",
    "outputs/gradanat_v1/anatomy.npz",
    "outputs/gradanat_probes_v1/anatomy.npz",
    "outputs/gradanat_probes_v1/metrics_analysis.json",
    "outputs/armheld_v1/metrics.json",
    "outputs/tokabl_v1/metrics.json",
    "outputs/tokabl_v1_s0/config.json",
    "outputs/portmap_v1/port_map.npz",
    "outputs/portmap_v1/metrics_analysis.json",
    "outputs/vvdepth_v1/metrics.json",
    "figures/fig3_why_differs_stats.json",
    "figures/fig5_causal_validation_stats.json",
    "ppt-fig/meeting_table_data.json",
    "experiments/paper/fig_why_differs.py",
    "experiments/paper/fig_criterion.py",
    "experiments/paper/make_claim_figure_pack.py",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legend(ax, labels, loc="lower left", ncol=1, above=False):
    handles, _ = ax.get_legend_handles_labels()
    options = {"bbox_to_anchor": (0.5, 1.01)} if above else {}
    ax.legend(
        handles, labels, loc=loc, ncol=ncol, fontsize=10,
        handlelength=1.15, handletextpad=0.4, columnspacing=0.8,
        borderaxespad=0.2, labelspacing=0.25, **options,
    )


def export_panel(fig, name, out, exported):
    if name not in ADDITIONS:
        plt.close(fig)
        return
    folder, _ = ADDITIONS[name]
    target = out / folder
    target.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(3.5, 2.8)
    ax = fig.axes[0]
    original_xlim = ax.get_xlim()
    ax.set_xlabel("VLM layer index")
    ax.set_xticks([0, 7, 14, 21, 28, 35])
    if name.startswith("fig4_same_token_"):
        ax.set_ylabel(r"Corrected Spearman’s $\rho$")
        legend(ax, ["All tokens", "Vision tokens", "Text tokens"])
        ax.get_legend().set_frame_on(True)
        ax.get_legend().get_frame().set_edgecolor("none")
        ax.get_legend().get_frame().set_alpha(0.95)
    elif name == "fig5_probe_ratio_mlp":
        ax.set_ylabel("Relative importance ratio")
        # Orange here denotes the cache readout, not the CoC objective alone.
        ax.lines[2].set_color("#ECAE3C")
        legend(ax, ["Actual objectives", "Expert / LM head", "Cache / LM head"])
    elif name.startswith("fig5_damage_by_band_"):
        ax.set_xlabel("Pruned VLM layers")
        ax.set_xticks([0, 1, 2], ["0–21", "22–35", "0–35"])
        ax.set_ylabel("FM loss increase" if name.endswith("fm") else "CoC NLL increase")
        legend(ax, ["Dual", "Traj", "CoC"], loc="lower center", ncol=3, above=True)
    elif name == "fig5_dual_saves":
        ax.set_xlabel("Removed unit set")
        ax.set_ylabel("CoC NLL increase (%)")
        ax.set_xticks([0, 1, 2, 3], ["Q\nvs Traj", "Q\nvs CoC", "MLP\nvs Traj", "MLP\nvs CoC"])
        for bar in ax.patches[::2]:
            bar.set_facecolor("#009E73")
        legend(ax, ["Selected set", "Random sets"], loc="lower center", ncol=2, above=True)
    elif name == "fig4_token_handoff_mlp":
        ax.set_ylabel("Vision-token contribution")
        legend(ax, [r"$I_{\mathrm{traj}}$", r"$I_{\mathrm{CoC}}$"], loc="upper right")
    elif name == "fig4_itraj_ports_mlp":
        ax.set_ylabel("Normalized importance")
        # get_legend_handles_labels returns the total Line2D before the BarContainers.
        legend(ax, ["Total", "Cache 1–21", "Cache 22–24", "Cache 25–29", "Cache 30–35"],
               loc="lower center", ncol=2, above=True)
    elif name == "fig5_vv_window_profile":
        ax.set_ylabel("Normalized profile")
        legend(
            ax,
            [r"$I_{\mathrm{traj}}$ (vision)", r"$I_{\mathrm{CoC}}$ (vision)", "Knockout effect"],
            loc="lower center", ncol=2, above=True,
        )

    ax.set_xlim(*original_xlim)

    for axis in fig.axes:
        axis.xaxis.label.set_fontsize(12)
        axis.yaxis.label.set_fontsize(12)
        axis.tick_params(axis="both", which="both", labelsize=10)
        for text in (*axis.texts, axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            text.set_fontsize(10)
        for line in axis.lines:
            if line.get_linestyle() not in ("None", "", "none"):
                line.set_linestyle("-")
        for collection in axis.collections:
            if isinstance(collection, LineCollection):
                collection.set_linestyle("solid")
        leg = axis.get_legend()
        if leg:
            for line in leg.get_lines():
                line.set_linestyle("-")
        axis.grid(axis="y", linestyle="-", alpha=0.7)
        axis.set_axisbelow(True)
    # The first layout can change the automatic tick locations (e.g. 8 -> 10.0).
    # Re-measure their final widths before saving on the fixed-size canvas.
    for _ in range(2):
        fig.tight_layout(pad=0.75)
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Keep long rotated labels inside the fixed canvas, as in fig_criterion.py.
    for axis in fig.axes:
        label = axis.yaxis.label
        box = label.get_window_extent(renderer)
        margin = 3 * fig.dpi / 72
        delta = max(margin - box.y0, 0) - max(box.y1 - fig.bbox.height + margin, 0)
        if delta:
            label.set_y(label.get_position()[1] + delta / axis.bbox.height)
    for extension in FORMATS:
        fig.savefig(target / f"{name}.{extension}", dpi=600)
    plt.close(fig)
    exported.add(name)


def same_token_mlp_panel(outputs, _stats):
    """MLP counterpart of anatomy_panels' Q-head plot, using the same correction."""
    metrics = json.loads((outputs / "gradanat_v1/metrics_analysis.json").read_text())
    fig, ax = source.depth_axes(r"Corrected Spearman’s $\rho$", (-0.3, 1.1))
    series = (
        ("pooled (the shipped score)", "All tokens", "black", 1.6),
        ("both at VISION positions", "Vision tokens", source.TRAJ, 1.1),
        ("both at TEXT-side positions", "Text tokens", source.PURPLE, 1.1),
    )
    for key, label, color, width in series:
        data = metrics["P2"]["mlp"][key]["per_layer"]
        st, sc, cross = (np.asarray(data[k]) for k in ("self_traj", "self_coc", "cross"))
        reliable = (st > 0.2) & (sc > 0.2)
        corrected = np.where(
            reliable, cross / np.sqrt(np.where(reliable, st * sc, 1.0)), np.nan,
        )
        ax.plot(np.arange(len(corrected)), corrected, color=color, lw=width, label=label)
    ax.axhline(0, color="black", lw=0.7)
    source.save(fig, "fig4_same_token_mlp")


def write_gallery(out):
    sections = [
        ("본문: 관찰 → 원인 → 실제 개입", {
            **{k: ("", v) for k, v in EXISTING.items() if k.startswith("fig1_")},
            **{k: v for k, v in ADDITIONS.items() if not v[0]},
        }),
        ("본문: expert 구조 선택의 동기", {
            k: ("", v) for k, v in EXISTING.items() if k.startswith("fig2_")
        }),
        ("보충: 단위 개입·gradient 분해·vision interaction", {
            k: v for k, v in ADDITIONS.items() if v[0]
        }),
        ("실제 성능과 압축 절충", {k: ("tables", v) for k, v in TABLES.items()}),
    ]
    body = []
    for title, panels in sections:
        cards = []
        for name, (folder, caption) in panels.items():
            base = f"{folder}/{name}" if folder else name
            cards.append(
                f'<article><a href="{base}.pdf"><img src="{base}.png" '
                f'alt="{html.escape(caption)}" loading="lazy"></a>'
                f'<p>{html.escape(caption)}</p><small>{name}</small><p>'
                + " · ".join(f'<a href="{base}.{e}">{e.upper()}</a>' for e in FORMATS)
                + "</p></article>"
            )
        body.append(f"<h2>{title}</h2><section>{''.join(cards)}</section>")
    (out / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Dual + expert MLP — 논문 피겨</title><style>'
        'body{max-width:1250px;margin:40px auto;padding:0 20px;font:16px/1.6 sans-serif;'
        'color:#203344;background:#fafafa}section{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}'
        'article{background:white;padding:16px;border:1px solid #ddd;border-radius:8px}'
        'img{width:100%;height:auto}a{color:#3568a3}small{overflow-wrap:anywhere}</style>'
        '<h1>Dual + expert MLP pruning</h1>'
        '<p>관찰 → 출력 경로 분석 → 실제 구조 제거 → expert 구조 선택 → 성능·비용 절충</p>'
        '<p><a href="CLAIMS.md">주장·캡션·해석 범위</a> · '
        '<a href="README.md">파일 안내·재현 방법</a> · '
        '<a href="manifest.json">출처·파일 해시</a></p>'
        '<p>기존 6개 피겨 유지. 추가 분석 피겨: 축 12 pt, 기타 10 pt, 실선, 5:4 비율. '
        '성능 표는 기존 검증된 파일의 복사본.</p>'
        + "".join(body) + "</html>", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "paper-fig")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    existing = {out / f"{name}.{ext}": digest(out / f"{name}.{ext}")
                for name in EXISTING for ext in FORMATS}
    for path in INPUTS:
        if not (REPO / path).exists():
            raise FileNotFoundError(REPO / path)
    exported = set()
    source.TRAJ, source.COC = "#618BC8", "#ECAE3C"
    original_save = source.save
    source.save = lambda fig, name: export_panel(fig, name, out, exported)
    try:
        for generate in (
            source.anatomy_panels, same_token_mlp_panel, source.port_panels, source.causal_panels,
        ):
            generate(REPO / "outputs", {})
    finally:
        source.save = original_save
    if exported != ADDITIONS.keys():
        raise RuntimeError(f"Missing panels: {ADDITIONS.keys() - exported}")
    if any(digest(path) != before for path, before in existing.items()):
        raise RuntimeError("An existing paper figure was changed")
    (out / "tables").mkdir(exist_ok=True)
    sources = list(INPUTS)
    for name in TABLES:
        for ext in (*FORMATS, "csv"):
            path = REPO / "ppt-fig" / f"{name}.{ext}"
            shutil.copy2(path, out / "tables" / path.name)
            sources.append(str(path.relative_to(REPO)))
    write_gallery(out)
    manifest = {
        "purpose": "Stored-result curation; no new model evaluation",
        "style": {"canvas_inches": [3.5, 2.8], "axis_pt": 12, "text_pt": 10,
                  "dpi": 600, "line_style": "solid", "traj": "#618BC8", "coc": "#ECAE3C",
                  "tables": "Unmodified copies; original table dimensions and typography"},
        "existing_unchanged": {str(p.relative_to(out)): h for p, h in existing.items()},
        "added_panels": {k: {"folder": v[0], "caption": v[1]} for k, v in ADDITIONS.items()},
        "sources_sha256": {p: digest(REPO / p) for p in sources},
        "artifacts_sha256": {str(p.relative_to(out)): digest(p)
                             for p in sorted(out.rglob("*"))
                             if p.suffix in (".png", ".pdf", ".svg", ".csv")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(exported)} additional panels and {len(TABLES)} tables to {out}")


if __name__ == "__main__":
    main()
