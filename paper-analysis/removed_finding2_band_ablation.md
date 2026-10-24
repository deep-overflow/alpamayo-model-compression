# Removed from analysis_v2.md on 2026-09-23 (former Finding 2, band-restricted ablation)

Kept here for reuse; the numbers and sources are unchanged.

## Which Objective Sees Function in Each Depth Band
<!-- label: sec:band_ablation -->

> **Finding 2.** In the late band only the CoC objective's importance survives removal: no late-layer selection changes the FM loss, while discarding the units the trajectory score alone drops raises the CoC NLL by 65%. In the trunk both losses depend on the units that either score ranks highly, and keeping their union preserves the action far better than either score alone.

An importance score is a first-order estimate of how much a loss depends on a unit, so we test what removing the selected units actually does. At the per-layer budget of Sec. [sec:vlm_pruning] we build three selections – by $I^{\mathrm{CoC}}$ alone, by $I^{\mathrm{traj}}$ alone, and by the larger of the two within-layer ranks (Eq. [eq:vlm_combined_score]) – and apply each as a mask to the dense model in the trunk only, in the late band only, or everywhere, on 100 held-out clips. The dense model's own CoC text is teacher-forced so that all selections are scored on identical tokens: CoC NLL, FM loss over the 10 steps with the same noise, and minADE over 8 samples; the expert and the KV projections are untouched.

**Table [tab:band_ablation].** Band-restricted ablation of the three selection rules on 100 held-out clips, relative to the dense model (CoC NLL $0.171$, FM loss $0.243$, minADE $0.84$ m). Late-band masks leave the FM loss and minADE unchanged for every rule; trunk masks separate the rules on both channels.

| mask band | selection rule | $\Delta$FM loss | $\Delta$CoC NLL | $\Delta$minADE |
|---|---|---|---|---|
| late (22–35) | $I^{\mathrm{traj}}$ only | $-0.1\%$ | $+65.5\%$ | $+0.00$ m |
|  | $I^{\mathrm{CoC}}$ only | $+0.1\%$ | $+6.3\%$ | $+0.00$ m |
|  | union (Eq. [eq:vlm_combined_score]) | $+0.1\%$ | $+10.1\%$ | $+0.00$ m |
| trunk (0–21) | $I^{\mathrm{traj}}$ only | $+26.1\%$ | $+51.2\%$ | $+2.64$ m |
|  | $I^{\mathrm{CoC}}$ only | $+17.0\%$ | $+28.3\%$ | $+0.46$ m |
|  | union | $+4.8\%$ | $+25.2\%$ | $+0.06$ m |
| all layers | $I^{\mathrm{traj}}$ only | $+6.6\%$ | $+240.6\%$ | $+0.06$ m |
|  | $I^{\mathrm{CoC}}$ only | $+15.8\%$ | $+67.2\%$ | $+0.56$ m |
|  | union | $+5.5\%$ | $+32.6\%$ | $+0.01$ m |
<!-- All entries are mean deltas over the 100 clips from outputs/armheld_v1/summary.txt -->
<!-- (section iii), divided by the dense values. Optional figure: paper-fig/ -->
<!-- fig5_damage_by_band_fm.pdf and _nll.pdf show the FM and NLL columns as bars. -->

In the late band (Table [tab:band_ablation], top) the FM loss does not depend on the units' outputs at this budget: no selection changes it by more than $0.1\%$ or moves minADE, nor do random selections of the same size. The trajectory importance earned there – half or more of it at language positions (Finding 1) – is thus a first-order quantity with no counterpart in removal (damage per unit of $I^{\mathrm{traj}}$ falls from $0.011$ in the trunk to zero). The CoC NLL does depend on those outputs, and on exactly the units the two scores disagree about: the trajectory-only selection costs $+65.5\%$ against $+6.3\%$ (CoC only) and $+10.1\%$ (union), and removing just the late-band heads that the trajectory score discards and the union keeps costs $+8.6\%$ against $1.3$–$3.2\%$ for random sets of the same size. Letting $I^{\mathrm{traj}}$ decide in the late layers discards outputs the CoC objective depends on and gains nothing for the action.

In the trunk (Table [tab:band_ablation], middle) both losses depend on the same units, and on more of them than either score alone identifies: the FM loss rises by $4.8\%$ under the union but by $17.0\%$ under $I^{\mathrm{CoC}}$ alone and $26.1\%$ under $I^{\mathrm{traj}}$ alone, minADE by $0.06$ m against $0.46$ m and $2.64$ m, and the CoC NLL is also lowest under the union. This is the band in which the two scores agree at $0.85$–$0.96$ and differ on about two heads per layer, so the gain is not a compromise between two rankings: keeping a unit whenever *either* objective's importance is high preserves the action better than the action objective's own ranking. Applied everywhere (Table [tab:band_ablation], bottom), the single-objective rules fail on different channels – trajectory-only on language (NLL $+240.6\%$), CoC-only on action (minADE $+0.56$ m) – while the union keeps both within a few percent of the dense model.

The whole-model rows also show why the trajectory score alone is not enough even for driving, although its FM loss ($+6.6\%$) is close to the union's ($+5.5\%$): its trunk selection alone costs $+26.1\%$, and pruning the late band by any rule cancels part of that damage ($+6.6\%$ with its own late selection, $+13.5\%$ with the CoC-only one, $+8.8\%$ with a random one). The near-parity is a cancellation, not an equivalence. A depth-split rule – $I^{\mathrm{traj}}$ in the trunk where the scores agree, $I^{\mathrm{CoC}}$ in the late band where they disagree – costs FM $+13.5\%$ and minADE $+0.98$ m, and even with the union's own late band a trajectory-only trunk costs $+7.7\%$ and $+0.26$ m against $+5.5\%$ and $+0.01$ m. The union is needed in the trunk itself, for the action.
<!-- Mixed-band masks: outputs/armmix_v1/summary.txt (same 100 clips, same protocol); bar panel -->
<!-- paper-analysis/figures/main/fig5_band_mix.pdf. On the full open-loop evaluation of the physically -->
<!-- pruned arms (Sec. Experiments), trajectory-only costs 1.2-2.8x the union's minADE in cruise, -->
<!-- decel/stop and turns (outputs/failure_by_situation_v1). -->
Sec. [sec:structured_pruning] builds the selection rule on these findings.

