# Method
<!-- label: sec:structured_pruning -->

<!-- NOTE 2026-09-23: the band-restricted ablation was removed from the analysis pending a new Finding 2. -->
<!-- Every "[band ablation, pending new Finding 2]" below, and Table [tab:band_ablation], cites that removed -->
<!-- finding (its text is parked in paper-analysis/removed_finding2_band_ablation.md); revise once the new -->
<!-- Finding 2 exists. "Finding 2" in the expert section means the action-expert finding (was Finding 3). -->
<!-- Revision of method.tex (2026-09-23): the rules are unchanged; each one now carries the finding -->
<!-- of Sec.~\ref{sec:analysis} that motivates it (formerly a separate "From Analysis to Design" -->
<!-- subsection at the end of the analysis). Equations, labels and budgets are the originals. -->

Our approach uses the preceding analysis to select VLM units with both CoC and trajectory importance and to select action expert MLP channels using importance measured across denoising steps.

## VLM Pruning with CoC and Trajectory Importance
<!-- label: sec:vlm_pruning -->

We score every VLM unit twice, with $I^{\mathrm{CoC}}_{\ell,u}$ and $I^{\mathrm{traj}}_{\ell,u}$ measured separately on the dense model, rather than differentiating one combined loss. Finding 1 is the reason: in the late layers the two objectives depend on what a unit writes at different positions and rank units differently, and neither score is a function of the other. A single combined loss could not represent this even in principle – the FM score is $6$–$10\times$ larger than the CE score in raw units, so any combination made before ranking is decided by the trajectory objective alone.

For each VLM layer, we rank Q heads and MLP channels separately using CoC and trajectory importance. Let $r^o_{\ell,u}$ denote the rank of unit $u$ under objective $o\in\{\mathrm{CoC},\mathrm{traj}\}$, with larger ranks indicating greater importance. We use the larger of the two ranks as the final score:

$$
S^{\mathrm V}_{\ell,u}
=\max\!\left(r^{\mathrm{CoC}}_{\ell,u},
r^{\mathrm{traj}}_{\ell,u}\right).
$$

This rule prioritizes units that rank highly under either objective. Ranking within a layer puts the two scores on a common scale, and taking the larger rank means a unit's score is never lowered by the objective that does not use it. [band ablation, pending new Finding 2] is why this union, rather than either score, is the right rule at both depths. In the late band only the CoC objective's importance corresponds to what its loss depends on once units are removed – no late-layer selection changes the FM loss, while discarding the units the trajectory score alone drops raises the CoC NLL by $65\%$ – so the CoC rank must be able to keep a unit on its own. In the trunk both losses depend on the units that either score ranks highly, and keeping their union preserves the action far better than either score alone (FM loss $+4.8\%$ against $+17.0\%$ and $+26.1\%$; Table [tab:band_ablation]). We claim only that both scores are needed and that their union avoids each single-objective failure, not that the maximum is the optimal combining operator.

For each unit type, we retain the units with the highest combined scores and remove a fixed number per pruned layer. The head and channel budgets are specified separately and remain constant across those layers. Four considerations support the uniform budget. First, the depth profiles of Finding 1 say *which* units to keep, not how many per layer: importance mass is not removal damage (the late-layer trajectory mass is a first-order shadow, [band ablation, pending new Finding 2]), and the two profiles would prescribe opposite tilts. Second, the within-layer spread of importance – the only first-order quantity that can inform an allocation, since scores are not comparable across layers ([band ablation, pending new Finding 2]) – itself prescribes a uniform ratio: importance is spread to a similar degree in every layer (Gini 0.2–0.3 for MLP channels in layers 6–34 under either objective, with no trend in depth), the uniform cut removes only 17–31% of a layer's MLP importance against 40% of its parameters, and an allocation that equalised the removed share across layers would set per-layer ratios of 0.35–0.57 (s.d. 0.03–0.05 around the uniform 0.40). The exception is MLP layers 1–5, where importance sits in a few channels (Gini 0.88) and a first-order allocation would remove far more; we leave that option untested. Third, [band ablation, pending new Finding 2] shows that the late band tolerates the trunk's rate provided the CoC score has a vote there. Fourth, at a matched 40% budget no non-uniform allocation improved on the uniform one (50 validation clips, dual criterion): the late-heavy profile that the trajectory curve suggests raised the CoC NLL by $+1.58$ against $+0.08$ and the CoC degeneracy rate to 30% against 6% with no minADE gain ($+0.11$ against $+0.10$ m); an allocation that follows the per-layer agreement of the two objectives matched uniform on NLL but lost on minADE ($+0.18$ m); a depth-tilted allocation matched on minADE but doubled the degeneracy rate. A uniform budget also keeps the comparison between selection rules one-factor.
<!-- Panels: paper-analysis/figures/supplementary/fig5_removed_mass_by_layer.pdf (scripts/removed_mass.py) -->
<!-- and fig5_layer_dispersion.pdf (scripts/layer_dispersion.py: Gini, equalising allocation). Grid: outputs/grid_v1/summary.txt (criterion x allocation, val 50 clips, K=8, KV and expert -->
<!-- fixed); report reports/archive/2026-07-29_criterion-vs-allocation-grid.html. -->

## Action Expert MLP Pruning
<!-- label: sec:expert_pruning -->

We use FM Taylor scores $A_{s,\ell,u}$ measured on the dense model
and averaged over calibration clips. Because MLP-channel rankings are stable across denoising steps and their importance concentrates in a small fraction of channels (Finding 2), one channel set can serve every step. For each layer and step,
we standardize MLP channel importance scores to zero mean and unit variance. The final score averages these values over the 10 steps:
$$
S^{\mathrm E}_{\ell,u}
= \frac{1}{10}\sum_{s=0}^{9}Z_{s,\ell,u}.
$$
In each of the 36 expert layers, we retain the 516 highest-scoring
channels out of $8{,}256$, removing $93.75\%$ of MLP channels.
Each layer's retained channels remain fixed across denoising steps,
and expert attention is unchanged: Q-head importance is broadly distributed (14 of 16 heads carry $68.5\%$ of a layer's importance, Finding 2), so heads are not pruned.

## Pruning Procedure
<!-- label: sec:pruning_procedure -->

We compute all importance scores on the original dense model,
then physically remove the selected structures and their dependent
weight slices. The retained weights remain unchanged, and the
compressed model is evaluated without post-pruning fine-tuning.

VLM pruning alone removes approximately $24\%$ of the original
model's parameters. Expert MLP pruning removes an additional
$16$ percentage points, bringing the total reduction to
approximately $40\%$.
