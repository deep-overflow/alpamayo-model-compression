# Analysis
<!-- label: sec:analysis -->

<!-- Revision of analysis.tex (2026-09-23). Finding 1 = rank agreement + token-level anatomy (two figures); -->
<!-- Finding 2 = action expert (unchanged). The band-restricted ablation that used to be Finding 2 is -->
<!-- parked in paper-analysis/removed_finding2_band_ablation.md pending the new Finding 2. -->

## Analysis Setup
<!-- label: sec:analysis_setup -->

We analyze Alpamayo-1.5-10B [wang2025alpamayo] to find out how importance scores distribute in the VLM backbone across the two objectives and in the action expert across denoising steps. We use 100 calibration clips selected from the PhysicalAI Autonomous Vehicles training dataset [wang2025alpamayo] for Taylor importance computation; the sampling procedure is described in Appendix [app:calibration_sampling]. For each VLM unit $u$ in layer $\ell$ (a query head or an MLP channel) we obtain two scores: $I^{\mathrm{CoC}}_{\ell,u}$ from the CE loss of the model's own CoC rollout and $I^{\mathrm{traj}}_{\ell,u}$ from the FM loss averaged over the 10 denoising steps.

## Im portance Score in VLM Backbone
<!-- label: sec:vlm_analysis -->

> **Finding 1.** The two objectives rank the units of a layer differently, and increasingly so with depth, because each objective's importance follows the token positions its output depends on: in the early layers both are earned mainly at vision positions and the rankings nearly agree; in the late layers CoC importance is earned almost entirely at text positions, whereas trajectory importance stays spread over vision, text and ego-history positions.

*[Figure: fig1_rank_agreement]*
<!-- paper-analysis/figures/main/fig1_rank_agreement.pdf -->

We first compare, within each layer, how the two objectives rank the layer's units (Figure [fig:vlm_rank_correlation]). The within-layer Spearman correlation between the two rankings is $0.75$–$0.78$ over layers 0–21 and falls to $0.34$ over layers 22–34, for Q heads and MLP channels alike – a genuine disagreement rather than calibration noise (corrected for each score's split-half reliability it is still $0.37$ and $0.54$ in the late band). The two objectives thus value different units, and increasingly so with depth; we refer to layers 0–21 as the *trunk* and to layers 22–35 as the *late* band.

*[Figure: fig4_token_handoff5_mlp]*
<!-- paper-analysis/figures/main/fig4_token_handoff5_mlp.pdf; Q heads: figures/main/fig4_token_handoff5_q_head_bands.pdf (supplement) -->

To see concretely which units each objective values, we split each score by the token position at which the unit's output enters the loss. A unit's gate gradient is a sum over positions, so a layer's score decomposes exactly into the shares earned at vision (2,880 tokens per clip), ego-history (48), prompt-text (157) and generated-CoC (about 16) positions (Figure [fig:token_handoff]; MLP channels shown, Q heads in the supplement). In the trunk both objectives earn their importance mainly at vision positions (90% of $I^{\mathrm{traj}}$ and 75% of $I^{\mathrm{CoC}}$ in layers 6–17); the CoC objective also draws on text positions (24% against 7% for $I^{\mathrm{traj}}$), which is where the small trunk disagreement comes from. In the late band the two objectives part: $I^{\mathrm{CoC}}$ is earned almost entirely at text positions (98% in layers 27–34), whereas $I^{\mathrm{traj}}$ stays spread over vision (26%), text (66%) and ego-history (8%) positions (Q heads: 35%, 49% and 16%). The disagreement is indeed a matter of position: read at the same kind of position, the late-band rankings agree again (ceiling-corrected agreement $0.38\!\to\!0.79$ at vision, $0.81$ at prompt-text and $0.94$ at ego-history positions for Q heads), except at the generated-CoC positions ($0.41$).

Each objective's importance therefore follows the inputs its output depends on – language positions for CoC generation; vision, ego-history and text positions for trajectory prediction – and a unit that matters for one output can be nearly invisible to the other objective's score. Selecting units with one loss alone would discard units the other objective values; this is the case for scoring with both. These are first-order shares of the importance scores; whether each loss still depends on those units once they are removed is a separate question, which the evaluation of the pruned models addresses.

## Importance Score in Action Expert
<!-- label: sec:expert_analysis -->

> **Finding 2.** In the action expert, MLP-channel importance is more stable across denoising steps and more concentrated than Q-head importance.

*[Figure: fig/fig4_expert_analysis]*

We next examine how the expert's importance rankings vary across the 10 denoising steps (Figure [fig:expert_importance]). For each layer and unit type we correlate the unit-score vectors $A_{s,\ell,:}$ of every pair of steps (Spearman) and average over the 36 expert layers. Q-head rankings are similar between adjacent steps but drift as the step distance grows (Fig. [fig:expert_importance]([fig:expert_qhead_rank])), whereas MLP-channel rankings stay similar across all steps (Fig. [fig:expert_importance]([fig:expert_mlp_rank])): over the 45 step pairs the mean correlation is $0.668$ for Q heads and $0.874$ for MLP channels.

Importance is also distributed differently within the two unit types. Summing the per-step scores and sorting units within each layer, we measure the fraction of a layer's importance held by its lowest-ranked units (Fig. [fig:expert_importance]([fig:expert_cumulative])). Removing 14 of 16 Q heads ($87.5\%$) removes about $68.5\%$ of a layer's importance on average, whereas removing $90\%$ of the $8{,}256$ MLP channels removes only $1.8\%$: Q-head importance is broadly distributed and MLP-channel importance is concentrated in the top $10\%$ of channels. The stable rankings and concentrated importance motivate retaining a small, fixed set of channels across all steps and leaving expert attention intact.
