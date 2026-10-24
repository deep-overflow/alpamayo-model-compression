# Figure manifest (2026-09-26)

Folders: `1_main` = in the paper now (the 7 analysis figures), `2_supplementary` = proposed for the
supplement, `3_unused` = superseded variants and analysis-style plots. Render scripts live in `scripts/`
(paper style: `paper_*.py`, `head_map_combined.py`; outputs land in `scripts/paperfigs/` or `scripts/figs/` and are copied here).

## 1_main

| figure | formats | what it shows |
|---|---|---|
| `fig4_same_position_q_head` | pdf png svg | same-position rank agreement, Q heads, raw (analysis) |
| `fig4_token_share_traj_q_head_split` | pdf png svg | I_traj share by position type, Q heads (analysis) |
| `fig4_token_share_coc_q_head_split` | pdf png svg | I_CoC share by position type, Q heads (analysis) |
| `fig6e_head_map_combined_sortq` | pdf png | position type + rank maps, Q heads / MLP, Q sorted by I_FM rank (analysis) |
| `fig2_mass_curve` | pdf png | importance removed vs fraction removed (analysis) |
| `fig2_steps_mlp` | pdf png | denoising-step rank agreement, MLP (analysis) |
| `fig2_steps_q_head` | pdf png | denoising-step rank agreement, Q heads (analysis) |

## 2_supplementary

| figure | formats | what it shows |
|---|---|---|
| `fig5_removed_mass_by_layer` | pdf png | method_v2.md panel (removed mass by layer; was in supplementary/ before) -- CONFIRM where it goes |
| `fig5_layer_dispersion` | pdf png | method_v2.md panel (Gini / equalising allocation; was in supplementary/ before) -- CONFIRM where it goes |
| `fig4_same_position_mlp` | pdf png svg | same-position agreement, MLP twin of the main figure (raw) |
| `fig4_token_share_traj_mlp_split` | pdf png svg | I_traj share by position type, MLP twin |
| `fig4_token_share_coc_mlp_split` | pdf png svg | I_CoC share by position type, MLP twin |
| `fig6e_head_map_combined_sortq_diff` | pdf png | main map + rank(I_FM) - rank(I_CE) column |
| `fig1_rank_agreement_val100` | pdf png svg | held-out replication: pooled agreement, val100 |
| `fig1_rank_agreement_ood` | pdf png svg | held-out replication: pooled agreement, OOD |
| `fig1_rank_agreement_test100` | pdf png svg | held-out replication: pooled agreement, test100 |
| `fig4_token_share_traj_q_head_val100` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_traj_q_head_ood` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_traj_q_head_test100` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_coc_q_head_val100` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_coc_q_head_ood` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_coc_q_head_test100` | pdf png svg | held-out replication (3-curve; re-render with split) |
| `fig4_token_share_traj_mlp_val100` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig4_token_share_traj_mlp_ood` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig4_token_share_traj_mlp_test100` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig4_token_share_coc_mlp_val100` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig4_token_share_coc_mlp_ood` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig4_token_share_coc_mlp_test100` | pdf png svg | held-out replication, MLP (3-curve) |
| `fig6_head_token_map_q_head_val100` | pdf png | held-out head-token map (old 3-column layout; re-render as combined) |
| `fig6_head_token_map_q_head_ood` | pdf png | held-out head-token map (old 3-column layout; re-render as combined) |
| `fig6_head_token_map_q_head_test100` | pdf png | held-out head-token map (old 3-column layout; re-render as combined) |
| `heldout_replication_sheet` | png | held-out numbers sheet (MLP) |
| `heldout_replication_sheet_q` | png | held-out numbers sheet (Q heads) |
| `fig9_coc_position_split` | pdf png svg | CoC exception = token mixture: boundary tokens + same-sub-span agreement |
| `fig10_coc_position_dissociation` | pdf png svg | double dissociation at CoC positions (run B) |
| `fig13_text_tokens` | pdf png svg | which text tokens carry I_FM / I_CE; ego-history recency (run D) |
| `fig8b_coc_attention_bars` | pdf png svg | attention census at CoC queries (paper style) |
| `fig12_coc_query_ego_history_by_position` | png | ego-history attention by CoC query position (analysis style; re-render for paper) |
| `fig11c_coc_words` | pdf png svg | CoC words carrying I_traj / I_CoC (action vs cause clause) |
| `fig5_dual_saves` | pdf png | dual-saves trunk test (units dual keeps over traj-only are language units) |
| `fig5_dissociation_q_head` | pdf png | k=6 chosen-head dissociation, FM vs NLL damage, Q heads |
| `fig5_dissociation_mlp` | pdf png | k=6 chosen-channel dissociation, MLP |
| `fig5_matched_fm` | pdf png | per-clip FM change of the three arms vs dense |
| `fig5_matched_nll` | pdf png | per-clip NLL change of the three arms vs dense |
| `fig1_depth_q_head` | pdf png | depth profile of I_traj / I_CoC, Q heads (was inside the old fig3 wrapper) |
| `fig1_depth_mlp` | pdf png | depth profile of I_traj / I_CoC, MLP |
| `fig1_rank_agreement` | pdf png svg | pooled rank agreement Q/MLP (analysis_v2.md still cites it; superseded by the 'All' curve of fig4_same_position) |
| `fig4_token_handoff5_mlp` | pdf png | vision->text hand-over, MLP (analysis_v2.md still cites it) |
| `fig4_token_handoff5_q_head_bands` | pdf png | vision->text hand-over, Q heads, band means (analysis_v2.md cites it) |
| `fig3_ratio_step` | pdf png | I_traj / I_CoC ratio step at layers 22-23 (README: 'candidate') |

## 3_unused

| figure | formats | what it shows |
|---|---|---|
| `fig4_same_position_q_head_corrected` | pdf png svg | ceiling-corrected variant (raw chosen) |
| `fig4_same_position_mlp_corrected` | pdf png svg | ceiling-corrected variant (raw chosen) |
| `fig4_same_token5_q_head` | pdf png | older corrected same-position figure |
| `fig4_same_token5_mlp` | pdf png | older corrected same-position figure |
| `fig4_same_token_q_head` | pdf png | oldest same-token figure |
| `fig4_token_share_traj_q_head` | pdf png svg | 3-curve version, superseded by _split |
| `fig4_token_share_coc_q_head` | pdf png svg | 3-curve version, superseded by _split |
| `fig4_token_share_traj_mlp` | pdf png svg | 3-curve version, superseded by _split |
| `fig4_token_share_coc_mlp` | pdf png svg | 3-curve version, superseded by _split |
| `fig4_token_handoff5_q_head` | pdf png | per-layer lines version of the hand-over |
| `fig4_token_handoff5_q_head_ci` | pdf png | hand-over with bootstrap bands |
| `fig4_token_handoff_mlp` | pdf png | 2-line hand-over (vision share only) |
| `fig4_token_handoff_q_head` | pdf png | 2-line hand-over (vision share only) |
| `fig4_pertoken_mlp` | pdf png | importance per token by type |
| `fig4_pertoken_q_head` | pdf png | importance per token by type |
| `fig4_ratio_by_token5_mlp` | pdf png | I_traj / I_CoC ratio by token type |
| `fig4_ratio_by_token5_q_head` | pdf png | I_traj / I_CoC ratio by token type |
| `fig3_noise_ceiling_q_head` | pdf png | split-half noise ceiling (background for the corrected values) |
| `fig3_noise_ceiling_mlp` | pdf png | split-half noise ceiling |
| `fig5_band_mix` | pdf png | removed finding 2 (band ablation) |
| `fig5_damage_by_band_fm` | pdf png | removed finding 2 (band ablation) |
| `fig5_damage_by_band_nll` | pdf png | removed finding 2 (band ablation) |
| `fig6_head_token_map_q_head` | pdf png | 3-column head-token map, superseded by fig6e |
| `fig6b_head_token_map_mlp` | pdf png | 3-column MLP map, superseded by fig6e |
| `fig6c_head_map_plain` | pdf png | rank-only maps, folded into fig6e |
| `fig6d_head_token_map_dom_q_head` | pdf png | dominant-share-only panels, folded into fig6e |
| `fig6d_head_token_map_dom_mlp` | pdf png | dominant-share-only panels, folded into fig6e |
| `fig6e_head_map_combined` | pdf png | combined map with Q heads in index order (variant) |
| `fig6e_head_map_combined_diff` | pdf png | variant |
| `fig6e_head_map_combined_idxmlp` | pdf png | variant |
| `fig7_failure_channels` | png | analysis-style failure channels plot |
| `fig8_coc_query_attention` | png | analysis-style census plot |
| `fig8b_coc_query_attention_bars` | png | analysis-style census bars |
| `fig8c_coc_path_split` | pdf png svg | path split (negative result), raw |
| `fig8c_coc_path_split_corrected` | pdf png svg | path split, corrected |
| `fig9_coc_same_subspan_agreement_analysis` | png | analysis-style twin of fig9 |
| `fig10_coc_position_dissociation_analysis` | png | analysis-style twin of fig10 |
| `fig11_coc_token_category_shares` | png | analysis-style category shares |
| `fig11b_coc_token_category_enrichment` | png | analysis-style category enrichment |
| `fig13_text_role_shares_analysis` | png | analysis-style twin of fig13 |
| `fig13b_prompt_position_share_analysis` | png | analysis-style prompt-position curves |
| `image` | png | pasted layout reference screenshot |

