# Q head별 지배 토큰 자리의 held-out 재현 (2026-09-24)

**질문.** `fig6_head_token_map_q_head`(calib_100)에서 head마다 정한 지배 자리(vision / prompt-text / 생성 CoC / ego-history)가
캘리브레이션에 쓰지 않은 클립(val100 / OOD / test100)에서도 같은가?

**방법.** 세트마다 head별 가법 분해 `C[t,l,u] = E_c[G sign(Σ_t G)]`(Σ_t C = I)로 지배 자리(argmax_t C)와 연속 몫(C_t / I)을
구해 calib_100과 비교. 잡음 천장은 calib_100을 50클립 두 반쪽으로 50번 나눠 반쪽끼리 비교한 값(라벨이 50클립에서
추정되므로 100 대 100 비교보다 낮은, 보수적인 천장). κ는 우연 일치를 뺀 Cohen's kappa.
계산: `paper-analysis/scripts/head_type_stability.py`. 지도: `figures/supplementary/fig6_head_token_map_q_head_{val100,ood,test100}.*`.

## 결과: 유지된다

- **지배 자리 일치율 75–86%(κ 0.61–0.79)**, 모든 셀에서 calib 안 반쪽끼리 일치율(69–81%, κ 0.53–0.69)과 같거나 높다.
  즉 세트 간 차이는 표본 잡음이 허용하는 수준 안이고, 세트 특유의 지배 자리는 보이지 않는다.
- calib 순위 상위 19개 head(출하 예산의 유지 수)로 제한하면 79–90%로 더 높다.
- 연속 몫(head별 vision / text / CoC 몫)의 세트 간 Spearman은 0.65–0.96으로 역시 반쪽 천장(0.50–0.91) 위. 예외는
  `I_CoC` 후반층의 ego-history 몫(0.22–0.33, 천장 0.15)인데, 이 몫은 거의 0이라 순위 자체가 잡음이다.
- 지배 자리별 head 비율(마진)도 세트 간 ±0.10 안. 가장 큰 차이는 후반층에서 held-out 세트가 calib보다 vision 지배
  head(`I_traj` 0.47 → 0.52–0.57)와 CoC 지배 head(`I_CoC` 0.64 → 0.67–0.74)를 조금 더 많이 두는 것.

## 표 (스크립트 출력)

## 1. Share of heads by dominant position type

| score | band | set | vision | prompt-text | generated CoC | ego-history | median dominant share |
|---|---|---|---:|---:|---:|---:|---:|
| I_traj | 0-21 | calib_100 | 0.62 | 0.17 | 0.10 | 0.12 | 0.68 |
| I_traj | 0-21 | val100 | 0.65 | 0.13 | 0.09 | 0.13 | 0.68 |
| I_traj | 0-21 | OOD | 0.69 | 0.13 | 0.07 | 0.12 | 0.72 |
| I_traj | 0-21 | test100 | 0.67 | 0.11 | 0.10 | 0.12 | 0.70 |
| I_traj | 22-34 | calib_100 | 0.47 | 0.13 | 0.32 | 0.08 | 0.56 |
| I_traj | 22-34 | val100 | 0.54 | 0.12 | 0.28 | 0.06 | 0.55 |
| I_traj | 22-34 | OOD | 0.57 | 0.12 | 0.26 | 0.04 | 0.57 |
| I_traj | 22-34 | test100 | 0.52 | 0.12 | 0.31 | 0.06 | 0.55 |
| I_coc | 0-21 | calib_100 | 0.46 | 0.25 | 0.27 | 0.03 | 0.63 |
| I_coc | 0-21 | val100 | 0.45 | 0.25 | 0.27 | 0.03 | 0.61 |
| I_coc | 0-21 | OOD | 0.46 | 0.26 | 0.25 | 0.03 | 0.62 |
| I_coc | 0-21 | test100 | 0.41 | 0.22 | 0.33 | 0.03 | 0.61 |
| I_coc | 22-34 | calib_100 | 0.01 | 0.34 | 0.64 | 0.00 | 0.70 |
| I_coc | 22-34 | val100 | 0.02 | 0.26 | 0.72 | 0.00 | 0.72 |
| I_coc | 22-34 | OOD | 0.02 | 0.31 | 0.67 | 0.00 | 0.73 |
| I_coc | 22-34 | test100 | 0.01 | 0.25 | 0.74 | 0.00 | 0.72 |

## 2. Head-wise agreement of the dominant type with calib_100 (raw fraction / Cohen's kappa)

| score | band | heads | within-calib split-half (ceiling) | val100 | OOD | test100 |
|---|---|---|---:|---:|---:|---:|
| I_traj | 0-21 | all | 0.73 / 0.54 | 0.83 / 0.70 | 0.83 / 0.68 | 0.83 / 0.68 |
| I_traj | 0-21 | top-19 (calib rank) | 0.78 / 0.60 | 0.88 / 0.77 | 0.86 / 0.73 | 0.86 / 0.73 |
| I_traj | 22-34 | all | 0.69 / 0.53 | 0.80 / 0.68 | 0.75 / 0.61 | 0.78 / 0.66 |
| I_traj | 22-34 | top-19 (calib rank) | 0.72 / 0.58 | 0.83 / 0.75 | 0.79 / 0.69 | 0.81 / 0.72 |
| I_coc | 0-21 | all | 0.79 / 0.69 | 0.86 / 0.79 | 0.84 / 0.76 | 0.85 / 0.78 |
| I_coc | 0-21 | top-19 (calib rank) | 0.85 / 0.76 | 0.90 / 0.84 | 0.88 / 0.81 | 0.88 / 0.82 |
| I_coc | 22-34 | all | 0.81 / 0.61 | 0.86 / 0.69 | 0.85 / 0.68 | 0.85 / 0.65 |
| I_coc | 22-34 | top-19 (calib rank) | 0.81 / 0.58 | 0.86 / 0.68 | 0.85 / 0.67 | 0.88 / 0.71 |

## 3. Per-head shares, Spearman with calib_100 across heads of the band (continuous version)

| score | band | share | within-calib split-half (ceiling) | val100 | OOD | test100 |
|---|---|---|---:|---:|---:|---:|
| I_traj | 0-21 | vision | 0.78 | 0.89 | 0.88 | 0.89 |
| I_traj | 0-21 | text (prompt+CoC) | 0.66 | 0.82 | 0.81 | 0.81 |
| I_traj | 0-21 | generated CoC | 0.50 | 0.70 | 0.71 | 0.71 |
| I_traj | 0-21 | ego-history | 0.61 | 0.80 | 0.76 | 0.78 |
| I_traj | 22-34 | vision | 0.72 | 0.86 | 0.84 | 0.88 |
| I_traj | 22-34 | text (prompt+CoC) | 0.72 | 0.86 | 0.84 | 0.88 |
| I_traj | 22-34 | generated CoC | 0.69 | 0.85 | 0.81 | 0.84 |
| I_traj | 22-34 | ego-history | 0.50 | 0.74 | 0.74 | 0.75 |
| I_coc | 0-21 | vision | 0.91 | 0.95 | 0.95 | 0.96 |
| I_coc | 0-21 | text (prompt+CoC) | 0.90 | 0.95 | 0.94 | 0.95 |
| I_coc | 0-21 | generated CoC | 0.82 | 0.91 | 0.89 | 0.92 |
| I_coc | 0-21 | ego-history | 0.51 | 0.67 | 0.62 | 0.67 |
| I_coc | 22-34 | vision | 0.60 | 0.65 | 0.69 | 0.70 |
| I_coc | 22-34 | text (prompt+CoC) | 0.62 | 0.68 | 0.70 | 0.72 |
| I_coc | 22-34 | generated CoC | 0.86 | 0.92 | 0.88 | 0.92 |
| I_coc | 22-34 | ego-history | 0.15 | 0.33 | 0.22 | 0.26 |
