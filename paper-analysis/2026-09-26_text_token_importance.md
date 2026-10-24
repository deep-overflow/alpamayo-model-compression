# I_FM이 중요하게 보는 텍스트 토큰 — prompt·ego-history·CoC 위치별 anatomy (2026-09-26, Run D)

출력 `outputs/textpos_v1/` (shard `textpos_v1_s0-3`; `summary.txt`, `metrics.json`에 위치별 몫·history 프로파일·역할별 일치도), 스크립트
`experiments/head_analysis/run_text_position_anatomy.py`, `analyze_text_position_anatomy.py` (브랜치 `worktree-dual-saves-trunk`).
그림: 논문 스타일 `figures/supplementary/fig13_text_tokens.{png,pdf,svg}` (`scripts/paper_text_tokens.py`), 분석 스타일
`fig13_text_role_shares_analysis.png`, `fig13b_prompt_position_share_analysis.png`(non-vision prompt 위치 206개를 따라 두 점수의 몫).
계획·예측: `plans/2026-09-25_coc-position-content.md` §D.

## 설정

calib_100 중 99클립(Run A와 같은 클립·rollout seed·expert pass), dense 모델. vision을 제외한 **모든 위치**(sink 1 + prompt 텍스트 157 + ego-history 48 +
생성 CoC ≤ 64)에 typed gate를 하나씩 두고(vision은 한 타입) CE backward 한 번, full-seed FM backward 한 번 → Q head별 위치별 부호 있는 기여. MLP는
위치별로 저장하지 않음(272 × 12,288 × 층). 몫은 |Q-head 기여|를 밴드 head·층으로 합쳐 **non-vision 위치 합 = 1**로 정규화(vision 포함 정규화도 summary에 있음:
0–21층 I_FM의 vision 몫 0.305, I_CE 0.246). prompt는 클립 간 동일해(history 토큰 id만 다름) 위치별 평균이 가능.

prompt 역할(토큰 수): 채팅/턴 마커 13, 시스템 문장 11, `<|vision_start|>`/`<|vision_end|>` 32, 카메라 레이블("Front left camera: frame 0 …") 80,
history 마커 2, ego-history 토큰 48(= 16 timestep × 3), 지시문 18("output the chain-of-thought reasoning of the driving process, then output the future trajectory."),
`<|cot_start|>` 1; CoC 단어·`<|cot_end|>`·`<|traj_future_start|>`.

## D1. 역할별 몫 (non-vision 정규화; enr = 몫 / 토큰 몫)

| 역할 | 토큰 몫 | I_CE 0–21 (enr) | I_FM 0–21 (enr) | FM−CE [CI] | I_CE 22–34 (enr) | I_FM 22–34 (enr) | FM−CE [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| ego-history 토큰 (48) | 0.217 | 0.164 (0.76) | **0.366 (1.69)** | **+0.202 [+0.177, +0.226]** | 0.024 (0.11) | **0.202 (0.93)** | +0.178 [+0.166, +0.191] |
| vision 마커 (32) | 0.145 | 0.121 (0.83) | **0.197 (1.36)** | +0.077 [+0.060, +0.095] | 0.007 (0.05) | 0.031 (0.22) | +0.024 |
| 카메라 레이블 (80) | 0.362 | 0.180 (0.50) | 0.139 (0.38) | −0.041 | 0.105 (0.29) | 0.131 (0.36) | +0.027 |
| 지시문 (18) | 0.081 | 0.085 (1.05) | 0.061 (0.76) | −0.024 | 0.076 (0.93) | 0.104 (1.28) | +0.028 |
| 시스템 문장 (11) | 0.050 | 0.009 (0.18) | 0.007 (0.14) | −0.002 | 0.003 (0.07) | 0.013 (0.27) | +0.010 |
| 턴 마커 (13) | 0.059 | 0.086 (1.47) | 0.023 (0.39) | −0.064 | 0.131 (2.23) | 0.072 (1.22) | −0.060 |
| history 마커 (2) | 0.009 | 0.009 (1.00) | 0.019 (2.15) | +0.010 | 0.003 (0.30) | 0.024 (2.68) | +0.022 |
| `<\|cot_start\|>` (1) | 0.005 | 0.057 (12.5) | 0.017 (3.9) | −0.039 | 0.154 (34.1) | 0.041 (9.1) | −0.113 |
| CoC 단어 (13) | 0.060 | **0.288 (4.83)** | 0.058 (0.97) | **−0.230 [−0.262, −0.199]** | **0.496 (8.30)** | 0.096 (1.61) | −0.400 |
| `<\|cot_end\|>` (1) | 0.005 | 0 | 0.056 (12.3) | +0.056 | 0 | 0.100 (22.1) | +0.100 |
| `<\|traj_future_start\|>` (1) | 0.005 | 0 | 0.055 (12.2) | +0.055 | 0 | **0.184 (40.8)** | +0.184 |

읽기:
- **I_FM의 텍스트 쪽 질량은 언어가 아니라 숫자와 경계에 있다.** 0–21층에서 ego-history 토큰 37% + `<|vision_end|>` 마커 20% + CoC 경계 토큰 11% = **68%**가 자연어가
  아닌 토큰이고, 자연어(시스템·지시문·카메라 레이블·CoC 단어)는 27%인데 그마저 토큰 몫보다 낮다(enr 0.14–0.97). 22–34층은 `<|traj_future_start|>` 18% + `<|cot_end|>`
  10% + history 20% = 48%.
- **I_CE는 언어에 있다.** CoC 단어(4.8× / 8.3×), `<|cot_start|>`(첫 CoC 단어를 예측하는 자리, 12.5× / 34×), 턴 마커(1.5× / 2.2×; assistant 턴의 `<|im_start|>` 하나가 22–34층
  CE의 12.6%), 지시문(1.0×). ego-history는 0.76× / 0.11×, vision 마커 0.83× / 0.05×.
- 두 점수가 갈리는 방향은 CoC 안에서 본 것과 같다: FM은 **읽히는 자리**(expert가 읽는 경계·요약 토큰과 ego 상태)에, CE는 **예측이 어려운 자리**(다음 토큰이 불확실한 언어 위치)에.
- 사전 예측 D1은 반은 맞았다. 마커/경계 토큰이 큰 것은 맞지만 가장 큰 것은 마커가 아니라 **ego-history 토큰**이고, `<|vision_end|>`는 0–21층에서만 크다(22–34층 0.22×).

## D1'. FM 몫이 큰 prompt 위치 (non-vision 몫, 클립 평균)

0–21층: `-th`(지시문 "chain-of-thought"의 subword, 위치 186) **0.038**(CE 0.058 — 두 손실 모두에 큰 sink성 토큰), 마지막 이미지의 `<|vision_end|>`(131) 0.033, 최신
history 토큰(178/175/172/169/166 = 마지막 timestep들의 첫 슬롯) 0.028/0.024/0.019/0.016/0.014, 다른 `<|vision_end|>`(125/119/102/47) 0.021–0.012, `<|cot_start|>` 0.018(CE 0.057),
`<|traj_history_end|>` 0.016, assistant 턴 `<|im_start|>`(202) 0.011(CE 0.070).
22–34층: `-th` 0.065, `<|im_start|>` 0.050(CE 0.126), `<|cot_start|>` 0.041(CE 0.154), 카메라 레이블의 공백 토큰(72) 0.023(CE 0.072), `<|traj_history_end|>` 0.021(CE 0.002),
`Front`(103) 0.017, 최신 history 토큰 0.010–0.005, `output`(182) 0.008.
주의: 위치 72(카메라 레이블 안의 공백)와 186(`-th`)은 두 손실 모두에서 튀는 **sink성 토큰**으로 보인다(residual gradient도 큼) — 의미가 아니라 활성 크기 문제일 가능성.

## D2. ego-history 48토큰 = 16 timestep × 3 슬롯

| 밴드·점수 | timestep별 몫 (가장 오래된 → 최신) | 마지막 8 / 처음 8 토큰 | Spearman(idx, 몫) | 슬롯 1/2/3 |
|---|---|---:|---:|---|
| 0–21 I_FM | 0.078, 0.049, 0.038, 0.036, 0.038, 0.038, 0.041, 0.045, 0.050, 0.054, 0.062, 0.068, 0.078, 0.087, 0.107, **0.130** | **1.78** | +0.39 | 0.57 / 0.16 / 0.27 |
| 0–21 I_CE | 0.119, 0.069, 0.051, 0.045, 0.045, 0.044, 0.046, 0.047, 0.050, 0.049, 0.052, 0.055, 0.060, 0.067, 0.081, 0.120 | 1.02 | +0.12 | 0.55 / 0.20 / 0.25 |
| 22–34 I_FM | 0.082, 0.058, 0.038, 0.033, 0.033, 0.034, 0.035, 0.039, 0.043, 0.051, 0.061, 0.074, 0.092, 0.102, 0.108, **0.118** | **1.74** | +0.50 | 0.42 / 0.19 / 0.39 |
| 22–34 I_CE | 0.154, 0.057, 0.042, 0.035, 0.036, 0.038, 0.041, 0.043, 0.049, 0.049, 0.054, 0.061, 0.070, 0.075, 0.088, 0.107 | 0.99 | +0.44 | 0.48 / 0.21 / 0.31 |

- **최근성 확인(D2 통과)**: I_FM은 최신 timestep이 가장 오래된 것(첫 토큰 효과로 두 번째로 큼)을 제외한 모든 step보다 크고, 4번째 step부터 단조 증가(0.036 → 0.130, 3.6배);
  마지막 8토큰 / 처음 8토큰 = 1.78 (0–21) / 1.74 (22–34) > 1.5. I_CE는 양끝이 높은 U자(1.02 / 0.99)이고 history 총량 자체가 작다.
- timestep 안의 3슬롯 중 첫 슬롯이 42–57%를 받는다(슬롯의 물리량은 토크나이저 정의를 봐야 함).

## D3. 역할별 순위 일치도 I_FM vs I_CE (Q head; raw / split-half 보정; 클립 단위)

0–21층: sink 0.83/0.95, 턴 마커 0.90/0.93, 시스템 0.95/0.98, vision 마커 0.94/0.99, 카메라 레이블 0.90/0.93, history 마커 0.97/0.99, **ego-history 0.98/1.00**,
지시문 0.95/0.97, `<|cot_start|>` 0.95/1.00, CoC 단어 0.89/0.95, `<|cot_end|>` 0.76/0.96. 22–34층: 0.79–0.93 raw / 0.86–0.98 보정(sink 0.61/0.74 제외).
클립 하나 안에서도 ego-history 0.94/0.82, vision 마커 0.93/0.71, CoC 단어 0.79/0.76.
→ **D3 통과**: prompt의 어느 역할에도 두 번째 CoC형 예외는 없다. 같은 토큰에서는 두 점수가 같은 head를 세우고, 다른 것은 어느 토큰을 얼마나 읽느냐뿐이다.

## residual-gradient norm 몫 (unit과 무관한 토큰 가중; non-vision)

0–21 FM: history 0.282, 카메라 레이블 0.226, vision 마커 0.160, 지시문 0.075, 턴 마커 0.074, CoC 단어 0.057, `<|cot_end|>` 0.045, `<|traj_future_start|>` 0.036.
0–21 CE: 카메라 레이블 0.286, 턴 마커 0.229, CoC 단어 0.189, history 0.097, vision 마커 0.075, 지시문 0.074. 22–34 FM: 카메라 레이블 0.221, history 0.217, 턴 마커 0.163, 지시문 0.109.

## 한 줄 결론과 논문 문장

**I_FM이 텍스트 쪽에서 읽는 것은 ego-history 숫자 토큰(최근 것일수록)과 각 입력 블록을 닫는 경계 토큰(`<|vision_end|>`, `<|traj_history_end|>`, `<|cot_end|>`, `<|traj_future_start|>`)이고,
자연어 문장(시스템·지시문·CoC 단어)은 토큰 몫 이하로 본다; I_CE는 정확히 그 반대다.** 그리고 어느 역할에서도 같은 토큰의 head 순위는 일치한다.

```latex
Outside the vision tokens, the two scores read different tokens. In layers 0--21 the FM score
puts $37\%$ of its non-vision mass on the 48 ego-history tokens ($1.7\times$ their token share,
rising monotonically toward the most recent timestep, last-to-first ratio $1.8$), $20\%$ on the
$\langle$vision\_end$\rangle$ markers that close each image ($1.4\times$) and $11\%$ on the two
tokens that close the reasoning ($12\times$), while the natural-language tokens (system prompt,
instruction, camera labels, CoC words) receive less than their token share. The CE score reads
the language: $29\%$ on the CoC words ($4.8\times$), $6\%$ on $\langle$cot\_start$\rangle$
($12.5\times$) and $9\%$ on the turn markers ($1.5\times$), and only $0.8\times$ on the ego
history. Within every role, however, the two scores rank the heads alike (split-half corrected
$\rho \ge 0.93$ in layers 0--21 and $\ge 0.86$ in 22--34, ego history $1.00$ / $0.97$): the
objectives differ in which tokens they read, not in which heads they need at a token.
```
