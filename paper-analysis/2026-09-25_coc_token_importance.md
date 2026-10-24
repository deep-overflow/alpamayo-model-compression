# CoC 토큰 단위 FM/CE importance — 단어 범주별 분석 (2026-09-25, 논문 에이전트 요청 1–5 + 6 제안)

출력 `outputs/coc_tokimp_v1/` (`summary.txt`, `metrics.json`에 단어별 표·범주 규칙 포함, `plots/category_shares.png`, `plots/category_enrichment.png`),
스크립트 `experiments/head_analysis/analyze_coc_token_importance.py` (브랜치 `worktree-dual-saves-trunk`). 데이터는 Run A
(`outputs/coc_posanat_v1_s0-3`, calib_100 중 99클립, dense, 자체 생성 CoC): CoC 위치 p마다 Q head u의 부호 있는 기여 o_{u,p}·∂L/∂o_{u,p}
(CE = CoC NLL, FM = 출하 expert pass 10스텝 합). 그림(분석 스타일) `figures/supplementary/fig11_coc_token_category_shares.png`,
`fig11b_coc_token_category_enrichment.png`.

## 정의

- **토큰 중요도** T^L_{c,p} = Σ_{u∈밴드 head} |o_{u,p}·∂L_c/∂o_{u,p}|, 클립마다 CoC 위치 합 = 1. 요청대로 **special 토큰 포함** 정규화가 기본이고,
  FM 질량의 63–75%가 두 special 토큰에 있어 단어 간 비교가 묻히므로 **단어만** 정규화도 병기한다.
- **subword 병합**: 공백/개행으로 시작하는 토큰이 새 단어(`Ad`+`apt` → adapt, `cut`+`-in` → cut-in). special 토큰은 별도 단위.
- **범주 규칙 (레포에 "v2 추출기"는 없다 — 가장 가까운 것은 `coc_action_consistency.py`의 `HEAD2BUCKET` 머리 어절 매핑. 아래 규칙은 이번에 정의했고
  범주별 단어 목록을 summary에 출력해 감사·재매핑이 가능하며 단어별 표가 `metrics.json`에 있다):**
  - decision/action = 행동 절: 첫 단어부터 첫 접속어(since/because/due/for/as/at/through/with/after/but/then/and/or/while/when/until) 또는
    결정 동사가 뒤따르지 않는 전치사(to/from/behind/in/into/on/following…) 앞까지, 기능어 제외. 단 nudge/turn/change 뒤의 "to the right/left"는 절에 포함.
    (n = 215: keep 51, distance 40, speed 15, lane 13, adapt 12, stop 10, nudge 7, left 7, slow 6, down 6, right 6, accelerate 5, proceed 4, yield 3 …; 새어 들어간 오류 약 8개)
  - scene(entity + state) = 절 이후의 나머지 내용어 (n = 358: lane 67, vehicle 47, lead 41, traffic 17, clear 14, light 13, curve 12, bends 12, red 8, intersection 8, green 7 …)
  - spatial (n = 152: ahead 75, directly 30, right 29, into 5, from 4, left 4, side 3 …)
  - other = 기능어·접속어·구두점 (n = 536: the 124, since 78, to 76, is 58, it 44, our 43, in 32, for 20, due 15 …)
  - cot_end = `<|cot_end|>`, traj_start = `<|traj_future_start|>` (99클립 전부 이 둘로 끝남)
  - 민감도: 엄격 정의 = 앞 두 단어(`HEAD2BUCKET` 관례)도 보고.
- enrichment = 중요도 몫 / 토큰 몫(둘 다 클립 평균의 비). 주 검정 = decision의 FM enrichment − CE enrichment, 클립 페어드 부트스트랩 95% CI(3,000회),
  몫 차이의 Wilcoxon p 병기.

## 1·3. 범주별 몫과 enrichment

**정규화: CoC 위치 전체(단어 + special 토큰)** — 요청 사양

| 밴드 | 범주 | 토큰 몫 | CE 몫 | CE enr. | FM 몫 | FM enr. | FM − CE enr. [CI] | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0–21 | decision | 0.151 | 0.213 | 1.42 | 0.075 | 0.50 | **−0.92 [−1.10, −0.76]** | 7e−17 |
| 0–21 | scene | 0.242 | 0.299 | 1.24 | 0.098 | 0.41 | −0.83 [−0.98, −0.70] | 2e−15 |
| 0–21 | spatial | 0.102 | 0.072 | 0.71 | 0.029 | 0.29 | −0.42 [−0.53, −0.32] | 1e−8 |
| 0–21 | other | 0.360 | 0.416 | 1.15 | 0.084 | 0.23 | −0.92 [−0.99, −0.85] | 6e−18 |
| 0–21 | cot_end | 0.073 | 0.000 | 0.00 | 0.330 | 4.51 | +4.51 [+4.10, +4.95] | 6e−18 |
| 0–21 | traj_start | 0.073 | 0.000 | 0.00 | 0.385 | 5.27 | +5.27 [+4.80, +5.75] | 6e−18 |
| 22–34 | decision | 0.151 | 0.182 | 1.21 | 0.066 | 0.44 | **−0.77 [−1.01, −0.56]** | 2e−10 |
| 22–34 | scene | 0.242 | 0.279 | 1.16 | 0.075 | 0.31 | −0.85 [−1.00, −0.70] | 2e−16 |
| 22–34 | spatial | 0.102 | 0.069 | 0.68 | 0.027 | 0.26 | −0.42 [−0.54, −0.31] | 4e−8 |
| 22–34 | other | 0.360 | 0.469 | 1.30 | 0.088 | 0.24 | −1.06 [−1.18, −0.95] | 7e−18 |
| 22–34 | cot_end | 0.073 | 0.000 | 0.00 | 0.258 | 3.53 | +3.53 [+3.16, +3.93] | 6e−18 |
| 22–34 | traj_start | 0.073 | 0.000 | 0.00 | 0.487 | 6.67 | +6.67 [+6.16, +7.15] | 6e−18 |

**정규화: 단어만**

| 밴드 | 범주 | 토큰 몫 | CE 몫 | CE enr. | FM 몫 | FM enr. | FM − CE enr. [CI] | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0–21 | decision | 0.177 | 0.213 | 1.20 | 0.271 | 1.53 | **+0.32 [+0.16, +0.47]** | 1e−6 |
| 0–21 | scene | 0.284 | 0.299 | 1.05 | 0.309 | 1.09 | +0.03 [−0.08, +0.15] | 0.70 |
| 0–21 | spatial | 0.118 | 0.072 | 0.61 | 0.115 | 0.97 | +0.36 [+0.25, +0.46] | 8e−8 |
| 0–21 | other | 0.420 | 0.416 | 0.99 | 0.306 | 0.73 | −0.26 [−0.32, −0.20] | 2e−10 |
| 22–34 | decision | 0.177 | 0.182 | 1.03 | 0.266 | 1.50 | **+0.47 [+0.28, +0.65]** | 4e−7 |
| 22–34 | scene | 0.284 | 0.279 | 0.98 | 0.278 | 0.98 | −0.01 [−0.13, +0.12] | 0.48 |
| 22–34 | spatial | 0.118 | 0.069 | 0.59 | 0.108 | 0.91 | +0.33 [+0.22, +0.43] | 8e−7 |
| 22–34 | other | 0.420 | 0.469 | 1.12 | 0.348 | 0.83 | −0.29 [−0.38, −0.19] | 5e−7 |

- 엄격 정의(앞 두 단어, 단어만 정규화): 토큰 몫 0.154; 0–21 CE enr. 1.25 / FM 1.64, **+0.39 [+0.21, +0.56]**; 22–34 CE 1.08 / FM 1.61, **+0.53 [+0.32, +0.74]**.
- 단어별(발생당 평균 몫, 0–21 FM 기준 상위): `<|traj_future_start|>` 0.385, `<|cot_end|>` 0.330, accelerate 0.106, nudge 0.073, down 0.062, stop 0.061, light 0.051, stopped 0.048, red 0.048 …;
  CE 상위: accelerate 0.333, light 0.182, clear 0.180, curve 0.166, stop 0.161, sign 0.153, red 0.138, down 0.128, is 0.121, keep 0.112.

## 4. 위치 통제

단어를 상대 위치 4구간으로 나누고 구간 안에서 정규화. 결정 단어는 **Q1에 몰려 있다**(Q1 단어의 61%; Q2/Q3/Q4에는 9 / 5 / 2개뿐이라 검정 불가).

| 밴드 | 구간 | 단어 수 | decision 토큰 몫 | CE dec 몫 (enr.) | FM dec 몫 (enr.) | FM − CE enr. [CI] |
|---|---|---:|---:|---:|---:|---:|
| 0–21 | Q1 | 350 | 0.608 | 0.668 (1.10) | 0.746 (1.23) | **+0.13 [+0.07, +0.18]** |
| 22–34 | Q1 | 350 | 0.608 | 0.662 (1.09) | 0.760 (1.25) | **+0.16 [+0.09, +0.23]** |

- special·마지막 토큰 별도(전체 정규화, 클립 평균): `<|cot_end|>` CE 0 / FM 0.330 | 0.258; `<|traj_future_start|>` CE 0 / FM 0.385 | 0.487;
  **마지막 단어**는 CE 0.066 | 0.089, FM 0.035 | 0.026 — FM에 특별하지 않다. 특별한 것은 두 special 토큰이다.

## 5. Unit 연결 (층마다 rank(I_traj@CoC) − rank(I_CoC@CoC) 상위 6 = FM 쪽, 하위 6 = CE 쪽; 그룹 자신의 CoC 질량 중 몫)

| 밴드 | 손실 | 그룹 | decision (전체) [CI] | special | scene | decision (단어만) [CI] | scene | spatial | other |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0–21 | FM | FM 쪽 | 0.038 [0.033, 0.043] | **0.857** | 0.054 | **0.290 [0.271, 0.312]** | 0.305 | 0.104 | 0.301 |
| 0–21 | FM | CE 쪽 | 0.120 [0.109, 0.133] | 0.455 | 0.189 | 0.223 [0.207, 0.243] | 0.335 | 0.131 | 0.311 |
| 0–21 | CE | FM 쪽 | 0.226 [0.194, 0.260] | 0 | 0.297 | 0.226 | 0.297 | 0.067 | 0.410 |
| 0–21 | CE | CE 쪽 | 0.173 [0.146, 0.204] | 0 | 0.323 | 0.173 | 0.323 | 0.080 | 0.424 |
| 22–34 | FM | FM 쪽 | 0.046 [0.040, 0.053] | **0.864** | 0.035 | **0.342 [0.318, 0.367]** | 0.238 | 0.095 | 0.325 |
| 22–34 | FM | CE 쪽 | 0.142 [0.128, 0.158] | 0.380 | 0.196 | 0.231 [0.211, 0.252] | 0.314 | 0.116 | 0.339 |
| 22–34 | CE | FM 쪽 | 0.221 [0.187, 0.258] | 0 | 0.259 | 0.221 | 0.259 | 0.061 | 0.459 |
| 22–34 | CE | CE 쪽 | 0.170 [0.136, 0.208] | 0 | 0.313 | 0.170 | 0.313 | 0.076 | 0.441 |

- I_FM의 decision 몫, FM 쪽 − CE 쪽: 전체 정규화 **−0.083 [−0.092, −0.074]** | **−0.096 [−0.108, −0.085]**; 단어만 **+0.067 [+0.058, +0.077]** | **+0.111 [+0.094, +0.127]** (Wilcoxon p ≤ 3e−16).
- I_CE의 decision 몫도 FM 쪽 head가 더 크다: +0.053 | +0.051 (p ≤ 8e−12); CE 쪽 head는 other(기능어)와 scene에 더 무겁다.

## 읽기

1. **요청한 가설("I_FM이 결정 단어에 몰린다")은 요청한 정규화에서는 반대로 나온다.** FM의 CoC 질량은 결정 단어가 아니라 두 special 토큰에 몰린다
   (72% | 75%; enrichment 4.5–6.7배), CE는 거기서 구조적으로 0. 그래서 decision의 FM enrichment(0.50 | 0.44)가 CE(1.42 | 1.21)보다 낮다(−0.92 | −0.77, CI 0 배제).
2. **단어 사이에서는 성립한다.** FM은 decision을 1.5배, spatial을 1.0배로 보고 CE는 decision 1.0–1.2배, spatial 0.6배, other 1.0–1.1배 — 차이 +0.32 | +0.47(엄격 정의 +0.39 | +0.53).
   scene은 같다. 즉 FM이 CE보다 상대적으로 **행동·공간어에 무겁고 기능어에 가볍다**.
3. **위치 통제 후에는 절반 크기로 남는다.** 결정 단어는 첫 구간에 몰려 있어(Q1의 61%) 위치와 범주가 거의 같은 것이고, Q1 안에서 비교하면 +0.13 | +0.16 —
   단어 전체의 +0.32 | +0.47 중 절반은 "앞쪽 위치", 절반은 "결정 단어"다. Q2–Q4는 결정 단어가 2–9개라 검정할 수 없다.
4. **Unit 연결.** FM 쪽 head는 I_FM의 86%를 special 토큰에 쓴다(CE 쪽 head는 38–46%) — 이 head들이 expert가 읽는 경계 토큰의 writer라는 Run A/B 결론과 같은 사실.
   단어만 보면 FM 쪽 head가 결정 단어에 더 무겁다(0.29–0.34 vs 0.22–0.23). 흥미롭게 I_CE에서도 FM 쪽 head가 결정 단어에 더 무겁고(0.22 vs 0.17)
   CE 쪽 head는 기능어·scene에 무겁다 — CE 쪽 head가 "다음 토큰 이어 쓰기" 기계라는 census 결과와 일치.
5. 논문 문장에 쓸 수 있는 요약: *Among the words of the reasoning, the FM score weights the action clause 1.5× its token share against 1.0–1.2× for the CE score
   (paired difference +0.32 / +0.47, CI excluding 0), but the bulk of the FM score at CoC positions (72–75%) sits on the two boundary tokens, where the CE
   score is zero by construction; within the first quarter of the sentence, where 61% of the action words are, the preference survives at half its size (+0.13 / +0.16).*

## 6. (선택) expert attention에서 결정 단어 CoC key 마스킹 — 제안 (미실행, 승인 시 진행)

- 설계: Run A와 같은 클립·rollout에서 cache를 한 번 만들고 expert의 `prefix_mask`로 (a) 결정 단어 위치, (b) 같은 개수의 무작위 CoC 단어 위치(클립별 3회),
  (c) 두 special 토큰, (d) 마스크 없음을 비교. 읽는 값은 ΔL_FM(10스텝, 같은 노이즈)과 minADE(K = 8). `pl.expert_fm_loss`는 내부에서 `prefix_mask = ones`를
  만들므로 `prefix_mask` 인자를 받도록 3줄 수정이 먼저 필요하다(`analysis_lib.denoise_with_cache`는 이미 받음).
- 예측(Run A·B에서): (c) ≫ (a), (b); (a) vs (b)는 단어 enrichment 1.5배가 인과로 옮겨간다면 (a) > (b), 작을 것.
- 비용: 클립당 rollout + forward + 6 조건 × FM 10스텝 + minADE ≈ 40 s → 99클립을 Ada 4장에 나누면 약 20분.

## 남은 주의점

- 범주 규칙은 이번에 정의한 것이다. 문장이 강하게 템플릿화되어 있어(머리 어절 27종, 어휘 130개) 규칙 오류는 결정 215개 중 한 자릿수지만, 논문에 쓰려면
  `metrics.json`의 단어별 표로 원하는 매핑을 다시 집계하는 편이 안전하다.
- "결정 단어"와 "문장 앞부분"은 이 데이터에서 거의 같은 것이라, 결정 단어의 효과를 위치와 분리한 추정치는 Q1 내부 비교(+0.13 | +0.16)뿐이다.

## 추가 (같은 날): 결정 절 / 원인 절로 나눠 본 FM의 단어 (`summary.txt` clause view, 단어만 정규화)

절 = 첫 단어부터 첫 접속어(since/for/due/because …) 앞까지가 **결정 절**, 접속어, 그 뒤가 **원인 절**, special 토큰 별도.

| 밴드 | 절 | 토큰 몫 | CE 몫 (enr.) | FM 몫 (enr.) | FM − CE enr. [CI] |
|---|---|---:|---:|---:|---:|
| 0–21 | 결정 절 | 0.194 | 0.230 (1.18) | 0.287 (1.48) | **+0.30 [+0.14, +0.46]** |
| 0–21 | 접속어 | 0.085 | 0.071 (0.84) | 0.106 (1.24) | +0.40 [+0.28, +0.51] |
| 0–21 | 원인 절 | 0.721 | 0.699 (0.97) | 0.607 (0.84) | −0.13 [−0.17, −0.09] |
| 22–34 | 결정 절 | 0.194 | 0.202 (1.04) | 0.282 (1.45) | **+0.41 [+0.23, +0.59]** |
| 22–34 | 접속어 | 0.085 | 0.043 (0.50) | 0.085 (0.99) | +0.49 [+0.40, +0.57] |
| 22–34 | 원인 절 | 0.721 | 0.756 (1.05) | 0.634 (0.88) | −0.17 [−0.21, −0.12] |

special 토큰 포함 정규화에서는 FM 질량이 결정 절 0.07–0.08 / 접속어 0.02–0.03 / 원인 절 0.16–0.18 / special 0.71–0.75 (0–21 | 22–34).
즉 **단어 안에서는** FM이 결정 절을 1.5배로 보고 원인 절은 0.84–0.88배로 보지만, 원인 절이 단어의 72%라 FM 단어 질량의 61–63%는 여전히 원인 절에 있다. 접속어(since/for/due)는 FM이 1.0–1.2배, CE는 0.5–0.8배로 본다.

**FM이 무겁게 보는 결정 절 단어** (발생당 FM 몫, n ≥ 3; 괄호는 FM/CE 비; 0–21 | 22–34)

| 단어 | n | FM/발생 0–21 (FM/CE) | FM/발생 22–34 (FM/CE) | 평균 클립 기여 FM 0–21 |
|---|---:|---:|---:|---:|
| accelerate | 5 | 0.309 (0.93) | 0.352 (1.46) | 0.016 |
| stop | 10 | 0.306 (1.41) | 0.260 (1.15) | 0.031 |
| nudge | 7 | 0.214 (2.53) | 0.199 (2.55) | 0.015 |
| yield | 3 | 0.177 (1.45) | 0.169 (2.12) | 0.005 |
| keep | 51 | 0.160 (1.41) | 0.155 (1.42) | **0.082** |
| turn | 3 | 0.154 (1.43) | 0.140 (1.32) | 0.005 |
| adapt | 12 | 0.152 (3.10) | 0.168 (8.43) | 0.019 |
| down | 6 | 0.133 (1.03) | 0.117 (0.76) | 0.008 |
| lane (keep lane) | 13 | 0.118 (1.29) | 0.119 (2.11) | 0.016 |
| speed | 15 | 0.102 (0.98) | 0.093 (0.83) | 0.015 |
| left / right (turn·nudge) | 7 / 6 | 0.101 / 0.120 (1.0 / 1.2) | 0.097 / 0.098 (1.9 / 1.0) | 0.007 / 0.007 |
| slow | 6 | 0.054 (1.51) | 0.100 (4.32) | 0.003 |
| distance | 40 | 0.050 (0.84) | 0.048 (1.34) | 0.020 |
| proceed | 4 | 0.058 (0.65) | 0.061 (0.79) | 0.002 |

읽기: **동사(행동 자체)**가 무겁다 — accelerate, stop, nudge, yield, keep, turn, adapt. 동사의 목적어·부사(distance, speed, down, proceed)는 FM에 가볍다.
CE 대비 FM이 특히 더 보는 것은 nudge(2.5×), adapt(3–8×), slow(1.5–4×), keep·stop·yield·turn(1.3–2×) — 종·횡 방향의 **속도·차선 조정 동사**.

**FM이 무겁게 보는 원인 절 단어** (발생당 FM 몫, n ≥ 3; 0–21 | 22–34)

| 단어 | n | FM/발생 0–21 (FM/CE) | FM/발생 22–34 (FM/CE) | 평균 클립 기여 FM 0–21 |
|---|---:|---:|---:|---:|
| slowly | 3 | 0.259 (5.0) | 0.078 (2.1) | 0.008 |
| light | 13 | 0.143 (0.78) | 0.114 (0.53) | 0.019 |
| cut-in | 4 | 0.139 (0.83) | 0.140 (0.97) | 0.006 |
| curve | 12 | 0.137 (0.82) | 0.084 (1.18) | 0.017 |
| stopped | 6 | 0.135 (1.68) | — | 0.008 |
| red | 8 | 0.125 (0.91) | 0.100 (0.99) | 0.010 |
| clear | 14 | 0.115 (0.64) | 0.094 (0.38) | 0.016 |
| sign | 6 | 0.109 (0.72) | 0.118 (0.55) | 0.007 |
| green | 7 | 0.107 (8.2) | — | 0.008 |
| intersection | 8 | 0.106 (1.85) | 0.086 (3.34) | 0.009 |
| moving | 3 | 0.105 (4.8) | — | 0.003 |
| since (접속어) | 64 | 0.099 (1.08) | 0.087 (1.17) | **0.064** |
| ahead | 75 | 0.096 (1.93) | 0.090 (1.45) | **0.072** |
| vehicle | 47 | 0.094 (1.17) | 0.082 (1.16) | 0.044 |
| lane | 67 | 0.087 (1.05) | 0.088 (1.02) | 0.059 |
| merging | 3 | 0.084 (6.1) | 0.079 (5.2) | 0.003 |
| blocking | 4 | 0.084 (1.05) | 0.081 (1.88) | 0.003 |
| straight | 5 | — | 0.091 (4.1) | — |

읽기: 원인 절에서 FM이 CE보다 상대적으로 더 보는 것은 **상대 차량·신호의 동적 상태와 공간 관계**(slowly 5×, moving 4.8×, merging 6×, green 8×,
stopped 1.7×, intersection 1.9–3.3×, ahead 1.5–1.9×, blocking 1.9× late, straight 4×), CE가 더 보는 것은 **정적 장면 명사**(clear 0.4–0.6×, light 0.5–0.8×,
sign 0.6–0.7×, curve 0.8×). 클립 평균 기여로는 원인 절의 ahead(0.072)·since(0.064)·lane(0.059)·vehicle(0.044)이 결정 절의 keep(0.082)과 같은 급 —
즉 FM이 단어에서 읽는 것은 "keep … vehicle … lane ahead"의 골격이다. 주의: n ≤ 7인 단어의 비율은 불안정하고, 문장 끝 단어(green, ahead, slowly)는
CE가 종료 토큰 예측에 붙어 작아지는 위치 효과가 섞일 수 있다(마지막 단어의 CE 몫은 0.066–0.089로 평균 단어와 비슷해 크지는 않음).

## 추가 2 (2026-09-26): ego-history를 많이 읽는 CoC 질의 위치 (`run_coc_census.py --per-query`, `outputs/coc_census_pos_v1/`)

census를 질의 위치별로 다시 측정(같은 calib 99클립·같은 rollout seed, Ada 4–7, shard당 5분). 단위는 위와 같은 단어 병합·범주·절. 값은 **CoC 질의 하나가 ego-history 키에 두는 attention 질량**(밴드 head 평균; FM 쪽/CE 쪽 = 층당 rank 차 상·하위 6 head). 그림 `figures/supplementary/fig12_coc_query_ego_history_by_position.png`.

| 질의 위치 | 0–21 전체 | 0–21 FM 쪽 | 0–21 CE 쪽 | 22–34 전체 | 22–34 FM 쪽 | 22–34 CE 쪽 |
|---|---:|---:|---:|---:|---:|---:|
| decision 단어 | 0.033 | 0.040 | 0.019 | 0.016 | 0.024 | 0.010 |
| scene 단어 | 0.028 | 0.033 | 0.019 | 0.015 | 0.020 | 0.011 |
| spatial 단어 | 0.027 | 0.030 | 0.018 | 0.017 | 0.022 | 0.014 |
| other | 0.027 | 0.029 | 0.018 | 0.014 | 0.016 | 0.010 |
| `<\|cot_end\|>` | 0.042 | **0.090** | 0.018 | 0.025 | **0.046** | 0.011 |
| `<\|traj_future_start\|>` | **0.079** | **0.193** | 0.027 | **0.065** | **0.155** | 0.014 |

- **ego-history를 가장 많이 읽는 CoC 질의는 두 경계 토큰**, 특히 `<|traj_future_start|>`다: 전체 head로 단어의 2.4배(0–21) / 4배(22–34), FM 쪽 head로는 단어의 5배(0.193 vs 0.029–0.040) / 7배.
  CoC 질의 전체가 ego-history에 두는 attention의 몫으로 보면 두 special 토큰(토큰의 14.6%)이 전체 head 27% / 34%, FM 쪽 head **43% / 47%**를 차지한다(`<|traj_future_start|>`만 17.6% / 25.0%, FM 쪽 29.1% / 36.2%).
- **CE 쪽 head는 어디서도 ego-history를 읽지 않는다**(0.010–0.027, 위치 무관). special 토큰에서 CE 쪽 head의 attention은 sink로 간다(0.41–0.42; 0–21) — 거기서 읽을 텍스트가 없기 때문.
  FM 쪽 − CE 쪽 차이: special 토큰 **+0.119 [+0.116, +0.122]** (0–21) / **+0.088 [+0.084, +0.092]** (22–34), 단어 +0.014 / +0.008.
- **단어 사이에서는 편평하다**(0.027–0.033; 범주별 몫/토큰 몫 비 0.8–1.0, 어느 범주도 농축되지 않음). 그나마 높은 단어는 **속도·조향 동사**: 0–21 FM 쪽 head 기준 down 0.055, turn 0.054, slow 0.051, adapt 0.046, nudge 0.046, yield 0.045, speed 0.044, accelerate 0.041, stop 0.040, keep 0.039 (scene 단어는 curve 0.044, vehicle 0.037, lane 0.034).
  22–34에서도 down 0.033, speed 0.029, slow 0.029.
- 대조로 **이전 CoC 읽기**는 문장을 따라 누적된다(0–21 전체 head: 첫 사분위 0.055 → 마지막 0.205; CE 쪽 head 0.10 → 0.31)이고 vision은 위치와 무관(22–34 모든 질의 0.49–0.63).

**읽기.** Run A(FM gradient의 63–75%가 두 경계 토큰), Run B(FM 쪽 head를 CoC 자리에서 끄면 행동만 손상), census(FM 쪽 head는 ego-history를 읽음)를 이 결과가 하나로 잇는다:
**추론이 끝나는 자리 `<|cot_end|>`·`<|traj_future_start|>`에서 FM 쪽 head가 ego-history를 끌어와 요약 표현을 쓰고, expert가 그것을 읽는다.** 단어 위치에서는 ego-history 읽기가 작고 편평하며 결정 동사에서만 조금 높다.
텍스트 이어 쓰기 head(CE 쪽)는 경계 토큰에서 할 일이 없어 sink를 본다.

**논문 문장 (English).**

```latex
The ego-history reading sits where the expert reads. At the CoC queries, attention
to the ego-history keys is flat across the words ($0.03$ per query in layers 0--21,
with no word category enriched) and rises at the two boundary tokens: $0.04$ at
$\langle$cot\_end$\rangle$ and $0.08$ at $\langle$traj\_start$\rangle$ for all heads,
$0.09$ and $0.19$ for the FM-favored heads against $0.02$--$0.03$ for the CE-favored
ones, whose attention there goes to the sink. The two boundary tokens thus carry
$27\%$ of the CoC queries' ego-history attention ($43\%$ for the FM-favored heads)
on $15\%$ of the tokens; among the words the ego history is read most at the speed
and steering verbs (down, turn, slow, adapt, nudge, yield; $0.04$--$0.05$). Layers
22--34 show the same profile at lower mass ($0.155$ vs.\ $0.014$ at
$\langle$traj\_start$\rangle$ for the two head groups).
```

**그림 (2026-09-26 추가).** 단어 수준 논문 스타일 그림 `figures/supplementary/fig11c_coc_words.{png,pdf,svg}` (`scripts/paper_coc_words.py`): 결정 절 상위 8 단어·원인 절 상위 10 단어(n ≥ 3)의 발생당 I_traj/I_CoC 몫, 두 밴드.
