# 캐시 보존 Taylor 기준 (`cachedual`) — dual의 I_traj를 "expert가 읽는 캐시를 덜 움직이는가"로 바꾼다

날짜: 2026-08-30. 세션: dual kv protect. 브랜치: `worktree-cache-jlens-criterion` (워크트리; `dualr-weighted` 위에 병합). 상태: **실행 중** (2026-08-30 22:30 KST 시작).

## 왜

- VLM 절단의 개루프 궤적 비용은 거의 전부 KV 캐시 이동으로 들어온다(프루닝 캐시 + dense expert = +0.07
  ≈ dual +0.058, `2026-08-28_cache-shift`), 손상은 이동 크기에 비례한다(Stage C ρ +0.89,
  `2026-08-29_cache-use-map`), 그 이동은 상류 절단의 누적이라 refit으로는 첫 층부터 전부 고쳐야 한다
  (`2026-08-29_cache-targeted-reconstruction`). 재구성은 캐시·궤적을 지키지만 reasoning을 잃는다
  (dualr_rep LingoQA 52.2 vs dual 68.8, `2026-08-30_dualr-weighted-hessian`).
- 남은 지렛대는 **선택**이다: 어떤 유닛을 자를지를 "캐시를 얼마나 움직이는가"로 고르면 refit 없이,
  reasoning을 잃지 않고(CE 절반은 dual 그대로) 이동을 줄일 수 있는가.
- dual의 `I_traj`는 이미 캐시 경유 신호(궤적 손실은 expert → 캐시 → VLM 게이트로만 흐른다)지만 클립당
  스칼라 하나라 분산이 크고 GT 궤적이 필요하며 스텝 집계 문제를 안고 있었다. 캐시 보존 신호는
  (층 × 그룹 × 위치)의 밀집 신호이고 라벨이 없다 — J-lens가 CoC-NLL Taylor를 대신했던 것과 같은 논리.

## 기준의 정의 — 1차 Taylor는 0이므로 Jacobian 노름을 쓴다

캐시 보존 손실 ‖cache(g) − cache(1)‖²는 g=1에서 값·기울기가 모두 0이라 게이트 Taylor |∂L/∂g|는 전부
0이다. 필요한 양은 유닛 u를 껐을 때 캐시가 움직이는 크기, 즉

  I_cache(u) = Σ_{l,g} S[l,g] · ‖∂K_{l,g}/∂g_u‖² + ‖∂V_{l,g}/∂g_u‖²   (기대값, 캘리브레이션 클립·위치 평균)

여기서 S[l,g]는 cache-use map Stage C의 **단위 이동당 민감도**(`outputs/cacheuse_v1/maps_swap.npz:sensitivity`,
층 0은 0). expert 어텐션 가중은 무효로 판명됐으므로 쓰지 않는다(`dualr_e`). 유닛별 backward는
36×(32+12288)회라 불가능 → **확률 프로브**: 층·그룹별 무작위 cotangent r_{l,g} ~ N(0, S[l,g]/(T·D)·I)를 K·V에
걸고 ⟨Σ r, cache⟩를 한 번 backward하면 게이트 기울기의 제곱 (Σ_l ∂⟨r_l, cache_l⟩/∂g_u)²의 기대값이
I_cache(u)다(독립 프로브라 층 간 교차항의 기대값 0). 클립당 P=16 프로브, 100클립 = 1,600 backward.

구현은 `run_importance.py`의 I_traj 경로 그대로다 — 그쪽은 expert 기울기를 캐시 cotangent로 써서
`prune_lib.vlm_backward_from_cache(cache_t, seed_grads)`로 VLM 게이트(`UnitGates`)에 한 번 backward한다;
여기서는 seed_grads를 프로브로 바꾸고 기울기를 **제곱해서** 누적한다(1차가 아니라 2차 양). 위치 범위는
prefill 전체(expert가 전 구간을 읽는다 — cache-use map), 프로브는 K와 V에 같은 무게. 층 l의 게이트는
층 > l의 캐시에만 닿으므로(K/V는 층 입력의 사영) **마지막 층의 I_cache는 정확히 0**이고 거기서는
I_CoC만 순위를 정한다.

## 설계 — dual과 one-factor

| arm | 기준 | 비고 |
|---|---|---|
| `dual_u40_v2` (기존) | max(rank I_traj, rank I_CoC) | 기준 |
| **`cachedual_u40_v2`** | **max(rank I_cache, rank I_CoC)** | I_traj → I_cache만 교체. 본명 |
| `cacheonly_u40_v2` | I_cache 단독 | CE 절반의 기여 분리 (traj/coc 단독 대조군과 같은 역할) |
| `traj_u40_v2` (기존) | I_traj 단독 | I_cache의 직접 대조군 (둘 다 캐시 경유) |

예산·할당·expert·KV 전부 u40_v2 가족과 동일(uniform 0.3985632694, −2,657,452,032, 순수 선택 → `--no-state`).
합치는 연산은 max(rank)로 고정 — 합/곱은 operator ablation에서 졌다(dualsum test +0.069, dualprod +0.226
vs dual +0.052).

## 사전 등록 게이트

- **G0 신호 검사** (기준 계산 직후, 빌드 전):
  (i) split-half 안정성(짝/홀 클립) Spearman ≥ 0.95, kept-set 겹침 ≥ 0.95 (jlens_coc32 수준);
  (ii) **kept-set 겹침 vs traj_u40_v2 / dual_u40_v2 < 0.95** — 그 이상 겹치면 I_traj와 같은 랭킹이라 실험이
  무의미, 여기서 접는다(30분이면 안다);
  (iii) I_cache의 층 프로파일 보고(깊은 층 캐시를 흔드는 상류 유닛이 어디에 있나).
- **G1 캐시 프록시** (`run_cacheproxy.py`, val500 앞 200클립, dual과 페어드): 민감도 가중 이동 비 vs dual,
  A10−A00. 판정선: 이동 비 CI 상한 < 0.9(≥10% 감소)이고 캐시 비용이 dual보다 유의하게 작음.
- **G2 val500**: Δ vs 무압축 중앙값이 dual(+0.058)보다 유의하게 작음(paired arm−dual CI 상한 < 0).
  손상 ∝ 이동이므로 "이동 X% 감소 → 비용 ~X% 감소"가 예상선.
- **G3 LingoQA**: cachedual ≥ dual(68.8) CI 안(paired 차 CI가 0 포함). CE 절반을 그대로 두었으니 떨어지면 기각.
- **G4 폐루프**(150씬, G1–G3 통과 시): score ≥ dual, 충돌 ≤ dual.

## 파일

| 파일 | 변경 |
|---|---|
| `experiments/head_analysis/run_cache_jlens.py` (신규) | 프로브 backward로 I_cache(vlm_q, vlm_mlp) 계산; `outputs/cachejlens_v1/importance.npz`에 `cache_vlm_q`, `cache_vlm_mlp` + 짝/홀 반쪽 + 프로브 분산; `--probes 16 --sensitivity cacheuse_v1/maps_swap.npz` |
| `experiments/head_analysis/analyze_cachejlens.py` (신규) | G0: 안정성, 겹침(traj/dual/coc), 층 프로파일, 그림; 여러 run의 프로브 가중 병합 |
| `experiments/head_analysis/make_slim.py` | `uni` 분기 `half()`에 `cache` 추가(`--cache-importance`), `cachedual`=("cache","coc"), `cacheonly`. **공유 파일** |
| `experiments/evaluation/paper_numbers.py` | `cachedual`, `cacheonly` arm. **공유 파일** |
| 보고서 템플릿 `cachejlens_report_template.html` + `reports/evaluation/2026-08-30_cache-jlens-criterion.html` | |

## 비용·순서

1. I_cache 계산: calib_100 × 16 프로브, GPU 1장 ≈ 25분(클립당 14 s).
2. G0 (분석만, 분) → 접거나 진행.
3. 빌드 2 arm(`--no-state`, 각 2분) → G1 프록시 2 arm(Ada 6·7, 30분) ‖ val500 2 arm(공유 큐에 `append`, 워커 6·7) → LingoQA 2 arm(13분씩).
4. 개루프까지 **약 2.5시간**. G4는 별도 8시간.

## 솔직한 기대치
- 선택만으로 줄일 수 있는 누적 이동의 상한은 모른다. 기준 간 kept-set 겹침이 보통 75–85%라 이동 감소는
  dualr의 46%보다 훨씬 작을 것이고, 10–20%면 성공, 5% 미만이면 "선택은 캐시를 못 지킨다"가 결론이다.
  어느 쪽이든 결론이 나온다.
- I_cache가 I_traj와 같은 랭킹이면(G0-ii) 이 방향은 닫히고, 그것도 결론이다.

## 범위 밖
- 할당(깊이별 비율) 변경, refit과의 결합(cachedual + safe_refit dualr) — G1–G3 결과 뒤에 별도.

## G0 결과 (2026-08-31 00:05 KST) — 신규성 PASS, 안정성은 기준 재조정

- 프로브 16 → 32(seed 43 추가 pass)로 늘려도 split-half가 Q 0.970/0.953 → 0.970/0.958, MLP 0.912/0.907 →
  0.918/0.910으로 거의 안 움직인다 → 분산의 몸통은 프로브가 아니라 **클립 표집**(반쪽 50클립)이다.
  사전 등록한 0.95(jlens_coc32 수준)는 MLP에서 미달이지만, 이 기준과 합쳐질 traj/coc 기준 자체의
  캘리브레이션 잡음 바닥이 Q 0.86 / MLP 0.78(racfit G3 기록)이므로 I_cache는 기존 기준보다 안정적이다.
  **판정 기준을 house floor(traj/coc 바닥) 초과로 재조정하고 진행** — 사전 등록 이탈로 기록.
- 신규성: cachedual vs dual kept 겹침 Q 0.895 / MLP 0.885(< 0.95), cacheonly vs traj 0.82/0.80,
  Spearman vs I_traj +0.72/+0.66 → 같은 랭킹이 아니다. 마지막 층 I_cache는 정의대로 0.
- `outputs/cachejlens_v1/importance.npz`는 두 pass의 프로브 가중 평균(3,200 프로브)으로 덮어썼다;
  단일 pass 원본은 `cachejlens_v1b`(seed 43)만 남는다.

## 결과 (2026-08-31 03:30 KST) — G1 이동 PASS, G2 기각, G3 PASS: 캐시는 지켰는데 생성이 무너진다

| arm | 캐시 이동 (dual 비) | 캐시 비용 A10−A00 | val500 Δ | CoC 붕괴율 | LingoQA |
|---|---|---|---|---|---|
| dual | 1.000 | +0.077 | +0.058 | 0.014 | 68.8 |
| **cachedual** | **0.871** [0.867, 0.875] | +0.053 (−0.002 n.s.; 평균 +0.051 vs +0.106) | **+0.115** [+0.084, +0.174] | **0.226** (soup, 길이 297) | 69.0 (+0.2 n.s.) |
| cacheonly | 0.807 | +0.173 (dual보다 나쁨) | +0.457 | 0.632 | 26.8 |
| traj (기존) | — | — | +0.088 | 0.014 | 37.0 |
| dualr_rep (참고) | 0.544 | +0.006 | +0.008 | 0.010 | 52.2 |

- **선택만으로 민감도 가중 캐시 이동 −13%**(예상선 10–20% 안) — 선택이 캐시를 어느 정도는 지킨다.
- 그러나 val500은 dual의 두 배: 클립 22.6%에서 CoC가 soup으로 붕괴(dual 1.4%, coc-only 14.2%). LingoQA는
  dual과 같다 — 잃은 것은 reasoning이 아니라 **자기회귀 생성의 안정성**.
- **기전**: I_cache는 층 l의 게이트가 층 > l의 캐시에만 닿아 상류에 몰린다(Q head 질량 층 9–17에 70%,
  27–35에 2.5%). max(rank I_cache, rank I_CoC)에서 후반 층은 I_CoC 단독이 정하고, cachedual은 dual이
  남기던 층 27–35 MLP 채널 10,230개를 더 잘랐다 → grid의 "late-heavy 할당이 CoC를 무너뜨린다"가
  기준 안의 암묵적 할당으로 재현. dual에서는 I_traj가 CoC 캐시 위치를 통해 후반 층을 지키고 있었다.
- **프록시 맹점**: G1 프록시는 dense CoC를 teacher-forcing으로 고정해 생성 붕괴를 못 본다(경고는
  own-CoC NLL +0.064뿐). 앞으로 프록시에 rollout 붕괴율을 넣을 것.
- G4 미실행. 보고서 `reports/evaluation/2026-08-30_cache-jlens-criterion.html`.

**남는 길**: (a) I_cache를 앞 절반 층(0–17)에만, 후반 층은 I_traj 유지 — 이동 감소의 대부분은 앞쪽에서
오므로 붕괴 없이 −13%를 지킬 수 있는지; (b) 후반 층 보호용 생성 안정성 신호(`infer` 모드처럼 rollout
위에서 잰 NLL); (c) 기대선은 (a)가 성립하면 궤적 비용 −10% 안팎.
