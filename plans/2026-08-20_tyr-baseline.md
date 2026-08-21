# Týr-the-Pruner 베이스라인 — tyr_u40 (2026-08-20)

## 목적과 가설

Týr-the-Pruner(AMD, NeurIPS 2025, arXiv:2503.09657)를 u40 예산의 **전역 sparsity 분배
최적화 베이스라인**으로 평가한다. 우리 기존 결과와의 관계: 그리드 트랙은 4개의 *고정*
할당(uniform/late/agree/depthprior)만 비교했고, Tyr는 할당을 **탐색**하며 로컬 프루닝도
선택이 아니라 **OSSCAR식 2차 근사 + 잔여 가중치 재구성**(least-squares 보정)이다.

**H-T1**: 재구성이 있는 Tyr-uniform(탐색 전 초기점)은 선택-만 하는 dual보다 낫거나 동급이다
(재구성 이득). **H-T2**: 탐색된 분배는 Tyr-uniform을 더 개선한다 (전역 탐색 이득).
두 arm을 분리 평가해 이득의 출처를 귀속한다.

## 원본 구현 확인 (AMD-AGI/Tyr-the-Pruner, Apache-2.0, 2026-08-20 clone)

1. **로컬 프루너** (`src/local_pruner.py`): o_proj·down_proj **입력**에서 Hessian
   H = ΣXXᵀ(fp32) 수집, G = H·W. `local_prune_core` = OSSCAR식 그룹 제거: 그룹별 2차
   목적값 argsort → H_inv 블록 업데이트로 축차 제거(o_proj는 head 그룹 = Q head,
   down_proj는 채널 단위 group_size 1; mlp_update_iter 16 / mha_update_iter 1) →
   **남은 가중치를 inv(H[kept])·G[kept]로 재구성**. 즉 선택+보정이 한 몸.
2. **Supernet** (`prune_to_supernet.py`, `src/pruner.py`): 블록 순차 처리. 레이어마다
   기대 sparsity를 중심으로 `weights_diff` 간격의 **9개 레벨**을 모두 절단·재구성해
   `<layer>/<level>.pth`로 저장. `--error_accumulation`: 기대 레벨의 가중치를 모델에
   기록한 뒤 다음 블록의 활성값을 생성(오차 전파 반영, median 버전).
3. **탐색** (`search_sparsity_dist.py`): 상태 = 레이어×모듈별 레벨 정수 벡터. 변이 =
   한 레이어 레벨 −1 + 다른 레이어 +1, **(decr+incr) 인덱스 짝수 제약** — 정렬된
   layer_names에서 mlp/o_proj가 교대로 나오므로 같은 모듈 타입 안에서만 예산 이동
   → 타입별 총예산이 구성상 보존. 세대당 offspring 128, 다단 선택(2048/16384/131072
   토큰, 생존 16/4/2, 엘리트 유지), 적합도 `sparse_kl` = dense teacher의 **top-8192
   로짓에 대한 KL**. 외부 4회 반복: 탐색 결과를 중심으로 step을 반씩 줄여 supernet 재구축.
4. 데이터: fineweb_edu 4.2M 토큰, 라벨 미사용 (teacher 로짓 = dense 모델 자신).

## Alpamayo 어댑테이션 (설계 결정)

- **축**: u40_v2 패밀리와 동일 — VLM Q head(o_proj 열 그룹)·MLP 채널(down_proj 열)만,
  expert·KV 불변. Tyr는 o_proj/down_proj만 수정하므로 mask 의미론과 정확히 일치하고,
  최종 모델은 재구성된 o_proj/down_proj를 기록한 뒤 기존 slim 수술로 물리 제거.
- **예산**: 타입별 총량을 u40_v2와 정확히 일치시켜 초기화 — 레이어마다 Q 13/32 절단,
  MLP 4898/12288 절단(= −2.657B). 변이의 타입-보존 제약으로 탐색 내내 예산 불변.
  레벨 정의: MHA step = **1 head**, MLP step = **256 채널**, 레벨 9개(±4).
- **적합도 (핵심 결정)**: 원본은 LM 로짓 KL 하나지만 우리 시스템은 이중 목적.
  coc-단독 기준의 주행 붕괴(폐루프 −0.089*) 전례가 있으므로 **dual-teacher 적합도**를
  제안한다: `fit = KL_sparse(CoC) / KL₀ + MSE_vf / MSE₀`
  - KL_sparse(CoC): dense teacher의 CoC rollout 텍스트를 양쪽에 teacher-forcing,
    **CoC 위치의 top-k(1024) 로짓 KL** (전체 vocab 저장 불필요, 라벨-free — teacher
    텍스트는 dense 모델 자신의 rollout).
  - MSE_vf: 같은 noise draw에서 dense expert의 vector field와 pruned의 field 간 MSE
    (trajectory 채널의 KL 대응물, GT 라벨 불필요).
  - 정규화 상수 KL₀·MSE₀ = 탐색 초기점(uniform supernet)의 값 — 한 번 재고 고정.
- **Calibration**: calib_100 (Hessian·teacher 로짓·적합도 모두), release-inference
  프롬프트. Hessian은 prefill 전체 토큰(원본이 전 토큰을 쓰는 것과 동일).
- **탐색 규모 (축소, 사전 등록)**: 외부 반복 **1회**(원 논문 4회의 첫 회에 이득이
  집중된다는 ablation + 예산; 결과가 유망하면 2회차를 후속 제안), 세대 20 ×
  offspring 32, 다단 선택 = 클립 (4, 16, 48) / 생존 (8, 2, 1), 엘리트 유지.
- **Supernet 저장**: bf16, 9레벨 × 36레이어 × (o 32MB + down 96MB) ≈ **41GB** —
  `outputs/tyr_supernet_u40/` (nvme1n1 562GB 여유 확인). 탐색 후 최종 config 외 삭제.

## 평가 arm (고정 프로토콜: rollout-only, @6, val500/test500/OOD-val262, Ada)

| arm | 무엇 | 분리하는 효과 |
|---|---|---|
| `tyr_uniform_u40` | supernet 초기점(uniform 분배 + OSSCAR 재구성) | 재구성 이득 (vs dual: 선택-만) |
| `tyr_u40` | 탐색된 분배 + 재구성 | 전역 탐색 이득 (vs tyr_uniform) |

## 사전 등록 게이트

- **T0 (무결성)**: 두 arm 모두 제거 파라미터 −2,657,452,032 정확 일치(타입-보존 확인),
  expert·KV 불변; 탐색 로그에 세대별 적합도 단조 개선 기록.
- **T1 (test_500)**: paired ΔminADE@6 (tyr_uniform − dual). CI < 0이면 재구성 이득 확정
  → 폐루프·LingoQA 후속 검증. CI ≥ 0이면 선택-만으로 충분하다는 근거.
- **T2 (test_500)**: paired ΔminADE@6 (tyr − tyr_uniform). CI < 0이면 전역 탐색 이득 확정.
- **T3 (부기록)**: degen, Δ vs baseline, minFDE@6, 탐색된 레이어별 분배 시각화
  (late-heavy 그리드 결과와 비교 — late 할당이 CoC를 붕괴시켰던 전례와 대조).

## 파일 구성 — `experiments/head_analysis/`

| 파일 | 역할 |
|---|---|
| `tyr_lib.py` (신규) | LocalPruner·local_prune_core 이식(Apache-2.0 출처 주석), Hessian 수집 훅, 레벨 생성 |
| `run_tyr_supernet.py` (신규) | 블록 순차: 블록 i 프리필 훅으로 H 수집 → 9레벨 절단·재구성·저장 → 기대 레벨 기록(error accumulation) |
| `run_tyr_search.py` (신규) | teacher 로짓/필드 캐시 → 진화 탐색(타입-보존 변이, 다단 선택) → `final_config.json` |
| `make_slim.py` (수정) | `tyr_u40`/`tyr_uniform_u40`: 재구성된 o/down 가중치 기록 후 마스크 수술 |
| `paper_numbers.py` (수정) | 두 arm × 3셋 ARMS 등록 |

## 실행 순서 / 예산 (Ada 1장, run_retry_host; wanda-eval 세션과 카드 공유 주의)

1. smoke: 2클립·2레이어·3레벨 supernet + 재구성 수치 검증 (dense 레벨 재구성 ≈ 원본
   가중치, prune_loss 감소 확인) — ~20분
2. supernet 본 구축 (36블록 × 100클립 프리필 + OSSCAR 9레벨) — ~4–6 h
3. `tyr_uniform_u40` slim 빌드 + 평가 3셋 (탐색과 병렬 가능) — ~3.5 h
4. 진화 탐색 20세대 — ~4–6 h
5. `tyr_u40` slim 빌드 + 평가 3셋 — ~3.5 h
6. T0–T3 판정 + 보고

합계 ~1.5 GPU-일 (카드 1장 직렬 기준).

## 상태

승인(2026-08-20, dual-teacher 적합도) — 구현 완료, 실행 대기 (사용자 지시로 GPU 실험 보류).

구현: tyr_lib.py(OSSCAR 이식, CPU 수학 검증 통과: 재구성=직접해, 그룹 수 정확, 목적값 단조),
run_tyr_supernet.py, run_tyr_search.py(mutate 예산 보존 2000회 검증), make_slim tyr 분기,
paper_numbers ARMS(tyr, tyr_uniform).

## v2 — damping 수정 (2026-08-20, 1차 결과 후 추가)

1차(업스트림 damping 1e-2 그대로) 결과: **tyr_uniform test minADE 2.11 / degen 74%,
tyr 2.06 / 87%** — 탐색 이전의 재구성 단계에서 이미 붕괴. 원인 진단(`outputs/tyr_hdiag.json`,
scratch `tyr_hdiag.py`): 프리필 calibration H의 조건수 1e34–1e38, 미소 고유값 수백~수천
개(L02 down_proj 12288 중 9946) → inv(H)가 재구성을 전 레이어 30–60%, 일부 레이어 2–12배
폭주시킴. 텍스트-토큰만의 H는 토큰 2만 개로 더 rank-deficient(탈락). damping을 mean-diag의
1.0으로 올리면 재구성 변화가 10–16%(o_proj)/2–15%(down_proj)로 정상화.

- **v2 설계**: 1차와 동일, `--damp 1.0`만 변경 (`tyr_supernet_u40_d1`, `tyr_search_u40_d1`,
  arm `tyr_uniform_u40_d1` / `tyr_u40_d1`). 사전 등록 게이트 T0–T3는 v2 arm에 그대로 적용.
- 감도 스크리닝: damp 0.1, level-0만, test 60클립 (`tyr_uniform_u40_d01_test60`) — 보고용.
- 1차(d=1e-2) 결과는 "업스트림 설정 그대로" 행으로 보존·보고한다 (T0 통과, T1/T2 붕괴).
- 탐색 적합도의 한계: teacher-forced KL은 rollout degeneration을 보지 못해 1차 탐색이
  degen을 74→87%로 키웠다. v2에서도 동일 적합도(업스트림 충실)를 쓰되 이 한계를 보고에 명시.

## 정정 (2026-08-20 22:30 UTC) — 평가된 arm은 selection-only였다

`tyr_sel_u40`(level-0 선택 + 원본 가중치) 60클립 결과가 v1 `tyr_uniform`과 소수점까지
동일(2.421/1.584/degen 0.717) → **`--no-state` 빌드는 slim_meta만 남기고 `load_slim`이
원본 가중치로 재구성하므로, 지금까지의 Tyr 평가(v1 uniform 3셋, v1 searched 3셋, d=0.1
스크리닝)는 전부 OSSCAR 재구성이 빠진 selection-only 모델이었다.** 따라서:
- 붕괴(degen 68–87%)의 원인은 **OSSCAR 선택 자체** — 프리필(비전 93%) 재구성 오차 목적이
  텍스트 생성에 필요한 유닛을 잘라낸다. 선택 겹침은 dual 76.8%/traj 80.4%로 "정상 범위"
  지만 결과는 coc-only·j-only급 붕괴.
- 재구성 효과는 미평가. 재평가: state 저장 빌드(`make_slim` 가드 추가: tyr 재구성 config는
  `--no-state` 거부)로 `tyr_uniform_u40_d1`(d=1.0, arm id `*_d1r_*`)·`tyr_u40_d1`(탐색)
  3셋 + v1 재구성(d=0.01) 60클립 스크리닝(`tyr_uniform_u40_recon_test60`).
- 기존 selection-only 결과는 `tyr_sel_uniform` / `tyr_sel_search`로 재라벨해 보존 (T0 통과,
  정확한 표기로 보고).

## 재정정 (2026-08-21 00:45 UTC) — 재구성이 해법이고 업스트림 damping이 옳았다

state 저장 빌드로 재구성을 실제로 평가한 결과 (test, run_baseline 요약 @8 평균):
- v1 재구성 (damp 1e-2 = 업스트림, "폭주"로 보였던 가중치) 60클립: **1.086 / degen 8.3%**
  — 같은 60클립 dual 1.116, baseline 1.025. 선택-only 2.42/72%에서 완전히 회복.
- v2 재구성 (damp 1.0): uniform 1.764 / 61%, 탐색 1.399 / 34% — 보정을 약하게 하면
  선택-only 붕괴 쪽으로 되돌아간다. 즉 큰 kept-가중치 변화는 수치 병리가 아니라 공선적
  head들 사이의 기능 이전(OSSCAR 보정) 자체였다. damping 강화(v2)는 잘못된 전제 위의
  수정이었고 감도 행으로만 보존한다.
- 정식 Tyr arm = **`tyr_uniform_u40_r` / `tyr_u40_r`** (d=1e-2 + 재구성, state 저장):
  3셋 평가 진행 중. 게이트 T0–T3는 이 두 arm에 적용. 탐색(`tyr_search_u40`)은 재구성된
  레벨 가중치를 로드해 적합도를 쟀으므로 탐색 자체는 유효.

## 최종 판정 (2026-08-21 02:50 UTC) — 정식 arm tyr_uniform_u40_r / tyr_u40_r, minADE@6 평균

| set | baseline | dual | tyr_uniform_r | tyr_r | T1 uniform_r−dual | tyr_r−dual | T2 tyr_r−uniform_r |
|---|---|---|---|---|---|---|---|
| test 500 | 0.842 | 0.950 | 0.999 (degen 4.8%) | 0.950 (0.8%) | +0.008 [−0.018,+0.026] | −0.001 [−0.027,+0.017] | **−0.010 [−0.017,−0.001]\*** |
| val 500 | 0.824 | 0.890 | 0.940 (4.2%) | 0.846 (0.8%) | +0.004 [−0.028,+0.027] | −0.023 [−0.048,+0.008] | **−0.006 [−0.017,−0.001]\*** (FDE −0.033\*) |
| OOD-val 262 | 1.000 | 1.119 | 1.289 (6.1%) | 1.176 (1.1%) | **+0.037 [+0.001,+0.103]\*** | +0.003 [−0.020,+0.059] | **−0.016 [−0.025,−0.001]\*** (FDE −0.083\*) |

- **T0 통과**: 두 arm 모두 −2,657,452,032, expert·KV 불변, 탐색 로그 보존.
- **T1 불충족**: OSSCAR 선택+재구성(uniform)은 Taylor dual과 in-dist 동급, OOD-val에서는 유의 열세.
  재구성 이득은 없다 — 단 선택-only OSSCAR는 붕괴(degen 68–87%)했으므로, 재구성은 "OSSCAR 선택의
  결함을 메우는" 역할이지 Taylor 선택을 넘는 역할이 아니다.
- **T2 통과 (3셋 일관)**: 전역 분배 탐색은 uniform 대비 ADE −0.006~−0.016\*, FDE val/OOD-val\*.
- **tyr_r vs dual: 3셋 모두 통계적 동급** (CI가 0 포함; 평균은 test 동일, val tyr_r 우세, OOD-val dual
  우세). degen은 tyr_r 0.8–1.1%로 dual 1.4–3.4%보다 낮다.
- 두 arm 모두 baseline 대비 유의 열세(+0.046~+0.135\*).
- 감도: damp 1.0(약한 재구성) uniform/탐색 test 1.76/1.40, degen 61/34% — 보정 약화는 해롭다.
- 비용: Tyr = supernet ~50분 + 탐색 ~50분(2–3 GPU) + 41GB 저장; dual = 역전파 1회 중요도, 재구성·탐색 없음.

**결론**: 충실한 Tyr(OSSCAR 재구성 + 전역 탐색)는 궤적에서 dual과 동급·추론 degen은 더 낮다; dual의
기여는 "선택만으로, 재구성·탐색 없이 같은 수준"이라는 단순성과 OOD 안정성. 후속 후보: dual 기준 +
Tyr 탐색(할당) 조합. 상태: **완료**.

## LingoQA (2026-08-21, 표준 프로토콜 VQA, Lingo-Judge, 500문항, dual 대비 세그먼트-클러스터 paired)

| arm | acc | Δ vs dual [95% CI] |
|---|---|---|
| baseline | 73.2% | +4.4pp [−0.4, +9.4] |
| dual | 68.8% | — |
| **tyr_r** (재구성+탐색) | **34.2%** | −34.6pp [−40.4, −28.6]\* |
| tyr_uniform_r (재구성, uniform) | 20.8% | −48.0pp [−53.4, −42.6]\* |
| traj / j / coc 단독 | 37.0 / 32.2 / 30.2 | −31.8 / −36.6 / −38.6\* |
| wanda | 9.2% | −59.6\* |

궤적 동급·CoC degen 0.8%였던 tyr_r도 **일반 VQA 지식은 단일-기준 arm 수준으로 붕괴**(34.2%,
답변 평균 1단어). 프리필 AV calibration 위의 OSSCAR 재구성 목적은 주행-영역 계산은 보존하지만
언어 지식은 보호하지 않는다 — calibration 분포 문제(coc/j/traj 단독과 같은 실패 양상). 탐색은
LingoQA를 20.8→34.2%로 크게 올렸지만(CoC-KL 항의 효과) dual(68.8%)과의 격차는 크다.
결론 보강: dual의 max(rank,rank)가 주는 "두 채널 중 하나라도 높이 평가한 유닛 보존"이 언어 능력
보존의 핵심이며, Tyr는 궤적 동급·언어 열세.

### 궤적↔언어 관계의 올바른 읽기 (2026-08-21, dual-global 세션 69efd68과 대조)

재구성 arm들을 나란히 놓으면 (test minADE@6, LingoQA)는
tyr_uniform_r (0.999, 20.8) → tyr_r (0.950, 34.2) → dualr (0.860, 41.8) → dualgr (0.864, 42.2)
로 **궤적이 좋을수록 언어도 좋다** — 둘 다 밑에 깔린 selection 품질을 따라가므로 arm 간 비교에서
트레이드오프로 읽으면 안 된다. 트레이드오프는 **같은 selection에서 재구성을 켤 때** 나타난다:
dual 0.950 / 68.8% → dualr 0.860 / 41.8% (궤적 −0.090\*, 언어 −27pp\*). OSSCAR selection 쪽은
재구성 off(=`tyr_sel_*`)의 LingoQA를 재지 않아 같은 토글 비교가 없다 (필요하면
`eval_lingo_tyr.sh --with-sensitivity`).

탐색 이득의 조건부성도 두 세션 결과가 일치: OSSCAR selection 위에서는 유의(T2, −0.006~−0.016\*)
하지만 dual selection 위에서는 0 (dualg +0.001 ns, tyr_r 할당 이식 +0.002 ns, dualgr−dualr ns)
— 전역 탐색은 selection이 약할수록 보상한다.

### 재구성 × selection 2×2 완성 (2026-08-21) — 언어에 대한 재구성 효과는 조건부다

> **[정정됨 / SUPERSEDED]** 아래 상호작용 해석은 무효다. OSSCAR 행의 두 셀(2.2%, 20.8%)이
> 모두 퇴화-상수 하한(37.0%) 아래여서 그 차이는 언어 능력이 아니다 — 다음 절을 볼 것.

빠져 있던 셀(OSSCAR selection + 재구성 off = `tyr_sel_u40`)을 측정해
{selection} × {OSSCAR 재구성 on/off}가 닫혔다. LingoQA는 세그먼트-클러스터 paired,
궤적은 test minADE@6:

| selection | 재구성 off | 재구성 on | Δ LingoQA (on−off) | Δ ADE |
|---|---|---|---|---|
| dual   | 68.8% / 0.950 | 41.8% / 0.860 | **−27.0pp [−32.6, −21.4]\*** | −0.090 (개선) |
| OSSCAR | **2.2%** / 2.292 | 20.8% / 0.999 | **+18.6pp [+14.4, +22.8]\*** | −1.293 (개선) |

- **궤적에서는 재구성이 두 selection 모두를 개선**한다.
- **언어에서는 부호가 뒤집힌다**: 좋은 selection(dual) 위에서는 깎고(−27pp\*), 무너진
  selection(OSSCAR 단독 2.2%) 위에서는 올린다(+18.6pp\*). 따라서 "재구성이 언어를 깎는다"는
  무조건적 명제가 아니라 **selection × 재구성 상호작용**이다.
- 해석: 최소자승 보정은 AV 프리필 분포에서 원래 레이어 출력을 복원하는 목적이므로, 언어
  유닛이 남아 있는 selection에서는 그 유닛을 AV 목적 쪽으로 끌어당겨 손상시키고, 언어가
  이미 파괴된 selection에서는 일반 기능을 부분 복구한다. 어느 쪽이든 dual(68.8%) 근처로는
  못 간다 — 언어 보존의 결정 요인은 여전히 selection이다.
- 소스: `lingo_vqa_slim_tyr_sel_u40`(2.2%), `lingo_vqa_scores_tyrsel`, 토글 분석
  `lingo_toggle_osscar` / `lingo_toggle_dual`; dualr 수치는 dual-global 세션(69efd68).


### LingoQA 퇴화-상수 하한 (2026-08-21) — 표 전체의 해석 기준

peer(dual 세션)가 재구성 arm들의 출력이 토큰 샐러드·초단문 쓰레기임을 지적해 대조군을 만들었다:
질문과 무관하게 **항상 같은 한 마디만 답하는** 예측 파일을 Lingo-Judge로 채점
(`lingo_vqa_const_{no,yes,2,empty}`, `lingo_vqa_scores_controls`).

    "No."  37.0%      "Yes."  37.6%      "2"  7.0%      "."  0.6%

즉 **이 벤치마크의 퇴화 하한은 ~37%**다 (질문 51%가 yes/no형이라 상수 부정이 절반 이상 맞는다).
세그먼트-클러스터 paired로 각 arm을 `const_no` 대비 검정하면 (`lingo_vs_constant_floor`):

| arm | acc | Δ vs 상수 하한 [95% CI] | 판정 |
|---|---|---|---|
| baseline | 73.2% | +36.2 [+30.8, +41.6]\* | 하한 초과 |
| dual | 68.8% | +31.8 [+26.0, +37.4]\* | 하한 초과 |
| dualr | 41.8% | +4.8 [−0.8, +10.2] | 구분 불가 |
| traj | 37.0% | +0.0 [−6.2, +6.0] | 구분 불가 |
| tyr_r | 34.2% | −2.8 [−7.6, +1.8] | 구분 불가 |
| j | 32.2% | −4.8 [−11.0, +1.4] | 구분 불가 |
| coc | 30.2% | −6.8 [−12.8, −0.8]\* | 하한 미만 |
| tyr_uniform_r | 20.8% | −16.2 [−21.2, −11.4]\* | 하한 미만 |
| wanda | 9.2% | −27.8 [−32.8, −23.0]\* | 하한 미만 |
| tyr_sel | 2.2% | −34.8 [−39.4, −30.2]\* | 하한 미만 |

**따라서**: 언어 능력이 실제로 남아 있다고 말할 수 있는 프루닝 arm은 **dual 하나**다. 나머지
arm들 사이의 LingoQA 서열(traj 37.0 > tyr_r 34.2 > j 32.2 > coc 30.2 …)은 능력 차이가 아니라
붕괴 출력이 판정기에서 얻는 형식 점수의 차이이며, 그 위에서 세운 비교(내 2×2 상호작용 포함)는
해석 불가다. 재구성 토글이 의미를 갖는 유일한 축은 **양쪽 셀이 하한 위인 경우**인데 현재는
dual(68.8, 하한 초과) → dualr(41.8, 구분 불가)뿐이라, 그 −27pp도 "하한 위에서 하한 부근으로
떨어졌다"까지만 말할 수 있다.

**보고 규약(이후 모든 LingoQA 표에 적용)**: 상수 하한 37.0%를 표에 함께 싣고, 하한과 구분되지
않는 셀은 능력 수치로 인용하지 않는다. 기존 논문 표 T3(`2026-08-19_results_tables.tex`)의
단일-기준 행(traj 37.0 / j 32.2 / coc 30.2)과 wanda·Tyr 행이 여기 해당하므로 수정이 필요하다.
정말로 상호작용을 검정하려면 붕괴하지 않은 두 번째 selection이 필요하다(peer 제안: traj 위
재구성 토글 — 단 traj도 하한과 구분 불가라 후보로 부적합; 하한을 넘는 selection은 현재 dual뿐).
