# Wanda 베이스라인 — wanda_u40_v2 (2026-08-20)

## 목적과 가설

Wanda(Sun et al. 2023: score = |W|·‖X‖₂, gradient-free·label-free·retraining-free)를
u40_v2 셀의 **훈련-신호 없는 베이스라인 기준**으로 추가한다. 리뷰어 관점에서 "gradient
기반 Taylor 기준이 activation-aware magnitude보다 정말 나은가"에 대한 직접 답이 된다.

**H-W**: 매칭 예산에서 Wanda는 dual보다 나쁘다 (paired ΔminADE@6 > 0). 만약 동급이거나
낫다면 — Taylor 측정(역전파, 이중 목적) 없이도 되는 것이므로 기준 트랙의 주장을 재검토해야
하는 중요 결과.

## 원본 구현 확인 (locuslab/wanda, 2026-08-20 clone)

`lib/layerwrapper.py`·`lib/prune.py`·`lib/data.py`에서 확인한 디테일과 반영 방식:

1. **활성값 통계** (`WrappedGPT.add_batch`): 각 Linear의 **입력** dim j에 대해
   `scaler_row[j] = 샘플 평균( Σ_tokens x_{t,j}² )` — 샘플별 토큰 제곱합의 러닝 평균,
   **fp32 누적** (`inp.type(torch.float32)`), (B·T, D)로 펴서 축적.
   → 동일 공식·동일 dtype으로 반영: 클립 1개 = 샘플 1개, 클립별 Σx²의 러닝 평균.
2. **점수**: `W_metric = |W| · sqrt(scaler_row)` (선택 시점에 sqrt) → 그대로.
3. **비교 그룹**: 비구조적 기본은 **output row별** 정렬·절단(dim=-1), N:M은 row×m-블록.
   유닛(열 전체) 제거에는 row별 그룹이 정의되지 않으므로 — 원본 repo에 구조적 유닛
   버전은 없음 — **레이어 내 유닛 간 비교**(기존 `select_mask_ratios`)가 우리의
   어댑테이션임을 명시. 유닛 집계도 원본에 없으므로 사전 등록으로 고정:
   **주 점수 = 열 L2 집계**, L1 집계는 npz에 병행 저장해 kept-set 겹침만 부기록.
   - MLP 채널 c (down_proj 입력): `sqrt(scaler_row[c]) · ‖W_down[:,c]‖₂`
   - Q head h (o_proj 입력 128차원): `sqrt( Σ_{j∈head} scaler_row[j] · ‖W_o[:,j]‖₂² )`
4. **Calibration**: 원본은 C4 train 128샘플 × seqlen 2048 (≈26만 토큰), 라벨 미사용,
   forward-only, use_cache=False, 레이어 순차 전파(메모리 트릭).
   → 우리는 calib_100의 release-inference fused 프롬프트 **prefill 전체 토큰**
   (vision 포함, CoC 참조 텍스트 없음 — label-free 유지), 전체 모델 1회 forward에
   o_proj/down_proj 입력 pre-hook (VRAM 충분, 순차 전파 불필요). 토큰 수는 클립당
   수천 × 100클립 ≈ 원본 이상.
5. **커버리지**: 원본은 블록 내 모든 Linear(q,k,v,o,gate,up,down)를 절단. 우리는
   u40_v2 축(Q head → q/o, MLP 채널 → gate/up/down)만 제거하고 **k/v는 불변** —
   expert가 읽는 KV 인터페이스(8×128)가 하드 제약이기 때문. 편차로 명시.
6. `--use_variant`(row별 sparsity 조정 α 탐색)는 기본 off이고 유닛 제거에 무의미 — 미사용.

훅 지점 일치 확인: 원본이 각 Linear의 입력을 잡는 것과 동일하게, down_proj 입력
(SwiGLU 활성 후)과 o_proj 입력(head 연결 출력)은 기존 mask_lib 게이트 지점 그대로다.

## 설계 (one-factor, u40_v2 셀)

- 예산·축 동일: uniform 0.3985632694, 레이어당 Q 13/32·MLP 4898/12288 절단, VLM만,
  expert·KV 불변, 제거 파라미터 2,657,452,032 정확 일치.
- Calibration: calib_100 동일 100클립 (importance_v2와 같은 클립 → 기준만 one-factor).
- 기존 기준들과의 kept-set 겹침 기록 (dual/traj/coc/j + L1/L2 집계 간).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `run_wanda.py` (신규) | pre-hook fp32 Σx² 러닝 평균 수집 → 가중치 norm 결합 → `outputs/wanda_v1/wanda.npz` (`q_w`, `mlp_w`, L1 변형 `q_w_l1`, `mlp_w_l1`) |
| `make_slim.py` (수정) | u40_v2 패밀리 `half()`에 `wanda` 스템 추가 (단일 기준, rank_norm 불필요) |
| `paper_numbers.py` (수정) | ARMS에 wanda 3셋 등록 |

## 사전 등록 게이트

- **W0 (무결성)**: 제거 파라미터 정확 일치, 레이어별 절단 수 동일, expert·KV 불변.
- **W1 (주 판정, test_500)**: paired ΔminADE@6 (wanda − dual), bootstrap median CI.
  - CI 전체 > 0 → 예상대로 Taylor 기준 우위 확인, T2 베이스라인 행으로 수록.
  - CI가 0 포함 또는 < 0 → 중요 결과: val·OOD-val 확인 후 LingoQA·폐루프 후속 검증
    필수 (기준 트랙 주장 재검토).
- **W2 (부기록)**: degen rate, paired Δ vs baseline, minFDE@6, kept-set 겹침
  (vs 기존 기준들, L1 vs L2 집계).

## 평가 (고정 프로토콜)

rollout-only(TF 금지), K=8 per-sample 배열(호라이즌 포함) 저장 → @6 축소,
**val 500 · test 500 · OOD-val 262** 3셋 전부 (T2 형식과 맞춤), Ada 카드, paired seed.

## 실행 순서 / 예산 (Ada, run_retry_host)

1. `run_wanda.py` smoke(클립 2, scaler_row 유한성·shape 확인) → 본 수집 100클립
   — forward-only ~1 h, 1장
2. `make_slim --config wanda_u40_v2 --no-state` + W0 검증 — ~30–40 분
3. `run_baseline` 3셋 — test·val 각 ~1.2 h, OOD-val ~0.6 h (가용 카드에 분산)
4. W1/W2 판정 + 보고, `paper_numbers.py`로 재계산 검증

합계: 카드 1장 기준 ~4 h, 2–3장이면 ~2.5 h.

## 상태

승인(2026-08-20 "진행해줘") — 실행 개시. 구현: run_wanda.py + make_slim --config wanda_u40_v2 + paper_numbers ARMS.

**완료 — H-W 확인, Wanda 기각 (2026-08-20).**
- W0: 제거 2,657,452,032 정확 일치, 19/7390, expert·KV 불변. kept-set 겹침
  dual 77.2/79.2%, traj 79.5/78.8%, coc 74.0/78.3%, j 63.0/67.0%.
- W1 (test_500): ΔminADE@6 vs dual **+1.0589 [+0.8987, +1.2902]*** — CI 전체 > 0,
  압도적 열세. val +1.0867*, OOD-val +0.8944*. FDE도 +3.12* (test).
- W2: degen **86.0/87.6/85.9%** (val/test/OOD-val; dual 1.4/3.0/3.4%) — 추론 채널 붕괴.
  절대치: test ADE@6 2.9754 / FDE@6 8.2687 (dual 0.9498/2.5459, baseline 0.8417/2.2673).
- 위치: 같은 24% 예산의 모든 gradient 기반 단독 기준보다 나쁨 (j-단독 2.158 < wanda 2.975
  < u55 제로샷 4.264). 해석: per-weight 비구조적 세팅에서 성립하던 활성값×크기 신호는
  유닛 단위로 굵어지면 이 VLA의 추론·주행 필수 유닛을 식별하지 못한다 — Taylor 기준
  (특히 dual)의 논문 방어 근거.

## 변형 arm: wanda_txt (2026-08-20 승인 "wanda_txt에 대해서도 진행해줘")

감사 결과 구현 오류는 없었으나(마스크 == 점수 top-k 36/36, 방향 정상, 스모크↔본수집
Spearman 0.95/0.87), 캘리브레이션 프롬프트의 **93%가 vision 토큰**이라 Wanda의 ‖X‖가
"이미지 토큰에서의 발화"를 재는 조건이었다. 이를 분리하기 위한 변형:

- `run_wanda.py --tokens text`: ‖X‖를 **프롬프트 text 스팬(compute_spans: vision·궤적이력·
  sink 제외) ∪ 모델 자체 rollout CoC 토큰**(clip_seed, temperature 0.6, max 256)에서만
  누적 — teacher-forced forward 1회. 여전히 라벨 없음. 예산·축·집계는 wanda와 동일.
  출력 `outputs/wanda_txt_v1`, config `wandatxt_u40_v2` (`make_slim --wanda-txt`).
- 평가: 고정 프로토콜 3셋 + LingoQA(`eval_lingo_arm.sh slim_wandatxt_u40_v2`).

사전 등록 게이트 (추가):
- **W0-txt**: wanda와 동일 무결성.
- **W3 (핵심 대조, test_500)**: paired ΔminADE@6 (wanda_txt − wanda) bootstrap CI.
  CI 전체 < 0 → vision 지배가 붕괴의 주 원인(텍스트 제한이 회복); CI 0 포함 → 구조적
  Wanda 자체의 한계. W1-txt(vs dual)는 그대로 보고.
- **L1-txt**: LingoQA paired Δ vs dual 및 vs wanda(세그먼트 군집 CI).


### wanda_txt 결과 (2026-08-20 완료)

W0-txt 통과: 제거 2,657,452,032 정확, 마스크 == 점수 top-k 36/36,
kept-set 겹침 **wanda 96.1/97.9%**, dual 78.2/79.9%.

| 셋 | wandatxt ADE@6/FDE@6 | wanda | dual | ΔADE vs wanda (W3) | ΔADE vs dual (W1-txt) |
|---|---|---|---|---|---|
| val 500 | 2.141 / 6.073 | 2.700 / 7.514 | 0.890 / 2.330 | **−0.372 [−0.535,−0.179]*** | +0.600* |
| test 500 | 2.320 / 6.392 | 2.975 / 8.269 | 0.950 / 2.546 | **−0.302 [−0.534,−0.104]*** | +0.625* |
| OOD-val 262 | 2.357 / 6.036 | 2.790 / 6.884 | 1.119 / 2.851 | −0.165 [−0.330,+0.053] | +0.788* |

degen: wandatxt 63.2/63.4/66.8% vs wanda 86.0/87.6/85.9%.
L1-txt (LingoQA): wandatxt **8.4%** vs wanda 9.2%, dual 68.8% —
Δ vs dual −60.4pp [−65.0,−55.6]*, ≤2단어 75.8%, token-junk 14.4%(wanda 56.2%).

**판정**: W3는 in-distribution 두 셋에서 유의한 부분 개선(−0.30~−0.37, degen −23pp)이지만
OOD-val에서는 미검출이고, 어느 셋에서도 dual과의 격차(+0.60~+0.79*)를 좁히지 못한다.
LingoQA는 회복이 전혀 없다(8.4 vs 9.2%, ns 수준). 즉 vision 토큰 지배는 붕괴의
**부차적 원인**일 뿐이고, 주 원인은 구조적 유닛 단위로 적용된 |W|·‖X‖ 신호 자체다.
논문에는 wanda(all)를 대표 베이스라인 행으로, wanda_txt를 "캘리브레이션 토큰을 바꿔도
회복되지 않는다"는 강건성 각주로 싣는다.