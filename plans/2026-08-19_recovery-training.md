# 회복 학습 (KI-LoRA) — dual_u55_v2 (2026-08-19)

## 가설과 배경

구조적 프루닝 33.1% (`slim_dual_u55_v2`, 7.41B)는 제로샷으로 붕괴한다:
test_500 minADE@6(rollout) **4.2644** (median 3.0397), CoC degen **0.714**
(@6 앵커: baseline 0.8417 / dual_u40_v2 0.9498, degen ~0.03). u40은 제로샷 유지가
이미 증명됐으므로, 회복 학습의 질문은 **"u55의 붕괴를 소량의 학습으로 되돌릴 수
있는가"**이다.

open-loop 지표는 **minADE best-of-6**로 보고한다 (사용자 결정). 샘플 seed가
base+k prefix이고 최근 런은 per-sample 배열(`ade_rollout_k`)을 저장하므로, @6는
"저장된 배열의 앞 6개 min"으로 정의된다 — K=8로 실행해도 지표는 동일하다.

**H1**: KI-LoRA 회복 학습(~4,800 샘플)으로 u55의 궤적·추론 붕괴가 대부분 회복된다.

사전 등록 게이트 (test_500, paired vs zero-shot u55):
- **G1 (궤적)**: paired ΔminADE@6 CI < 0 (개선 확정), 목표선 mean ≤ 1.6
  (u40 @6 0.9498까지 gap 3.31 m의 80% 회복).
- **G2 (추론)**: CoC degen 0.714 → **≤ 0.05**.
- **G3 (폐루프)**: alpasim 150 scenes, paired d_score(vs baseline) CI 하한 > −0.080
  (powered effect 기준 non-inferior). 리포트 중심, 통과/실패는 G1·G2로 판정.

## 사용자 확정 설계

1. **KI-LoRA** (π0.5 Knowledge Insulation): VLM forward 1회(use_cache)로 CE logits +
   KV cache를 얻고, **cache를 detach**한 뒤 expert가 FM 스텝을 돈다.
   → FM grad는 expert LoRA에만, CE grad는 VLM LoRA에만 흐른다.
   - LoRA: 살아남은 q/k/v/o + gate/up/down, VLM 36층 + expert 36층, r=32 / α=64.
   - lm_head, embed, k/v_norm 등 나머지 전부 동결. FM 규약은
     `prune_lib.expert_fm_grads`와 동일 (x_t=(1−t)·noise+t·x1, v_target=x1−noise).
   - 두 손실은 파라미터가 분리되므로 가중 없이 L = L_ce + L_fm.
2. **손실**: FM + CE. CE 대상 CoC:
   - OOD train 1,271: 큐레이션된 `gt_coc` (ood 캐시 npz에 저장돼 있음).
   - official train: `/mnt/nvme1n1/ad_vla/data/coc_generated/train/coc_train.parquet`
     (full 모델 생성, 119,260 샘플 / 10,040 클립 / t0 그리드 2–13 s, 샘플별 minADE 동봉).
3. **학습 데이터** (~2,471 샘플):
   - OOD train **1,271 전량** (per-clip t0, 기존 ood 캐시에서 로드).
   - official train **1,200**: coc_train.parquet에서
     ① calib_100 겹침 9클립 제외, ② coc degenerate/빈 텍스트 제외,
     ③ 품질 게이트 minADE ≤ 3.0 (≈75th pct — 나쁜 rollout의 CoC를 distill하지 않기 위함),
     ④ 클립당 1샘플(t0는 후보 중 랜덤)로 클립 다양성 확보,
     ⑤ 시나리오 버킷 **{left_turn, right_turn, decel_stop, accel, cruise}** 균형 샘플링
     (240/버킷, 부족 버킷은 잔여 재배분). 버킷은 `eval_lib.gt_geometry` 확장
     (net_turn에 부호 추가)으로 GT future 궤적에서 계산하며, 궤적은
     `labels/egomotion` zip만 읽어 산출 (video decode 없음).
   - 선택된 official 1,200 샘플은 sample_cache 형식 npz로 캐시 빌드
     (`pre_processed/train/samples`, ~4 GB; coc 텍스트는 recovery_sets parquet에서 참조).
4. **Validation (500클립)**: ood_val **262 전량** + official val **238**
   (eval 캐시 18,868 중 **val_500(=indist_500) 제외** 풀에서 동일 버킷 균형 샘플링 —
   보고 세트 오염 방지). Probe는 K=1 **rollout** (u55의 주 증상이 degen이므로
   teacher-forced가 아니라 rollout이어야 함): mean/median minADE + degen rate,
   ood/official 분리 리포트. 매 150 step (+step 0, 최종) ≈ 40분/회 × 5회.
   모델 선정: val mean minADE 최소 체크포인트 (degen은 게이트로 함께 기록).
5. **대상**: `dual_u55_v2`부터. slim_state.pt가 없으므로 `make_slim`으로 재구성(~1.5 h)
   후 학습. 학습 후 평가: **val_500, test_500, ood_val 262, alpasim 150 scenes**.
   - open-loop 지표는 **minADE@6**. baseline·dual_u40은 기존 `_ps_` rows
     (`baseline_ada_ps_{indist,test,oodval}`, `dual_u40_v2_ps_{indist,test,ood}`)에서
     @6 재환산 — 재실행 0. @6 앵커: test 0.8417/0.9498, val_500 0.8236/0.8904,
     ood_val 0.9995/1.1192 (baseline/u40).
   - **u55 zero-shot은 val_500·ood_val 런이 없어 신규 실행** (~2 h; K 문제와 무관하게
     decision 5에 필요). recovered와 u55 신규 런은 K=8로 실행해 per-sample 배열을
     저장하고 @6로 보고 (클립당 시간은 CoC rollout decode가 지배해 K=8의 추가
     비용은 미미; @8은 metrics.json에 부기록).
   - OOD 평가는 ood_val 262만 클린 (train 1,271은 학습에 사용). 비교 arm들은 기존
     rows에서 같은 262 부분집합을 재집계.
   - open-loop은 Ada, paired seed (`clip_seed`); alpasim은 shards 경로 (Ada, OMP 8).
6. **체크포인트**: adapter만 저장 (`adapter_state.pt` ~0.3 GB + config + val log).
   `rebuild_merged.py`가 slim 재구성 → LoRA merge → merged(17 GB)를 필요 시 생성
   (recipes 방식). 평가·alpasim 직전에만 materialize.

## 파일 구성 — `experiments/recovery/` (신규)

| 파일 | 역할 |
|---|---|
| `recover_lib.py` | KI-LoRA attach(peft), FM/CE loss (in-repo, phase21 의존 제거), adapter save/merge |
| `make_train_set.py` | 후보 필터 + egomotion 버킷팅 + 균형 샘플링 → `outputs/recovery_sets/*.parquet` + 분포 plot |
| `build_train_cache.py` | 선택 official 샘플 npz 캐시 빌드 |
| `train_recover.py` | 본 학습 (Design-B forward + KI detach), outputs 규약 준수 |
| `rebuild_merged.py` | adapter + slim_meta → merged 체크포인트 재구성 |

평가는 기존 `experiments/evaluation/run_baseline.py`를 merged 경로로 재사용.

## 실행 순서 / 예산 (Ada 1장, run_retry_host)

1. `make_train_set` + `build_train_cache` — ~1 h
2. `make_slim --config dual_u55_v2` state 재구성 — ~1.5 h
3. **smoke** (10 step + probe 20클립) — 게이트: peak < 46 GB, loss 하강,
   **KI 단위 검증** (CE 끈 스텝에서 VLM LoRA grad가 전부 0인지 확인)
4. 본 학습 600 step × accum 8, lr 1e-4 cosine (warmup 50), clip 1.0, bf16, seed 42
   — ~4 h + probe ~3 h
5. 평가: recovered open-loop 3세트 ~3 h + u55 zero-shot 보충 런 ~2 h
   + alpasim 150 scenes ~8 h

합계 ~1.5일 wall-clock.

## 리스크 / 후속

- KI로 VLM은 궤적 신호를 전혀 받지 않는다. CE만으로 VLM이 회복되지 않아
  minADE가 정체하면, cache-attach(FM→VLM 허용) 대조 arm이 자연스러운 후속.
- OOD gt_coc(사람 큐레이션)과 official 생성 CoC의 스타일 차이가 CE를 흔들 수 있어
  probe에서 ood/official을 분리해 관찰.
- **파인튜닝-베이스라인 대조군**(unpruned + 동일 레시피): u55 회복이 성공하면
  "회복 vs 단순 파인튜닝 이득" 분리를 위해 즉시 후속으로 제안.
