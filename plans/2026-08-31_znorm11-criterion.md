# 11-손실 znorm 기준 — dual과 얼마나 같은 유닛을 남기는가 (코드/오프라인)

날짜: 2026-08-31. 브랜치: `znorm11-criterion`. 상태: **코드 + 오프라인 분석 완료, 빌드/평가 미실행**
(사용자 요청: "코드만").

## 질문

출시 기준 `dual` = `max(rank_norm I_traj, rank_norm I_CoC)`에서 `I_traj`는 10개 디노이징 스텝의
기울기를 **합산**한 `mean_clips |Σ_s ∂L_s/∂g|` 하나다. 스텝을 합치지 않고 **CoC 손실 1개 + FM 스텝
손실 10개 = 11개를 각각 레이어 내 z-정규화한 뒤 평균**하면(=`znorm11`) 남는 유닛이 dual과 얼마나
겹치는가.

## 재료 (재측정 없음)

- per-step VLM 기울기: `outputs/importance_stepvlm_v1/step_importance_vlm.npz`
  (`q_abs_step` (10, 36, 32), `mlp_abs_step` (10, 36, 12288), calib_100, Ada)
- CoC 반: `outputs/importance_v2_ada/importance.npz` — 같은 클립·같은 아키텍처
  (`make_stepvlm_importance.py`가 드롭인을 만들 때 쓰는 그 참조 파일)
- 예산: u40 균일(레이어당 Q 19/32, MLP 7390/12288 유지), `select_mask_ratios`의 레이어 내 argsort

## 결과 (오프라인, `analyze_criterion_overlap.py`)

| 기준 | Q 일치 | MLP 일치 | Q Jaccard | MLP Jaccard | Q ρ | MLP ρ | 파라미터 churn |
|---|---|---|---|---|---|---|---|
| **znorm11** (CoC + 10 step) | 0.871 | 0.881 | 0.776 | 0.790 | +0.867 | +0.869 | **12.1%** |
| znorm10 (step만) | 0.857 | 0.865 | 0.755 | 0.766 | +0.825 | +0.839 | 13.7% |
| znorm5050 (CoC ½ + step평균 ½) | 0.918 | 0.907 | 0.852 | 0.832 | +0.939 | +0.913 | 9.1% |
| traj 단독 | 0.861 | 0.867 | 0.762 | 0.771 | +0.850 | +0.854 | 13.4% |
| CoC 단독 | 0.857 | 0.865 | 0.756 | 0.766 | +0.849 | +0.839 | 13.7% |
| **dualfix** (아래) | 0.994 | 0.993 | 0.990 | 0.989 | +0.992 | +0.989 | 0.6% |

- **znorm11은 dual과 87~88% 같고 12%가 다르다** — 남는 3.99B 중 483M(헤드 88개 + 채널 31,794개).
- 차이의 대부분은 **스텝 축 집계가 아니라 CoC 가중치**에서 온다: znorm10(0.857) ≈ traj 단독(0.861)이고,
  CoC를 1/11(9%)에서 1/2로 올린 znorm5050은 0.918로 dual에 훨씬 가깝다. dual의 `max(rank, rank)`는
  두 목적을 대등하게 다루므로, 11개 균등 평균은 FM 쪽으로 10:1 기울어진 기준이다.
- 레이어별 편차: Q 0.79–0.95, MLP 0.76–0.99(얕은 층 MLP가 가장 잘 맞고 깊을수록 갈린다).

## 부수 발견 — 출시 dual에 들어 있는 퇴화 (층 35)

`traj_vlm_q`·`traj_vlm_mlp`는 **층 35에서 정확히 0**이다(다른 35개 층은 정상). 마지막 층의
o_proj/down_proj 출력은 그 층의 K/V를 만들지 않아 expert가 읽는 캐시에 도달하지 못하므로 궤적
기울기가 구조적으로 0이다. 그런데 `rank_norm(0벡터)` = `argsort(argsort(zeros))/(n−1)` = **인덱스
순서**이고, `max(rank traj, rank coc)`가 그 인덱스 순서를 CoC와 경쟁시킨다.

검증: `slim_dual_u40_v2/slim_meta.json`의 층 35 kept Q = `[0,2,5,8,12,14,19,...,31]`이 재현되고
(importance_v2 / v2_ada 모두), 그중 **8/19가 인덱스 순서 쪽이 더 커서 선택**됐으며 kept set의 74%가
"인덱스 상위 19개"와 겹친다(정상 층 63%). **출시된 체크포인트에 실재하는 결함**이며 영향 범위는
36층 중 1층(파라미터의 2.8%)이다.

수정 `dualfix`: 반쪽이 상수인 레이어는 max 경쟁에서 빠지고(−inf) 나머지 반쪽 단독으로 결정한다.
dual과 99.4% 동일 — 즉 **층 35만 바뀌고 나머지는 그대로**다(파라미터 churn 0.6%).

## 코드 (이번에 추가/변경, 전부 GPU 불필요)

| 파일 | 변경 |
|---|---|
| `experiments/head_analysis/analyze_criterion_overlap.py` (신규) | 임의의 두 기준의 kept set 비교(일치율·Jaccard·Spearman·파라미터 churn·레이어별 곡선), 퇴화 레이어 진단 |
| `experiments/head_analysis/make_slim.py` | `half()`에 `znorm11` 스템(11-손실 z-평균, `--stepvlm`로 per-step 파일 지정), `dualfix` 스템(상수 레이어 가드 `rank_or_nan`), 도크스트링·CLI 추가. 기존 분기 동작 불변 |

빌드 명령(미실행):
```
bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/make_slim.py \
     --config znorm11_u40_v2 --importance importance_v2_ada --stepvlm importance_stepvlm_v1 \
     --gpu 0 --out outputs/slim_znorm11_u40_v2
bash experiments/head_analysis/run_retry_host.sh 30 experiments/head_analysis/make_slim.py \
     --config dualfix_u40_v2 --importance importance_v2_ada --gpu 1 --out outputs/slim_dualfix_u40_v2
```
평가는 고정 프로토콜(val500/test500/OOD-val, Ada)로 dual과 페어드. **주의**: `--importance`는
per-step 파일과 같은 아키텍처여야 한다(Ada 본 `importance_v2_ada`), 그렇지 않으면 두 반쪽이 서로
다른 커널에서 측정된 이중 요인이 된다.

## 다음(승인 시)

1. 두 체크포인트 빌드(Blackwell 가능, 각 ~3분) → val500 paired로 12% 선택 차이가 성능 차이인지.
2. `dualfix`는 사실상 무비용 수정이므로, 차이가 없다면 "결함은 실재하나 영향 없음"으로 기록하고
   기준 코드에만 반영하는 선택지도 있다.
