# 단일 기준 arm 3개 추가 — `traj` / `coc` / `j` @ u40_v2

날짜: 2026-08-12 · 상태: 실행 중

## 가설

`dual_u40_v2` = `max(rank I_traj, rank I_CoC)`, `jtraj_u40_v2` = `max(rank I_traj, rank J)` 는
**조합 기준**이다. 두 arm의 차이가 "라벨(참조 CoC 텍스트)이 필요한가"를 격리하지만,
`max()` 안의 **각 항이 단독으로 무엇을 하는지**는 아직 측정되지 않았다.

측정하려는 것:

1. **H1 (조합 이득)** — `max(traj, X)` 가 `traj` 단독보다 나은가. 아니라면 dual/jtraj의
   추론 항은 장식이고, 궤적 Taylor 하나로 충분하다는 뜻이 된다.
2. **H2 (추론 항의 단독 능력)** — `coc` 단독과 `j` 단독이 각각 얼마나 무너지는가.
   J-lens가 CoC Taylor의 대체재라면 두 단독 arm도 비슷하게 거동해야 한다.
3. **H3 (개루프↔폐루프 부호 반전)** — 기존 2 arm에서 개루프는 −(나빠짐), 폐루프는
   +(좋아짐)로 뒤집혔다. 5개 arm으로 늘리면 이 반전이 arm 전반의 경향인지,
   두 점의 우연인지 구분된다.

## 설계 (1요인, 나머지 전부 고정)

기존 쌍과 **완전히 동일한 셀**을 쓴다. 바꾸는 것은 within-layer 점수 하나뿐:

| 고정 | 값 |
|---|---|
| 할당 | uniform, 레이어별 동일 비율 |
| 목표 예산 | `run_grid.allocations()["uniform"]` = **0.3985632694** (0.40 아님) |
| expert | 미개입 (16/16 Q, 8256/8256 MLP) |
| KV | drop 없음 |
| calibration | `importance_v2` + `jlens_v2` (동일한 100 클립) |
| 모델 리비전 | `MODEL_REV = 7aba829…` |

| config | within-layer 점수 | 성격 |
|---|---|---|
| `traj_u40_v2` | `I_traj` | 태스크 전용 (추론 정보 0) |
| `coc_u40_v2` | `I_CoC` | 추론 전용, **라벨 필요** |
| `j_u40_v2` | `J` | 추론 전용, **라벨 없음** |
| `dual_u40_v2` (기존) | `max(rank I_traj, rank I_CoC)` | 조합, 라벨 필요 |
| `j_traj_u40_v2` (기존) | `max(rank I_traj, rank J)` | 조합, 라벨 없음 |

`select_mask_ratios` 는 레이어 안에서 argsort로 고르므로 `rank_norm` 은 단일 기준에서
**선택을 바꾸지 않는다**(레이어별 단조 변환). 두 점수를 한 저울에 올릴 때만 의미가 있으므로
단일 arm은 원점수를 그대로 쓴다.

## 실행

1. `make_slim.py` 의 `*_u40_v2` 분기를 일반화 — 기존 두 체크포인트가 **비트 동일**하게
   재생성되는지 kept-index 비교로 먼저 검증한 뒤에만 빌드한다.
2. 체크포인트 3개 빌드: Ada 4/5/6 병렬, 각 ~1.5 h, 각 16 GB.
3. **개루프**: `indist_500` / `test_500` / `ood_1533`, best-of-8, `launch_arms.sh` 큐로
   Ada 4–7. 3 arm × 3 셋 = 9 잡, ~6 h. baseline은 `baseline_ada_*` 를 재사용.
4. **폐루프**: `public_2601` 150 씬, driver Ada 4–7 / renderer·physics Blackwell 2–3,
   `DRIVER_OMP_THREADS=8`. config당 ~8 h. **Blackwell이 비어야 시작 가능.**
5. `analyze_arms.py` 를 6-arm으로 확장, 보고서 두 개(개루프/폐루프)에 5행 표로 갱신.

## 사전 등록 게이트

- **G1**: 개루프 minADE@8 (test_500) 에서 `traj_u40_v2` ≤ `dual_u40_v2` 이면 H1 기각 —
  조합 이득 없음. 차이는 paired bootstrap 95% CI로 판정.
- **G2**: `coc_u40_v2` 와 `j_u40_v2` 의 paired 차이가 CI에 0을 포함하면 "J는 CoC Taylor의
  대체재" 주장이 단독 기준 수준에서도 유지된다.
- **G3**: 폐루프 씬 점수에서 5 arm 중 baseline보다 나은 arm이 2개 이상이면 "압축이
  정규화로 작동" 가설을 지지, 1개뿐이면 `dual_uniform` 단일 관측의 우연 가능성이 커진다.

## 위험

- `/mnt/nvme1n1` 여유 989 GB — 체크포인트 3개(48 GB)는 문제 없음.
- 폐루프는 arm당 8 h이므로 3개면 24 h. Blackwell 점유가 풀리는 시점이 임계 경로.
- 개루프/폐루프 모두 **Ada 드라이버 고정** — 아키텍처 혼용은 3–4% 클립에서 CoC가 갈린다.
