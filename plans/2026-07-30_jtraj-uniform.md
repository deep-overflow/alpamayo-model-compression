# j_traj_uniform: 라벨-프리 기준을 dual_uniform 구조에 이식

## Hypothesis

**H**: `j_traj_full_r20`의 폐루프 종방향 열화(lead_thw_median −0.749 s, p=0.0028)는
기준(criterion)이 아니라 **얹힌 축**(expert magnitude pruning, Q-head 유도 J-mass 기반 KV1 drop)
탓이다. 근거:

- Stage E(80클립, VLM-only·uniform·KV 없음·expert 없음)에서 j_traj는 dual과 r20에서 구분
  불가(dNLL p=0.96), r30에서는 dADE가 더 좋았다(−0.104, p=0.006). **구조가 dual_uniform과 같을
  때 기준 차이는 open-loop에서 검출되지 않았다.**
- dual_uniform(−24.0%, VLM-only)은 폐루프 최선 config인 반면, 같은 dual 기준에 expert+KV를 얹은
  cocsafe_full_r20(−18.1%)은 baseline보다 나빴다 → 축이 손해를 만든다는 독립 증거.
- j_traj_full의 KV 점수는 J-lens가 직접 채점하지 못해 Q-head J-score 제곱합으로 **유도**한
  약한 고리다. VLM-only 구조에서는 이 고리가 아예 사라진다.

검증 방식: dual_uniform과 **모든 것이 동일하고 기준만** dual → j_traj로 바뀐 `j_traj_uniform`을
만들어 open-loop → 폐루프 순서로 비교한다. 완전한 one-factor 실험.

## Config 정의

| | dual_uniform (기존) | **j_traj_uniform (신규)** |
|---|---|---|
| 기준 | max(rank I_traj, rank I_CoC) | max(rank I_traj, rank J) — CoC 라벨 불필요 |
| 배분 | uniform, target 0.3986 | 동일 |
| VLM | 전 레이어 Q 19/32, MLP 7390/12288 유지 | 동일 개수 (유닛은 점수가 결정) |
| expert / KV | 무손 / drop 없음 | 동일 |
| 총 파라미터 | −2.657B (−24.0%) | 동일 (배분이 같으므로) |

점수 출처: `outputs/importance_v1/importance.npz`(I_traj), `outputs/jlens_coc/jlens.npz`(J).

## Stages & 사전 등록 게이트

### Stage 1 — open-loop, dual_uniform 예산에서 기준 비교 (~3h, GPU 2장)

`run_jspace_sweep.py --ratios 0.3986`으로 기존 6개 기준 × 1비율 + baseline = 7 config를
80클립(40클립 × 2샤드, `--clip-offset 0/40`)에서 마스크 스윕. 기존 r20/30 스윕과 동일한
클립·시드·조화(harness)라 프론티어에 r40 점이 그대로 추가된다.

**Gate O (진행 조건 — 평균·중앙값·꼬리 3중)**: paired j_traj−dual에서
- 중앙값: dNLL 차 ≤ +0.010, dADE 차 ≤ +0.050 (Wilcoxon 유의한 열세 없음)
- 평균: paired bootstrap 95% CI 상한이 dADE +0.10, dNLL +0.020 이하
  (minADE 델타가 heavy-tailed라 중앙값만으로는 "가끔 파국"을 놓친다 — 평균이 꼬리 감지기)
- 꼬리: per-clip dADE의 p95가 dual 대비 +0.5 m 이내 (단일 클립 파국 없음)

셋 중 하나라도 실패 시: 40%에서는 기준이 갈린다는 뜻 → j_traj_uniform은 만들지 않고
negative result로 기록, 비율을 낮춘 절충(예: 30%)은 별도 논의.
(참고: 기존 r20/r30 스윕은 평균으로 읽어도 결론 동일 — r30 dADE 평균 CI [−0.258,−0.045]로
오히려 j_traj 우위가 더 강함. 2026-07-30 확인.)

### Stage 2 — 물리 수술 (~1.5h, GPU 1장)

`make_slim.py`에 `j_traj_uniform` branch 추가: `select_mask_ratios`에
`np.full(36, 0.3986)`(grid의 `allocations()`와 동일하게 `slim_integrated_mag` meta에서 역산),
expert identity, kvonly 없음. `verify_slim.py`로 mask↔slim 일치 확인,
파라미터 수 8.42B(=dual_uniform과 동일) 확인.

### Stage 3 — 폐루프 (~하룻밤)

`launch_alpasim_matrix.sh`에 `slim_j_traj_uniform` GPU 맵 추가(**Ada** driver — 커널 confound
방지). 30씬 × 2롤아웃. **baseline도 같은 세션에서 재실행**하여 세션 드리프트 confound 제거
(기존 j_traj_full 비교에서 지적된 한계). GPU 여유 시 dual_uniform도 재실행(3 config).

**주 종점(사전 등록, 단일)**: `lead_thw_median`의 baseline 대비 paired delta.
지난번 36개 검정 사후 선택 문제의 재발 방지 — 이번엔 이 하나가 primary이고 나머지
(thw_p05, frac_thw_below_1s, mean_speed_when_close, scene score, CoC degen)는 secondary.

**Gate C (성공 조건)**: j_traj_uniform의 lead_thw_median delta가 유의하게 나쁘지 않고
(p≥0.05 또는 방향 양호), CoC degen < 0.05.
- 통과 → "CoC 라벨 없이 dual_uniform급 폐루프 성능" — 논문의 핵심 주장 성립.
- 실패 → 열화가 기준 탓으로 확정(H 기각). 이 역시 가치 있는 attribution.

## 산출물

- `outputs/jsweep40_s0`, `jsweep40_s40`, `jsweep40_summary` (analyze_jsweep 재사용)
- `outputs/slim_j_traj_uniform/` (slim_meta.json + slim_state.pt는 gitignore)
- `/home/cvlab21/project/chan/alpasim-runs/matrix_slim_j_traj_uniform/` (+ baseline 재실행)
- 보고서 `reports/2026-07-29_jspace-label-free-pruning.html` §6 갱신 + §9 결론 수정

## Risks

| Risk | Mitigation |
|---|---|
| 40%에서 j_traj가 무너짐 | Gate O가 폐루프 비용 전에 차단 |
| GPU 점유 | run_retry_host 재시도, 폐루프는 야간 |
| 세션 드리프트 | baseline 동일 세션 재실행 (사전 등록) |
| 다중비교 재발 | 주 종점 1개 사전 등록 |
| int(0.3986*100)=39 라벨 혼동 | config 이름은 `_r39`로 생성됨 — 분석 시 라벨만 주의 |
