# 카메라 시점 불일치 — 개루프, test2000 (2026-10-26)

## 목적
Alpamayo 1.5 10B는 카메라 4대 × 시점 4개(t0−0.3, −0.2, −0.1, t0초)를 받지만 **타임스탬프는 받지 않는다**
(`alpamayo1_5/models`는 타임스탬프를 참조하지 않고, 프롬프트는 `"<카메라>: frame 0..3"` + 이미지뿐).
모델은 4대가 동기화된 10 Hz 입력이라고 암묵적으로 가정한다. 이 가정이 깨질 때(카메라 한 대가 늦게
도착하거나, 영상 전체가 ego 궤적보다 늦거나, 프레임 간격이 바뀔 때) 주행 예측이 얼마나 나빠지는지 잰다.
사용자 요청(2026-10-26): "카메라 입력 시점을 바꿔서 실험, 카메라 촬영 시점 불일치도 테스트", test2000 먼저,
Blackwell 0–3 전부 사용.

**데이터의 자연 어긋남** (test 20클립 실측, 2026-10-26): 4대 모두 30 Hz. 슬롯 간격은 99.97–100.11 ms.
카메라는 {전방 광각, 전방 망원}과 {좌 크로스, 우 크로스} 두 묶음으로 찍히고, 묶음 안 차이는 0.1 ms 미만,
묶음 사이는 클립마다 고정된 0.9–27.4 ms(평균 12.5)다. 그러므로 학습 분포 안의 카메라 간 어긋남은 1프레임
(33 ms) 이하다.

가설
- H1: 카메라 1대가 100 ms(3프레임) 늦으면 minADE@6이 증가한다. 비용 순서는 블랙아웃 실험과 같아서
  전방 망원이 가장 크다(2026-10-18 카메라별 블랙아웃).
- H2: 전방 망원 지연의 비용은 Δ에 따라 단조 증가한다(67 < 100 < 200 < 300 ms). 67 ms는 자연 어긋남의
  두 배 수준이라 비용이 0에 가까울 수 있다.
- H3: 4대 전체를 같은 Δ만큼 늦추면(영상이 ego 궤적보다 늦음) 단일 카메라 지연보다 비용이 크고,
  움직임이 큰 bucket(회전·가감속)에서 더 크다.
- H4 (탐색): 프레임 간격을 바꾸면(200 ms: 겉보기 움직임 2배, 33 ms: 1/3배) 비용이 생긴다. 방향은 사전에
  정하지 않는다.
- 반대 가능성: 4프레임 중 일부만 바뀌거나 이동이 작아서 효과가 측정 한계 아래일 수 있고, 그 자체가 결과다.

## 설계
**프레임 창 캐시** — `build_timing_cache.py`가 test2000 클립마다 카메라별로 [t0−1.0 s, t0]의 30 Hz
프레임 전부(31장)를 캡처 타임스탬프와 함께 JPEG q95로 저장한다(`pre_processed/test_timing`, ~86 GB).
원래 4개 시점 프레임은 기존 `test` 캐시와 **바이트 단위로 같아야 한다**(G0, 클립마다 검사·기록).

**개입** — `run_baseline.py --lag-ms L --stride-ms S --timing-cams C`: 대상 카메라 c의 슬롯 i(0..3)는
`t0 − L − (3−i)·S` 이전 가장 가까운 프레임(로더와 같은 규칙), 나머지 카메라는 원래 시점. ego history,
프롬프트 텍스트, 토큰 수는 그대로이고 픽셀만 바뀐다. 모든 조작은 30 Hz 프레임 단위로 정확히 떨어진다
(67/100/200/300 ms = 2/3/6/9프레임, 간격 33/200 ms = 1/6프레임). 실제로 고른 캡처 시각을
`frame_ts_ms`로 클립마다 저장한다.

**arm** (우선순위 순; 각 arm은 test2000 전체, 4 shard)
| arm | 대상 | 지연 | 간격 | 묻는 것 |
|---|---|---|---|---|
| sync | – | 0 | 100 | 기준 |
| tm6_lag100 / tm0 / tm1 / tm2 | 카메라 1대 | 100 ms | 100 | H1 카메라별 |
| tmall_lag100 | 4대 | 100 ms | 100 | H3 |
| tmall_lag300 | 4대 | 300 ms | 100 | H3 |
| tmall_st200 | 4대 | 0 | 200 ms | H4 |
| tm6_lag300 / tm6_lag67 / tm6_lag200 | 전방 망원 | 300 / 67 / 200 ms | 100 | H2 |
| tmall_st33 | 4대 | 0 | 33 ms | H4 |

H2의 지연 강도 축 카메라는 블랙아웃 결과를 근거로 실행 전에 전방 망원으로 고정했다.

**모델** — `nvidia/Alpamayo-1.5-10B` (revision `7aba829`, 무압축). k=8, 시드는 클립 id에서(`clip_seed`),
모든 arm이 같은 샘플링 시드를 공유한다. rollout만(프로토콜 2026-08-19), `--save-pred`.

**하드웨어** — Blackwell 0–3(사용자 지시). shard i는 모든 arm에서 카드 i에서 돈다. 기준(sync)도 같은
Blackwell에서 새로 돌린다(Ada 결과와는 커널이 달라 페어링 불가, test2000 baseline은 원래 없음).
비용: ~9.4 s/clip → arm당 ~78분, 12 arm ~16시간.

## 지표
- 주: minADE@6(6.4 s) 평균의 sync 대비 페어드 Δ, 부트스트랩 95% CI(클립 재표집).
  부: 중앙값 Δ + Wilcoxon, minFDE@6, 지평 1.6 / 3.2 s, 상대 변화(%).
- CoC: 퇴화율, sync와 CoC 문장이 같은 클립 비율.
- 분해: bucket별, t0 속도 3분위별(ego history 마지막 구간).
- 비교 척도: 블랙아웃(test500, Ada)의 비용과 test500 부분집합에서 상대 % 비교만 한다(카드가 달라 절대값 페어링 불가).

## 사전 등록 게이트
- **G0a (캐시)**: test_timing의 원래 4시점 프레임이 test 캐시와 2000/2000 클립 바이트 동일. 실패하면 조작 arm을
  돌리지 않는다(런처가 `READY` 없으면 정지).
- **G0b (재현)**: sync arm이 기존 Blackwell baseline(`baseline_bw_ps_test`, test500)과 겹치는 클립에서
  비트 동일(`ade_rollout_k`, `gen_coc`). 스모크 2/2 통과(2026-10-26).
- **G1 (효과 존재)**: arm별 평균 ΔminADE@6 부트스트랩 CI가 0을 배제하는지. 0에 걸치면 10만 회 이상으로
  재추정한다(2026-10 부트스트랩 수렴 교훈).
- H2 단조성: 망원 67/100/200/300 ms의 평균 Δ가 순서대로인지 보고, 인접 쌍 차이의 CI도 같이 낸다.

## 해석상 주의
- 지연 arm은 "옛 프레임"이 들어간 것이지 손상된 프레임이 아니다. 블랙아웃(정보 없음 + 분포 밖)과 달리
  분포 안 이미지다.
- 4대 전체 지연(tmall_lag*)은 카메라 간 동기는 유지되고 ego 궤적과만 어긋난다. 카메라 간 불일치는 단일 카메라 arm이다.
- 개루프는 6.4 s를 채점하지만 폐루프 실효 지평은 ~1.9 s다. 1.6 s 지평을 같이 본다.

## 추가: 프레임별 무작위 촬영 시점 지터 (사용자 설계, 2026-10-26)
사용자가 원래 생각한 설정. 위 체계적 지연 arm은 망원 100 ms(`tm6_lag100`)까지만 돌리고 나머지 10개는 보류.

**개입** — 16장(카메라 4 × 시점 4) 각각에 독립적으로 dt ~ U(−50, +50) ms를 뽑아
`t0 − (3−i)·100 ms + dt` 이전 가장 가까운 프레임을 쓴다. |dt| ≤ 50 ms라 인접 슬롯(100 ms 간격)의 순서는
뒤집히지 않는다. t0 슬롯의 dt > 0이면 미래 프레임이므로 **검은 이미지**로 대체한다(이미지를 아예 빼는 방식은
다음 단계). dt는 `sha256(f"jitter:{seed}:{clip_id}")`에서 뽑아 모든 arm이 같은 값을 쓰고, 샘플링 시드와 분리한다.
`run_baseline.py --jitter-ms 50 --jitter-future black` (`sample_cache.jitter_requests`).

**사전 실측** (test2000 캐시의 실제 캡처 시각에 같은 규칙 적용, 모델 없이):
- sync 대비 프레임 이동: −2프레임 5.0%, −1 33.1%, 0 33.0%, +1 25.1%, +2 3.8%. 실제 시각 이동 −68 ~ +67 ms.
- 순서 역전 0건. 인접 슬롯이 같은 프레임을 받는 클립 410/2000(20.5%) — 그대로 둔다(기본안, 사용자 이견 없음).
- 검게 되는 t0 프레임 수/클립: 0장 130, 1장 512, 2장 719, 3장 509, 4장 130 (평균 2.0).

**arm** (같은 dt, 이 순서로)
| arm | 지터 | 미래 t0 | 묻는 것 |
|---|---|---|---|
| jit50_black | 16장 전부 | 검은 이미지 | 사용자 설계 |
| jit50_drop | 16장 전부 | 이미지와 그 "frame 3" 라벨을 프롬프트에서 제거(카메라 이름 라벨은 유지) | 제외 방식의 차이 |

분해 arm(`jit50_keep`: 미래 t0는 원래 프레임 유지 / `jit50_black_noshift`: 지터 없이 검정만)은 사용자 결정으로
큐에서 뺐다(2026-10-26). 러너 옵션(`--jitter-future keep`, `--jitter-no-shift`)은 남아 있다.

**드롭 구현** — `analysis_lib.build_inputs(..., drop_images=[c*4+3 ...])`가 릴리스 `create_message`의 결과에서 해당
이미지와 바로 앞의 `"frame 3 "` 텍스트만 지운다. CPU 확인(2026-10-26): 빈 목록이면 input_ids·pixel_values가 기존과
비트 동일, 2장 제거 시 이미지 16→14, 프롬프트 3344→2972 토큰, "frame 3" 4→2, 카메라 이름 라벨 4개 유지, 남은
14장의 pixel 패치 동일. 모델 코드는 이미지 수를 고정하지 않는다(`image_grid_thw`로 전달).

**가설**
- H5: jit50_black의 minADE@6이 sync보다 크다(G1과 같은 기준).
- H6 (탐색): jit50_drop과 jit50_black의 차이. 검은 이미지는 "정보 없음 + 분포 밖 입력"이고, 제거는 정보만 없다.
  제거가 덜 나쁘면 검은 이미지의 비용 일부는 분포 밖 입력 때문이다. 방향은 사전에 정하지 않는다.
- H7 (탐색): 비용은 미래로 표시된 t0 프레임 수(0~4장)에 따라 늘고, 카메라별로는 전방 망원이 가장 크다(OLS).

## 추가: Alpamayo 2 Super (2026-10-27, 사용자 요청)
같은 지터 실험을 A2에서. arm 3개(사용자 지정): `sync`, `jit50_black`, `jit50_drop`. 세트는 test500(= test2000 앞 500;
사용자 이견 없어 권장안), 지터는 A2 궤적 입력 6대 전부(`[0,1,2,3,5,6]`, 권장안).
- **입력**: A2도 공개 코드상 타임스탬프를 모델에 넣지 않는다(`image_frames`, `camera_indices`, ego만; 타임스탬프는 자동 라벨링 검증용).
  프레임은 `test_timing`(0,1,2,6) + 새 `test_timing_rear`(3,5; 500클립, 11 GB), ego는 test 캐시(A2 로더와 동일, `build_cache_a2.py --verify` 3/3).
- **지터 추첨**: 카메라 0,1,2,6은 A1.5 실험과 같은 dt(`jitter_requests`, 같은 sha256), 3,5는 별도 스트림(`...:rear`).
  그래서 A2는 클립당 미래 t0가 평균 3장(6대 중) 빠진다(A1.5는 2장/4대).
- **러너**: `run_a2_timing.py` — alpamayo2-distill `run_oodval.py`의 생성·디노이징·층 분할을 import(수정 없음). 제거는
  메시지와 이미지 목록에서 같은 이미지와 그 `"frame 3"` 라벨을 함께 지운다.
- **하드웨어**: Blackwell 0,1(shard 0) / 2,3(shard 1), 층 분할 [30, 34]. ~14 s/clip → arm당 ~60분.
- **G0**: (a) A2 공식 경로(`build_cache_a2.py` → `select_task_input`) 3클립 vs 우리 조립 sync 입력 — 프레임 24장·캡처 시각·ego·
  메시지 동일(3/3); (b) 러너가 sync의 첫 클립에서 자체 빌더 = `run_oodval.build_inputs` 단언(통과);
  (c) 제거 시험 2클립: 겹치는 4대의 dt·프레임이 A1.5와 동일, 이미지 수 = 24 − 미래 t0 수.
- **비교**: A2 내부는 클립별 페어드. A1.5와는 같은 500클립에서 상대 %와 클립별 Δ의 상관으로만(모델이 다름).

## 추가: 모든 카메라의 t0 제외 (2026-11-03, 사용자 요청)
A1.5, test2000, 지터 없이 4대의 t0 프레임 전부 제외. 나머지 12장(t0−0.3 … t0−0.1)은 그대로.
- arm: `t0x_black`(t0 4장 검정, 이미지 16장 유지), `t0x_drop`(t0 4장과 그 `"frame 3"` 라벨을 프롬프트에서 제거, 이미지 12장).
  `run_baseline.py --t0-exclude {black,drop}`.
- 카드: 0번은 다른 작업(cvpr27 tyr supernet)이 사용 중이라 Blackwell 1–3, 3 shard(사용자 지시 "1,2,3번").
- G0: 조작 없는 경로를 새 배치(3 shard, 카드 1)로 10클립 → `camtiming_sync_test2000`과 10/10 비트 동일. 그래서 sync와 클립별 페어링.
- 질문: 도착 지연 관점에서 "최신 프레임이 하나도 오지 않은 채 추론"의 비용, 그리고 그때 검정 vs 제거 중 무엇이 나은지.
  지터 arm(t0 평균 2/4 제외)과 비교해 제외 장수에 따른 비용도 본다.

## 구현
- `experiments/evaluation/build_timing_cache.py` (새 파일)
- `experiments/evaluation/sample_cache.py`: `pick_timed_frames()`
- `experiments/evaluation/run_baseline.py`: `--lag-ms`, `--stride-ms`, `--timing-cams`, `--timing-cache`
- `experiments/evaluation/launch_camera_timing.sh`: 캐시 빌드(CPU) ‖ sync(GPU 0–3) → READY → 조작 arm (tm6_lag100 이후 사용자 지시로 중단)
- `experiments/evaluation/launch_camera_jitter.sh`: jit50_black → jit50_drop (`WAIT_EXP`로 앞 런 뒤에 대기)
- `experiments/head_analysis/analysis_lib.py`: `build_inputs(drop_images=...)`
- 출력: `outputs/camtiming_<arm>_test2000/`, 로그 `outputs/camtiming_logs/`
- 분석: `experiments/evaluation/analyze_camera_timing.py` → `outputs/camtiming_analysis/`, 리포트 `reports/evaluation/`
