# 최신 프레임 카메라 블랙아웃 — 개루프 (2026-10-18)

## 목적
Alpamayo 1.5 10B는 매 추론마다 카메라 4대 × 시점 4개(t0−0.3, −0.2, −0.1, t0초) = 16장을 받는다
(alpasim 150씬·hard100 런 전부 이 설정임을 2026-10-18 ASL 전수 검사로 확인).
**최신 시점(t0) 프레임 4장 중 무작위 1장이 들어오지 않고 검은 이미지로 대체될 때 주행 성능이 얼마나
떨어지는가**를 잰다. alpasim 폐루프 전에 개루프로 효과 크기를 먼저 확인하고, 폐루프 조건은 이 결과를
보고 정한다.

가설
- H1: 최신 프레임 1장을 잃으면 minADE@6이 증가한다.
- H2 (탐색): 가려진 카메라가 전방 광각(`camera_front_wide_120fov`)일 때 하락이 가장 크다.
- 반대 가능성: 과거 프레임 12장과 ego history가 남아 있어 거의 보상된다. 그 경우 효과가 0에 가깝고,
  그 자체가 결과다.

## 설계
**개입** — `sc.load_cached()` 직후, `build_inputs()` 전에:
```python
# data["image_frames"]: (4 cam, 4 frame, 3, H, W) uint8, frame -1 = t0
cam = blackout_cam(args.blackout_seed, clip_id)   # 0..3, sha256(f"blackout:{seed}:{clip_id}") % 4
data["image_frames"][cam, -1] = 0                 # (3, H, W) 완전 검정
```
- 카메라 선택은 클립 id에서 정해지므로 shard 수·순서·재시작과 무관하다.
- 선택 시드는 **궤적 샘플링 시드(`clip_seed`)와 분리**한다. 블랙아웃 조건과 기존 baseline이 같은
  샘플링 시드를 공유하므로 페어드 차이에서 잡음이 상쇄된다.
- 이미지 슬롯·해상도·카메라 이름 텍스트·`frame 3` 텍스트는 그대로다. 토큰 수와 위치가 바뀌지 않으므로
  바뀌는 것은 그 이미지의 픽셀 내용뿐이다.
- 블랙은 JPEG 디코드 **후**에 0으로 채우므로 정확히 0이다.

**조건** — 블랙아웃 1개 조건. 기준(블랙아웃 없음)은 기존 Ada baseline(`baseline_pred_test`,
`baseline_pred_oodval`)을 재사용하고, G0로 재현성을 먼저 확인한다.

**모델** — `nvidia/Alpamayo-1.5-10B` (revision `7aba829`, 무압축).

**평가셋** — test500 (`--set test`), OOD-val262 (`--set ood --manifest ood_val --no-tf`).
프로토콜(2026-08-19)대로 rollout만 측정하고 TF는 돌리지 않는다. k=8 샘플을 저장하고 앞 6개로 @6을
축소한다. `--save-pred`로 경로 자체도 저장한다(기존 `baseline_pred_*`와 같은 형식).

**하드웨어** — Ada 4–7(평가는 Ada 전용, 실행 직전 빈 카드 확인). 세트마다 4 shard로 나눈다.

## 지표
- 주: minADE@6 평균, 기준 대비 페어드 Δ(부트스트랩 CI). 부: 중앙값 + Wilcoxon, minFDE@6.
- 지평 1.6 / 3.2 / 6.4초.
- CoC: 퇴화율(`coc_degenerate`), 기준과 CoC 텍스트가 동일한 클립 비율.
- 분해(탐색): **가려진 카메라별**(test500 카메라당 ~125클립, OOD ~65클립), bucket별.

## 사전 등록 게이트
- **G0 (파이프라인)**: 수정한 러너로 블랙아웃 없이 test500 앞 10클립을 돌린 결과가
  `baseline_pred_test`와 비트 동일(`ade_rollout_k`, `gen_coc`). 실패하면 기준을 재사용하지 않고 새로 돈다.
- **G1 (효과 존재)**: 평균 ΔminADE@6의 부트스트랩 CI가 0을 배제하는지 세트별로 보고한다.
  두 세트 모두 0을 포함하면 "최신 프레임 1장 손실은 개루프에서 측정 한계 아래"로 기록하고, 폐루프에서
  더 강한 개입(여러 대, 또는 블랙아웃 지속)을 쓸지 다시 정한다.
- 카메라별 분해는 탐색이다. 검정력이 낮아 방향만 보고, 결론으로 쓰지 않는다.

## 해석상 주의
- 검은 이미지는 "정보 없음"과 "분포 밖 입력"이 섞인 개입이다. 이번에는 둘을 분리하지 않는다
  (대조 실험 제외, 2026-10-18 결정).
- 개루프는 6.4초를 채점하지만 폐루프 실효 지평은 ~1.9초다(2026-09-11). 1.6초 지평 결과를 함께 본다.

## 구현
- `experiments/evaluation/run_baseline.py`: `--blackout`(t0 무작위 1대), `--blackout-seed`(기본 0) 추가.
  tag에 `_bk1`을 붙이고 레코드에 `blackout_cam`(카메라 index: 0 좌측 / 1 전방 광각 / 2 우측 /
  6 전방 망원)을 저장한다.
- `experiments/evaluation/analyze_blackout.py` (신규): 위 지표와 G0/G1 판정을 계산한다.
- 출력:
  - `outputs/blackout_test/`, `outputs/blackout_oodval/` — `baseline_bk1_s{i}of4.json`, `config.json`
  - `outputs/blackout_analysis/` — `config.json`, `metrics.json`, `summary.txt`, `plots/*.png`
    (지평별 Δ, 카메라별 Δ)

## 비용
762클립 × ~8초 ≈ 1.7 GPU-h. Ada 4장 병렬로 **약 30–40분**(모델 로드 포함). G0 확인 ~3분.

## 결과 (2026-10-18, 21:30–21:58 KST 실행)
- **G0 PASS**: 10/10 클립이 `baseline_pred_test`와 비트 동일(`ade_rollout_k`, `gen_coc`, `pred_xy_k`).
- **G1 FAIL (두 세트)**: 6.4초 minADE@6 Δ test500 +0.0189 [−0.0023, +0.0406], OOD-val262 +0.0261
  [−0.0234, +0.0773]. 1.6초·3.2초는 두 세트 모두 평균 CI가 0을 배제(+2.2–3.7%)하고, 중앙값은 전 칸 양수.
  6.4초 CI가 넓은 것은 ±2–3 m 양방향 꼬리 때문이다(test500 10% 절사평균 +0.0085 [+0.0025, +0.0153]).
- 크기는 24% 프루닝(`dual_u40_v2`) 비용의 14–27%. CoC 텍스트 동일 92.2% / 86.3%, 퇴화율 변화 ≤ +0.8pp.
- 리포트: `reports/evaluation/2026-10-18_camera-blackout-openloop.html`.
- 사전 등록 규칙대로, 폐루프 전에 더 강한 개입(4대 모두 t0 블랙, 또는 카메라 1대의 4프레임 전부 블랙)을 정할 것.

## 추가: 카메라별 고정 블랙아웃 (2026-10-18, 사용자 지시)
무작위 1대 설계로는 카메라당 60–132클립뿐이라 어느 카메라가 중요한지 말할 수 없었다. 그래서 카메라 4대 각각에
대해 **모든 클립**에서 그 카메라의 t0 프레임만 검게 바꾼다(4조건 × test500 + OOD-val262).
- 러너: `run_baseline.py --blackout-cam {0,1,2,6}`, tag `baseline_bkcam{idx}`,
  출력 `outputs/blackout_cam{idx}_{test,oodval}/`. 실행: `launch_blackout_cam.sh` (카드당 카메라 1대, Ada 4–7).
- 지표: minADE@6 / minFDE@6 (1.6 / 3.2 / 6.4초), baseline 대비 페어드 Δ, 카메라 간 페어드 비교(같은 클립이므로).
- **G0' (재현성)**: 무작위 설계에서 그 카메라가 가려졌던 클립(카메라당 ~1/4)은 같은 계산이므로
  `blackout_test`/`blackout_oodval` 행과 비트 동일해야 한다.
- **H2 판정**: 전방 광각의 6.4초 minADE@6 Δ가 다른 세 카메라보다 크다(카메라 간 페어드 CI > 0, Holm 보정 Wilcoxon)가
  두 세트 모두에서 성립하면 지지. 6.4초는 꼬리가 크므로 1.6초 결과를 함께 보고한다.
- 분석: `analyze_blackout_cam.py` → `outputs/blackout_cam_analysis/`.

### 결과 (2026-10-18 22:42 – 10-19 00:28 KST)
- **G0' PASS**: 겹치는 762클립(test500 117/132/131/120, OOD 73/66/60/63) 전부 비트 동일.
- **H2 지지 안 됨**: 전방 광각은 두 세트 모두 비용이 거의 없다(6.4초 minADE@6 +2.3% / −1.1%, 중앙값 p=0.57 / 0.54).
- **가장 중요한 카메라는 전방 망원(30°)**: 6.4초 minADE@6 test500 +0.0252 (+3.0%, 중앙값 p=0.006, 절사평균 CI>0),
  OOD +0.0649 [+0.0102, +0.1239] (+6.5%), minFDE@6 +0.1027 / +0.1839 (CI>0). OOD에서 전방 광각 − 망원 −0.0754
  [−0.1369, −0.0189], Holm p=0.015. OOD 1.6·3.2초에서도 망원만 CI>0.
- 좌·우 크로스는 0–3%, 우측 크로스는 test500에서 작지만 CI>0(+1.2%). CoC 퇴화율 변화 ≤ +1.1pp.
- 리포트: `reports/evaluation/2026-10-18_camera-blackout-percamera.html`.

## 폐루프로 넘길 때 정할 것 (이번 범위 밖)
- 가릴 카메라를 매 스텝 새로 뽑을지, rollout 동안 고정할지.
- 블랙 프레임이 다음 스텝에서 t−0.1 자리로 밀려 들어가게 할지(카메라 고장 모사), t0 자리에만
  둘지(개루프와 같은 개입). alpasim 드라이버의 `FrameCache`는 원본을 보관하므로 후자가 구현이 쉽다.
